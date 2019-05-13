import tensorflow as tf
import tensorlayer as tl
import numpy as np
import argparse
import os
import sys
sys.path.append("/Adaptive_Affinity_Fields")
from seg_models.models.pspnet import pspnet_resnet101 as model
from seg_models.models.d_net import d_net
from seg_models.image_reader import ImageReader
import network.common.layers as nn
import utils.general
from tqdm import tqdm
import math
import time
IMG_MEAN = np.array((122.675, 116.669, 104.008), dtype=np.float32)
num_iterations_to_alternate_trained_model = 2000

def get_arguments():
    parser = argparse.ArgumentParser(description='Train args')
    parser.add_argument('--batch-size',type=int,default=8,help='batch size. rely on the GPU memory')
    parser.add_argument('--data-dir',type=str,default='/Adaptive_Affinity_Fields/data/VOCdevkit/',help='dataset dir')
    parser.add_argument('--data-list',type=str,default='/Adaptive_Affinity_Fields/dataset/voc12/train.txt',help='list name')
    parser.add_argument('--label-ignore', type=int, default=255,help='The index of the label to ignore.')
    parser.add_argument('--img-mean', type=float, default=IMG_MEAN,help='imgs avg in each channel')
    parser.add_argument('--input-size', type=str, default='336,336',help='Comma-separated string with H and W of image.')
    parser.add_argument('--random-seed', type=int, default=1234,help='Random seed to have reproducible results.')
    parser.add_argument('--is-training', default='True',help='Whether to updates weights.')
    parser.add_argument('--use-global-status', default='False',help='Whether to updates moving mean and variance.')
    parser.add_argument('--learning-rate', type=float, default=4e-5,help='Base learning rate.')
    parser.add_argument('--learning-rate-D', type=float, default=1e-4,help='Base learning rate.')
    parser.add_argument('--power', type=float, default=0.9,help='Decay for poly learing rate policy.')
    parser.add_argument('--power-D', type=float, default=0.5,help='Decay for poly learing rate policy.')
    parser.add_argument('--momentum', type=float, default=0.9,help='Momentum component of the optimiser.')
    parser.add_argument('--weight-decay', type=float, default=5e-4,help='Regularisation parameter for L2-loss.')
    parser.add_argument('--weight-DLoss', type=float, default=1e-3,help='DLoss weight')
    parser.add_argument('--kd', type=float, default=1,help=' weight')
    parser.add_argument('--num-classes', type=int, default=21,help='Number of classes to predict.')
    parser.add_argument('--num-steps', type=int, default=10,help='Number of training steps.')
    parser.add_argument('--iter-size', type=int, default=10,help='Number of iteration to update weights')
    parser.add_argument('--keep-prob', type=float, default=0.5,help='Number of iteration to update weights')
    parser.add_argument('--random-mirror', default='True',help='Whether to randomly mirror the inputs.')
    parser.add_argument('--random-crop', default='True',help='Whether to randomly crop the inputs.')
    parser.add_argument('--random-scale', default='True',help='Whether to randomly scale the inputs.')
    parser.add_argument('--menet-restore-from', type=str, default='/Adaptive_Affinity_Fields/snapshot/model.ckpt-10000',#'/home/wangyang/Desktop/Semantic Segmentation Code/MENet/pre_model/resnet_v1_101.ckpt'
                        help='Where menet restore model parameters from.')
    parser.add_argument('--ranet-restore-from', type=str, default=None,#'/home/wangyang/Desktop/Semantic Segmentation Code/MENet/pre_model/vgg_16.ckpt'
                        help='Where ranet restore model parameters from.')
    parser.add_argument('--save-pred-every', type=int, default=1,help='Save summaries and checkpoint every often.')
    parser.add_argument('--update-tb-every', type=int, default=10,help='Update summaries every often.')
    parser.add_argument('--snapshot-dir', type=str, default='/Adaptive_Affinity_Fields/train_d_model',
                        help='Where to save snapshots of the model.')
    parser.add_argument('--not-restore-classifier', action='store_true',help='Whether to not restore classifier layers.')

    return parser.parse_args()


def save(saver,sess,logdir,step):
    model_name = 'model.cpkt'
    checkpoint_path = os.path.join(logdir,model_name)
    saver.save(sess,checkpoint_path,global_step=step)
    print("model has been stored")


def save_D(saver,sess,logdir,step):
    model_name = 'model_D.cpkt'
    checkpoint_path = os.path.join(logdir,model_name)
    saver.save(sess,checkpoint_path,global_step=step)
    print("model has been stored")

def load(sess,saver,store_path):
    saver.restore(sess,store_path)
    print('load checkpoint from {}'.format(store_path))



def main():
    arg = get_arguments()
    h,w = map(int,arg.input_size.split(','))
    input_size = (h,w)
    out_size = (int(math.ceil(h/8)), int(math.ceil(w/8)))
    tf.set_random_seed(arg.random_seed)
    coord = tf.train.Coordinator()
    step_ph = tf.placeholder(dtype=tf.float32,shape=())
    with tf.device('/cpu:0'):
      with tf.name_scope('create_inputs'):
        reader = ImageReader(arg.data_dir,arg.data_list,input_size,arg.random_scale,arg.random_mirror,arg.random_crop,arg.label_ignore,IMG_MEAN)
        image_batch, label_batch = reader.dequeue(arg.batch_size)
    ###define GAN
    output = model(image_batch,num_classes=arg.num_classes,is_training=arg.is_training,use_global_status=arg.use_global_status,reuse=False)
    #RSNet_out = RSNet.resnet_v1(image_batch,name='resnetv2',filters=[64,128,256,512],num_blocks=[3,4,23,3],strides=[2,1,1,1],dilations=[None, None, 2, 4],is_training=arg.is_training,use_global_status=arg.use_global_status,reuse=False)

    real_disciminator_input, fake_disciminator_input = utils.general.make_dis_input(output[0], image_batch, label_batch, arg.num_classes)

    output_D_G = d_net(fake_disciminator_input,arg.num_classes,arg.is_training,arg.use_global_status,reuse=False)


    w = arg.weight_decay
    label = tf.image.resize_nearest_neighbor(label_batch,out_size,name='label_shrink')
    label_int = tf.cast(label,tf.int32)
    label_flat = tf.reshape(label,[-1,])
    not_ignore_pixel = tf.where(tf.less_equal(label_flat,arg.num_classes-1))
    pixel_inds = tf.squeeze(not_ignore_pixel, 1)

    labels_gather = tf.to_int32(tf.gather(label_flat,pixel_inds))
    output_2d = tf.gather(tf.reshape(output[0],[-1,arg.num_classes]),pixel_inds)
    loss_soft1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_2d,labels=labels_gather)
    loss_soft = tf.reduce_mean(loss_soft1)
    """
        output_softmax = tf.nn.softmax(output_2d)
        output_softmax = tf.cast(output_softmax,tf.float32)
        one = tf.constant(1.0)
        output_softmax = tf.subtract(one,output_softmax)
        output_softmax_log = tf.log(output_softmax)
        labels_one_hot = tf.one_hot(labels_gather,depth=arg.num_classes)
        pixel_wise_loss = -tf.multiply(output_softmax_log,labels_one_hot)
        pixel_wise_loss = tf.reduce_sum(pixel_wise_loss,-1)
    """
    output_D_G_flat = tf.reshape(output_D_G,[-1,])
    output_D_G_flat = tf.gather(output_D_G_flat,pixel_inds)
    output_D_G_normal = tf.identity(output_D_G_flat/tf.reduce_sum(output_D_G_flat))
    loss_D_G = output_D_G_normal*loss_soft1
    loss_D_G = tf.reduce_sum(loss_D_G)
    loss_D = loss_D_G

    G_var = tl.layers.get_variables_with_name('resnet_v1_101', True, False)
    D_var = tl.layers.get_variables_with_name('d_net', True, True)


    base_lr = tf.constant(arg.learning_rate)
    learning_rate_D = tf.constant(arg.learning_rate_D)
    learning_rate = tf.scalar_mul(base_lr,tf.pow((1-step_ph/arg.num_steps), arg.power))

    d_optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate_D)
    d_grads = d_optimiser.compute_gradients(loss_D, var_list=D_var)
    d_optim = d_optimiser.apply_gradients(d_grads)
    ###config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)


    ###define vis
    output_vis_full = tf.image.resize_nearest_neighbor(output[0],tf.shape(image_batch)[1:3,])
    output_vis_full = tf.argmax(output_vis_full, axis=3)
    output_vis_full = tf.expand_dims(output_vis_full, dim=3)
    output_vis_full = tf.cast(output_vis_full, tf.uint8)
    label_vis = tf.cast(label_batch, tf.uint8)
    image_summary = tf.py_func(utils.general.inv_preprocess,[image_batch,IMG_MEAN],np.uint8)
    gt_summary = tf.py_func(utils.general.decode_labels,[label_vis,arg.num_classes],np.uint8)
    output_summary = tf.py_func(utils.general.decode_labels,[output_vis_full,arg.num_classes],np.uint8)
    total_image_summary = tf.summary.image('image',tf.concat(axis=2, values=[image_summary, gt_summary, output_summary]),max_outputs=arg.batch_size)
    loss_soft_summary = tf.summary.scalar('DLoss',loss_D)

    #total_loss_summary = tf.summary,scalar('total_loss',total_loss)
    total_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(arg.snapshot_dir,graph=tf.get_default_graph())



    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)
    menet_restore_var = [v for v in tf.global_variables() if 'resnet_v1_101' in v.name and 'Adam' not in v.name]
    #ranet_restore_var = [v for v in tf.global_variables() if 'RANet' in v.name and 'fc8' not in v.name and 'fc9' not in v.name and 'Adam' not in v.name]
    if arg.menet_restore_from is not None:
        loader1 = tf.train.Saver(var_list=menet_restore_var)
        load(sess,loader1,arg.menet_restore_from)

    if arg.ranet_restore_from is not None:
        loader2 = tf.train.Saver(var_list=ranet_restore_var)
        load(sess,loader2,arg.ranet_restore_from)

    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    ###run
    pbar = tqdm(range(arg.num_steps))
    for step in pbar:
        start_time = time.time()
        feed_dict = {step_ph:step}
        step_loss_G = 0
        step_loss_D = 0

        itt = 0
        while itt < num_iterations_to_alternate_trained_model:
            sess_outs = [d_optim,loss_D,output_D_G_normal]
            _ , DLoss_out,output_D_G_normal_out= sess.run(sess_outs,feed_dict=feed_dict)
            itt = itt + 1

            print('DLoss is {:.6f}'.format(DLoss_out))
            step_loss_D += DLoss_out
        learning_rate_D = learning_rate_D/2
        step_loss_D = step_loss_D/num_iterations_to_alternate_trained_model
        save_D(saver, sess, arg.snapshot_dir, step)
        duration = time.time() - start_time
        desc = 'DLoss = {:.3f}'.format(step_loss_D)
        pbar.set_description(desc)
    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main()
