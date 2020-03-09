from __future__ import print_function

import argparse
import os
import time
import math

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tqdm import tqdm
import sys
sys.path.append("/home/f523/wangyang/segmentation/Adaptive_Affinity_Fields/")
sys.path.append("/home/f523/wangyang/segmentation/Adaptive_Affinity_Fields/utils")
#from seg_models.models.pspnet_v2 import pspnet_v2_resnet101 as model
#from seg_models.models.pspnet_v2_1 import pspnet_v2_resnet101 as model
from seg_models.models.pspnet_v2_9 import pspnet_v2_resnet101 as model
#from seg_models.models.pspnet import pspnet_resnet101 as model
from seg_models.image_reader2 import ImageReader
import network.common.layers as nn
import general
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
IMG_MEAN = np.array((122.675, 116.669, 104.008), dtype=np.float32)
IMG_MEAN_ndsm = np.array((122.675, 116.669, 104.008, 100.0), dtype=np.float32)


def get_arguments():
  """

  Returns:
    A list of parsed arguments.
  """
  parser = argparse.ArgumentParser(description='Semantic Segmentation')
  # Data parameters
  parser.add_argument('--batch-size', type=int, default=4,
	             help='Number of images in one step.')
  parser.add_argument('--data-dir', type=str, default='/home/f523/wangyang/segmentation/Vaihingen/Split/',
 	             help='/path/to/dataset/.')#/media/f523/7cf72e9a-af1d-418e-b1f1-94d2a5e0f0d5/f523/wangyang/potsdam/Split
  parser.add_argument('--data-list', type=str, default='/home/f523/wangyang/segmentation/Adaptive_Affinity_Fields/dataset/Vaihingen/Vaihingen_train_ndsm_hist.txt',
	             help='/path/to/datalist/file.')
  parser.add_argument('--ignore-label', type=int, default=255,
	             help='The index of the label to ignore.')
  parser.add_argument('--input-size', type=str, default='336,336',
	             help='Comma-separated string with H and W of image.')
  parser.add_argument('--random-seed', type=int, default=1234,
	             help='Random seed to have reproducible results.')
 # Training paramters
  parser.add_argument('--is-training', action='store_true',default=True,
	             help='Whether to updates weights.')
  parser.add_argument('--use-global-status', action='store_true',default=True,
	             help='Whether to updates moving mean and variance.')
  parser.add_argument('--learning-rate', type=float, default=5e-5,
	             help='Base learning rate.')
  parser.add_argument('--power', type=float, default=0.9,
	             help='Decay for poly learing rate policy.')
  parser.add_argument('--momentum', type=float, default=0.9,
	             help='Momentum component of the optimiser.')
  parser.add_argument('--weight-decay', type=float, default=5e-5,
	             help='Regularisation parameter for L2-loss.')
  parser.add_argument('--num-classes', type=int, default=6,
	             help='Number of classes to predict.')
  parser.add_argument('--num-steps', type=int, default=9000,
	             help='Number of training steps.')
  parser.add_argument('--iter-size', type=int, default=10,
	             help='Number of iteration to update weights')
  parser.add_argument('--random-mirror', action='store_true',default=True,
	             help='Whether to randomly mirror the inputs.')
  parser.add_argument('--random-crop', action='store_true',default=True,
	             help='Whether to randomly crop the inputs.')
  parser.add_argument('--random-scale', action='store_true',default=True,
	             help='Whether to randomly scale the inputs.')
  parser.add_argument('--kld-margin', type=float,default=3,
	             help='margin for affinity loss')
  parser.add_argument('--kld-lambda-1', type=float,default=1,
	             help='Lambda for affinity loss at edge.')
  parser.add_argument('--kld-lambda-2', type=float,default=1,
	             help='Lambda for affinity loss not at edge.')
 # Misc paramters '/home/f523/wangyang/segmentation/Adaptive_Affinity_Fields/resnet_v1_101.ckpt'
  parser.add_argument('--restore-from', type=str, default="/home/f523/wangyang/segmentation/Adaptive_Affinity_Fields/snapshot/snapshot_v9_ndsm_hist/model.ckpt-5500",#
	             help='Where restore model parameters from.')
  parser.add_argument('--save-pred-every', type=int, default=500,
	             help='Save summaries and checkpoint every often.')
  parser.add_argument('--update-tb-every', type=int, default=20,
	             help='Update summaries every often.')
  parser.add_argument('--snapshot-dir', type=str, default='/home/f523/wangyang/segmentation/Adaptive_Affinity_Fields/snapshot/snapshot_v9_ndsm_hist_temp',
	             help='Where to save snapshots of the model.')
  parser.add_argument('--not-restore-classifier', action='store_true',
	             help='Whether to not restore classifier layers.')


  return parser.parse_args()



def save(saver, sess, logdir, step):
  """Saves the trained weights.

  Args:
    saver: TensorFlow Saver object.
    sess: TensorFlow session.
    logdir: path to the snapshots directory.
    step: current training step.
  """
  model_name = 'model.ckpt'
  checkpoint_path = os.path.join(logdir, model_name)

  if not os.path.exists(logdir):
    os.makedirs(logdir)
  saver.save(sess, checkpoint_path, global_step=step)
  print('The checkpoint has been created.')


def load(saver, sess, ckpt_path):
  """Loads the trained weights.

  Args:
    saver: TensorFlow Saver object.
    sess: TensorFlow session.
    ckpt_path: path to checkpoint file with parameters.
  """
  saver.restore(sess, ckpt_path)
  print('Restored model parameters from {}'.format(ckpt_path))

def focal_loss(onehot_labels, cls_preds, gamma=2, scope="focal"):

    with tf.name_scope(scope, 'focal_loss', [cls_preds, onehot_labels]) as sc:
        onehot_labels = slim.one_hot_encoding(onehot_labels, 6)
        logits = tf.convert_to_tensor(cls_preds,tf.float32)
        onehot_labels = tf.convert_to_tensor(onehot_labels)
        onehot_labels = tf.cast(onehot_labels, logits.dtype)
        predictions = tf.nn.softmax(logits)
        pt = tf.reshape(tf.reduce_sum(tf.multiply(onehot_labels,predictions),axis=1),[-1,1])
        loss=-tf.reduce_sum(tf.pow(1-pt,gamma)*(tf.log(pt)),axis=1)
        loss=tf.reduce_mean(loss)
        return loss

def main():
  """Create the model and start training.
  """
  # Read CL arguments and snapshot the arguments into text file.

  args = get_arguments()
  if not os.path.exists(args.snapshot_dir):
      os.mkdir(args.snapshot_dir)
  general.snapshot_arg(args)

  # The segmentation network is stride 8 by default.
  h, w = map(int, args.input_size.split(','))
  input_size = (h, w)
  innet_size = (int(math.ceil(h/2)), int(math.ceil(w/2)))
  innet_size1 = (int(math.ceil(h/4)), int(math.ceil(w/4)))
  innet_size2 = (int(math.ceil(h/8)), int(math.ceil(w/8)))
  # Initialize the random seed.
  tf.set_random_seed(args.random_seed)

  # Create queue coordinator.
  coord = tf.train.Coordinator()

  # current step
  step_ph = tf.placeholder(dtype=tf.float32, shape=())

  # Load the data reader.
  with tf.device('/cpu:0'):
    with tf.name_scope('create_inputs'):
      reader = ImageReader(
          args.data_dir,
          args.data_list,
          input_size,
          args.random_scale,
          args.random_mirror,
          args.random_crop,
          args.ignore_label,
          IMG_MEAN_ndsm)

      image_batch, label_batch = reader.dequeue(args.batch_size)

  # Shrink labels to the size of the network output.

  labels = tf.image.resize_nearest_neighbor(
      label_batch, innet_size, name='label_shrink')


  labels_flat = tf.reshape(labels, [-1,])


  # Ignore the location where the label value is larger than args.num_classes.
  not_ignore_pixel = tf.less_equal(labels_flat, args.num_classes-1)

  # Extract the indices of pixel where the gradients are propogated.
  pixel_inds = tf.squeeze(tf.where(not_ignore_pixel), 1)


  labels1 = tf.image.resize_nearest_neighbor(
      label_batch, innet_size1, name='label_shrink1')
  labels_flat1 = tf.reshape(labels1, [-1,])

  # Ignore the location where the label value is larger than args.num_classes.
  not_ignore_pixel1 = tf.less_equal(labels_flat1, args.num_classes-1)

  # Extract the indices of pixel where the gradients are propogated.
  pixel_inds1 = tf.squeeze(tf.where(not_ignore_pixel1), 1)

  labels2 = tf.image.resize_nearest_neighbor(
      label_batch, innet_size2, name='label_shrink2')
  labels_flat2 = tf.reshape(labels2, [-1,])

  # Ignore the location where the label value is larger than args.num_classes.
  not_ignore_pixel2 = tf.less_equal(labels_flat2, args.num_classes-1)

  # Extract the indices of pixel where the gradients are propogated.
  pixel_inds2 = tf.squeeze(tf.where(not_ignore_pixel2), 1)

  # Create network and predictions.
  outputs,outputs1,outputs2 = model(image_batch,
                  args.num_classes,
                  args.is_training,
                  args.use_global_status)

  # Grab variable names which should be restored from checkpoints.nonlocal
  # restore_var = [
  #   v for v in tf.global_variables() if 'block5' not in v.name and 'struct_multi' not in v.name and 'nonlocal' not in v.name and "resnet_v1_101/conv1" not in v.name]

  # Define softmax loss.
  restore_var = [
    v for v in tf.global_variables()]
  # restore_var = [
  #   v for v in tf.global_variables() if "resnet_v1_101/conv1/weights" not in v.name]

  weight1 = [v for v in tf.global_variables() if v.name == "resnet_v1_101/conv1/weights:0"]
  # weight1 = 0
  # Momentum1 = 0
  # for v in tf.global_variables():
  #     if "resnet_v1_101/conv1/weights" in v.name:
  #         print("fuckkkkkkkkkkkkk")
  #         print(v.name)
  #     if v.name == "resnet_v1_101/conv1/weights:0":
  #         global weight1
  #         weight1 = v
  #         print("fuck in weight")
  #     if v.name == "resnet_v1_101/conv1/weights/Momentum":
  #         global Momentum1
  #         Momentum1 = v
  #         print("fuck in Momentum1")

  # Momentum1 = v for v in tf.global_variables() if v.name == "resnet_v1_101/conv1/weights/Momentum"
  # print(weight1)
  # weight1_np = np.load("/home/f523/wangyang/segmentation/Adaptive_Affinity_Fields/snapshot/snapshot_v9/weight.npy")
  # Momentum1_np = np.load("/home/f523/wangyang/segmentation/Adaptive_Affinity_Fields/snapshot/snapshot_v9/Momentum.npy")

  labels_gather = tf.to_int32(tf.gather(labels_flat, pixel_inds))
  labels_gather1 = tf.to_int32(tf.gather(labels_flat1, pixel_inds1))
  labels_gather2 = tf.to_int32(tf.gather(labels_flat2, pixel_inds2))
  seg_losses = []
  seg_losses1 = []
  seg_losses2 = []
  for i,output in enumerate(outputs):
    output_2d = tf.reshape(output, [-1, args.num_classes])
    output_gather = tf.gather(output_2d, pixel_inds)
    #loss,inter_class_loss = loss_total(output_gather,labels_gather)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_gather, labels=labels_gather)
    #loss = focal_loss(labels_gather, output_gather, gamma=2, scope="focal")
    seg_losses.append(loss)
  for i,output1 in enumerate(outputs1):
    output_2d1 = tf.reshape(output1, [-1, args.num_classes])
    output_gather1 = tf.gather(output_2d1, pixel_inds1)
    #loss,inter_class_loss = loss_total(output_gather,labels_gather)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_gather1, labels=labels_gather1)
    #loss = focal_loss(labels_gather1, output_gather1, gamma=2, scope="focal1")
    seg_losses1.append(loss)
  for i,output2 in enumerate(outputs2):
    output_2d2 = tf.reshape(output2, [-1, args.num_classes])
    output_gather2 = tf.gather(output_2d2, pixel_inds2)
    #loss,inter_class_loss = loss_total(output_gather,labels_gather)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_gather2, labels=labels_gather2)
    #loss = loss = focal_loss(labels_gather2, output_gather2, gamma=2, scope="focal")
    seg_losses2.append(loss)

  # Define weight regularization loss.
  w = args.weight_decay
  l2_losses = [w*tf.nn.l2_loss(v) for v in tf.trainable_variables()
                 if 'weights' in v.name]

  # Sum all loss terms.
  mean_seg_loss = tf.reduce_mean(seg_losses)
  mean_seg_loss1 = tf.reduce_mean(seg_losses1)
  mean_seg_loss2 = tf.reduce_mean(seg_losses2)
  mean_l2_loss = tf.reduce_mean(l2_losses)
  reduced_loss = mean_seg_loss + mean_l2_loss +mean_seg_loss1+ mean_seg_loss2


  # Grab variable names which are used for training.
  all_trainable = tf.trainable_variables()
  fc_trainable = [v for v in all_trainable if 'block5' in v.name or 'struct_multi' in v.name or 'nonlocal' in v.name] # lr*10
  base_trainable = [v for v in all_trainable if 'block5' not in v.name and 'struct_multi' not in v.name and 'nonlocal' not in v.name] # lr*1

  # Computes gradients per iteration.
  grads = tf.gradients(reduced_loss, base_trainable+fc_trainable)
  grads_base = grads[:len(base_trainable)]
  grads_fc = grads[len(base_trainable):]

  # Define optimisation parameters.
  base_lr = tf.constant(args.learning_rate)
  learning_rate = tf.scalar_mul(
    base_lr,
    tf.pow((1-step_ph/args.num_steps), args.power))

  opt_base = tf.train.MomentumOptimizer(learning_rate*1.0, args.momentum)
  opt_fc = tf.train.MomentumOptimizer(learning_rate*10.0, args.momentum)

  # Define tensorflow operations which apply gradients to update variables.
  train_op_base = opt_base.apply_gradients(zip(grads_base, base_trainable))
  train_op_fc = opt_fc.apply_gradients(zip(grads_fc, fc_trainable))
  train_op = tf.group(train_op_base, train_op_fc)

  # Process for visualisation.
  print("The input image shape is {}".format(tf.shape(image_batch)))
  with tf.device('/cpu:0'):
    # Image summary for input image, ground-truth label and prediction.
    output_vis = tf.image.resize_nearest_neighbor(
        outputs[-1], tf.shape(image_batch)[1:3,])
    output_vis = tf.argmax(output_vis, axis=3)
    output_vis = tf.expand_dims(output_vis, dim=3)
    output_vis = tf.cast(output_vis, dtype=tf.uint8)

    labels_vis = tf.cast(label_batch, dtype=tf.uint8)

    gt_summary = tf.py_func(
       general.decode_labels,
       [labels_vis,args.num_classes],tf.uint8)
    out_summary = tf.py_func(
        general.decode_labels,
        [output_vis, args.num_classes],tf.uint8)
    # Concatenate image summaries in a row.
    total_summary = tf.summary.image(
        'images',
        tf.concat(axis=2, values=[gt_summary,out_summary]),# gt_summary,out_summary
        max_outputs=args.batch_size)

    # Scalar summary for different loss terms.
    seg_loss_summary = tf.summary.scalar(
        'seg_loss', mean_seg_loss)
    #intra_class_loss_summary = tf.summary.scalar(
    #    'intra_class_loss', intra_class_loss)

    total_summary = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(
        args.snapshot_dir,
        graph=tf.get_default_graph())

  # Set up tf session and initialize variables.
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  init = tf.global_variables_initializer()
  init_local = tf.local_variables_initializer()
  sess.run(init)
  sess.run(init_local)

  # Saver for storing checkpoints of the model.
  saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=0)

  # Load variables if the checkpoint is provided.
  if args.restore_from is not None:
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, args.restore_from)

  # Start queue threads.
  threads = tf.train.start_queue_runners(
      coord=coord, sess=sess)

  # Iterate over training steps.
  # sess.run(tf.assign(weight1[0], weight1_np))
  # sess.run(tf.assign(Momentum1, Momentum1_np))

  pbar = tqdm(range(args.num_steps))
  for step in pbar:
    start_time = time.time()
    feed_dict = {step_ph : step}

    step_loss = 0

    for it in range(args.iter_size):
      # Update summary periodically.
      if it == args.iter_size-1 and step % args.update_tb_every == 0:
        sess_outs = [reduced_loss, total_summary, train_op]
        loss_value, summary, _ = sess.run(sess_outs,
                                          feed_dict=feed_dict)
        summary_writer.add_summary(summary, step)

      else:
        sess_outs = [reduced_loss, train_op]
        loss_value, _ = sess.run(sess_outs, feed_dict=feed_dict)

      step_loss += loss_value


    step_loss /= args.iter_size

    lr = sess.run(learning_rate, feed_dict=feed_dict)

    # Save trained model periodically.
    if step % args.save_pred_every == 0 and step > 0:
      save(saver, sess, args.snapshot_dir, step)

    duration = time.time() - start_time
    desc = 'loss = {:.3f}, lr = {:.6f}'.format(step_loss,lr)
    pbar.set_description(desc)

  coord.request_stop()
  coord.join(threads)

if __name__ == '__main__':
  main()
