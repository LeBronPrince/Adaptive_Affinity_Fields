import os
import tensorflow as tf
from PIL import Image
import numpy as np
import scipy.io


#LABEL_COLORS = scipy.io.loadmat('misc/colormapvoc.mat')['colormapvoc']
#LABEL_COLORS *= 255
#LABEL_COLORS = LABEL_COLORS.astype(np.uint8)

#label_colours = [(255, 255, 255), (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)]
label_colours = [(255, 255, 255), (255, 0, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255), (0, 0, 255)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor(0, 255, 255)



def decode_labels(mask, num_classes):
    n, h, w, c = mask.shape
    outputs = np.zeros((n, h, w, 3), dtype=np.uint8)
    for i in range(n):
      img = Image.new('RGB', (h, w))
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :, 0]):
          for k_, k in enumerate(j):
              if k < num_classes:
                  pixels[k_,j_] = label_colours[k]
      outputs[i] = np.array(img)
    return outputs

def inv_preprocess(imgs, img_mean):
  """Inverses image preprocessing of the input images.

  This function adds back the mean vector and convert BGR to RGB.

  Args:
    imgs: A tensor of size [batch_size, height_in, width_in, 3]
    img_mean: A 1-D tensor indicating the vector of mean colour values.

  Returns:
    A tensor of size [batch_size, height_in, width_in, 3]
  """
  n, h, w, c = imgs.shape
  outputs = np.zeros((n, h, w, c), dtype=np.uint8)
  for i in range(n):
    outputs[i] = (imgs[i] + img_mean).astype(np.uint8)

  return outputs

def decode_labels_inference(mask,num_classes=6):
    h, w = mask.shape

    outputs = np.zeros((h, w, 3), dtype=np.uint8)

    for i,i_ in enumerate(mask):
        for j,j_ in enumerate(i_):
            if j_ < num_classes and j_>=0:
                outputs[i,j] = label_colours[j_]
    #outputs = np.array(img)
    print('output shape is {}'.format(outputs.shape))
    return outputs


def make_dis_input(G_logits, imgs, labels, num_classes):

    logits_shape = tf.shape(G_logits)
    downsampling_shape = [logits_shape[1], logits_shape[2]]
    labels = tf.image.resize_nearest_neighbor(labels,downsampling_shape)
    labels = tf.cast(labels, dtype=tf.int32)
    imgs = tf.image.resize_images(imgs, size = downsampling_shape)
    # 1. one hot representation of labels
    G_probs = tf.nn.softmax(G_logits, name='softmax_tensor')
    batch = tf.cast(tf.shape(labels)[0], dtype=tf.int32)
    height = tf.cast(tf.shape(labels)[1], dtype=tf.int32)
    width = tf.cast(tf.shape(labels)[2], dtype=tf.int32)
    one_hot_flat_y = tf.one_hot(tf.reshape(labels, [-1, ]), num_classes, axis=1)
    one_hot_y = tf.reshape(one_hot_flat_y,[batch, height, width, num_classes])
    # define operations between generator and discriminator - version "product"


    # 2. Slice r,g,b components
    blue = tf.slice(imgs, [0,0,0,0], [1, height, width, 1])
    green = tf.slice(imgs, [0, 0, 0, 1], [1, height, width, 1])
    red = tf.slice(imgs, [0, 0, 0, 2], [1, height, width, 1])


    # 3. Generate fake discriminator input
    product_b = G_probs * blue
    product_g = G_probs * green
    product_r = G_probs * red

    fake_disciminator_input = tf.concat([product_b, product_g, product_r], axis=-1)
    #fake_disciminator_input = tf.squeeze(fake_disciminator_input,squeeze_dims=0)
    print(fake_disciminator_input.shape)

    # 4. Generate also real discriminator input
    product_b = one_hot_y * blue
    product_g = one_hot_y * green
    product_r = one_hot_y * red
    real_disciminator_input = tf.concat([product_b, product_g, product_r], axis=-1)
    #real_disciminator_input = tf.squeeze(real_disciminator_input,squeeze_dims=0)

    return real_disciminator_input, fake_disciminator_input

def DLoss(real_D_logits, fake_D_logits):
    in_shape = tf.shape(fake_D_logits)
    batch_size = tf.cast(in_shape[0], dtype=tf.int32)
    height = tf.cast(in_shape[1], dtype=tf.int32)
    width = tf.cast(in_shape[2], dtype=tf.int32)

    # Reshape logits by num of classes
    fake_D_logits_by_num_classes = tf.reshape(fake_D_logits, [-1, 2])
    real_D_logits_by_num_classes = tf.reshape(real_D_logits, [-1, 2])

    # Define real/fake labels
    label_real = tf.cast(tf.fill([batch_size*height*width], 1.0), dtype=tf.int32)
    label_fake = tf.cast(tf.fill([batch_size*height*width], 0.0), dtype=tf.int32)

    # Compute loss
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=real_D_logits_by_num_classes,
                                                                          labels=label_real,
                                                                          name="bce_1"))
    loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fake_D_logits_by_num_classes,
                                                                          labels=label_fake,
                                                                          name="bce_2"))
    total_loss = loss
    return total_loss

def GLoss(labels, G_logits, fake_D_logits,num_classes,loss_bce_weight):
    logits_by_num_classes = tf.reshape(G_logits, [-1, num_classes])
    preds = tf.argmax(G_logits, axis=3, output_type=tf.int32)
    preds_flat = tf.reshape(preds, [-1, ])
    labels_flat = tf.reshape(labels, [-1, ])
    print(preds_flat.shape)
    print(labels_flat.shape)
    valid_indices = tf.multiply(tf.to_int32(labels_flat <= num_classes - 1), tf.to_int32(labels_flat > -1))

    # Prepare segmentation model logits and labels
    valid_logits = tf.dynamic_partition(logits_by_num_classes, valid_indices, num_partitions=2)[1]
    valid_labels = tf.dynamic_partition(labels_flat, valid_indices, num_partitions=2)[1]
    valid_preds = tf.dynamic_partition(preds_flat, valid_indices, num_partitions=2)[1]

    in_shape = tf.shape(fake_D_logits)
    batch_size = tf.cast(in_shape[0], dtype=tf.int32)
    height = tf.cast(in_shape[1], dtype=tf.int32)
    width = tf.cast(in_shape[2], dtype=tf.int32)

    fake_D_logits_by_num_classes = tf.reshape(fake_D_logits, [-1, 2])
    label_real = tf.cast(tf.fill([batch_size*height*width], 1.0), dtype=tf.int32)

    l_mce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=valid_logits,
                                                           labels=valid_labels,
                                                           name="g_mce"))
    l_bce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fake_D_logits_by_num_classes,
                                                                          labels=label_real,
                                                                          name="l_bce"))
    loss = l_mce + loss_bce_weight * l_bce

    total_loss = loss

    return total_loss,l_bce

def snapshot_arg(args):
  """Print and snapshots Command-Line arguments to a text file.
  """
  snap_dir = args.snapshot_dir
  dictargs = vars(args)

  print('-----------------------------------------------')
  print('-----------------------------------------------')
  with open(os.path.join(snap_dir, 'config'), 'w') as argsfile:
    for key, val in dictargs.items():
      line = '| {0} = {1}'.format(key, val)
      print(line)
      argsfile.write(line+'\n')
  print('-----------------------------------------------')
  print('-----------------------------------------------')
