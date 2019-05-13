from __future__ import print_function

import argparse
import os
import time
import math
import sys
import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc
from PIL import Image
sys.path.append("/home/wangyang/Desktop/Semantic Segmentation Code/Adaptive_Affinity_Fields")
from seg_models.models.pspnet import pspnet_resnet101 as model
from seg_models.image_reader import ImageReader
import utils.general

IMG_MEAN = np.array((122.675, 116.669, 104.008), dtype=np.float32)


def get_arguments():
  """Parse all the arguments provided from the CLI.

  Returns:
    A list of parsed arguments.
  """
  parser = argparse.ArgumentParser(
      description='Inference for Semantic Segmentation')
  parser.add_argument('--data-dir', type=str, default='/home/wangyang/Desktop/dataset/VOCdevkit/',#/home/wangyang/Desktop/dataset/VOCdevkit/
                      help='/path/to/dataset.')
  parser.add_argument('--data-list', type=str, default='/home/wangyang/Desktop/Semantic Segmentation Code/Adaptive_Affinity_Fields/dataset/voc12/val.txt',
                      help='/path/to/datalist/file.')
  parser.add_argument('--input-size', type=str, default='336,336',
                      help='Comma-separated string with H and W of image.')
  parser.add_argument('--strides', type=str, default='336,336',
                      help='Comma-separated string with strides of H and W.')
  parser.add_argument('--num-classes', type=int, default=21,
                      help='Number of classes to predict.')
  parser.add_argument('--ignore-label', type=int, default=255,
                      help='Index of label to ignore.')
  parser.add_argument('--restore-from', type=str, default='/home/wangyang/Desktop/Semantic Segmentation Code/Adaptive_Affinity_Fields/snapshot/model.ckpt-10000',
                      help='Where restore model parameters from.')
  parser.add_argument('--save-dir', type=str, default='/home/wangyang/Desktop/Semantic Segmentation Code/Adaptive_Affinity_Fields/inference_results',
                      help='/path/to/save/predictions.')
  parser.add_argument('--colormap', type=str, default=None,
                      help='/path/to/colormap/file.')

  return parser.parse_args()

def load(saver, sess, ckpt_path):
  """Load the trained weights.

  Args:
    saver: TensorFlow saver object.
    sess: TensorFlow session.
    ckpt_path: path to checkpoint file with parameters.
  """
  saver.restore(sess, ckpt_path)
  print('Restored model parameters from {}'.format(ckpt_path))

def parse_commastr(str_comma):
  """Read comma-sperated string.
  """
  if '' == str_comma:
    return None
  else:
    a, b =  map(int, str_comma.split(','))

  return [a,b]

def main():
  """Create the model and start the Inference process.
  """
  args = get_arguments()

  # Parse image processing arguments.
  input_size = parse_commastr(args.input_size)
  strides = parse_commastr(args.strides)
  assert(input_size is not None and strides is not None)
  h, w = input_size
  innet_size = (int(math.ceil(h/8)), int(math.ceil(w/8)))


  # Create queue coordinator.
  coord = tf.train.Coordinator()

  # Load the data reader.
  with tf.name_scope('create_inputs'):
    reader = ImageReader(
        args.data_dir,
        args.data_list,
        None,
        False, # No random scale.
        False, # No random mirror.
        False, # No random crop, center crop instead
        args.ignore_label,
        IMG_MEAN)
    image = reader.image
    image_list = reader.image_list
  image_batch = tf.expand_dims(image, dim=0)

  # Create input tensor to the Network.
  crop_image_batch = tf.placeholder(
      name='crop_image_batch',
      shape=[1,input_size[0],input_size[1],3],
      dtype=tf.float32)

  # Create network and output prediction.
  outputs = model(crop_image_batch,
                  args.num_classes,
                  False,
                  True)

  # Grab variable names which should be restored from checkpoints.
  restore_var = [
    v for v in tf.global_variables() if 'crop_image_batch' not in v.name]

  # Output predictions.
  output = outputs[-1]
  output = tf.image.resize_bilinear(
      output,
      tf.shape(crop_image_batch)[1:3,])
  output = tf.nn.softmax(output, dim=3)

  # Set up tf session and initialize variables.
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  init = tf.global_variables_initializer()

  sess.run(init)
  sess.run(tf.local_variables_initializer())

  # Load weights.
  loader = tf.train.Saver(var_list=restore_var)
  if args.restore_from is not None:
    load(loader, sess, args.restore_from)

  # Start queue threads.
  threads = tf.train.start_queue_runners(coord=coord, sess=sess)

  # Get colormap.
  #map_data = scipy.io.loadmat(args.colormap)
  #key = os.path.basename(args.colormap).replace('.mat','')
  #colormap = map_data[key]
  #colormap *= 255
  #colormap = colormap.astype(np.uint8)

  # Create directory for saving predictions.
  pred_dir = os.path.join(args.save_dir, 'gray')
  color_dir = os.path.join(args.save_dir, 'color')
  if not os.path.isdir(pred_dir):
    os.makedirs(pred_dir)
  if not os.path.isdir(color_dir):
    os.makedirs(color_dir)

  # Iterate over testing steps.
  with open(args.data_list, 'r') as listf:
    num_steps = len(listf.read().split('\n'))-1

  for step in range(num_steps):
    img_batch = sess.run(image_batch)
    img_size = img_batch.shape
    padimg_size = list(img_size) # deep copy of img_size

    padimg_h, padimg_w = padimg_size[1:3]
    input_h, input_w = input_size

    if input_h > padimg_h:
      padimg_h = input_h
    if input_w > padimg_w:
      padimg_w = input_w

    # Update padded image size.
    padimg_size_divisible = padimg_size
    padimg_size_divisible[1] = math.ceil(padimg_h/input_h)*input_h
    padimg_size_divisible[2] = math.ceil(padimg_w/input_w)*input_w
    padimg_size[1] = padimg_h
    padimg_size[2] = padimg_w
    padimg_batch = np.zeros(padimg_size_divisible, dtype=np.float32)
    img_h, img_w = img_size[1:3]
    padimg_batch[:, :img_h, :img_w, :] = img_batch

    # Create padded label array.
    lab_size = list(padimg_size_divisible)
    lab_size[-1] = args.num_classes
    lab_batch = np.zeros(lab_size, dtype=np.float32)
    lab_batch.fill(args.ignore_label)


    npatches_h = math.ceil(padimg_h/input_h)
    npatches_w = math.ceil(padimg_w/input_w)

    # Crate the ending index of each patch.


    for indh in range(npatches_h):
      for indw in range(npatches_w):
        sh, eh = indh*input_h, (indh+1)*input_h-1 # start&end ind of H
        sw, ew = indw*input_w, (indw+1)*input_w-1 # start&end ind of W
        cropimg_batch = padimg_batch[:, sh:eh, sw:ew, :]
        feed_dict = {crop_image_batch: cropimg_batch}

        out = sess.run(output, feed_dict=feed_dict)
        lab_batch[:, sh:eh, sw:ew, :] += out

    lab_batch = lab_batch[0, :img_h, :img_w, :]
    lab_batch = np.argmax(lab_batch, axis=-1)
    lab_batch = lab_batch.astype(np.uint8)

    basename = os.path.basename(image_list[step])
    basename = basename.replace('jpg', 'png')

    predname = os.path.join(pred_dir, basename)
    Image.fromarray(lab_batch, mode='L').save(predname)

    colorname = os.path.join(color_dir, basename)
    color = utils.general.decode_labels_inference(lab_batch)
    Image.fromarray(color, mode='RGB').save(colorname)

  coord.request_stop()
  coord.join(threads)

if __name__ == '__main__':
    main()
