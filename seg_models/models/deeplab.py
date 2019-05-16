import tensorflow as tf

from network.common.resnet_v1 import resnet_v1_101
import network.common.layers as nn

def _deeplab_builder(x,
                     name,
                     cnn_fn,
                     num_classes,
                     is_training,
                     use_global_status,
                     reuse=False):
  """Helper function to build Deeplab v2 model for semantic segmentation.

  The Deeplab v2 model is composed of one base network (ResNet101) and
  one ASPP module (4 Atrous Convolutional layers of different size). The
  segmentation prediction is the summation of 4 outputs of the ASPP module.

  Args:
    x: A tensor of size [batch_size, height_in, width_in, channels].
    name: The prefix of tensorflow variables defined in this network.
    cnn_fn: A function which builds the base network (ResNet101).
    num_classes: Number of predicted classes for classification tasks.
    is_training: If the tensorflow variables defined in this network
      would be used for training.
    use_global_status: enable/disable use_global_status for batch
      normalization. If True, moving mean and moving variance are updated
      by exponential decay.
    reuse: enable/disable reuse for reusing tensorflow variables. It is
      useful for sharing weight parameters across two identical networks.

  Returns:
    A tensor of size [batch_size, height_in/8, width_in/8, num_classes].
  """

  # Build the base network.
  h, w = x.get_shape().as_list()[1:3]
  x = cnn_fn(x, name, is_training, use_global_status, reuse)
  pool_k = int(h/8)
  with tf.variable_scope(name, reuse=reuse) as scope:
    # Build the ASPP module.    

    score1 = nn.atrous_conv(
          x,
          name='block5/fc1_c{:d}'.format(0),
          filters=256,
          kernel_size=3,
          dilation=12,
          padding='SAME',
          relu=False,
          biased=True,
          bn=False,
          is_training=is_training)
    score2 = nn.atrous_conv(
          x,
          name='block5/fc1_c{:d}'.format(1),
          filters=256,
          kernel_size=3,
          dilation=24,
          padding='SAME',
          relu=False,
          biased=True,
          bn=False,
          is_training=is_training)

    score3 = nn.atrous_conv(
          x,
          name='block5/fc1_c{:d}'.format(2),
          filters=256,
          kernel_size=3,
          dilation=36,
          padding='SAME',
          relu=False,
          biased=True,
          bn=False,
          is_training=is_training)
    pool1 = tf.nn.avg_pool(x,
                           name='block5/pool1',
                           ksize=[1,pool_k,pool_k,1],
                           strides=[1,pool_k,pool_k,1],
                           padding='VALID')
    pool1 = nn.conv(pool1,
                    'block5/pool1/conv1',
                    256,
                    1,
                    1,
                    padding='SAME',
                    biased=False,
                    bn=True,
                    relu=True,
                    is_training=is_training,
                    decay=0.99,
                    use_global_status=use_global_status)
    pool1 = tf.image.resize_bilinear(pool1, [pool_k, pool_k])
    pool2 = nn.conv(x,
                    'block5/pool2/conv1',
                    256,
                    1,
                    1,
                    padding='SAME',
                    biased=False,
                    bn=True,
                    relu=True,
                    is_training=is_training,
                    decay=0.99,
                    use_global_status=use_global_status)

    score = tf.concat([score1,score2,score3,pool1,pool2], axis=3,name='fc1_sum')
    score = nn.conv(score,
                    'block5/conv5',
                    256,
                    1,
                    1,
                    padding='SAME',
                    biased=False,
                    bn=True,
                    relu=True,
                    is_training=is_training,
                    decay=0.99,
                    use_global_status=use_global_status)
    score = nn.conv(score,
                    'block5/fc',
                    num_classes,
                    1,
                    1,
                    padding='SAME',
                    biased=False,
                    bn=True,
                    relu=True,
                    is_training=is_training,
                    decay=0.99,
                    use_global_status=use_global_status)
  return score


def deeplab_resnet101(x,
                      num_classes,
                      is_training,
                      use_global_status,
                      reuse=False):
  """Builds Deeplab v2 based on ResNet101.

  Args:
    x: A tensor of size [batch_size, height_in, width_in, channels].
    name: The prefix of tensorflow variables defined in this network.
    num_classes: Number of predicted classes for classification tasks.
    is_training: If the tensorflow variables defined in this network
      would be used for training.
    use_global_status: enable/disable use_global_status for batch
      normalization. If True, moving mean and moving variance are updated
      by exponential decay.
    reuse: enable/disable reuse for reusing tensorflow variables. It is
      useful for sharing weight parameters across two identical networks.

  Returns:
    A tensor of size [batch_size, height_in/8, width_in/8, num_classes].
  """
  h, w = x.get_shape().as_list()[1:3] # NxHxWxC

  scores = []
  for i,scale in enumerate([1]):
    with tf.name_scope('scale_{:d}'.format(i)) as scope:
      x_in = x

      score = _deeplab_builder(
          x_in,
          'resnet_v1_101',
          resnet_v1_101,
          num_classes,
          is_training,
          use_global_status,
          reuse=reuse)

      scores.append(score)

  return scores
