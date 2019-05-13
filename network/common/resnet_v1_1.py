import tensorflow as tf

import network.common.layers as nn


def bottleneck(x,
               name,
               filters,
               strides=None,
               dilation=None,
               is_training=True,
               use_global_status=True):
  """Builds the bottleneck module in ResNet.

  This function stack 3 convolutional layers and fuse the output with
  the residual connection.

  Args:
    x: A tensor of size [batch_size, height_in, width_in, channels].
    name: The prefix of tensorflow variables defined in this layer.
    filters: A number indicating the number of output channels.
    strides: A number indicating the stride of the sliding window for
      height and width.
    dilation: A number indicating the dilation factor for height and width.
    is_training: If the tensorflow variables defined in this layer
      would be used for training.
    use_global_status: enable/disable use_global_status for batch
      normalization. If True, moving mean and moving variance are updated
      by exponential decay.

  Returns:
    A tensor of size [batch_size, height_out, width_out, channels_out].
  """
  if strides is None and dilation is None:
    raise ValueError('None of strides or dilation is specified, '
                     +'set one of them to 1 or bigger number.')
  elif strides > 1 and dilation is not None and dilation > 1:
    raise ValueError('strides and dilation are both specified, '
                     +'set one of them to 1 or None.')

  with tf.variable_scope(name) as scope:
    c_i = x.get_shape().as_list()[-1]

    if c_i != filters*4:
      # Use a convolutional layer as residual connection when the
      # number of input channels is different from output channels.
      shortcut = nn.conv(x,
                         name='shortcut',
                         filters=filters*4,
                         kernel_size=1,
                         strides=strides,
                         padding='VALID',
                         biased=False,
                         bn=True,
                         relu=False,
                         is_training=is_training,
                         use_global_status=use_global_status)
    elif strides > 1:
      # Use max-pooling as residual connection when the number of
      # input channel is same as output channels, but stride is
      # larger than 1.
      shortcut = nn.max_pool(x,
                             name='shortcut',
                             kernel_size=1,
                             strides=strides,
                             padding='VALID')
    else:
      # Otherwise, keep the original input as residual connection.
      shortcut = x

    # Build the 1st convolutional layer.
    x = nn.conv(x,
                name='conv1',
                filters=filters,
                kernel_size=1,
                strides=1,
                padding='SAME',
                biased=False,
                bn=True,
                relu=True,
                is_training=is_training,
                use_global_status=use_global_status)

    if dilation is not None and dilation > 1:
      # If dilation > 1, apply atrous conv to the 2nd convolutional layer.
      x = nn.atrous_conv(
          x,
          name='conv2',
          filters=filters,
          kernel_size=3,
          dilation=dilation,
          padding='SAME',
          biased=False,
          bn=True,
          relu=True,
          is_training=is_training,
          use_global_status=use_global_status)
    else:
      padding = 'VALID' if strides > 1 else 'SAME'
      x = nn.conv(
          x,
          name='conv2',
          filters=filters,
          kernel_size=3,
          strides=strides,
          padding=padding,
          biased=False,
          bn=True,
          relu=True,
          is_training=is_training,
          use_global_status=use_global_status)

    # Build the 3rd convolutional layer (increase the channels).
    x = nn.conv(x,
                name='conv3',
                filters=filters*4,
                kernel_size=1,
                strides=1,
                padding='SAME',
                biased=False,
                bn=True,
                relu=False,
                is_training=is_training,
                use_global_status=use_global_status)

    # Fuse the convolutional outputs with residual connection.
    x = tf.add_n([x, shortcut], name='add')
    x = tf.nn.relu(x, name='relu')

  return x


def resnet_v1(x,
              name,
              filters=[64,128,256,512],
              num_blocks=[3,4,23,3],
              strides=[2,1,1,1],
              dilations=[None, None, 2, 2],
              is_training=True,
              use_global_status=True,
              reuse=False):
  """Helper function to build ResNet.

  Args:
    x: A tensor of size [batch_size, height_in, width_in, channels].
    name: The prefix of tensorflow variables defined in this network.
    filters: A list of numbers indicating the number of output channels
      (The output channels would be 4 times to the numbers).
    strides: A list of numbers indicating the stride of the sliding window for
      height and width.
    dilation: A number indicating the dilation factor for height and width.
    is_training: If the tensorflow variables defined in this layer
      would be used for training.
    use_global_status: enable/disable use_global_status for batch
      normalization. If True, moving mean and moving variance are updated
      by exponential decay.
    reuse: enable/disable reuse for reusing tensorflow variables. It is
      useful for sharing weight parameters across two identical networks.

  Returns:
    A tensor of size [batch_size, height_out, width_out, channels_out].
  """
  if len(filters) != len(num_blocks) or len(filters) != len(strides):
    raise ValueError('length of lists are not consistent')

  with tf.variable_scope(name, reuse=reuse) as scope:
    # Build conv1.
    x = nn.conv(x,
                name='conv1',
                filters=64,
                kernel_size=7,
                strides=2,
                padding='VALID',
                biased=False,
                bn=True,
                relu=True,
                is_training=is_training,
                use_global_status=use_global_status)

    # Build pool1.
    x = nn.max_pool(x,
                    name='pool1',
                    kernel_size=3,
                    strides=2,
                    padding='VALID')

    res0 = x
    for iu in range(num_blocks[0]):
        name_format = 'block{:d}/unit_{:d}/bottleneck_v1'
        block_name = name_format.format(1, iu+1)

        c_o = filters[0] # output channel
        s = strides[0] if iu == num_blocks[0]-1 else 1
        d = dilations[0]
        x = bottleneck(x,name=block_name,filters=c_o,strides=s,dilation=d,is_training=is_training,use_global_status=use_global_status)
    res1 = x
    for iu in range(num_blocks[1]):
        name_format = 'block{:d}/unit_{:d}/bottleneck_v1'
        block_name = name_format.format(2, iu+1)

        c_o = filters[1] # output channel
        s = strides[1] if iu == num_blocks[1]-1 else 1
        d = dilations[1]
        x = bottleneck(x,name=block_name,filters=c_o,strides=s,dilation=d,is_training=is_training,use_global_status=use_global_status)
    res2 = x
    for iu in range(num_blocks[2]):
        name_format = 'block{:d}/unit_{:d}/bottleneck_v1'
        block_name = name_format.format(3, iu+1)

        c_o = filters[2] # output channel
        s = strides[2] if iu == num_blocks[2]-1 else 1
        d = dilations[2]
        x = bottleneck(x,name=block_name,filters=c_o,strides=s,dilation=d,is_training=is_training,use_global_status=use_global_status)
    res3 = x
    for iu in range(num_blocks[3]):
        name_format = 'block{:d}/unit_{:d}/bottleneck_v1'
        block_name = name_format.format(4, iu+1)

        c_o = filters[3] # output channel
        s = strides[3] if iu == num_blocks[3]-1 else 1
        d = dilations[3]
        x = bottleneck(x,name=block_name,filters=c_o,strides=s,dilation=d,is_training=is_training,use_global_status=use_global_status)
    res4 = x

    # Build residual bottleneck blocks.
    """
    for ib in range(len(filters)):
      for iu in range(num_blocks[ib]):
        name_format = 'block{:d}/unit_{:d}/bottleneck_v1'
        block_name = name_format.format(ib+1, iu+1)

        c_o = filters[ib] # output channel
        # Apply strides to the last block.
        s = strides[ib] if iu == num_blocks[ib]-1 else 1
        d = dilations[ib]
        x = bottleneck(x,
                       name=block_name,
                       filters=c_o,
                       strides=s,
                       dilation=d,
                       is_training=is_training,
                       use_global_status=use_global_status)
                       """
    return res0,res1,res2,res3,res4


def resnet_v1_101(x,
                  name,
                  is_training,
                  use_global_status,
                  reuse=False):
  """Builds ResNet101 v1.

  Args:
    x: A tensor of size [batch_size, height_in, width_in, channels].
    name: The prefix of tensorflow variables defined in this network.
    is_training: If the tensorflow variables defined in this layer
      would be used for training.
    use_global_status: enable/disable use_global_status for batch
      normalization. If True, moving mean and moving variance are updated
      by exponential decay.
    reuse: enable/disable reuse for reusing tensorflow variables. It is
      useful for sharing weight parameters across two identical networks.

  Returns:
    A tensor of size [batch_size, height_out, width_out, channels_out].
  """
  return resnet_v1(x,
                   name=name,
                   filters=[64,128,256,512],
                   num_blocks=[3,4,23,3],
                   strides=[2,1,1,1],
                   dilations=[None, None, 4, 2],
                   is_training=is_training,
                   use_global_status=use_global_status,
                   reuse=reuse)
