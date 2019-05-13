import tensorflow as tf
import sys
sys.path.append("/home/f523/wangyang/segmentation/Adaptive_Affinity_Fields")
import network.common.layers as nn
from network.common.resnet_v1_1 import resnet_v1_101 as pspnet_builder
from network.common.pymaid import nonlocal_dot
def cab(batch_input,name,is_training,use_global_status):
    with tf.variable_scope(name) as scope:
        shape = batch_input.get_shape().as_list()
        global_avg_pool = tf.nn.avg_pool(batch_input, ksize=[1, batch_input.get_shape().as_list()[1], batch_input.get_shape().as_list()[2], 1], strides=[1, 		   batch_input.get_shape().as_list()[1], batch_input.get_shape().as_list()[2], 1], padding='VALID')
        conv_1 = nn.conv(global_avg_pool,name='conv1',filters=shape[-1]/4,kernel_size=1,strides=1,padding='VALID',biased=False,bn=False,relu=True,
                    is_training=is_training,use_global_status=use_global_status)
        conv_2 = nn.conv(conv_1,name='conv2',filters=shape[-1],kernel_size=1,strides=1,padding='VALID',biased=False,bn=False,relu=False,
                    is_training=is_training,use_global_status=use_global_status)
        activation_2 = tf.sigmoid(conv_2)
        mul = tf.multiply(batch_input, activation_2)

    return  mul

def pspnet_v2(x,name,num_classes,is_training,use_global_status,reuse=False):

    # Ensure that the size of input data is valid (should be multiple of 6x8=48).
    h, w = x.get_shape().as_list()[1:3] # NxHxWxC
    assert(h%48 == 0 and w%48 == 0 and h == w)

    # Build the base network.
    res0,res1,res2,res3,res4 = pspnet_builder(x, name,is_training=is_training, use_global_status=use_global_status, reuse=reuse)

    with tf.variable_scope('struct_multi', reuse=reuse) as scope:
        cab0 = cab(res0,'res0',is_training,use_global_status)
        cab0_conv = nn.conv(cab0,name='cab0_conv',filters=256,kernel_size=3,strides=2,padding='VALID',biased=False,bn=True,relu=True,
                    is_training=is_training,use_global_status=use_global_status)
        cab1 = cab(res1,'res1',is_training,use_global_status)
        cab1_conv = nn.conv(cab1,name='cab1_conv',filters=256,kernel_size=3,strides=1,padding='SAME',biased=False,bn=True,relu=True,
                    is_training=is_training,use_global_status=use_global_status)
        cab2 = cab(res2,'res2',is_training,use_global_status)
        cab2_conv = nn.conv(cab2,name='cab2_conv',filters=256,kernel_size=3,strides=1,padding='SAME',biased=False,bn=True,relu=True,
                    is_training=is_training,use_global_status=use_global_status)
        cab3 = cab(res3,'res3',is_training,use_global_status)
        cab3_conv = nn.conv(cab3,name='cab3_conv',filters=256,kernel_size=3,strides=1,padding='SAME',biased=False,bn=True,relu=True,
                    is_training=is_training,use_global_status=use_global_status)
        cab4 = cab(res4,'res4',is_training,use_global_status)
        cab4_conv = nn.conv(cab4,name='cab4_conv',filters=256,kernel_size=3,strides=1,padding='SAME',biased=False,bn=True,relu=True,
                    is_training=is_training,use_global_status=use_global_status)

        cab_cat = tf.concat([cab0_conv,cab1_conv,cab2_conv,cab3_conv,cab4_conv],axis=3)

        cab_out = nn.conv(cab_cat,name='cab_out',filters=512,kernel_size=1,strides=1,padding='VALID',biased=False,bn=True,relu=True,
                    is_training=is_training,use_global_status=use_global_status)

    with tf.variable_scope(name, reuse=reuse) as scope:
    # Build the PSP module
        pool_k = int(h/8) # the base network is stride 8 by default.

    # Build pooling layer results in 1x1 output.
        pool1 = tf.nn.avg_pool(res4,
                           name='block5/pool1',
                           ksize=[1,pool_k,pool_k,1],
                           strides=[1,pool_k,pool_k,1],
                           padding='VALID')
        pool1 = nn.conv(pool1,
                    'block5/pool1/conv1',
                    512,
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

    # Build pooling layer results in 2x2 output.
        pool2 = tf.nn.avg_pool(res4,
                           name='block5/pool2',
                           ksize=[1,pool_k//2,pool_k//2,1],
                           strides=[1,pool_k//2,pool_k//2,1],
                           padding='VALID')
        pool2 = nn.conv(pool2,
                    'block5/pool2/conv1',
                    512,
                    1,
                    1,
                    padding='SAME',
                    biased=False,
                    bn=True,
                    relu=True,
                    is_training=is_training,
                    decay=0.99,
                    use_global_status=use_global_status)
        pool2 = tf.image.resize_bilinear(pool2, [pool_k, pool_k])

    # Build pooling layer results in 3x3 output.
        pool3 = tf.nn.avg_pool(res4,
                           name='block5/pool3',
                           ksize=[1,pool_k//3,pool_k//3,1],
                           strides=[1,pool_k//3,pool_k//3,1],
                           padding='VALID')
        pool3 = nn.conv(pool3,
                    'block5/pool3/conv1',
                    512,
                    1,
                    1,
                    padding='SAME',
                    biased=False,
                    bn=True,
                    relu=True,
                    is_training=is_training,
                    decay=0.99,
                    use_global_status=use_global_status)
        pool3 = tf.image.resize_bilinear(pool3, [pool_k, pool_k])

    # Build pooling layer results in 6x6 output.
        pool6 = tf.nn.avg_pool(res4,
                           name='block5/pool6',
                           ksize=[1,pool_k//6,pool_k//6,1],
                           strides=[1,pool_k//6,pool_k//6,1],
                           padding='VALID')
        pool6 = nn.conv(pool6,
                    'block5/pool6/conv1',
                    512,
                    1,
                    1,
                    padding='SAME',
                    biased=False,
                    bn=True,
                    relu=True,
                    is_training=is_training,
                    decay=0.99,
                    use_global_status=use_global_status)
        pool6 = tf.image.resize_bilinear(pool6, [pool_k, pool_k])
        #nonlocal1 = nonlocal_dot(res4, 512, embed=True, softmax=True, scope="nonlocal/scale_1",scale=1)
    # Fuse the pooled feature maps with its input, and generate
    # segmentation prediction.


        #pool_cat = tf.concat([pool1, pool2, pool3, pool6],axis=3)

        #pool_out = nn.conv(pool_cat,name='block5/pool_out',filters=512,kernel_size=1,strides=1,padding='VALID',biased=False,bn=True,relu=True,
        #            is_training=is_training,use_global_status=use_global_status)
        #x = tf.concat([pool_out, res4, cab_out],name='block5/concat',axis=3)
        x = tf.concat([pool1, pool2, pool3, pool6, res4, cab_out],name='block5/concat',axis=3)
        x = nn.conv(x,
                'block5/conv2',
                512,
                3,
                1,
                padding='SAME',
                biased=False,
                bn=True,
                relu=True,
                is_training=is_training,
                decay=0.99,
                use_global_status=use_global_status)
        x = nn.conv(x,
                'block5/fc1_voc12',
                num_classes,
                1,
                1,
                padding='SAME',
                biased=True,
                bn=False,
                relu=False,
                is_training=is_training)

    return x

def pspnet_v2_resnet101(x,
                     num_classes,
                     is_training,
                     use_global_status,
                     reuse=False):
  """Helper function to build PSPNet model for semantic segmentation.

  The PSPNet model is composed of one base network (ResNet101) and
  one pyramid spatial pooling (PSP) module, followed with concatenation
  and two more convlutional layers for segmentation prediction.

  Args:
    x: A tensor of size [batch_size, height_in, width_in, channels].
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

  scores = []
  with tf.name_scope('scale_0') as scope:
    score = pspnet_v2(
        x,
        'resnet_v1_101',
        num_classes,
        is_training,
        use_global_status,
        reuse=reuse)

    scores.append(score)

  return scores
