import tensorflow as tf
import numpy as np
import network.common.layers as nn

def d_net(x,num_classes,is_training,use_global_status,reuse=False):
    initializer = tf.random_normal_initializer(stddev=0.02)
    with tf.variable_scope('d_net',reuse=reuse) as scope:
        x = nn.conv(x,name='conv0',filters=64,kernel_size=3, strides=1, padding='SAME',biased=False,bn=True,relu=True,
                    is_training=is_training,use_global_status=use_global_status)
        x = nn.conv(x,name='conv1',filters=128,kernel_size=3, strides=2, padding='SAME',biased=False,bn=True,relu=True,
                    is_training=is_training,use_global_status=use_global_status)
        x = nn.conv(x,name='conv2',filters=256,kernel_size=3, strides=1, padding='SAME',biased=False,bn=True,relu=True,
                    is_training=is_training,use_global_status=use_global_status)
        x = nn.conv(x,name='conv3',filters=512,kernel_size=3, strides=2, padding='SAME',biased=False,bn=True,relu=True,
                    is_training=is_training,use_global_status=use_global_status)
        x = nn.conv(x,name='conv4',filters=1024,kernel_size=3, strides=1, padding='SAME',biased=False,bn=True,relu=True,
                    is_training=is_training,use_global_status=use_global_status)
        x = nn.conv(x,name='conv5',filters=1024,kernel_size=3, strides=1, padding='SAME',biased=False,bn=True,relu=True,
                    is_training=is_training,use_global_status=use_global_status)

        x = tf.layers.conv2d_transpose(x, 512, kernel_size=3, strides=(2, 2), padding="same", kernel_initializer=initializer,name='conv6')

        x = nn.conv(x,name='conv7',filters=256,kernel_size=3, strides=1, padding='SAME',biased=False,bn=True,relu=True,
                    is_training=is_training,use_global_status=use_global_status)
        x = tf.layers.conv2d_transpose(x, 128, kernel_size=3, strides=(2, 2), padding="same", kernel_initializer=initializer,name='conv8')
        x = nn.conv(x,name='conv9',filters=64,kernel_size=3, strides=1, padding='SAME',biased=False,bn=True,relu=True,
                    is_training=is_training,use_global_status=use_global_status)
        x = nn.conv(x,name='conv10',filters=num_classes,kernel_size=3, strides=1, padding='SAME',biased=False,bn=True,relu=True,
                    is_training=is_training,use_global_status=use_global_status)
        x = nn.conv(x,name='conv11',filters=1,kernel_size=3, strides=1, padding='SAME',biased=False,bn=False,relu=False,
                    is_training=is_training,use_global_status=use_global_status)
        return x
