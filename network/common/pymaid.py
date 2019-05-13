import tensorflow as tf
import tensorflow.contrib.slim as slim


def nonlocal_dot(net, depth, embed=True, softmax=True, scope=None,scale=1):

    batch_size, h, w, c = net.get_shape().as_list()
    local_x = []
    local_y = []
    step_h, step_w = h//scale, w//scale
    for i in range(0, scale):
        for j in range(0, scale):
            start_x, start_y = i*step_h, j*step_w
            end_x, end_y = min(start_x+step_h, h), min(start_y+step_w, w)
            if i == (scale-1):
                end_x = h
            if j == (scale-1):
                end_y = w
            local_x += [start_x, end_x]
            local_y += [start_y, end_y]
    local_list = []
    local_block_cnt = 2*scale*scale
    with tf.variable_scope(scope, 'nonlocal', values=[net]) as sc:
        with slim.arg_scope([slim.conv2d]):
            if embed:#slim.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate,padding='SAME', scope=scope)
                a = slim.conv2d(net, depth, 1, stride=1, scope='embA')
                b = slim.conv2d(net, depth, 1, stride=1, scope='embB')
            else:
                a, b = net, net
            g_orig = g = slim.conv2d(net, depth, 1, stride=1, scope='g')
    for i in range(0,local_block_cnt, 2):
    # Flatten from (B,H,W,C) to (B,HW,C) or similar
        value_local = g[:,local_x[i]:local_x[i+1],local_y[i]:local_y[i+1],:]
        query_local = a[:,local_x[i]:local_x[i+1],local_y[i]:local_y[i+1],:]
        key_local = b[:,local_x[i]:local_x[i+1],local_y[i]:local_y[i+1],:]
        a_flat = tf.reshape(query_local, [tf.shape(query_local)[0], -1, tf.shape(query_local)[-1]])
        b_flat = tf.reshape(key_local, [tf.shape(key_local)[0], -1, tf.shape(key_local)[-1]])
        g_flat = tf.reshape(value_local, [tf.shape(value_local)[0], -1, tf.shape(value_local)[-1]])
        #a_flat.set_shape([query_local.shape[0], query_local.shape[1] * query_local.shape[2] if None not in a.shape[1:3] else None, a.shape[-1]])
        #b_flat.set_shape([b.shape[0], b.shape[1] * b.shape[2] if None not in b.shape[1:3] else None, b.shape[-1]])
        #g_flat.set_shape([g.shape[0], g.shape[1] * g.shape[2] if None not in g.shape[1:3] else None, g.shape[-1]])
        # Compute f(a, b) -> (B,HW,HW)
        f = tf.matmul(a_flat, tf.transpose(b_flat, [0, 2, 1]))
        if softmax:
            f = tf.nn.softmax(f)
        else:
            f = f / tf.cast(tf.shape(f)[-1], tf.float32)
    # Compute f * g ("self-attention") -> (B,HW,C)
        fg = tf.matmul(f, g_flat)
    # Expand and fix the static shapes TF lost track of.
        fg = tf.reshape(fg, tf.shape(value_local))
        local_list.append(fg)
    context_list = []
    for i in range(0, scale):
        row_tmp = []
        for j in range(0, scale):
            row_tmp.append(local_list[j+i*scale])
        context_list.append(tf.concat(row_tmp, 2))
    context = tf.concat(context_list, 1)
    context = slim.conv2d(context,c,1,1,scope=scope+'nonlocal_final')
    out = context + net
    return out
