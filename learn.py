# -- coding: utf-8 --
import tensorflow as tf
import numpy as np
y_true = [1,2,2,1]
y_pred = [2,2,2,2]
#y_true = tf.one_hot(y_true,3,dtype=tf.int32)
pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
sess = tf.Session()
e = sess.run(pt_1)
print(e)





"""
a = np.array([[1,2,3],[4,5,6]])
b = np.array([[2],[3]])
c = a*b
print(c)

from utils.general import snapshot_arg
a = [1,2,3,4]
b = [1,3,2,4]
c = tf.not_equal(a,b)
with tf.Session() as sess:
    print(c.eval())


text = ""
pbar = tqdm(["a", "b", "c", "d"])
for char in pbar:
    pbar.set_description("Processing %s" % char)
    sleep(1)

a = [1,2,3,4,5,6,7]
b = tf.less_equal(a,4)
c = tf.where(b)
d = tf.squeeze(c,1)
e = tf.gather(a,d)
with tf.Session() as sess:
    print(b.eval())
    print(c.eval())
    print(d.eval())
    print(e.eval())


class R:
    a = 1
    b = 2
    c = 3

snap_dir ='/home/wangyang/Desktop/'
dictargs = vars(R)

print('-----------------------------------------------')
print('-----------------------------------------------')
with open(os.path.join(snap_dir, 'config'), 'w') as argsfile:
    for key, val in dictargs.items():
        line = '| {0} = {1}'.format(key, val)
        print(line)
        argsfile.write(line+'\n')
print('-----------------------------------------------')
print('-----------------------------------------------')



a = tf.constant([0])
b = tf.constant([1])
c = tf.stack([a,b])
d = tf.squeeze(c,1)
with tf.Session() as sess:
    print(c.eval())
    print(d.eval())
"""
