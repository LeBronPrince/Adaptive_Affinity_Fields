# -- coding: utf-8 --

import cv2
import numpy as np
import os
path = "/home/f523/wangyang/segmentation/Vaihingen/ISPRS_semantic_labeling_Vaihingen/ndsm/"#/dsm_09cm_matching_area11_normalized.jpg
path1 = "/home/f523/wangyang/segmentation/Vaihingen/ISPRS_semantic_labeling_Vaihingen/ndsm_hist/"
mean = 0
mean1 = 0
num = 0
for p in os.listdir(path):
    im = cv2.imread(path+p)
    mean += np.mean(im[:,:,0])
    im1 = cv2.equalizeHist(im[:,:,0])
    mean1 += np.mean(im1)
    cv2.imwrite(path1+p, im1)
    num += 1
print("the original mean is {}".format(mean/num))
print("the new mean is {}".format(mean1/num))


"""
import tensorflow as tf
import numpy as np

import os
from tensorflow.python import pywrap_tensorflow
checkpoint_path = "/home/f523/wangyang/segmentation/Adaptive_Affinity_Fields/snapshot/snapshot_v9/model.ckpt-6500"
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
weight_new = np.zeros((7,7,4,64))
Momentum_new = np.zeros((7,7,4,64))
for key in var_to_shape_map:
    if key == "resnet_v1_101/conv1/weights":
        print("tensor_name: ", key, end=' ')
        weight = reader.get_tensor(key)#7*7*3*64
    if key == "resnet_v1_101/conv1/weights/Momentum":
        print("tensor_name: ", key, end=' ')
        Momentum = reader.get_tensor(key)#7*7*3*64

weight_new[:,:,:3,:] = weight
mean = np.mean(weight, axis = 2)
weight_new[:,:,3,:] = mean
np.save("/home/f523/wangyang/segmentation/Adaptive_Affinity_Fields/snapshot/snapshot_v9/weight.npy", weight_new)

Momentum_new[:,:,:3,:] = Momentum
mean = np.mean(Momentum, axis = 2)
Momentum_new[:,:,3,:] = mean
np.save("/home/f523/wangyang/segmentation/Adaptive_Affinity_Fields/snapshot/snapshot_v9/Momentum.npy", Momentum_new)
print(weight_new)



import scipy.misc
import numpy as np
path = "/home/f523/wangyang/segmentation/Vaihingen/ISPRS_semantic_labeling_Vaihingen/ndsm/dsm_09cm_matching_area11_normalized.jpg"
im = scipy.misc.imread(path)
im = np.array(im)
print(np.max(im))
print(np.min(im))
print(np.mean(im))


path = "/home/f523/wangyang/segmentation/Vaihingen/ISPRS_semantic_labeling_Vaihingen/top/top_mosaic_09cm_area11.tif"
im = scipy.misc.imread(path)
im = np.array(im)
print("red channel mean")
print(np.max(im[:,:,0]))
print(np.min(im[:,:,0]))
print(np.mean(im[:,:,0]))

print("green channel mean")
print(np.max(im[:,:,1]))
print(np.min(im[:,:,1]))
print(np.mean(im[:,:,1]))

print("bule channel mean")
print(np.max(im[:,:,2]))
print(np.min(im[:,:,2]))
print(np.mean(im[:,:,2]))




y_true = [1,2,2,1]
y_pred = [2,2,2,2]
#y_true = tf.one_hot(y_true,3,dtype=tf.int32)
pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
sess = tf.Session()
e = sess.run(pt_1)
print(e)






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
