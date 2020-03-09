import scipy.misc
import os
import os.path as osp
import numpy as np
path = "/home/f523/wangyang/segmentation/Vaihingen/ISPRS_semantic_labeling_Vaihingen/ndsm"
num = 0
mean = 0
for root, path, files in os.walk(path):
    for file in files:
        num += 1
        name = osp.join(root, file)
        im = np.array(scipy.misc.imread(name))
        mean += np.mean(im)
mean /= num
print("there are %d images"%num)
print("The mean is %f"%mean)
