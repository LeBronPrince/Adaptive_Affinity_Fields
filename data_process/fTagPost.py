"""
This script is used to adapt the tiff tag image to single channel .png tag image
name rule:
    VH_XX_AAAAB: XX from top split at AAAA,
     B: 1 - stands for origin split, 2 - flip, 3 - rotation,
usage:
    1) run the x_adapt_tags(), setting the data_root and all the path, to adapt the
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import scipy.misc
import os
import cv2

# ------------------------------------------------------------- #
# BASIC original image folder settings
#
# ! set the data root first !


training_set = [1, 3, 5, 7, 13, 17, 21, 23, 26, 32, 37, 11, 15, 28, 30, 34]

# hard written index: it is only 6 class...
# reference file VH_class.txt
# [S]elect the corresponding index of colors

# the following is for VH
d_index = [(255, 255, 255), (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)]

# the following option is for MA
#d_index = [(0, 0, 0), (255, 0, 0)]

# index fo
# d_index = [(255, 255, 255),     # background -> label 0 don't count
#            (0, 0, 0),           # Roads
#            (100, 100, 100),     # Buildings
#            (0, 125, 0),         # Trees
#            (0, 255, 0),         # Grass
#            (150, 80, 0),        # Bare Soil
#            (0, 0, 150),         # Water
#            (255, 255, 0),       # Railways
#            (150, 150, 255)]     # Swimming Pools

# Initialisation of the set
# training_set = [1, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 17, 18, 20]
# test_set = [2, 6, 12, 15, 16, 19]


def x_adapt_one(im, tt_path):
    if not os.path.isfile(tt_path):
        print("in")
        im_arr = np.array(im)

        weight = im_arr.shape[1]
        print("weight is {}".format(weight))
        height = im_arr.shape[0]
        t_im_arr = np.zeros([height, weight], dtype=np.uint8)

        for ix in range(height):
            for iy in range(weight):
                #print(ix*6000+iy)
                p_cur = tuple(im_arr[ix][iy])
                if (p_cur in d_index):
                    t_im_arr[ix][iy] = d_index.index(p_cur)
                else:
                    t_im_arr[ix][iy] = 255  # 255 mark as ignored
        print("save img")
        cv2.imwrite(tt_path,t_im_arr)
        #Image.fromarray(t_im_arr).save(tt_path)
    else:
        print("fuck")




def xf_adapt_PD_tags():

    folder_root = '/home/wangyang/Desktop/dataset/Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE/'
    folder_root1 = '/home/wangyang/Desktop/dataset/Vaihingen/label_train/'
    for root, dirs, files in os.walk(folder_root):
        for onefile in files:
            fname = folder_root + onefile
            print('adapting tag file ' + onefile)
            t_fname = folder_root1 + onefile[0:-4] + '.png'
            im = cv2.imread(fname)
            x_adapt_one(im, t_fname)


if __name__ == '__main__':
    xf_adapt_PD_tags()
