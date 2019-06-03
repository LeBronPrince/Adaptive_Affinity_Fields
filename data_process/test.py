import cv2
import numpy as np
import scipy.misc
from PIL import Image
folder_root = "/home/wangyang/Desktop/dataset/Vaihingen/ISPRS_semantic_labeling_Vaihingen/dsm/"
t_tag_folder = "/home/wangyang/Desktop/dataset/Vaihingen/Split/test/11/dsm/"
fname = folder_root + 'dsm_09cm_matching_area' + str(11) + '.tif'
im = cv2.imread(fname,3)
gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
#cv2.imwrite("/home/wangyang/Desktop/dataset/Vaihingen/ISPRS_semantic_labeling_Vaihingen/dsm/11.jpg",gray)
print(np.array(im).shape)
print(np.array(gray).shape)
