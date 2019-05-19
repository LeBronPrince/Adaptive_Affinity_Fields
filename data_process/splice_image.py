import numpy as np
import os
import cv2
datapath = '/home/f523/wangyang/segmentation/Vaihingen/Split/test/original_11'
im_path = '/home/f523/wangyang/segmentation/Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE/top_mosaic_09cm_area11.tif'
im = cv2.imread(im_path)
im = np.array(im)
w = im.shape(0)
h = im.shape(1)
w_num = int(w/336)
h_num = int(h/336)
num = w_num*h_num
shape = [w_num*336,h_num*336,3]
shape1 = [w,h,3]
output = np.zeros(shape)
output1 = np.zeros(shape1)
data = []
for root,dirs,files in os.walk(datapath):
    path = os.join(datapath,files)
    image = cv2.imread(path)
    image = np.array(image)
    data.append(image)

w = h =0
for i in range(num):
    if i >= w_num:
        w = 0
        h += 1
    output[patch_size*h: patch_size*(h+1),patch_size*w: patch_size*(w+1),:] = data[i]
    i += 1

print("process splice image sucessfully")
save_path = "/home/f523/wangyang/segmentation/Vaihingen/Split/test/original_11/VH_11.jpg"
output1 = output[:w,:h,:]
cv2.imsave(save_path,output1)
