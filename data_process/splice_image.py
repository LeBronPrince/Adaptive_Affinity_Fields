import numpy as np
import os
import cv2
import math
datapath = '/home/f523/wangyang/segmentation/Adaptive_Affinity_Fields/inference/inference_final/color'
im_path = '/home/f523/wangyang/segmentation/Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE/top_mosaic_09cm_area11.tif'
patch_size = 336
im = cv2.imread(im_path)
im = np.array(im)
print(im.shape)
im_h = im.shape[0]
im_w = im.shape[1]
w_num = math.ceil(im_w/patch_size)
print(w_num)
h_num =  math.ceil(im_h/patch_size)
print(h_num)
num = w_num*h_num
shape = [h_num*patch_size,w_num*patch_size,3]
shape1 = [im_h,im_w,3]
output = np.zeros(shape)
output1 = np.zeros(shape1)
data = []
files = os.listdir(datapath)
files.sort(key=lambda x:int(x[-7:-4]))

for file in files:
    path = os.path.join(datapath,file)
    image = cv2.imread(path)
    image = np.array(image)
    data.append(image)
print(len(data))
w = h = 0
for i in range(num):
    if w >= w_num:
        w = 0
        h += 1
    output[patch_size*h: patch_size*(h+1),patch_size*w: patch_size*(w+1),:] = data[i]
    print("i is {}".format(i))
    print("w is {} h is {}".format(w,h))
    #print(np.array(data[i]).shape)
    i += 1
    w += 1

print("process splice image sucessfully")
save_path = "/home/f523/wangyang/segmentation/Vaihingen/Split/test/original_11/VH_11.jpg"
output1 = output[:im_h,:im_w,:]
print(output1.shape)
cv2.imwrite(save_path,output1)
