import numpy as np
import os
import cv2
datapath = '/home/f523/wangyang/segmentation/Adaptive_Affinity_Fields/inference/inference_final/color'
im_path = '/home/f523/wangyang/segmentation/Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE/top_mosaic_09cm_area11.tif'
patch_size = 336
im = cv2.imread(im_path)
im = np.array(im)
w = im.shape[0]
h = im.shape[1]
w_num = int(w/patch_size)
h_num = int(h/patch_size)
num = w_num*h_num
shape = [w_num*patch_size,h_num*patch_size,3]
shape1 = [w,h,3]
output = np.zeros(shape)
output1 = np.zeros(shape1)
data = []
for root,dirs,files in os.walk(datapath):
    for file in files:
        path = os.path.join(root,file)
        print(path)
        image = cv2.imread(path)
        image = np.array(image)
        data.append(image)
print(len(data))
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
