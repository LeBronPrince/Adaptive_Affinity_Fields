import numpy as np
import os
import cv2
datapath = ''
data = []
list1 =
num =
shape =
output = np.zeros(shape)
for root,dirs,files in os.walk(datapath):
    path = os.join(datapath,files)
    image = cv2.imread(path)
    image = np.array(image)
    data.append(image)

w = h =0
for i in range(num):
    if i >= width:
        w = 0
        h += 1
    output[patch_size*h: patch_size*(h+1),patch_size*w: patch_size*(w+1),:] = data[i]

print("process splice image sucessfully")
save_path = ""
cv2.imsave(save_path,output)
