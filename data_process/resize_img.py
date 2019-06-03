import os
import cv2
data_path = '/media/wangyang/F523/wangyang/segmentation/inference_v9/11'

for root,dirs,files in os.walk(data_path):
    for file1 in files:

        im_path = os.path.join(root,file1)
        im = cv2.imread(im_path)
        im = cv2.resize(im,(256,256))
        save_path = root+'/256'
        save_path = os.path.join(save_path,file1)
        cv2.imwrite(save_path,im)
