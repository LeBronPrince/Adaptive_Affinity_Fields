import os
file_path = '/home/wangyang/Desktop/dataset/VOCdevkit/VOC2012/ImageSets/Main/aeroplane_train.txt'
f = open(file_path,'r')
for line in f:
    name,bool1 = line.strip("\n").split(' ')
    if bool1 == '-1':
        line.delete
