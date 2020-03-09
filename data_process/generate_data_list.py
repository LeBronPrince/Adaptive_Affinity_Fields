import os
read_file = "/home/f523/wangyang/segmentation/Adaptive_Affinity_Fields/dataset/Vaihingen/Vaihingen_test.txt"
write_file = "/home/f523/wangyang/segmentation/Adaptive_Affinity_Fields/dataset/Vaihingen/Vaihingen_test_ndsm_hist.txt"

fr = open(read_file, 'r')
fw = open(write_file, 'w')

lines = fr.readlines()

for line in lines:
    image, mask = line.strip("\n").split(' ')
    name = image.strip().split('/')[1]
    w = line.strip() + ' ' +'ndsm_hist/' + name
    fw.write(w+"\n")
fr.close()
fw.close()
