import os
def ListFilesToTxt(dir,file1):
  files = os.listdir(dir)
  print(len(files))
  for name in files:
    file1.write("original/" + name +" "+"label_train/"+name+"\n")
def Test():
  dir="/home/wangyang/Desktop/dataset/Vaihingen/Split/original"
  outfile="/home/wangyang/Desktop/Semantic Segmentation Code/Adaptive_Affinity_Fields/dataset/Vaihingen/Vaihingen_train.txt"

  file1 = open(outfile,"w")

  ListFilesToTxt(dir,file1)

  file1.close()
Test()
