import os
def ListFilesToTxt(dir,file1):
  files = os.listdir(dir)
  print(len(files))
  for name in files:
    #file1.write("original_15/" + name +" "+"label_train/"+name+"\n")
    file1.write(name[0:-4]+"\n")
def Test():
  dir="/home/wangyang/Desktop/DOTA_devkit/examplesplit1/images"
  outfile="/home/wangyang/Desktop/DOTA_devkit/examplesplit1/train.txt"

  file1 = open(outfile,"w")

  ListFilesToTxt(dir,file1)

  file1.close()
Test()
