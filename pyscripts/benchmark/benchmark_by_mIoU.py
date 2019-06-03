import os
import argparse

from PIL import Image
import numpy as np
import sys
sys.path.append("/home/f523/wangyang/segmentation/Adaptive_Affinity_Fields/")
from utils.metrics import iou_stats
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(
  description='Benchmark segmentation predictions'
)
parser.add_argument('--pred-dir', type=str, default='/home/f523/wangyang/segmentation/Adaptive_Affinity_Fields/inference/inference_fcn/gray',
                    help='/path/to/prediction.')
parser.add_argument('--gt-dir', type=str, default='/home/f523/wangyang/segmentation/Vaihingen/Split/test/label_test',
                    help='/path/to/ground-truths')
parser.add_argument('--num-classes', type=int, default=6,
                    help='number of segmentation classes')
parser.add_argument('--string-replace', type=str, default=',',
                    help='replace the first string with the second one')
args = parser.parse_args()

assert(os.path.isdir(args.pred_dir))
assert(os.path.isdir(args.gt_dir))
tp_fn = np.zeros(args.num_classes, dtype=np.float64)
tp_fp = np.zeros(args.num_classes, dtype=np.float64)
tp = np.zeros(args.num_classes, dtype=np.float64)
for dirpath, dirnames, filenames in os.walk(args.pred_dir):
  for filename in filenames:
    predname = os.path.join(dirpath, filename)
    gtname = predname.replace(args.pred_dir, args.gt_dir)
    if args.string_replace != '':
      stra, strb = args.string_replace.split(',')
      gtname = gtname.replace(stra, strb)

    pred = np.array(
        Image.open(predname),
        dtype=np.uint8)
    gt = np.array(
        Image.open(gtname),
        dtype=np.uint8)
    #gt = gt[:,:,0]
    _tp_fn, _tp_fp, _tp = iou_stats(
        pred,
        gt,
        num_classes=args.num_classes,
        background=0)

    tp_fn += _tp_fn
    tp_fp += _tp_fp
    tp += _tp

print('tp {}'.format(tp))
print('tp_fn {}'.format(tp_fn))
print('tp_fp {}'.format(tp_fp))
iou = tp / (tp_fn + tp_fp - tp + 1e-12) * 100.0
precision = tp / (tp_fp+ 1e-12)
recall = tp / (tp_fn+ 1e-12)
F1 = 2*(precision*recall)/(precision + recall) * 100.0
class_names = ['surfaces', 'background', 'Car', 'Tree', 'vegetation', 'Building']
#print(tp_fn.sum())
#print(tp_fp.sum())
oa = tp.sum()/(tp_fn.sum())* 100.0#+tp_fp.sum()-tp_fp[1]-tp.sum()
print("overall accuracy is {:4.4f}%".format(oa))
for i in range(args.num_classes):
  print('class {:10s}: {:02d}, iou:{:4.4f} ,recall: {:4.4f} ,precision: {:4.4f} ,F1: {:4.4f}%'.format(
      class_names[i], i, iou[i], recall[i] ,precision[i], F1[i]))
mean_recall = (recall.sum()-recall[1]) / (args.num_classes-1)* 100.0
mean_precision = (precision.sum()-precision[1]) / (args.num_classes-1)* 100.0
mean_F1 = (F1[0]+F1[2]+F1[3]+F1[4]+F1[5]) / (args.num_classes-1)
mean_iou = (iou.sum()-iou[1]) / (args.num_classes-1)

print('mean recall: {:4.4f}%'.format(mean_recall))
print('mean precision: {:4.4f}%'.format(mean_precision))
print('mean F1: {:4.4f}%'.format(mean_F1))
print('mean iou: {:4.4f}%'.format(mean_iou))

#mean_pixel_acc = tp.sum() / (tp_fp.sum() + 1e-12)
#print('mean Pixel Acc: {:4.4f}%'.format(mean_pixel_acc))
