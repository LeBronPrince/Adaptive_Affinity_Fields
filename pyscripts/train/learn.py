a = "resnet_v1_101/block5/conv2/BatchNorm/beta"

if 'block5' in a:
    print('in')
else:
    print('out')
"""
import sys
sys.path.append("/home/wangyang/Desktop/Semantic Segmentation Code/Adaptive_Affinity_Fields")
sys.path.append("/home/wangyang/Desktop/Semantic Segmentation Code/Adaptive_Affinity_Fields/seg_models/models")
print(sys.path)
from seg_models.models.pspnet import pspnet_resnet101 as model
"""
