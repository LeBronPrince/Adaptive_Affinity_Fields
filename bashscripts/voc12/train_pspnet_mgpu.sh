#!/bin/bash
# This script is used for training, inference and benchmarking
# the baseline method with PSPNet on PASCAL VOC 2012 with
# multi-gpus. Users could also modify from this script for
# their use case.
#
# Usage:
#   # From Adaptive_Affinity_Fields/ directory.
#   bash bashscripts/voc12/train_pspnet_mgpu.sh
#
#

# Set up parameters for training.
BATCH_SIZE=16
TRAIN_INPUT_SIZE=480,480
WEIGHT_DECAY=5e-4
ITER_SIZE=1
NUM_STEPS=30000
NUM_CLASSES=21
NUM_GPU=4

# Set up parameters for inference.
INFERENCE_INPUT_SIZE=480,480
INFERENCE_STRIDES=320,320
INFERENCE_SPLIT=val

# Set up path for saving models.
SNAPSHOT_DIR=snapshots/voc12/pspnet/p480_bs16_lr1e-3_it30k

# Set up the procedure pipeline.
IS_TRAIN_1=1
IS_INFERENCE_1=1
IS_BENCHMARK_1=1
IS_TRAIN_2=1
IS_INFERENCE_2=1
IS_BENCHMARK_2=1

# Update PYTHONPATH.
export PYTHONPATH=`pwd`:$PYTHONPATH

# Set up the data directory.
DATAROOT=/path/to/data

# Train for the 1st stage.
if [ ${IS_TRAIN_1} -eq 1 ]; then
  python3 pyscripts/train/train_mgpu.py\
    --snapshot-dir ${SNAPSHOT_DIR}/stage1\
    --restore-from snapshots/imagenet/trained/resnet_v1_101.ckpt\
    --data-list dataset/voc12/train+.txt\
    --data-dir ${DATAROOT}/VOCdevkit/\
    --batch-size ${BATCH_SIZE}\
    --save-pred-every ${NUM_STEPS}\
    --update-tb-every 50\
    --input-size ${TRAIN_INPUT_SIZE}\
    --learning-rate 1e-3\
    --weight-decay ${WEIGHT_DECAY}\
    --iter-size ${ITER_SIZE}\
    --num-classes ${NUM_CLASSES}\
    --num-steps $(($NUM_STEPS+1))\
    --num-gpu ${NUM_GPU}\
    --random-mirror\
    --random-scale\
    --random-crop\
    --not-restore-classifier\
    --is-training
fi

# Inference for the 1st stage.
if [ ${IS_INFERENCE_1} -eq 1 ]; then
  python3 pyscripts/inference/inference.py\
    --data-dir ${DATAROOT}/VOCdevkit/\
    --data-list dataset/voc12/${INFERENCE_SPLIT}.txt\
    --input-size ${INFERENCE_INPUT_SIZE}\
    --strides ${INFERENCE_STRIDES}\
    --restore-from ${SNAPSHOT_DIR}/stage1/model.ckpt-${NUM_STEPS}\
    --colormap misc/colormapvoc.mat\
    --num-classes ${NUM_CLASSES}\
    --ignore-label 255\
    --save-dir ${SNAPSHOT_DIR}/stage1/results/${INFERENCE_SPLIT}
fi

# Benchmark for the 1st stage.
if [ ${IS_BENCHMARK_1} -eq 1 ]; then
  python3 pyscripts/benchmark/benchmark_by_mIoU.py\
    --pred-dir ${SNAPSHOT_DIR}/stage1/results/${INFERENCE_SPLIT}/gray/\
    --gt-dir ${DATAROOT}/VOCdevkit/VOC2012/segcls/\
    --num-classes ${NUM_CLASSES}
fi

# Train for the 2nd stage.
if [ ${IS_TRAIN_2} -eq 1 ]; then
  python3 pyscripts/train/train_mgpu.py\
    --snapshot-dir ${SNAPSHOT_DIR}/stage2\
    --restore-from ${SNAPSHOT_DIR}/stage1/model.ckpt-30000\
    --data-list dataset/voc12/train.txt\
    --data-dir ${DATAROOT}/VOCdevkit/\
    --batch-size ${BATCH_SIZE}\
    --save-pred-every ${NUM_STEPS}\
    --update-tb-every 50\
    --input-size ${TRAIN_INPUT_SIZE}\
    --learning-rate 1e-4\
    --weight-decay ${WEIGHT_DECAY}\
    --iter-size ${ITER_SIZE}\
    --num-classes ${NUM_CLASSES}\
    --num-steps $(($NUM_STEPS+1))\
    --num-gpu ${NUM_GPU}\
    --random-mirror\
    --random-scale\
    --random-crop\
    --is-training
fi

# Inference for the 2nd stage.
if [ ${IS_INFERENCE_2} -eq 1 ]; then
  python3 pyscripts/inference/inference_msc.py\
    --data-dir ${DATAROOT}/VOCdevkit/\
    --data-list dataset/voc12/${INFERENCE_SPLIT}.txt\
    --input-size ${INFERENCE_INPUT_SIZE}\
    --strides ${INFERENCE_STRIDES}\
    --restore-from ${SNAPSHOT_DIR}/stage2/model.ckpt-${NUM_STEPS}\
    --colormap misc/colormapvoc.mat\
    --num-classes ${NUM_CLASSES}\
    --ignore-label 255\
    --flip-aug\
    --scale-aug\
    --save-dir ${SNAPSHOT_DIR}/stage2/results/${INFERENCE_SPLIT}
fi

# Benchmark for the 2nd stage.
if [ ${IS_BENCHMARK_2} -eq 1 ]; then
  python3 pyscripts/benchmark/benchmark_by_mIoU.py\
    --pred-dir ${SNAPSHOT_DIR}/stage2/results/${INFERENCE_SPLIT}/gray/\
    --gt-dir ${DATAROOT}/VOCdevkit/VOC2012/segcls/\
    --num-classes ${NUM_CLASSES}
fi
