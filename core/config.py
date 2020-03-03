#! /usr/bin/env python
# coding=utf-8
import os
from easydict import EasyDict as edict


__C                           = edict()
cfg                           = __C

# YOLO options
__C.YOLO                      = edict()

__C.YOLO.CLASSES              = os.path.abspath('.') + "\data\classes\coco.names"
__C.YOLO.ANCHORS              = os.path.abspath('.') + "\data\\anchors\\basline_anchors.txt"
__C.YOLO.STRIDES              = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE     = 3
__C.YOLO.IOU_LOSS_THRESH      = 0.5

# Train options
__C.TRAIN                     = edict()

__C.TRAIN.ANNOT_PATH          = os.path.abspath('.')+"\data\dataset\\train.txt"
__C.TRAIN.BATCH_SIZE          = 4
__C.TRAIN.INPUT_SIZE          = [416]
__C.TRAIN.DATA_AUG            = True
__C.TRAIN.LR_INIT             = 1e-3
__C.TRAIN.LR_END              = 1e-6
__C.TRAIN.WARMUP_EPOCHS       = 2
__C.TRAIN.EPOCHS              = 30




