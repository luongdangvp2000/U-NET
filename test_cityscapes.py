import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.datasets import Cityscapes

from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import numpy as np
#import cityscapesScripts

from dataset import (
    CityscapesDataset,
)

    
if __name__ == "__main__":
    ignore_index=255
    void_classes = [0,12,3,4,5,6,9,10,14,15,16,18,29,30,-1]
    valid_classes = [ignore_index,7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]

    class_map = dict(zip(valid_classes, range(len(valid_classes))))
    n_classes=len(valid_classes)

    colors = [   [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(n_classes), colors))
    
    transform = A.Compose([
        A.Resize(256, 512),
        A.HorizontalFlip(),
        A.Normalize(mean=(0.485, 0.456, 0.486), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    dataset = CityscapesDataset(
        './data/cityscapes', 
        split='val', 
        mode='fine',
        target_type='semantic',
        transforms= transform
    )

    #print(dataset[0][0].size)

    img, seg = dataset[20]
    print(img.shape, seg.shape)
    