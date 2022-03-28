import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
#import cityscapesScripts

from dataset import (
    CityscapesDataset,
)

if __name__ == "__main__":
    dataset = Cityscapes(
        './data/cityscapes', 
        split='train', 
        mode='fine',
        target_type='semantic',
        transforms= transform
    )

    img, seg = dataset[20]
    print(img.shape, seg.shape)