import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
#import cityscapesScripts

from dataset import (
    Cityscapes,
)

if __name__ == "__main__":
    dataset = Cityscapes(
        './data/cityscapes', 
        split='train', 
        mode='fine',
        target_type='semantic'
    )

    img, smnt = dataset[0]