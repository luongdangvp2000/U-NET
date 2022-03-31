import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes
import numpy as np

import json
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from torchvision.datasets.utils import extract_archive, verify_str_arg, iterable_to_str
from torchvision.datasets import VisionDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from typing import Any, Callable, Dict, List, Optional, Union, Tuple

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace('.jpg', '_mask.gif'))
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image= image, mask= mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

class CityscapesDataset(Cityscapes):
    def __init__(self):
        super(CityscapesDataset, self).__init__()
        self.transform = A.Compose([
            A.Resize(256, 512),
            A.HorizontalFlip(),
            A.Normalize(mean=(0.485, 0.456, 0.486), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert('RGB')

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])
            targets.append(target)
        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            transformed=transform(image=np.array(image), mask=np.array(target))            
        return transformed['image'],transformed['mask']
