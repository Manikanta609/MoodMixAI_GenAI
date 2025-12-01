import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class FER2013Dataset(Dataset):
    """
    Custom Dataset for FER2013.
    Expected structure:
    data/fer2013/
        train/
            angry/
            disgust/
            ...
        test/
            angry/
            disgust/
            ...
    """
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.classes = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.images = []
        self.labels = []

        self._load_dataset()

    def _load_dataset(self):
        split_dir = os.path.join(self.root_dir, self.split)
        if not os.path.exists(split_dir):
            print(f"Warning: Directory {split_dir} not found. Dataset might be empty.")
            return

        for cls_name in self.classes:
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue
            
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                self.images.append(img_path)
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load image in grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
             # Handle missing/corrupt images gracefully-ish
             image = np.zeros((48, 48), dtype=np.uint8)
        
        # Convert to PIL Image for transforms
        image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)

        return image, label
