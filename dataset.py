from torch.utils.data import Dataset
import random
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch

class FacesDataset(Dataset):
    def __init__(self, dir, split="train", per_subject_samples = 4, transforms = None):
        super().__init__()
        self.base = os.path.join(dir, split)
        self.subjects = os.listdir(self.base)
        self.per_subject_samples = per_subject_samples
        self.transforms = transforms

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, index): # index -> person, from person, we sample 4 images
        subject = self.subjects[index]
        subject_path = os.path.join(self.base, subject)
        images = os.listdir(subject_path)
        sample = [random.randint(0, len(images)-1) for _ in range(self.per_subject_samples)]
        ret_images = []
        for idx in sample:
            path = os.path.join(subject_path, images[idx])
            image = Image.open(path)
            if self.transforms:
                image = self.transforms(image)
            ret_images.append(image)
            # print(image.shape)
        
        ret_tensor = torch.stack(ret_images)

        return ret_tensor