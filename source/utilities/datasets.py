from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd
import random
import numpy as np
import torch

class CAG_Dataset(Dataset):
    def __init__(self, df, image_dir, mask_dir, default_transform, aug_transform=None, Prompt_Args:dict=None):
        self.df = df
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.default_transform = default_transform
        self.aug_transform = aug_transform
        self.Prompt_Args = Prompt_Args

        self.image_paths = [os.path.join(image_dir, path) for path in self.df["image_filename"]]
        self.mask_paths = [os.path.join(mask_dir, path) for path in self.df["mask_save_path"]]

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        if self.aug_transform:
            augmented = self.aug_transform(image=np.array(image), mask=np.array(mask))
            image = augmented['image']
            mask = augmented['mask']

        transformed = self.default_transform(image=np.array(image), mask=np.array(mask))
        image = transformed['image']
        mask = transformed['mask']
        mask = mask.unsqueeze(0)

        # numpy -> torch 변환 (ToTensorV2가 있으면 이미 tensor임)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        mask = (mask > 0).float()

        if self.Prompt_Args is not None:
            prompt = self.Few_Shot_Prompt(mask)
            if prompt is None:
                return image, mask
            return image, mask, prompt

        return image, mask
    
    def Few_Shot_Prompt(self, mask):
        """
        전처리된 마스크(Binary Mask)로 Few-shot prompt를 생성합니다.
        Prompt_Args에서 n_shot만큼 현재 idx를 제외한 샘플을 무작위로 선택하여,
        각 샘플의 이미지와 전처리된 binary mask를 (img, msk) 튜플로 리스트에 담아 반환합니다.
        """
        if self.Prompt_Args is None:
            return None
        n_shot = self.Prompt_Args.get("n_shot", 0)
        # mask는 이미 전처리된 binary mask임
        candidates = [i for i in range(len(self.df))]
        shot_indices = random.sample(candidates, min(n_shot, len(candidates)))
        def make_prompt(i):
            img = self.default_transform(Image.open(self.image_paths[i]))
            msk = (self.default_transform(Image.open(self.mask_paths[i])) != 0).float()
            return (img, msk)
        return list(map(make_prompt, shot_indices))

    
    def __get_image_path(self, idx):
        return self.image_paths[idx]
    
    def __get_mask_path(self, idx):
        return self.mask_paths[idx]
    
    def __get_image_filename(self, idx):
        return os.path.basename(self.image_paths[idx])
    
    def __get_mask_filename(self, idx):
        return os.path.basename(self.mask_paths[idx])
    
    def __get_image_save_filename(self, idx):
        return os.path.basename(self.image_paths[idx])
    
    def __get_mask_save_filename(self, idx):
        return os.path.basename(self.mask_paths[idx])
    
    def __get_image_save_path(self, idx):
        return os.path.join(self.image_dir, self.__get_image_save_filename(idx))    
            
