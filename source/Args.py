from torchvision import transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import BCEWithLogitsLoss, MSELoss
import albumentations as A
from albumentations.pytorch import ToTensorV2
from monai.losses.dice import DiceCELoss
from utilities.losses.boundary_loss import BoundaryLoss
from utilities.losses.active_boundary_loss import Active_BoundaryLoss
import torch
import os

class Args_Train_Loader:
    train_bs = 4
    num_workers = 20
    pin_memory = True
    shuffle = True
    drop_last = True

    @staticmethod
    def _get_default_transform():
        return A.Compose([
            A.Resize(224, 224),
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ])
    @staticmethod
    def _get_aug_transform():
        return A.Compose([
            A.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            # A.OneOf([
            #     A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2, hue=0.1, p=0.5),
            #     A.NoOp()
            # ], p=0.5),
            A.OneOf([
                A.ColorJitter(brightness=0.7, contrast=0.8, saturation=0.3, hue=0.15, p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=0.7),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.5),
            ], p=0.8),
        ])

class Args_Valid_Loader:
    valid_bs = 4
    num_workers = 20
    pin_memory = True
    shuffle = False
    drop_last = True

    @staticmethod
    def _get_default_transform():
        return A.Compose([
            A.Resize(224, 224),
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ])

#%% [INFO] 실험용 
class Args_experiments:
    device = 'cuda'
    device_ids = [0, 1] if torch.cuda.device_count() > 1 else [0]
    
    lr = 42e-3
    weight_deacy = 1e-4
    epoch = 100
    patience = 20
    warmup_epoch = 10
    T_max = 20
    
    optimizer_fn = staticmethod(lambda params: AdamW(params, lr=Args_experiments.lr, weight_decay=1e-4))
    scheduler_fn = staticmethod(lambda optimizer: CosineAnnealingLR(optimizer, T_max=20))

    #% Loss Collection
    loss_fn = staticmethod(lambda: BCEWithLogitsLoss()) # DICE : 0.79
    # loss_fn = staticmethod(lambda: DiceCELoss(include_background=True)) # include_background=False 로 설정
    # loss_fn = staticmethod(lambda: BoundaryLoss()) # DICE : 0.7443
    # loss_fn = staticmethod(lambda: Active_BoundaryLoss()) # DICE : None 
    # loss_fn = staticmethod(lambda: MSELoss()) # DICE : 0.3058
