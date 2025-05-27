from torchvision import transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import BCEWithLogitsLoss, MSELoss
import albumentations as A
from albumentations.pytorch import ToTensorV2
from monai.losses.dice import DiceCELoss
from utils.losses.boundary_loss import BoundaryLoss
from utils.losses.active_boundary_loss import Active_BoundaryLoss

class Args_Train_Loader:
    train_bs = 16
    num_workers = 20
    pin_memory = True
    shuffle = True
    drop_last = True

    @staticmethod
    def _get_default_transform():
        return A.Compose([
            A.Resize(512, 512),
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ])
    @staticmethod
    def _get_aug_transform():
        return A.Compose([
            A.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.OneOf([
                A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2, hue=0.1, p=0.5),
                A.NoOp()
            ], p=0.5),
        ])

class Args_Valid_Loader:
    valid_bs = 8
    num_workers = 20
    pin_memory = True
    shuffle = False
    drop_last = True

    @staticmethod
    def _get_default_transform():
        return A.Compose([
            A.Resize(512, 512),
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ])

#%% [INFO] 실험용 
class Args_experiments:
    device = 'cuda'
    
    lr = 42e-3
    weight_deacy = 1e-4
    epoch = 100
    patience = 20
    warmup_epoch = 10
    T_max = 20
    
    optimizer_fn = staticmethod(lambda params: AdamW(params, lr=Args_experiments.lr, weight_decay=1e-4))
    scheduler_fn = staticmethod(lambda optimizer: CosineAnnealingLR(optimizer, T_max=20))

    #% Loss Collection
    # loss_fn = staticmethod(lambda: BCEWithLogitsLoss()) # DICE : 0.79
    # loss_fn = staticmethod(lambda: DiceCELoss(include_background=True)) # include_background=False 로 설정
    # loss_fn = staticmethod(lambda: BoundaryLoss()) # DICE : 0.7443
    # loss_fn = staticmethod(lambda: Active_BoundaryLoss()) # DICE : None 
    loss_fn = staticmethod(lambda: MSELoss()) # DICE : 0.3058
