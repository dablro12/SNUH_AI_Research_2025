from dotenv import load_dotenv
import os, sys
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from utils.seed import seed_everything
from utils.datasets import CAG_Dataset
from utils.metrics import SegmentationMetrics
from Args import Args_Train_Loader, Args_Valid_Loader, Args_experiments
import torch
from tqdm import tqdm

seed_everything()
load_dotenv('.env')
metrics = SegmentationMetrics()

class Trainer:
    def __init__(self):
        self.train_loader, self.valid_loader = self.data_load()
        self.model, self.optimizer, self.scheduler, self.loss_fn = self.model_load()

    def data_load(self):
        tuning_df = pd.read_csv(os.getenv('TUNING_CSV'))
        kf = KFold(n_splits=5, shuffle=True, random_state=os.getenv('SEED'))
        train_idx, val_idx = next(kf.split(tuning_df))
        train_df = tuning_df.iloc[train_idx].reset_index(drop=True)
        valid_df = tuning_df.iloc[val_idx].reset_index(drop=True)

        train_dataset = CAG_Dataset(
            df=train_df,
            image_dir=os.getenv('IMAGE_DIR'),
            mask_dir=os.getenv('MASK_DIR'),
            default_transform=Args_Train_Loader._get_default_transform(),
            aug_transform=Args_Train_Loader._get_aug_transform()
        )
        valid_dataset = CAG_Dataset(
            df=valid_df,
            image_dir=os.getenv('IMAGE_DIR'),
            mask_dir=os.getenv('MASK_DIR'),
            default_transform=Args_Valid_Loader._get_default_transform()
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=Args_Train_Loader.train_bs,
            shuffle=Args_Train_Loader.shuffle,
            num_workers=Args_Train_Loader.num_workers,
            pin_memory=Args_Train_Loader.pin_memory,
            drop_last=Args_Train_Loader.drop_last
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=Args_Valid_Loader.valid_bs,
            shuffle=Args_Valid_Loader.shuffle,
            num_workers=Args_Valid_Loader.num_workers,
            pin_memory=Args_Valid_Loader.pin_memory,
            drop_last=Args_Valid_Loader.drop_last
        )

        sample_imgs, sample_masks = next(iter(train_loader))
        print(f"[INFO] sample_imgs.shape: {sample_imgs.shape}, sample_masks.shape: {sample_masks.shape}")
        return train_loader, valid_loader

    def model_load(self):
        from models.DeepSA.model import build_model
        model = build_model(
            ckpt_path=os.getenv('deepsa_ckpt_path'),
            device=Args_experiments.device
        ).to(Args_experiments.device)
        optimizer = Args_experiments.optimizer_fn(model.parameters())
        scheduler = Args_experiments.scheduler_fn(optimizer)
        loss_fn = Args_experiments.loss_fn()
        return model, optimizer, scheduler, loss_fn

    @staticmethod
    def set_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    @staticmethod
    def get_lr(optimizer):
        return optimizer.param_groups[0]['lr']

    def train_one_epoch(self, model, loader, optimizer, loss_fn, device):
        model.train()
        epoch_loss, metric_sum = 0, None
        for imgs, masks in tqdm(loader, desc="Train", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)
            batch_metrics = metrics.evaluate(torch.sigmoid(outputs), masks)
            if metric_sum is None:
                metric_sum = {k: v * imgs.size(0) for k, v in batch_metrics.items()}
            else:
                for k in metric_sum:
                    metric_sum[k] += batch_metrics[k] * imgs.size(0)
        n = len(loader.dataset)
        avg_metrics = {k: v / n for k, v in metric_sum.items()}
        return epoch_loss / n, avg_metrics

    @torch.no_grad()
    def valid_one_epoch(self, model, loader, loss_fn, device):
        model.eval()
        epoch_loss, metric_sum = 0, None
        for imgs, masks in tqdm(loader, desc="Valid", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, masks)
            epoch_loss += loss.item() * imgs.size(0)
            batch_metrics = metrics.evaluate(torch.sigmoid(outputs), masks)
            if metric_sum is None:
                metric_sum = {k: v * imgs.size(0) for k, v in batch_metrics.items()}
            else:
                for k in metric_sum:
                    metric_sum[k] += batch_metrics[k] * imgs.size(0)
        n = len(loader.dataset)
        avg_metrics = {k: v / n for k, v in metric_sum.items()}
        return epoch_loss / n, avg_metrics

    def run_training(self, num_epochs, patience, exp_name):
        best_dice = 0
        best_valid_loss = float('inf')
        patience_counter = 0
        save_dir = os.path.join(os.getenv("EXPERIMENT_DIR", "./EXPERIMENT_DIR"), exp_name)
        os.makedirs(save_dir, exist_ok=True)
        
        best_weight_path = os.path.join(save_dir, "best_weight_dice.pth")
        best_weight_loss_path = os.path.join(save_dir, "best_weight_loss.pth")
        warmup_epoch = getattr(Args_experiments, "warmup_epoch", 0)
        base_lr = Args_experiments.lr

        for epoch in range(1, num_epochs + 1):
            print(f"Epoch {epoch}/{num_epochs}")
            if warmup_epoch > 0 and epoch <= warmup_epoch:
                warmup_lr = base_lr * epoch / warmup_epoch
                self.set_lr(self.optimizer, warmup_lr)
                print(f"Warmup lr: {self.get_lr(self.optimizer):.6f}")
            elif warmup_epoch > 0 and epoch == warmup_epoch + 1:
                self.set_lr(self.optimizer, base_lr)
                print(f"Set lr to base: {self.get_lr(self.optimizer):.6f}")

            train_loss, train_metrics = self.train_one_epoch(
                self.model, self.train_loader, self.optimizer, self.loss_fn, Args_experiments.device
            )
            valid_loss, valid_metrics = self.valid_one_epoch(
                self.model, self.valid_loader, self.loss_fn, Args_experiments.device
            )
            self.scheduler.step()

            # dice 기준 best
            if valid_metrics["dice_coef"] > best_dice:
                best_dice = valid_metrics["dice_coef"]
                patience_counter = 0
                torch.save(self.model.state_dict(), best_weight_path)
                print(f"[Best Dice] Train Loss: {train_loss:.4f} | " + " | ".join([f'{k}: {v:.4f}' for k, v in train_metrics.items()]))
                print(f"[Best Dice] Valid Loss: {valid_loss:.4f} | " + " | ".join([f'{k}: {v:.4f}' for k, v in valid_metrics.items()]))
                print(f"Best Dice model saved at {best_weight_path} (Dice: {best_dice:.4f})")
            else:
                patience_counter += 1
                print(f"Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

            # loss 기준 best
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), best_weight_loss_path)
                print(f"Best Loss model saved at {best_weight_loss_path} (Valid Loss: {best_valid_loss:.4f})")

if __name__ == "__main__":
    trainer = Trainer()
    num_epochs = Args_experiments.epoch
    patience = Args_experiments.patience
    exp_name = "seg_experiment"
    trainer.run_training(num_epochs, patience, exp_name)