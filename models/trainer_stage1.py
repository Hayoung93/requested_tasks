import os
import numpy as np
from torch.optim import lr_scheduler
import torch
from sklearn.metrics import accuracy_score, average_precision_score
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import time
from tqdm import tqdm
from torch.amp import autocast
from torch.cuda.amp import GradScaler

from models.network.net_stage1 import net_stage1


class Trainer_stage1:
    def __init__(self, opt):
        self.model = net_stage1()

        # Load pretrained weights if specified
        if opt.pretrained_stage1_path and os.path.exists(opt.pretrained_stage1_path):
            print(f"\n{'='*60}")
            print(f"Loading pretrained Stage 1 model from:")
            print(f"  {opt.pretrained_stage1_path}")
            print(f"{'='*60}")

            checkpoint = torch.load(opt.pretrained_stage1_path, map_location='cpu')

            # Check checkpoint structure
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"Checkpoint info:")
                if 'epoch' in checkpoint:
                    print(f"  - Trained epoch: {checkpoint['epoch']}")
                if 'loss' in checkpoint:
                    print(f"  - Validation loss: {checkpoint['loss']:.4f}")
            else:
                # Direct state_dict (fallback)
                state_dict = checkpoint
                print("Loading direct state_dict (no checkpoint wrapper)")

            # Load state_dict with strict=True to ensure key matching
            try:
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

                if len(missing_keys) == 0 and len(unexpected_keys) == 0:
                    print("✓ All keys matched perfectly!")
                else:
                    if missing_keys:
                        print(f"⚠ Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
                    if unexpected_keys:
                        print(f"⚠ Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")

                print(f"{'='*60}\n")
            except Exception as e:
                print(f"✗ Error loading pretrained weights: {e}")
                print("Starting from scratch (CLIP ImageNet weights)")
                print(f"{'='*60}\n")
        elif opt.pretrained_stage1_path:
            print(f"\n⚠ Pretrained path specified but not found: {opt.pretrained_stage1_path}")
            print("Starting from scratch (CLIP ImageNet weights)\n")
        else:
            print("\nNo pretrained weights specified. Starting from scratch (CLIP ImageNet weights)\n")

        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"Total model parameters: {total_params:.2f}M")

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6
        trainable_ratio = (trainable_params / total_params) * 100
        print(f"Trainable parameters: {trainable_params:.2f}M ({trainable_ratio:.2f}%)")

        # Multi-GPU setup
        self.use_multi_gpu = opt.use_multi_gpu if hasattr(opt, 'use_multi_gpu') else False
        if self.use_multi_gpu and torch.cuda.device_count() > 1:
            gpu_ids = [int(x) for x in opt.gpu_ids.split(',')]
            print(f"\n{'='*60}")
            print(f"Multi-GPU Training Enabled")
            print(f"  Available GPUs: {torch.cuda.device_count()}")
            print(f"  Using GPUs: {gpu_ids}")
            print(f"  Effective batch size: {opt.stage1_batch_size} x {len(gpu_ids)} = {opt.stage1_batch_size * len(gpu_ids)}")
            print(f"{'='*60}\n")

            self.model = torch.nn.DataParallel(self.model, device_ids=gpu_ids)
            self.device = torch.device(f"cuda:{gpu_ids[0]}")
            self.is_parallel = True
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.is_parallel = False
            if self.use_multi_gpu:
                print("\n⚠ Multi-GPU requested but only 1 GPU available. Using single GPU.\n")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.stage1_learning_rate, betas=(0.9, 0.999))
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=opt.stage1_lr_decay_step, gamma=opt.stage1_lr_decay_factor)
        self.scaler = GradScaler()

        self.best_val_loss = float('inf')

    def train_epoch(self, dataloader: DataLoader, criterion):
        total_loss = 0.0
        total_batches = 0

        running_loss = 0.0
        batch_number = 0

        self.model.to(self.device)
        self.model.train()

        for batch_idx, (data, target) in enumerate(tqdm(dataloader)):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            with autocast('cuda'):
                output, _ = self.model(data)
                loss = criterion(output.squeeze(1), target.type(torch.float32))

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            total_loss += loss.item()
            batch_number += 1
            total_batches += 1

        return total_loss / (total_batches + 1)

    def validate_epoch(self, dataloader: DataLoader, criterion, epoch: int, writer: SummaryWriter = None):
        self.model.to(self.device)
        self.model.eval()
        running_loss = 0.0
        dataset_preds = []
        dataset_targets = []

        for data, target in tqdm(dataloader):
            data, target = data.to(self.device), target.to(self.device)

            with torch.no_grad():
                with autocast('cuda'):
                    pre, _ = self.model(data)
                    loss = criterion(pre.squeeze(1), target.type(torch.float32))
                    running_loss += loss.item()
                    pre_prob = pre.cpu().numpy()
                    target = target.cpu().numpy()
                    dataset_preds.append(pre_prob)
                    dataset_targets.append(target)

        # Handle empty dataloader case
        if len(dataset_preds) == 0:
            print("\nWARNING: Validation dataloader is empty. Returning default metrics.")
            return 0.0, 0.0, 0.0

        dataset_preds = np.concatenate(dataset_preds)
        dataset_targets = np.concatenate(dataset_targets)

        acc = accuracy_score(dataset_targets, dataset_preds > 0)
        ap = average_precision_score(dataset_targets, dataset_preds)

        if writer is not None:
            writer.add_scalar('Loss/Validation', running_loss / len(dataloader), epoch)
            writer.add_scalar('Accuracy', acc, epoch)
            writer.add_scalar('Average Precision', ap, epoch)

        return running_loss / len(dataloader), acc, ap

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, criterion, num_epochs: int,
              checkpoint_dir: str = None, writer: SummaryWriter = None):
        best_val_loss = float('inf')
        best_val_acc = 0.0
        for epoch in range(num_epochs):
            print(f"Training" + "-" * 60)
            time.sleep(1)
            train_loss = self.train_epoch(train_dataloader, criterion)
            print(f"Validating" + "*" * 60)
            time.sleep(1)
            val_loss, acc, ap = self.validate_epoch(val_dataloader, criterion, epoch, writer=writer)

            print(
                f'{time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())}\nTrain Epoch: {epoch+1}: \n'
                f'train loss: {train_loss}\nval_loss:{val_loss}\nacc:{acc}\nap:{ap}')

            os.makedirs(checkpoint_dir, exist_ok=True)

            # Get the actual model (unwrap DataParallel if needed)
            model_to_save = self.model.module if self.is_parallel else self.model

            if (epoch+1) % 1 == 0:
                checkpoint_path_1 = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_acc': acc,
                    'val_ap': ap,
                }, checkpoint_path_1)
                print(f'Model checkpoint saved to {checkpoint_path_1}')

            # Save checkpoint if validation loss improved
            if val_loss < best_val_loss and checkpoint_dir is not None:
                best_val_loss = val_loss
                best_val_acc = acc
                checkpoint_path_2 = os.path.join(checkpoint_dir, f'intermediate_model_best.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_acc': acc,
                    'val_ap': ap,
                    'best_val_loss': best_val_loss,
                    'best_val_acc': best_val_acc,
                }, checkpoint_path_2)
                print(f'Model checkpoint saved to {checkpoint_path_2}')

            self.scheduler.step()

        print('Training complete.')
