import os
import argparse
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import CityscapesInstanceDataset
from src.model import ProposalFreeInstanceSegNet
from src.loss import ProposalFreeInstanceLoss
from src.instance_postproc import parse_instance_segmentation
from src.metrics import compute_semantic_iou, evaluate_instance_predictions


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    sem_loss_sum = 0.0
    center_loss_sum = 0.0
    offset_loss_sum = 0.0

    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        targets = {
            'semantic_target': batch['semantic_target'].to(device),
            'center_target': batch['center_target'].to(device),
            'offset_target': batch['offset_target'].to(device),
            'offset_weight': batch['offset_weight'].to(device)
        }

        optimizer.zero_grad()
        predictions = model(images)
        loss_dict = criterion(predictions, targets)

        loss = loss_dict['total_loss']
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        sem_loss_sum += loss_dict['sem_loss'].item()
        center_loss_sum += loss_dict['center_loss'].item()
        offset_loss_sum += loss_dict['offset_loss'].item()

    num_batches = len(dataloader)
    return {
        'total_loss': total_loss / num_batches,
        'sem_loss': sem_loss_sum / num_batches,
        'center_loss': center_loss_sum / num_batches,
        'offset_loss': offset_loss_sum / num_batches
    }


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    sem_loss_sum = 0.0
    center_loss_sum = 0.0
    offset_loss_sum = 0.0
    
    all_sem_ious = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            targets = {
                'semantic_target': batch['semantic_target'].to(device),
                'center_target': batch['center_target'].to(device),
                'offset_target': batch['offset_target'].to(device),
                'offset_weight': batch['offset_weight'].to(device)
            }

            predictions = model(images)
            loss_dict = criterion(predictions, targets)

            total_loss += loss_dict['total_loss'].item()
            sem_loss_sum += loss_dict['sem_loss'].item()
            center_loss_sum += loss_dict['center_loss'].item()
            offset_loss_sum += loss_dict['offset_loss'].item()
            
            # Compute semantic IoU
            sem_pred = torch.argmax(predictions['semantic_logits'], dim=1)
            for i in range(images.size(0)):
                ious = compute_semantic_iou(sem_pred[i], targets['semantic_target'][i])
                all_sem_ious.append(ious['mIoU_fg'])

    num_batches = len(dataloader)
    mean_sem_iou = np.mean(all_sem_ious) if len(all_sem_ious) > 0 else 0.0
    
    return {
        'total_loss': total_loss / num_batches,
        'sem_loss': sem_loss_sum / num_batches,
        'center_loss': center_loss_sum / num_batches,
        'offset_loss': offset_loss_sum / num_batches,
        'mIoU_fg': mean_sem_iou
    }


def run_training(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Datasets and Loaders
    train_dataset = CityscapesInstanceDataset(
        root_dir=args.data_dir, split='train', target_size=(args.height, args.width), is_synthetic=args.is_synthetic
    )
    val_dataset = CityscapesInstanceDataset(
        root_dir=args.data_dir, split='val', target_size=(args.height, args.width), is_synthetic=args.is_synthetic
    )
    
    if len(train_dataset) == 0:
        raise ValueError(f"No training images found in '{args.data_dir}'. Ensure dataset paths or synthetic data generation is set up.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 2. Model, Loss, Optimizer
    model = ProposalFreeInstanceSegNet(backbone_name=args.backbone, pretrained=True).to(device)
    criterion = ProposalFreeInstanceLoss().to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')

    print(f"\n--- Starting Training ({args.epochs} Epochs) ---")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        v_metrics = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"Train Loss: {t_metrics['total_loss']:.4f} (Sem: {t_metrics['sem_loss']:.4f}, Ctr: {t_metrics['center_loss']:.4f}, Off: {t_metrics['offset_loss']:.4f}) | "
              f"Val Loss: {v_metrics['total_loss']:.4f} | Val mIoU: {v_metrics['mIoU_fg']:.4f}")

        # Save Checkpoint
        if v_metrics['total_loss'] < best_val_loss:
            best_val_loss = v_metrics['total_loss']
            ckpt_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'args': vars(args)
            }, ckpt_path)

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/60:.2f} minutes. Best Val Loss: {best_val_loss:.4f}")
    print(f"Saved best checkpoint to '{os.path.join(args.save_dir, 'best_model.pth')}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Cityscapes Proposal-Free Instance Segmentation Model")
    parser.add_argument('--data_dir', type=str, default='data/synthetic', help="Dataset root directory")
    parser.add_argument('--is_synthetic', action='store_true', default=True, help="Flag for synthetic dataset")
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--height', type=int, default=256, help="Target image height")
    parser.add_argument('--width', type=int, default=512, help="Target image width")
    parser.add_argument('--epochs', type=int, default=15, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
    parser.add_argument('--lr', type=float, default=5e-4, help="Learning rate")
    parser.add_argument('--save_dir', type=str, default='checkpoints', help="Checkpoint save directory")
    parser.add_argument('--cpu', action='store_true', help="Force CPU training")
    
    args = parser.parse_args()
    run_training(args)
