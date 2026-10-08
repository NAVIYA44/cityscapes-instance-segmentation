import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import CityscapesInstanceDataset
from src.model import ProposalFreeInstanceSegNet
from src.instance_postproc import parse_instance_segmentation
from src.metrics import compute_semantic_iou, evaluate_instance_predictions


def evaluate_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Evaluating model on device: {device}", flush=True)
    
    val_dataset = CityscapesInstanceDataset(
        root_dir=args.data_dir, split='val', target_size=(args.height, args.width), is_synthetic=args.is_synthetic
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    model = ProposalFreeInstanceSegNet(backbone_name=args.backbone, pretrained=False).to(device)
    
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded trained checkpoint from '{args.checkpoint}'")
    else:
        print(f"Warning: Checkpoint '{args.checkpoint}' not found. Evaluating with initialized model weights.")

    model.eval()
    
    all_sem_ious = []
    instance_metrics_list = []
    
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            image = batch['image'].to(device)
            target_sem = batch['semantic_target'].squeeze(0).numpy()
            target_inst_path = val_dataset.instance_paths[idx]
            from PIL import Image
            target_inst_map = np.array(Image.open(target_inst_path).resize((args.width, args.height), Image.NEAREST))

            preds = model(image)
            
            # Instance post-processing
            inst_mask, sem_pred, inst_details = parse_instance_segmentation(
                preds['semantic_logits'], preds['center_heatmap'], preds['offset_map'],
                center_thresh=args.center_thresh
            )
            
            # Attach predicted masks to inst_details for AP computation
            for inst in inst_details:
                inst['mask'] = (inst_mask == inst['instance_id'])
                
            # Semantic IoU
            sem_iou = compute_semantic_iou(sem_pred, target_sem)
            all_sem_ious.append(sem_iou)
            
            # Instance Mask AP / Precision / Recall
            inst_metrics = evaluate_instance_predictions(inst_details, target_inst_map)
            instance_metrics_list.append(inst_metrics)

    # Aggregate metrics
    mean_person_iou = np.mean([m['Person'] for m in all_sem_ious])
    mean_car_iou = np.mean([m['Car'] for m in all_sem_ious])
    mean_fg_iou = np.mean([m['mIoU_fg'] for m in all_sem_ious])
    
    mean_ap50 = np.mean([m.get('AP_50', 0.0) for m in instance_metrics_list])
    mean_ap75 = np.mean([m.get('AP_75', 0.0) for m in instance_metrics_list])
    mean_rec50 = np.mean([m.get('Recall_50', 0.0) for m in instance_metrics_list])
    mean_f1_50 = np.mean([m.get('F1_50', 0.0) for m in instance_metrics_list])
    
    print("\n=======================================================")
    print("        EVALUATION REPORT — INSTANCE SEGMENTATION      ")
    print("=======================================================")
    print(f" Semantic mIoU (Person)       : {mean_person_iou*100:.2f}%")
    print(f" Semantic mIoU (Car)          : {mean_car_iou*100:.2f}%")
    print(f" Semantic mIoU (Foreground)   : {mean_fg_iou*100:.2f}%")
    print("-------------------------------------------------------")
    print(f" Instance Mask AP @ IoU 0.50  : {mean_ap50*100:.2f}%")
    print(f" Instance Mask AP @ IoU 0.75  : {mean_ap75*100:.2f}%")
    print(f" Instance Mask Recall @ 0.50  : {mean_rec50*100:.2f}%")
    print(f" Instance Mask F1-Score @ 0.50: {mean_f1_50*100:.2f}%")
    print("=======================================================\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Cityscapes Proposal-Free Instance Segmentation Model")
    parser.add_argument('--data_dir', type=str, default='data/synthetic')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--is_synthetic', action='store_true', default=True)
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--center_thresh', type=float, default=0.2)
    parser.add_argument('--cpu', action='store_true')
    
    args = parser.parse_args()
    evaluate_model(args)
