import numpy as np
import torch


def compute_iou(mask_a, mask_b):
    """Computes Intersection over Union (IoU) between two binary masks."""
    intersection = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def compute_semantic_iou(pred_sem, target_sem, num_classes=3, ignore_index=255):
    """
    Computes per-class IoU and mean IoU for semantic segmentation.
    
    Classes: 0: Background, 1: Person, 2: Car
    """
    if isinstance(pred_sem, torch.Tensor):
        pred_sem = pred_sem.cpu().numpy()
    if isinstance(target_sem, torch.Tensor):
        target_sem = target_sem.cpu().numpy()
        
    valid_mask = (target_sem != ignore_index)
    pred_valid = pred_sem[valid_mask]
    target_valid = target_sem[valid_mask]
    
    ious = {}
    class_names = {0: 'Background', 1: 'Person', 2: 'Car'}
    
    for c in range(num_classes):
        pred_c = (pred_valid == c)
        target_c = (target_valid == c)
        
        intersection = np.logical_and(pred_c, target_c).sum()
        union = np.logical_or(pred_c, target_c).sum()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        ious[class_names.get(c, f"Class_{c}")] = float(iou)
        
    # mIoU over Person and Car foreground classes
    fg_ious = [ious['Person'], ious['Car']]
    ious['mIoU_fg'] = float(np.mean(fg_ious))
    ious['mIoU_all'] = float(np.mean(list(ious.values())))
    
    return ious


def evaluate_instance_predictions(pred_instances, target_inst_map, iou_thresholds=[0.5, 0.75]):
    """
    Evaluates instance-level predictions against ground truth instance map.
    
    Args:
        pred_instances: list of dicts [{'instance_id': 1, 'class_id': 1, 'score': 0.9, ...}, ...]
        target_inst_map: numpy array (H, W) of encoded ground truth instance IDs
        iou_thresholds: list of IoU thresholds (e.g. 0.50, 0.75)
        
    Returns:
        metrics: dict with Precision, Recall, F1, and AP at specified IoU thresholds
    """
    from src.dataset import parse_instance_id, CLASS_MAP
    
    # Extract GT instance masks and class IDs
    gt_instances = []
    unique_gt_ids = np.unique(target_inst_map)
    
    for gt_id in unique_gt_ids:
        if gt_id == 0:
            continue
        cls_id, inst_idx = parse_instance_id(gt_id)
        if cls_id in CLASS_MAP and inst_idx > 0:
            target_cls = CLASS_MAP[cls_id]
            gt_mask = (target_inst_map == gt_id)
            gt_instances.append({
                'gt_id': gt_id,
                'class_id': target_cls,
                'mask': gt_mask,
                'area': np.sum(gt_mask)
            })
            
    results = {}
    
    for iou_thresh in iou_thresholds:
        tp = 0
        fp = 0
        fn = 0
        
        matched_gt = set()
        
        # Sort predictions by confidence score descending
        sorted_preds = sorted(pred_instances, key=lambda x: x.get('score', 0.5), reverse=True)
        
        for pred in sorted_preds:
            p_mask = pred.get('mask', None)
            if p_mask is None:
                continue
            p_cls = pred['class_id']
            
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gt_instances):
                if gt_idx in matched_gt:
                    continue
                if gt['class_id'] != p_cls:
                    continue
                    
                iou = compute_iou(p_mask, gt['mask'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
                    
            if best_iou >= iou_thresh and best_gt_idx != -1:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
                
        fn = len(gt_instances) - len(matched_gt)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results[f'AP_{int(iou_thresh*100)}'] = precision
        results[f'Precision_{int(iou_thresh*100)}'] = precision
        results[f'Recall_{int(iou_thresh*100)}'] = recall
        results[f'F1_{int(iou_thresh*100)}'] = f1
        
    return results
