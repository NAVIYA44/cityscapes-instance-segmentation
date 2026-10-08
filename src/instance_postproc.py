import numpy as np
import torch
import torch.nn.functional as F


def extract_center_peaks(center_heatmap, sem_pred=None, kernel_size=5, threshold=0.40, min_dist=14.0, top_k=20):
    """
    Extracts local maxima (peaks) from center heatmaps using NMS max pooling,
    dilated semantic mask filtering, confidence thresholding, and top-k suppression.
    Threshold can be a single float or a dict {0: person_thresh, 1: car_thresh}.
    """
    if center_heatmap.dim() == 4:
        center_heatmap = center_heatmap.squeeze(0)
        
    num_classes, H, W = center_heatmap.shape
    padding = (kernel_size - 1) // 2
    
    max_pooled = F.max_pool2d(center_heatmap.unsqueeze(0), kernel_size=kernel_size, stride=1, padding=padding).squeeze(0)
    peaks_per_class = {}
    
    # Class-specific spatial minimum distances (Cars are larger than Persons)
    class_min_dists = {0: max(12.0, min_dist), 1: max(45.0, min_dist * 2.2)} # 0: Person, 1: Car
    
    # If threshold is a float, apply calibrated class thresholds: Person converges to 0.44-0.46 while Car to 0.70-1.0
    if isinstance(threshold, (int, float)):
        # Scale person threshold appropriately if user supplies a single global slider
        thresh_dict = {
            0: float(threshold),
            1: max(float(threshold), 0.50)
        }
    elif isinstance(threshold, dict):
        thresh_dict = threshold
    else:
        thresh_dict = {0: 0.40, 1: 0.50}

    for c in range(num_classes):
        target_cls = c + 1 # 1: Person, 2: Car
        cls_map = center_heatmap[c]
        c_thresh = thresh_dict.get(c, 0.40)
        
        # Candidate peaks must be local maxima and exceed threshold
        keep = (cls_map == max_pooled[c]) & (cls_map >= c_thresh)
        
        if sem_pred is not None:
            sem_mask = torch.from_numpy(sem_pred == target_cls).float().unsqueeze(0).unsqueeze(0)
            # Dilate semantic mask by 7x7 so candidate centers are not rejected due to slight 1-2px centroid offsets
            dilated_sem = F.max_pool2d(sem_mask, kernel_size=7, stride=1, padding=3).squeeze().bool().to(cls_map.device)
            keep = keep & dilated_sem
            
        y_coords, x_coords = torch.where(keep)
        scores = cls_map[keep]
        
        if len(scores) == 0:
            peaks_per_class[c] = []
            continue
            
        # Sort candidates by score descending
        sorted_indices = torch.argsort(scores, descending=True)
        
        peaks = []
        c_min_dist = class_min_dists.get(c, min_dist)
        
        for idx in sorted_indices:
            x = float(x_coords[idx])
            y = float(y_coords[idx])
            score = float(scores[idx])
            
            # Check spatial distance against already accepted peaks (Spatial NMS)
            too_close = False
            for prev_p in peaks:
                dist = np.sqrt((x - prev_p['x'])**2 + (y - prev_p['y'])**2)
                if dist < c_min_dist:
                    too_close = True
                    break
                    
            if not too_close:
                peaks.append({'x': x, 'y': y, 'score': score})
                if len(peaks) >= top_k:
                    break
                
        peaks_per_class[c] = peaks
        
    return peaks_per_class


def parse_instance_segmentation(semantic_logits, center_heatmap, offset_map, 
                                center_thresh=0.40, sem_thresh=0.30, max_dist=96.0, 
                                min_instance_pixels=40, top_k=20, min_dist=14.0):
    """
    Proposal-Free Panoptic/Instance Post-Processing via Center + Offset Regression.
    Incorporates class-aware confidence gating, spatial NMS, and size filtering.
    """
    if semantic_logits.dim() == 4:
        semantic_logits = semantic_logits.squeeze(0)
    if center_heatmap.dim() == 4:
        center_heatmap = center_heatmap.squeeze(0)
    if offset_map.dim() == 4:
        offset_map = offset_map.squeeze(0)
        
    H, W = semantic_logits.shape[1], semantic_logits.shape[2]
    
    # 1. Semantic predictions with foreground confidence gating
    sem_probs = torch.softmax(semantic_logits, dim=0).cpu().numpy() # (3, H, W)
    sem_pred = np.argmax(sem_probs, axis=0) # (H, W)
    
    # Class-aware semantic thresholding
    if isinstance(sem_thresh, (int, float)):
        sem_thresh_dict = {1: min(float(sem_thresh), 0.35), 2: max(float(sem_thresh), 0.45)}
    elif isinstance(sem_thresh, dict):
        sem_thresh_dict = sem_thresh
    else:
        sem_thresh_dict = {1: 0.30, 2: 0.45}

    for c_id in [1, 2]:
        mask_c = (sem_pred == c_id)
        drop = mask_c & (sem_probs[c_id] < sem_thresh_dict.get(c_id, 0.35))
        sem_pred[drop] = 0
    
    # 2. Extract center peaks
    peaks_dict = extract_center_peaks(
        center_heatmap, sem_pred=sem_pred, kernel_size=5, 
        threshold=center_thresh, min_dist=min_dist, top_k=top_k
    )
    
    # Coordinate grid
    grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Predicted center per pixel: (x + dx, y + dy)
    dx = offset_map[0].cpu().numpy()
    dy = offset_map[1].cpu().numpy()
    
    target_x = grid_x + dx
    target_y = grid_y + dy
    
    instance_mask = np.zeros((H, W), dtype=np.int32)
    instance_details = []
    next_instance_id = 1
    
    class_names = {1: 'Person', 2: 'Car'}
    # Cars need higher minimum pixel count than persons (persons are slender)
    class_min_pixels = {1: max(35, int(min_instance_pixels)), 2: max(80, int(min_instance_pixels * 1.8))}
    
    # Process each class
    for cls_idx in [1, 2]:
        cls_channel = cls_idx - 1
        peaks = peaks_dict.get(cls_channel, [])
        fg_mask = (sem_pred == cls_idx)
        min_pixels = class_min_pixels.get(cls_idx, min_instance_pixels)
        
        if not np.any(fg_mask):
            continue
            
        if len(peaks) == 0:
            # Fallback: Connected components if no peaks detected for this class
            from scipy.ndimage import label
            labeled_mask, num_features = label(fg_mask)
            for feat_i in range(1, min(num_features + 1, top_k + 1)):
                feat_mask = (labeled_mask == feat_i)
                if np.sum(feat_mask) < min_pixels:
                    continue
                instance_mask[feat_mask] = next_instance_id
                y_idx, x_idx = np.where(feat_mask)
                instance_details.append({
                    'instance_id': next_instance_id,
                    'class_id': cls_idx,
                    'class_name': class_names[cls_idx],
                    'score': float(np.mean(sem_probs[cls_idx][feat_mask])),
                    'center': (float(np.mean(x_idx)), float(np.mean(y_idx))),
                    'bbox': [int(np.min(x_idx)), int(np.min(y_idx)), int(np.max(x_idx)), int(np.max(y_idx))],
                    'area': int(np.sum(feat_mask))
                })
                next_instance_id += 1
            continue
            
        # Match each foreground pixel to closest peak
        peak_coords = np.array([[p['x'], p['y']] for p in peaks])  # (K, 2)
        peak_scores = [p['score'] for p in peaks]
        
        fg_y, fg_x = np.where(fg_mask)
        fg_target_x = target_x[fg_y, fg_x]
        fg_target_y = target_y[fg_y, fg_x]
        fg_targets = np.stack([fg_target_x, fg_target_y], axis=1)  # (N_fg, 2)
        
        dists = np.linalg.norm(fg_targets[:, None, :] - peak_coords[None, :, :], axis=2)
        min_peak_idx = np.argmin(dists, axis=1)
        min_dists = np.min(dists, axis=1)
        
        # Assign pixel to nearest peak if within max_dist threshold
        for k in range(len(peaks)):
            inst_pixel_indices = np.where((min_peak_idx == k) & (min_dists <= max_dist))[0]
            if len(inst_pixel_indices) < min_pixels:
                continue
                
            inst_y = fg_y[inst_pixel_indices]
            inst_x = fg_x[inst_pixel_indices]
            
            curr_inst_id = next_instance_id
            instance_mask[inst_y, inst_x] = curr_inst_id
            
            instance_details.append({
                'instance_id': curr_inst_id,
                'class_id': cls_idx,
                'class_name': class_names[cls_idx],
                'score': peak_scores[k],
                'center': (peaks[k]['x'], peaks[k]['y']),
                'bbox': [int(np.min(inst_x)), int(np.min(inst_y)), int(np.max(inst_x)), int(np.max(inst_y))],
                'area': len(inst_pixel_indices)
            })
            next_instance_id += 1

    return instance_mask, sem_pred, instance_details
