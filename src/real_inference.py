import torch
import torchvision.models.segmentation as seg
import numpy as np
from scipy.ndimage import label, center_of_mass, distance_transform_edt, maximum_filter

_REAL_MODEL = None

def get_real_model(device='cpu'):
    global _REAL_MODEL
    if _REAL_MODEL is None:
        _REAL_MODEL = seg.deeplabv3_resnet50(weights=seg.DeepLabV3_ResNet50_Weights.DEFAULT).to(device)
        _REAL_MODEL.eval()
    return _REAL_MODEL

def predict_real_world(img_tensor, device='cpu'):
    """
    Predicts proposal-free dense representations (Semantic logits, Center Heatmaps, Offset vectors)
    for real-world photographic street images using deep feature representation.
    
    Returns a dictionary identical to ProposalFreeInstanceSegNet:
      - 'semantic_logits': (1, 3, H, W) -> [Background, Person, Car]
      - 'center_heatmap':  (1, 2, H, W) -> [Person Heatmap, Car Heatmap]
      - 'offset_map':      (1, 2, H, W) -> [dx, dy] 2D spatial offsets towards centroid
    """
    model = get_real_model(device)
    with torch.no_grad():
        out = model(img_tensor.to(device))['out']  # (1, 21, H, W)

    probs = torch.softmax(out, dim=1)[0]  # (21, H, W)
    H, W = probs.shape[1], probs.shape[2]

    # COCO class 15 = person, class 7 = car
    person_prob = probs[15]
    car_prob = probs[7]
    bg_prob = torch.clamp(1.0 - person_prob - car_prob, min=1e-5)

    # 3-class semantic probability stack
    sem_probs = torch.stack([bg_prob, person_prob, car_prob], dim=0)  # (3, H, W)
    sem_logits = torch.log(sem_probs + 1e-7).unsqueeze(0)             # (1, 3, H, W)

    # Synthesize Gaussian center heatmaps and 2D offset vector fields per object instance
    center_heat = np.zeros((2, H, W), dtype=np.float32)
    offset_map = np.zeros((2, H, W), dtype=np.float32)
    grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    sem_pred = torch.argmax(sem_probs, dim=0).cpu().numpy()

    for cls_idx in [1, 2]:
        fg_mask = (sem_pred == cls_idx)
        labeled, num_features = label(fg_mask)
        cls_channel = cls_idx - 1

        for i in range(1, num_features + 1):
            comp = (labeled == i)
            area = np.sum(comp)
            min_area = 40 if cls_idx == 1 else 100
            if area < min_area:
                continue

            cy, cx = center_of_mass(comp)
            sigma = 8.0
            dist_sq = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
            heat = np.exp(-dist_sq / (2 * (sigma ** 2))).astype(np.float32)
            center_heat[cls_channel] = np.maximum(center_heat[cls_channel], heat)

            offset_map[0, comp] = (cx - grid_x[comp]).astype(np.float32)
            offset_map[1, comp] = (cy - grid_y[comp]).astype(np.float32)

    center_heat_t = torch.from_numpy(center_heat).unsqueeze(0).to(device)
    offset_map_t = torch.from_numpy(offset_map).unsqueeze(0).to(device)

    return {
        'semantic_logits': sem_logits.to(device),
        'center_heatmap': center_heat_t,
        'offset_map': offset_map_t
    }
