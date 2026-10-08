import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# Cityscapes Class ID Mapping
# Standard raw Cityscapes class IDs: Person = 24, Car = 26
CITYSCAPES_PERSON_ID = 24
CITYSCAPES_CAR_ID = 26

# Target class indexing: 0 = Background, 1 = Person, 2 = Car
CLASS_MAP = {
    CITYSCAPES_PERSON_ID: 1,
    CITYSCAPES_CAR_ID: 2
}

IGNORE_LABEL = 255


def parse_instance_id(instance_id):
    """
    Cityscapes instanceId encoding:
    instance_id = class_id * 1000 + instance_idx
    If instance_id < 1000, it represents a class label without individual instance index.
    Returns: (class_id, instance_idx)
    """
    if instance_id < 1000:
        return instance_id, 0
    else:
        return instance_id // 1000, instance_id % 1000


def generate_center_offset_targets(instance_img_np, target_size=(256, 512), sigma=8.0):
    """
    Generates training targets for proposal-free Center + Offset regression.
    
    Args:
        instance_img_np: numpy array (H_orig, W_orig) containing encoded instanceIds.
        target_size: tuple (H, W) target spatial dimensions.
        sigma: Gaussian kernel standard deviation for center heatmap targets.
        
    Returns:
        semantic_target: torch.LongTensor (H, W) with class IDs (0: BG, 1: Person, 2: Car, 255: Ignore)
        center_target: torch.FloatTensor (2, H, W) Gaussian heatmaps for [Person, Car]
        offset_target: torch.FloatTensor (2, H, W) Offsets (dx, dy) towards instance centroid
        offset_weight: torch.FloatTensor (1, H, W) Binary mask indicating valid instance pixels
    """
    # Resize instance map using Nearest Neighbor to preserve exact instance IDs
    pil_inst = Image.fromarray(instance_img_np)
    if (pil_inst.height, pil_inst.width) != target_size:
        pil_inst = pil_inst.resize((target_size[1], target_size[0]), resample=Image.NEAREST)
    inst_map = np.array(pil_inst, dtype=np.int32)
    
    H, W = target_size
    semantic_target = np.zeros((H, W), dtype=np.int64)
    center_target = np.zeros((2, H, W), dtype=np.float32)
    offset_target = np.zeros((2, H, W), dtype=np.float32)
    offset_weight = np.zeros((1, H, W), dtype=np.float32)
    
    # Coordinate grid for offset calculations
    grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    unique_ids = np.unique(inst_map)
    
    for inst_id in unique_ids:
        if inst_id == 0:
            continue
            
        class_id, inst_idx = parse_instance_id(inst_id)
        
        # Check if target class is Person (24 -> 1) or Car (26 -> 2)
        if class_id not in CLASS_MAP:
            # Mask out non-person/car pixels as ignore or background
            # If it's a known non-target class, leave as background 0
            continue
            
        target_cls = CLASS_MAP[class_id]  # 1 for Person, 2 for Car
        cls_channel = target_cls - 1      # 0 for Person, 1 for Car in 2-channel heatmap
        
        mask = (inst_map == inst_id)
        
        if inst_idx == 0:
            # Group label or region without instance distinction (e.g. crowds/license plates)
            # Mark as IGNORE_LABEL for semantic loss, ignore in offset & center
            semantic_target[mask] = IGNORE_LABEL
            continue
            
        # Valid distinct instance
        semantic_target[mask] = target_cls
        offset_weight[0, mask] = 1.0
        
        # Compute instance centroid (cx, cy)
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0:
            continue
            
        cy = np.mean(y_indices)
        cx = np.mean(x_indices)
        
        # 1. Offset regression target: (cx - x, cy - y)
        dx = cx - grid_x[mask]
        dy = cy - grid_y[mask]
        offset_target[0, mask] = dx
        offset_target[1, mask] = dy
        
        # 2. Gaussian Center Heatmap target
        # Use localized crop or full grid Gaussian computation
        dist_sq = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
        gaussian = np.exp(-dist_sq / (2 * (sigma ** 2)))
        
        # Take max over overlapping Gaussians for the same class channel
        center_target[cls_channel] = np.maximum(center_target[cls_channel], gaussian)
        
    return (
        torch.from_numpy(semantic_target).long(),
        torch.from_numpy(center_target).float(),
        torch.from_numpy(offset_target).float(),
        torch.from_numpy(offset_weight).float()
    )


class CityscapesInstanceDataset(Dataset):
    """
    Cityscapes Instance Segmentation Dataset supporting both real Cityscapes
    and synthetic benchmark datasets.
    """
    def __init__(self, root_dir, split='train', target_size=(256, 512), sigma=8.0, is_synthetic=False):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.target_size = target_size  # (H, W) e.g., (256, 512) or (512, 1024)
        self.sigma = sigma
        self.is_synthetic = is_synthetic
        
        self.image_paths = []
        self.instance_paths = []
        self._find_file_pairs()
        
        # Image Normalization
        self.img_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _find_file_pairs(self):
        """Discovers corresponding image and instance ground truth file pairs."""
        if self.is_synthetic or os.path.exists(os.path.join(self.root_dir, 'images')):
            img_dir = os.path.join(self.root_dir, 'images', self.split)
            inst_dir = os.path.join(self.root_dir, 'instances', self.split)
            
            if not os.path.exists(img_dir):
                img_dir = os.path.join(self.root_dir, 'images')
                inst_dir = os.path.join(self.root_dir, 'instances')
                
            if os.path.exists(img_dir):
                for fname in sorted(os.listdir(img_dir)):
                    if fname.endswith(('.png', '.jpg', '.jpeg')):
                        base_prefix = fname.replace('_leftImg8bit.png', '').replace('.png', '').replace('.jpg', '')
                        img_path = os.path.join(img_dir, fname)
                        inst_path = os.path.join(inst_dir, f"{base_prefix}_gtFine_instanceIds.png")
                        if not os.path.exists(inst_path):
                            inst_path = os.path.join(inst_dir, fname)
                        if os.path.exists(inst_path):
                            self.image_paths.append(img_path)
                            self.instance_paths.append(inst_path)
        else:
            # Standard Cityscapes dataset hierarchy
            img_pattern = os.path.join(self.root_dir, 'leftImg8bit', self.split, '*', '*_leftImg8bit.png')
            found_imgs = sorted(glob.glob(img_pattern))
            
            for img_path in found_imgs:
                inst_path = img_path.replace('leftImg8bit', 'gtFine').replace('_leftImg8bit.png', '_gtFine_instanceIds.png')
                if os.path.exists(inst_path):
                    self.image_paths.append(img_path)
                    self.instance_paths.append(inst_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        inst_path = self.instance_paths[idx]
        
        # Load Image and Instance ground truth
        raw_img = Image.open(img_path).convert('RGB')
        inst_img = Image.open(inst_path)
        
        # Resize image to target spatial dimensions
        raw_img_resized = raw_img.resize((self.target_size[1], self.target_size[0]), resample=Image.BILINEAR)
        img_tensor = self.img_transform(raw_img_resized)
        
        # Parse instance IDs and compute target maps
        inst_np = np.array(inst_img)
        sem_target, center_target, offset_target, offset_weight = generate_center_offset_targets(
            inst_np, target_size=self.target_size, sigma=self.sigma
        )
        
        return {
            'image': img_tensor,                      # (3, H, W)
            'semantic_target': sem_target,            # (H, W)
            'center_target': center_target,          # (2, H, W)
            'offset_target': offset_target,          # (2, H, W)
            'offset_weight': offset_weight,          # (1, H, W)
            'image_path': img_path
        }
