import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.dataset import CityscapesInstanceDataset
from src.utils import visualize_semantic_segmentation, visualize_offset_vectors, visualize_instance_segmentation


def validate_data_pipeline(data_dir='data/synthetic', is_synthetic=True, output_plot='check_data_output.png'):
    """
    Stage 1 Validation Script:
    Verifies instanceId parsing, object mask extraction, center heatmaps, and offset vector targets.
    """
    print(f"--- Stage 1 Data Pipeline Validation ---")
    dataset = CityscapesInstanceDataset(data_dir, split='train', is_synthetic=is_synthetic)
    
    if len(dataset) == 0:
        print(f"Error: No data found in '{data_dir}'! Please generate synthetic data first.")
        return False
        
    print(f"Successfully loaded dataset with {len(dataset)} samples.")
    sample = dataset[0]
    
    img_tensor = sample['image']          # (3, H, W) normalized
    sem_target = sample['semantic_target']# (H, W)
    center_target = sample['center_target']# (2, H, W)
    offset_target = sample['offset_target']# (2, H, W)
    offset_weight = sample['offset_weight']# (1, H, W)
    
    # Denormalize Image
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_np = (img_tensor.numpy() * std + mean).transpose(1, 2, 0)
    img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    
    sem_np = sem_target.numpy()
    center_person = center_target[0].numpy()
    center_car = center_target[1].numpy()
    offset_np = offset_target.numpy()
    weight_np = offset_weight[0].numpy()
    
    # Extract object count statistics
    person_pixels = np.sum(sem_np == 1)
    car_pixels = np.sum(sem_np == 2)
    bg_pixels = np.sum(sem_np == 0)
    
    print(f"Sample 0 Statistics:")
    print(f"  - Image shape: {img_np.shape}")
    print(f"  - Person pixels: {person_pixels}")
    print(f"  - Car pixels: {car_pixels}")
    print(f"  - Background pixels: {bg_pixels}")
    print(f"  - Center Person Heatmap max score: {center_person.max():.4f}")
    print(f"  - Center Car Heatmap max score: {center_car.max():.4f}")
    print(f"  - Offset Range (dx, dy): [{offset_np[0].min():.1f}, {offset_np[0].max():.1f}], [{offset_np[1].min():.1f}, {offset_np[1].max():.1f}]")
    
    # Create Diagnostic 6-Panel Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. RGB Street Image
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title("1. Input Street Image", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. Semantic Segmentation Ground Truth Overlay
    sem_vis = visualize_semantic_segmentation(img_np, sem_np, alpha=0.5)
    axes[0, 1].imshow(sem_vis)
    axes[0, 1].set_title("2. Semantic GT (Green: Person, Blue: Car)", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 3. Target Center Heatmap (Person)
    im3 = axes[0, 2].imshow(center_person, cmap='inferno')
    axes[0, 2].set_title("3. Target Person Center Heatmap", fontsize=12, fontweight='bold')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
    axes[0, 2].axis('off')
    
    # 4. Target Center Heatmap (Car)
    im4 = axes[1, 0].imshow(center_car, cmap='inferno')
    axes[1, 0].set_title("4. Target Car Center Heatmap", fontsize=12, fontweight='bold')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)
    axes[1, 0].axis('off')
    
    # 5. Offset Field Quiver Overlay
    offset_vis = visualize_offset_vectors(img_np, weight_np.astype(int), offset_np, step=16)
    axes[1, 1].imshow(offset_vis)
    axes[1, 1].set_title("5. 2D Offset Vector Field Target", fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # 6. Instance Weight Mask
    im6 = axes[1, 2].imshow(weight_np, cmap='gray')
    axes[1, 2].set_title("6. Foreground Instance Weight Mask", fontsize=12, fontweight='bold')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Data pipeline validation successful! Saved visualization to '{output_plot}'.")
    return True


if __name__ == '__main__':
    validate_data_pipeline()
