import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import torch

# Distinct color palette for semantic classes
SEMANTIC_COLOR_MAP = {
    0: (30, 30, 40),       # Background (Dark Slate)
    1: (46, 204, 113),     # Person (Emerald Green)
    2: (52, 152, 219),     # Car (Royal Blue)
    255: (100, 100, 100)   # Ignore
}


def generate_distinct_colors(num_colors=50, seed=42):
    """Generates a list of distinct RGB color tuples for instance visualization."""
    np.random.seed(seed)
    colors = []
    for _ in range(num_colors):
        # Generate vibrant HSL/RGB colors
        color = tuple(np.random.randint(50, 255, size=3).tolist())
        colors.append(color)
    return colors


INSTANCE_COLORS = generate_distinct_colors(100)


def visualize_semantic_segmentation(image_np, sem_pred, alpha=0.5):
    """
    Overlays semantic predictions on input RGB image.
    
    Args:
        image_np: numpy array (H, W, 3) uint8 image
        sem_pred: numpy array (H, W) class map (0: BG, 1: Person, 2: Car)
        alpha: overlay blending factor
    """
    H, W, _ = image_np.shape
    sem_color_img = np.zeros((H, W, 3), dtype=np.uint8)
    
    for cls_id, color in SEMANTIC_COLOR_MAP.items():
        if cls_id == 255:
            continue
        sem_color_img[sem_pred == cls_id] = color
        
    blended = (image_np.astype(np.float32) * (1 - alpha) + sem_color_img.astype(np.float32) * alpha).astype(np.uint8)
    return blended


def visualize_instance_segmentation(image_np, instance_mask, instance_details, alpha=0.6, draw_centers=True):
    """
    Overlays instance segmentation masks with distinct colors per instance object.
    
    Args:
        image_np: numpy array (H, W, 3) uint8 image
        instance_mask: numpy array (H, W) integer instance ID map
        instance_details: list of dicts with instance metadata
        alpha: mask blending opacity
        draw_centers: if True, draws center keypoints and bounding boxes
    """
    H, W, _ = image_np.shape
    overlay = image_np.copy()
    
    # Create mask canvas
    mask_canvas = np.zeros((H, W, 3), dtype=np.uint8)
    
    for inst in instance_details:
        inst_id = inst['instance_id']
        color = INSTANCE_COLORS[inst_id % len(INSTANCE_COLORS)]
        
        mask = (instance_mask == inst_id)
        mask_canvas[mask] = color
        
    # Blend image with instance masks
    has_instance = (instance_mask > 0)
    blended = image_np.copy()
    blended[has_instance] = (image_np[has_instance].astype(np.float32) * (1 - alpha) + 
                             mask_canvas[has_instance].astype(np.float32) * alpha).astype(np.uint8)
    
    # Convert to PIL for crisp text and marker drawing
    pil_img = Image.fromarray(blended)
    draw = ImageDraw.Draw(pil_img)
    
    if draw_centers:
        for inst in instance_details:
            inst_id = inst['instance_id']
            cls_name = inst['class_name']
            bbox = inst['bbox']  # [xmin, ymin, xmax, ymax]
            cx, cy = inst['center']
            score = inst.get('score', 1.0)
            
            color = INSTANCE_COLORS[inst_id % len(INSTANCE_COLORS)]
            
            # Draw Bounding Box
            draw.rectangle(bbox, outline=color, width=2)
            
            # Draw Center Keypoint Marker
            r = 4
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 0), outline=(0, 0, 0), width=1)
            
            # Draw Label Tag (e.g. "Car (1.00)", "Person (1.00)")
            label_text = f"{cls_name} ({score:.2f})"
            draw.rectangle([bbox[0], max(0, bbox[1] - 16), bbox[0] + len(label_text) * 8 + 4, bbox[1]], fill=color)
            draw.text((bbox[0] + 4, max(0, bbox[1] - 14)), label_text, fill=(255, 255, 255))
            
    return np.array(pil_img)


def visualize_offset_vectors(image_np, instance_mask, offset_map, step=16):
    """
    Visualizes the predicted 2D offset vector field pointing to instance centers.
    Great for whiteboard defense & visual understanding!
    """
    H, W, _ = image_np.shape
    vis_img = image_np.copy()
    
    dx = offset_map[0] if isinstance(offset_map, np.ndarray) else offset_map[0].cpu().numpy()
    dy = offset_map[1] if isinstance(offset_map, np.ndarray) else offset_map[1].cpu().numpy()
    
    # Draw vector arrows at grid intervals
    for y in range(step // 2, H, step):
        for x in range(step // 2, W, step):
            if instance_mask[y, x] > 0:
                end_x = int(round(x + dx[y, x]))
                end_y = int(round(y + dy[y, x]))
                
                # Draw line towards center
                cv2.arrowedLine(vis_img, (x, y), (end_x, end_y), (0, 255, 255), 1, tipLength=0.2)
                cv2.circle(vis_img, (x, y), 2, (255, 0, 0), -1)
                
    return vis_img
