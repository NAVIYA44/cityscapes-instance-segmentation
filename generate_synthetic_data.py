import os
import numpy as np
from PIL import Image, ImageDraw


def generate_synthetic_cityscapes_sample(width=512, height=256, num_cars=4, num_persons=5, seed=None):
    """
    Generates a synthetic street scene image and matching Cityscapes gtFine instanceId map.
    
    Cityscapes instance ID encoding:
    instance_id = class_id * 1000 + instance_index
    Person class ID: 24 -> instance IDs: 24001, 24002...
    Car class ID: 26 -> instance IDs: 26001, 26002...
    """
    if seed is not None:
        np.random.seed(seed)
        
    # 1. Base Image Canvas
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    
    # Sky (Top 40%)
    sky_color = (135 + np.random.randint(-10, 10), 180 + np.random.randint(-10, 10), 222 + np.random.randint(-10, 10))
    draw.rectangle([0, 0, width, int(height * 0.4)], fill=sky_color)
    
    # Buildings/Trees background (Middle 20%)
    bg_color = (80 + np.random.randint(-10, 10), 90 + np.random.randint(-10, 10), 95 + np.random.randint(-10, 10))
    draw.rectangle([0, int(height * 0.4), width, int(height * 0.55)], fill=bg_color)
    
    # Sidewalk & Road (Bottom 45%)
    road_color = (60 + np.random.randint(-5, 5), 62 + np.random.randint(-5, 5), 65 + np.random.randint(-5, 5))
    draw.rectangle([0, int(height * 0.55), width, height], fill=road_color)
    
    # Road lane lines
    for lane_x in range(40, width, 80):
        draw.rectangle([lane_x, int(height * 0.75), lane_x + 30, int(height * 0.77)], fill=(220, 220, 220))
        
    # 2. Instance ID Map Canvas (int32)
    inst_map = np.zeros((height, width), dtype=np.int32)
    inst_pil = Image.fromarray(inst_map)
    inst_draw = ImageDraw.Draw(inst_pil)
    
    # 3. Add Cars (Class 26)
    car_inst_count = 1
    for i in range(num_cars):
        w_car = np.random.randint(60, 110)
        h_car = int(w_car * 0.5)
        x1 = np.random.randint(10, width - w_car - 10)
        y1 = np.random.randint(int(height * 0.55), height - h_car - 10)
        x2, y2 = x1 + w_car, y1 + h_car
        
        encoded_inst_id = 26 * 1000 + car_inst_count
        car_inst_count += 1
        
        # Draw Car on Image (Body + Roof + Wheels)
        car_color = (np.random.randint(150, 240), np.random.randint(30, 100), np.random.randint(30, 100))
        draw.rectangle([x1, y1 + int(h_car*0.4), x2, y2], fill=car_color)
        draw.rectangle([x1 + int(w_car*0.2), y1, x2 - int(w_car*0.2), y1 + int(h_car*0.5)], fill=(40, 40, 50))
        # Wheels
        draw.ellipse([x1 + 5, y2 - 8, x1 + 20, y2 + 5], fill=(10, 10, 10))
        draw.ellipse([x2 - 20, y2 - 8, x2 - 5, y2 + 5], fill=(10, 10, 10))
        
        # Draw on Instance Map
        inst_draw.rectangle([x1, y1, x2, y2], fill=encoded_inst_id)
        
    # 4. Add Persons / Pedestrians (Class 24)
    person_inst_count = 1
    for i in range(num_persons):
        h_person = np.random.randint(35, 65)
        w_person = int(h_person * 0.38)
        x1 = np.random.randint(10, width - w_person - 10)
        y1 = np.random.randint(int(height * 0.48), height - h_person - 5)
        x2, y2 = x1 + w_person, y1 + h_person
        
        encoded_inst_id = 24 * 1000 + person_inst_count
        person_inst_count += 1
        
        # Draw Person on Image (Head + Body + Legs)
        shirt_color = (np.random.randint(20, 220), np.random.randint(150, 240), np.random.randint(100, 220))
        pants_color = (30, 40, 80)
        
        # Head
        draw.ellipse([x1 + int(w_person*0.2), y1, x2 - int(w_person*0.2), y1 + int(h_person*0.25)], fill=(230, 190, 160))
        # Torso
        draw.rectangle([x1, y1 + int(h_person*0.25), x2, y1 + int(h_person*0.65)], fill=shirt_color)
        # Legs
        draw.rectangle([x1 + 1, y1 + int(h_person*0.65), x2 - 1, y2], fill=pants_color)
        
        # Draw on Instance Map
        inst_draw.rectangle([x1, y1, x2, y2], fill=encoded_inst_id)
        
    inst_map = np.array(inst_pil, dtype=np.int32)
    return img, inst_map


def build_synthetic_dataset(output_dir='data/synthetic', train_samples=40, val_samples=10):
    """Populates train and val dataset directories with synthetic Cityscapes image/instance pairs."""
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'instances', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'instances', 'val'), exist_ok=True)
    
    print(f"Generating {train_samples} synthetic training samples...")
    for i in range(train_samples):
        img, inst = generate_synthetic_cityscapes_sample(seed=100 + i)
        fname = f"synth_city_{i:04d}"
        img.save(os.path.join(output_dir, 'images', 'train', f"{fname}_leftImg8bit.png"))
        Image.fromarray(inst.astype(np.uint32)).save(os.path.join(output_dir, 'instances', 'train', f"{fname}_gtFine_instanceIds.png"))
        
    print(f"Generating {val_samples} synthetic validation samples...")
    for i in range(val_samples):
        img, inst = generate_synthetic_cityscapes_sample(seed=500 + i)
        fname = f"synth_city_val_{i:04d}"
        img.save(os.path.join(output_dir, 'images', 'val', f"{fname}_leftImg8bit.png"))
        Image.fromarray(inst.astype(np.uint32)).save(os.path.join(output_dir, 'instances', 'val', f"{fname}_gtFine_instanceIds.png"))
        
    print(f"Successfully generated synthetic Cityscapes dataset in '{output_dir}'!")


if __name__ == '__main__':
    build_synthetic_dataset()
