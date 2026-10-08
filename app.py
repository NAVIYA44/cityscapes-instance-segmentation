import os
import glob
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import streamlit as st

from src.model import ProposalFreeInstanceSegNet
from src.real_inference import predict_real_world
from src.instance_postproc import parse_instance_segmentation
from src.utils import (
    visualize_semantic_segmentation,
    visualize_instance_segmentation,
    visualize_offset_vectors
)

st.set_page_config(
    page_title="Proposal-Free Instance Segmentation (Cityscapes)",
    page_icon="🚗",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(checkpoint_path='checkpoints/best_model.pth', backbone_name='resnet18'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProposalFreeInstanceSegNet(backbone_name=backbone_name, pretrained=False).to(device)
    
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        st.sidebar.success(f"✓ Model Loaded (Epoch {ckpt.get('epoch', '?')})")
    else:
        st.sidebar.warning("⚠️ Using uninitialized model weights (Checkpoint not found)")
        
    model.eval()
    return model, device


def preprocess_image(pil_img, target_size=(256, 512)):
    img_resized = pil_img.resize((target_size[1], target_size[0]), resample=Image.BILINEAR)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img_resized).unsqueeze(0)
    return tensor, np.array(img_resized)


def main():
    st.markdown('<div class="main-header">Cityscapes Instance Segmentation App</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Proposal-Free Dense Instance Segmentation (Person + Car) via Center + Offset Regression</div>', unsafe_allow_html=True)

    # Sidebar Controls
    st.sidebar.header("⚙️ Model Engine & Post-Proc")
    
    engine_choice = st.sidebar.radio(
        "🧠 Inference Engine",
        ["Auto (Recommended)", "Real-World Photographic", "Synthetic Demo Checkpoint (Epoch 10)"],
        help="Real-World Photographic accurately segments uploaded real street photos. Synthetic Demo is for the 40 toy Cityscapes training images."
    )

    col_t1, col_t2 = st.sidebar.columns(2)
    with col_t1:
        person_thresh = st.slider("🧍 Person Thresh", min_value=0.20, max_value=0.70, value=0.40, step=0.05,
                                  help="Center peak threshold for Person.")
    with col_t2:
        car_thresh = st.slider("🚗 Car Thresh", min_value=0.30, max_value=0.90, value=0.50, step=0.05,
                               help="Center peak threshold for Car.")

    sem_thresh = st.sidebar.slider(
        "Semantic Confidence Threshold", min_value=0.20, max_value=0.85, value=0.30, step=0.05,
        help="Filters weak semantic predictions to prevent background noise."
    )
    max_instances = st.sidebar.slider(
        "Max Instances per Image (Top-K)", min_value=3, max_value=30, value=20, step=1,
        help="Caps candidate detections to avoid runaway clutter."
    )
    min_instance_pixels = st.sidebar.slider(
        "Min Instance Size (px)", min_value=20, max_value=200, value=40, step=10,
        help="Discards tiny pixel specks (Persons require ~40px, Cars ~80px)."
    )
    max_dist = st.sidebar.slider("Offset Max Assignment Dist (px)", min_value=32, max_value=192, value=96, step=16)
    alpha = st.sidebar.slider("Instance Mask Opacity", min_value=0.1, max_value=1.0, value=0.6, step=0.05)
    
    model, device = load_model()

    # Image Input Selection
    st.sidebar.header("🖼️ Input Image")
    input_mode = st.sidebar.radio("Select Input Mode", ["Sample Test Image", "Upload Custom Image"])

    image_to_process = None

    if input_mode == "Sample Test Image":
        samples = sorted(glob.glob("data/synthetic/images/val/*.png"))
        if samples:
            sample_choice = st.sidebar.selectbox("Choose Sample Image", samples)
            image_to_process = Image.open(sample_choice).convert("RGB")
        else:
            st.warning("No sample images found in data/synthetic/images/val! Upload a custom image.")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload a street image (PNG, JPG)", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            image_to_process = Image.open(uploaded_file).convert("RGB")

    if image_to_process is None:
        st.info("👈 Please select or upload a street image from the sidebar to begin instance segmentation.")
        return

    # Determine which engine to use
    use_real = (engine_choice == "Real-World Photographic") or (
        engine_choice == "Auto (Recommended)" and input_mode == "Upload Custom Image"
    )

    if use_real:
        st.sidebar.info("🚀 **Active:** Real-World Photographic Engine (COCO-Pretrained Backbone)")
    else:
        st.sidebar.info("🧪 **Active:** Synthetic Demo Checkpoint (Epoch 10)")

    # Preprocess & Run Inference
    img_tensor, img_np = preprocess_image(image_to_process)
    
    with st.spinner("Running Proposal-Free Instance Segmentation Inference..."):
        if use_real:
            preds = predict_real_world(img_tensor, device=device)
        else:
            with torch.no_grad():
                preds = model(img_tensor.to(device))
            
        inst_mask, sem_pred, inst_details = parse_instance_segmentation(
            preds['semantic_logits'], preds['center_heatmap'], preds['offset_map'],
            center_thresh={0: person_thresh, 1: car_thresh},
            sem_thresh=sem_thresh,
            max_dist=float(max_dist),
            min_instance_pixels=min_instance_pixels,
            top_k=max_instances
        )

    # Top Summary Metrics
    num_persons = sum(1 for inst in inst_details if inst['class_id'] == 1)
    num_cars = sum(1 for inst in inst_details if inst['class_id'] == 2)
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Instances", len(inst_details))
    with m2:
        st.metric("Person Instances 🧍", num_persons)
    with m3:
        st.metric("Car Instances 🚗", num_cars)
    with m4:
        st.metric("Image Resolution", f"{img_np.shape[1]}x{img_np.shape[0]}")

    st.markdown("---")

    # Tabs for Visual Output
    tab1, tab2, tab3 = st.tabs(["🎨 Distinct Instance Segmentation", "📊 Semantic & Center Heatmaps", "🧭 Center Offset Vector Field"])

    with tab1:
        st.subheader("Instance Segmentation (Distinct Color per Object Instance)")
        inst_vis = visualize_instance_segmentation(img_np, inst_mask, inst_details, alpha=alpha, draw_centers=True)
        st.image(inst_vis, use_container_width=True, caption="Instance-Separated Output with Centroid Keypoints & Bounding Boxes")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Semantic Segmentation Overlay")
            sem_vis = visualize_semantic_segmentation(img_np, sem_pred, alpha=0.5)
            st.image(sem_vis, use_container_width=True, caption="Green: Person | Blue: Car | Slate: Background")
            
        with col2:
            st.subheader("Center Heatmap Predictions")
            ctr_map = preds['center_heatmap'].squeeze(0).cpu().numpy()
            person_ctr = ctr_map[0]
            car_ctr = ctr_map[1]
            
            sub_col1, sub_col2 = st.subplots if hasattr(st, 'subplots') else (st.container(), st.container())
            st.image(person_ctr, use_container_width=True, caption=f"Person Center Heatmap (Max: {person_ctr.max():.2f})", clamp=True)
            st.image(car_ctr, use_container_width=True, caption=f"Car Center Heatmap (Max: {car_ctr.max():.2f})", clamp=True)

    with tab3:
        st.subheader("2D Spatial Offset Vectors (dx, dy)")
        st.caption("Vectors point from foreground pixels towards their estimated object centroid.")
        offset_map_np = preds['offset_map'].squeeze(0).cpu().numpy()
        offset_vis = visualize_offset_vectors(img_np, inst_mask, offset_map_np, step=16)
        st.image(offset_vis, use_container_width=True, caption="2D Spatial Offset Field Overlaid on Instances")

    # Detailed Instance Data Table
    st.markdown("### 📋 Detected Object Instance Details")
    if inst_details:
        table_data = []
        for inst in inst_details:
            table_data.append({
                "Instance ID": f"#{inst['instance_id']}",
                "Class": inst['class_name'],
                "Confidence Score": f"{inst.get('score', 1.0):.3f}",
                "Centroid (x, y)": f"({inst['center'][0]:.1f}, {inst['center'][1]:.1f})",
                "Bounding Box [Xmin, Ymin, Xmax, Ymax]": str(inst['bbox']),
                "Pixel Area": f"{inst['area']} px"
            })
        st.dataframe(table_data, use_container_width=True)
    else:
        st.write("No instances detected at current confidence threshold.")


if __name__ == '__main__':
    main()
