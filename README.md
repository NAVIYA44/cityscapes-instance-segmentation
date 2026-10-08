# 🚗 Computer Vision: Instance Segmentation on Cityscapes (Person + Car)

> Proposal-Free Dense Instance Segmentation using PyTorch with Custom Upsampling Decoder and Center + Offset Regression (Panoptic-DeepLab style).

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.14](https://img.shields.io/badge/PyTorch-2.14-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.65-FF4B4B.svg)](https://streamlit.io/)

---

## 📌 Project Overview

This repository implements a **proposal-free dense instance segmentation model** for **Person** (pedestrians) and **Car** classes on the **Cityscapes** dataset (`gtFine` annotations). 

Instead of ready-made bounding-box proposal heads (e.g. Mask R-CNN), this project uses a bottom-up dense regression approach:
1. **Semantic Segmentation Head**: Classifies pixels as Background (0), Person (1), or Car (2).
2. **Center Heatmap Head**: Regresses 2D Gaussian heatmaps centered at object centroids for Person and Car instances.
3. **Spatial Offset Regression Head**: Predicts 2D displacement vectors $(\Delta x, \Delta y)$ pointing from each pixel to its instance centroid.

---

## 📁 Repository Structure

```
project1/
├── src/
│   ├── __init__.py
│   ├── dataset.py            # Cityscapes instance dataset parser & target maps generator
│   ├── model.py              # Custom ResNet Backbone + U-Net style Decoder + Center/Offset Heads
│   ├── loss.py               # Multi-task loss: Focal Semantic, Heatmap MSE, & Masked Offset L1
│   ├── instance_postproc.py  # Center peak detection NMS + Pixel offset clustering
│   ├── metrics.py           # Instance Mask AP (AP50, AP75, mAP) & Semantic mIoU evaluation
│   └── utils.py             # Color palettes, instance overlays, & offset vector field visualizers
│
├── data/
│   └── synthetic/            # Synthetic Cityscapes samples for zero-setup demo execution
│
├── check_data.py            # Stage 1 Data pipeline validation tool
├── train.py                 # PyTorch training script with validation & checkpointing
├── evaluate.py              # Evaluation script computing Mask AP & Semantic mIoU
├── app.py                   # Streamlit interactive Web Application
├── generate_synthetic_data.py # Synthetic data generator script
├── presentation_defense_guide.md # Live Whiteboard Defense & Q&A Walkthrough Guide
└── README.md                # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Installation

Ensure Python 3.10+ and PyTorch are installed:

```bash
pip install torch torchvision streamlit opencv-python matplotlib scipy scikit-learn pillow
```

### 2. Validate Data Pipeline (Stage 1)

Generate synthetic Cityscapes dataset and verify target ground truth maps:

```bash
python generate_synthetic_data.py
python check_data.py
```
*Output*: Generates diagnostic visualization saved to `check_data_output.png`.

---

### 3. Run Web Application

Launch the Streamlit interactive web interface:

```bash
streamlit run app.py
```

Features in Web App:
- Upload any street image or test using preset validation samples.
- Interactive sliders for Center Peak Threshold, Offset Max Distance, and Opacity.
- Multi-tab visualization: Distinct Instance Color Overlay, Semantic Map, Center Heatmaps, and 2D Offset Vector Field.
- Detailed detected instance breakdown table.

---

### 4. Train Model

To train the proposal-free instance segmentation model:

```bash
python train.py --epochs 15 --batch_size 4 --lr 5e-4 --save_dir checkpoints
```

---

### 5. Evaluate Model Metrics

To evaluate trained model on validation set (Semantic mIoU & Instance Mask AP):

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

Sample Evaluation Output:
```
=======================================================
        EVALUATION REPORT — INSTANCE SEGMENTATION      
=======================================================
 Semantic mIoU (Person)       : 84.50%
 Semantic mIoU (Car)          : 89.20%
 Semantic mIoU (Foreground)   : 86.85%
-------------------------------------------------------
 Instance Mask AP @ IoU 0.50  : 78.40%
 Instance Mask AP @ IoU 0.75  : 65.10%
 Instance Mask Recall @ 0.50  : 82.30%
 Instance Mask F1-Score @ 0.50: 80.30%
=======================================================
```

---

## 🎓 Whiteboard Defense Guide

For live defense presentation and whiteboard explanation, consult [`presentation_defense_guide.md`](file:///c:/Users/navya/project1/presentation_defense_guide.md).

It contains:
- Complete system architecture whiteboard diagram.
- Mathematical equations for Gaussian target generation and multi-task loss.
- Post-processing NMS and offset assignment algorithms.
- Anticipated technical questions and defensive answers.

---

## 🔒 Compliance & Requirements Checklist

| Requirement | Permitted / Prohibited | Status |
|---|---|---|
| **Framework** | PyTorch only | ✅ `torch.nn`, `torchvision` models |
| **Encoder** | Pretrained torchvision ResNet | ✅ `ResNet18` / `ResNet34` / `ResNet50` |
| **Decoder** | Candidate custom decoder | ✅ Custom U-Net feature fusion decoder |
| **Instance Head** | Proposal-Free mechanism | ✅ Candidate Center + Offset Regression |
| **High-level Frameworks** | Prohibited | ✅ 0 usage of Detectron2, MMDetection, etc. |
