# 🎓 Live Whiteboard Defense & Presentation Guide
## Computer Vision: Proposal-Free Instance Segmentation on Cityscapes (Person + Car)

> **Core Focus**: Custom Decoder + Proposal-Free Dense Instance Separation (Center + Offset Regression)

---

## 1. 📐 Architectural Whiteboard Diagram

During your live defense, draw this diagram step-by-step on the whiteboard:

```
[ Input Street Image ] (B, 3, H, W)
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│              Pretrained Encoder Backbone               │ (torchvision ResNet18/34/50)
│   C1 (H/2, 64) ──► C2 (H/4, 64) ──► C3 (H/8, 128)      │
│                         │                │              │
│                         ▼                ▼              │
│                   C4 (H/16, 256) ──► C5 (H/32, 512)     │
└─────────────────────────┬───────────────────────────────┘
                          │ (Skip Connections)
                          ▼
┌─────────────────────────────────────────────────────────┐
│            Custom Feature Fusion Decoder               │ (torch.nn candidate design)
│   D4 (H/16, 128) ◄── D3 (H/8, 128) ◄── D2 (H/4, 128)    │
│                         │                               │
│                         ▼                               │
│                 Final Feature Map F (B, 64, H, W)       │
└─────────────────────────┬───────────────────────────────┘
                          │
       ┌──────────────────┼──────────────────┐
       ▼                  ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Semantic Head│   │ Center Head  │   │ Offset Head  │
│  (3, H, W)   │   │  (2, H, W)   │   │  (2, H, W)   │
│ Logits: BG,  │   │ Sigmoid:     │   │ 2D Vectors:  │
│ Person, Car  │   │ Person/Car   │   │  (dx, dy)    │
└──────────────┘   └──────────────┘   └──────────────┘
       │                  │                  │
       └──────────────────┼──────────────────┘
                          ▼
        ┌──────────────────────────────────┐
        │ Center Peak NMS & Pixel-Centroid │
        │        Offset Clustering         │
        └──────────────────────────────────┘
                          │
                          ▼
      [ Distinct Instance Segmentation Output ]
```

---

## 2. 🧮 Mathematical Formulation & Losses

### A. Ground Truth Target Generation
1. **Cityscapes Instance ID Encoding**:
   $$\text{instance\_id} = \text{class\_id} \times 1000 + \text{instance\_index}$$
   - Person ($\text{class\_id} = 24$) $\rightarrow$ target class 1.
   - Car ($\text{class\_id} = 26$) $\rightarrow$ target class 2.
   - Ignore region ($\text{instance\_index} = 0$ / crowd groups) $\rightarrow$ mapped to `ignore_index = 255`.

2. **Center Heatmap Target** $S_{c, y, x}$:
   For each object instance $k$ of class $c$ with centroid $(c_x^{(k)}, c_y^{(k)})$:
   $$S_{c, y, x} = \max_k \exp \left( -\frac{(x - c_x^{(k)})^2 + (y - c_y^{(k)})^2}{2 \sigma^2} \right)$$
   where $\sigma = 8.0$ controls the Gaussian radius.

3. **Center Spatial Offset Target** $O_{y, x} = (\Delta x, \Delta y)$:
   For pixel $(x, y)$ inside instance $k$:
   $$\Delta x(x, y) = c_x^{(k)} - x, \quad \Delta y(x, y) = c_y^{(k)} - y$$

---

### B. Loss Functions

$$\mathcal{L}_{\text{total}} = \lambda_{\text{sem}} \mathcal{L}_{\text{sem}} + \lambda_{\text{center}} \mathcal{L}_{\text{center}} + \lambda_{\text{offset}} \mathcal{L}_{\text{offset}}$$

1. **Semantic Focal Loss** ($\mathcal{L}_{\text{sem}}$):
   $$\mathcal{L}_{\text{sem}} = -\frac{1}{N} \sum_{i, y, x} (1 - p_{i, y, x})^\gamma \log(p_{i, y, x})$$
   (Evaluated on valid pixels where target $\neq 255$).

2. **Center Heatmap MSE Loss** ($\mathcal{L}_{\text{center}}$):
   $$\mathcal{L}_{\text{center}} = \frac{1}{2 \cdot H \cdot W} \sum_{c=1}^2 \sum_{y=1}^H \sum_{x=1}^W \left( \hat{S}_{c, y, x} - S_{c, y, x} \right)^2$$

3. **Masked L1 Offset Loss** ($\mathcal{L}_{\text{offset}}$):
   $$\mathcal{L}_{\text{offset}} = \frac{1}{\sum w_{y,x}} \sum_{y, x} w_{y, x} \cdot \left( \left| \hat{\Delta x}_{y,x} - \Delta x_{y,x} \right| + \left| \hat{\Delta y}_{y,x} - \Delta y_{y,x} \right| \right)$$
   where $w_{y,x} = 1$ for foreground instance pixels, 0 for background.

---

## 3. 🔍 Proposal-Free Inference & Instance Clustering

How pixels are grouped into distinct individual instances:

1. **Center Keypoint NMS**:
   - Apply $5 \times 5$ max pooling on predicted heatmap $\hat{S}_c$.
   - Extract local maxima where $\hat{S}_c(x, y) == \text{MaxPool}(\hat{S}_c)(x, y)$ and $\hat{S}_c(x, y) \ge T_{\text{center}}$.
   - Obtain peak centroid coordinates $\{(c_x^{(k)}, c_y^{(k)})\}_{k=1}^K$.

2. **Pixel-to-Centroid Assignment**:
   - For pixel $(x, y)$ predicted as foreground class $c$:
     $$\text{Predicted Centroid} = (x + \hat{\Delta x}(x, y), y + \hat{\Delta y}(x, y))$$
   - Calculate Euclidean distance to all detected peaks of class $c$:
     $$d_k = \sqrt{(x + \hat{\Delta x} - c_x^{(k)})^2 + (y + \hat{\Delta y} - c_y^{(k)})^2}$$
   - Assign pixel to instance $\arg\min_k d_k$ if $\min d_k \le D_{\text{max}}$.

---

## 4. 💡 Potential Defense Questions & Model Answers

### Q1: Why choose proposal-free Center + Offset over Mask R-CNN or Pixel Embedding?
**Answer**: 
- **Over Mask R-CNN**: Mask R-CNN relies on anchor boxes and Region Proposal Networks (RPN) which struggle with dense, non-axis-aligned, or highly overlapping pedestrians. Proposal-free methods process the image in a single bottom-up dense pass ($O(HW)$), avoiding thousands of candidate bounding boxes.
- **Over Pixel Embedding**: Center + Offset regression uses fixed 2D vector offset targets, making loss optimization deterministic and stable with standard Smooth $L_1$ loss. Embedding methods require distance-based margin losses (discriminative loss) and expensive iterative clustering like DBSCAN at inference.

### Q2: How does your model handle small pedestrians or crowded overlapping vehicles?
**Answer**:
- Center heatmaps pinpoint exact object centers regardless of overlap.
- Offset vectors allow overlapping object pixels to point to completely different centroids even if they are adjacent in image space.
- Higher resolution decoder skip connections retain spatial granularity for small distant pedestrians.

### Q3: What is the accuracy vs. compute trade-off of your resolution choice?
**Answer**:
- Cityscapes native resolution is $2048 \times 1024$ (2.1 Mpixels), which requires high VRAM and compute per batch.
- Resizing to $512 \times 256$ (0.13 Mpixels) reduces GFLOPs by $16\times$, allowing fast training and real-time 30+ FPS inference on 8GB VRAM while preserving key structural features of vehicles and pedestrians.

---

## 5. 🛠️ Prohibited vs. Permitted Compliance Audit

| Required / Constraint | Status | Implementation Detail |
|---|---|---|
| PyTorch only | ✅ Compliant | Built strictly with `torch`, `torch.nn`, `torchvision.models` backbone |
| Allowed Encoder | ✅ Compliant | `torchvision.models.resnet18` with ImageNet pretrained weights |
| Custom Decoder | ✅ Compliant | Custom implemented `CustomDecoderBlock` with skip fusion |
| Proposal-free head | ✅ Compliant | Center Heatmap + 2D Spatial Offset Regression |
| No high-level frameworks | ✅ Compliant | Zero usage of Detectron2, MMDetection, PyTorch Lightning, fastai |
