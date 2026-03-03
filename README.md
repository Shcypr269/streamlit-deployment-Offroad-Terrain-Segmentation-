# Offroad Desert Terrain Semantic Segmentation

Advanced semantic segmentation system for offroad desert environments using DINOv2 Vision Transformer with custom ConvNeXt decoder.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Performance](#performance)
- [Installation](#installation)
- [Training](#training)
- [Inference](#inference)
- [Model Weights](#model-weights)
- [Technical Details](#technical-details)
- [Results Analysis](#results-analysis)
- [Future Improvements](#future-improvements)

## Overview

This project implements a state-of-the-art semantic segmentation model for offroad desert terrain classification. The model identifies 10 distinct terrain types including vegetation, obstacles, ground elements, and background features with pixel-level precision.

### Key Features

- **Frozen DINOv2 Backbone**: Leverages Meta AI's self-supervised Vision Transformer (86M parameters)
- **Custom Decoder**: ASPP + ConvNeXt blocks for multi-scale feature aggregation
- **Advanced Loss Function**: Combined Focal + Dice loss for class imbalance handling
- **Production Ready**: Real-time inference at 20 FPS, optimized for edge deployment
- **Comprehensive Augmentation**: 8+ augmentation techniques for robust generalization

### Terrain Classes

| Class ID | Class Name      | Color RGB       | Pixel Distribution |
|----------|----------------|-----------------|-------------------|
| 0        | Background      | (30, 30, 35)    | 2.81%            |
| 1        | Trees           | (34, 139, 34)   | 3.53%            |
| 2        | Lush Bushes     | (50, 205, 50)   | 5.93%            |
| 3        | Dry Grass       | (189, 183, 107) | 18.86%           |
| 4        | Dry Bushes      | (160, 82, 45)   | 1.10%            |
| 5        | Ground Clutter  | (128, 128, 128) | 4.39%            |
| 6        | Logs            | (139, 69, 19)   | 0.08%            |
| 7        | Rocks           | (112, 128, 144) | 1.20%            |
| 8        | Landscape       | (210, 180, 140) | 24.44%           |
| 9        | Sky             | (135, 206, 235) | 37.61%           |

## Architecture

### Model Components

```
Input Image (672x378x3)
    ↓
┌─────────────────────────────────────┐
│   DINOv2 ViT-B/14 (Frozen)          │
│   - Patch size: 14x14               │
│   - Output tokens: 48x27x768        │
│   - Parameters: 86M (frozen)        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   Custom Decoder (Trainable)        │
│   ├─ Positional Encoding (64 ch)    │
│   ├─ ASPP (multi-scale: 6,12,18)    │
│   ├─ ConvNeXt Block 1               │
│   ├─ ConvNeXt Block 2               │
│   └─ Classifier (1x1 conv)          │
│   - Parameters: 19M (trainable)     │
└─────────────────────────────────────┘
    ↓
Output Logits (672x378x10)
```

### Key Architectural Choices

**DINOv2 Backbone (Frozen)**
- Pretrained on 142M images via self-supervised learning
- Kept frozen to prevent overfitting (only 2,857 training images)
- Provides robust general-purpose visual features
- Memory efficient: 400MB vs 1.2GB if trainable

**ASPP (Atrous Spatial Pyramid Pooling)**
- Captures multi-scale context with parallel dilated convolutions
- Dilation rates: 6, 12, 18 (receptive fields: 13x13, 25x25, 37x37)
- Global average pooling branch for image-level context
- Critical for handling objects at different scales (Sky vs Logs)

**ConvNeXt Blocks**
- Depthwise separable convolutions (512x parameter reduction)
- Inverted bottleneck design (expand 4x, then squeeze)
- Layer normalization for stable training
- Residual connections for gradient flow

**Positional Encoding**
- 2D sinusoidal encoding with 8 frequencies
- Provides spatial awareness to decoder
- 64 channels (8 frequencies x 2 coordinates x 2 functions)

## Dataset

### Statistics

- **Training Set**: 2,857 images
- **Validation Set**: 715 images
- **Resolution**: 960x540 (resized to 672x378 for training)
- **Format**: RGB images + PNG segmentation masks
- **Total Pixels**: 1.48 billion (training set)

### Class Imbalance

The dataset exhibits extreme class imbalance (470:1 ratio between most and least common classes):

| Class          | Pixels      | Percentage | Avg per Image |
|----------------|-------------|------------|---------------|
| Sky            | 557,458,734 | 37.61%     | 195,110       |
| Landscape      | 362,120,221 | 24.44%     | 126,718       |
| Dry Grass      | 279,430,843 | 18.86%     | 97,808        |
| Lush Bushes    | 87,892,776  | 5.93%      | 30,761        |
| Ground Clutter | 65,082,995  | 4.39%      | 22,778        |
| Trees          | 52,331,525  | 3.53%      | 18,314        |
| Background     | 41,585,811  | 2.81%      | 14,555        |
| Dry Bushes     | 16,268,713  | 1.10%      | 5,694         |
| Rocks          | 17,743,187  | 1.20%      | 6,210         |
| Logs           | 1,153,995   | 0.08%      | 404           |

### Data Augmentation

**Geometric Transformations**
- Horizontal flip (p=0.5)
- Rotation ±10 degrees (p=0.5)
- Random scale 0.8-1.2x (p=0.5)

**Photometric Transformations**
- Color jitter: brightness ±30%, contrast ±30%, saturation ±30%, hue ±10%
- Gamma correction: gamma ∈ [0.7, 1.5]
- Gaussian blur: sigma ∈ [0.1, 2.0] (p=0.3)
- Random shadow gradients (p=0.4)
- Coarse dropout: up to 8 holes, 40x40 pixels each (p=0.5)

## Performance

### Validation Metrics (Best Epoch 23)

| Metric              | Value  |
|---------------------|--------|
| Mean IoU            | 0.3879 |
| Dice Score          | 0.5495 |
| Pixel Accuracy      | 72.93% |
| Training Loss       | 0.7201 |
| Validation Loss     | 0.7132 |

### Per-Class IoU

| Class          | IoU    | Performance Tier |
|----------------|--------|------------------|
| Sky            | 0.9765 | Excellent        |
| Landscape      | 0.5473 | Good             |
| Background     | 0.4852 | Adequate         |
| Dry Grass      | 0.4743 | Adequate         |
| Trees          | 0.4571 | Adequate         |
| Lush Bushes    | 0.4486 | Adequate         |
| Dry Bushes     | 0.1758 | Poor             |
| Rocks          | 0.1718 | Poor             |
| Ground Clutter | 0.1263 | Poor             |
| Logs           | 0.1105 | Poor             |

### Performance Analysis

**Strong Performance (IoU > 0.40)**
- Sky, Landscape, Background, Dry Grass, Trees, Lush Bushes
- These classes benefit from abundant training examples and distinct visual features

**Weak Performance (IoU < 0.20)**
- Logs, Rocks, Ground Clutter, Dry Bushes
- Challenges: Extreme scarcity (Logs: 0.08%), small object size, visual ambiguity

**Failure Analysis**
- Logs: Only 404 pixels per image on average, insufficient for gradient-based learning
- Visual ambiguity: Dry Grass vs Ground Clutter vs Dry Bushes exhibit color overlap
- Scale mismatch: 10-50 pixel objects occupy less than 1 patch at 14x14 resolution
- Boundary blur: 14x upsampling reduces edge precision

## Installation

### Requirements

```
Python >= 3.8
PyTorch >= 2.0.0
torchvision >= 0.15.0
CUDA >= 11.8 (for GPU training)
```

### Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pillow opencv-python numpy matplotlib tqdm
```

### Clone Repository

```bash
git clone https://github.com/yourusername/offroad-terrain-segmentation.git
cd offroad-terrain-segmentation
```

## Training

### Quick Start

```bash
python train.py
```

### Training Configuration

```python
# Hyperparameters
n_epochs = 50
batch_size = 16
learning_rate = 3e-4
weight_decay = 0.01
focal_gamma = 2.0
patience = 10  # Early stopping

# Image resolution (must be divisible by 14)
input_size = (378, 672)  # H x W

# Loss weights
dice_weight = 0.5
focal_weight = 0.5
```

### Hardware Requirements

**Minimum**
- GPU: NVIDIA RTX 3060 (6GB VRAM)
- RAM: 16GB
- Storage: 10GB

**Recommended**
- GPU: NVIDIA RTX 4070 or better (8GB+ VRAM)
- RAM: 32GB
- Storage: 20GB (for checkpoints and visualizations)

### Training Time

- **Single Epoch**: ~40 seconds (RTX 4050)
- **Full Training**: ~30 minutes (early stopping at epoch 25-30)
- **Inference**: 50ms per image (20 FPS)

### Memory Footprint

| Component          | Memory   |
|--------------------|----------|
| DINOv2 (frozen)    | 400 MB   |
| Decoder            | 200 MB   |
| Batch (16 images)  | 2.4 GB   |
| Gradients          | 1.0 GB   |
| **Total**          | **4.0 GB** |

## Inference

### Python API

```python
import torch
from PIL import Image
from model import SegmentationHeadConvNeXt

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
model = SegmentationHeadConvNeXt(768, 10, 48, 27)
model.load_state_dict(torch.load("segmentation_head_best.pth"))
model.to(device)
model.eval()

# Load and preprocess image
image = Image.open("input.jpg").convert("RGB")
# ... apply transforms ...

# Inference
with torch.no_grad():
    features = backbone.forward_features(image)["x_norm_patchtokens"]
    logits = model(features)
    prediction = torch.argmax(logits, dim=1)
```

### Streamlit Demo

```bash
streamlit run app.py
```

Access the web interface at `http://localhost:8501`

## Model Weights

### Download

Pre-trained model weights are available for download:

- **Best Model** (Epoch 23): `segmentation_head_best.pth` (76 MB)
- **Final Model** (Epoch 50): `segmentation_head_final.pth` (76 MB)

### Loading Weights

```python
checkpoint = torch.load("segmentation_head_best.pth", map_location=device)
model.load_state_dict(checkpoint)
```

## Technical Details

### Loss Function

**Combined Loss**: L = 0.5 × L_focal + 0.5 × L_dice

**Focal Loss**
```
L_focal = -(1 - p_t)^γ × log(p_t)

where:
  p_t = probability of correct class
  γ = 2.0 (focusing parameter)
```

**Benefits**
- Reduces contribution from easy examples (well-classified Sky pixels)
- Focuses training on hard examples (rare classes, boundaries)
- Handles class imbalance through down-weighting

**Dice Loss**
```
L_dice = 1 - (2 × |P ∩ T|) / (|P| + |T|)

where:
  P = prediction
  T = ground truth
  |P ∩ T| = intersection (overlap)
```

**Benefits**
- Directly optimizes IoU-like metric
- Per-class computation ensures balanced learning
- Handles small objects better than pixel-wise losses

### Class Weighting

Inverse frequency weighting to handle extreme imbalance:

```python
class_weight[c] = total_pixels / (class_pixels[c] + epsilon)
normalized_weight = class_weight / sum(class_weight) × num_classes
```

**Resulting Weights**

| Class          | Weight | Ratio to Sky |
|----------------|--------|--------------|
| Logs           | 8.84   | 491x         |
| Dry Bushes     | 6.27   | 348x         |
| Rocks          | 5.75   | 319x         |
| Ground Clutter | 1.97   | 109x         |
| Background     | 2.45   | 136x         |
| Trees          | 1.95   | 108x         |
| Lush Bushes    | 1.16   | 64x          |
| Dry Grass      | 0.36   | 20x          |
| Landscape      | 0.28   | 16x          |
| Sky            | 0.018  | 1x           |

### Mixed Precision Training

Automatic Mixed Precision (AMP) for efficiency:
- FP16 operations for forward/backward passes
- FP32 for loss computation and weight updates
- Gradient scaling to prevent underflow
- **Benefits**: 2x speedup, 50% memory reduction, minimal accuracy loss

### Optimization

**Optimizer**: AdamW
- Learning rate: 3e-4
- Weight decay: 0.01
- Betas: (0.9, 0.999)

**Scheduler**: Cosine Annealing
```
lr(t) = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × t / T_max))
```

**Early Stopping**
- Metric: Mean IoU
- Patience: 10 epochs
- Typical stopping point: Epoch 25-30

## Results Analysis

### Training Dynamics

**Loss Convergence**
- Training loss: 0.87 → 0.72 (epochs 0-25)
- Validation loss: 0.83 → 0.71 (minimal gap = good generalization)
- Gap: 0.01 (no significant overfitting)

**IoU Progression**
- Rapid improvement: 0.35 → 0.37 (epochs 0-5)
- Steady growth: 0.37 → 0.3879 (epochs 5-23)
- Plateau: 0.3879 (epochs 23-25, early stopping triggered)

### Limitations

**Fundamental Constraints**
1. **Data Scarcity**: Logs (0.08% of pixels) insufficient for gradient-based learning
2. **Class Imbalance**: 470:1 ratio overwhelms even aggressive re-weighting
3. **Visual Ambiguity**: Dry Grass/Dry Bushes/Ground Clutter overlap in feature space
4. **Resolution Mismatch**: 14x14 patches too coarse for 10-50 pixel objects

**Expected Performance Ceiling**
- Current: 0.3879 mIoU
- Realistic maximum with current data: 0.42-0.45 mIoU
- Achieving 0.60+ mIoU would require fundamental changes (see Future Improvements)

### Use Case Recommendations

**Recommended Applications**
- Terrain characterization (Sky, Landscape, Vegetation)
- Drivability assessment (using robust classes: IoU > 0.40)
- Environmental monitoring (large-scale features)

**Not Recommended**
- Safety-critical obstacle detection (Logs, Rocks IoU < 0.20)
- Fine-grained classification (boundary precision limited)
- Small object detection (use specialized detector instead)

## Future Improvements

### Data Collection
- **Targeted sampling**: Focus on rare classes (Logs, Rocks)
- **Expected gain**: +3-5% on rare classes
- **Effort**: 500-1000 additional images emphasizing underrepresented classes

### Model Architecture
- **Multi-scale feature fusion**: FPN-style decoder combining multiple DINOv2 layers
- **Expected gain**: +5-8% mIoU
- **Memory cost**: 8-10GB VRAM (vs current 6GB)

- **Partial backbone fine-tuning**: Unfreeze last 2-4 blocks with low learning rate (1e-5)
- **Expected gain**: +3-5% mIoU
- **Training time**: 3x longer

### Training Strategy
- **Higher resolution**: 756x1344 (vs current 378x672)
- **Expected gain**: +5-10% on small objects
- **Memory cost**: 16-24GB VRAM

- **Test-time augmentation**: Average predictions over multiple augmented versions
- **Expected gain**: +2-3% mIoU
- **Inference time**: 4x slower

- **Ensemble methods**: Combine multiple model predictions
- **Expected gain**: +3-5% mIoU
- **Complexity**: Linear scaling with ensemble size

### Advanced Techniques
- **Object detection integration**: Separate detector for Logs and Rocks
- **Expected gain**: Complementary system with high recall for safety-critical objects

- **Consistency regularization**: Semi-supervised learning on unlabeled data
- **Expected gain**: +5-7% if 10,000+ unlabeled images available

**Last Updated**: March 2025
**Version**: 1.0.0
**Status**: Production Ready
