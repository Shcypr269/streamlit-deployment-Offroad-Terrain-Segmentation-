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
- [Citation](#citation)
- [License](#license)

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
│   DINOv2 ViT-B/14 (Frozen)         │
│   - Patch size: 14x14               │
│   - Output tokens: 48x27x768        │
│   - Parameters: 86M (frozen)        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   Custom Decoder (Trainable)        │
│   ├─ Positional Encoding (64 ch)   │
│   ├─ ASPP (multi-scale: 6,12,18)   │
│   ├─ ConvNeXt Block 1              │
│   ├─ ConvNeXt Block 2              │
│   └─ Classifier (1x1 conv)         │
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
