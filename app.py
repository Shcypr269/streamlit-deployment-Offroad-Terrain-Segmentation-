import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import gc

# Force PyTorch to cache downloaded models in the local directory
os.environ["TORCH_HOME"] = "./.cache/torch"

# ==============================
# CONFIG
# ==============================

DEVICE = torch.device("cpu")  # Streamlit Cloud uses CPU
IMG_HEIGHT = 378
IMG_WIDTH = 672
PATCH_SIZE = 14
NUM_CLASSES = 10

tokenW = IMG_WIDTH // PATCH_SIZE  # 48
tokenH = IMG_HEIGHT // PATCH_SIZE  # 27

# ==============================
# SEGMENTATION HEAD
# ==============================


class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H = tokenH
        self.W = tokenW

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, out_channels, 1),
        )

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        return self.decoder(x)


# ==============================
# LOAD MODEL (CACHED)
# ==============================


@st.cache_resource
def load_model():
    """Load DINOv2 backbone and trained segmentation head."""

    with st.spinner("Loading DINOv2 backbone..."):
        # Load frozen backbone
        backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14_reg", trust_repo=True
        )
        backbone.to(DEVICE)
        backbone.eval()

        # Freeze all parameters
        for param in backbone.parameters():
            param.requires_grad = False

    # Detect embedding dimension
    dummy_input = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).to(DEVICE)
    with torch.no_grad():
        dummy_features = backbone.forward_features(dummy_input)["x_norm_patchtokens"]
        embed_dim = dummy_features.shape[2]

    # Create segmentation head
    classifier = SegmentationHeadConvNeXt(
        in_channels=embed_dim, out_channels=NUM_CLASSES, tokenW=tokenW, tokenH=tokenH
    ).to(DEVICE)

    # Load checkpoint
    checkpoint_paths = [
        "segmentation_head_best.pth",
        "best_model.pth",
        "checkpoint.pth",
    ]

    checkpoint = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
                st.success(f"Loaded checkpoint: {path}")
                break
            except Exception as e:
                continue

    if checkpoint is None:
        st.error(
            "No checkpoint found! Please ensure model checkpoint is in the repository."
        )
        st.info(
            """
        **To fix this:**
        1. Train your model using train_first.py
        2. Save the checkpoint as 'segmentation_head_best.pth'
        3. Upload it to your GitHub repository (use Git LFS if >100MB)
        4. Redeploy the app
        """
        )
        st.stop()

    # Load weights
    try:
        if isinstance(checkpoint, dict):
            if "head" in checkpoint:
                classifier.load_state_dict(checkpoint["head"])
            elif "model" in checkpoint:
                classifier.load_state_dict(checkpoint["model"])
            elif "state_dict" in checkpoint:
                classifier.load_state_dict(checkpoint["state_dict"])
            else:
                classifier.load_state_dict(checkpoint)
        else:
            classifier.load_state_dict(checkpoint)

        classifier.eval()
        st.success("Model loaded successfully!")

    except Exception as e:
        st.error(f"Error loading checkpoint: {str(e)}")
        st.stop()

    return backbone, classifier


# Load model once at startup
try:
    backbone, classifier = load_model()
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# ==============================
# IMAGE TRANSFORM
# ==============================

transform = transforms.Compose(
    [
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# ==============================
# CLASS COLORS AND NAMES
# ==============================

CLASS_COLORS = [
    (0, 0, 0),  # Background
    (34, 139, 34),  # Trees
    (50, 205, 50),  # Lush Bushes
    (189, 183, 107),  # Dry Grass
    (160, 82, 45),  # Dry Bushes
    (128, 128, 128),  # Ground Clutter
    (139, 69, 19),  # Logs
    (112, 128, 144),  # Rocks
    (210, 180, 140),  # Landscape
    (135, 206, 235),  # Sky
]

CLASS_NAMES = [
    "Background",
    "Trees",
    "Lush Bushes",
    "Dry Grass",
    "Dry Bushes",
    "Ground Clutter",
    "Logs",
    "Rocks",
    "Landscape",
    "Sky",
]

# ==============================
# INFERENCE FUNCTION
# ==============================


def segment_image(image):
    """Perform segmentation on input image."""
    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = backbone.forward_features(input_tensor)["x_norm_patchtokens"]
        logits = classifier(features)
        logits = F.interpolate(
            logits, size=(IMG_HEIGHT, IMG_WIDTH), mode="bilinear", align_corners=False
        )
        prediction = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

    # Create colored mask
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    for class_id, color in enumerate(CLASS_COLORS):
        mask[prediction == class_id] = color

    return Image.fromarray(mask), prediction


# ==============================
# STREAMLIT UI
# ==============================

st.set_page_config(
    page_title="Offroad Terrain Segmentation",
    layout="wide",
    page_icon="",
    initial_sidebar_state="expanded",
)

# Header
st.title("Offroad Terrain Segmentation")
st.markdown(
    """
AI-powered terrain segmentation for offroad environments using DINOv2 + Custom Decoder.
Upload an image to identify different terrain types in real-time.
"""
)

# Sidebar
with st.sidebar:
    st.header("Model Information")
    st.markdown(
        f"""
    - **Backbone**: DINOv2 ViT-B/14
    - **Input Size**: {IMG_WIDTH}×{IMG_HEIGHT}
    - **Classes**: {NUM_CLASSES}
    - **Device**: CPU
    """
    )

    st.markdown("---")

    st.header("Class Legend")
    for name, color in zip(CLASS_NAMES, CLASS_COLORS):
        col1, col2 = st.columns([1, 4])
        with col1:
            color_box = np.ones((20, 20, 3), dtype=np.uint8)
            color_box[:, :] = color
            st.image(color_box, width=20)
        with col2:
            st.markdown(f"**{name}**")

    st.markdown("---")

    st.header("About")
    st.markdown(
        """
    This model segments offroad terrain into 10 categories including:
    - Vegetation (trees, bushes, grass)
    - Obstacles (logs, rocks)
    - Ground elements
    - Background features
    """
    )

# Main content
uploaded_file = st.file_uploader(
    "Upload an offroad terrain image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG",
)

if uploaded_file is not None:
    # Use file ID to prevent re-running same image
    file_id = uploaded_file.file_id
    
    if "current_file_id" not in st.session_state or st.session_state.current_file_id != file_id:
        image = Image.open(uploaded_file)
        with st.spinner("Segmenting terrain..."):
            output_mask, prediction = segment_image(image)
            
        # Store in session state
        st.session_state.current_file_id = file_id
        st.session_state.image = image
        st.session_state.output_mask = output_mask
        st.session_state.prediction = prediction

    # Display from session state
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(st.session_state.image, use_container_width=True)
    with col2:
        st.subheader("Segmentation Result")
        st.image(st.session_state.output_mask, use_container_width=True)

    # Statistics
    st.markdown("---")
    st.subheader("Terrain Distribution")

    unique, counts = np.unique(prediction, return_counts=True)
    total_pixels = prediction.size

    # Create metrics in columns
    cols = st.columns(5)
    for i, class_id in enumerate(range(NUM_CLASSES)):
        with cols[i % 5]:
            if class_id in unique:
                count = counts[unique == class_id][0]
                percentage = (count / total_pixels) * 100
                st.metric(
                    label=CLASS_NAMES[class_id],
                    value=f"{percentage:.1f}%",
                    delta=f"{count:,} px",
                )
            else:
                st.metric(label=CLASS_NAMES[class_id], value="0.0%", delta="0 px")

else:
    # Instructions when no image is uploaded
    st.info("Upload an image to get started!")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 1. Upload")
        st.markdown("Choose an offroad terrain image (JPG, JPEG, or PNG format)")

    with col2:
        st.markdown("### 2. Process")
        st.markdown("The AI model will segment different terrain types automatically")

    with col3:
        st.markdown("### 3. Analyze")
        st.markdown("View the segmentation map and terrain distribution statistics")

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center'>
    <p>Powered by DINOv2 (Meta AI) | Built with Streamlit</p>
</div>
""",
    unsafe_allow_html=True,
)
