import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# ==============================
# PROFESSIONAL UI STYLING
# ==============================

def inject_custom_css():
    """Inject modern dark mode CSS with professional styling."""
    st.markdown("""
    <style>
    /* Import Professional Font - Inter */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Background - Deep Charcoal */
    .stApp {
        background: linear-gradient(135deg, #1a1d23 0%, #252932 100%);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e2228 0%, #2a2f38 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    [data-testid="stSidebar"] .element-container {
        padding: 0.5rem 0;
    }
    
    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        border-radius: 16px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 50%, #2563eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        font-weight: 400;
        line-height: 1.6;
    }
    
    /* Status Indicator */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 8px;
        font-size: 0.875rem;
        color: #10b981;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        margin-right: 0.5rem;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Card Containers */
    .card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(255, 255, 255, 0.12);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(96, 165, 250, 0.3);
    }
    
    /* Class Legend Item */
    .legend-item {
        display: flex;
        align-items: center;
        padding: 0.6rem;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 8px;
        margin-bottom: 0.5rem;
        transition: all 0.2s ease;
        border: 1px solid transparent;
    }
    
    .legend-item:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(255, 255, 255, 0.1);
        transform: translateX(4px);
    }
    
    .legend-color {
        width: 24px;
        height: 24px;
        border-radius: 6px;
        margin-right: 0.75rem;
        border: 2px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    
    .legend-name {
        font-size: 0.9rem;
        color: #cbd5e1;
        font-weight: 500;
    }
    
    /* File Uploader Styling */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.03);
        border: 2px dashed rgba(96, 165, 250, 0.3);
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(96, 165, 250, 0.6);
        background: rgba(96, 165, 250, 0.05);
    }
    
    [data-testid="stFileUploader"] label {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
    }
    
    /* Metric Cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(37, 99, 235, 0.05) 100%);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid rgba(59, 130, 246, 0.2);
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.2);
    }
    
    [data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #60a5fa !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #94a3b8 !important;
        font-size: 0.75rem !important;
    }
    
    /* Images */
    [data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Info Boxes */
    .stAlert {
        background: rgba(59, 130, 246, 0.1) !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        border-radius: 10px !important;
        color: #93c5fd !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #60a5fa !important;
    }
    
    /* Model Info Card */
    .model-info-card {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(37, 99, 235, 0.05) 100%);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid rgba(59, 130, 246, 0.2);
        margin-bottom: 1.5rem;
    }
    
    .model-info-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .model-info-item:last-child {
        border-bottom: none;
    }
    
    .model-info-label {
        color: #94a3b8;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .model-info-value {
        color: #e2e8f0;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    /* Instructions Grid */
    .instruction-card {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .instruction-card:hover {
        background: rgba(255, 255, 255, 0.04);
        border-color: rgba(96, 165, 250, 0.3);
        transform: translateY(-4px);
    }
    
    .instruction-number {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .instruction-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 0.5rem;
    }
    
    .instruction-text {
        font-size: 0.9rem;
        color: #94a3b8;
        line-height: 1.5;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #64748b;
        font-size: 0.875rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 3rem;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.02);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(96, 165, 250, 0.3);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(96, 165, 250, 0.5);
    }
    </style>
    """, unsafe_allow_html=True)


# ==============================
# CONFIG
# ==============================

DEVICE = torch.device("cpu")
IMG_HEIGHT = 378
IMG_WIDTH = 672
PATCH_SIZE = 14
NUM_CLASSES = 10

tokenW = IMG_WIDTH // PATCH_SIZE
tokenH = IMG_HEIGHT // PATCH_SIZE


# ==============================
# SEGMENTATION HEAD (UNCHANGED)
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
# LOAD MODEL (UNCHANGED)
# ==============================

@st.cache_resource
def load_model():
    """Load DINOv2 backbone and trained segmentation head."""
    
    # Load frozen backbone
    backbone = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitb14_reg", trust_repo=True
    )
    backbone.to(DEVICE)
    backbone.eval()

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
                break
            except Exception:
                continue

    if checkpoint is None:
        st.error("❌ No checkpoint found! Please ensure model checkpoint is in the repository.")
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

    except Exception as e:
        st.error(f"❌ Error loading checkpoint: {str(e)}")
        st.stop()

    return backbone, classifier


# ==============================
# IMAGE TRANSFORM (UNCHANGED)
# ==============================

transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ==============================
# CLASS COLORS AND NAMES
# ==============================

CLASS_COLORS = [
    (30, 30, 35),      # Background - Dark charcoal
    (34, 139, 34),     # Trees - Forest green
    (50, 205, 50),     # Lush Bushes - Lime green
    (189, 183, 107),   # Dry Grass - Dark khaki
    (160, 82, 45),     # Dry Bushes - Sienna
    (128, 128, 128),   # Ground Clutter - Gray
    (139, 69, 19),     # Logs - Saddle brown
    (112, 128, 144),   # Rocks - Slate gray
    (210, 180, 140),   # Landscape - Tan
    (135, 206, 235),   # Sky - Sky blue
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

CLASS_ICONS = ["⬛", "🌲", "🌿", "🌾", "🍂", "🪨", "🪵", "🗿", "🏜️", "☁️"]


# ==============================
# INFERENCE FUNCTION (UNCHANGED)
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
# STREAMLIT UI - PROFESSIONAL REDESIGN
# ==============================

st.set_page_config(
    page_title="Offroad Terrain Segmentation",
    layout="wide",
    page_icon="🚜",
    initial_sidebar_state="expanded",
)

# Inject custom CSS
inject_custom_css()

# Load model
try:
    with st.spinner("🔄 Loading AI Model..."):
        backbone, classifier = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    model_loaded = False
    st.stop()


# ==============================
# SIDEBAR
# ==============================

with st.sidebar:
    # Status Indicator
    if model_loaded:
        st.markdown("""
        <div class="status-indicator">
            <div class="status-dot"></div>
            <span>Model Ready</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Model Information Card
    st.markdown('<div class="section-header">⚙️ Model Information</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="model-info-card">
        <div class="model-info-item">
            <span class="model-info-label">Backbone</span>
            <span class="model-info-value">DINOv2 ViT-B/14</span>
        </div>
        <div class="model-info-item">
            <span class="model-info-label">Input Size</span>
            <span class="model-info-value">672×378</span>
        </div>
        <div class="model-info-item">
            <span class="model-info-label">Classes</span>
            <span class="model-info-value">10 Terrain Types</span>
        </div>
        <div class="model-info-item">
            <span class="model-info-label">Device</span>
            <span class="model-info-value">CPU</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Class Legend
    st.markdown('<div class="section-header">🎨 Class Legend</div>', unsafe_allow_html=True)
    
    for name, color, icon in zip(CLASS_NAMES, CLASS_COLORS, CLASS_ICONS):
        color_hex = f"rgb({color[0]}, {color[1]}, {color[2]})"
        st.markdown(f"""
        <div class="legend-item">
            <div class="legend-color" style="background-color: {color_hex};"></div>
            <span class="legend-name">{icon} {name}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # About Section
    st.markdown('<div class="section-header">📖 About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color: #94a3b8; font-size: 0.9rem; line-height: 1.6;">
    Advanced AI-powered terrain segmentation system designed for offroad environments.
    Identifies 10 distinct terrain categories including vegetation, obstacles, and ground elements.
    </div>
    """, unsafe_allow_html=True)


# ==============================
# MAIN CONTENT
# ==============================

# Hero Section
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🚜 Offroad Terrain Segmentation</div>
    <div class="hero-subtitle">
        AI-powered real-time terrain analysis using DINOv2 Vision Transformer. 
        Upload an image to identify different terrain types with pixel-level precision.
    </div>
</div>
""", unsafe_allow_html=True)

# File Uploader
uploaded_file = st.file_uploader(
    "📁 Upload Terrain Image",
    type=["jpg", "jpeg", "png"],
    help="Supported: JPG, JPEG, PNG • Max 200MB",
    label_visibility="collapsed"
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    
    # Display original and segmented side by side
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown('<div class="section-header">📷 Original Image</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">🎨 Segmentation Result</div>', unsafe_allow_html=True)
        
        # Run segmentation with spinner
        with st.spinner("🔄 Analyzing terrain..."):
            output_mask, prediction = segment_image(image)
        
        st.image(output_mask, use_container_width=True)
    
    # Statistics Section
    st.markdown("---")
    st.markdown('<div class="section-header">📊 Terrain Distribution Analysis</div>', unsafe_allow_html=True)
    
    unique, counts = np.unique(prediction, return_counts=True)
    total_pixels = prediction.size
    
    # Calculate percentages and sort by dominance
    terrain_data = []
    for class_id in range(NUM_CLASSES):
        if class_id in unique:
            count = counts[unique == class_id][0]
            percentage = (count / total_pixels) * 100
            terrain_data.append((class_id, percentage, count))
        else:
            terrain_data.append((class_id, 0.0, 0))
    
    # Sort by percentage (descending)
    terrain_data_sorted = sorted(terrain_data, key=lambda x: x[1], reverse=True)
    
    # Display top 5 terrain types as featured metrics
    st.markdown("### 🏆 Dominant Terrain Types")
    cols = st.columns(5)
    
    for i, (class_id, percentage, count) in enumerate(terrain_data_sorted[:5]):
        with cols[i]:
            st.metric(
                label=f"{CLASS_ICONS[class_id]} {CLASS_NAMES[class_id]}",
                value=f"{percentage:.1f}%",
                delta=f"{count:,} pixels"
            )
    
    # Display all terrain types in grid
    st.markdown("### 📋 Complete Breakdown")
    cols = st.columns(5)
    
    for i, (class_id, percentage, count) in enumerate(terrain_data_sorted):
        with cols[i % 5]:
            st.metric(
                label=f"{CLASS_ICONS[class_id]} {CLASS_NAMES[class_id]}",
                value=f"{percentage:.1f}%",
                delta=f"{count:,} px"
            )

else:
    # Instructions when no image uploaded
    st.markdown('<div class="section-header">🚀 Get Started</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="instruction-card">
            <div class="instruction-number">1️⃣</div>
            <div class="instruction-title">Upload</div>
            <div class="instruction-text">
                Choose an offroad terrain image in JPG, JPEG, or PNG format from your device
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="instruction-card">
            <div class="instruction-number">2️⃣</div>
            <div class="instruction-title">Process</div>
            <div class="instruction-text">
                Our AI model analyzes the image and segments different terrain types automatically
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="instruction-card">
            <div class="instruction-number">3️⃣</div>
            <div class="instruction-title">Analyze</div>
            <div class="instruction-text">
                View the segmentation map and detailed terrain distribution statistics
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Sample use cases
    st.markdown("---")
    st.markdown('<div class="section-header">💡 Use Cases</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h4 style="color: #e2e8f0; margin-bottom: 0.5rem;">🚗 Autonomous Navigation</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">
                Enable self-driving vehicles to understand terrain composition and plan optimal routes
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h4 style="color: #e2e8f0; margin-bottom: 0.5rem;">🌍 Environmental Monitoring</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">
                Track vegetation patterns, erosion, and landscape changes over time
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h4 style="color: #e2e8f0; margin-bottom: 0.5rem;">🎮 Game Development</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">
                Generate realistic terrain maps for racing games and simulation environments
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h4 style="color: #e2e8f0; margin-bottom: 0.5rem;">🛰️ Satellite Analysis</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">
                Analyze aerial and satellite imagery for agricultural and geological surveys
            </p>
        </div>
        """, unsafe_allow_html=True)
