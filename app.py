import glob
import os

import streamlit as st
import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")

MODEL_MAP = {
    "efficientnetb1": "efficientnet_b1",
    "inceptionv3": "inception_v3",
    "mobilenetv3": "mobilenetv3_large_100",
    "mobilenetv4": "mobilenetv4_conv_medium",
    "resnet50": "resnet50",
}

NUM_CLASSES = 10

IDX_TO_CLASS = {
    0: "Bacterial_spot",
    1: "Early_blight",
    2: "healthy",
    3: "Late_blight",
    4: "Leaf_Mold",
    5: "Septoria_leaf_spot",
    6: "Spider_mites Two-spotted_spider_mite",
    7: "Target_Spot",
    8: "Tomato_mosaic_virus",
    9: "Tomato_Yellow_Leaf_Curl_Virus",
}

FRIENDLY_LABELS = {
    "Bacterial_spot": "Bacterial Spot",
    "Early_blight": "Early Blight",
    "healthy": "Healthy",
    "Late_blight": "Late Blight",
    "Leaf_Mold": "Leaf Mold",
    "Septoria_leaf_spot": "Septoria Leaf Spot",
    "Spider_mites Two-spotted_spider_mite": "Spider Mites (Two-spotted)",
    "Target_Spot": "Target Spot",
    "Tomato_mosaic_virus": "Tomato Mosaic Virus",
    "Tomato_Yellow_Leaf_Curl_Virus": "Yellow Leaf Curl Virus",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_image_size(model_name: str) -> int:
    """Return expected input size — 299 for Inception, 224 otherwise."""
    return 299 if "inception" in model_name else 224


def _build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _discover_weights() -> list[str]:
    """Return sorted list of .pth filenames in models/."""
    pth_files = glob.glob(os.path.join(MODELS_DIR, "*.pth"))
    return sorted([os.path.basename(f) for f in pth_files])


def _discover_sample_images() -> list[str]:
    """Return sorted list of image filenames in images/."""
    exts = ("*.jpg", "*.jpeg", "*.png")
    files: list[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(IMAGES_DIR, ext)))
    return sorted([os.path.basename(f) for f in files])


def _load_model(weight_filename: str) -> tuple[nn.Module, str]:
    """Create a timm model and load weights. Returns (model, timm_name)."""
    stem = os.path.splitext(weight_filename)[0]  # e.g. "mobilenetv3"
    timm_name = MODEL_MAP.get(stem)
    if timm_name is None:
        raise ValueError(
            f"Unknown model '{stem}'. Expected one of: {list(MODEL_MAP.keys())}"
        )

    weight_path = os.path.join(MODELS_DIR, weight_filename)

    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, nn.Module):
        # Full model was saved
        model = checkpoint
    else:
        # state_dict (plain or wrapped in a checkpoint dict)
        model = timm.create_model(timm_name, pretrained=False, num_classes=NUM_CLASSES)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    model.eval()
    return model, timm_name


@torch.no_grad()
def _predict(
    model: nn.Module, image: Image.Image, timm_name: str
) -> list[tuple[str, str, float]]:
    """Run inference. Returns list of (class_key, friendly_label, prob) sorted descending."""
    img_size = _get_image_size(timm_name)
    transform = _build_transform(img_size)
    tensor = transform(image.convert("RGB")).unsqueeze(0)

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    results = []
    for idx, prob in enumerate(probs.tolist()):
        class_key = IDX_TO_CLASS[idx]
        friendly = FRIENDLY_LABELS.get(class_key, class_key)
        results.append((class_key, friendly, prob))

    results.sort(key=lambda x: x[2], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Tomato Leaf Disease Classifier", layout="wide")
st.title("Tomato Leaf Disease Classifier")

# ---- Sidebar: model selection ----
with st.sidebar:
    st.header("Model Setup")

    weight_files = _discover_weights()
    if not weight_files:
        st.error(f"No .pth files found in `{MODELS_DIR}`")
        st.stop()

    selected_weight = st.selectbox("Select model weights", weight_files)

    if st.button("Load Model"):
        with st.spinner("Loading model..."):
            try:
                model, timm_name = _load_model(selected_weight)
                st.session_state["model"] = model
                st.session_state["timm_name"] = timm_name
                st.session_state["model_label"] = selected_weight
                st.success(f"Loaded **{selected_weight}**")
            except Exception as e:
                st.error(f"Failed to load model: {e}")

    if "model_label" in st.session_state:
        st.info(f"Active model: **{st.session_state['model_label']}**")

# ---- Main area: image input & prediction ----
col_upload, col_result = st.columns([1, 1])

with col_upload:
    st.subheader("Input Image")

    input_mode = st.radio("Image source", ["Upload", "Sample images"], horizontal=True)

    image: Image.Image | None = None

    if input_mode == "Upload":
        uploaded = st.file_uploader(
            "Upload a tomato leaf image", type=["jpg", "jpeg", "png"]
        )
        if uploaded is not None:
            image = Image.open(uploaded)
    else:
        sample_files = _discover_sample_images()
        if sample_files:
            chosen = st.selectbox("Pick a sample image", sample_files)
            image = Image.open(os.path.join(IMAGES_DIR, chosen))
        else:
            st.warning("No sample images found in `images/` folder.")

    if image is not None:
        st.image(image, caption="Input image", use_container_width=True)

with col_result:
    st.subheader("Prediction")

    if image is not None and "model" in st.session_state:
        with st.spinner("Running inference..."):
            results = _predict(
                st.session_state["model"], image, st.session_state["timm_name"]
            )

        top_key, top_label, top_prob = results[0]

        st.metric(label="Predicted Class", value=top_label, delta=f"{top_prob:.1%}")

        if top_key == "healthy":
            st.success(f"The leaf appears **healthy** ({top_prob:.1%} confidence)")
        else:
            st.warning(
                f"Possible disease: **{top_label}** ({top_prob:.1%} confidence)"
            )

        # Bar chart of all class probabilities
        st.subheader("Class Probabilities")
        chart_data = {friendly: prob for _, friendly, prob in results}
        st.bar_chart(chart_data)

    elif image is not None:
        st.warning("Please load a model first (see sidebar).")
    else:
        st.info("Upload or select an image to get started.")
