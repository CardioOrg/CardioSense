

import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import DenseNet121_Weights, densenet121


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "cardiosense_cxr_densenet121.pth"
LABEL_MAP_PATH = BASE_DIR / "label_map.json"
IMG_SIZE = 224


st.set_page_config(
    page_title="CardioSense CXR",
    page_icon="CXR",
    layout="centered",
)

# Light styling to keep the layout readable
st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(135deg, #0b1728, #122840);
            color: #e5edf5;
        }
        .stApp {
            background: transparent;
        }
        .card {
            background: rgba(255, 255, 255, 0.04);
            padding: 1rem 1.25rem;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


#Load Labels

@st.cache_resource
def load_labels(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing label map at {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    labels = data.get("labels")
    if not labels:
        raise ValueError("label_map.json does not contain a 'labels' list")
    return labels


@st.cache_resource
def load_model_and_device() -> Tuple[torch.nn.Module, List[str], torch.device]:
    labels = load_labels(LABEL_MAP_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, len(labels))

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model weights at {MODEL_PATH}. "
            "Place cardiosense_cxr_densenet121.pth next to app.py."
        )
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    return model, labels, device


@st.cache_resource
def get_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=DenseNet121_Weights.IMAGENET1K_V1.transforms().mean,
                std=DenseNet121_Weights.IMAGENET1K_V1.transforms().std,
            ),
        ]
    )


def predict(image: Image.Image, threshold: float) -> Tuple[pd.DataFrame, List[Tuple[str, float]]]:
    model, labels, device = load_model_and_device()
    tfm = get_transform()

    tensor = tfm(image).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()

    rows = []
    positives = []
    for label, prob in sorted(zip(labels, probs), key=lambda x: x[1], reverse=True):
        rows.append({"Label": label, "Probability": prob})
        if prob >= threshold:
            positives.append((label, prob))

    df = pd.DataFrame(rows)
    return df, positives


def main():
    st.title("CardioSense Chest X-ray Triage")
    st.caption("Multi-label DenseNet121 classifier for cardiothoracic findings")

    with st.sidebar:
        st.header("Settings")
        threshold = st.slider(
            "Positive call threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.05,
            help="Predictions at or above this probability are flagged as positive.",
        )
        try:
            _, labels, device = load_model_and_device()
            st.success(f"Model ready on {device}. Labels: {', '.join(labels)}")
        except Exception as e:
            st.error(str(e))
            return

        st.write("Upload a frontal chest X-ray (PNG/JPG).")

    uploaded = st.file_uploader("Drag and drop a chest X-ray", type=["png", "jpg", "jpeg"])

    if not uploaded:
        st.info("Waiting for an image...")
        return

    try:
        image = Image.open(uploaded)
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception as e:
        st.error(f"Unable to read the image: {e}")
        return

    st.subheader("Preview")
    st.image(image, caption=uploaded.name, use_column_width=True, output_format="PNG")

    with st.spinner("Running inference..."):
        results_df, positives = predict(image, threshold)

    st.subheader("Predictions")
    st.dataframe(results_df.style.format({"Probability": "{:.1%}"}), use_container_width=True)

    st.markdown("### Positive findings")
    if positives:
        for label, prob in positives:
            st.write(f"- {label}: {prob:.1%}")
    else:
        st.write("No labels crossed the threshold.")

    st.markdown("---")
    st.markdown(
        "Model: DenseNet121 trained on NIH ChestX-ray (labels: "
        + ", ".join(load_labels(LABEL_MAP_PATH))
        + ")."
    )


if __name__ == "__main__":
    main()
