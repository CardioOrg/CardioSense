"""
Component 2 — Chest X-Ray Analysis.
Loads the trained DenseNet-121 from the existing pth weights file.
Includes Grad-CAM generation for the top finding.
"""

import io
import base64
import json
import logging

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import DenseNet121_Weights, densenet121

logger = logging.getLogger(__name__)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import config

IMG_SIZE = config.CXR_IMG_SIZE
IMAGENET_MEAN = list(DenseNet121_Weights.IMAGENET1K_V1.transforms().mean)
IMAGENET_STD = list(DenseNet121_Weights.IMAGENET1K_V1.transforms().std)

# ── Singleton model cache ──────────────────────────────────────────
_cache = {}


def _load_labels():
    with open(config.COMPONENT2_LABEL_MAP_PATH, "r") as f:
        return json.load(f)["labels"]


def _load_model():
    if "model" in _cache:
        return _cache["model"], _cache["labels"], _cache["device"]

    labels = _load_labels()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, len(labels))

    if not config.COMPONENT2_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"CXR model weights not found at {config.COMPONENT2_MODEL_PATH}"
        )

    state = torch.load(
        config.COMPONENT2_MODEL_PATH, map_location=device, weights_only=True
    )
    model.load_state_dict(state)
    model.to(device).eval()
    torch.set_grad_enabled(False)

    _cache["model"] = model
    _cache["labels"] = labels
    _cache["device"] = device
    return model, labels, device


def _get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _pil_to_base64(pil_image, fmt="PNG"):
    """Convert a PIL image to a base64-encoded data URI."""
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _generate_gradcam(model, image, tensor, labels, device, target_idx):
    """Generate Grad-CAM overlay for the given target label index."""
    try:
        torch.set_grad_enabled(True)

        target_layers = [model.features.denseblock4]
        cam = GradCAM(model=model, target_layers=target_layers)

        targets = [ClassifierOutputTarget(target_idx)]
        grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]  # H x W

        # Build the RGB base image (inverse-normalise the tensor)
        inv_mean = [-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]
        inv_std = [1.0 / s for s in IMAGENET_STD]
        inv_norm = transforms.Normalize(mean=inv_mean, std=inv_std)
        rgb = inv_norm(tensor.squeeze(0).cpu()).clamp(0, 1).permute(1, 2, 0).numpy()

        overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

        torch.set_grad_enabled(False)

        # Convert overlay (numpy uint8 array) to PIL → base64
        overlay_pil = Image.fromarray(overlay)
        return _pil_to_base64(overlay_pil)
    except Exception:
        torch.set_grad_enabled(False)
        return None


def analyze_cxr(image_source):
    """
    Analyse a chest X-ray image.
    image_source: file path (str/Path) or PIL.Image or file-like object.
    Returns: {
        "class_probabilities": {label: prob},
        "predicted_labels": [labels above threshold],
        "positive_findings": int,
        "top_finding": str,
        "top_probability": float,
        "all_results": [(label, prob), ...]  # sorted descending
        "threshold": float,
        "input_image_b64": str,     # base64 data URI of the input image
        "gradcam_image_b64": str,   # base64 data URI of the Grad-CAM overlay
    }
    """
    model, labels, device = _load_model()

    if isinstance(image_source, Image.Image):
        image = image_source.convert("RGB")
    elif hasattr(image_source, "read"):
        image = ImageOps.exif_transpose(Image.open(image_source)).convert("RGB")
    else:
        image = ImageOps.exif_transpose(Image.open(str(image_source))).convert("RGB")

    # Create resized version for display
    display_image = image.copy()
    display_image.thumbnail((512, 512), Image.LANCZOS)
    input_b64 = _pil_to_base64(display_image)

    tensor = _get_transform()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    results = sorted(
        [(lab, float(p)) for lab, p in zip(labels, probs)],
        key=lambda x: x[1],
        reverse=True,
    )

    threshold = 0.7
    positives = [lab for lab, p in results if p >= threshold]

    class_probs = {lab: round(float(p), 4) for lab, p in zip(labels, probs)}

    # Generate Grad-CAM for the top finding
    gradcam_b64 = None
    if results:
        top_label = results[0][0]
        top_idx = labels.index(top_label)
        gradcam_b64 = _generate_gradcam(
            model, image, tensor, labels, device, top_idx
        )

    return {
        "class_probabilities": class_probs,
        "predicted_labels": positives,
        "positive_findings": len(positives),
        "top_finding": results[0][0] if results else "",
        "top_probability": round(results[0][1], 4) if results else 0.0,
        "all_results": [(lab, round(p, 4)) for lab, p in results],
        "threshold": threshold,
        "input_image_b64": input_b64,
        "gradcam_image_b64": gradcam_b64,
    }
