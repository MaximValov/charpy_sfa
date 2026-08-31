"""FractureMask — CNN segmentation with YOLO scale-bar measurement.

The CNN is expected to return three channels in this order:
    0: ductile, 1: brittle, 2: background

The original predictor used a 256 x 256 grayscale input. That contract is
kept here so an existing .h5 model can be used without retraining.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
import requests
from PIL import Image

RESIZED_IMAGE_LENGTH = 256
CLASS_NAMES = ("Ductile", "Brittle", "Background")
CLASS_CHARPY = 0
CLASS_SCALEBAR = 1
COLOR_CHARPY = (0, 255, 0)       # Green, BGR
COLOR_SCALEBAR = (0, 165, 255)   # Orange, BGR

# RGB colors matching the source script's OpenCV palette:
# ductile = blue, brittle = red, background = green.
CLASS_COLORS_RGB = np.array(
    [
        [0, 0, 255],   # Ductile - Blue
        [255, 0, 0],   # Brittle - Red
        [0, 255, 0],   # Background - Green
    ],
    dtype=np.uint8,
)

# GitHub repository configuration
GITHUB_REPO = "MaximValov/charpy_sfa"
GITHUB_BRANCH = "main"
GITHUB_RAW_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}"
GITHUB_RELEASE_URL = f"https://github.com/{GITHUB_REPO}/releases/download/v1.0"

# Model URLs - Using raw GitHub URLs first, fallback to releases
DEFAULT_CNN_URL = f"https://github.com/MaximValov/charpy_sfa/releases/download/v1/unet23jan_model.h5"
DEFAULT_YOLO_URL = f"https://github.com/MaximValov/charpy_sfa/releases/download/v1/best.pt"


def custom_conv2d_transpose(**kwargs: Any) -> Any:
    """Allow models saved with a newer Keras `groups` argument to load."""
    from keras.layers import Conv2DTranspose

    kwargs.pop("groups", None)
    return Conv2DTranspose(**kwargs)


def validate_model_file(file_path: str, model_type: str) -> bool:
    """Validate that the model file exists and is readable."""
    path = Path(file_path)
    if not path.exists():
        return False

    # Check file size
    file_size = path.stat().st_size
    if file_size < 1024:  # Less than 1KB
        return False

    # For H5 files, check header
    if model_type == "cnn" and (file_path.endswith('.h5') or file_path.endswith('.keras')):
        try:
            with open(file_path, 'rb') as f:
                header = f.read(10)
                # HDF5 files start with the magic bytes
                if header[:8] != b'\x89HDF\r\n\x1a\n':
                    return False
        except Exception:
            return False

    return True


@st.cache_resource(show_spinner=False)
def download_model_from_github(url: str, model_kind: str) -> str:
    """Download a model from GitHub and save it to a temporary file."""
    model_dir = Path(tempfile.gettempdir()) / "fracturemask-models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create a filename from the URL
    filename = Path(url).name
    if not filename:
        filename = f"{model_kind}_model"

    # Add hash to avoid conflicts
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:8]
    target = model_dir / f"{model_kind}-{url_hash}-{filename}"

    if target.exists() and validate_model_file(str(target), model_kind):
        return str(target)

    with st.spinner(f"Downloading {model_kind} model from GitHub..."):
        try:
            # Check if URL is accessible
            head_response = requests.head(url)
            if head_response.status_code != 200:
                st.warning(f"Model not found at: {url}")
                st.warning("Please upload the model manually or check the URL.")
                raise requests.exceptions.RequestException(f"URL not accessible: {url}")

            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            progress_bar = st.progress(0)

            with open(target, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    if total_size > 0:
                        downloaded += len(chunk)
                        progress_bar.progress(min(downloaded / total_size, 1.0))

            progress_bar.empty()

            if not validate_model_file(str(target), model_kind):
                st.error(f"Downloaded {model_kind} model appears to be invalid.")
                target.unlink(missing_ok=True)
                raise ValueError(f"Invalid {model_kind} model file")

            return str(target)
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to download model: {e}")
            raise


@st.cache_resource(show_spinner=False)
def load_cnn_model(model_path: str) -> Any:
    """Load the user's U-Net/CNN once per process."""
    from tensorflow import keras

    # Check if it's a URL and download if needed
    if model_path.startswith(("http://", "https://")):
        model_path = download_model_from_github(model_path, "cnn")

    if not validate_model_file(model_path, "cnn"):
        raise FileNotFoundError(f"Invalid CNN model: {model_path}")

    try:
        model = keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={"Conv2DTranspose": custom_conv2d_transpose},
        )
        return model
    except Exception as e:
        st.error(f"Failed to load CNN model: {e}")
        raise


@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path: str) -> Any:
    """Load the scale-bar detector once per process."""
    from ultralytics import YOLO

    if model_path.startswith(("http://", "https://")):
        model_path = download_model_from_github(model_path, "yolo")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")

    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        raise


@st.cache_resource(show_spinner=False)
def load_ocr_reader() -> Any:
    """Create EasyOCR lazily because its model download is relatively large."""
    import easyocr
    return easyocr.Reader(["en"], gpu=False, verbose=False)


def persist_uploaded_model(uploaded_file: Any, model_kind: str) -> str:
    """Write an uploaded model to a stable temporary path for model loaders."""
    payload = uploaded_file.getvalue()
    digest = hashlib.sha256(payload).hexdigest()[:16]
    suffix = Path(uploaded_file.name).suffix.lower()
    model_dir = Path(tempfile.gettempdir()) / "fracturemask-models"
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / f"{model_kind}-{digest}{suffix}"
    if not target.exists():
        target.write_bytes(payload)

    if not validate_model_file(str(target), model_kind):
        st.error(f"Uploaded {model_kind} model appears to be invalid.")
        target.unlink(missing_ok=True)
        raise ValueError(f"Invalid uploaded {model_kind} model")

    return str(target)


def first_existing_path(candidates: list[str]) -> str | None:
    """Pick the first local model path that exists, if any."""
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.is_file():
            return str(path)
    return None


def preprocess_image(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the original grayscale image and normalized CNN input."""
    if image_rgb.ndim == 2:
        gray = image_rgb
    else:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    resized = cv2.resize(
        gray,
        (RESIZED_IMAGE_LENGTH, RESIZED_IMAGE_LENGTH),
        interpolation=cv2.INTER_AREA,
    )
    normalized = resized.astype(np.float32) / 255.0
    return gray, np.expand_dims(normalized, axis=(0, -1))


def postprocess_predictions(
    prediction: np.ndarray,
    brittle_class_idx: int = 1,
    min_brittle_area: int = 100,
    keep_only_largest: bool = True,
    morphology_kernel: int = 3,
) -> np.ndarray:
    """Clean small brittle components while keeping a valid 3-class output."""
    if prediction.ndim != 3 or prediction.shape[-1] < 3:
        raise ValueError(
            f"The CNN must return H x W x 3 probabilities; got {prediction.shape}."
        )

    processed = np.asarray(prediction[..., :3], dtype=np.float32).copy()
    labels = np.argmax(processed, axis=-1)
    brittle_mask = (labels == brittle_class_idx).astype(np.uint8)

    if np.sum(brittle_mask) > 0:
        num_labels, component_labels, stats, _ = cv2.connectedComponentsWithStats(
            brittle_mask, connectivity=8
        )
        valid_components: list[tuple[int, int]] = []
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= min_brittle_area:
                valid_components.append((label, area))

        cleaned_brittle = np.zeros_like(brittle_mask)
        if keep_only_largest and valid_components:
            largest_label = max(valid_components, key=lambda item: item[1])[0]
            cleaned_brittle[component_labels == largest_label] = 1
        else:
            for label, _ in valid_components:
                cleaned_brittle[component_labels == label] = 1

        kernel_size = max(1, int(morphology_kernel))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        cleaned_brittle = cv2.morphologyEx(
            cleaned_brittle, cv2.MORPH_CLOSE, kernel
        )
        cleaned_brittle = cv2.morphologyEx(
            cleaned_brittle, cv2.MORPH_OPEN, kernel
        )

        # Make brittle pixels unambiguously brittle
        processed[cleaned_brittle == 1, :] = 0.0
        processed[..., brittle_class_idx] = cleaned_brittle.astype(np.float32)
        outside = cleaned_brittle == 0
        processed[outside, brittle_class_idx] = np.maximum(
            processed[outside, brittle_class_idx], 0.0
        )
        sums = processed[..., :3].sum(axis=-1, keepdims=True)
        processed = np.divide(
            processed,
            np.maximum(sums, 1e-7),
            out=np.zeros_like(processed),
            where=sums > 1e-7,
        )
        processed[cleaned_brittle == 1, :] = 0.0
        processed[..., brittle_class_idx][cleaned_brittle == 1] = 1.0

    return processed


def run_cnn(
    model: Any,
    image_rgb: np.ndarray,
    apply_postprocessing: bool,
    min_brittle_area: int,
    keep_only_largest: bool,
    morphology_kernel: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the CNN and return raw probabilities, final probabilities, labels."""
    _, model_input = preprocess_image(image_rgb)
    raw_prediction = np.asarray(model.predict(model_input, verbose=0)[0])
    if raw_prediction.ndim != 3 or raw_prediction.shape[-1] < 3:
        raise ValueError(
            "The loaded CNN does not produce three segmentation classes "
            f"(received {raw_prediction.shape})."
        )

    if apply_postprocessing:
        final_prediction = postprocess_predictions(
            raw_prediction,
            min_brittle_area=min_brittle_area,
            keep_only_largest=keep_only_largest,
            morphology_kernel=morphology_kernel,
        )
    else:
        final_prediction = raw_prediction[..., :3]
    labels_256 = np.argmax(final_prediction, axis=-1).astype(np.uint8)
    return raw_prediction[..., :3], final_prediction, labels_256


def normalize_yolo_class_names(names: Any) -> dict[int, str]:
    """Convert Ultralytics class names into a JSON-friendly integer mapping."""
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    return {
        CLASS_CHARPY: "charpy",
        CLASS_SCALEBAR: "scalebar",
    }


def yolo_box_records(result: Any, class_names: dict[int, str]) -> list[dict[str, Any]]:
    """Serialize all kept YOLO boxes for drawing, metadata, and downloads."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy()
    class_ids = (
        boxes.cls.detach().cpu().numpy().astype(int)
        if boxes.cls is not None
        else np.zeros(len(xyxy), dtype=int)
    )
    confidences = (
        boxes.conf.detach().cpu().numpy()
        if boxes.conf is not None
        else np.ones(len(xyxy), dtype=np.float32)
    )

    records: list[dict[str, Any]] = []
    for coordinates, class_id, box_confidence in zip(
        xyxy, class_ids, confidences
    ):
        records.append(
            {
                "xyxy": [int(round(value)) for value in coordinates.tolist()],
                "class_id": int(class_id),
                "class_name": class_names.get(int(class_id), f"class_{class_id}"),
                "confidence": float(box_confidence),
            }
        )
    return records


def draw_yolo_boxes(
    image_bgr: np.ndarray,
    boxes: list[Any],
    class_names: dict[int, str],
) -> np.ndarray:
    """
    Draw bounding boxes for Charpy (green) and scale bar (orange).
    """
    out = image_bgr.copy()
    for box in boxes:
        if isinstance(box, dict):
            cls_id = int(box.get("class_id", -1))
            cls_name = str(
                box.get("class_name")
                or class_names.get(cls_id, f"class_{cls_id}")
            )
            confidence = float(box.get("confidence", 0.0))
            x1, y1, x2, y2 = [int(value) for value in box["xyxy"]]
        else:
            # Also accept native Ultralytics Boxes items
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            cls_name = class_names.get(cls_id, f"class_{cls_id}")
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        normalized_name = cls_name.lower().replace("_", "").replace("-", "")
        if "scale" in normalized_name or "bar" in normalized_name:
            color = COLOR_SCALEBAR
        elif "charpy" in normalized_name:
            color = COLOR_CHARPY
        elif cls_id == CLASS_SCALEBAR:
            color = COLOR_SCALEBAR
        else:
            color = COLOR_CHARPY

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        label_top = max(0, y1 - text_height - baseline - 6)
        cv2.rectangle(
            out,
            (x1, label_top),
            (x1 + text_width + 6, y1),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            out,
            label,
            (x1 + 3, max(text_height + 2, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def detect_scale_bar(
    yolo_model: Any,
    image_rgb: np.ndarray,
    charpy_confidence: float,
    scalebar_confidence: float,
    use_ocr: bool,
) -> dict[str, Any]:
    """Detect, measure, and annotate the highest-confidence YOLO scale bar."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Run YOLO with the lower confidence threshold to catch both classes
    # We'll filter by class-specific confidence later
    min_confidence = min(charpy_confidence, scalebar_confidence)
    result = yolo_model(image_rgb, conf=min_confidence, verbose=False)[0]
    class_names = normalize_yolo_class_names(
        getattr(result, "names", None) or getattr(yolo_model, "names", None)
    )
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return {
            "found": False,
            "message": "YOLO did not detect any objects.",
            "boxes": [],
            "class_names": class_names,
        }

    all_boxes = yolo_box_records(result, class_names)

    # Filter boxes by class-specific confidence thresholds
    filtered_boxes = []
    for box in all_boxes:
        cls_name = box["class_name"].lower()
        cls_id = box["class_id"]
        confidence = box["confidence"]

        # Check if this is a scale bar
        is_scale = "scale" in cls_name or "bar" in cls_name or cls_id == CLASS_SCALEBAR
        is_charpy = "charpy" in cls_name or cls_id == CLASS_CHARPY

        if is_scale and confidence >= scalebar_confidence:
            filtered_boxes.append(box)
        elif is_charpy and confidence >= charpy_confidence:
            filtered_boxes.append(box)
        elif not is_scale and not is_charpy:
            # For unknown classes, use the lower of the two thresholds
            if confidence >= min_confidence:
                filtered_boxes.append(box)

    if not filtered_boxes:
        return {
            "found": False,
            "message": f"No objects met confidence thresholds (Charpy: {charpy_confidence:.2f}, Scale: {scalebar_confidence:.2f}).",
            "boxes": all_boxes,  # Return all boxes for display
            "class_names": class_names,
        }

    # Find scale bar candidates from filtered boxes
    scale_indices = [
        index
        for index, box in enumerate(filtered_boxes)
        if (
            "scale" in box["class_name"].lower()
            or "bar" in box["class_name"].lower()
            or box["class_id"] == CLASS_SCALEBAR
        )
    ]

    # If no scale bar found, use the highest confidence box as fallback
    if not scale_indices:
        # Use the highest confidence box from filtered results
        best_idx = max(range(len(filtered_boxes)), key=lambda i: filtered_boxes[i]["confidence"])
        scale_indices = [best_idx]
        is_fallback = True
    else:
        is_fallback = False

    best_index = max(
        scale_indices,
        key=lambda index: filtered_boxes[index]["confidence"],
    )
    best_box = filtered_boxes[best_index]
    x1, y1, x2, y2 = best_box["xyxy"]
    height, width = image_rgb.shape[:2]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return {
            "found": False,
            "message": "YOLO returned an invalid scale-bar box.",
            "boxes": filtered_boxes,
            "class_names": class_names,
        }

    # Only a scale-bar class is valid for measuring the scale
    best_name = best_box["class_name"].lower()
    if not any(term in best_name for term in ["scale", "bar"]) and not is_fallback:
        return {
            "found": False,
            "message": "YOLO detected objects, but no scale-bar class was found.",
            "boxes": filtered_boxes,
            "class_names": class_names,
        }

    crop = gray[y1:y2, x1:x2]
    crop_height, crop_width = crop.shape[:2]

    # A scale bar can be either light-on-dark or dark-on-light. Search both
    # threshold polarities and select the longest compact horizontal component.
    candidates: list[tuple[int, int, int, int, int]] = []
    for threshold_type in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, threshold = cv2.threshold(
            crop, 0, 255, threshold_type | cv2.THRESH_OTSU
        )
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(3, crop_width // 10), 1)
        )
        horizontal = cv2.morphologyEx(
            threshold, cv2.MORPH_CLOSE, horizontal_kernel
        )
        contours, _ = cv2.findContours(
            horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            cx, cy, cw, ch = cv2.boundingRect(contour)
            if cw >= max(5, int(crop_width * 0.12)) and cw >= ch * 3:
                # Prefer long, thin components; penalize text-like blobs.
                score = int(cw * 100 / max(ch, 1))
                candidates.append((score, cw, cx, cy, ch))

    if candidates:
        _, line_px, line_cx, line_cy, line_ch = max(
            candidates, key=lambda candidate: (candidate[1], candidate[0])
        )
        line_x1 = x1 + line_cx
        line_y1 = y1 + line_cy + line_ch // 2
        line_x2 = line_x1 + line_px
        line_y2 = line_y1
        measurement_source = "horizontal component inside YOLO box"
    else:
        # The detector box remains the authoritative outline. If thresholding
        # cannot isolate the bar, expose the box width instead of hiding px.
        line_px = crop_width
        line_x1, line_x2 = x1, x2
        line_y1 = line_y2 = y1 + max(1, crop_height // 2)
        measurement_source = "YOLO box width fallback"

    ocr_text = ""
    scale_value_mm = None
    scale_unit = None
    pixel_size_mm = None

    if use_ocr:
        try:
            reader = load_ocr_reader()
            # Upscaling improves recognition for small labels in microscopy
            # images while keeping the OCR crop restricted to YOLO's box.
            ocr_crop = cv2.resize(
                crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC
            )
            ocr_text = " ".join(
                str(text) for text in reader.readtext(ocr_crop, detail=0)
            ).strip()
            normalized_text = (
                ocr_text.lower()
                .replace("μ", "µ")
                .replace("㎛", "µm")
                .replace("µ m", "µm")
                .replace("u m", "um")
            )
            match = re.search(
                r"(?P<value>\d+(?:[.,]\d+)?)\s*"
                r"(?P<unit>mm|µm|um|nm)\b",
                normalized_text,
                flags=re.IGNORECASE,
            )
            if match:
                value = float(match.group("value").replace(",", "."))
                unit = match.group("unit").lower()
                conversion_to_mm = {"nm": 1e-6, "µm": 1e-3, "um": 1e-3, "mm": 1.0}
                scale_value_mm = value * conversion_to_mm[unit]
                scale_unit = unit
                pixel_size_mm = scale_value_mm / line_px
        except Exception as exc:
            st.warning(f"OCR failed; px measurement is still available: {exc}")

    # Find Charpy boxes from filtered results
    charpy_boxes = [
        box
        for box in filtered_boxes
        if (
            "charpy" in box["class_name"].lower()
            or box["class_id"] == CLASS_CHARPY
        )
    ]
    charpy_width_px = 0
    if charpy_boxes:
        charpy_width_px = max(
            int(box["xyxy"][2] - box["xyxy"][0]) for box in charpy_boxes
        )
    charpy_width_mm = (
        charpy_width_px * pixel_size_mm
        if charpy_width_px > 0 and pixel_size_mm is not None
        else None
    )

    return {
        "found": True,
        "box": (x1, y1, x2, y2),
        "line": (line_x1, line_y1, line_x2, line_y2),
        "line_px": int(line_px),
        "box_width_px": int(x2 - x1),
        "confidence": float(best_box["confidence"]),
        "ocr_text": ocr_text,
        "scale_value_mm": scale_value_mm,
        "scale_unit": scale_unit,
        "pixel_size_mm": pixel_size_mm,
        "measurement_source": measurement_source,
        "boxes": filtered_boxes,  # Only boxes that passed confidence thresholds
        "all_boxes": all_boxes,   # All boxes for reference
        "class_names": class_names,
        "charpy_width_px": charpy_width_px,
        "charpy_width_mm": charpy_width_mm,
        "is_fallback": is_fallback,
    }


def render_mask(labels: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize class labels and apply the visible class palette."""
    width, height = target_size
    labels_original = cv2.resize(
        labels,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    return CLASS_COLORS_RGB[labels_original]


def build_overlay(
    image_rgb: np.ndarray,
    colored_mask: np.ndarray,
    scale: dict[str, Any] | None,
    opacity: float,
) -> np.ndarray:
    """Blend segmentation, measured scale, and all YOLO detections."""
    blended = cv2.addWeighted(
        image_rgb,
        1.0 - opacity,
        colored_mask,
        opacity,
        0,
    )
    blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

    # Draw YOLO boxes if they exist
    if scale and scale.get("boxes"):
        blended_bgr = draw_yolo_boxes(
            blended_bgr,
            scale["boxes"],
            scale.get("class_names", {}),
        )

    # Draw scale measurement if found
    if scale and scale.get("found"):
        x1, y1, x2, y2 = scale["box"]
        lx1, ly, lx2, _ = scale["line"]
        label = f"{scale['line_px']:,} px"
        if scale.get("scale_value_mm") is not None:
            label += f"  |  {scale['scale_value_mm']:.6g} mm"

        font = cv2.FONT_HERSHEY_SIMPLEX
        (label_width, label_height), baseline = cv2.getTextSize(
            label, font, 0.62, 2
        )
        label_x = max(4, min(lx1, blended_bgr.shape[1] - label_width - 8))
        label_y = max(label_height + baseline + 4, ly - 10)

        # Draw the measured line and its px/mm value first.
        cv2.line(
            blended_bgr,
            (lx1, ly),
            (lx2, ly),
            (0, 0, 255),
            3,
        )
        cv2.rectangle(
            blended_bgr,
            (label_x - 4, label_y - label_height - baseline - 4),
            (label_x + label_width + 4, label_y + 4),
            (0, 0, 0),
            cv2.FILLED,
        )
        cv2.putText(
            blended_bgr,
            label,
            (label_x, label_y),
            font,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if scale and scale.get("charpy_width_mm") is not None:
        cv2.putText(
            blended_bgr,
            f"Charpy width: {scale['charpy_width_mm']:.6g} mm",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            COLOR_CHARPY,
            2,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)


def png_bytes(image_rgb: np.ndarray) -> bytes:
    """Encode an RGB ndarray as a downloadable PNG."""
    ok, encoded = cv2.imencode(
        ".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    )
    if not ok:
        raise ValueError("Could not encode the result as PNG.")
    return encoded.tobytes()


def grayscale_mask_bytes(labels: np.ndarray, target_size: tuple[int, int]) -> bytes:
    width, height = target_size
    full_mask = cv2.resize(
        labels,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    ok, encoded = cv2.imencode(".png", full_mask)
    if not ok:
        raise ValueError("Could not encode the class mask as PNG.")
    return encoded.tobytes()


def create_download_bundle(
    original_rgb: np.ndarray,
    colored_mask: np.ndarray,
    overlay: np.ndarray,
    labels: np.ndarray,
    metadata: dict[str, Any],
) -> bytes:
    """Package the three image products and metadata in one ZIP."""
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("original.png", png_bytes(original_rgb))
        bundle.writestr("segmentation_mask.png", png_bytes(colored_mask))
        bundle.writestr("class_mask.png", grayscale_mask_bytes(labels, (original_rgb.shape[1], original_rgb.shape[0])))
        bundle.writestr("overlay.png", png_bytes(overlay))
        bundle.writestr("metadata.json", json.dumps(metadata, indent=2))
    return archive.getvalue()


def show_legend() -> None:
    legend = "  •  ".join(
        f"{name}: {'blue' if index == 0 else 'red' if index == 1 else 'green'}"
        for index, name in enumerate(CLASS_NAMES)
    )
    st.caption(legend)


def main() -> None:
    st.set_page_config(
        page_title="FractureMask",
        page_icon="FM",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("FractureMask")
    st.write(
        "Segment brittle fracture, ductile fracture, and background from "
        "microscopy images. YOLO finds the scale bar so results can include "
        "a pixel-to-millimetre estimate."
    )

    with st.sidebar:
        st.header("Models")
        st.caption(
            "Models are automatically loaded from GitHub. You can optionally upload custom models."
        )

        st.info(f"📦 Default models from: {GITHUB_REPO}")

        cnn_upload = st.file_uploader(
            "CNN / U-Net model (.h5, .keras) - upload to override",
            type=["h5", "keras"],
            key="cnn_upload",
        )
        cnn_path = st.text_input(
            "CNN model path or URL",
            value=DEFAULT_CNN_URL,
            help="Path to a local file or a URL to a model hosted on GitHub.",
        )
        yolo_upload = st.file_uploader(
            "YOLO scale-bar model (.pt) - upload to override",
            type=["pt"],
            key="yolo_upload",
        )
        yolo_path = st.text_input(
            "YOLO model path or URL",
            value=DEFAULT_YOLO_URL,
            help="Path to a local file or a URL to a model hosted on GitHub.",
        )

        st.header("Segmentation")
        apply_postprocessing = st.checkbox(
            "Clean brittle mask",
            value=True,
            help="Removes small connected brittle regions and applies morphology.",
        )
        min_brittle_area = st.slider(
            "Minimum brittle region (CNN pixels)",
            min_value=0,
            max_value=5000,
            value=100,
            step=25,
            disabled=not apply_postprocessing,
        )
        keep_only_largest = st.checkbox(
            "Keep only largest brittle region",
            value=True,
            disabled=not apply_postprocessing,
        )
        morphology_kernel = st.slider(
            "Morphology kernel",
            min_value=1,
            max_value=9,
            value=3,
            step=2,
            disabled=not apply_postprocessing,
        )
        overlay_opacity = st.slider(
            "Mask opacity",
            min_value=0.1,
            max_value=0.8,
            value=0.3,
            step=0.05,
        )

        st.header("YOLO Detection")
        st.caption("Separate confidence thresholds for Charpy and scale bar detection")

        charpy_confidence = st.slider(
            "Charpy confidence threshold",
            min_value=0.01,
            max_value=0.99,
            value=0.25,
            step=0.01,
            help="Minimum confidence for Charpy notch detection"
        )

        scalebar_confidence = st.slider(
            "Scale bar confidence threshold",
            min_value=0.01,
            max_value=0.99,
            value=0.01,
            step=0.01,
            help="Minimum confidence for scale bar detection"
        )

        use_ocr = st.checkbox(
            "Read scale value with EasyOCR",
            value=True,
            help="The first run may take longer while EasyOCR prepares its reader.",
        )

        # Manual scale override
        st.header("Manual Scale (if OCR fails)")
        use_manual_scale = st.checkbox("Use manual scale instead of OCR", value=False)
        if use_manual_scale:
            col1, col2 = st.columns(2)
            with col1:
                manual_px = st.number_input(
                    "Scale bar pixels", min_value=1, value=1035, step=10
                )
            with col2:
                manual_mm = st.number_input(
                    "Scale length (mm)",
                    min_value=0.0,
                    value=0.1,
                    step=0.01,
                    format="%.4f",
                )

    uploaded_image = st.file_uploader(
        "Upload a fracture image",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        help="The CNN converts the image to grayscale and resizes it to 256 × 256.",
    )
    show_legend()

    if uploaded_image is None:
        st.info(
            "Upload an image to begin. Models will be automatically downloaded from GitHub "
            "when you run the analysis."
        )
        with st.expander("Expected model contract"):
            st.markdown(
                "- **CNN:** Keras model with a grayscale `256 × 256 × 1` input "
                "and three output channels in the order **ductile, brittle, background**.\n"
                "- **YOLO:** Ultralytics detector trained to find the scale-bar region.\n"
                "- **OCR:** EasyOCR reads labels such as `50 µm` or `0.1 mm` inside the YOLO crop.\n"
                f"\n**Default models are loaded from:** {GITHUB_REPO}"
            )
        return

    image_bytes = uploaded_image.getvalue()
    image_signature = hashlib.sha256(image_bytes).hexdigest()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_rgb = np.asarray(image)
    st.image(image_rgb, caption=f"{uploaded_image.name} · {image_rgb.shape[1]} × {image_rgb.shape[0]} px")

    # Resolve model paths
    if cnn_upload is not None:
        resolved_cnn_path = persist_uploaded_model(cnn_upload, "cnn")
        st.sidebar.success(f"Using uploaded CNN: {cnn_upload.name}")
    else:
        if cnn_path.startswith(("http://", "https://")):
            resolved_cnn_path = cnn_path
        else:
            resolved_cnn_path = first_existing_path([cnn_path, "unet23jan_model.h5", "models/unet23jan_model.h5"])
            if resolved_cnn_path is None:
                resolved_cnn_path = DEFAULT_CNN_URL
                st.sidebar.info("Using default CNN model from GitHub")

    if yolo_upload is not None:
        resolved_yolo_path = persist_uploaded_model(yolo_upload, "yolo")
        st.sidebar.success(f"Using uploaded YOLO: {yolo_upload.name}")
    else:
        if yolo_path.startswith(("http://", "https://")):
            resolved_yolo_path = yolo_path
        else:
            resolved_yolo_path = first_existing_path([yolo_path, "best.pt", "models/best.pt"])
            if resolved_yolo_path is None:
                resolved_yolo_path = DEFAULT_YOLO_URL
                st.sidebar.info("Using default YOLO model from GitHub")

    run_analysis = st.button("Run fracture analysis", type="primary", use_container_width=True)
    if run_analysis:
        with st.spinner("Loading models and running analysis..."):
            try:
                cnn_model = load_cnn_model(resolved_cnn_path)
                raw_pred, final_pred, labels_256 = run_cnn(
                    cnn_model,
                    image_rgb,
                    apply_postprocessing,
                    min_brittle_area,
                    keep_only_largest,
                    morphology_kernel,
                )

                scale_result = None
                yolo_warning = None
                if resolved_yolo_path:
                    try:
                        yolo_model = load_yolo_model(resolved_yolo_path)
                        scale_result = detect_scale_bar(
                            yolo_model,
                            image_rgb,
                            charpy_confidence,
                            scalebar_confidence,
                            use_ocr and not use_manual_scale,
                        )

                        # Override with manual scale if enabled
                        if use_manual_scale and scale_result and scale_result.get("found"):
                            scale_result["line_px"] = int(manual_px)
                            scale_result["pixel_size_mm"] = manual_mm / max(manual_px, 1)
                            scale_result["scale_value_mm"] = manual_mm
                            scale_result["scale_unit"] = "mm"
                            if scale_result.get("charpy_width_px", 0) > 0:
                                scale_result["charpy_width_mm"] = (
                                    scale_result["charpy_width_px"]
                                    * scale_result["pixel_size_mm"]
                                )
                            scale_result["manual"] = True

                    except Exception as exc:
                        yolo_warning = f"YOLO scale-bar detection failed: {exc}"
                else:
                    yolo_warning = "No YOLO model available."

                colored_mask = render_mask(
                    cv2.resize(
                        labels_256,
                        (image_rgb.shape[1], image_rgb.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ),
                    (image_rgb.shape[1], image_rgb.shape[0]),
                )
                overlay = build_overlay(
                    image_rgb, colored_mask, scale_result, overlay_opacity
                )
                pixel_counts = np.bincount(
                    cv2.resize(
                        labels_256,
                        (image_rgb.shape[1], image_rgb.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).ravel(),
                    minlength=3,
                )
                total_pixels = int(pixel_counts.sum())
                metadata = {
                    "image": uploaded_image.name,
                    "image_width_px": int(image_rgb.shape[1]),
                    "image_height_px": int(image_rgb.shape[0]),
                    "cnn_input": "256x256 grayscale normalized to [0, 1]",
                    "classes": list(CLASS_NAMES),
                    "postprocessing": apply_postprocessing,
                    "min_brittle_area_cnn_pixels": min_brittle_area,
                    "keep_only_largest": keep_only_largest,
                    "charpy_confidence_threshold": charpy_confidence,
                    "scalebar_confidence_threshold": scalebar_confidence,
                    "pixel_counts": pixel_counts.tolist(),
                    "percent_area": (pixel_counts / max(total_pixels, 1) * 100).round(4).tolist(),
                    "scale": scale_result,
                }
                st.session_state["result"] = {
                    "signature": image_signature,
                    "raw_pred": raw_pred,
                    "final_pred": final_pred,
                    "labels_256": labels_256,
                    "colored_mask": colored_mask,
                    "overlay": overlay,
                    "scale": scale_result,
                    "metadata": metadata,
                    "pixel_counts": pixel_counts,
                    "yolo_warning": yolo_warning,
                }
            except Exception as exc:
                st.session_state.pop("result", None)
                st.error(f"Analysis failed: {exc}")

    result = st.session_state.get("result")
    if not result or result.get("signature") != image_signature:
        st.caption("Choose the analysis settings, then run the analysis.")
        return

    if result.get("yolo_warning"):
        st.warning(result["yolo_warning"])
    elif result.get("scale", {}).get("found"):
        scale = result["scale"]
        scale_text = (
            f"YOLO found the scale bar at {scale['confidence']:.0%} confidence"
            f" · {scale['line_px']:,} px"
        )
        if scale.get("pixel_size_mm") is not None:
            if scale.get("manual"):
                scale_text += (
                    f" · {scale['scale_value_mm']:.6g} mm"
                    f" · {scale['pixel_size_mm']:.6g} mm/px (manual)"
                )
            else:
                scale_text += (
                    f" · {scale['scale_value_mm']:.6g} mm"
                    f" · {scale['pixel_size_mm']:.6g} mm/px"
                )
        elif scale.get("ocr_text"):
            scale_text += f" · OCR: “{scale['ocr_text']}”"
        else:
            scale_text += " · mm value not read"
        st.success(scale_text)
    else:
        st.info(result.get("scale", {}).get("message", "No scale bar detected."))

    st.subheader("Segmentation result")
    result_columns = st.columns(2)
    with result_columns[0]:
        st.image(result["overlay"], caption="Overlay with scale-bar annotation", use_container_width=True)
    with result_columns[1]:
        st.image(result["colored_mask"], caption="Class mask", use_container_width=True)

    scale = result.get("scale")
    if scale and scale.get("found"):
        st.subheader("Detected scale bar")
        scale_columns = st.columns(4)
        with scale_columns[0]:
            st.metric("Bar length", f"{scale['line_px']:,} px")
        with scale_columns[1]:
            if scale.get("scale_value_mm") is not None:
                st.metric("Bar value", f"{scale['scale_value_mm']:.6g} mm")
            else:
                st.metric("Bar value", "Not read")
        with scale_columns[2]:
            if scale.get("pixel_size_mm") is not None:
                st.metric("Scale", f"{scale['pixel_size_mm']:.6g} mm/px")
            else:
                st.metric("Scale", "Unavailable")
        with scale_columns[3]:
            st.metric("YOLO confidence", f"{scale['confidence']:.1%}")
        if scale.get("ocr_text"):
            st.caption(f"OCR text: {scale['ocr_text']}")

    counts = result["pixel_counts"]
    total = max(int(counts.sum()), 1)
    metric_columns = st.columns(3)
    for index, column in enumerate(metric_columns):
        with column:
            st.metric(
                CLASS_NAMES[index],
                f"{int(counts[index]):,} px",
                f"{counts[index] / total * 100:.2f}% of image",
            )

    if result.get("scale", {}).get("pixel_size_mm") is not None:
        pixel_size_mm = float(result["scale"]["pixel_size_mm"])
        brittle_area_mm2 = int(counts[1]) * pixel_size_mm**2
        st.metric("Estimated brittle area", f"{brittle_area_mm2:.6g} mm²")

    st.subheader("Download results")
    download_columns = st.columns(4)
    with download_columns[0]:
        st.download_button(
            "Download overlay",
            png_bytes(result["overlay"]),
            file_name=f"{Path(uploaded_image.name).stem}_overlay.png",
            mime="image/png",
            use_container_width=True,
        )
    with download_columns[1]:
        st.download_button(
            "Download color mask",
            png_bytes(result["colored_mask"]),
            file_name=f"{Path(uploaded_image.name).stem}_mask.png",
            mime="image/png",
            use_container_width=True,
        )
    with download_columns[2]:
        st.download_button(
            "Download class mask",
            grayscale_mask_bytes(
                result["labels_256"],
                (image_rgb.shape[1], image_rgb.shape[0]),
            ),
            file_name=f"{Path(uploaded_image.name).stem}_class_mask.png",
            mime="image/png",
            use_container_width=True,
        )
    with download_columns[3]:
        st.download_button(
            "Download ZIP bundle",
            create_download_bundle(
                image_rgb,
                result["colored_mask"],
                result["overlay"],
                result["labels_256"],
                result["metadata"],
            ),
            file_name=f"{Path(uploaded_image.name).stem}_fracturemask.zip",
            mime="application/zip",
            use_container_width=True,
        )

    with st.expander("Run metadata"):
        st.json(result["metadata"])


if __name__ == "__main__":
    main()
