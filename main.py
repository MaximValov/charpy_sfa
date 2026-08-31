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
from PIL import Image

RESIZED_IMAGE_LENGTH = 256
CLASS_NAMES = ("Ductile", "Brittle", "Background")

# RGB colors matching the source script's OpenCV palette:
# ductile = blue, brittle = red, background = green.
CLASS_COLORS_RGB = np.array(
    [
        [0, 0, 255],
        [255, 0, 0],
        [0, 255, 0],
    ],
    dtype=np.uint8,
)


def custom_conv2d_transpose(**kwargs: Any) -> Any:
    """Allow models saved with a newer Keras `groups` argument to load."""
    from keras.layers import Conv2DTranspose

    kwargs.pop("groups", None)
    return Conv2DTranspose(**kwargs)


@st.cache_resource(show_spinner=False)
def load_cnn_model(model_path: str) -> Any:
    """Load the user's U-Net/CNN once per process."""
    from tensorflow import keras

    return keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={"Conv2DTranspose": custom_conv2d_transpose},
    )


@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path: str) -> Any:
    """Load the scale-bar detector once per process."""
    from ultralytics import YOLO

    return YOLO(model_path)


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

    # Make brittle pixels unambiguously brittle, and normalize the remaining
    # two classes so downstream argmax/area calculations are consistent.
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


def detect_scale_bar(
    yolo_model: Any,
    image_rgb: np.ndarray,
    confidence: float,
    use_ocr: bool,
) -> dict[str, Any]:
    """Detect a scale bar with YOLO, measure its line, and OCR its label."""
    result = yolo_model(image_rgb, conf=confidence, verbose=False)[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return {"found": False, "message": "YOLO did not detect a scale bar."}

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidences = (
        boxes.conf.detach().cpu().numpy()
        if boxes.conf is not None
        else np.ones(len(xyxy))
    )
    best_index = int(np.argmax(confidences))
    x1, y1, x2, y2 = np.rint(xyxy[best_index]).astype(int).tolist()
    height, width = image_rgb.shape[:2]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return {"found": False, "message": "YOLO returned an invalid scale-bar box."}

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    crop = gray[y1:y2, x1:x2]
    _, threshold = cv2.threshold(
        crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    horizontal_lines: list[tuple[int, int, int, int]] = []
    for contour in contours:
        cx, cy, cw, ch = cv2.boundingRect(contour)
        if cw > ch * 3:
            horizontal_lines.append((cx, cy, cw, ch))

    if horizontal_lines:
        cx, cy, line_px, line_height = max(
            horizontal_lines, key=lambda item: item[2]
        )
        line_x1 = x1 + cx
        line_x2 = line_x1 + line_px
        line_y = y1 + cy + line_height // 2
    else:
        line_px = max(1, crop.shape[1])
        line_x1, line_x2 = x1, x2
        line_y = y1 + max(1, (y2 - y1) // 2)

    ocr_text = ""
    scale_value = None
    scale_unit = None
    pixel_size_mm = None
    if use_ocr:
        reader = load_ocr_reader()
        ocr_text = " ".join(reader.readtext(crop, detail=0))
        match = re.search(
            r"(\d+(?:\.\d+)?)\s*(µm|um|mm|nm)\b",
            ocr_text.lower(),
        )
        if match:
            value = float(match.group(1))
            scale_unit = match.group(2)
            conversion = {"nm": 1e-6, "µm": 1e-3, "um": 1e-3, "mm": 1.0}
            scale_value = value * conversion[scale_unit]
            pixel_size_mm = scale_value / line_px

    return {
        "found": True,
        "box": (x1, y1, x2, y2),
        "line": (line_x1, line_y, line_x2, line_y),
        "line_px": int(line_px),
        "confidence": float(confidences[best_index]),
        "ocr_text": ocr_text,
        "scale_value_mm": scale_value,
        "scale_unit": scale_unit,
        "pixel_size_mm": pixel_size_mm,
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
    """Blend the segmentation and annotate the detected scale bar."""
    overlay = cv2.addWeighted(
        image_rgb,
        1.0 - opacity,
        colored_mask,
        opacity,
        0,
    )
    if scale and scale.get("found"):
        x1, y1, x2, y2 = scale["box"]
        lx1, ly, lx2, _ = scale["line"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 200, 0), 2)
        cv2.line(overlay, (lx1, ly), (lx2, ly), (255, 0, 0), 3)
        label = f"{scale['line_px']} px"
        if scale.get("scale_value_mm") is not None:
            label += f"  {scale['scale_value_mm']:.4g} mm"
        cv2.putText(
            overlay,
            label,
            (max(4, lx1), max(22, ly - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return overlay


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
            "Upload the trained weights or enter a path inside the project. "
            "The CNN is required for segmentation; YOLO and OCR are optional."
        )
        cnn_upload = st.file_uploader(
            "CNN / U-Net model (.h5, .keras)",
            type=["h5", "keras"],
            key="cnn_upload",
        )
        cnn_path = st.text_input(
            "CNN model path",
            value="models/unet_model.h5",
            help="Used when no CNN file is uploaded.",
        )
        yolo_upload = st.file_uploader(
            "YOLO scale-bar model (.pt)",
            type=["pt"],
            key="yolo_upload",
        )
        yolo_path = st.text_input(
            "YOLO model path",
            value="models/scalebar.pt",
            help="Used when no YOLO file is uploaded.",
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

        st.header("Scale bar")
        yolo_confidence = st.slider(
            "YOLO confidence",
            min_value=0.05,
            max_value=0.99,
            value=0.5,
            step=0.05,
        )
        use_ocr = st.checkbox(
            "Read scale value with EasyOCR",
            value=True,
            help="The first run may take longer while EasyOCR prepares its reader.",
        )

    uploaded_image = st.file_uploader(
        "Upload a fracture image",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        help="The CNN converts the image to grayscale and resizes it to 256 × 256.",
    )
    show_legend()

    if uploaded_image is None:
        st.info(
            "Upload an image to begin. Add the trained CNN model in the sidebar "
            "before running analysis."
        )
        with st.expander("Expected model contract"):
            st.markdown(
                "- **CNN:** Keras model with a grayscale `256 × 256 × 1` input "
                "and three output channels in the order **ductile, brittle, background**.\n"
                "- **YOLO:** Ultralytics detector trained to find the scale-bar region.\n"
                "- **OCR:** EasyOCR reads labels such as `50 µm` or `0.1 mm` inside the YOLO crop."
            )
        return

    image_bytes = uploaded_image.getvalue()
    image_signature = hashlib.sha256(image_bytes).hexdigest()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_rgb = np.asarray(image)
    st.image(image_rgb, caption=f"{uploaded_image.name} · {image_rgb.shape[1]} × {image_rgb.shape[0]} px")

    if cnn_upload is not None:
        resolved_cnn_path = persist_uploaded_model(cnn_upload, "cnn")
        st.sidebar.success(f"Using uploaded CNN: {cnn_upload.name}")
    else:
        resolved_cnn_path = first_existing_path(
            [
                cnn_path,
                "unet23jan_model.h5",
                "models/unet23jan_model.h5",
                # "models/unet_model.h5",
            ]
        )

    if yolo_upload is not None:
        resolved_yolo_path = persist_uploaded_model(yolo_upload, "yolo")
        st.sidebar.success(f"Using uploaded YOLO: {yolo_upload.name}")
    else:
        resolved_yolo_path = first_existing_path(
            [
                yolo_path,
                "best.pt",
                "models/best.pt",
                "runs/detect/charpy_scalebar/weights/best.pt",
            ]
        )

    if resolved_cnn_path is None:
        st.warning(
            "No CNN model is available yet. Upload a `.h5` or `.keras` model "
            "in the sidebar, or place it at `models/unet_model.h5`."
        )
        return

    run_analysis = st.button("Run fracture analysis", type="primary", use_container_width=True)
    if run_analysis:
        with st.spinner("Running CNN segmentation and scale-bar detection…"):
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
                            yolo_model, image_rgb, yolo_confidence, use_ocr
                        )
                    except Exception as exc:
                        yolo_warning = f"YOLO scale-bar detection failed: {exc}"
                else:
                    yolo_warning = (
                        "No YOLO model available. Segmentation completed without "
                        "scale-bar measurement."
                    )

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
        scale_text = f"YOLO found the scale bar at {scale['confidence']:.0%} confidence"
        if scale.get("pixel_size_mm"):
            scale_text += f" · {scale['pixel_size_mm']:.6g} mm/px"
        elif scale.get("ocr_text"):
            scale_text += f" · OCR: “{scale['ocr_text']}”"
        st.success(scale_text)
    else:
        st.info(result.get("scale", {}).get("message", "No scale bar detected."))

    st.subheader("Segmentation result")
    result_columns = st.columns(2)
    with result_columns[0]:
        st.image(result["overlay"], caption="Overlay with scale-bar annotation", use_container_width=True)
    with result_columns[1]:
        st.image(result["colored_mask"], caption="Class mask", use_container_width=True)

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

    if result.get("scale", {}).get("pixel_size_mm"):
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
