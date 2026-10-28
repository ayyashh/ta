"""Preprocess images using the OpenCV pipeline for the selected paper approach.

Input images are read recursively and saved as PNG images. The in-memory
pipeline output is an RGB float32 array in [0, 1].
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Sesuaikan INPUT_DIR bila gambar Anda berada di folder dataset yang berbeda.
INPUT_DIR = PROJECT_ROOT / "Dataset" / "Data Gambar" / "gambar_terpilih"
OUTPUT_DIR = Path(__file__).resolve().parent / "hasil_preprocessing_opencv"

ROI_AREA_FRACTION = 0.80
OUTPUT_WIDTH = 1024
OUTPUT_HEIGHT = 768
MEDIAN_KERNEL_SIZE = 3
DEBUG_VISUALIZE = True  # Tampilkan perbandingan tahap untuk sejumlah gambar.
DEBUG_IMAGE_LIMIT = 5

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def center_crop(image: np.ndarray, area_fraction: float = ROI_AREA_FRACTION) -> np.ndarray:
    """Crop a centered ROI covering approximately ``area_fraction`` of pixels."""
    if not 0 < area_fraction <= 1:
        raise ValueError("area_fraction harus berada pada rentang (0, 1].")

    # Use the same scale for width and height so the crop remains centered
    # and keeps the input aspect ratio. The resulting pixel area is ~80%.
    scale = area_fraction**0.5
    height, width = image.shape[:2]
    crop_width = max(1, round(width * scale))
    crop_height = max(1, round(height * scale))
    x0 = (width - crop_width) // 2
    y0 = (height - crop_height) // 2
    return image[y0 : y0 + crop_height, x0 : x0 + crop_width]


def normalize_brightness_and_equalize(image_bgr: np.ndarray) -> np.ndarray:
    """Normalize luminance range, then equalize its histogram, preserving color."""
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    luminance, cr, cb = cv2.split(ycrcb)

    # Stretch luminance to the full 8-bit range before histogram equalization.
    normalized_luminance = cv2.normalize(
        luminance, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    ).astype(np.uint8)
    equalized_luminance = cv2.equalizeHist(normalized_luminance)

    adjusted = cv2.merge((equalized_luminance, cr, cb))
    return cv2.cvtColor(adjusted, cv2.COLOR_YCrCb2BGR)


def preprocess_image(image_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Apply pipeline and return final RGB float image plus intermediate stages."""
    stages = {"Gambar asli": cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)}
    roi = center_crop(image_bgr)
    stages["Cropping ROI"] = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        roi,
        (OUTPUT_WIDTH, OUTPUT_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    stages["Resizing 1024 x 768"] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    corrected = normalize_brightness_and_equalize(resized)
    stages["Normalisasi kecerahan + ekualisasi"] = cv2.cvtColor(
        corrected, cv2.COLOR_BGR2RGB
    )
    denoised = cv2.medianBlur(corrected, MEDIAN_KERNEL_SIZE)
    stages["Median filter 3 x 3"] = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)

    # OpenCV loads BGR; convert to RGB for typical machine-learning pipelines.
    rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    stages["Normalisasi skala 0-1"] = normalized
    return normalized, stages


def show_debug_stages(image_path: Path, stages: dict[str, np.ndarray]) -> None:
    """Display each pipeline stage and print its shape and value range."""
    print(f"\\nDebug: {image_path.name}")
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    for axis, (stage_name, image) in zip(axes.flat, stages.items()):
        axis.imshow(image)
        axis.set_title(stage_name)
        axis.axis("off")
        print(
            f"  {stage_name}: shape={image.shape}, "
            f"rentang=[{image.min():.4f}, {image.max():.4f}]"
        )
    figure.suptitle(f"Debug preprocessing: {image_path.name}")
    figure.tight_layout()
    plt.show()
    plt.close(figure)


def main() -> None:
    input_dir = INPUT_DIR.resolve()
    output_dir = OUTPUT_DIR.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Folder input tidak ditemukan: {input_dir}")

    image_paths = sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        print(f"Tidak ada gambar ditemukan di: {input_dir}")
        return

    debug_count = 0
    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"Lewati gambar yang tidak dapat dibaca: {image_path}")
            continue

        processed, stages = preprocess_image(image_bgr)
        if DEBUG_VISUALIZE and debug_count < DEBUG_IMAGE_LIMIT:
            show_debug_stages(image_path, stages)
            debug_count += 1

        # Normalisasi 0-1 dipakai di RAM; PNG menyimpan piksel 8-bit.
        output_rgb = np.rint(processed * 255.0).clip(0, 255).astype(np.uint8)
        output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
        relative_path = image_path.relative_to(input_dir).with_suffix(".png")
        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), output_bgr):
            print(f"Gagal menyimpan: {output_path}")
            continue
        print(f"Tersimpan: {output_path}")


if __name__ == "__main__":
    main()
