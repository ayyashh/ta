"""Extract normalized RGB color histograms from preprocessed images."""

import csv
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "hasil_preprocessing_opencv"
OUTPUT_CSV = SCRIPT_DIR / "fitur_histogram_rgb.csv"

BIN_COUNT = 256  # 256 bin per kanal: total 768 fitur per gambar.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def extract_rgb_histogram(image_bgr: np.ndarray) -> np.ndarray:
    """Return concatenated, sum-normalized R, G, B histograms (768 values)."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    channel_histograms = []

    for channel_index in range(3):  # RGB order
        histogram = cv2.calcHist(
            [image_rgb], [channel_index], None, [BIN_COUNT], [0, 256]
        ).flatten()
        total = histogram.sum()
        if total > 0:
            histogram /= total
        channel_histograms.append(histogram)

    return np.concatenate(channel_histograms).astype(np.float32)


def main() -> None:
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"Folder hasil preprocessing tidak ditemukan: {INPUT_DIR}")

    image_paths = sorted(
        path for path in INPUT_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        print(f"Tidak ada gambar ditemukan di: {INPUT_DIR}")
        return

    feature_names = [
        f"{channel}_bin_{bin_index:03d}"
        for channel in ("r", "g", "b")
        for bin_index in range(BIN_COUNT)
    ]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_path", *feature_names])

        for image_path in image_paths:
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                print(f"Lewati gambar yang tidak dapat dibaca: {image_path}")
                continue

            features = extract_rgb_histogram(image_bgr)
            relative_path = image_path.relative_to(INPUT_DIR).as_posix()
            writer.writerow([relative_path, *[f"{value:.8f}" for value in features]])
            print(
                f"Diproses: {relative_path} | jumlah fitur={features.size} | "
                f"jumlah histogram R/G/B={features[:BIN_COUNT].sum():.3f}/"
                f"{features[BIN_COUNT:2 * BIN_COUNT].sum():.3f}/"
                f"{features[2 * BIN_COUNT:].sum():.3f}"
            )

    print(f"Fitur tersimpan di: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
