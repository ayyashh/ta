"""Extract per-channel statistics, 2D DFT power, entropy, and S20 features."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

INPUT_DIR = Path(r"D:\00 Kuliah\TA\Experiment\Model\Analysis of Microalgal Density Estimation by Using LASSO and Image Texture Features\cropped_roi_50x50")
OUTPUT_ALL = Path(r"D:\00 Kuliah\TA\Experiment\Model\Analysis of Microalgal Density Estimation by Using LASSO and Image Texture Features\fitur_tekstur_lengkap.csv")
OUTPUT_S20 = Path(r"D:\00 Kuliah\TA\Experiment\Model\Analysis of Microalgal Density Estimation by Using LASSO and Image Texture Features\fitur_skenario_S20.csv")

ROI_WIDTH = 50
ROI_HEIGHT = 50
PIXEL_COUNT = ROI_WIDTH * ROI_HEIGHT
CI_Z_VALUE = 1.96
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

S20_FEATURES = [
    "R_mean", "G_mean", "B_mean",
    "R_entropy", "G_entropy", "B_entropy",
    "R_entropy_squared", "G_entropy_squared", "B_entropy_squared",
    "R_entropy_x_G_entropy", "R_entropy_x_B_entropy", "G_entropy_x_B_entropy",
]


def shannon_entropy(channel: np.ndarray) -> float:
    """Calculate Shannon entropy from 256 one-intensity-wide histogram bins."""
    histogram = np.bincount(channel.reshape(-1), minlength=256).astype(np.float64)
    probabilities = histogram / histogram.sum()
    nonzero = probabilities > 0
    return float(-np.sum(probabilities[nonzero] * np.log2(probabilities[nonzero])))


def extract_features(image_bgr: np.ndarray) -> dict[str, float]:
    """Calculate all 21 listed features for a 50x50 RGB image."""
    height, width = image_bgr.shape[:2]
    if (width, height) != (ROI_WIDTH, ROI_HEIGHT):
        raise ValueError(
            f"ROI harus berukuran {ROI_WIDTH}x{ROI_HEIGHT} px, "
            f"ditemukan {width}x{height} px."
        )

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    features: dict[str, float] = {}
    entropy_values: dict[str, float] = {}

    for index, channel_name in enumerate(("R", "G", "B")):
        channel = image_rgb[:, :, index]
        values = channel.astype(np.float64)
        mean = float(np.mean(values))
        std_population = float(np.std(values, ddof=0))
        margin = CI_Z_VALUE * std_population / np.sqrt(PIXEL_COUNT)

        features[f"{channel_name}_mean"] = mean
        features[f"{channel_name}_L"] = mean - margin
        features[f"{channel_name}_R"] = mean + margin

        # Power spectrum is |2D-DFT|^2, averaged over all frequency bins.
        dft = np.fft.fft2(values)
        power_spectrum = np.abs(dft) ** 2
        features[f"{channel_name}_FT"] = float(np.mean(power_spectrum))

        entropy = shannon_entropy(channel)
        entropy_values[channel_name] = entropy
        features[f"{channel_name}_entropy"] = entropy
        features[f"{channel_name}_entropy_squared"] = entropy**2

    features["R_entropy_x_G_entropy"] = entropy_values["R"] * entropy_values["G"]
    features["R_entropy_x_B_entropy"] = entropy_values["R"] * entropy_values["B"]
    features["G_entropy_x_B_entropy"] = entropy_values["G"] * entropy_values["B"]
    return features


def main() -> None:
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"Folder ROI tidak ditemukan: {INPUT_DIR}")

    image_paths = sorted(
        path for path in INPUT_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        print(f"Tidak ada citra ROI ditemukan di: {INPUT_DIR}")
        return

    rows = []
    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"Lewati gambar yang tidak dapat dibaca: {image_path}")
            continue

        try:
            extracted = extract_features(image_bgr)
        except ValueError as error:
            print(f"Lewati {image_path.name}: {error}")
            continue

        relative_path = image_path.relative_to(INPUT_DIR).as_posix()
        rows.append({"image_path": relative_path, **extracted})
        print(
            f"OK: {relative_path} | ukuran={image_bgr.shape[1]}x{image_bgr.shape[0]} "
            f"| mean RGB=({extracted['R_mean']:.3f}, {extracted['G_mean']:.3f}, "
            f"{extracted['B_mean']:.3f}) | entropy RGB=({extracted['R_entropy']:.3f}, "
            f"{extracted['G_entropy']:.3f}, {extracted['B_entropy']:.3f})"
        )

    if not rows:
        print("Tidak ada ROI valid untuk diekstrak.")
        return

    all_features = pd.DataFrame(rows)
    OUTPUT_ALL.parent.mkdir(parents=True, exist_ok=True)
    all_features.to_csv(OUTPUT_ALL, index=False, encoding="utf-8-sig")
    all_features[["image_path", *S20_FEATURES]].to_csv(
        OUTPUT_S20, index=False, encoding="utf-8-sig"
    )
    print(f"Fitur lengkap ({len(all_features.columns) - 1} fitur) tersimpan: {OUTPUT_ALL}")
    print(f"Fitur skenario S20 (12 fitur) tersimpan: {OUTPUT_S20}")


if __name__ == "__main__":
    main()
