"""Ekstrak rerata RGB gambar sebagai proksi pembacaan warna tiap sampel.

Tidak ada koreksi latar belakang karena nilai acuan putih tidak tersedia.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


INPUT_DIR = Path(r"D:\00 Kuliah\TA\Experiment\Dataset\Data Gambar\gambar_terpilih")
OUTPUT_CSV = Path(r"D:\00 Kuliah\TA\Experiment\Model\Cost-effective wireless system for image-based microalgae biomass estimation with adaptive background correction\fitur_rgb_sampel.csv")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main() -> None:
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"Folder gambar tidak ditemukan: {INPUT_DIR}")

    image_paths = sorted(
        path
        for path in INPUT_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise FileNotFoundError(f"Tidak ada gambar di: {INPUT_DIR}")

    rows = []

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image_bgr is None:
            print(f"Lewati, gambar tidak dapat dibaca: {image_path}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_float = image_rgb.astype(np.float32)

        row = {
            "image_path": image_path.relative_to(INPUT_DIR).as_posix(),
            "R_mean": float(image_float[:, :, 0].mean()),
            "G_mean": float(image_float[:, :, 1].mean()),
            "B_mean": float(image_float[:, :, 2].mean()),
            "R_std": float(image_float[:, :, 0].std()),
            "G_std": float(image_float[:, :, 1].std()),
            "B_std": float(image_float[:, :, 2].std()),
            "width": int(image_rgb.shape[1]),
            "height": int(image_rgb.shape[0]),
        }
        rows.append(row)

        print(
            f"OK: {image_path.name} | "
            f"RGB mean=({row['R_mean']:.2f}, "
            f"{row['G_mean']:.2f}, {row['B_mean']:.2f})"
        )

    if not rows:
        raise ValueError("Tidak ada gambar yang berhasil diproses.")

    features = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\nJumlah gambar diproses: {len(features)}")
    print(f"Fitur RGB tersimpan di: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
