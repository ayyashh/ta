"""Visualisasikan jumlah patch, posisi window, dan contoh patch."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = Path("/kaggle/working/CNN_Moving_Window")
MANIFEST_PATH = OUTPUT_DIR / "moving_window_manifest.csv"
PLOT_DIR = OUTPUT_DIR / "visualisasi_dataset"

if not MANIFEST_PATH.is_file():
    raise FileNotFoundError(
        f"Manifest tidak ditemukan: {MANIFEST_PATH}. Jalankan Kode 1 dulu."
    )

manifest = pd.read_csv(MANIFEST_PATH, encoding="utf-8-sig")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

split_order = ["train", "validation", "test"]

patch_counts = (
    manifest["split"]
    .value_counts()
    .reindex(split_order, fill_value=0)
)
source_counts = (
    manifest.groupby("split")["source_image"]
    .nunique()
    .reindex(split_order, fill_value=0)
)

# Jumlah patch dan jumlah gambar sumber per subset.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].bar(split_order, patch_counts.values)
axes[0].set_title("Jumlah patch per subset")
axes[0].set_ylabel("Jumlah patch")

axes[1].bar(split_order, source_counts.values)
axes[1].set_title("Jumlah gambar sumber per subset")
axes[1].set_ylabel("Jumlah gambar")

for axis in axes:
    axis.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(
    PLOT_DIR / "jumlah_patch_dan_gambar.png",
    dpi=160,
    bbox_inches="tight",
)

# Koordinat awal moving window.
fig, axis = plt.subplots(figsize=(7, 6))

for split_name in split_order:
    subset = manifest[manifest["split"] == split_name]
    if not subset.empty:
        axis.scatter(
            subset["x"],
            subset["y"],
            alpha=0.4,
            label=split_name,
        )

axis.set_xlabel("Koordinat x awal window (px)")
axis.set_ylabel("Koordinat y awal window (px)")
axis.set_title("Posisi moving window")
axis.invert_yaxis()
axis.grid(alpha=0.2)
axis.legend(title="Subset")

fig.tight_layout()
fig.savefig(
    PLOT_DIR / "sebaran_posisi_window.png",
    dpi=160,
    bbox_inches="tight",
)

# Contoh maksimal empat patch per subset.
available_splits = [
    name for name in split_order
    if (manifest["split"] == name).any()
]

fig, axes = plt.subplots(
    len(available_splits),
    4,
    figsize=(10, 2.8 * len(available_splits)),
    squeeze=False,
)

for row_index, split_name in enumerate(available_splits):
    subset = manifest[manifest["split"] == split_name]
    samples = subset.sample(
        n=min(4, len(subset)),
        random_state=42,
    )

    for column_index, axis in enumerate(axes[row_index]):
        axis.axis("off")

        if column_index >= len(samples):
            continue

        item = samples.iloc[column_index]
        patch_path = OUTPUT_DIR / item["patch_path"]

        if patch_path.is_file():
            axis.imshow(plt.imread(patch_path))
            axis.set_title(
                f"{split_name}: x={item['x']}, y={item['y']}"
            )
        else:
            axis.set_title("Patch tidak ditemukan")

fig.tight_layout()
fig.savefig(
    PLOT_DIR / "contoh_patch_per_subset.png",
    dpi=160,
    bbox_inches="tight",
)

plt.show()
plt.close("all")
print(f"Grafik tersimpan di: {PLOT_DIR}")