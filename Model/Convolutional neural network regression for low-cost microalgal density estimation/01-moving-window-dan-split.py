"""Split gambar sumber, buat moving-window patch, dan simpan manifest."""

from pathlib import Path
import re

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split


KAGGLE_INPUT = Path("/kaggle/input")
IMAGE_FOLDER_NAME = Path(r"D:/00 Kuliah/TA/Experiment/Dataset/Data Gambar/gambar_terpilih")
LABELS_FILENAME = Path(r"D:\00 Kuliah\TA\Experiment\Dataset\Data Sensor Kultivasi\csv\gabungan_dengan_jumlah_sel.csv")
IMAGE_COLUMN = "Nama_Gambar"
TARGET_COLUMN = "Jumlah Sel (sel/mL)"

WINDOW_SIZE = 32
STRIDE = 20

TEST_RATIO = 0.20
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def image_key(value):
    normalized = str(value).replace("\\", "/")
    return Path(normalized).stem.strip().lower()


def parse_target(value):
    """Baca target integer dengan pemisah ribuan titik atau koma."""
    text = str(value).strip().replace(" ", "")
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", text):
        text = text.replace(".", "").replace(",", "")
    return float(text)


if KAGGLE_INPUT.is_dir():
    image_dirs = [
        path for path in KAGGLE_INPUT.rglob(IMAGE_FOLDER_NAME.name)
        if path.is_dir()
    ]
    label_files = list(KAGGLE_INPUT.rglob(LABELS_FILENAME.name))

    if len(image_dirs) != 1:
        raise FileNotFoundError(
            f"Harus ditemukan tepat satu folder {IMAGE_FOLDER_NAME.name}; "
            f"ditemukan {len(image_dirs)}."
        )
    if len(label_files) != 1:
        raise FileNotFoundError(
            f"Harus ditemukan tepat satu file {LABELS_FILENAME.name}; "
            f"ditemukan {len(label_files)}."
        )

    IMAGE_DIR = image_dirs[0]
    LABELS_CSV = label_files[0]
    OUTPUT_DIR = Path("/kaggle/working/CNN_Moving_Window")
else:
    IMAGE_DIR = IMAGE_FOLDER_NAME
    LABELS_CSV = LABELS_FILENAME
    OUTPUT_DIR = Path(__file__).resolve().parent / "CNN_Moving_Window"

PATCH_DIR = OUTPUT_DIR / "patches"
MANIFEST_PATH = OUTPUT_DIR / "moving_window_manifest.csv"

if not IMAGE_DIR.is_dir():
    raise FileNotFoundError(f"Folder gambar tidak ditemukan: {IMAGE_DIR}")
if not LABELS_CSV.is_file():
    raise FileNotFoundError(f"CSV label tidak ditemukan: {LABELS_CSV}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

labels = pd.read_csv(
    LABELS_CSV,
    encoding="utf-8-sig",
    dtype={IMAGE_COLUMN: "string", TARGET_COLUMN: "string"},
)

required_columns = {IMAGE_COLUMN, TARGET_COLUMN}
missing_columns = required_columns - set(labels.columns)
if missing_columns:
    raise KeyError(f"Kolom label tidak ditemukan: {sorted(missing_columns)}")

labels["_key"] = labels[IMAGE_COLUMN].map(image_key)

if labels["_key"].duplicated().any():
    raise ValueError("Nama gambar pada CSV label duplikat.")

label_lookup = {
    row["_key"]: {
        "label_name": str(row[IMAGE_COLUMN]),
        "target": parse_target(row[TARGET_COLUMN]),
    }
    for _, row in labels.iterrows()
}

records = []
image_paths = sorted(
    path for path in IMAGE_DIR.rglob("*")
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
)

for image_path in image_paths:
    label_info = label_lookup.get(image_key(image_path.name))

    if label_info is None:
        print(f"Lewati, label tidak ditemukan: {image_path.name}")
        continue

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Lewati, gambar tidak dapat dibaca: {image_path}")
        continue

    records.append({
        "path": image_path,
        "source_image": image_path.relative_to(IMAGE_DIR).as_posix(),
        "Nama_Gambar": label_info["label_name"],
        "target_density": label_info["target"],
    })

if len(records) < 3:
    raise ValueError("Pasangan gambar-label valid kurang dari tiga.")

# Split dilakukan pada gambar sumber sebelum patch dibuat.
all_indices = list(range(len(records)))

train_indices, test_indices = train_test_split(
    all_indices,
    test_size=TEST_RATIO,
    random_state=SEED,
    shuffle=True,
)

split_indices = {
    "train": train_indices,
    "test": test_indices,
}

manifest_rows = []
patch_counts = {name: 0 for name in split_indices}

for split_name, source_indices in split_indices.items():
    for source_index in source_indices:
        record = records[source_index]
        image = cv2.imread(str(record["path"]), cv2.IMREAD_COLOR)
        height, width = image.shape[:2]

        if width < WINDOW_SIZE or height < WINDOW_SIZE:
            print(f"Lewati gambar terlalu kecil: {record['path'].name}")
            continue

        patch_number = 0

        for y in range(0, height - WINDOW_SIZE + 1, STRIDE):
            for x in range(0, width - WINDOW_SIZE + 1, STRIDE):
                patch = image[y:y + WINDOW_SIZE, x:x + WINDOW_SIZE]
                patch_number += 1

                patch_name = (
                    f"{record['path'].stem}_y{y:04d}_x{x:04d}.png"
                )
                patch_path = (
                    PATCH_DIR
                    / split_name
                    / Path(record["source_image"]).parent
                    / patch_name
                )
                patch_path.parent.mkdir(parents=True, exist_ok=True)

                if not cv2.imwrite(str(patch_path), patch):
                    print(f"Gagal menyimpan patch: {patch_path}")
                    continue

                manifest_rows.append({
                    "patch_path": patch_path.relative_to(OUTPUT_DIR).as_posix(),
                    "source_image": record["source_image"],
                    "Nama_Gambar": record["Nama_Gambar"],
                    "split": split_name,
                    "patch_number": patch_number,
                    "x": x,
                    "y": y,
                    "window_width": WINDOW_SIZE,
                    "window_height": WINDOW_SIZE,
                    "target_density": record["target_density"],
                })
                patch_counts[split_name] += 1

manifest = pd.DataFrame(manifest_rows)
if manifest.empty:
    raise ValueError("Tidak ada patch yang berhasil dibuat.")

manifest.to_csv(
    MANIFEST_PATH,
    index=False,
    encoding="utf-8-sig",
)

print(f"Jumlah pasangan gambar-label: {len(records)}")
print(
    "Jumlah gambar sumber — "
    f"train: {len(train_indices)}, "
    f"test: {len(test_indices)}"
)
print(
    "Jumlah patch — "
    f"train: {patch_counts['train']}, "
    f"test: {patch_counts['test']}"
)
print(f"Manifest: {MANIFEST_PATH}")
