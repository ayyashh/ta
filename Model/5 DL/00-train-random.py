"""Image + selected conventional-feature regression using five DL models."""

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from pathlib import Path
import gc
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# ============================================================
# 1. Konfigurasi input, target, model, dan training
# ============================================================

IMAGE_DIR = Path(
    r"D:\00 Kuliah\TA\Experiment\Dataset\Data Gambar\Gambar Final"
)
TARGET_CSV = Path(
    r"D:\00 Kuliah\TA\Experiment\Dataset\Data Sensor Kultivasi\csv"
    r"\sel mikroalga\gabungan_sensor_cell_density_with_image.csv"
)
FEATURE_CSV = Path(
    r"D:\00 Kuliah\TA\Experiment\Dataset\Data Sensor Kultivasi"
    r"\hasil_fitur_konvensional.csv"
)

KAGGLE_INPUT = Path("/kaggle/input")

def find_kaggle_path(filename, is_directory=False):
    matches = [
        path for path in KAGGLE_INPUT.rglob(filename)
        if path.is_dir() == is_directory
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Harus ditemukan tepat satu {filename!r} di {KAGGLE_INPUT}; "
            f"ditemukan {len(matches)}. Atur path input secara eksplisit."
        )
    return matches[0]


if not IMAGE_DIR.is_dir() and KAGGLE_INPUT.is_dir():
    IMAGE_DIR = find_kaggle_path("Gambar Final", is_directory=True)
if not TARGET_CSV.is_file() and KAGGLE_INPUT.is_dir():
    TARGET_CSV = find_kaggle_path(
        "gabungan_sensor_cell_density_with_image.csv"
    )
if not FEATURE_CSV.is_file() and KAGGLE_INPUT.is_dir():
    FEATURE_CSV = find_kaggle_path("hasil_fitur_konvensional.csv")

if Path("/kaggle/working").is_dir():
    OUTPUT_DIR = Path("/kaggle/working/hasil_dl_fusion_random")
else:
    OUTPUT_DIR = Path(
        r"D:\00 Kuliah\TA\Experiment\Model\5 DL\Hasil_Random_Fusion"
    )

IMAGE_COLUMN = "Nama_Gambar"
FEATURE_IMAGE_COLUMN = "file_name"
TARGET_COLUMN = "sensor_1_bga(cells/mL)"
FEATURE_PREFIX = "fitur_"

TEST_RATIO = 0.20
N_SPLITS = 5
EPOCHS = 15
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
SEED = 42
FEATURE_COUNTS = (5, 10, 20, 30, 50)  # Jumlah fitur; semua fitur selalu ditambahkan.
MODEL_NAMES = [
    "Custom CNN 29-Layer",
    "VGG16",
    "EfficientNet-B0",
    "Multilayer ANN",
    "ResNeXt50 32x4d",
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for required_path in (IMAGE_DIR, TARGET_CSV, FEATURE_CSV):
    if not required_path.exists():
        raise FileNotFoundError(f"Input tidak ditemukan: {required_path}")
if not IMAGE_DIR.is_dir():
    raise NotADirectoryError(f"Folder gambar bukan direktori: {IMAGE_DIR}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Device: {DEVICE}")
print(f"Folder gambar: {IMAGE_DIR}")
print(f"CSV target: {TARGET_CSV}")
print(f"CSV fitur tambahan: {FEATURE_CSV}")
print(f"Output: {OUTPUT_DIR}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def parse_target(values):
    """Baca angka Indonesia 425.584,73 maupun desimal titik 425584.73."""
    text = values.astype("string").str.strip()
    decimal_comma = text.str.contains(",", regex=False).fillna(False)
    text.loc[decimal_comma] = (
        text.loc[decimal_comma]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    dot_thousands = (
        ~decimal_comma
        & text.str.fullmatch(r"\d{1,3}(?:\.\d{3})+").fillna(False)
    )
    text.loc[dot_thousands] = text.loc[dot_thousands].str.replace(
        ".", "", regex=False
    )
    return pd.to_numeric(text, errors="coerce").to_numpy(dtype=np.float32)


# ============================================================
# 2. Cocokkan gambar, label target, dan fitur konvensional
# ============================================================

target_df = pd.read_csv(
    TARGET_CSV,
    encoding="utf-8-sig",
    dtype={IMAGE_COLUMN: "string", TARGET_COLUMN: "string"},
)
feature_df = pd.read_csv(
    FEATURE_CSV,
    encoding="utf-8-sig",
    dtype={FEATURE_IMAGE_COLUMN: "string"},
)

required_target_columns = {IMAGE_COLUMN, TARGET_COLUMN}
missing_target_columns = required_target_columns - set(target_df.columns)
if missing_target_columns:
    raise KeyError(f"Kolom target tidak ditemukan: {sorted(missing_target_columns)}")
if FEATURE_IMAGE_COLUMN not in feature_df.columns:
    raise KeyError(f"Kolom nama gambar fitur tidak ditemukan: {FEATURE_IMAGE_COLUMN}")

feature_columns = [
    column for column in feature_df.columns if column.startswith(FEATURE_PREFIX)
]
if not feature_columns:
    raise ValueError(f"Tidak ada kolom fitur berawalan {FEATURE_PREFIX!r}.")

target_df["_image_key"] = target_df[IMAGE_COLUMN].astype("string").str.strip().str.lower()
feature_df["_image_key"] = (
    feature_df[FEATURE_IMAGE_COLUMN].astype("string").str.strip().str.lower()
)
for frame, name_column, description in (
    (target_df, IMAGE_COLUMN, "CSV target"),
    (feature_df, FEATURE_IMAGE_COLUMN, "CSV fitur"),
):
    if frame[name_column].isna().any() or frame[name_column].str.strip().eq("").any():
        raise ValueError(f"Ada nama gambar kosong pada {description}.")
    if frame["_image_key"].duplicated().any():
        raise ValueError(f"Ada nama gambar duplikat pada {description}.")

target_df["target_numeric"] = parse_target(target_df[TARGET_COLUMN])
if not np.isfinite(target_df["target_numeric"].to_numpy()).all():
    raise ValueError(f"Target {TARGET_COLUMN!r} memiliki nilai kosong/tidak valid.")

feature_df[feature_columns] = feature_df[feature_columns].apply(
    pd.to_numeric, errors="coerce"
)
if not np.isfinite(feature_df[feature_columns].to_numpy(dtype=np.float64)).all():
    raise ValueError("Fitur konvensional memiliki nilai kosong/tidak valid.")

image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
image_files = [
    path for path in IMAGE_DIR.iterdir()
    if path.is_file() and path.suffix.lower() in image_extensions
]
image_map = {path.name.strip().lower(): path for path in image_files}
if len(image_map) != len(image_files):
    raise ValueError("Ada nama gambar duplikat pada folder gambar.")

target_keys = set(target_df["_image_key"])
feature_keys = set(feature_df["_image_key"])
image_keys = set(image_map)
missing_images = sorted(target_keys - image_keys)
missing_features = sorted(target_keys - feature_keys)
if missing_images:
    raise FileNotFoundError(
        f"{len(missing_images)} gambar target tidak ada di folder; contoh: "
        f"{missing_images[:5]}"
    )
if missing_features:
    raise ValueError(
        f"{len(missing_features)} gambar target tidak punya fitur; contoh: "
        f"{missing_features[:5]}"
    )

# Hanya pasangan target-gambar-fitur yang lengkap digunakan; gambar ekstra diabaikan.
data = target_df[[IMAGE_COLUMN, "_image_key", "target_numeric"]].merge(
    feature_df[["_image_key", *feature_columns]],
    on="_image_key",
    how="inner",
    validate="one_to_one",
)
data["image_path"] = data["_image_key"].map(lambda key: str(image_map[key]))
data = data.reset_index(drop=True)

X_images = data["image_path"].astype(str).to_numpy()
X_features = data[feature_columns].to_numpy(dtype=np.float32)
y = data["target_numeric"].to_numpy(dtype=np.float32)
sample_names = data[IMAGE_COLUMN].astype(str).to_numpy()

if not np.isfinite(X_features).all() or not np.isfinite(y).all():
    raise ValueError("Input fitur atau target berisi nilai tidak valid.")

print(f"Sampel lengkap yang digunakan: {len(data)}")
print(f"Gambar tanpa target yang diabaikan: {len(image_keys - target_keys)}")
print(f"Jumlah fitur konvensional tersedia: {len(feature_columns)}")
print(f"Target: {TARGET_COLUMN}; rentang {y.min():,.2f}–{y.max():,.2f} sel/mL")


# ============================================================
# 3. Random split 80:20 dan fold CV pada bagian train
# ============================================================

all_indices = np.arange(len(y))
train_indices, test_indices = train_test_split(
    all_indices,
    test_size=max(1, int(np.ceil(TEST_RATIO * len(y)))),
    shuffle=True,
    random_state=SEED,
)
train_images, y_train = X_images[train_indices], y[train_indices]
train_features = X_features[train_indices]
test_images, y_test = X_images[test_indices], y[test_indices]
test_features = X_features[test_indices]
test_names = sample_names[test_indices]

if len(y_train) <= N_SPLITS:
    raise ValueError("Data train tidak cukup untuk 5-fold cross-validation.")
if len(y_test) < 2:
    raise ValueError("Test set memerlukan minimal dua sampel untuk menghitung R².")

feature_counts = sorted(
    {min(count, len(feature_columns)) for count in FEATURE_COUNTS}
    | {len(feature_columns)}
)
fold_splits = list(
    KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(y_train)
)
estimated_cv_runs = len(MODEL_NAMES) * len(feature_counts) * N_SPLITS
print(f"Train: {len(y_train)} | Test: {len(y_test)}")
print(f"Kandidat jumlah fitur: {feature_counts}")
print(f"Perkiraan pelatihan CV: {estimated_cv_runs} model-run + 5 final fit")


# ============================================================
# 4. Dataset gambar + fitur tabular
# ============================================================

class ImageFeatureDataset(Dataset):
    def __init__(self, image_paths, features, labels_scaled, transform):
        self.image_paths = image_paths
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = np.asarray(labels_scaled, dtype=np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        with Image.open(self.image_paths[index]) as image:
            image_tensor = self.transform(image.convert("RGB"))
        feature_tensor = torch.as_tensor(self.features[index], dtype=torch.float32)
        target_tensor = torch.tensor(self.labels[index], dtype=torch.float32)
        return image_tensor, feature_tensor, target_tensor


imagenet_normalize = transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)


def make_transform(model_name):
    image_size = 32 if model_name == "Multilayer ANN" else (
        128 if model_name == "Custom CNN 29-Layer" else 224
    )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            imagenet_normalize,
        ]
    )


def make_loader(image_paths, features, targets_scaled, transform, shuffle):
    dataset = ImageFeatureDataset(image_paths, features, targets_scaled, transform)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )


# ============================================================
# 5. Image encoders dan fusion head untuk lima model
# ============================================================

class CustomCNN29(nn.Module):
    """Image encoder: 13 Conv2d + 13 BatchNorm2d + 3 Linear layers."""

    def __init__(self):
        super().__init__()
        conv_config = [
            (3, 32), (32, 32),
            (32, 64), (64, 64),
            (64, 128), (128, 128), (128, 128),
            (128, 256), (256, 256), (256, 256),
            (256, 512), (512, 512), (512, 512),
        ]
        block_end_indices = {1, 3, 6, 9, 12}
        layers = []
        for index, (in_channels, out_channels) in enumerate(conv_config):
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size=3,
                        padding=1, bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            if index in block_end_indices:
                layers.append(nn.MaxPool2d(kernel_size=2))
        layers.extend(
            [
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
            ]
        )
        self.network = nn.Sequential(*layers)

    def forward(self, images):
        return self.network(images)


class ImageANNEncoder(nn.Module):
    """ANN pada piksel RGB 32 x 32; menghasilkan embedding 128 dimensi."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def forward(self, images):
        return self.network(images)


class FusionRegressor(nn.Module):
    """Gabungkan embedding gambar dan embedding fitur terpilih."""

    def __init__(self, model_name, number_of_features):
        super().__init__()
        self.model_name = model_name

        if model_name == "Custom CNN 29-Layer":
            self.image_encoder = CustomCNN29()
            image_embedding_size = 64
        elif model_name == "Multilayer ANN":
            self.image_encoder = ImageANNEncoder()
            image_embedding_size = 128
        elif model_name == "VGG16":
            backbone = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self.image_encoder = backbone.features
            self.image_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
            self._freeze_image_encoder()
            image_embedding_size = 512
        elif model_name == "EfficientNet-B0":
            backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT
            )
            self.image_encoder = backbone.features
            self.image_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
            self._freeze_image_encoder()
            image_embedding_size = 1280
        elif model_name == "ResNeXt50 32x4d":
            backbone = models.resnext50_32x4d(
                weights=models.ResNeXt50_32X4D_Weights.DEFAULT
            )
            self.image_encoder = nn.Sequential(*list(backbone.children())[:-2])
            self.image_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
            self._freeze_image_encoder()
            image_embedding_size = 2048
        else:
            raise ValueError(f"Nama model tidak dikenal: {model_name}")

        self.feature_encoder = nn.Sequential(
            nn.Linear(number_of_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.regression_head = nn.Sequential(
            nn.Linear(image_embedding_size + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def _freeze_image_encoder(self):
        for parameter in self.image_encoder.parameters():
            parameter.requires_grad = False

    def forward(self, images, features):
        if self.model_name in {"VGG16", "EfficientNet-B0", "ResNeXt50 32x4d"}:
            self.image_encoder.eval()
            with torch.no_grad():
                image_embedding = self.image_encoder(images)
                image_embedding = self.image_pool(image_embedding)
        else:
            image_embedding = self.image_encoder(images)

        feature_embedding = self.feature_encoder(features)
        combined = torch.cat((image_embedding, feature_embedding), dim=1)
        return self.regression_head(combined)


# ============================================================
# 6. Ranking fitur, training, prediksi, dan evaluasi
# ============================================================

def rank_features(fit_features, fit_targets):
    scores, _ = f_regression(fit_features, fit_targets)
    scores = np.nan_to_num(
        scores,
        nan=-np.inf,
        posinf=np.finfo(np.float64).max,
        neginf=-np.inf,
    )
    return np.argsort(scores)[::-1], scores


def evaluate_loss(model, loader, target_std):
    model.eval()
    scaled_absolute_error = 0.0
    absolute_error = 0.0
    sample_count = 0
    with torch.no_grad():
        for images, features, targets_scaled in loader:
            images = images.to(DEVICE, non_blocking=True)
            features = features.to(DEVICE, non_blocking=True)
            targets_scaled = targets_scaled.to(DEVICE, non_blocking=True)
            predictions_scaled = model(images, features).squeeze(1)
            errors = predictions_scaled - targets_scaled
            scaled_absolute_error += torch.abs(errors).sum().item()
            absolute_error += torch.abs(errors * target_std).sum().item()
            sample_count += len(images)
    return scaled_absolute_error / sample_count, absolute_error / sample_count


def fit_one_run(
    model_name,
    feature_count,
    fit_images,
    fit_features,
    fit_targets,
    validation_images=None,
    validation_features=None,
    validation_targets=None,
    seed=SEED,
):
    set_seed(seed)
    transform = make_transform(model_name)

    # Selector, scaler, dan statistik target hanya memakai subset fit saat ini.
    feature_order, feature_scores = rank_features(fit_features, fit_targets)
    selected_count = min(fit_features.shape[1], feature_count)
    selected_indices = feature_order[:selected_count]
    selected_names = [feature_columns[index] for index in selected_indices]

    fit_selected = fit_features[:, selected_indices]
    feature_mean = fit_selected.mean(axis=0)
    feature_scale = fit_selected.std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    fit_scaled = (fit_selected - feature_mean) / feature_scale

    target_mean = float(fit_targets.mean())
    target_std = float(fit_targets.std())
    if target_std < 1e-8:
        target_std = 1.0
    fit_targets_scaled = (fit_targets - target_mean) / target_std
    fit_loader = make_loader(
        fit_images, fit_scaled, fit_targets_scaled, transform, shuffle=True
    )
    fit_eval_loader = make_loader(
        fit_images, fit_scaled, fit_targets_scaled, transform, shuffle=False
    )

    validation_loader = None
    if validation_images is not None:
        validation_selected = validation_features[:, selected_indices]
        validation_scaled = (validation_selected - feature_mean) / feature_scale
        validation_targets_scaled = (validation_targets - target_mean) / target_std
        validation_loader = make_loader(
            validation_images,
            validation_scaled,
            validation_targets_scaled,
            transform,
            shuffle=False,
        )

    network = FusionRegressor(model_name, selected_count).to(DEVICE)
    optimizer = torch.optim.Adam(
        [p for p in network.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )
    loss_function = nn.L1Loss()
    history = []

    for epoch in range(1, EPOCHS + 1):
        network.train()
        for images, features, targets_scaled in fit_loader:
            images = images.to(DEVICE, non_blocking=True)
            features = features.to(DEVICE, non_blocking=True)
            targets_scaled = targets_scaled.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predictions_scaled = network(images, features).squeeze(1)
            loss = loss_function(predictions_scaled, targets_scaled)
            loss.backward()
            optimizer.step()

        train_loss, train_mae = evaluate_loss(network, fit_eval_loader, target_std)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mae": train_mae,
        }
        if validation_loader is not None:
            validation_loss, validation_mae = evaluate_loss(
                network, validation_loader, target_std
            )
            row["validation_loss"] = validation_loss
            row["validation_mae"] = validation_mae
        history.append(row)

    preprocessing = {
        "selected_indices": selected_indices,
        "selected_features": selected_names,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "target_mean": target_mean,
        "target_std": target_std,
        "transform": transform,
    }
    return network, preprocessing, history, feature_scores


def predict(model, image_paths, features, preprocessing):
    selected = features[:, preprocessing["selected_indices"]]
    scaled = (selected - preprocessing["feature_mean"]) / preprocessing["feature_scale"]
    dummy_targets = np.zeros(len(image_paths), dtype=np.float32)
    loader = make_loader(
        image_paths,
        scaled,
        dummy_targets,
        preprocessing["transform"],
        shuffle=False,
    )
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, batch_features, _ in loader:
            images = images.to(DEVICE, non_blocking=True)
            batch_features = batch_features.to(DEVICE, non_blocking=True)
            predictions_scaled = model(images, batch_features).squeeze(1)
            predictions.extend(predictions_scaled.cpu().numpy())
    return (
        np.asarray(predictions, dtype=np.float32) * preprocessing["target_std"]
        + preprocessing["target_mean"]
    )


def calculate_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": r2_score(actual, predicted),
    }


# ============================================================
# 7. Cross-validation semua model dan kandidat jumlah fitur
# ============================================================

cv_epoch_rows = []
cv_fold_metric_rows = []
cv_prediction_rows = []

for model_index, model_name in enumerate(MODEL_NAMES):
    for feature_count in feature_counts:
        print(
            f"\n===== {model_name} | top-{feature_count} fitur | "
            f"{N_SPLITS}-fold CV ====="
        )
        for fold_number, (fold_train_idx, fold_validation_idx) in enumerate(
            fold_splits, start=1
        ):
            network, preprocessing, history, _ = fit_one_run(
                model_name=model_name,
                feature_count=feature_count,
                fit_images=train_images[fold_train_idx],
                fit_features=train_features[fold_train_idx],
                fit_targets=y_train[fold_train_idx],
                validation_images=train_images[fold_validation_idx],
                validation_features=train_features[fold_validation_idx],
                validation_targets=y_train[fold_validation_idx],
                seed=SEED + model_index * 1000 + feature_count * 10 + fold_number,
            )
            cv_epoch_rows.extend(
                {
                    "Model": model_name,
                    "n_features": feature_count,
                    "Fold": fold_number,
                    **item,
                }
                for item in history
            )

            validation_predictions = predict(
                network,
                train_images[fold_validation_idx],
                train_features[fold_validation_idx],
                preprocessing,
            )
            fold_metrics = calculate_metrics(
                y_train[fold_validation_idx], validation_predictions
            )
            cv_fold_metric_rows.append(
                {
                    "Model": model_name,
                    "n_features": feature_count,
                    "Fold": fold_number,
                    **fold_metrics,
                }
            )
            original_indices = train_indices[fold_validation_idx]
            for original_index, actual_value, predicted_value in zip(
                original_indices,
                y_train[fold_validation_idx],
                validation_predictions,
            ):
                cv_prediction_rows.append(
                    {
                        "Model": model_name,
                        "n_features": feature_count,
                        "Fold": fold_number,
                        "Nama_Gambar": sample_names[original_index],
                        "actual": actual_value,
                        "predicted": predicted_value,
                        "residual": actual_value - predicted_value,
                    }
                )

            del network
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()


# ============================================================
# 8. Ringkasan CV dan pemilihan jumlah fitur per model
# ============================================================

cv_epoch_df = pd.DataFrame(cv_epoch_rows)
cv_fold_metrics_df = pd.DataFrame(cv_fold_metric_rows)
cv_predictions_df = pd.DataFrame(cv_prediction_rows)
cv_epoch_summary_df = (
    cv_epoch_df.groupby(["Model", "n_features", "epoch"], as_index=False)
    .agg(
        train_loss_mean=("train_loss", "mean"),
        train_loss_std=("train_loss", "std"),
        validation_loss_mean=("validation_loss", "mean"),
        validation_loss_std=("validation_loss", "std"),
        train_mae_mean=("train_mae", "mean"),
        train_mae_std=("train_mae", "std"),
        validation_mae_mean=("validation_mae", "mean"),
        validation_mae_std=("validation_mae", "std"),
    )
)
cv_feature_summary_df = (
    cv_fold_metrics_df.groupby(["Model", "n_features"], as_index=False)
    .agg(
        MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        R2_mean=("R2", "mean"), R2_std=("R2", "std"),
        n_folds=("Fold", "nunique"),
    )
    .sort_values(["Model", "MAE_mean", "n_features"])
    .reset_index(drop=True)
)
selected_rows = cv_feature_summary_df.loc[
    cv_feature_summary_df.groupby("Model")["MAE_mean"].idxmin()
].sort_values("MAE_mean").reset_index(drop=True)
selected_rows = selected_rows.rename(columns={"n_features": "selected_n_features"})
selected_rows["total_fitur_dipilih"] = selected_rows[
    "selected_n_features"
].astype(int)
best_model_name = selected_rows.iloc[0]["Model"]
selected_count_by_model = dict(
    zip(selected_rows["Model"], selected_rows["selected_n_features"].astype(int))
)

cv_epoch_df.to_csv(
    OUTPUT_DIR / "dl_cv_loss_per_fold_per_epoch.csv", index=False, encoding="utf-8-sig"
)
cv_epoch_summary_df.to_csv(
    OUTPUT_DIR / "dl_cv_loss_mean_per_epoch.csv", index=False, encoding="utf-8-sig"
)
cv_fold_metrics_df.to_csv(
    OUTPUT_DIR / "dl_cv_metrics_per_fold.csv", index=False, encoding="utf-8-sig"
)
cv_predictions_df.to_csv(
    OUTPUT_DIR / "dl_cv_predictions.csv", index=False, encoding="utf-8-sig"
)
cv_feature_summary_df.to_csv(
    OUTPUT_DIR / "dl_cv_feature_count_summary.csv", index=False, encoding="utf-8-sig"
)
selected_rows.to_csv(
    OUTPUT_DIR / "dl_cv_model_summary.csv", index=False, encoding="utf-8-sig"
)
print("\nRingkasan jumlah fitur terbaik per model (berdasarkan MAE CV):")
print(selected_rows.to_string(index=False))


# ============================================================
# 9. Fit final pada 80%, evaluasi train/test, dan simpan fitur terpilih
# ============================================================

final_history_rows = []
final_metric_rows = []
test_predictions = {}
selected_feature_rows = []
best_checkpoint = None

for model_index, model_name in enumerate(MODEL_NAMES):
    selected_count = selected_count_by_model[model_name]
    print(f"\n===== Final fit: {model_name} | top-{selected_count} fitur =====")

    network, preprocessing, history, final_scores = fit_one_run(
        model_name=model_name,
        feature_count=selected_count,
        fit_images=train_images,
        fit_features=train_features,
        fit_targets=y_train,
        seed=SEED + 10000 + model_index,
    )
    final_history_rows.extend(
        {"Model": model_name, "n_features": selected_count, **item}
        for item in history
    )

    train_predictions = predict(
        network, train_images, train_features, preprocessing
    )
    test_predictions_for_model = predict(
        network, test_images, test_features, preprocessing
    )
    test_predictions[model_name] = test_predictions_for_model
    final_metric_rows.extend(
        [
            {
                "Model": model_name,
                "n_features": selected_count,
                "total_fitur_dipilih": selected_count,
                "Data": "Train",
                **calculate_metrics(y_train, train_predictions),
            },
            {
                "Model": model_name,
                "n_features": selected_count,
                "total_fitur_dipilih": selected_count,
                "Data": "Test",
                **calculate_metrics(y_test, test_predictions_for_model),
            },
        ]
    )

    final_order = np.argsort(final_scores)[::-1]
    ranks = np.empty(len(feature_columns), dtype=int)
    ranks[final_order] = np.arange(1, len(feature_columns) + 1)
    selected_set = set(preprocessing["selected_indices"].tolist())
    for feature_index, feature_name in enumerate(feature_columns):
        selected_feature_rows.append(
            {
                "Model": model_name,
                "selected_n_features": selected_count,
                "total_fitur_dipilih": selected_count,
                "feature": feature_name,
                "f_score": final_scores[feature_index],
                "rank": ranks[feature_index],
                "selected": feature_index in selected_set,
            }
        )

    if model_name == best_model_name:
        best_checkpoint = {
            "model_name": model_name,
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in network.state_dict().items()
            },
            "selected_features": preprocessing["selected_features"],
            "selected_indices": preprocessing["selected_indices"],
            "feature_mean": preprocessing["feature_mean"],
            "feature_scale": preprocessing["feature_scale"],
            "target_mean": preprocessing["target_mean"],
            "target_std": preprocessing["target_std"],
            "feature_columns": feature_columns,
            "target_column": TARGET_COLUMN,
            "seed": SEED + 10000 + model_index,
        }

    del network
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

final_history_df = pd.DataFrame(final_history_rows)
final_metrics_df = pd.DataFrame(final_metric_rows)
selected_features_df = pd.DataFrame(selected_feature_rows)
test_prediction_df = pd.DataFrame(
    {"Nama_Gambar": test_names, "actual": y_test}
)
for model_name, predictions in test_predictions.items():
    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    test_prediction_df[f"total_fitur_dipilih_{safe_name}"] = (
        selected_count_by_model[model_name]
    )
    test_prediction_df[f"predicted_{safe_name}"] = predictions
    test_prediction_df[f"residual_{safe_name}"] = y_test - predictions

final_history_df.to_csv(
    OUTPUT_DIR / "dl_final_train_loss_per_epoch.csv", index=False, encoding="utf-8-sig"
)
final_metrics_df.to_csv(
    OUTPUT_DIR / "dl_final_metrics_train_test.csv", index=False, encoding="utf-8-sig"
)
selected_features_df.to_csv(
    OUTPUT_DIR / "dl_selected_features_by_model.csv", index=False, encoding="utf-8-sig"
)
test_prediction_df.to_csv(
    OUTPUT_DIR / "dl_test_predictions_all_models.csv", index=False, encoding="utf-8-sig"
)
torch.save(best_checkpoint, OUTPUT_DIR / "dl_best_model_by_cv.pth")


# ============================================================
# 10. Ringkasan evaluasi test
# ============================================================

test_metrics_df = pd.DataFrame(
    [
        {
            "Model": model_name,
            "n_features": selected_count_by_model[model_name],
            "total_fitur_dipilih": selected_count_by_model[model_name],
            **calculate_metrics(y_test, test_predictions[model_name]),
        }
        for model_name in MODEL_NAMES
    ]
)
test_metrics_df.to_csv(
    OUTPUT_DIR / "dl_test_metrics_all_models.csv", index=False, encoding="utf-8-sig"
)

print("\nEvaluasi final train dan test:")
print(
    final_metrics_df.pivot(
        index="Model", columns="Data", values=["MAE", "RMSE", "R2"]
    ).round(3).to_string()
)
print("\nJumlah fitur terpilih per model:")
print(selected_rows[["Model", "selected_n_features", "MAE_mean"]].to_string(index=False))
print(f"Model terbaik berdasarkan rata-rata MAE CV: {best_model_name}")
print(f"CSV dan checkpoint tersimpan di: {OUTPUT_DIR}")
