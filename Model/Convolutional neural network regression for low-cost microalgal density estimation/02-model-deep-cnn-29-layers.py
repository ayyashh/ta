"""Train CNN moving-window, jalankan 5-fold CV, dan simpan laporan DL."""

from pathlib import Path
import gc
from zipfile import BadZipFile, ZipFile

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# 1. Konfigurasi dan deteksi GPU
# ============================================================

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
IS_KAGGLE = KAGGLE_INPUT.is_dir()

SEED = 42
WINDOW_SIZE = 32
STRIDE = 20
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
N_SPLITS = 5

LOSS_NAME = "mse"  # Pilihan: "mse", "mae", atau "huber".
if LOSS_NAME not in {"mse", "mae", "huber"}:
    raise ValueError("LOSS_NAME harus 'mse', 'mae', atau 'huber'.")

MODEL_NAME = f"Deep CNN 29 Layers ({LOSS_NAME.upper()})"
MANIFEST_FILENAME = "moving_window_manifest.csv"

physical_gpus = tf.config.list_physical_devices("GPU")

if physical_gpus:
    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as error:
            print(f"Tidak dapat mengatur memory growth untuk {gpu}: {error}")

    TF_DEVICE = "/GPU:0"
    print(f"GPU terdeteksi: {len(physical_gpus)}")
    for gpu in physical_gpus:
        print(f"  {gpu}")
else:
    TF_DEVICE = "/CPU:0"
    print("GPU tidak terdeteksi oleh TensorFlow; model akan memakai CPU.")


# ============================================================
# 2. Tentukan lokasi input dan output
# ============================================================

if IS_KAGGLE:
    # Input selalu berasal dari dataset yang dipasang di /kaggle/input.
    # Working hanya dipakai untuk hasil model dan ekstraksi ZIP sementara.
    OUTPUT_DIR = KAGGLE_WORKING / "CNN_Moving_Window"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_manifests = list(KAGGLE_INPUT.rglob(MANIFEST_FILENAME))

    if len(input_manifests) == 1:
        MANIFEST_PATH = input_manifests[0]
        INPUT_DATA_DIR = MANIFEST_PATH.parent
    elif len(input_manifests) > 1:
        raise FileNotFoundError(
            f"Pasang tepat satu dataset yang berisi {MANIFEST_FILENAME}; "
            f"ditemukan {len(input_manifests)} manifest di {KAGGLE_INPUT}."
        )
    else:
        # Jika Kaggle Dataset berisi ZIP, ekstrak satu arsip yang memuat manifest.
        matching_archives = []
        for archive_path in KAGGLE_INPUT.rglob("*.zip"):
            try:
                with ZipFile(archive_path) as archive:
                    manifest_members = [
                        name for name in archive.namelist()
                        if Path(name).name == MANIFEST_FILENAME
                    ]
                    if manifest_members:
                        matching_archives.append(archive_path)
            except BadZipFile:
                continue

        if len(matching_archives) != 1:
            raise FileNotFoundError(
                f"Tidak ditemukan satu dataset unik dengan {MANIFEST_FILENAME} "
                f"di {KAGGLE_INPUT}; manifest langsung={len(input_manifests)}, "
                f"arsip cocok={len(matching_archives)}."
            )

        extracted_input_dir = OUTPUT_DIR / "input_dataset"
        extracted_input_dir.mkdir(parents=True, exist_ok=True)
        with ZipFile(matching_archives[0]) as archive:
            root = extracted_input_dir.resolve()
            for member in archive.infolist():
                destination = (root / member.filename).resolve()
                if destination != root and root not in destination.parents:
                    raise ValueError(
                        f"Path tidak aman di dalam arsip: {member.filename}"
                    )
            archive.extractall(root)

        extracted_manifests = list(
            extracted_input_dir.rglob(MANIFEST_FILENAME)
        )
        if len(extracted_manifests) != 1:
            raise FileNotFoundError(
                "Arsip dataset harus berisi tepat satu "
                f"{MANIFEST_FILENAME}; ditemukan {len(extracted_manifests)}."
            )
        MANIFEST_PATH = extracted_manifests[0]
        INPUT_DATA_DIR = MANIFEST_PATH.parent

else:
    # Saat lokal, Kode 1 menyimpan manifest dan patch di folder ini.
    SCRIPT_DIR = Path(__file__).resolve().parent
    INPUT_DATA_DIR = SCRIPT_DIR / "CNN_Moving_Window"
    OUTPUT_DIR = INPUT_DATA_DIR
    MANIFEST_PATH = INPUT_DATA_DIR / MANIFEST_FILENAME
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


if not MANIFEST_PATH.is_file():
    raise FileNotFoundError(
        f"Manifest tidak ditemukan: {MANIFEST_PATH}. Jalankan Kode 1 terlebih dahulu."
    )

OUTPUT_DIR = OUTPUT_DIR / f"loss_{LOSS_NAME}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Mode: {'Kaggle' if IS_KAGGLE else 'Lokal'}")
print(f"Manifest input: {MANIFEST_PATH}")
print(f"Folder patch input: {INPUT_DATA_DIR}")
print(f"Folder output: {OUTPUT_DIR}")
print(f"Perangkat training: {TF_DEVICE}")

manifest = pd.read_csv(MANIFEST_PATH, encoding="utf-8-sig")

required_columns = {
    "patch_path",
    "source_image",
    "Nama_Gambar",
    "split",
    "target_density",
}
missing_columns = required_columns - set(manifest.columns)

if missing_columns:
    raise KeyError(
        f"Kolom manifest tidak ditemukan: {sorted(missing_columns)}"
    )


# ============================================================
# 3. Definisi model CNN 29 layer
# ============================================================

def build_model():
    inputs = keras.Input(
        shape=(WINDOW_SIZE, WINDOW_SIZE, 3),
        name="rgb_patch",
    )
    x = inputs

    # Block 1: 3 Conv2D (16 filter), masing-masing dengan BatchNorm, lalu pooling.
    for index in range(1, 4):
        x = layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
            activation="relu",
            name=f"block1_conv{index}",
        )(x)
        x = layers.BatchNormalization(name=f"block1_bn{index}")(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=2,
        name="block1_pool",
    )(x)

    # Block 2: 3 Conv2D (32 filter), masing-masing dengan BatchNorm, lalu pooling.
    for index in range(1, 4):
        x = layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
            activation="relu",
            name=f"block2_conv{index}",
        )(x)
        x = layers.BatchNormalization(name=f"block2_bn{index}")(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=2,
        name="block2_pool",
    )(x)

    # Block 3: 3 Conv2D (64 filter), masing-masing dengan BatchNorm, lalu pooling.
    for index in range(1, 4):
        x = layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
            activation="relu",
            name=f"block3_conv{index}",
        )(x)
        x = layers.BatchNormalization(name=f"block3_bn{index}")(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=2,
        name="block3_pool",
    )(x)

    # Bottleneck 1x1 dengan ReLU dan BatchNorm, lalu dropout.
    x = layers.Conv2D(
        64,
        kernel_size=1,
        padding="same",
        activation="relu",
        name="bottleneck_64",
    )(x)
    x = layers.BatchNormalization(name="bottleneck_64_bn")(x)
    x = layers.Dropout(0.20, name="dropout_64")(x)

    x = layers.Conv2D(
        16,
        kernel_size=1,
        padding="same",
        activation="relu",
        name="bottleneck_16",
    )(x)
    x = layers.BatchNormalization(name="bottleneck_16_bn")(x)
    x = layers.Dropout(0.20, name="dropout_16")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(64, activation="relu", name="dense_64")(x)
    outputs = layers.Dense(1, activation="linear", name="density")(x)

    model = keras.Model(inputs, outputs, name="deep_cnn_29_layers")

    # Metodologi menghitung 29 lapisan utama (11 Conv, 11 BN, 3 Pool,
    # 2 Dropout, 2 Dense); Keras juga menghitung InputLayer dan Flatten.
    if len(model.layers) != 31:
        raise RuntimeError(
            f"Jumlah layer Keras={len(model.layers)}, seharusnya 31 "
            "(29 lapisan utama + InputLayer + Flatten)."
        )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={
            "mse": "mean_squared_error",
            "mae": "mean_absolute_error",
            "huber": keras.losses.Huber(),
        }[LOSS_NAME],
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


# ============================================================
# 4. Fungsi memuat patch dan menghitung metrik
# ============================================================

def load_source_patches(source_names, source_manifest):
    """Muat patch yang berasal dari gambar sumber tertentu."""
    subset = source_manifest[
        source_manifest["source_image"].isin(source_names)
    ].copy().reset_index(drop=True)

    images = []

    for relative_path in subset["patch_path"]:
        patch_path = INPUT_DATA_DIR / Path(relative_path)

        if not patch_path.is_file():
            raise FileNotFoundError(f"Patch tidak ditemukan: {patch_path}")

        image_bgr = cv2.imread(str(patch_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Patch tidak dapat dibaca: {patch_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        images.append(image_rgb)

    if not images:
        raise ValueError(
            f"Tidak ada patch untuk {len(source_names)} gambar sumber."
        )

    X = np.asarray(images, dtype=np.float32) / 255.0
    # Jaringan mempelajari target dalam juta sel/mL.
    y = subset["target_density"].to_numpy(dtype=np.float32) / 1_000_000.0

    return subset, X, y


def aggregate_predictions(subset, patch_predictions):
    """Rata-ratakan prediksi patch menjadi satu prediksi per gambar sumber."""
    result = subset[
        ["source_image", "Nama_Gambar", "target_density"]
    ].copy()

    result["predicted"] = patch_predictions

    return (
        result.groupby(
            ["source_image", "Nama_Gambar"],
            as_index=False,
        )
        .agg(
            actual=("target_density", "first"),
            predicted=("predicted", "mean"),
            patch_count=("predicted", "size"),
        )
    )


def calculate_metrics(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


# ============================================================
# 5. 5-fold CV sebagai evaluasi tambahan
# ============================================================

# Ikuti split tunggal yang dibuat preprocessing pada gambar sumber.
all_sources = manifest["source_image"].drop_duplicates().to_numpy()
split_per_source = manifest[["source_image", "split"]].drop_duplicates()
if split_per_source["source_image"].duplicated().any():
    raise ValueError("Satu gambar sumber muncul pada lebih dari satu split.")
if not set(split_per_source["split"].unique()).issubset({"train", "test"}):
    raise ValueError(
        "Manifest harus berisi split 'train' dan 'test' saja. "
        "Jalankan ulang preprocessing 01 untuk memperbarui manifest."
    )

train_sources = (
    split_per_source.loc[split_per_source["split"] == "train", "source_image"]
    .to_numpy()
)
test_sources = (
    split_per_source.loc[split_per_source["split"] == "test", "source_image"]
    .to_numpy()
)
if len(train_sources) + len(test_sources) != len(all_sources):
    raise ValueError("Sebagian gambar sumber tidak masuk ke train atau test.")

train_source_manifest = manifest[
    manifest["source_image"].isin(train_sources)
].copy()

if len(train_sources) < N_SPLITS:
    raise ValueError(
        f"Perlu minimal {N_SPLITS} gambar sumber train untuk CV; "
        f"ditemukan {len(train_sources)}."
    )

cv_metric_rows = []
cv_prediction_rows = []
cv_epoch_rows = []

kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED,
)

for fold_number, (outer_train_idx, outer_valid_idx) in enumerate(
    kf.split(train_sources),
    start=1,
):
    print(f"\nCV fold {fold_number}/{N_SPLITS}")

    outer_train_sources = train_sources[outer_train_idx]
    outer_valid_sources = train_sources[outer_valid_idx]

    fit_manifest, X_fit, y_fit = load_source_patches(
        outer_train_sources,
        train_source_manifest,
    )
    outer_val_manifest, X_outer_val, y_outer_val = load_source_patches(
        outer_valid_sources,
        train_source_manifest,
    )

    y_fit_scaled = y_fit
    y_outer_val_scaled = y_outer_val

    # Seed CV terpisah; hasilnya tidak dipakai untuk mengubah training final.
    tf.keras.utils.set_random_seed(SEED + fold_number)

    with tf.device(TF_DEVICE):
        fold_model = build_model()
        fold_history = fold_model.fit(
            X_fit,
            y_fit_scaled,
            validation_data=(X_outer_val, y_outer_val_scaled),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            shuffle=True,
            verbose=0,
        )

    # Simpan riwayat loss/MAE per fold dan epoch.
    for epoch_index in range(len(fold_history.history["loss"])):
        cv_epoch_rows.append({
            "Model": MODEL_NAME,
            "Fold": fold_number,
            "epoch": epoch_index + 1,
            "train_loss": fold_history.history["loss"][epoch_index],
            "train_mae": fold_history.history["mae"][epoch_index],
            "validation_loss": fold_history.history["val_loss"][epoch_index],
            "validation_mae": fold_history.history["val_mae"][epoch_index],
        })

    # Evaluasi outer fold validation; prediksi patch dikembalikan ke sel/mL.
    outer_pred_scaled = fold_model.predict(
        X_outer_val,
        batch_size=BATCH_SIZE,
        verbose=0,
    ).reshape(-1)

    outer_pred = outer_pred_scaled * 1_000_000.0
    source_predictions = aggregate_predictions(outer_val_manifest, outer_pred)

    fold_scores = calculate_metrics(
        source_predictions["actual"].to_numpy(),
        source_predictions["predicted"].to_numpy(),
    )

    cv_metric_rows.append({
        "Model": MODEL_NAME,
        "Fold": fold_number,
        **fold_scores,
    })

    for _, row in source_predictions.iterrows():
        cv_prediction_rows.append({
            "Model": MODEL_NAME,
            "Fold": fold_number,
            "Nama_Gambar": row["Nama_Gambar"],
            "actual": row["actual"],
            "predicted": row["predicted"],
            "residual": row["actual"] - row["predicted"],
            "patch_count": row["patch_count"],
        })

    del fold_model
    tf.keras.backend.clear_session()
    gc.collect()


cv_metrics_df = pd.DataFrame(cv_metric_rows)

cv_summary_df = pd.DataFrame([{
    "Model": MODEL_NAME,
    "MAE_mean": cv_metrics_df["MAE"].mean(),
    "MAE_std": cv_metrics_df["MAE"].std(ddof=1),
    "RMSE_mean": cv_metrics_df["RMSE"].mean(),
    "RMSE_std": cv_metrics_df["RMSE"].std(ddof=1),
    "R2_mean": cv_metrics_df["R2"].mean(),
    "R2_std": cv_metrics_df["R2"].std(ddof=1),
    "n_folds": len(cv_metrics_df),
}])

cv_epoch_df = pd.DataFrame(cv_epoch_rows)

if not cv_epoch_df.empty:
    cv_epoch_summary_df = (
        cv_epoch_df.groupby(["Model", "epoch"], as_index=False)
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
else:
    cv_epoch_summary_df = pd.DataFrame()

cv_metrics_df.to_csv(
    OUTPUT_DIR / "dl_cv_metrics_per_fold.csv",
    index=False,
    encoding="utf-8-sig",
)
cv_summary_df.to_csv(
    OUTPUT_DIR / "dl_cv_model_summary.csv",
    index=False,
    encoding="utf-8-sig",
)
pd.DataFrame(cv_prediction_rows).to_csv(
    OUTPUT_DIR / "dl_cv_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)
cv_epoch_df.to_csv(
    OUTPUT_DIR / "dl_cv_loss_per_fold_per_epoch.csv",
    index=False,
    encoding="utf-8-sig",
)
cv_epoch_summary_df.to_csv(
    OUTPUT_DIR / "dl_cv_loss_mean_per_epoch.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nSkor CV setiap fold:")
print(cv_metrics_df.to_string(index=False))

print("\nRingkasan CV (mean ± std antar-fold):")
for metric in ("MAE", "RMSE", "R2"):
    print(
        f"{metric}: "
        f"{cv_summary_df.loc[0, f'{metric}_mean']:.4g} ± "
        f"{cv_summary_df.loc[0, f'{metric}_std']:.4g}"
    )


# ============================================================
# 6. Fit final pada seluruh 80% train selama 15 epoch
# ============================================================

FINAL_EPOCHS = EPOCHS
print(f"Epoch CV dan fit final: {EPOCHS}")

train_manifest, X_train, y_train = load_source_patches(
    train_sources, manifest
)
test_manifest, X_test, y_test = load_source_patches(
    test_sources, manifest
)

# y_train sudah dinyatakan dalam juta sel/mL oleh load_source_patches.
y_train_scaled = y_train

# Reset seed supaya CV tambahan tidak mengubah seed training final.
tf.keras.utils.set_random_seed(SEED)

model_path = OUTPUT_DIR / "dl_final_model.keras"

with tf.device(TF_DEVICE):
    model = build_model()
    final_history = model.fit(
        X_train,
        y_train_scaled,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        shuffle=True,
        verbose=1,
    )
    model.save(model_path)

# Simpan riwayat training final.
final_history_df = pd.DataFrame(final_history.history)
final_history_df.insert(0, "epoch", np.arange(1, len(final_history_df) + 1))
final_history_df.insert(0, "Model", MODEL_NAME)

final_history_df.to_csv(
    OUTPUT_DIR / "dl_final_train_loss_per_epoch.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 7. Evaluasi final per gambar sumber
# ============================================================

def predict_per_source(subset, X):
    predicted_scaled = model.predict(
        X,
        batch_size=BATCH_SIZE,
        verbose=0,
    ).reshape(-1)

    predicted = predicted_scaled * 1_000_000.0
    result = aggregate_predictions(subset, predicted)
    result["residual"] = result["actual"] - result["predicted"]
    return result


train_predictions = predict_per_source(train_manifest, X_train)
test_predictions = predict_per_source(test_manifest, X_test)

metric_rows = []

for split_name, prediction_df in [
    ("Train", train_predictions),
    ("Test", test_predictions),
]:
    metric_rows.append({
        "Model": MODEL_NAME,
        "Data": split_name,
        **calculate_metrics(
            prediction_df["actual"].to_numpy(),
            prediction_df["predicted"].to_numpy(),
        ),
    })

all_metrics_df = pd.DataFrame(metric_rows)

# Format utama sesuai laporan DL: train dan test.
final_metrics_df = all_metrics_df[
    all_metrics_df["Data"].isin(["Train", "Test"])
].copy()

final_metrics_df.to_csv(
    OUTPUT_DIR / "dl_final_metrics_train_test.csv",
    index=False,
    encoding="utf-8-sig",
)

final_metrics_df[final_metrics_df["Data"] == "Test"][
    ["Model", "MAE", "RMSE", "R2"]
].to_csv(
    OUTPUT_DIR / "dl_test_metrics_all_models.csv",
    index=False,
    encoding="utf-8-sig",
)

# Satu baris per gambar test; nilai prediksi sudah dirata-ratakan dari patch.
test_report_df = test_predictions.rename(columns={
    "predicted": "predicted_deep_cnn_29_layers",
    "residual": "residual_deep_cnn_29_layers",
})

test_report_df.to_csv(
    OUTPUT_DIR / "dl_test_predictions_all_models.csv",
    index=False,
    encoding="utf-8-sig",
)

pd.DataFrame([{
    "target_scale_divisor": 1_000_000.0,
    "window_size": WINDOW_SIZE,
    "stride": STRIDE,
    "seed": SEED,
    "n_splits": N_SPLITS,
    "train_source_count": len(train_sources),
    "test_source_count": len(test_sources),
    "train_source_ratio": len(train_sources) / len(all_sources),
    "test_source_ratio": len(test_sources) / len(all_sources),
    "final_epochs": FINAL_EPOCHS,
    "epochs_per_run": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "loss_name": LOSS_NAME,
    "device": TF_DEVICE,
}]).to_csv(
    OUTPUT_DIR / "dl_training_metadata.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 8. Ringkasan proses dan pemeriksaan penggunaan GPU
# ============================================================

print("\nEvaluasi final train/test:")
print(final_metrics_df.to_string(index=False))

print("\nFile utama:")
print(f"- CV per fold: {OUTPUT_DIR / 'dl_cv_metrics_per_fold.csv'}")
print(f"- Ringkasan CV: {OUTPUT_DIR / 'dl_cv_model_summary.csv'}")
print(f"- Metrik final: {OUTPUT_DIR / 'dl_final_metrics_train_test.csv'}")
print(f"- Prediksi test: {OUTPUT_DIR / 'dl_test_predictions_all_models.csv'}")
print(f"- Model final: {model_path}")

if physical_gpus:
    try:
        gpu_memory = tf.config.experimental.get_memory_info("GPU:0")
        print(
            "\nMemori GPU TensorFlow — "
            f"saat ini: {gpu_memory['current'] / (1024 ** 2):.1f} MiB, "
            f"puncak: {gpu_memory['peak'] / (1024 ** 2):.1f} MiB"
        )
    except (ValueError, RuntimeError, AttributeError) as error:
        print(f"Tidak bisa membaca statistik memori GPU: {error}")
