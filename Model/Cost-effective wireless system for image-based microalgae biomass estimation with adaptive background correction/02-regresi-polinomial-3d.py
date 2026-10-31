"""Prediksi densitas dari rerata RGB gambar dengan regresi polinomial orde 2.

Fitur RGB mentah gambar digunakan sebagai proksi warna sampel. Koreksi
latar belakang tidak diterapkan karena nilai acuan putih tidak tersedia.
"""

from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

FEATURES_CSV = SCRIPT_DIR / "fitur_rgb_sampel.csv"
LABELS_CSV = (
    PROJECT_ROOT
    / "Dataset"
    / "Data Sensor Kultivasi"
    / "csv"
    / "gabungan_dengan_jumlah_sel.csv"
)
OUTPUT_DIR = SCRIPT_DIR / "hasil_regresi_polinomial_rgb"

IMAGE_COLUMN_FEATURES = "image_path"
IMAGE_COLUMN_LABELS = "Nama_Gambar"
TARGET_COLUMN = "Jumlah Sel (sel/mL)"
INPUT_FEATURES = [
    "R_mean", "G_mean", "B_mean",
    "R_std", "G_std", "B_std",
]

TEST_SIZE = 0.20
N_SPLITS = 5
RANDOM_STATE = 42


def parse_numeric_target(value) -> float:
    """Baca target angka, termasuk pemisah ribuan titik atau koma."""
    text = str(value).strip().replace(" ", "")

    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", text):
        text = text.replace(".", "").replace(",", "")
    else:
        text = text.replace(",", ".")

    return float(text)


def image_key(value) -> str:
    """Samakan kunci gambar berdasarkan nama file tanpa ekstensi."""
    normalized = str(value).replace("\\", "/").strip()
    return Path(normalized).stem.lower()


def main() -> None:
    if not FEATURES_CSV.is_file():
        raise FileNotFoundError(
            f"CSV fitur RGB tidak ditemukan: {FEATURES_CSV}. "
            "Jalankan 01 terlebih dahulu."
        )

    if not LABELS_CSV.is_file():
        raise FileNotFoundError(f"CSV target tidak ditemukan: {LABELS_CSV}")

    color_data = pd.read_csv(FEATURES_CSV, encoding="utf-8-sig")
    labels = pd.read_csv(
        LABELS_CSV,
        encoding="utf-8-sig",
        dtype={IMAGE_COLUMN_LABELS: "string", TARGET_COLUMN: "string"},
    )

    missing_features = {
        IMAGE_COLUMN_FEATURES,
        *INPUT_FEATURES,
    } - set(color_data.columns)

    missing_labels = {
        IMAGE_COLUMN_LABELS,
        TARGET_COLUMN,
    } - set(labels.columns)

    if missing_features:
        raise KeyError(f"Kolom fitur tidak ditemukan: {sorted(missing_features)}")

    if missing_labels:
        raise KeyError(f"Kolom target tidak ditemukan: {sorted(missing_labels)}")

    color_data["_key"] = color_data[IMAGE_COLUMN_FEATURES].map(image_key)
    labels["_key"] = labels[IMAGE_COLUMN_LABELS].map(image_key)

    if color_data["_key"].duplicated().any():
        raise ValueError("Nama gambar duplikat ditemukan pada CSV fitur RGB.")

    if labels["_key"].duplicated().any():
        raise ValueError("Nama gambar duplikat ditemukan pada CSV target.")

    labels["_target"] = labels[TARGET_COLUMN].map(parse_numeric_target)

    data = color_data.merge(
        labels[["_key", "_target"]],
        on="_key",
        how="inner",
        validate="one_to_one",
    )

    if data.empty:
        raise ValueError(
            "Tidak ada nama gambar yang cocok antara CSV fitur dan CSV target."
        )

    X = data[INPUT_FEATURES].to_numpy(dtype=np.float64)
    y = data["_target"].to_numpy(dtype=np.float64)

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Fitur RGB atau target mengandung nilai tidak valid.")

    if len(y) < N_SPLITS + 1:
        raise ValueError(
            f"Diperlukan minimal {N_SPLITS + 1} sampel agar train/test split "
            f"dan {N_SPLITS}-fold CV dapat dilakukan."
        )

    indices = np.arange(len(y))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    def make_model() -> Pipeline:
        """Buat pipeline regresi yang sama untuk CV dan fit final."""
        return Pipeline([
            ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
            ("regression", LinearRegression()),
        ])

    # CV hanya memakai 80% train. Setiap fold membuat pipeline baru agar
    # validasi fold tidak ikut digunakan saat fit.
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_rows = []
    for fold_number, (fit_indices, validation_indices) in enumerate(
        cv.split(X_train), start=1
    ):
        fold_model = make_model()
        fold_model.fit(X_train[fit_indices], y_train[fit_indices])
        validation_prediction = fold_model.predict(X_train[validation_indices])
        y_validation = y_train[validation_indices]
        cv_rows.append({
            "fold": fold_number,
            "n_fit": len(fit_indices),
            "n_validation": len(validation_indices),
            "MAE": mean_absolute_error(y_validation, validation_prediction),
            "RMSE": np.sqrt(mean_squared_error(y_validation, validation_prediction)),
            "R2": r2_score(y_validation, validation_prediction),
        })

    # Fit ulang model identik pada seluruh 80% train untuk evaluasi final.
    model = make_model()
    model.fit(X_train, y_train)

    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cv_metrics = pd.DataFrame(cv_rows)
    cv_metrics.to_csv(
        OUTPUT_DIR / "hasil_cv_per_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cv_summary = cv_metrics[["MAE", "RMSE", "R2"]].agg(["mean", "std"])
    cv_summary.index.name = "statistic"
    cv_summary.to_csv(
        OUTPUT_DIR / "hasil_cv_ringkasan.csv",
        encoding="utf-8-sig",
    )

    test_predictions = pd.DataFrame({
        "nama_file": data.iloc[test_indices][IMAGE_COLUMN_FEATURES].to_numpy(),
        "actual": y_test,
        "predicted": test_prediction,
        "residual": y_test - test_prediction,
    })
    test_predictions.to_csv(
        OUTPUT_DIR / "prediksi_test_semua_model.csv",
        index=False,
        encoding="utf-8-sig",
    )

    polynomial = model.named_steps["polynomial"]
    regression = model.named_steps["regression"]
    term_names = polynomial.get_feature_names_out(INPUT_FEATURES)

    pd.DataFrame({
        "term": term_names,
        "coefficient": regression.coef_,
    }).to_csv(
        OUTPUT_DIR / "koefisien_polinomial.csv",
        index=False,
        encoding="utf-8-sig",
    )

    train_test_metrics = pd.DataFrame([
        {
            "Model": "Regresi Polinomial Orde 2",
            "Data": "Train",
            "MAE": mean_absolute_error(y_train, train_prediction),
            "RMSE": np.sqrt(mean_squared_error(y_train, train_prediction)),
            "R2": r2_score(y_train, train_prediction),
        },
        {
            "Model": "Regresi Polinomial Orde 2",
            "Data": "Test",
            "MAE": mean_absolute_error(y_test, test_prediction),
            "RMSE": np.sqrt(mean_squared_error(y_test, test_prediction)),
            "R2": r2_score(y_test, test_prediction),
        },
    ])
    train_test_metrics.to_csv(
        OUTPUT_DIR / "hasil_evaluasi_train_test_semua_model.csv",
        index=False,
        encoding="utf-8-sig",
    )

    joblib.dump(
        model,
        OUTPUT_DIR / "model_regresi_polinomial_rgb.joblib",
    )

    print(
        f"Sampel cocok: {len(data)} "
        f"(train={len(train_indices)}, test={len(test_indices)})"
    )
    print(
        f"5-fold CV (train saja) MAE: {cv_metrics['MAE'].mean():.6g} "
        f"± {cv_metrics['MAE'].std():.6g}"
    )
    print(
        f"5-fold CV (train saja) RMSE: {cv_metrics['RMSE'].mean():.6g} "
        f"± {cv_metrics['RMSE'].std():.6g}"
    )
    print(
        f"5-fold CV (train saja) R²: {cv_metrics['R2'].mean():.6g} "
        f"± {cv_metrics['R2'].std():.6g}"
    )
    test_row = train_test_metrics[train_test_metrics["Data"] == "Test"].iloc[0]
    print(f"Test MAE: {test_row['MAE']:.6g}")
    print(f"Test RMSE: {test_row['RMSE']:.6g}")
    print(f"Test R²: {test_row['R2']:.6g}")
    print(f"Hasil tersimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
