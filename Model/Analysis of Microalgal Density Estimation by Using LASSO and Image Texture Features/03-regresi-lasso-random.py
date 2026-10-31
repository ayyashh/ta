"""LASSO S20 dengan split acak dan format laporan 5 ML."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Kaggle: cari file pada seluruh dataset terpasang.
# Lokal: cari file di folder kerja skrip.
SCRIPT_DIR = Path.cwd()
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

FEATURES_FILENAME = "fitur_skenario_S20.csv"
LABELS_FILENAME = "gabungan_dengan_jumlah_sel.csv"

if KAGGLE_INPUT.is_dir():
    feature_candidates = list(KAGGLE_INPUT.rglob(FEATURES_FILENAME))
    label_candidates = list(KAGGLE_INPUT.rglob(LABELS_FILENAME))

    if len(feature_candidates) != 1:
        raise FileNotFoundError(
            f"Harus ditemukan tepat satu {FEATURES_FILENAME} di {KAGGLE_INPUT}; "
            f"ditemukan {len(feature_candidates)}."
        )
    if len(label_candidates) != 1:
        raise FileNotFoundError(
            f"Harus ditemukan tepat satu {LABELS_FILENAME} di {KAGGLE_INPUT}; "
            f"ditemukan {len(label_candidates)}."
        )

    FEATURES_CSV = feature_candidates[0]
    LABELS_CSV = label_candidates[0]
    OUTPUT_DIR = KAGGLE_WORKING / "Hasil" / "Random"
else:
    FEATURES_CSV = SCRIPT_DIR / FEATURES_FILENAME
    LABELS_CSV = SCRIPT_DIR / LABELS_FILENAME
    OUTPUT_DIR = SCRIPT_DIR / "Hasil" / "Random"


FEATURE_IMAGE_COLUMN = "image_path"
LABEL_IMAGE_COLUMN = "Nama_Gambar"
TARGET_COLUMN = "Jumlah Sel (sel/mL)"

TEST_RATIO = 0.20
N_SPLITS = 5
RANDOM_STATE = 42

# Alpha diuji pada target yang distandardisasi di dalam setiap fit. Kandidat
# 0.001 tetap dimasukkan, tetapi interpretasinya hanya sebanding dengan paper
# jika paper juga menstandardisasi target dengan cara yang sama.
ALPHA_VALUES = np.concatenate(([0.0], np.geomspace(1e-3, 10.0, 99)))
ALPHA_VALUES = np.unique(np.append(ALPHA_VALUES, 1e-3))
CV_FOLD_COUNTS = (5, 10, 20)

S20_FEATURES = [
    "R_mean",
    "G_mean",
    "B_mean",
    "R_entropy",
    "G_entropy",
    "B_entropy",
    "R_entropy_squared",
    "G_entropy_squared",
    "B_entropy_squared",
    "R_entropy_x_G_entropy",
    "R_entropy_x_B_entropy",
    "G_entropy_x_B_entropy",
]


def image_key(value) -> str:
    """Cocokkan nama gambar tanpa memperhatikan folder atau ekstensi."""
    normalized = str(value).replace("\\", "/")
    return Path(normalized).stem.strip().lower()


def parse_target(values: pd.Series) -> np.ndarray:
    """Mendukung target numerik dan format ribuan Indonesia seperti 1.234."""
    text = values.astype("string").str.strip()
    thousands = text.str.fullmatch(r"\d{1,3}(?:\.\d{3})+").fillna(False)
    text.loc[thousands] = text.loc[thousands].str.replace(".", "", regex=False)
    return pd.to_numeric(text, errors="coerce").to_numpy(dtype=np.float64)


def make_estimator(alpha: float) -> TransformedTargetRegressor:
    """Skalakan X dan y hanya dari data fit; prediksi dikembalikan ke unit asli."""
    if np.isclose(alpha, 0.0):
        regressor = LinearRegression()
    else:
        # Toleransi 1e-7 terlalu ketat untuk target berskala besar dan fitur
        # yang saling berkorelasi; beri ruang iterasi lebih banyak agar solver
        # coordinate descent dapat mencapai konvergensi.
        regressor = Lasso(
            alpha=float(alpha),
            max_iter=500_000,
            tol=1e-5,
        )

    feature_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", regressor),
    ])
    return TransformedTargetRegressor(
        regressor=feature_pipeline,
        transformer=StandardScaler(),
    )


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def select_alpha(X: np.ndarray, y: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Pilih alpha dengan MSE terendah lintas 5-, 10-, dan 20-fold acak."""
    valid_fold_counts = [folds for folds in CV_FOLD_COUNTS if len(y) > folds]
    if not valid_fold_counts:
        raise ValueError(
            f"Data untuk pencarian alpha hanya {len(y)} sampel; "
            f"diperlukan lebih dari {min(CV_FOLD_COUNTS)}."
        )

    cv_splits_by_fold = {
        folds: list(KFold(
            n_splits=folds,
            shuffle=True,
            random_state=RANDOM_STATE,
        ).split(X))
        for folds in valid_fold_counts
    }
    rows = []

    for alpha in ALPHA_VALUES:
        for folds, cv_splits in cv_splits_by_fold.items():
            fold_mse = []

            for fit_indices, validation_indices in cv_splits:
                estimator = make_estimator(float(alpha))
                estimator.fit(X[fit_indices], y[fit_indices])
                prediction = estimator.predict(X[validation_indices])
                fold_mse.append(
                    mean_squared_error(y[validation_indices], prediction)
                )

            rows.append({
                "alpha": float(alpha),
                "n_splits": folds,
                "mean_cv_mse": float(np.mean(fold_mse)),
                "std_cv_mse": float(np.std(fold_mse, ddof=0)),
            })

    results = pd.DataFrame(rows)
    # Pemilihan akhir memakai MSE rata-rata yang sama bobotnya untuk k=5,10,20.
    alpha_summary = results.groupby("alpha", as_index=False).agg(
        mean_cv_mse=("mean_cv_mse", "mean"),
        std_cv_mse=("mean_cv_mse", "std"),
    )
    best_index = alpha_summary["mean_cv_mse"].idxmin()
    best_alpha = float(alpha_summary.loc[best_index, "alpha"])
    results["mean_cv_mse_all_k"] = results["alpha"].map(
        alpha_summary.set_index("alpha")["mean_cv_mse"]
    )
    results["selected_alpha"] = best_alpha
    return best_alpha, results


def main() -> None:
    if not FEATURES_CSV.is_file():
        raise FileNotFoundError(f"CSV fitur tidak ditemukan: {FEATURES_CSV}")
    if not LABELS_CSV.is_file():
        raise FileNotFoundError(f"CSV label tidak ditemukan: {LABELS_CSV}")

    features = pd.read_csv(FEATURES_CSV, encoding="utf-8-sig")
    labels = pd.read_csv(LABELS_CSV, encoding="utf-8-sig")

    missing_features = set([FEATURE_IMAGE_COLUMN, *S20_FEATURES]) - set(features.columns)
    missing_labels = {LABEL_IMAGE_COLUMN, TARGET_COLUMN} - set(labels.columns)

    if missing_features:
        raise KeyError(f"Kolom fitur tidak ditemukan: {sorted(missing_features)}")
    if missing_labels:
        raise KeyError(f"Kolom label tidak ditemukan: {sorted(missing_labels)}")

    features["_key"] = features[FEATURE_IMAGE_COLUMN].map(image_key)
    labels["_key"] = labels[LABEL_IMAGE_COLUMN].map(image_key)

    if features["_key"].duplicated().any():
        raise ValueError("Ada nama gambar fitur duplikat setelah ekstensi dihapus.")
    if labels["_key"].duplicated().any():
        raise ValueError("Ada nama gambar label duplikat setelah ekstensi dihapus.")

    data = features.merge(
        labels[["_key", LABEL_IMAGE_COLUMN, TARGET_COLUMN]],
        on="_key",
        how="inner",
        validate="one_to_one",
    )

    if data.empty:
        raise ValueError(
            "Tidak ada gambar yang cocok antara CSV fitur dan CSV label."
        )

    data["nama_file"] = data[LABEL_IMAGE_COLUMN].astype("string")

    # Acak urutan data dengan seed tetap agar hasil dapat direproduksi.
    data = data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    X = data[S20_FEATURES].to_numpy(dtype=np.float64)
    y = parse_target(data[TARGET_COLUMN])

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Fitur atau target mengandung nilai kosong/tidak valid.")

    # Split acak: 80% train dan 20% test.
    n_samples = len(data)
    all_indices = np.arange(n_samples)
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=TEST_RATIO,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    train_names = data.iloc[train_indices]["nama_file"].to_numpy()
    test_names = data.iloc[test_indices]["nama_file"].to_numpy()

    if len(y_train) <= N_SPLITS:
        raise ValueError("Data train terlalu sedikit untuk CV acak 5-fold.")

    outer_cv = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    cv_rows = []

    # CV bersarang acak: alpha dipilih hanya dari bagian fit pada setiap fold.
    for fold_number, (fit_indices, validation_indices) in enumerate(
        outer_cv.split(X_train), start=1
    ):
        print(f"Evaluasi CV fold {fold_number}/{N_SPLITS}")

        fold_alpha, _ = select_alpha(
            X_train[fit_indices],
            y_train[fit_indices],
        )

        fold_model = make_estimator(fold_alpha)
        fold_model.fit(X_train[fit_indices], y_train[fit_indices])
        fold_prediction = fold_model.predict(X_train[validation_indices])

        cv_rows.append({
            "Model": "LASSO",
            "Fold": fold_number,
            **calculate_metrics(
                y_train[validation_indices],
                fold_prediction,
            ),
        })
    cv_per_fold = pd.DataFrame(cv_rows)
    
    cv_summary = pd.DataFrame([{
    "Model": "LASSO",
    "MAE_mean": cv_per_fold["MAE"].mean(),
    "MAE_std": cv_per_fold["MAE"].std(ddof=1),
    "RMSE_mean": cv_per_fold["RMSE"].mean(),
    "RMSE_std": cv_per_fold["RMSE"].std(ddof=1),
    "R2_mean": cv_per_fold["R2"].mean(),
    "R2_std": cv_per_fold["R2"].std(ddof=1),
    "n_folds": len(cv_per_fold),
    }])
    
    for metric in ("MAE", "RMSE", "R2"):
        cv_summary[f"{metric}_mean ± std"] = cv_summary.apply(
            lambda row: f"{row[f'{metric}_mean']:.4g} ± {row[f'{metric}_std']:.4g}",
            axis=1,
        )
    # Pilih alpha final menggunakan seluruh data train saja.
    print("Mencari alpha final menggunakan data train...")
    best_alpha, alpha_results = select_alpha(X_train, y_train)

    final_model = make_estimator(best_alpha)
    final_model.fit(X_train, y_train)

    train_prediction = final_model.predict(X_train)
    test_prediction = final_model.predict(X_test)

    train_test_metrics = pd.DataFrame([
        {
            "Model": "LASSO",
            "Data": "Train",
            **calculate_metrics(y_train, train_prediction),
        },
        {
            "Model": "LASSO",
            "Data": "Test",
            **calculate_metrics(y_test, test_prediction),
        },
    ])

    # Struktur kolom mengikuti prediksi_test_semua_model.csv dari notebook.
    predictions = pd.DataFrame({
        "nama_file": test_names,
        "actual": y_test,
        "predicted_LASSO": test_prediction,
        "residual_LASSO": y_test - test_prediction,
    })

    # Simpan empat CSV laporan utama dengan nama yang sama seperti notebook.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    cv_per_fold.to_csv(
        OUTPUT_DIR / "hasil_cv_per_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cv_summary.to_csv(
        OUTPUT_DIR / "hasil_cv_ringkasan.csv",
        index=False,
        encoding="utf-8-sig",
    )
    train_test_metrics.to_csv(
        OUTPUT_DIR / "hasil_evaluasi_train_test_semua_model.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        OUTPUT_DIR / "prediksi_test_semua_model.csv",
        index=False,
        encoding="utf-8-sig",
    )
    joblib.dump(final_model, OUTPUT_DIR / "model_terbaik.joblib")

    # CSV tambahan untuk memeriksa pencarian alpha.
    alpha_results.to_csv(
        OUTPUT_DIR / "hasil_cv_alpha.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Dataset fitur: {FEATURES_CSV}")
    print(f"Dataset label: {LABELS_CSV}")
    print(f"Jumlah sampel yang cocok: {n_samples}")
    print(f"Jumlah fitur S20: {len(S20_FEATURES)}")
    print(f"Jumlah data train: {len(train_indices)}")
    print(f"Jumlah data test: {len(test_indices)}")
    print(f"Jumlah fold CV: {N_SPLITS}")
    print(f"Alpha terbaik pada data train: {best_alpha:.8g}")

    print("\nSkor CV per fold:")
    print(cv_per_fold.to_string(index=False))

    print("\nRingkasan CV (mean ± std antar-fold):")
    print(cv_summary[
        ["Model", "MAE_mean ± std", "RMSE_mean ± std", "R2_mean ± std", "n_folds"]
    ].to_string(index=False))

    print("\nEvaluasi train dan test:")
    print(train_test_metrics.to_string(index=False))

    print(f"\nFile hasil disimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
