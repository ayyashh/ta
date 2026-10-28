"""Train four regressors using RGB histogram features and biomass labels."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_CSV = SCRIPT_DIR / "fitur_histogram_rgb.csv"
LABELS_CSV = SCRIPT_DIR.parents[1] / "Dataset" / "Data Sensor Kultivasi" / "csv" / "gabungan_dengan_jumlah_sel.csv"
OUTPUT_DIR = SCRIPT_DIR / "hasil_model_histogram_rgb"

IMAGE_COLUMN_FEATURES = "image_path"
IMAGE_COLUMN_LABELS = "Nama_Gambar"
TARGET_COLUMN = "Jumlah Sel (sel/mL)"
TEST_SIZE = 0.20
N_SPLITS = 5
N_QUANTILE_BINS = 5
RANDOM_STATE = 42


def make_quantile_bins(target: np.ndarray) -> np.ndarray:
    """Create quantile strata with enough samples for test split and 5-fold CV."""
    for requested_bins in range(N_QUANTILE_BINS, 1, -1):
        bins = pd.qcut(target, q=requested_bins, labels=False, duplicates="drop")
        bins = np.asarray(bins, dtype=int)
        counts = np.bincount(bins)
        actual_bins = len(counts)
        test_count = int(np.ceil(TEST_SIZE * len(target)))

        if (
            actual_bins >= 2
            and counts.min() >= N_SPLITS
            and test_count >= actual_bins
            and len(target) - test_count >= actual_bins
        ):
            print(f"Stratifikasi menggunakan {actual_bins} kuantil biomassa.")
            return bins

    raise ValueError(
        "Data terlalu sedikit atau distribusi biomassa terlalu banyak nilai duplikat "
        "untuk split dan Stratified 5-Fold. Tambah data atau kurangi jumlah fold."
    )


def main() -> None:
    if not FEATURES_CSV.is_file():
        raise FileNotFoundError(f"Fitur histogram tidak ditemukan: {FEATURES_CSV}")
    if not LABELS_CSV.is_file():
        raise FileNotFoundError(f"File label tidak ditemukan: {LABELS_CSV}")

    features = pd.read_csv(FEATURES_CSV, encoding="utf-8-sig")
    labels = pd.read_csv(LABELS_CSV, encoding="utf-8-sig")
    required_features = {IMAGE_COLUMN_FEATURES}
    required_labels = {IMAGE_COLUMN_LABELS, TARGET_COLUMN}
    if not required_features.issubset(features.columns):
        raise KeyError(f"Kolom fitur wajib tidak ditemukan: {required_features}")
    if not required_labels.issubset(labels.columns):
        raise KeyError(f"Kolom label wajib tidak ditemukan: {required_labels}")

    feature_columns = [column for column in features.columns if column != IMAGE_COLUMN_FEATURES]
    expected_columns = {f"{channel}_bin_{i:03d}" for channel in ("r", "g", "b") for i in range(256)}
    if set(feature_columns) != expected_columns:
        raise ValueError("Kolom fitur harus berupa histogram RGB 256 bin per kanal (768 fitur).")

    # Samakan key berdasarkan nama file agar label dapat dicocokkan dengan
    # feature path yang mungkin memiliki subfolder.
    features["_key"] = features[IMAGE_COLUMN_FEATURES].map(lambda value: Path(str(value)).name.lower())
    labels["_key"] = labels[IMAGE_COLUMN_LABELS].map(lambda value: Path(str(value)).name.lower())
    if features["_key"].duplicated().any():
        raise ValueError("Ada nama gambar duplikat pada fitur histogram.")
    if labels["_key"].duplicated().any():
        raise ValueError("Ada nama gambar duplikat pada file label.")

    data = features.merge(
        labels[["_key", TARGET_COLUMN]], on="_key", how="inner", validate="one_to_one"
    )
    if data.empty:
        raise ValueError("Tidak ada nama gambar yang cocok antara fitur dan label.")
    print(f"Gambar dengan fitur dan label cocok: {len(data)}")
    print(f"Gambar fitur tanpa label: {len(features) - len(data)}")

    X = data[feature_columns].to_numpy(dtype=np.float32)
    y = pd.to_numeric(data[TARGET_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Fitur atau target mengandung nilai kosong/tidak valid.")
    if len(y) < N_SPLITS + 2:
        raise ValueError("Sampel tidak cukup untuk split train/test dan CV 5-fold.")

    strata = make_quantile_bins(y)
    indices = np.arange(len(y))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=strata,
    )
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    train_strata = strata[train_indices]

    # GridSearchCV stratifikasi kuantil dilakukan pada bagian train saja.
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    models = {
        "DTS": (
            DecisionTreeRegressor(random_state=RANDOM_STATE),
            {
                "max_depth": [None, 3, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        ),
        "RF": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "n_estimators": [200, 500],
                "max_depth": [None, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": [0.5, 1.0],
            },
        ),
        "GBM": (
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.03, 0.1],
                "max_depth": [2, 3],
                "min_samples_leaf": [1, 3],
            },
        ),
        "k-NN": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor()),
            ]),
            {
                "model__n_neighbors": [3, 5, 7],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    prediction_rows = []
    cv_result_frames = []
    for model_name, (estimator, parameter_grid) in models.items():
        print(f"Grid search untuk {model_name}...")
        search = GridSearchCV(
            estimator=estimator,
            param_grid=parameter_grid,
            scoring="neg_mean_absolute_error",
            cv=list(cv.split(X_train, train_strata)),
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        train_prediction = search.best_estimator_.predict(X_train)
        test_prediction = search.best_estimator_.predict(X_test)
        results.append({
            "model": model_name,
            "cv_MAE": -search.best_score_,
            "test_MAE": mean_absolute_error(y_test, test_prediction),
            "test_RMSE": np.sqrt(mean_squared_error(y_test, test_prediction)),
            "test_R2": r2_score(y_test, test_prediction),
            "best_params": str(search.best_params_),
        })

        for split_name, split_indices, actual, predicted in (
            ("train", train_indices, y_train, train_prediction),
            ("test", test_indices, y_test, test_prediction),
        ):
            for image_path, actual_value, predicted_value in zip(
                data[IMAGE_COLUMN_FEATURES].iloc[split_indices], actual, predicted
            ):
                prediction_rows.append({
                    "model": model_name,
                    "split": split_name,
                    "image_path": image_path,
                    "actual_density_cells_per_mL": actual_value,
                    "predicted_density_cells_per_mL": predicted_value,
                    "residual_actual_minus_predicted": actual_value - predicted_value,
                })

        cv_results = pd.DataFrame(search.cv_results_)
        cv_results.insert(0, "model", model_name)
        cv_result_frames.append(cv_results)
        joblib.dump(search.best_estimator_, OUTPUT_DIR / f"model_{model_name}.joblib")
        print(f"  CV MAE terbaik: {-search.best_score_:.6g}")
        print(f"  Parameter terbaik: {search.best_params_}")

    results_df = pd.DataFrame(results).sort_values("cv_MAE")
    results_df.to_csv(OUTPUT_DIR / "ringkasan_hasil.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv(
        OUTPUT_DIR / "prediksi_train_test.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(cv_result_frames, ignore_index=True, sort=False).to_csv(
        OUTPUT_DIR / "hasil_grid_search_cv.csv", index=False, encoding="utf-8-sig"
    )
    print(f"\\nRingkasan hasil (target: {TARGET_COLUMN}):")
    print(results_df.to_string(index=False))
    print(f"\\nHasil disimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
