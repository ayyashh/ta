"""Train four regressors using RGB histogram features and biomass labels."""

from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.base import clone
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


def image_key(value) -> str:
    """Buat kunci gambar dari nama dasar tanpa folder atau ekstensi."""
    normalized = str(value).replace("\\", "/").strip()
    return Path(normalized).stem.casefold()


def parse_numeric_target(value) -> float:
    """Parse angka target dengan format Indonesia atau internasional."""
    if pd.isna(value):
        return np.nan

    text = str(value).strip().replace(" ", "").replace("\u00a0", "")
    if not text:
        return np.nan

    has_comma = "," in text
    has_dot = "." in text

    if has_comma and has_dot:
        # Separator terakhir dianggap desimal: 425.584,73 atau 425,584.73.
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif has_comma:
        if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+", text):
            text = text.replace(",", "")
        else:
            text = text.replace(",", ".")
    elif has_dot and re.fullmatch(r"[+-]?\d{1,3}(?:\.\d{3})+", text):
        text = text.replace(".", "")

    try:
        return float(text)
    except ValueError:
        return np.nan


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
    labels = pd.read_csv(
        LABELS_CSV,
        encoding="utf-8-sig",
        dtype={TARGET_COLUMN: "string"},
    )
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

    # Cocokkan nama dasar gambar saja. Ekstensi .jpg/.png dan subfolder
    # boleh berbeda antara CSV fitur dan CSV label.
    features["_key"] = features[IMAGE_COLUMN_FEATURES].map(image_key)
    labels["_key"] = labels[IMAGE_COLUMN_LABELS].map(image_key)
    if features["_key"].duplicated().any():
        raise ValueError("Ada nama gambar duplikat pada fitur histogram.")
    if labels["_key"].duplicated().any():
        raise ValueError("Ada nama gambar duplikat pada file label.")

    labels["_target_numeric"] = labels[TARGET_COLUMN].map(parse_numeric_target)
    data = features.merge(
        labels[["_key", TARGET_COLUMN, "_target_numeric"]],
        on="_key",
        how="inner",
        validate="one_to_one",
    )
    if data.empty:
        raise ValueError("Tidak ada nama gambar yang cocok antara fitur dan label.")
    print(f"Gambar dengan fitur dan label cocok: {len(data)}")
    print(f"Gambar fitur tanpa label: {len(features) - len(data)}")

    X_frame = data[feature_columns].apply(pd.to_numeric, errors="coerce")
    X = X_frame.to_numpy(dtype=np.float64)
    y = pd.to_numeric(data["_target_numeric"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    invalid_feature_count = int((~np.isfinite(X)).sum())
    invalid_target_mask = ~np.isfinite(y)
    if invalid_feature_count or invalid_target_mask.any():
        invalid_columns = [
            column
            for column in feature_columns
            if not np.isfinite(X_frame[column].to_numpy(dtype=np.float64)).all()
        ]
        invalid_target_examples = (
            data.loc[invalid_target_mask, TARGET_COLUMN]
            .astype("string")
            .head(10)
            .tolist()
        )
        raise ValueError(
            "Ditemukan nilai kosong/tidak valid setelah fitur dan target digabung. "
            f"Nilai fitur tidak valid: {invalid_feature_count} "
            f"(kolom: {invalid_columns[:10]}). "
            f"Target tidak valid: {int(invalid_target_mask.sum())} "
            f"(contoh nilai asli: {invalid_target_examples})."
        )

    X = X.astype(np.float32)
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
    test_prediction_rows = []
    cv_fold_rows = []
    train_test_metric_rows = []
    cv_result_frames = []
    cv_splits = list(cv.split(X_train, train_strata))

    for model_name, (estimator, parameter_grid) in models.items():
        print(f"Grid search untuk {model_name}...")
        search = GridSearchCV(
            estimator=estimator,
            param_grid=parameter_grid,
            scoring="neg_mean_absolute_error",
            cv=cv_splits,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        train_prediction = search.best_estimator_.predict(X_train)
        test_prediction = search.best_estimator_.predict(X_test)

        # Evaluasi parameter terbaik pada setiap fold, dengan fold yang sama
        # seperti GridSearchCV. Jumlah fitur tetap 768 histogram RGB.
        for fold_number, (fit_indices, validation_indices) in enumerate(
            cv_splits, start=1
        ):
            fold_estimator = clone(estimator).set_params(**search.best_params_)
            fold_estimator.fit(X_train[fit_indices], y_train[fit_indices])
            validation_prediction = fold_estimator.predict(
                X_train[validation_indices]
            )
            y_validation = y_train[validation_indices]
            cv_fold_rows.append({
                "Model": model_name,
                "jumlah_fitur": len(feature_columns),
                "Fold": fold_number,
                "n_train": len(fit_indices),
                "n_validation": len(validation_indices),
                "MAE": mean_absolute_error(y_validation, validation_prediction),
                "RMSE": np.sqrt(
                    mean_squared_error(y_validation, validation_prediction)
                ),
                "R2": r2_score(y_validation, validation_prediction),
                "parameter_terbaik": str(search.best_params_),
            })

        results.append({
            "model": model_name,
            "cv_MAE": -search.best_score_,
            "test_MAE": mean_absolute_error(y_test, test_prediction),
            "test_RMSE": np.sqrt(mean_squared_error(y_test, test_prediction)),
            "test_R2": r2_score(y_test, test_prediction),
            "best_params": str(search.best_params_),
        })

        for split_name, actual, predicted in (
            ("Train", y_train, train_prediction),
            ("Test", y_test, test_prediction),
        ):
            train_test_metric_rows.append({
                "Model": model_name,
                "Data": split_name,
                "jumlah_fitur": len(feature_columns),
                "n_sampel": len(actual),
                "MAE": mean_absolute_error(actual, predicted),
                "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
                "R2": r2_score(actual, predicted),
                "parameter_terbaik": str(search.best_params_),
            })

        for split_name, split_indices, actual, predicted in (
            ("train", train_indices, y_train, train_prediction),
            ("test", test_indices, y_test, test_prediction),
        ):
            for image_path, actual_value, predicted_value in zip(
                data[IMAGE_COLUMN_FEATURES].iloc[split_indices], actual, predicted
            ):
                prediction_row = {
                    "model": model_name,
                    "split": split_name,
                    "image_path": image_path,
                    "actual_density_cells_per_mL": actual_value,
                    "predicted_density_cells_per_mL": predicted_value,
                    "residual_actual_minus_predicted": actual_value - predicted_value,
                }
                prediction_rows.append(prediction_row)
                if split_name == "test":
                    test_prediction_rows.append({
                        "Model": model_name,
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
    cv_fold_df = pd.DataFrame(cv_fold_rows)
    cv_summary_df = (
        cv_fold_df.groupby("Model", as_index=False)
        .agg(
            jumlah_fitur=("jumlah_fitur", "first"),
            n_fold=("Fold", "nunique"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
        )
    )
    cv_summary_df = cv_summary_df.merge(
        results_df[["model", "best_params"]],
        left_on="Model",
        right_on="model",
        how="left",
    ).drop(columns="model")

    cv_fold_df.to_csv(
        OUTPUT_DIR / "hasil_cv_per_fold.csv", index=False, encoding="utf-8-sig"
    )
    cv_summary_df.to_csv(
        OUTPUT_DIR / "hasil_cv_ringkasan.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(train_test_metric_rows).to_csv(
        OUTPUT_DIR / "hasil_evaluasi_train_test_semua_model.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(test_prediction_rows).to_csv(
        OUTPUT_DIR / "prediksi_test_semua_model.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # File lama tetap ditulis agar skrip visualisasi 04 masih dapat digunakan.
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
