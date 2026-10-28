"""Chronological ML training with fold-level and summary CSV outputs."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
except ImportError as error:
    raise ImportError("Instal xgboost untuk menjalankan model XGBoost.") from error


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_NAME = "dataset_fitur_jumlah_sel.csv"
KAGGLE_INPUT = Path("/kaggle/input")
if KAGGLE_INPUT.is_dir():
    candidates = list(KAGGLE_INPUT.rglob(DATASET_NAME))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Harus ditemukan tepat satu {DATASET_NAME} di {KAGGLE_INPUT}; "
            f"ditemukan {len(candidates)}. Atur file_data secara eksplisit."
        )
    FILE_DATA = candidates[0]
else:
    FILE_DATA = SCRIPT_DIR / DATASET_NAME

if not FILE_DATA.is_file():
    raise FileNotFoundError(f"Dataset tidak ditemukan: {FILE_DATA}")

if Path("/kaggle/working").is_dir():
    OUTPUT_DIR = Path("/kaggle/working/Hasil/Chronological")
else:
    OUTPUT_DIR = SCRIPT_DIR / "Hasil" / "Chronological"

IMAGE_COLUMN = "nama_file"
TARGET_COLUMN = "Jumlah Sel (sel/mL)"
TEST_RATIO = 0.20
N_SPLITS = 5
SEED = 42


def parse_target(values: pd.Series) -> np.ndarray:
    text = values.astype("string").str.strip()
    thousands = text.str.fullmatch(r"\d{1,3}(?:\.\d{3})+").fillna(False)
    text.loc[thousands] = text.loc[thousands].str.replace(".", "", regex=False)
    return pd.to_numeric(text, errors="coerce").to_numpy(dtype=np.float64)


def make_pipeline(estimator, feature_count: int, scale: bool = False) -> Pipeline:
    steps = [
        ("select", SelectKBest(score_func=f_regression, k=feature_count)),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": r2_score(actual, predicted),
    }


def main() -> None:
    header = pd.read_csv(FILE_DATA, encoding="utf-8-sig", nrows=0).columns.tolist()
    feature_columns = [column for column in header if column.startswith("fitur_")]
    required = {IMAGE_COLUMN, TARGET_COLUMN}
    if not required.issubset(header) or not feature_columns:
        raise KeyError(
            f"Dataset harus memiliki {sorted(required)} dan kolom fitur_*."
        )

    dtypes = {column: np.float32 for column in feature_columns}
    dtypes[IMAGE_COLUMN] = "string"
    dtypes[TARGET_COLUMN] = "string"
    df = pd.read_csv(FILE_DATA, encoding="utf-8-sig", dtype=dtypes)
    if df[IMAGE_COLUMN].isna().any() or df[IMAGE_COLUMN].duplicated().any():
        raise ValueError("nama_file kosong atau duplikat.")

    target = parse_target(df[TARGET_COLUMN])
    timestamps_text = df[IMAGE_COLUMN].str.extract(
        r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", expand=False
    )
    timestamps = pd.to_datetime(
        timestamps_text, format="%Y-%m-%d_%H-%M-%S", errors="coerce"
    )
    if timestamps.isna().any():
        raise ValueError("Timestamp pada sebagian nama_file tidak dikenali.")

    order = np.argsort(timestamps.to_numpy(), kind="stable")
    df = df.iloc[order].reset_index(drop=True)
    X = df[feature_columns].to_numpy(dtype=np.float32, copy=True)
    y = target[order]
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Fitur atau target mengandung nilai kosong/tidak valid.")

    n_samples = len(y)
    n_test = max(1, int(np.ceil(TEST_RATIO * n_samples)))
    split_index = n_samples - n_test
    if split_index <= N_SPLITS:
        raise ValueError("Data train terlalu sedikit untuk TimeSeriesSplit 5-fold.")
    train_indices = np.arange(0, split_index)
    test_indices = np.arange(split_index, n_samples)
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    n_selected = min(50, X.shape[1])
    cv = TimeSeriesSplit(n_splits=N_SPLITS)
    folds = list(cv.split(X_train))
    models = {
        "DTS": make_pipeline(
            DecisionTreeRegressor(max_depth=5, min_samples_leaf=3, random_state=SEED),
            n_selected,
        ),
        "Random Forest": make_pipeline(
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                max_features=0.5,
                random_state=SEED,
                n_jobs=-1,
            ),
            n_selected,
        ),
        "Gradient Boosting": make_pipeline(
            GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.03, max_depth=2, random_state=SEED
            ),
            n_selected,
        ),
        "KNN": make_pipeline(
            KNeighborsRegressor(n_neighbors=5, weights="distance", p=2),
            n_selected,
            scale=True,
        ),
        "XGBoost": make_pipeline(
            XGBRegressor(
                objective="reg:squarederror",
                n_estimators=300,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=SEED,
                n_jobs=-1,
            ),
            n_selected,
        ),
    }

    cv_rows = []
    summary_rows = []
    test_rows = []
    predictions = {"nama_file": df.iloc[test_indices][IMAGE_COLUMN].to_numpy(), "actual": y_test}
    fitted_models = {}

    for model_name, estimator in models.items():
        print(f"CV chronological: {model_name}")
        fold_scores = []
        for fold_number, (fit_indices, validation_indices) in enumerate(folds, start=1):
            estimator.fit(X_train[fit_indices], y_train[fit_indices])
            fold_prediction = estimator.predict(X_train[validation_indices])
            scores = metrics(y_train[validation_indices], fold_prediction)
            fold_scores.append(scores)
            cv_rows.append({"Model": model_name, "Fold": fold_number, **scores})

        scores_frame = pd.DataFrame(fold_scores)
        summary_rows.append({
            "Model": model_name,
            "MAE_mean": scores_frame["MAE"].mean(),
            "MAE_std": scores_frame["MAE"].std(ddof=1),
            "RMSE_mean": scores_frame["RMSE"].mean(),
            "RMSE_std": scores_frame["RMSE"].std(ddof=1),
            "R2_mean": scores_frame["R2"].mean(),
            "R2_std": scores_frame["R2"].std(ddof=1),
            "n_folds": len(scores_frame),
        })

        # Fit final model only on the chronological train period, evaluate later period.
        estimator.fit(X_train, y_train)
        fitted_models[model_name] = estimator
        train_prediction = estimator.predict(X_train)
        test_prediction = estimator.predict(X_test)
        test_rows.extend([
            {"Model": model_name, "Data": "Train", **metrics(y_train, train_prediction)},
            {"Model": model_name, "Data": "Test", **metrics(y_test, test_prediction)},
        ])
        predictions[f"predicted_{model_name}"] = test_prediction
        predictions[f"residual_{model_name}"] = y_test - test_prediction

    cv_per_fold = pd.DataFrame(cv_rows)
    cv_summary = pd.DataFrame(summary_rows).sort_values("MAE_mean").reset_index(drop=True)
    train_test_metrics = pd.DataFrame(test_rows)
    best_model_name = cv_summary.iloc[0]["Model"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv_per_fold.to_csv(OUTPUT_DIR / "hasil_cv_per_fold.csv", index=False, encoding="utf-8-sig")
    cv_summary.to_csv(OUTPUT_DIR / "hasil_cv_ringkasan.csv", index=False, encoding="utf-8-sig")
    train_test_metrics.to_csv(
        OUTPUT_DIR / "hasil_evaluasi_train_test_semua_model.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(predictions).to_csv(
        OUTPUT_DIR / "prediksi_test_semua_model.csv", index=False, encoding="utf-8-sig"
    )
    joblib.dump(fitted_models[best_model_name], OUTPUT_DIR / "model_terbaik.joblib")

    print(f"Sampel chronological: train={len(train_indices)}, test={len(test_indices)}")
    print(f"Model terbaik berdasarkan MAE CV: {best_model_name}")
    print(f"Semua CSV dan model disimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
