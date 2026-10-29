"""Random-split ML training with cross-validated feature-count selection."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
except ImportError as error:
    raise ImportError("Instal xgboost untuk menjalankan model XGBoost.") from error


SCRIPT_DIR = Path.cwd()
DATASET_NAME = "dataset_fitur_jumlah_sel_dari_biomass.csv"
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
    OUTPUT_DIR = Path("/kaggle/working")
else:
    OUTPUT_DIR = SCRIPT_DIR / "Hasil" / "Random"

IMAGE_COLUMN = "file_name"
TARGET_COLUMN = "Jumlah Sel dari Biomass (sel/mL)"
TEST_RATIO = 0.20
N_SPLITS = 5
SEED = 42
FEATURE_COUNTS = (5, 10, 20, 30, 50)


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
    df = df.reset_index(drop=True)
    X = df[feature_columns].to_numpy(dtype=np.float32, copy=True)
    y = target
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Fitur atau target mengandung nilai kosong/tidak valid.")

    n_samples = len(y)
    train_indices, test_indices = train_test_split(
        np.arange(n_samples), test_size=TEST_RATIO,
        random_state=SEED, shuffle=True,
    )
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    if len(train_indices) <= N_SPLITS:
        raise ValueError("Data train terlalu sedikit untuk KFold 5-fold.")
    feature_counts = sorted({min(k, X.shape[1]) for k in FEATURE_COUNTS} | {X.shape[1]})
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(cv.split(X_train))
    model_factories = {
        "DTS": lambda: (
            DecisionTreeRegressor(
                max_depth=5, min_samples_leaf=3, random_state=SEED
            ),
            False,
        ),
        "Random Forest": lambda: (
            RandomForestRegressor(
                n_estimators=300, min_samples_leaf=2, max_features=0.5,
                random_state=SEED, n_jobs=-1,
            ),
            False,
        ),
        "Gradient Boosting": lambda: (
            GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.03,
                max_depth=2, random_state=SEED,
            ),
            False,
        ),
        "KNN": lambda: (
            KNeighborsRegressor(n_neighbors=5, weights="distance", p=2),
            True,
        ),
        "XGBoost": lambda: (
            XGBRegressor(
                objective="reg:squarederror", n_estimators=300,
                learning_rate=0.03, max_depth=3, subsample=0.8,
                colsample_bytree=0.8, reg_lambda=1.0,
                random_state=SEED, n_jobs=-1,
            ),
            False,
        ),
    }

    cv_rows = []
    summary_rows = []
    feature_selection_rows = []
    test_rows = []
    predictions = {"nama_file": df.iloc[test_indices][IMAGE_COLUMN].to_numpy(), "actual": y_test}
    fitted_models = {}

    for model_name, model_factory in model_factories.items():
        print(f"CV random + pemilihan fitur: {model_name}")
        per_k_rows = []
        for feature_count in feature_counts:
            for fold_number, (fit_indices, validation_indices) in enumerate(
                folds, start=1
            ):
                estimator, scale = model_factory()
                pipeline = make_pipeline(estimator, feature_count, scale=scale)
                pipeline.fit(X_train[fit_indices], y_train[fit_indices])
                fold_prediction = pipeline.predict(X_train[validation_indices])
                scores = metrics(y_train[validation_indices], fold_prediction)
                row = {
                    "Model": model_name,
                    "n_features": feature_count,
                    "Fold": fold_number,
                    **scores,
                }
                cv_rows.append(row)
                per_k_rows.append(row)

        per_k_summary = (
            pd.DataFrame(per_k_rows)
            .groupby("n_features", as_index=False)
            .agg(
                MAE_mean=("MAE", "mean"),
                MAE_std=("MAE", "std"),
                RMSE_mean=("RMSE", "mean"),
                RMSE_std=("RMSE", "std"),
                R2_mean=("R2", "mean"),
                R2_std=("R2", "std"),
                n_folds=("Fold", "count"),
            )
        )
        selected_summary = per_k_summary.sort_values(["MAE_mean", "n_features"]).iloc[0]
        selected_count = int(selected_summary["n_features"])
        summary_rows.append({
            "Model": model_name,
            "n_features": selected_count,
            **selected_summary.drop(labels="n_features").to_dict(),
        })
        feature_selection_rows.extend(
            {"Model": model_name, **row.to_dict()}
            for _, row in per_k_summary.iterrows()
        )

        estimator, scale = model_factory()
        final_model = make_pipeline(estimator, selected_count, scale=scale)
        final_model.fit(X_train, y_train)
        fitted_models[model_name] = final_model
        train_prediction = final_model.predict(X_train)
        test_prediction = final_model.predict(X_test)
        test_rows.extend([
            {"Model": model_name, "Data": "Train", **metrics(y_train, train_prediction)},
            {"Model": model_name, "Data": "Test", **metrics(y_test, test_prediction)},
        ])
        predictions[f"predicted_{model_name}"] = test_prediction
        predictions[f"residual_{model_name}"] = y_test - test_prediction

    cv_per_fold = pd.DataFrame(cv_rows)
    cv_summary = pd.DataFrame(summary_rows).sort_values("MAE_mean").reset_index(drop=True)
    feature_selection_summary = pd.DataFrame(feature_selection_rows)
    train_test_metrics = pd.DataFrame(test_rows)
    best_model_name = cv_summary.iloc[0]["Model"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv_per_fold.to_csv(OUTPUT_DIR / "hasil_cv_per_fold.csv", index=False, encoding="utf-8-sig")
    cv_summary.to_csv(OUTPUT_DIR / "hasil_cv_ringkasan.csv", index=False, encoding="utf-8-sig")
    feature_selection_summary.to_csv(
        OUTPUT_DIR / "hasil_pemilihan_jumlah_fitur.csv", index=False, encoding="utf-8-sig"
    )
    selected_feature_rows = []
    for model_name, pipeline in fitted_models.items():
        selector = pipeline.named_steps["select"]
        for feature, score, selected in zip(
            feature_columns, selector.scores_, selector.get_support()
        ):
            selected_feature_rows.append({
                "Model": model_name,
                "selected_n_features": int(selector.get_support().sum()),
                "feature": feature,
                "f_score": score,
                "selected": bool(selected),
            })
    pd.DataFrame(selected_feature_rows).to_csv(
        OUTPUT_DIR / "fitur_terpilih_per_model.csv", index=False, encoding="utf-8-sig"
    )
    train_test_metrics.to_csv(
        OUTPUT_DIR / "hasil_evaluasi_train_test_semua_model.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(predictions).to_csv(
        OUTPUT_DIR / "prediksi_test_semua_model.csv", index=False, encoding="utf-8-sig"
    )
    joblib.dump(fitted_models[best_model_name], OUTPUT_DIR / "model_terbaik.joblib")

    print(f"Sampel random split: train={len(train_indices)}, test={len(test_indices)}")
    print(f"Model terbaik berdasarkan MAE CV: {best_model_name}")
    print(f"Semua CSV dan model disimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
