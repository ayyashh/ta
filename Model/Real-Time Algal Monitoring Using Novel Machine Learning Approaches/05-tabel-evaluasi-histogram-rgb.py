"""Buat tabel evaluasi histogram RGB untuk CV 5-fold, train, dan test."""

from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
FOLDER_HASIL = SCRIPT_DIR / "hasil_model_histogram_rgb"

CV_FOLD_FILE = FOLDER_HASIL / "hasil_cv_per_fold.csv"
TRAIN_TEST_FILE = FOLDER_HASIL / "hasil_evaluasi_train_test_semua_model.csv"
METRICS = ("MAE", "RMSE", "R2")


def format_number(value: float, decimals: int = 0) -> str:
    """Format angka dengan koma sebagai pemisah ribuan."""
    if not np.isfinite(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def get_split_metric(rows: pd.DataFrame, split: str, metric: str) -> float:
    selected = rows[
        rows["Data"].astype(str).str.strip().str.casefold() == split.casefold()
    ]
    if len(selected) != 1:
        return np.nan
    value = pd.to_numeric(selected.iloc[0][metric], errors="coerce")
    return float(value) if pd.notna(value) else np.nan


def main() -> None:
    for path in (CV_FOLD_FILE, TRAIN_TEST_FILE):
        if not path.is_file():
            raise FileNotFoundError(
                f"File hasil tidak ditemukan: {path}\n"
                "Jalankan skrip 03-pemodelan-regresi-histogram-rgb.py terlebih dahulu."
            )

    cv = pd.read_csv(CV_FOLD_FILE, encoding="utf-8-sig")
    train_test = pd.read_csv(TRAIN_TEST_FILE, encoding="utf-8-sig")

    required_cv = {"Model", "jumlah_fitur", "Fold", *METRICS}
    required_eval = {"Model", "Data", "jumlah_fitur", *METRICS}
    missing_cv = required_cv - set(cv.columns)
    missing_eval = required_eval - set(train_test.columns)
    if missing_cv:
        raise KeyError(f"Kolom tidak ada di CSV CV: {sorted(missing_cv)}")
    if missing_eval:
        raise KeyError(
            f"Kolom tidak ada di CSV train/test: {sorted(missing_eval)}"
        )

    warnings = []
    presentation_rows = []
    numeric_rows = []

    for model_name in cv["Model"].dropna().astype(str).drop_duplicates():
        model_cv = cv[cv["Model"].astype(str) == model_name].copy()
        model_eval = train_test[train_test["Model"].astype(str) == model_name]
        fold_count = pd.to_numeric(model_cv["Fold"], errors="coerce").nunique()
        if len(model_cv) != 5 or fold_count != 5:
            warnings.append(
                f"{model_name}: diharapkan 5 fold, ditemukan {fold_count} fold "
                f"dalam {len(model_cv)} baris."
            )

        feature_values = pd.to_numeric(
            model_cv["jumlah_fitur"], errors="coerce"
        ).dropna().unique()
        feature_count = int(feature_values[0]) if len(feature_values) == 1 else np.nan
        if len(feature_values) != 1:
            warnings.append(
                f"{model_name}: jumlah fitur tidak tunggal/valid pada hasil CV."
            )

        shown = {"Model": model_name, "Jumlah fitur": feature_count}
        numeric = {"Model": model_name, "Jumlah fitur": feature_count}

        for metric in METRICS:
            values = pd.to_numeric(model_cv[metric], errors="coerce").to_numpy(float)
            values = values[np.isfinite(values)]
            mean = float(np.mean(values)) if len(values) else np.nan
            std = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            if len(values) != 5:
                warnings.append(
                    f"{model_name} CV {metric}: {len(values)} nilai valid dari 5 fold."
                )

            decimals = 3 if metric == "R2" else 0
            shown[f"CV {metric} (mean ± SD)"] = (
                f"{format_number(mean, decimals)} ± {format_number(std, decimals)}"
                if np.isfinite(mean) and np.isfinite(std)
                else "N/A"
            )
            numeric[f"CV {metric} mean"] = mean
            numeric[f"CV {metric} SD"] = std

            for split, split_label in (("Train", "Train 80%"), ("Test", "Test 20%")):
                value = get_split_metric(model_eval, split, metric)
                if not np.isfinite(value):
                    warnings.append(
                        f"{model_name}: metrik {metric} untuk {split_label} "
                        "tidak ditemukan atau tidak valid."
                    )
                shown[f"{split_label} {metric}"] = format_number(value, decimals)
                numeric[f"{split_label} {metric}"] = value

        presentation_rows.append(shown)
        numeric_rows.append(numeric)

    presentation = pd.DataFrame(presentation_rows).sort_values("Model")
    numeric = pd.DataFrame(numeric_rows).sort_values("Model")

    presentation_path = FOLDER_HASIL / "tabel_evaluasi_histogram_rgb.csv"
    numeric_path = FOLDER_HASIL / "tabel_evaluasi_histogram_rgb_numerik.csv"
    presentation.to_csv(presentation_path, index=False, encoding="utf-8-sig")
    numeric.to_csv(numeric_path, index=False, encoding="utf-8-sig")

    print("Tabel evaluasi histogram RGB:")
    print(presentation.to_string(index=False))
    if warnings:
        print("\nPemeriksaan data:")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("\nSemua model memiliki 5 fold dan metrik train/test lengkap.")
    print(f"\nTabel presentasi: {presentation_path}")
    print(f"Tabel numerik:    {numeric_path}")


if __name__ == "__main__":
    main()
