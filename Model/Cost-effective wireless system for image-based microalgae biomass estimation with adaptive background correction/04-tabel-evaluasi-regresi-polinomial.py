"""Buat tabel evaluasi regresi polinomial: CV 5-fold, train, dan test."""

from pathlib import Path

import numpy as np
import pandas as pd


# Folder ini berisi CSV yang dihasilkan oleh skrip regresi.
SCRIPT_DIR = Path(__file__).resolve().parent
FOLDER_HASIL = SCRIPT_DIR / "Hasil_sel_mikroalga"

CV_FILE = FOLDER_HASIL / "hasil_cv_per_fold.csv"
TRAIN_TEST_FILE = FOLDER_HASIL / "hasil_evaluasi_train_test_semua_model.csv"
METRICS = ("MAE", "RMSE", "R2")


def format_number(value: float, decimals: int = 0) -> str:
    """Format angka dengan pemisah ribuan koma."""
    if not np.isfinite(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def main() -> None:
    for path in (CV_FILE, TRAIN_TEST_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"File evaluasi tidak ditemukan: {path}")

    cv = pd.read_csv(CV_FILE, encoding="utf-8-sig")
    train_test = pd.read_csv(TRAIN_TEST_FILE, encoding="utf-8-sig")

    required_cv = {"fold", *METRICS}
    required_splits = {"Data", *METRICS}
    missing_cv = required_cv - set(cv.columns)
    missing_splits = required_splits - set(train_test.columns)
    if missing_cv:
        raise KeyError(f"Kolom tidak ada di hasil CV: {sorted(missing_cv)}")
    if missing_splits:
        raise KeyError(
            "Kolom tidak ada di hasil evaluasi train/test: "
            f"{sorted(missing_splits)}"
        )

    warnings = []
    fold_count = pd.to_numeric(cv["fold"], errors="coerce").nunique()
    if len(cv) != 5 or fold_count != 5:
        warnings.append(
            f"Diharapkan 5 fold; ditemukan {fold_count} fold dalam {len(cv)} baris."
        )

    presentation = {"Pendekatan": "Regresi Polinomial Orde 2"}
    numeric = {"Pendekatan": "Regresi Polinomial Orde 2"}

    for metric in METRICS:
        values = pd.to_numeric(cv[metric], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        mean = float(np.mean(values)) if len(values) else np.nan
        # SD sampel (ddof=1), konsisten dengan ringkasan CV.
        std = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
        if len(values) != 5:
            warnings.append(
                f"CV {metric}: ditemukan {len(values)} nilai valid dari 5 fold."
            )

        decimals = 3 if metric == "R2" else 0
        presentation[f"CV {metric} (mean ± SD, 5-fold)"] = (
            f"{format_number(mean, decimals)} ± {format_number(std, decimals)}"
            if np.isfinite(mean) and np.isfinite(std)
            else "N/A"
        )
        numeric[f"CV {metric} mean"] = mean
        numeric[f"CV {metric} SD"] = std

        for split, label in (("Train", "Train 80%"), ("Test", "Test 20%")):
            rows = train_test[
                train_test["Data"].astype(str).str.strip().str.casefold()
                == split.casefold()
            ]
            if len(rows) != 1:
                warnings.append(
                    f"Baris {label} seharusnya satu; ditemukan {len(rows)}."
                )
                value = np.nan
            else:
                value = pd.to_numeric(rows.iloc[0][metric], errors="coerce")
                value = float(value) if pd.notna(value) else np.nan

            if not np.isfinite(value):
                warnings.append(f"Nilai {metric} untuk {label} tidak tersedia.")
            presentation[f"{label} {metric}"] = format_number(value, decimals)
            numeric[f"{label} {metric}"] = value

    presentation_table = pd.DataFrame([presentation])
    numeric_table = pd.DataFrame([numeric])

    presentation_path = FOLDER_HASIL / "tabel_evaluasi_regresi_polinomial.csv"
    numeric_path = FOLDER_HASIL / "tabel_evaluasi_regresi_polinomial_numerik.csv"
    presentation_table.to_csv(presentation_path, index=False, encoding="utf-8-sig")
    numeric_table.to_csv(numeric_path, index=False, encoding="utf-8-sig")

    print("Tabel evaluasi regresi polinomial:")
    print(presentation_table.to_string(index=False))
    if warnings:
        print("\nPemeriksaan data:")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("\nSemua 5 fold serta metrik train dan test tersedia.")
    print(f"\nTabel presentasi: {presentation_path}")
    print(f"Tabel numerik:    {numeric_path}")


if __name__ == "__main__":
    main()
