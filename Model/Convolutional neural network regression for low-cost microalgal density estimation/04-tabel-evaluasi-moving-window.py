"""Buat tabel evaluasi CNN moving-window: CV 5-fold, train, dan test."""

from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
LOSS_NAME = "mse"  # Samakan dengan LOSS_NAME di 02-model-deep-cnn-29-layers.py.
FOLDER_HASIL = SCRIPT_DIR / "hasil" / f"loss_{LOSS_NAME}"

CV_FILE = FOLDER_HASIL / "dl_cv_metrics_per_fold.csv"
FINAL_FILE = FOLDER_HASIL / "dl_final_metrics_train_test.csv"
METRICS = ("MAE", "RMSE", "R2")


def format_number(value: float, decimals: int = 0) -> str:
    """Format angka dengan koma sebagai pemisah ribuan."""
    if not np.isfinite(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def main() -> None:
    for path in (CV_FILE, FINAL_FILE):
        if not path.is_file():
            raise FileNotFoundError(
                f"File evaluasi tidak ditemukan: {path}\n"
                "Pastikan skrip 02 selesai dijalankan dan LOSS_NAME cocok."
            )

    cv = pd.read_csv(CV_FILE, encoding="utf-8-sig")
    final = pd.read_csv(FINAL_FILE, encoding="utf-8-sig")

    required_cv = {"Fold", *METRICS}
    required_final = {"Model", "Data", *METRICS}
    missing_cv = required_cv - set(cv.columns)
    missing_final = required_final - set(final.columns)
    if missing_cv:
        raise KeyError(f"Kolom tidak tersedia pada CSV CV: {sorted(missing_cv)}")
    if missing_final:
        raise KeyError(
            f"Kolom tidak tersedia pada CSV train/test: {sorted(missing_final)}"
        )

    warnings = []
    fold_count = pd.to_numeric(cv["Fold"], errors="coerce").nunique()
    if len(cv) != 5 or fold_count != 5:
        warnings.append(
            f"Diharapkan 5 fold; ditemukan {fold_count} fold pada {len(cv)} baris."
        )

    model_names = final["Model"].dropna().astype(str).unique()
    model_name = model_names[0] if len(model_names) == 1 else "Deep CNN 29 Layers"
    if len(model_names) != 1:
        warnings.append(
            "Nama model pada CSV train/test tidak tunggal atau tidak tersedia."
        )

    presentation = {"Model": model_name}
    numeric = {"Model": model_name}

    for metric in METRICS:
        values = pd.to_numeric(cv[metric], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        mean = float(np.mean(values)) if len(values) else np.nan
        std = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
        if len(values) != 5:
            warnings.append(
                f"CV {metric}: hanya {len(values)} nilai valid dari 5 fold."
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
            rows = final[
                final["Data"].astype(str).str.strip().str.casefold()
                == split.casefold()
            ]
            if len(rows) != 1:
                warnings.append(
                    f"Baris {label} seharusnya satu; ditemukan {len(rows)}."
                )
                value = np.nan
            else:
                parsed = pd.to_numeric(rows.iloc[0][metric], errors="coerce")
                value = float(parsed) if pd.notna(parsed) else np.nan

            if not np.isfinite(value):
                warnings.append(f"Nilai {metric} untuk {label} tidak tersedia.")
            presentation[f"{label} {metric}"] = format_number(value, decimals)
            numeric[f"{label} {metric}"] = value

    presentation_table = pd.DataFrame([presentation])
    numeric_table = pd.DataFrame([numeric])
    presentation_path = FOLDER_HASIL / "dl_tabel_evaluasi.csv"
    numeric_path = FOLDER_HASIL / "dl_tabel_evaluasi_numerik.csv"
    presentation_table.to_csv(presentation_path, index=False, encoding="utf-8-sig")
    numeric_table.to_csv(numeric_path, index=False, encoding="utf-8-sig")

    print("Tabel evaluasi CNN moving-window:")
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
