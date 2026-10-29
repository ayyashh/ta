"""Susun tabel evaluasi 5 DL: CV 5-fold, train 80%, dan test 20%."""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. Tentukan folder hasil training 5 DL
# ============================================================


SCRIPT_DIR = Path.cwd()

KAGGLE_OUTPUT = Path("/kaggle/working/hasil_dl_fusion_random")
LOCAL_OUTPUT = SCRIPT_DIR / "Hasil_sel_mikroalga"
FOLDER_HASIL = KAGGLE_OUTPUT if KAGGLE_OUTPUT.is_dir() else LOCAL_OUTPUT

REQUIRED_FILES = [
    "dl_cv_metrics_per_fold.csv",
    "dl_cv_model_summary.csv",
    "dl_final_metrics_train_test.csv",
]
missing_files = [
    filename for filename in REQUIRED_FILES
    if not (FOLDER_HASIL / filename).is_file()
]
if missing_files:
    raise FileNotFoundError(
        "Hasil training 5 DL belum lengkap.\n"
        f"Folder yang diperiksa: {FOLDER_HASIL}\n"
        "File yang tidak ditemukan:\n"
        + "\n".join(f"- {filename}" for filename in missing_files)
    )


# ============================================================
# 2. Baca dan validasi CSV hasil training
# ============================================================

cv_folds = pd.read_csv(
    FOLDER_HASIL / "dl_cv_metrics_per_fold.csv", encoding="utf-8-sig"
)
cv_summary = pd.read_csv(
    FOLDER_HASIL / "dl_cv_model_summary.csv", encoding="utf-8-sig"
)
train_test = pd.read_csv(
    FOLDER_HASIL / "dl_final_metrics_train_test.csv", encoding="utf-8-sig"
)

required_columns = {
    "CV per fold": {"Model", "n_features", "Fold", "MAE", "RMSE", "R2"},
    "Ringkasan CV": {"Model", "selected_n_features"},
    "Train/test": {"Model", "Data", "MAE", "RMSE", "R2"},
}
frames = {
    "CV per fold": cv_folds,
    "Ringkasan CV": cv_summary,
    "Train/test": train_test,
}
for frame_name, required in required_columns.items():
    missing_columns = required - set(frames[frame_name].columns)
    if missing_columns:
        raise KeyError(
            f"Kolom {sorted(missing_columns)} tidak ditemukan pada CSV {frame_name}."
        )

if "total_fitur_dipilih" in cv_summary.columns:
    cv_summary["selected_n_features"] = pd.to_numeric(
        cv_summary["total_fitur_dipilih"], errors="coerce"
    )

metric_names = ("MAE", "RMSE", "R2")
models = cv_summary["Model"].drop_duplicates().tolist()
missing_values = []
presentation_rows = []
numeric_rows = []


# ============================================================
# 3. Fungsi format dan pembacaan nilai
# ============================================================

def metric_value(row, metric):
    """Ambil nilai metrik sebagai float; NaN jika baris/nilai tidak tersedia."""
    if row is None:
        return np.nan
    value = pd.to_numeric(row.get(metric), errors="coerce")
    if pd.isna(value) or not np.isfinite(value):
        return np.nan
    return float(value)


def format_number(value, decimals=0):
    if not np.isfinite(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def format_mean_std(mean, std, metric):
    if not np.isfinite(mean):
        return "N/A"
    decimals = 3 if metric == "R2" else 0
    return f"{format_number(mean, decimals)} ± {format_number(std, decimals)}"


# ============================================================
# 4. Gabungkan CV pada k terpilih dengan hasil train dan test
# ============================================================

for model_name in models:
    summary_matches = cv_summary[cv_summary["Model"] == model_name]
    if len(summary_matches) != 1:
        missing_values.append(
            f"{model_name}: ringkasan CV seharusnya satu baris, "
            f"ditemukan {len(summary_matches)}."
        )
    summary_row = summary_matches.iloc[0] if not summary_matches.empty else None

    selected_count = (
        pd.to_numeric(summary_row.get("selected_n_features"), errors="coerce")
        if summary_row is not None else np.nan
    )
    if not np.isfinite(selected_count):
        missing_values.append(f"{model_name}: jumlah fitur terpilih tidak tersedia.")
        selected_count = np.nan

    if np.isfinite(selected_count):
        model_cv = cv_folds[
            (cv_folds["Model"] == model_name)
            & (
                pd.to_numeric(cv_folds["n_features"], errors="coerce")
                == selected_count
            )
        ].copy()
    else:
        model_cv = cv_folds.iloc[0:0].copy()

    cv_values = {}
    for metric in metric_names:
        values = pd.to_numeric(model_cv[metric], errors="coerce").to_numpy(float)
        values = values[np.isfinite(values)]
        if len(values) != 5:
            missing_values.append(
                f"{model_name}: CV {metric} pada top-{selected_count:g} fitur "
                f"memiliki {len(values)} dari 5 nilai fold."
            )
        mean = float(np.mean(values)) if len(values) else np.nan
        std = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
        cv_values[metric] = (mean, std)

    split_rows = {}
    for split_name in ("Train", "Test"):
        matches = train_test[
            (train_test["Model"] == model_name)
            & (train_test["Data"].astype(str).str.casefold() == split_name.casefold())
        ]
        if len(matches) != 1:
            missing_values.append(
                f"{model_name}: baris {split_name} seharusnya satu, "
                f"ditemukan {len(matches)}."
            )
            split_rows[split_name] = None
        else:
            split_rows[split_name] = matches.iloc[0]

    pretty_row = {
        "Model": model_name,
        "Total fitur konvensional dipilih": (
            int(selected_count) if np.isfinite(selected_count) else "N/A"
        ),
    }
    numeric_row = {
        "Model": model_name,
        "Total fitur konvensional dipilih": selected_count,
    }

    for metric in metric_names:
        mean, std = cv_values[metric]
        pretty_row[f"CV {metric} (mean ± SD, 5-fold)"] = format_mean_std(
            mean, std, metric
        )
        numeric_row[f"CV {metric} mean"] = mean
        numeric_row[f"CV {metric} SD"] = std

        for split_name, split_label in (("Train", "Train 80%"), ("Test", "Test 20%")):
            value = metric_value(split_rows[split_name], metric)
            if not np.isfinite(value):
                missing_values.append(
                    f"{model_name}: {metric} {split_label} tidak tersedia."
                )
            pretty_row[f"{split_label} {metric}"] = format_number(
                value, 3 if metric == "R2" else 0
            )
            numeric_row[f"{split_label} {metric}"] = value

    presentation_rows.append(pretty_row)
    numeric_rows.append(numeric_row)


# ============================================================
# 5. Tampilkan tabel dan simpan versi presentasi/numerik
# ============================================================

presentation_table = pd.DataFrame(presentation_rows)
numeric_table = pd.DataFrame(numeric_rows)

presentation_path = FOLDER_HASIL / "tabel_hasil_evaluasi_5_dl.csv"
numeric_path = FOLDER_HASIL / "tabel_hasil_evaluasi_5_dl_numerik.csv"
presentation_table.to_csv(presentation_path, index=False, encoding="utf-8-sig")
numeric_table.to_csv(numeric_path, index=False, encoding="utf-8-sig")

print(f"Folder hasil: {FOLDER_HASIL}")
print("\nTabel evaluasi: CV mean ± SD, train 80%, dan test 20%")
print(presentation_table.to_string(index=False))

if missing_values:
    print("\nPERHATIAN — nilai/baris yang tidak lengkap:")
    for item in missing_values:
        print(f"- {item}")
else:
    print(
        "\nPemeriksaan kelengkapan: setiap model memiliki 5 nilai fold "
        "dan metrik MAE, RMSE, serta R² untuk train dan test."
    )

print(f"\nTabel presentasi disimpan: {presentation_path}")
print(f"Tabel numerik disimpan: {numeric_path}")
