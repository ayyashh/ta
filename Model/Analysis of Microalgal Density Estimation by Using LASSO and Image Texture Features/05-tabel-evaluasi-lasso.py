"""Buat tabel evaluasi LASSO: CV 5-fold, train, dan test."""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. Temukan folder hasil training
# ============================================================


SCRIPT_DIR = Path.cwd()


FOLDER_CANDIDATES = [
    SCRIPT_DIR / "Hasil_biomass"
]

REQUIRED_FILES = [
    "hasil_cv_per_fold.csv",
    "hasil_evaluasi_train_test_semua_model.csv",
]
FOLDER_HASIL = next(
    (
        folder
        for folder in FOLDER_CANDIDATES
        if all((folder / filename).is_file() for filename in REQUIRED_FILES)
    ),
    None,
)
if FOLDER_HASIL is None:
    checked = "\n".join(f"- {folder}" for folder in FOLDER_CANDIDATES)
    raise FileNotFoundError(
        "File evaluasi LASSO belum ditemukan. Folder yang diperiksa:\n"
        f"{checked}\nFile wajib: {', '.join(REQUIRED_FILES)}"
    )


# ============================================================
# 2. Baca hasil CV, train/test, dan alpha
# ============================================================

cv_folds = pd.read_csv(
    FOLDER_HASIL / "hasil_cv_per_fold.csv", encoding="utf-8-sig"
)
train_test = pd.read_csv(
    FOLDER_HASIL / "hasil_evaluasi_train_test_semua_model.csv",
    encoding="utf-8-sig",
)

alpha_path = FOLDER_HASIL / "hasil_cv_alpha.csv"
alpha_df = (
    pd.read_csv(alpha_path, encoding="utf-8-sig")
    if alpha_path.is_file()
    else None
)

required_cv_columns = {"Fold", "MAE", "RMSE", "R2"}
required_eval_columns = {"Data", "MAE", "RMSE", "R2"}
missing_cv = required_cv_columns - set(cv_folds.columns)
missing_eval = required_eval_columns - set(train_test.columns)
if missing_cv:
    raise KeyError(f"Kolom CV tidak tersedia: {sorted(missing_cv)}")
if missing_eval:
    raise KeyError(f"Kolom evaluasi train/test tidak tersedia: {sorted(missing_eval)}")

missing_values = []
metric_names = ("MAE", "RMSE", "R2")


# ============================================================
# 3. Fungsi format nilai
# ============================================================

def format_number(value, decimals=0):
    if not np.isfinite(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def format_mean_std(mean, std, metric):
    if not np.isfinite(mean):
        return "N/A"
    decimals = 3 if metric == "R2" else 0
    mean_text = format_number(mean, decimals)
    std_text = format_number(std, decimals)
    return f"{mean_text} ± {std_text}"


def get_metric(row, metric):
    if row is None:
        return np.nan
    value = pd.to_numeric(row.get(metric), errors="coerce")
    return float(value) if pd.notna(value) and np.isfinite(value) else np.nan


# ============================================================
# 4. Hitung mean ± SD CV dan ambil metrik train/test
# ============================================================

if "Model" in cv_folds.columns:
    cv_folds = cv_folds[
        cv_folds["Model"].astype(str).str.casefold() == "lasso"
    ]
if "Model" in train_test.columns:
    train_test = train_test[
        train_test["Model"].astype(str).str.casefold() == "lasso"
    ]

presentation_row = {
    "Pendekatan": "LASSO",
    "Jumlah fitur S20": 12,
}
numeric_row = {
    "Pendekatan": "LASSO",
    "Jumlah fitur S20": 12,
}

fold_numbers = pd.to_numeric(cv_folds["Fold"], errors="coerce").dropna().nunique()
if fold_numbers != 5 or len(cv_folds) != 5:
    missing_values.append(
        f"CV seharusnya memiliki 5 fold; ditemukan {fold_numbers} fold "
        f"dan {len(cv_folds)} baris."
    )

for metric in metric_names:
    values = pd.to_numeric(cv_folds[metric], errors="coerce").to_numpy(float)
    values = values[np.isfinite(values)]
    mean = float(np.mean(values)) if len(values) else np.nan
    std = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
    if len(values) != 5:
        missing_values.append(
            f"CV {metric} memiliki {len(values)} nilai valid dari 5 fold."
        )

    presentation_row[f"CV {metric} (mean ± SD, 5-fold)"] = format_mean_std(
        mean, std, metric
    )
    numeric_row[f"CV {metric} mean"] = mean
    numeric_row[f"CV {metric} SD"] = std

    for split_name, split_label in (("Train", "Train 80%"), ("Test", "Test 20%")):
        matches = train_test[
            train_test["Data"].astype(str).str.casefold() == split_name.casefold()
        ]
        if len(matches) != 1:
            missing_values.append(
                f"Evaluasi {split_label} seharusnya satu baris; ditemukan {len(matches)}."
            )
            row = None
        else:
            row = matches.iloc[0]

        value = get_metric(row, metric)
        if not np.isfinite(value):
            missing_values.append(f"Metrik {metric} pada {split_label} tidak tersedia.")
        decimals = 3 if metric == "R2" else 0
        presentation_row[f"{split_label} {metric}"] = format_number(value, decimals)
        numeric_row[f"{split_label} {metric}"] = value


# ============================================================
# 5. Tambahkan alpha terpilih jika file pencarian tersedia
# ============================================================

best_alpha = np.nan
if alpha_df is not None:
    if "selected_alpha" not in alpha_df.columns:
        missing_values.append("Kolom selected_alpha tidak ada pada hasil_cv_alpha.csv.")
    else:
        alpha_values = pd.to_numeric(
            alpha_df["selected_alpha"], errors="coerce"
        ).dropna().unique()
        if len(alpha_values) == 1:
            best_alpha = float(alpha_values[0])
        else:
            missing_values.append(
                "Alpha terpilih seharusnya satu nilai; "
                f"ditemukan {len(alpha_values)} nilai."
            )
else:
    missing_values.append("hasil_cv_alpha.csv tidak tersedia; alpha tidak ditampilkan.")

presentation_row["Alpha terpilih"] = (
    format_number(best_alpha, 8) if np.isfinite(best_alpha) else "N/A"
)
numeric_row["Alpha terpilih"] = best_alpha


# ============================================================
# 6. Tampilkan dan simpan tabel presentasi/numerik
# ============================================================

presentation_table = pd.DataFrame([presentation_row])
numeric_table = pd.DataFrame([numeric_row])

presentation_path = FOLDER_HASIL / "tabel_evaluasi_lasso.csv"
numeric_path = FOLDER_HASIL / "tabel_evaluasi_lasso_numerik.csv"
presentation_table.to_csv(presentation_path, index=False, encoding="utf-8-sig")
numeric_table.to_csv(numeric_path, index=False, encoding="utf-8-sig")

print(f"Folder hasil: {FOLDER_HASIL}")
print("\nTabel evaluasi LASSO:")
print(presentation_table.to_string(index=False))

if missing_values:
    print("\nPERHATIAN — data yang tidak lengkap:")
    for item in missing_values:
        print(f"- {item}")
else:
    print("\nPemeriksaan kelengkapan: semua metrik dan lima fold tersedia.")

print(f"\nTabel presentasi disimpan: {presentation_path}")
print(f"Tabel numerik disimpan: {numeric_path}")
