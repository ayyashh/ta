"""Gabungkan metrik CV 5-fold, train, dan test ke tabel evaluasi."""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. Temukan folder hasil
# ============================================================

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

KAGGLE_WORKING = Path("/kaggle/working")
REQUIRED_FILES = [
    "hasil_cv_per_fold.csv",
    "hasil_cv_ringkasan.csv",
    "hasil_evaluasi_train_test_semua_model.csv",
]

if KAGGLE_WORKING.is_dir():
    FOLDER_CANDIDATES = [
        KAGGLE_WORKING,
        KAGGLE_WORKING / "Hasil",
        KAGGLE_WORKING / "Hasil" / "Random",
    ]
else:
    FOLDER_CANDIDATES = [
        SCRIPT_DIR / "Hasil",
        SCRIPT_DIR / "Hasil" / "Random",
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
        "CSV hasil training tidak ditemukan. Folder yang diperiksa:\n"
        f"{checked}\nPastikan ketiga file hasil training sudah tersedia."
    )


# ============================================================
# 2. Baca file hasil training dan periksa kolom
# ============================================================

cv_folds = pd.read_csv(
    FOLDER_HASIL / "hasil_cv_per_fold.csv",
    encoding="utf-8-sig",
)
cv_summary = pd.read_csv(
    FOLDER_HASIL / "hasil_cv_ringkasan.csv",
    encoding="utf-8-sig",
)
train_test = pd.read_csv(
    FOLDER_HASIL / "hasil_evaluasi_train_test_semua_model.csv",
    encoding="utf-8-sig",
)

required_columns = {
    "CV per fold": {
        "Model", "n_features", "Fold", "MAE", "RMSE", "R2",
    },
    "Ringkasan CV": {"Model", "n_features"},
    "Train/test": {"Model", "Data", "MAE", "RMSE", "R2"},
}

frames = {
    "CV per fold": cv_folds,
    "Ringkasan CV": cv_summary,
    "Train/test": train_test,
}

for name, required in required_columns.items():
    missing_columns = required - set(frames[name].columns)
    if missing_columns:
        raise KeyError(
            f"Kolom {sorted(missing_columns)} tidak ditemukan pada CSV {name}."
        )

metric_names = {
    "MAE": "MAE",
    "RMSE": "RMSE",
    "R2": "R2",
}
models = cv_summary["Model"].drop_duplicates().tolist()
missing_values = []
table_rows = []
numeric_rows = []


# ============================================================
# 3. Susun metrik CV (mean ± SD), train 80%, dan test 20%
# ============================================================

def numeric_metric(row: pd.Series | None, metric: str) -> float:
    """Ambil metrik numerik; kembalikan NaN jika baris/nilai tidak ada."""
    if row is None:
        return np.nan

    value = pd.to_numeric(row.get(metric), errors="coerce")
    if pd.isna(value) or not np.isfinite(value):
        return np.nan
    return float(value)


def format_number(value: float, decimals: int = 0) -> str:
    if not np.isfinite(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def format_mean_std(mean: float, std: float, metric: str) -> str:
    if not np.isfinite(mean):
        return "N/A"

    decimals = 3 if metric == "R2" else 0
    mean_text = format_number(mean, decimals)
    std_text = format_number(std, decimals)
    return f"{mean_text} +/- {std_text}"


for model_name in models:
    summary_rows = cv_summary[cv_summary["Model"] == model_name]
    if len(summary_rows) != 1:
        missing_values.append(
            f"{model_name}: ringkasan CV seharusnya satu baris, "
            f"ditemukan {len(summary_rows)}."
        )
        summary_row = summary_rows.iloc[0] if not summary_rows.empty else None
    else:
        summary_row = summary_rows.iloc[0]

    if summary_row is None:
        selected_k = np.nan
        cv_rows = cv_folds.iloc[0:0]
    else:
        selected_k = pd.to_numeric(
            summary_row.get("n_features"), errors="coerce"
        )
        cv_rows = cv_folds[
            (cv_folds["Model"] == model_name)
            & (pd.to_numeric(cv_folds["n_features"], errors="coerce") == selected_k)
        ].copy()

    cv_metrics = {}
    for metric in metric_names:
        values = pd.to_numeric(
            cv_rows[metric], errors="coerce"
        ).to_numpy(dtype=float)
        values = values[np.isfinite(values)]

        if len(values) != 5:
            missing_values.append(
                f"{model_name}: metrik CV {metric} pada k={selected_k} "
                f"memiliki {len(values)} dari 5 nilai fold."
            )

        mean = float(np.mean(values)) if len(values) else np.nan
        std = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
        cv_metrics[metric] = (mean, std)

    split_rows = {}
    for split_name in ("Train", "Test"):
        matches = train_test[
            (train_test["Model"] == model_name)
            & (train_test["Data"].astype(str).str.casefold() == split_name.casefold())
        ]

        if len(matches) != 1:
            missing_values.append(
                f"{model_name}: hasil {split_name} seharusnya satu baris, "
                f"ditemukan {len(matches)}."
            )
            split_rows[split_name] = None
        else:
            split_rows[split_name] = matches.iloc[0]

    row_for_table = {
        "Model": model_name,
        "Fitur terpilih (k)": (
            int(selected_k) if pd.notna(selected_k) else "N/A"
        ),
    }
    row_for_numeric = {
        "Model": model_name,
        "Fitur terpilih (k)": selected_k,
    }

    for metric, label in metric_names.items():
        mean, std = cv_metrics[metric]
        row_for_table[f"CV {label} (mean +/- SD) 5-fold"] = format_mean_std(
            mean, std, metric
        )
        row_for_numeric[f"CV {metric} mean"] = mean
        row_for_numeric[f"CV {metric} SD"] = std

        for split_name, split_label in (("Train", "Train 80%"), ("Test", "Test 20%")):
            value = numeric_metric(split_rows[split_name], metric)
            if not np.isfinite(value):
                missing_values.append(
                    f"{model_name}: nilai {metric} untuk {split_label} tidak tersedia."
                )

            row_for_table[f"{split_label} {label}"] = format_number(
                value, 3 if metric == "R2" else 0
            )
            row_for_numeric[f"{split_label} {metric}"] = value

    table_rows.append(row_for_table)
    numeric_rows.append(row_for_numeric)


# ============================================================
# 4. Tampilkan tabel dan simpan hasil
# ============================================================

table = pd.DataFrame(table_rows)
numeric_table = pd.DataFrame(numeric_rows)

presentation_path = FOLDER_HASIL / "tabel_hasil_evaluasi_5_ml.csv"
numeric_path = FOLDER_HASIL / "tabel_hasil_evaluasi_5_ml_numerik.csv"

table.to_csv(presentation_path, index=False, encoding="utf-8-sig")
numeric_table.to_csv(numeric_path, index=False, encoding="utf-8-sig")

print(f"Folder hasil: {FOLDER_HASIL}")
print("\nTabel evaluasi (CV mean ± SD, train 80%, dan test 20%):")
print(table.to_string(index=False))

if missing_values:
    print("\nPERHATIAN — nilai/baris yang tidak tersedia:")
    for item in missing_values:
        print(f"- {item}")
else:
    print(
        "\nPemeriksaan kelengkapan: semua model memiliki 5 nilai fold "
        "dan metrik MAE, RMSE, R2 untuk train serta test."
    )

print(f"\nTabel presentasi disimpan: {presentation_path}")
print(f"Tabel numerik disimpan: {numeric_path}")
