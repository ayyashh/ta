"""Visualize CSV outputs from the RGB histogram regression experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "hasil_model_histogram_rgb"
PLOT_DIR = RESULT_DIR / "visualisasi"


def main() -> None:
    summary_path = RESULT_DIR / "ringkasan_hasil.csv"
    prediction_path = RESULT_DIR / "prediksi_train_test.csv"
    grid_path = RESULT_DIR / "hasil_grid_search_cv.csv"
    for path in (summary_path, prediction_path, grid_path):
        if not path.is_file():
            raise FileNotFoundError(f"Hasil CSV belum ditemukan: {path}")

    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    predictions = pd.read_csv(prediction_path, encoding="utf-8-sig")
    grid = pd.read_csv(grid_path, encoding="utf-8-sig")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Perbandingan skor test untuk empat model.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis, metric, title in zip(
        axes,
        ("test_MAE", "test_RMSE", "test_R2"),
        ("Test MAE", "Test RMSE", "Test R²"),
    ):
        ordered = summary.sort_values(metric, ascending=(metric != "test_R2"))
        axis.bar(ordered["model"], ordered[metric], color="steelblue")
        axis.set_title(title)
        axis.set_ylabel(metric)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "perbandingan_metrik_test.png", dpi=160, bbox_inches="tight")

    # Actual vs predicted dan residual pada test, satu panel per model.
    test = predictions[predictions["split"] == "test"]
    models = list(test["model"].drop_duplicates())
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), squeeze=False)
    for axis, model_name in zip(axes.flat, models):
        part = test[test["model"] == model_name]
        axis.scatter(
            part["actual_density_cells_per_mL"],
            part["predicted_density_cells_per_mL"],
            alpha=0.8,
        )
        low = min(
            part["actual_density_cells_per_mL"].min(),
            part["predicted_density_cells_per_mL"].min(),
        )
        high = max(
            part["actual_density_cells_per_mL"].max(),
            part["predicted_density_cells_per_mL"].max(),
        )
        axis.plot([low, high], [low, high], "r--", label="Prediksi ideal")
        axis.set(
            title=model_name,
            xlabel="Densitas aktual (sel/mL)",
            ylabel="Prediksi (sel/mL)",
        )
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "aktual_vs_prediksi_test.png", dpi=160, bbox_inches="tight")

    # Sebaran MAE CV dari seluruh kombinasi grid search, per model.
    if {"model", "mean_test_score"}.issubset(grid.columns):
        grid["cv_MAE"] = -grid["mean_test_score"]
        fig, axis = plt.subplots(figsize=(9, 5))
        grid.boxplot(column="cv_MAE", by="model", ax=axis, grid=False)
        axis.set(title="Sebaran MAE pada kombinasi Grid Search", xlabel="Model", ylabel="CV MAE")
        fig.suptitle("")
        axis.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "sebaran_grid_search_cv_mae.png", dpi=160, bbox_inches="tight")

    plt.show()
    plt.close("all")
    print(f"Grafik tersimpan di: {PLOT_DIR}")


if __name__ == "__main__":
    main()
