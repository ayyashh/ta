"""Visualisasikan rerata RGB mentah, prediksi, residual, dan koefisien model."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_CSV = SCRIPT_DIR / "fitur_rgb_sampel.csv"
MODEL_DIR = SCRIPT_DIR / "hasil_regresi_polinomial_rgb"
PLOT_DIR = SCRIPT_DIR / "visualisasi_hasil"


def main() -> None:
    prediction_path = MODEL_DIR / "prediksi_train_test.csv"
    coefficient_path = MODEL_DIR / "koefisien_polinomial.csv"

    for path in (FEATURES_CSV, prediction_path, coefficient_path):
        if not path.is_file():
            raise FileNotFoundError(f"File belum ditemukan: {path}")

    features = pd.read_csv(FEATURES_CSV, encoding="utf-8-sig")
    predictions = pd.read_csv(prediction_path, encoding="utf-8-sig")
    coefficients = pd.read_csv(coefficient_path, encoding="utf-8-sig")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Distribusi rerata nilai RGB sampel.
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.boxplot(
        [features["R_mean"], features["G_mean"], features["B_mean"]],
        tick_labels=["R", "G", "B"],
        showfliers=False,
    )
    axis.set(
        title="Distribusi rerata kanal RGB gambar sampel",
        ylabel="Intensitas RGB rata-rata (0–255)",
    )
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        PLOT_DIR / "distribusi_rerata_rgb.png",
        dpi=160,
        bbox_inches="tight",
    )

    # Aktual vs prediksi dan residual pada data test.
    test = predictions[predictions["split"] == "test"]

    if test.empty:
        raise ValueError("Tidak ada prediksi test untuk divisualisasikan.")

    actual = test["actual_density_cells_per_mL"]
    predicted = test["predicted_density_cells_per_mL"]
    residual = test["residual_actual_minus_predicted"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(actual, predicted, alpha=0.8)
    low = min(actual.min(), predicted.min())
    high = max(actual.max(), predicted.max())
    axes[0].plot(
        [low, high],
        [low, high],
        "r--",
        label="Prediksi ideal",
    )
    axes[0].set(
        xlabel="Densitas aktual (sel/mL)",
        ylabel="Prediksi (sel/mL)",
        title="Test: aktual vs prediksi",
    )
    axes[0].legend()

    axes[1].scatter(predicted, residual, alpha=0.8)
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set(
        xlabel="Prediksi (sel/mL)",
        ylabel="Residual aktual − prediksi",
        title="Test: residual",
    )

    for axis in axes:
        axis.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        PLOT_DIR / "prediksi_dan_residual_test.png",
        dpi=160,
        bbox_inches="tight",
    )

    # Koefisien fitur polinomial.
    ordered = coefficients.sort_values("coefficient")
    colors = np.where(ordered["coefficient"] >= 0, "teal", "tomato")

    fig, axis = plt.subplots(figsize=(9, 6))
    axis.barh(ordered["term"], ordered["coefficient"], color=colors)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set(
        xlabel="Koefisien",
        title="Koefisien regresi polinomial RGB orde 2",
    )
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        PLOT_DIR / "koefisien_polinomial_rgb.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.show()
    plt.close("all")
    print(f"Grafik tersimpan di: {PLOT_DIR}")


if __name__ == "__main__":
    main()
