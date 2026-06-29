"""
Shared evaluation/plotting helpers for the model notebooks.

Lifted from the inline definitions that were duplicated across
DescitionTree_Band_Gap.ipynb and DescisionTree_Hull_Regressor.ipynb.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def plot_confusion_matrix(model, X_test, y_test, labels_x, labels_y):
    """Row-normalised (percentage) confusion matrix for a classifier."""
    y_pred = model.predict(X_test)

    cm_percentage = confusion_matrix(y_test, y_pred, normalize="true")

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_percentage,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        xticklabels=labels_x,
        yticklabels=labels_y,
        cbar=True,
        square=True,
        vmin=0,
        vmax=1,
    )
    plt.title("Confusion Matrix (Prozentual) - Metall-Klassifikation", fontsize=14, pad=20)
    plt.xlabel("Vorhersage des Modells (Predicted)", fontsize=12)
    plt.ylabel("Tatsächliche Struktur (Actual)", fontsize=12)
    plt.tight_layout()
    plt.show()


def evaluate_model_performance(model, X_test, y_test, title=None):
    """Regression metrics + true-vs-predicted scatter, all in original units.

    Assumes the model was trained on the raw target (no y-scaling), so its
    predictions are already in the original unit (eV / eV per atom).
    """
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    max_err = max_error(y_test, y_pred)

    print("==================================================")
    print("   MODELL-PERFORMANCE AUF DEM TESTSET (ORIGINAL)")
    print("==================================================")
    print(f"Mean Absolute Error (MAE):      {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE):  {rmse:.4f}")
    print(f"R² Score (Bestimmtheitsmaß):    {r2:.4f}")
    print(f"Maximaler Fehler (Max Error):   {max_err:.4f}")
    print("==================================================")

    lo = float(min(y_test.min(), y_pred.min()))
    hi = float(max(y_test.max(), y_pred.max()))

    plt.figure(figsize=(5, 5))
    plt.scatter(y_test, y_pred, s=10, alpha=0.5)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title(title)

    x = np.linspace(lo, hi, 100)
    plt.plot(x, x, color="red", linestyle="--", label="y=x")
    plt.legend()
    plt.tight_layout()
    plt.show()
