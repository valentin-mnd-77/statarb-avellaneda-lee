"""Build the README figures from the tables saved by ``run_experiments.py``.

Reads only ``results/*.csv``, so it is fast and does not re-run any backtest.

Usage:
    python scripts/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
    }
)


def equity_curves() -> None:
    """Net equity of the hedged and unhedged books."""
    curves = pd.read_csv(
        RESULTS / "01_equity_curves.csv", index_col=0, parse_dates=True
    )

    figure, axis = plt.subplots(figsize=(9, 4))
    for label in curves.columns:
        axis.plot(curves.index, curves[label], linewidth=1.2, label=label)
    axis.axhline(1.0, color="grey", linewidth=0.8)
    axis.set_title("Net equity, 5 bps all-in costs")
    axis.set_ylabel("Growth of 1")
    axis.legend(frameon=False)
    figure.savefig(FIGURES / "equity_curves.png")
    plt.close(figure)


def model_state() -> None:
    """How the model's view of the market moves through time."""
    diagnostics = pd.read_csv(
        RESULTS / "01_diagnostics.csv", index_col=0, parse_dates=True
    )

    figure, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(
        diagnostics.index,
        diagnostics["explained_variance"],
        linewidth=0.8,
        color="tab:blue",
    )
    axes[0].set_ylabel("Explained variance")
    axes[0].set_title("Variance captured by the first 4 principal components")

    axes[1].fill_between(
        diagnostics.index, 0, diagnostics["n_long"], color="tab:green", alpha=0.6,
        label="long",
    )
    axes[1].fill_between(
        diagnostics.index, 0, -diagnostics["n_short"], color="tab:red", alpha=0.6,
        label="short",
    )
    axes[1].set_ylabel("Signals")
    axes[1].legend(frameon=False, ncol=2)
    axes[1].set_title("Open positions")

    axes[2].plot(
        diagnostics.index,
        diagnostics["median_half_life_days"],
        linewidth=0.8,
        color="tab:purple",
    )
    axes[2].set_ylabel("Days")
    axes[2].set_title("Median estimated half-life of mean reversion")

    figure.tight_layout()
    figure.savefig(FIGURES / "model_state.png")
    plt.close(figure)


def factor_counts() -> None:
    """Number of components needed to reach each variance target."""
    counts = pd.read_csv(
        RESULTS / "04_factor_counts.csv", index_col=0, parse_dates=True
    )

    figure, axis = plt.subplots(figsize=(9, 4))
    for label in counts.columns:
        if label == "fixed m = 4":
            continue
        axis.plot(counts.index, counts[label], linewidth=0.8, label=label)
    axis.axhline(4, color="black", linestyle="--", linewidth=1.2, label="fixed m = 4")
    axis.set_ylabel("Number of factors")
    axis.set_title(
        "Components required per variance target: fewer in stress, more in calm"
    )
    axis.legend(frameon=False, ncol=3, fontsize=8)
    figure.savefig(FIGURES / "factor_counts.png")
    plt.close(figure)


def cost_sensitivity() -> None:
    """Where the alpha dies as costs rise."""
    costs = pd.read_csv(RESULTS / "05_cost_sensitivity.csv", index_col=0)
    rates = [0, 1, 2, 5, 10]

    figure, axis = plt.subplots(figsize=(6.5, 4))
    for book in ["hedged", "unhedged"]:
        sharpes = [
            costs.loc[f"{book}, {rate} bps", "sharpe_ratio"] for rate in rates
        ]
        axis.plot(rates, sharpes, marker="o", linewidth=1.4, label=book)
    axis.axhline(0.0, color="grey", linewidth=0.8)
    axis.set_xlabel("All-in one-way cost (bps)")
    axis.set_ylabel("Net Sharpe ratio")
    axis.set_title("The strategy is a transaction-cost story")
    axis.legend(frameon=False)
    figure.savefig(FIGURES / "cost_sensitivity.png")
    plt.close(figure)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    equity_curves()
    model_state()
    factor_counts()
    cost_sensitivity()
    print(f"Figures written to {FIGURES}")


if __name__ == "__main__":
    main()
