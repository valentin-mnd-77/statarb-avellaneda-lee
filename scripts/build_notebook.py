"""Generate notebooks/01_research.ipynb from a list of cells.

Keeping the notebook in a script makes it reviewable in a diff and impossible to
desynchronise from the package. Run this, then execute the notebook.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# Statistical arbitrage on the Euro Stoxx 50

A walkthrough of Avellaneda & Lee (2008), PCA strand.

This notebook opens up **one** rebalance date to show what the model sees, then
runs the full walk-forward backtest. Every table and figure in the README comes
from `scripts/run_experiments.py`; nothing is computed twice.""",
    ),
    (
        "code",
        """import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd().parent))

from statarb.config import StrategyConfig
from statarb.data import (
    compute_returns,
    coverage_report,
    load_prices,
    load_ticker_details,
    prepare_estimation_window,
)
from statarb.factor_model import extract_factors, fit_factor_regression
from statarb.metrics import performance_statistics
from statarb.ou import estimate_ou_parameters, false_positive_rate
from statarb.signals import compute_s_scores
from statarb.strategy import run_strategy

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})

DATA = Path.cwd().parent / "data"
config = StrategyConfig(no_trade_band=0.005)
config""",
    ),
    ("markdown", "## 1. Data\n\nTotal-return index levels in EUR, forward-filled across local market holidays."),
    (
        "code",
        """prices = load_prices(DATA / "sx5e_underlyings.csv")
returns = compute_returns(prices)
sectors = load_ticker_details(DATA / "ticker_details.csv")["GICS"]

print(f"{prices.shape[1]} names, {prices.index[0].date()} to {prices.index[-1].date()}")
coverage_report(prices, returns).head(8).round(4)""",
    ),
    (
        "markdown",
        """The three names at the top joined the index mid-sample, so they carry long
leading gaps; the coverage filter keeps them out of the estimation window until
they have enough history. The `zero_return_share` column measures how much the
forward fill contributes: a few percent for names whose local exchange closes on
days the rest of Europe trades.""",
    ),
    ("markdown", "## 2. One rebalance date\n\n### 2.1 PCA on the correlation matrix"),
    (
        "code",
        """rebalance_date = returns.index[-1]
window, diagnostics = prepare_estimation_window(
    returns, rebalance_date, config.estimation_window, config.min_coverage
)
print(f"{window.index[0].date()} to {window.index[-1].date()}  -> {window.shape}")

decomposition = extract_factors(window, n_factors=config.n_factors)

shares = decomposition.metadata["explained_variance_per_component"]
cumulative = decomposition.metadata["cumulative_explained_variance"]

figure, axes = plt.subplots(1, 2, figsize=(11, 3.6))
axes[0].bar(range(1, 16), shares[:15], color="tab:blue")
axes[0].set(xlabel="Component", ylabel="Share of variance", title="Eigenvalue spectrum")
axes[1].plot(range(1, 26), cumulative[:25], "o-", markersize=3)
axes[1].axvline(config.n_factors, color="tab:green", ls="--", label=f"m = {config.n_factors}")
axes[1].axhline(0.55, color="tab:red", ls="--", label="55% of variance")
axes[1].set(xlabel="Components retained", ylabel="Cumulative", title="Cumulative variance")
axes[1].legend(frameon=False)
plt.tight_layout()\nplt.show()

print(f"first component alone: {shares[0]:.1%}")
print(f"m = {config.n_factors}: {decomposition.total_explained_variance:.1%}")""",
    ),
    (
        "markdown",
        """### 2.2 The factors are eigenportfolios

Equation (9) sets the dollar holdings of portfolio *j* to `Q_i = v_i / σ_i`, so a
factor return is a portfolio return. Two properties follow, and both are worth
checking rather than assuming.""",
    ),
    (
        "code",
        """correlation = decomposition.factors.corr().to_numpy()
print(f"max |corr(F_j, F_k)|, j != k : {np.abs(correlation - np.eye(config.n_factors)).max():.2e}")

first = decomposition.eigenportfolios["PC1"]
print(f"PC1 weights positive          : {(first > 0).sum()}/{len(first)}")
print(f"corr(Q1, 1/sigma)             : {np.corrcoef(first, 1 / decomposition.volatilities)[0, 1]:.2f}")""",
    ),
    (
        "markdown",
        """The factors are orthogonal to machine precision. That is not luck: with
`D = diag(σ)` and `Σ = D ρ D`, the covariance of two eigenportfolio returns is
`(v_j/σ)' Σ (v_k/σ) = v_j' ρ v_k = λ_j δ_jk`. The design matrix of the factor
regression is orthogonal by construction, so the loadings do not move when a
component is added — which is exactly what ETF-based factors cannot offer, since
sector ETFs are correlated with one another.

The first eigenportfolio is long every name with weights inversely proportional
to volatility: the market mode.

### 2.3 Sector coherence of the higher components

The paper's section 2.2 claims that, once sorted, neighbouring coefficients of
the second and third eigenvectors belong to the same industry. On 50 European
names it reads more cleanly than on the 1400 US names of the paper.""",
    ),
    (
        "code",
        """for component in ["PC2", "PC3"]:
    loadings = decomposition.eigenvectors[component].sort_values(ascending=False)
    top = pd.DataFrame({"loading": loadings.head(5).round(3),
                        "sector": sectors.reindex(loadings.head(5).index)})
    bottom = pd.DataFrame({"loading": loadings.tail(5).round(3),
                           "sector": sectors.reindex(loadings.tail(5).index)})
    print(f"\\n=== {component} ===")
    print(top.to_string())
    print("   ...")
    print(bottom.to_string())""",
    ),
    (
        "markdown",
        """### 2.4 Residuals and the Ornstein–Uhlenbeck fit

Loadings come from the last 60 days, factors from the full 252-day window. The
cumulative residual `X_k = Σ ε_j` is the auxiliary process the appendix fits.""",
    ),
    (
        "code",
        """regression_window = window.loc[:, decomposition.eigenportfolios.index].iloc[-config.ou_window:]
regression = fit_factor_regression(
    regression_window, decomposition.factors.iloc[-config.ou_window:]
)
levels = regression.cumulative_residuals

print(f"mean R^2 across names : {regression.r_squared.mean():.1%}")
print(f"max |X_T|             : {np.abs(levels.iloc[-1]).max():.2e}   <- forced to zero by the regression")

parameters = estimate_ou_parameters(
    levels, dt=config.dt,
    min_mean_reversion_speed=config.min_mean_reversion_speed,
    max_ar1_coefficient=config.max_ar1_coefficient,
    center_means=config.center_ou_means,
)
parameters.loc[parameters["is_tradeable"], ["b", "kappa", "m", "sigma_eq", "half_life_days"]].head(8).round(4)""",
    ),
    (
        "code",
        """asset = parameters.loc[parameters["is_tradeable"], "kappa"].idxmax()
row = parameters.loc[asset]

figure, axis = plt.subplots(figsize=(9, 3.6))
axis.plot(levels.index, levels[asset], color="tab:blue", lw=1.2, label="X(t)")
axis.axhline(row["m"], color="tab:red", ls="--", lw=1, label="equilibrium m")
for sign in (-1, 1):
    axis.axhline(row["m"] + sign * 1.25 * row["sigma_eq"], color="grey", ls=":", lw=1)
axis.set(title=f"{asset}: cumulative residual, kappa={row['kappa']:.1f}, "
               f"half-life={row['half_life_days']:.1f} days",
         ylabel="Cumulative residual")
axis.legend(frameon=False)\nplt.show()""",
    ),
    (
        "markdown",
        """The dotted lines are the ±1.25 σ_eq entry bands. Because the regression forces
`X_T = 0`, today's s-score reduces to `s = −m / σ_eq`, equation (22).

### 2.5 The cross-section of s-scores today""",
    ),
    (
        "code",
        """scores = compute_s_scores(levels, parameters, regression.alphas,
                          modified=config.use_modified_s_score).dropna().sort_values()

figure, axis = plt.subplots(figsize=(9, 3.2))
colours = ["tab:green" if s < -config.s_open_long else
           "tab:red" if s > config.s_open_short else "lightgrey" for s in scores]
axis.bar(range(len(scores)), scores.values, color=colours)
for level in (-config.s_open_long, config.s_open_short):
    axis.axhline(level, color="black", ls="--", lw=0.8)
axis.set(xticks=range(len(scores)),
         xticklabels=[t.split(".")[0] for t in scores.index],
         ylabel="s-score", title="Green = buy to open, red = sell to open")
axis.tick_params(axis="x", rotation=90, labelsize=6)
plt.show()

print(f"tradeable after the kappa filter : {int(parameters['is_tradeable'].sum())}/{len(parameters)}")
print(f"long signals  : {(scores < -config.s_open_long).sum()}")
print(f"short signals : {(scores > config.s_open_short).sum()}")""",
    ),
    (
        "markdown",
        """### 2.6 How much does the κ filter actually reject?

Almost every name passes it, which is suspicious. Simulating pure random walks —
for which κ = 0, so none should ever qualify — shows why.""",
    ),
    (
        "code",
        """rates = pd.DataFrame(
    {"raw": {n: false_positive_rate(n, n_paths=1500) for n in [60, 120, 252, 504]},
     "Kendall-corrected": {n: false_positive_rate(n, n_paths=1500, bias_correction=True)
                           for n in [60, 120, 252, 504]}}
)
rates.index.name = "window length"
(rates * 100).round(1)""",
    ),
    (
        "markdown",
        """On the paper's 60-day window, roughly three quarters of random walks look like
week-long mean-reverters. OLS on a near-unit-root AR(1) is biased towards zero by
about `(1 + 3b) / T`, which is ~0.07 here.

Correcting the bias, however, collapses the gross Sharpe from 0.91 to 0.15 (see
the design table in the README). The signal is not really a bet on stationarity:
`s = −m/σ_eq` is a normalised short-term reversal measure, and repairing the
estimator removes the very distortion it was exploiting.

## 3. Full walk-forward backtest""",
    ),
    (
        "code",
        """%%time
hedged = run_strategy(returns, config)
unhedged = run_strategy(returns, config.variant(hedge_factor_exposure=False))""",
    ),
    (
        "code",
        """comparison = pd.DataFrame({
    "hedged": hedged.statistics,
    "unhedged": unhedged.statistics,
}).loc[["annualised_return", "annualised_volatility", "sharpe_ratio", "max_drawdown",
        "hit_rate", "annualised_turnover", "average_gross_exposure"]]

gross = pd.Series({
    "hedged": performance_statistics(hedged.backtest.gross_returns)["sharpe_ratio"],
    "unhedged": performance_statistics(unhedged.backtest.gross_returns)["sharpe_ratio"],
}, name="gross_sharpe")

pd.concat([gross.to_frame().T, comparison]).round(3)""",
    ),
    (
        "code",
        """figure, axis = plt.subplots(figsize=(10, 4))
hedged.backtest.net_equity.plot(ax=axis, lw=1.2, label="hedged, net of 5 bps")
unhedged.backtest.net_equity.plot(ax=axis, lw=1.2, label="unhedged, net of 5 bps")
hedged.backtest.gross_equity.plot(ax=axis, lw=1, ls="--", color="grey", label="hedged, gross")
axis.axhline(1.0, color="black", lw=0.8)
axis.set(ylabel="Growth of 1", title="The gap between the dashed and solid blue lines is the cost")
axis.legend(frameon=False)\nplt.show()""",
    ),
    (
        "markdown",
        """The hedge works exactly as advertised and still loses on a net basis: the
eigenportfolios are re-estimated daily, so the hedge basket churns even when no
signal changed, pushing turnover from 45 to 77 times a year.""",
    ),
    (
        "code",
        """print("max residual factor exposure after hedging:",
      hedged.diagnostics["factor_exposure_after_hedge"].max())
print("mean factor exposure before hedging       :",
      hedged.diagnostics["factor_exposure_before_hedge"].mean().round(3))

hedged.diagnostics[["universe_size", "n_tradeable", "n_long", "n_short",
                    "explained_variance", "median_half_life_days",
                    "gross_exposure"]].describe().loc[["mean", "min", "50%", "max"]].round(3)""",
    ),
    (
        "markdown",
        """## 4. Everything else

The remaining studies — entry thresholds, fixed versus variable factor counts,
cost sensitivity, trading time — take a few minutes each and are produced by:

```bash
python scripts/run_experiments.py
python scripts/make_figures.py
```

Their tables land in `results/` and are discussed in the README.""",
    ),
]


def build() -> dict:
    cells = []
    for kind, source in CELLS:
        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
        if kind == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cells.append(cell)

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    target = ROOT / "notebooks" / "01_research.ipynb"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(build(), indent=1))
    print(f"wrote {target}")
