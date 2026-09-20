"""Run every experiment reported in the README and save the tables to ``results/``.

Usage:
    python scripts/run_experiments.py

Each block answers one question:
  1. headline    - the paper's configuration, with and without the factor hedge
  2. design      - does each of the paper's design choices earn its place?
  3. thresholds  - robustness to the entry cutoff
  4. factors     - fixed factor count versus a variable explained-variance target
  5. costs       - how much of the alpha survives at different cost levels
  6. trading     - signals estimated in trading time (volume-adjusted returns)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from statarb.backtest import run_backtest  # noqa: E402
from statarb.config import StrategyConfig  # noqa: E402
from statarb.data import (  # noqa: E402
    compute_returns,
    coverage_report,
    load_prices,
    load_volume,
    volume_adjusted_returns,
)
from statarb.metrics import performance_statistics, statistics_table  # noqa: E402
from statarb.portfolio import apply_no_trade_band  # noqa: E402
from statarb.strategy import generate_signals, run_strategy  # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "results"
HEADLINE_COLUMNS = [
    "gross_sharpe",
    "sharpe_ratio",
    "annualised_return",
    "annualised_volatility",
    "max_drawdown",
    "annualised_turnover",
    "cost_drag",
    "average_gross_exposure",
]


def evaluate(signals, returns, config, label_extra=None):
    """Price a signal set under a configuration and return gross+net statistics."""
    weights = apply_no_trade_band(signals["weights"], config.no_trade_band)
    backtest = run_backtest(
        weights, returns, config.transaction_costs, config.execution_lag
    )
    gross = performance_statistics(backtest.gross_returns)
    net = performance_statistics(
        backtest.net_returns,
        turnover=backtest.turnover,
        positions=backtest.held_weights,
    )
    net["gross_sharpe"] = gross["sharpe_ratio"]
    net["gross_annualised_return"] = gross["annualised_return"]
    net["cost_drag"] = net["annualised_turnover"] * config.transaction_costs
    net["average_signals"] = float((signals["positions"] != 0).sum(axis=1).mean())
    if label_extra:
        net.update(label_extra)
    return net, backtest


def main() -> None:
    RESULTS.mkdir(exist_ok=True)

    prices = load_prices(DATA / "sx5e_underlyings.csv")
    volume = load_volume(DATA / "volume.csv")
    returns = compute_returns(prices)

    coverage_report(prices, returns).to_csv(RESULTS / "00_data_coverage.csv")

    base = StrategyConfig(no_trade_band=0.005)

    # ------------------------------------------------------------------
    print("1/6  headline")
    headline = {}
    curves = {}
    for label, config in {
        "paper (hedged)": base,
        "unhedged": base.variant(hedge_factor_exposure=False),
    }.items():
        started = time.time()
        signals = generate_signals(returns, config)
        statistics, backtest = evaluate(signals, returns, config)
        headline[label] = statistics
        curves[label] = backtest.net_equity
        if label == "paper (hedged)":
            signals["diagnostics"].to_csv(RESULTS / "01_diagnostics.csv")
            signals["s_scores"].to_csv(RESULTS / "01_s_scores.csv")
        print(f"     {label:<18} net SR {statistics['sharpe_ratio']:+.2f}"
              f"  ({time.time() - started:.0f}s)")
    statistics_table(headline).to_csv(RESULTS / "01_headline.csv")
    pd.DataFrame(curves).to_csv(RESULTS / "01_equity_curves.csv")

    # ------------------------------------------------------------------
    print("2/6  design choices")
    design_variants = {
        "paper": base,
        "no factor hedge": base.variant(hedge_factor_exposure=False),
        "no centring of m": base.variant(center_ou_means=False),
        "modified s-score": base.variant(use_modified_s_score=True),
        "symmetric exits (0.50)": base.variant(s_close_short=0.50),
        "execution lag 2": base.variant(execution_lag=2),
        "equal gross 100%": base.variant(weighting="equal_gross"),
        "no AR(1) bound": base.variant(max_ar1_coefficient=1.0),
        "Kendall bias correction": base.variant(bias_correction=True),
        "no no-trade band": base.variant(no_trade_band=0.0),
    }
    design = {}
    for label, config in design_variants.items():
        signals = generate_signals(returns, config)
        design[label], _ = evaluate(signals, returns, config)
        print(f"     {label:<24} gross SR {design[label]['gross_sharpe']:+.2f}"
              f"  net SR {design[label]['sharpe_ratio']:+.2f}")
    statistics_table(design)[HEADLINE_COLUMNS].to_csv(RESULTS / "02_design_choices.csv")

    # ------------------------------------------------------------------
    print("3/6  entry thresholds")
    thresholds = {}
    for cutoff in [1.00, 1.25, 1.50, 2.00]:
        config = base.variant(s_open_long=cutoff, s_open_short=cutoff)
        signals = generate_signals(returns, config)
        thresholds[f"s_open = {cutoff:.2f}"], _ = evaluate(signals, returns, config)
    statistics_table(thresholds)[HEADLINE_COLUMNS + ["average_signals"]].to_csv(
        RESULTS / "03_entry_thresholds.csv"
    )

    # ------------------------------------------------------------------
    print("4/6  factor selection")
    factor_variants = {"fixed m = 4": base}
    for target in [0.40, 0.55, 0.65, 0.75]:
        factor_variants[f"variable, {target:.0%} variance"] = base.variant(
            n_factors=None, target_variance=target
        )
    factors = {}
    factor_counts = {}
    for label, config in factor_variants.items():
        signals = generate_signals(returns, config)
        factors[label], _ = evaluate(signals, returns, config)
        factors[label]["mean_n_factors"] = float(
            signals["diagnostics"]["n_factors"].mean()
        )
        factors[label]["mean_explained_variance"] = float(
            signals["diagnostics"]["explained_variance"].mean()
        )
        factor_counts[label] = signals["diagnostics"]["n_factors"]
    statistics_table(factors)[
        HEADLINE_COLUMNS + ["mean_n_factors", "mean_explained_variance"]
    ].to_csv(RESULTS / "04_factor_selection.csv")
    pd.DataFrame(factor_counts).to_csv(RESULTS / "04_factor_counts.csv")

    # ------------------------------------------------------------------
    print("5/6  cost sensitivity")
    costs = {}
    for label, config in {
        "hedged": base,
        "unhedged": base.variant(hedge_factor_exposure=False),
    }.items():
        signals = generate_signals(returns, config)
        for rate in [0.0, 1e-4, 2e-4, 5e-4, 10e-4]:
            variant = config.variant(transaction_costs=rate)
            costs[f"{label}, {rate * 1e4:.0f} bps"], _ = evaluate(
                signals, returns, variant
            )
    statistics_table(costs)[HEADLINE_COLUMNS].to_csv(RESULTS / "05_cost_sensitivity.csv")

    # ------------------------------------------------------------------
    print("6/6  trading time")
    trading = {}
    for label, config in {
        "hedged": base,
        "unhedged": base.variant(hedge_factor_exposure=False),
    }.items():
        adjusted = volume_adjusted_returns(
            returns, volume, config.volume_window, config.max_volume_adjustment
        ).dropna(how="all")
        result_calendar = run_strategy(returns, config)
        result_trading = run_strategy(returns, config, signal_returns=adjusted)
        trading[f"{label}, calendar time"] = result_calendar.statistics
        trading[f"{label}, trading time"] = result_trading.statistics
        for key in (f"{label}, calendar time", f"{label}, trading time"):
            trading[key]["cost_drag"] = (
                trading[key]["annualised_turnover"] * config.transaction_costs
            )
            trading[key]["gross_sharpe"] = float("nan")
    statistics_table(trading)[
        [column for column in HEADLINE_COLUMNS if column != "gross_sharpe"]
    ].to_csv(RESULTS / "06_trading_time.csv")

    print(f"\nTables written to {RESULTS}")


if __name__ == "__main__":
    main()
