"""The walk-forward loop that ties every component together.

On each trading day, with information available at that close only:

1. build the 252-day estimation window and its investable universe;
2. diagonalise the correlation matrix and form the eigenportfolio factors;
3. regress the last 60 days of returns on those factors;
4. cumulate the residuals and fit the AR(1) / Ornstein-Uhlenbeck model;
5. keep the assets that mean-revert fast enough and score their dislocation;
6. advance the position state machine, size the book and hedge its factor
   exposure.

Nothing in the loop reads a return dated after the rebalance date, so the
simulation is walk-forward by construction rather than by convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from statarb.backtest import BacktestResult, run_backtest
from statarb.config import StrategyConfig
from statarb.data import prepare_estimation_window
from statarb.factor_model import extract_factors, fit_factor_regression
from statarb.metrics import performance_statistics
from statarb.ou import estimate_ou_parameters
from statarb.portfolio import (
    apply_no_trade_band,
    hedge_factor_exposure,
    rescale_gross_exposure,
    residual_factor_exposure,
    size_positions,
)
from statarb.signals import SignalThresholds, update_positions


@dataclass
class StrategyResult:
    """Complete output of one backtest run."""

    config: StrategyConfig
    weights: pd.DataFrame
    """Target weights by signal date."""

    positions: pd.DataFrame
    """Raw signals in {-1, 0, +1} by signal date."""

    s_scores: pd.DataFrame
    """S-score of every asset on every rebalance date."""

    diagnostics: pd.DataFrame
    """Per-date state of the model: universe size, factors, exposures."""

    backtest: BacktestResult
    statistics: dict[str, float] = field(default_factory=dict)

    def summary(self) -> pd.Series:
        """One-line view of the run."""
        return pd.Series(self.statistics)


def _thresholds(config: StrategyConfig) -> SignalThresholds:
    return SignalThresholds(
        open_long=config.s_open_long,
        open_short=config.s_open_short,
        close_long=config.s_close_long,
        close_short=config.s_close_short,
    )


def generate_signals(
    signal_returns: pd.DataFrame,
    config: StrategyConfig,
    verbose: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run the estimation loop and produce weights, positions and diagnostics.

    Parameters:
        signal_returns (pd.DataFrame): Returns used for estimation. Calendar
            returns normally; volume-adjusted returns under trading time.
        config (StrategyConfig): Strategy parameters.
        verbose (bool): Print progress every 250 rebalance dates.

    Returns:
        dict[str, pd.DataFrame]: ``weights``, ``positions``, ``s_scores`` and
            ``diagnostics``, all indexed by rebalance date.
    """
    thresholds = _thresholds(config)

    open_positions: dict[str, float] = {}
    weight_history: dict[pd.Timestamp, pd.Series] = {}
    position_history: dict[pd.Timestamp, dict[str, float]] = {}
    score_history: dict[pd.Timestamp, pd.Series] = {}
    diagnostic_rows: list[dict] = []

    for processed, rebalance_date in enumerate(signal_returns.index):
        window, window_diagnostics = prepare_estimation_window(
            returns=signal_returns,
            rebalance_date=rebalance_date,
            lookback=config.estimation_window,
            min_coverage=config.min_coverage,
        )

        # Skip until a full window of history exists: a shorter one would make
        # the correlation matrix and the PCA unreliable.
        if (
            window_diagnostics["row_count"] < config.estimation_window
            or window.shape[1] == 0
        ):
            continue

        decomposition = extract_factors(
            window,
            n_factors=config.n_factors,
            target_variance=config.target_variance,
        )

        # Loadings on the short window, factors from the long one: the paper
        # wants betas that reflect the current regime and factors that are
        # stably estimated.
        regression_window = window.loc[:, decomposition.eigenportfolios.index].iloc[
            -config.ou_window :
        ]
        regression = fit_factor_regression(
            regression_window,
            decomposition.factors.iloc[-config.ou_window :],
        )

        ou_parameters = estimate_ou_parameters(
            regression.cumulative_residuals,
            dt=config.dt,
            min_mean_reversion_speed=config.min_mean_reversion_speed,
            max_ar1_coefficient=config.max_ar1_coefficient,
            center_means=config.center_ou_means,
            bias_correction=config.bias_correction,
        )

        s_scores = compute_scores(regression, ou_parameters, config)
        score_history[rebalance_date] = s_scores

        open_positions = update_positions(open_positions, s_scores, thresholds)
        position_history[rebalance_date] = open_positions.copy()

        weights = size_positions(
            open_positions,
            weighting=config.weighting,
            notional_per_position=config.notional_per_position,
        )

        exposure_before = residual_factor_exposure(weights, regression.betas)
        if config.hedge_factor_exposure and not weights.empty:
            weights = hedge_factor_exposure(
                weights, regression.betas, decomposition.eigenportfolios
            )
            if config.weighting == "equal_gross":
                weights = rescale_gross_exposure(weights, target=1.0)
        exposure_after = residual_factor_exposure(weights, regression.betas)

        weight_history[rebalance_date] = weights

        diagnostic_rows.append(
            {
                "date": rebalance_date,
                "universe_size": window.shape[1],
                "n_factors": decomposition.n_factors,
                "explained_variance": decomposition.total_explained_variance,
                "n_tradeable": int(ou_parameters["is_tradeable"].sum()),
                "median_kappa": float(
                    ou_parameters.loc[ou_parameters["is_tradeable"], "kappa"].median()
                ),
                "median_half_life_days": float(
                    ou_parameters.loc[
                        ou_parameters["is_tradeable"], "half_life_days"
                    ].median()
                ),
                "mean_r_squared": float(regression.r_squared.mean()),
                "n_long": sum(1 for value in open_positions.values() if value > 0),
                "n_short": sum(1 for value in open_positions.values() if value < 0),
                "gross_exposure": float(weights.abs().sum()),
                "net_exposure": float(weights.sum()),
                "factor_exposure_before_hedge": float(np.abs(exposure_before).max())
                if exposure_before.size
                else 0.0,
                "factor_exposure_after_hedge": float(np.abs(exposure_after).max())
                if exposure_after.size
                else 0.0,
            }
        )

        if verbose and processed % 250 == 0:
            print(f"  {rebalance_date.date()}  positions={len(open_positions)}")

    if not weight_history:
        raise ValueError(
            "no rebalance date had a full estimation window; check the data range"
        )

    weights_frame = (
        pd.DataFrame(weight_history).T.sort_index().fillna(0.0).astype(float)
    )
    positions_frame = (
        pd.DataFrame(position_history).T.sort_index().fillna(0.0).astype(float)
    )
    scores_frame = pd.DataFrame(score_history).T.sort_index()
    diagnostics_frame = pd.DataFrame(diagnostic_rows).set_index("date")

    return {
        "weights": weights_frame,
        "positions": positions_frame,
        "s_scores": scores_frame,
        "diagnostics": diagnostics_frame,
    }


def compute_scores(regression, ou_parameters, config: StrategyConfig) -> pd.Series:
    """S-score for the current date, pure or drift-adjusted per the config.

    Parameters:
        regression (FactorRegression): Output of the factor regression.
        ou_parameters (pd.DataFrame): Fitted O-U parameters.
        config (StrategyConfig): Strategy parameters.

    Returns:
        pd.Series: S-scores, NaN for assets that failed the filters.
    """
    from statarb.signals import compute_s_scores

    return compute_s_scores(
        regression.cumulative_residuals,
        ou_parameters,
        alphas=regression.alphas,
        modified=config.use_modified_s_score,
    )


def run_strategy(
    returns: pd.DataFrame,
    config: StrategyConfig,
    signal_returns: pd.DataFrame | None = None,
    verbose: bool = False,
) -> StrategyResult:
    """Full pipeline: generate signals, then price them on realised returns.

    Parameters:
        returns (pd.DataFrame): Calendar-time returns. The PnL is always
            computed on these, never on transformed ones.
        config (StrategyConfig): Strategy parameters.
        signal_returns (pd.DataFrame | None): Returns used for estimation when
            they differ from the PnL returns, i.e. under trading time.
        verbose (bool): Print progress.

    Returns:
        StrategyResult: Weights, diagnostics, PnL and statistics.
    """
    estimation_returns = returns if signal_returns is None else signal_returns

    signals = generate_signals(estimation_returns, config, verbose=verbose)

    # Cost-aware filtering happens on the weight schedule, after the model has
    # spoken: the signal is a research question, execution is an engineering one.
    traded_weights = apply_no_trade_band(signals["weights"], config.no_trade_band)

    backtest_result = run_backtest(
        weights=traded_weights,
        returns=returns,
        transaction_costs=config.transaction_costs,
        execution_lag=config.execution_lag,
    )

    statistics = performance_statistics(
        backtest_result.net_returns,
        turnover=backtest_result.turnover,
        positions=backtest_result.held_weights,
    )

    return StrategyResult(
        config=config,
        weights=traded_weights,
        positions=signals["positions"],
        s_scores=signals["s_scores"],
        diagnostics=signals["diagnostics"],
        backtest=backtest_result,
        statistics=statistics,
    )
