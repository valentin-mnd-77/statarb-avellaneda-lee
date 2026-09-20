"""Backtest engine: align a weight schedule with returns and compute the PnL.

The only subtle part is the timing. A weight row labelled ``t`` was computed
from information available at the close of day ``t``. With ``execution_lag = 1``
the trade happens at that close and the position earns the return of day t+1,
which is the paper's convention. With ``execution_lag = 2`` the trade happens at
the close of t+1 and the position earns the return of t+2, which is what the
Politecnico assignment specifies. Getting this wrong is worth a spurious day of
mean-reversion PnL, which on a one-day-horizon strategy is a large fraction of
the total.

Costs are charged on realised turnover, sum_i |w_i,t - w_i,t-1|, at the all-in
one-way rate. That includes the cost of scaling an existing position, which is
why the sizing scheme in :mod:`statarb.portfolio` matters.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    """Everything the performance layer needs, in one object."""

    gross_returns: pd.Series
    """Daily returns before costs."""

    net_returns: pd.Series
    """Daily returns after transaction costs."""

    turnover: pd.Series
    """Daily one-way turnover, sum of absolute weight changes."""

    held_weights: pd.DataFrame
    """Weights actually held on each date, after the execution lag."""

    @property
    def gross_equity(self) -> pd.Series:
        """Cumulative gross performance, starting at 1."""
        return _to_equity(self.gross_returns)

    @property
    def net_equity(self) -> pd.Series:
        """Cumulative net performance, starting at 1."""
        return _to_equity(self.net_returns)


def _to_equity(returns: pd.Series) -> pd.Series:
    """Compound a return series into an equity curve anchored at 1.

    The leading 1.0 matters: without it the first day's return is invisible to
    any statistic computed as ``equity[-1] / equity[0] - 1``, and the drawdown
    from inception is understated.

    Parameters:
        returns (pd.Series): Periodic returns.

    Returns:
        pd.Series: Equity curve with one extra leading point at 1.0.
    """
    equity = (1.0 + returns).cumprod()
    if equity.empty:
        return equity

    if len(equity) > 1:
        start = equity.index[0] - (equity.index[1] - equity.index[0])
    else:
        start = equity.index[0]
    return pd.concat([pd.Series([1.0], index=[start]), equity])


def align_weights(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    execution_lag: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Line up a weight schedule with the returns it is entitled to earn.

    Parameters:
        weights (pd.DataFrame): Target weights indexed by signal date.
        returns (pd.DataFrame): Asset returns.
        execution_lag (int): Days between the signal and the first return earned.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Held weights and matching returns,
            on a common index and asset set.
    """
    if weights.empty or weights.shape[1] == 0:
        raise ValueError("weights must contain at least one asset")

    shared_assets = weights.columns.intersection(returns.columns)
    if shared_assets.empty:
        raise ValueError("weights and returns share no asset")

    held = (
        weights.loc[:, shared_assets]
        .reindex(returns.index)
        .ffill()
        .shift(execution_lag)
        .dropna(axis=0, how="all")
        .fillna(0.0)
    )

    if held.empty:
        raise ValueError("weights and returns share no investable date")

    return held, returns.loc[held.index, shared_assets]


def run_backtest(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    transaction_costs: float = 0.0,
    execution_lag: int = 1,
) -> BacktestResult:
    """Compute gross and net PnL from a weight schedule.

    Parameters:
        weights (pd.DataFrame): Target weights indexed by signal date.
        returns (pd.DataFrame): Asset returns.
        transaction_costs (float): All-in one-way rate, e.g. 5e-4 for 5 bps.
        execution_lag (int): See module docstring.

    Returns:
        BacktestResult: Return series, turnover and realised weights.
    """
    held, aligned_returns = align_weights(weights, returns, execution_lag)

    # A non-zero weight on an asset with a missing return would be silently
    # dropped by the skipna sum, so make that case visible instead.
    exposure_without_return = (held.ne(0.0) & aligned_returns.isna()).to_numpy().sum()
    if exposure_without_return:
        raise ValueError(
            f"{exposure_without_return} position-days have a weight but no "
            f"return; check the investable universe construction"
        )

    gross_returns = held.multiply(aligned_returns).sum(axis=1)

    # Turnover on the first row is measured against an empty book, so the
    # initial build-up of the portfolio is correctly charged.
    previous = held.shift().fillna(0.0)
    turnover = (held - previous).abs().sum(axis=1)

    net_returns = gross_returns - transaction_costs * turnover

    return BacktestResult(
        gross_returns=gross_returns,
        net_returns=net_returns,
        turnover=turnover,
        held_weights=held,
    )
