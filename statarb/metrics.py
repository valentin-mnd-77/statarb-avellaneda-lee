"""Performance statistics.

Two conventions worth stating, because different choices are defensible and
silently mixing them is how Sharpe ratios drift:

* The Sharpe ratio uses the *arithmetic* mean of daily returns annualised by
  252, divided by the annualised standard deviation. Pairing a geometric
  (compounded) return with an arithmetic volatility understates the ratio by
  roughly half the variance, which is material for a levered long-short book.
* The annualised return reported next to it is the compounded one, because that
  is what an investor actually receives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from statarb.config import TRADING_DAYS_PER_YEAR


def performance_statistics(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    positions: pd.DataFrame | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """Summarise a strategy return series.

    Parameters:
        returns (pd.Series): Periodic strategy returns, net of costs.
        turnover (pd.Series | None): Daily one-way turnover, to report the
            annualised trading intensity and the implied cost drag.
        positions (pd.DataFrame | None): Held weights, to report average book size.
        risk_free_rate (float): Annual rate; the assignment ignores it.
        periods_per_year (int): Periods per year for annualisation.

    Returns:
        dict[str, float]: Named statistics.
    """
    returns = returns.dropna()
    if returns.empty:
        raise ValueError("no returns to evaluate")

    equity = (1.0 + returns).cumprod()
    n_years = len(returns) / periods_per_year

    total_return = float(equity.iloc[-1] - 1.0)
    annualised_return = (
        float((1.0 + total_return) ** (1.0 / n_years) - 1.0) if n_years > 0 else np.nan
    )
    annualised_volatility = float(returns.std(ddof=1) * np.sqrt(periods_per_year))

    arithmetic_annualised = float(returns.mean() * periods_per_year)
    sharpe = (
        (arithmetic_annualised - risk_free_rate) / annualised_volatility
        if annualised_volatility > 0
        else np.nan
    )

    downside = returns[returns < 0]
    downside_volatility = (
        float(downside.std(ddof=1) * np.sqrt(periods_per_year))
        if len(downside) > 1
        else np.nan
    )
    sortino = (
        (arithmetic_annualised - risk_free_rate) / downside_volatility
        if downside_volatility and downside_volatility > 0
        else np.nan
    )

    # Drawdown is measured from inception, hence the leading 1.0.
    equity_from_one = pd.concat([pd.Series([1.0]), equity.reset_index(drop=True)])
    drawdown = equity_from_one / equity_from_one.cummax() - 1.0
    max_drawdown = float(drawdown.min())

    statistics = {
        "total_return": total_return,
        "annualised_return": annualised_return,
        "annualised_volatility": annualised_volatility,
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino) if sortino == sortino else np.nan,
        "max_drawdown": max_drawdown,
        "calmar_ratio": float(annualised_return / abs(max_drawdown))
        if max_drawdown < 0
        else np.nan,
        "hit_rate": float((returns > 0).mean()),
        "skew": float(returns.skew()),
        "excess_kurtosis": float(returns.kurtosis()),
        "n_days": int(len(returns)),
    }

    if turnover is not None:
        aligned_turnover = turnover.reindex(returns.index).dropna()
        statistics["daily_turnover"] = float(aligned_turnover.mean())
        statistics["annualised_turnover"] = float(
            aligned_turnover.mean() * periods_per_year
        )

    if positions is not None:
        aligned_positions = positions.reindex(returns.index)
        statistics["average_gross_exposure"] = float(
            aligned_positions.abs().sum(axis=1).mean()
        )
        statistics["average_net_exposure"] = float(
            aligned_positions.sum(axis=1).mean()
        )
        statistics["average_positions"] = float(
            aligned_positions.ne(0.0).sum(axis=1).mean()
        )

    return statistics


def statistics_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Stack several statistic dictionaries into a comparison table.

    Parameters:
        results (dict[str, dict[str, float]]): Statistics keyed by strategy name.

    Returns:
        pd.DataFrame: Strategies as rows, statistics as columns.
    """
    table = pd.DataFrame(results).T
    table.index.name = "strategy"
    return table


def format_statistics_table(table: pd.DataFrame) -> pd.DataFrame:
    """Render a statistics table for display: percentages and ratios.

    Parameters:
        table (pd.DataFrame): Output of :func:`statistics_table`.

    Returns:
        pd.DataFrame: String-formatted copy.
    """
    percentage_columns = {
        "total_return",
        "annualised_return",
        "annualised_volatility",
        "max_drawdown",
        "hit_rate",
    }

    formatted = table.copy()
    for column in formatted.columns:
        if column in percentage_columns:
            formatted[column] = table[column].map(lambda value: f"{value:.2%}")
        elif column in {"n_days", "average_positions"}:
            formatted[column] = table[column].map(lambda value: f"{value:.0f}")
        else:
            formatted[column] = table[column].map(lambda value: f"{value:.2f}")

    return formatted
