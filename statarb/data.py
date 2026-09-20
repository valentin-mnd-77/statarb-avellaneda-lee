"""Loading and shaping the market data.

Everything that touches raw files lives here, so the rest of the package only
ever sees clean DataFrames indexed by date with assets as columns.

Two data-hygiene points are handled explicitly rather than silently:

* Prices are forward-filled because the universe spans several exchanges with
  different holiday calendars. A German name on a German holiday would
  otherwise look like a missing observation on a day the rest of Europe traded.
  The cost is an artificial zero return on those days, which is reported by
  :func:`coverage_report` rather than hidden.
* Estimation windows are strictly point-in-time: ``returns.loc[:date]`` never
  looks past the rebalance date.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_prices(path: str | Path, forward_fill: bool = True) -> pd.DataFrame:
    """Load the total-return index levels.

    Parameters:
        path (str | Path): CSV with a ``Date`` column and one column per ticker.
        forward_fill (bool): Carry the last observation forward across local
            market holidays. Leading gaps (a stock that joined the index later)
            stay missing.

    Returns:
        pd.DataFrame: Price levels, dates as index, tickers as columns.
    """
    prices = pd.read_csv(path, index_col="Date", parse_dates=True).sort_index()
    return prices.ffill() if forward_fill else prices


def load_volume(path: str | Path) -> pd.DataFrame:
    """Load daily traded volume.

    The vendor file encodes gaps as ``#N/A N/A``, which pandas already treats as
    missing; the dtypes are asserted numeric so that a change of encoding fails
    loudly instead of silently producing object columns.

    Parameters:
        path (str | Path): CSV with a ``Date`` column and one column per ticker.

    Returns:
        pd.DataFrame: Volumes, dates as index, tickers as columns.
    """
    volume = pd.read_csv(path, index_col="Date", parse_dates=True).sort_index()

    non_numeric = [
        column
        for column in volume.columns
        if not pd.api.types.is_numeric_dtype(volume[column])
    ]
    if non_numeric:
        raise ValueError(
            f"volume columns are not numeric: {non_numeric[:5]}; check the "
            f"missing-value encoding in {path}"
        )

    return volume


def load_ticker_details(path: str | Path) -> pd.DataFrame:
    """Load the ticker reference table (name, Bloomberg code, GICS sector).

    Parameters:
        path (str | Path): CSV whose second column holds the ticker.

    Returns:
        pd.DataFrame: Reference data indexed by ticker.
    """
    details = pd.read_csv(path, index_col=1)
    return details.drop(columns=[column for column in details.columns if column == "0"])


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns, dropping the unusable first row.

    Parameters:
        prices (pd.DataFrame): Total-return index levels.

    Returns:
        pd.DataFrame: Simple returns.
    """
    return prices.pct_change().iloc[1:]


def prepare_estimation_window(
    returns: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    lookback: int,
    min_coverage: float = 0.95,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the trailing window available at the close of ``rebalance_date``.

    Assets whose history is too sparse over the window are dropped rather than
    imputed wholesale: a stock that joined the index six months ago has no
    meaningful correlation estimate. The residual gaps of retained assets are
    filled with zero, which is the least-committal choice for a return series
    and matches how the strategy would treat a non-trading day in production.

    Parameters:
        returns (pd.DataFrame): Full return history.
        rebalance_date (pd.Timestamp): Last date included, inclusive.
        lookback (int): Target number of rows.
        min_coverage (float): Minimum share of observed returns per asset.

    Returns:
        tuple[pd.DataFrame, dict]: The window, and diagnostics describing how
            many rows and assets survived.
    """
    trailing = (
        returns.loc[:rebalance_date]
        .dropna(axis=0, how="all")
        .dropna(axis=1, how="all")
        .iloc[-lookback:]
    )

    coverage = trailing.notna().mean(axis=0)
    eligible = coverage.index[coverage >= min_coverage]
    window = trailing.loc[:, eligible].fillna(0.0)

    diagnostics = {
        "row_count": trailing.shape[0],
        "assets_before_filter": trailing.shape[1],
        "assets_after_filter": window.shape[1],
        "dropped_assets": list(coverage.index[coverage < min_coverage]),
        "imputed_fraction": float(
            trailing.loc[:, eligible].isna().to_numpy().mean()
        )
        if len(eligible)
        else 0.0,
    }

    return window, diagnostics


def volume_adjusted_returns(
    returns: pd.DataFrame,
    volume: pd.DataFrame,
    trailing_window: int = 60,
    max_adjustment: float = 10.0,
) -> pd.DataFrame:
    """Rescale returns into "trading time" (section 6 of the paper).

    Equation (20): R_bar_{i,t} = R_{i,t} * <dV_i> / V_{i,t}, where <dV_i> is the
    trailing average daily volume. A move on a quiet day is amplified, a move on
    a heavy day is damped, so the strategy is less willing to fade a price
    change that many participants agreed on.

    The trailing average includes day t, which is legitimate: the volume of day
    t is known at the close of day t, the same moment the signal is formed.

    Parameters:
        returns (pd.DataFrame): Calendar-time simple returns.
        volume (pd.DataFrame): Daily traded volume.
        trailing_window (int): Days in the average volume.
        max_adjustment (float): Cap on the ratio, which diverges when volume
            collapses towards zero.

    Returns:
        pd.DataFrame: Adjusted returns on the common index and columns. The
            first ``trailing_window - 1`` rows are missing by construction.
    """
    shared_dates = returns.index.intersection(volume.index)
    shared_assets = returns.columns.intersection(volume.columns)

    aligned_returns = returns.loc[shared_dates, shared_assets]
    aligned_volume = volume.loc[shared_dates, shared_assets]

    average_volume = aligned_volume.rolling(
        trailing_window, min_periods=trailing_window
    ).mean()

    # A reported volume of zero is a data gap, not a real quiet day; clipping at
    # one share keeps the ratio finite and the cap below handles the rest.
    adjustment = average_volume / aligned_volume.clip(lower=1.0)
    adjustment = adjustment.clip(upper=max_adjustment)

    return aligned_returns * adjustment


def coverage_report(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Per-asset data-quality summary, to be looked at before trusting a backtest.

    Parameters:
        prices (pd.DataFrame): Forward-filled price levels.
        returns (pd.DataFrame): Simple returns.

    Returns:
        pd.DataFrame: First observation, missing count and the share of exactly
            zero returns, which measures how much the forward fill contributes.
    """
    return pd.DataFrame(
        {
            "first_observation": prices.apply(lambda column: column.first_valid_index()),
            "missing_returns": returns.isna().sum(),
            "zero_return_share": (returns == 0).mean(),
            "annualised_volatility": returns.std() * np.sqrt(252),
        }
    ).sort_values("first_observation")
