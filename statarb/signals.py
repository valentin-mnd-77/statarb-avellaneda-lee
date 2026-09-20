"""From O-U parameters to trading signals.

Two steps: turn the current dislocation into a dimensionless s-score, then run
the entry/exit state machine that converts scores into positions.

The state machine is the part of the strategy that carries memory. A position
opened when the score crossed 1.25 stays on until the score comes back inside
the closing band, which may take days. That path dependence is why the backtest
has to be run sequentially over dates rather than vectorised across them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from statarb.config import TRADING_DAYS_PER_YEAR

FLAT = 0.0
LONG = 1.0
SHORT = -1.0


@dataclass(frozen=True)
class SignalThresholds:
    """Entry and exit cutoffs on the s-score."""

    open_long: float = 1.25
    open_short: float = 1.25
    close_long: float = 0.50
    close_short: float = 0.75


def compute_s_scores(
    cumulative_residuals: pd.DataFrame,
    ou_parameters: pd.DataFrame,
    alphas: pd.Series | None = None,
    modified: bool = False,
) -> pd.Series:
    """S-score of every asset at the last date of the window.

    Equation (15): s = (X(t) - m) / sigma_eq, the dislocation measured in
    equilibrium standard deviations. Because the regression forces the residuals
    to sum to zero, X at the last date is zero and this reduces to -m/sigma_eq,
    which is equation (22).

    Equation (17) adds the drift: s_mod = s - alpha / (kappa sigma_eq). A stock
    whose residual drifts upward becomes harder to short and easier to buy, so
    the modified score behaves like a momentum overlay on the mean-reversion
    signal.

    Parameters:
        cumulative_residuals (pd.DataFrame): X series over the O-U window.
        ou_parameters (pd.DataFrame): Output of :func:`statarb.ou.estimate_ou_parameters`.
        alphas (pd.Series | None): *Daily* regression intercepts. Required when
            ``modified`` is True; they are annualised here (alpha = intercept
            times 252, per the appendix) so that alpha and kappa share units.
        modified (bool): Whether to apply the drift correction.

    Returns:
        pd.Series: S-score per asset; NaN where the O-U fit was rejected.
    """
    last_level = cumulative_residuals.iloc[-1]
    aligned = ou_parameters.reindex(last_level.index)

    with np.errstate(divide="ignore", invalid="ignore"):
        s_score = (last_level - aligned["m"]) / aligned["sigma_eq"]

    if modified:
        if alphas is None:
            raise ValueError("the modified s-score needs the regression alphas")
        annualised_alpha = alphas.reindex(last_level.index) * TRADING_DAYS_PER_YEAR
        with np.errstate(divide="ignore", invalid="ignore"):
            drift_correction = annualised_alpha / (aligned["kappa"] * aligned["sigma_eq"])
        s_score = s_score - drift_correction

    # An asset that failed the kappa or b filter has no usable score.
    tradeable = aligned["is_tradeable"].reindex(last_level.index).astype("boolean")
    return s_score.where(tradeable.fillna(False).astype(bool), np.nan)


def update_positions(
    current_positions: dict[str, float],
    s_scores: pd.Series,
    thresholds: SignalThresholds,
) -> dict[str, float]:
    """Advance the position state machine by one day.

    Rules, symmetric in structure but asymmetric in the closing cutoffs:

    * flat  -> long  when s < -open_long
    * flat  -> short when s > +open_short
    * long  -> flat  when s > -close_long
    * short -> flat  when s < +close_short

    Assets whose score is missing this day have failed the O-U filter. Any
    position held in them is closed: the model that justified the trade no
    longer holds, and carrying an unmonitorable position is the worse of the two
    errors. Assets that left the universe entirely are closed by the same rule.

    Parameters:
        current_positions (dict[str, float]): Yesterday's positions, +1/-1/0.
        s_scores (pd.Series): Today's s-scores, NaN where untradeable.
        thresholds (SignalThresholds): Entry and exit cutoffs.

    Returns:
        dict[str, float]: Today's positions, holding only non-zero entries.
    """
    new_positions: dict[str, float] = {}

    for asset, score in s_scores.items():
        previous = current_positions.get(asset, FLAT)

        if not np.isfinite(score):
            continue  # filter failed today: stay flat / close out.

        if previous == LONG:
            # Hold the long until the residual has reverted close enough.
            if score <= -thresholds.close_long:
                new_positions[asset] = LONG
        elif previous == SHORT:
            if score >= thresholds.close_short:
                new_positions[asset] = SHORT
        else:
            if score < -thresholds.open_long:
                new_positions[asset] = LONG
            elif score > thresholds.open_short:
                new_positions[asset] = SHORT

    return new_positions


def run_state_machine(
    s_scores: pd.DataFrame,
    thresholds: SignalThresholds,
) -> pd.DataFrame:
    """Run the state machine over a whole history of s-scores.

    Convenience wrapper used for single-asset illustrations and for the
    threshold sensitivity study, where the s-scores are already known and only
    the cutoffs change.

    Parameters:
        s_scores (pd.DataFrame): S-scores, dates as index, assets as columns.
        thresholds (SignalThresholds): Entry and exit cutoffs.

    Returns:
        pd.DataFrame: Positions in {-1, 0, +1}, same shape as the input.
    """
    positions: dict[str, float] = {}
    history = {}

    for date, row in s_scores.iterrows():
        positions = update_positions(positions, row, thresholds)
        history[date] = positions.copy()

    return (
        pd.DataFrame(history)
        .T.reindex(columns=s_scores.columns)
        .fillna(0.0)
        .astype(float)
    )
