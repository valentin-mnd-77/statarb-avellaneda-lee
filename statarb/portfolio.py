"""Turning trading signals into dollar weights.

Two things happen here.

**Sizing.** Each open signal receives a notional. The paper (section 5) fixes a
stock-independent Lambda so a position's size is set once when it opens and is
never revisited; the alternative is to rescale daily to a constant gross
exposure. The distinction matters for costs: under constant gross, the arrival
of a single new signal reprices every existing position and books turnover that
no trader would actually pay. Footnote 8 of the paper says as much.

**Hedging.** Opening a long in stock i means buying one dollar of the stock and
selling beta_ij dollars of each eigenportfolio j. Since the eigenportfolios are
themselves baskets of the same stocks, the hedge folds back into stock weights:

    w = q - P (P^T B)^{-1} B^T q      (exact projection)
    w = q - P B^T q                   (the paper's direct form)

with q the raw stock positions, B the (N x m) loadings and P the (N x m)
eigenportfolio weights. The two coincide because B^T P = I whenever the
loadings and the eigenportfolios come from the same sample: writing Sigma for
the covariance matrix, F = R P gives Cov(R, F) = Sigma P and Var(F) = diag(lambda),
so B = Sigma P diag(lambda)^{-1} and B^T P = diag(lambda)^{-1} P^T Sigma P = I.
Here the loadings are fitted on 60 days while the eigenportfolios come from 252,
so the identity holds only approximately and the residual factor exposure is
reported as a diagnostic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def size_positions(
    positions: dict[str, float],
    weighting: str = "fixed_notional",
    notional_per_position: float = 0.05,
) -> pd.Series:
    """Give each open signal a dollar notional.

    Parameters:
        positions (dict[str, float]): Signals in {-1, +1}, flat assets absent.
        weighting (str): ``fixed_notional`` gives every position the same size
            and lets gross exposure float; ``equal_gross`` rescales to 100%
            gross every day.
        notional_per_position (float): Lambda, used by ``fixed_notional``.

    Returns:
        pd.Series: Signed weights indexed by asset; empty when flat.
    """
    if not positions:
        return pd.Series(dtype=float)

    signals = pd.Series(positions, dtype=float)

    if weighting == "fixed_notional":
        return signals * notional_per_position

    if weighting == "equal_gross":
        gross = signals.abs().sum()
        return signals / gross if gross > 0 else signals

    raise ValueError(f"unknown weighting scheme: {weighting}")


def hedge_factor_exposure(
    weights: pd.Series,
    betas: pd.DataFrame,
    eigenportfolios: pd.DataFrame,
    exact: bool = False,
) -> pd.Series:
    """Neutralise the portfolio's exposure to the statistical factors.

    Parameters:
        weights (pd.Series): Raw stock weights from :func:`size_positions`.
        betas (pd.DataFrame): Factor loadings, assets as index.
        eigenportfolios (pd.DataFrame): Eigenportfolio weights, assets as index.
        exact (bool): Use the exact projection, which forces the residual factor
            exposure to zero even when the loadings and the eigenportfolios come
            from different estimation windows.

    Returns:
        pd.Series: Hedged weights over the union of traded and hedging assets.
    """
    if weights.empty:
        return weights

    universe = eigenportfolios.index
    aligned_weights = weights.reindex(universe).fillna(0.0)
    loadings = betas.reindex(universe).fillna(0.0).to_numpy(dtype=float)
    hedging_basket = eigenportfolios.to_numpy(dtype=float)

    # Portfolio exposure to each factor: b_j = sum_i w_i beta_ij.
    factor_exposure = loadings.T @ aligned_weights.to_numpy(dtype=float)

    if exact:
        gram = loadings.T @ hedging_basket
        try:
            factor_exposure = np.linalg.solve(gram, factor_exposure)
        except np.linalg.LinAlgError:
            factor_exposure = np.linalg.lstsq(gram, factor_exposure, rcond=None)[0]

    hedged = aligned_weights.to_numpy(dtype=float) - hedging_basket @ factor_exposure

    return pd.Series(hedged, index=universe)


def residual_factor_exposure(
    weights: pd.Series,
    betas: pd.DataFrame,
) -> np.ndarray:
    """Portfolio beta on each factor, which the hedge is meant to drive to zero.

    Parameters:
        weights (pd.Series): Portfolio weights.
        betas (pd.DataFrame): Factor loadings.

    Returns:
        np.ndarray: One exposure per factor.
    """
    universe = betas.index
    aligned = weights.reindex(universe).fillna(0.0).to_numpy(dtype=float)
    return betas.to_numpy(dtype=float).T @ aligned


def rescale_gross_exposure(weights: pd.Series, target: float = 1.0) -> pd.Series:
    """Scale weights to a target gross exposure, leaving factor neutrality intact.

    Parameters:
        weights (pd.Series): Portfolio weights.
        target (float): Desired sum of absolute weights.

    Returns:
        pd.Series: Rescaled weights.
    """
    gross = weights.abs().sum()
    return weights * (target / gross) if gross > 0 else weights


def apply_no_trade_band(
    target_weights: pd.DataFrame,
    band: float,
) -> pd.DataFrame:
    """Suppress weight changes smaller than ``band``, per asset and per day.

    The factor hedge is re-estimated every day, so even a book whose signals did
    not change sees every weight move by a few basis points as the loadings and
    the eigenportfolios drift. Paying the spread on that noise is pure cost. A
    no-trade band holds the existing weight until the target has moved far
    enough to be worth the round trip, which is the simplest form of
    cost-aware rebalancing.

    A position that closes (target exactly zero) is always executed: leaving a
    stale position open because the band was not breached would change the
    strategy, not just its implementation.

    Parameters:
        target_weights (pd.DataFrame): Desired weights by date.
        band (float): Minimum absolute weight change worth trading. Zero
            reproduces the unfiltered schedule.

    Returns:
        pd.DataFrame: Weights actually carried, same shape as the input.
    """
    if band <= 0:
        return target_weights

    targets = target_weights.to_numpy(dtype=float)
    held = np.zeros_like(targets)
    current = np.zeros(targets.shape[1])

    for row in range(targets.shape[0]):
        desired = targets[row]
        # Trade when the move is large enough, or when the target is flat.
        should_trade = (np.abs(desired - current) >= band) | (desired == 0.0)
        current = np.where(should_trade, desired, current)
        held[row] = current

    return pd.DataFrame(
        held, index=target_weights.index, columns=target_weights.columns
    )
