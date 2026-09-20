"""Ornstein-Uhlenbeck estimation on the cumulative residuals.

The paper models the cumulative idiosyncratic return of each stock as

    dX(t) = kappa (m - X(t)) dt + sigma dW(t),

whose exact discretisation over a step dt is an AR(1),

    X_{n+1} = a + b X_n + zeta,   b = exp(-kappa dt),   a = m (1 - b),
    Var(zeta) = sigma^2 (1 - b^2) / (2 kappa).

Inverting those relations (appendix, equation 21) gives

    kappa   = -log(b) / dt
    m       = a / (1 - b)
    sigma   = sqrt(Var(zeta) * 2 kappa / (1 - b^2))
    sigma_eq= sqrt(Var(zeta) / (1 - b^2)) = sigma / sqrt(2 kappa)

The estimation is run for all assets at once: an AR(1) is a univariate OLS with
a closed form, so the whole cross-section is three column-wise reductions rather
than a loop of regressions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from statarb.config import TRADING_DAYS_PER_YEAR


def estimate_ou_parameters(
    cumulative_residuals: pd.DataFrame,
    dt: float = 1.0 / TRADING_DAYS_PER_YEAR,
    min_mean_reversion_speed: float = TRADING_DAYS_PER_YEAR / 30,
    max_ar1_coefficient: float = 0.9672,
    center_means: bool = True,
    bias_correction: bool = False,
) -> pd.DataFrame:
    """Fit the AR(1) and map it to O-U parameters for every asset.

    Parameters:
        cumulative_residuals (pd.DataFrame): X_k series over the O-U window,
            dates as index, assets as columns.
        dt (float): Time step in years.
        min_mean_reversion_speed (float): Reject assets below this kappa.
        max_ar1_coefficient (float): Reject assets whose b exceeds this bound.
            Together with b > 0 this is the appendix's validity domain.
        center_means (bool): Subtract the cross-sectional average of m,
            equation (18). Only assets passing the b bounds contribute to the
            average, otherwise a single divergent m would poison it.
        bias_correction (bool): Apply Kendall's small-sample correction to b.
            OLS on a near-unit-root AR(1) is biased downward by roughly
            (1 + 3b) / T, which on a 60-day window is about 0.07 -- enough to
            make a random walk look like it reverts in a week. See
            :func:`false_positive_rate` for the measured size of the problem.

    Returns:
        pd.DataFrame: One row per asset with columns ``a``, ``b``,
            ``var_zeta``, ``kappa``, ``m``, ``m_raw``, ``sigma``, ``sigma_eq``,
            ``half_life`` (in years), ``half_life_days`` and ``is_tradeable``.
    """
    levels = cumulative_residuals.to_numpy(dtype=float)
    n_observations = levels.shape[0]

    if n_observations < 4:
        raise ValueError(
            f"need at least 4 points to fit an AR(1), got {n_observations}"
        )

    # Regress X_{n+1} on X_n over the n_observations - 1 overlapping pairs.
    lagged = levels[:-1]
    current = levels[1:]
    n_pairs = lagged.shape[0]

    lagged_mean = lagged.mean(axis=0)
    current_mean = current.mean(axis=0)
    centred_lagged = lagged - lagged_mean
    centred_current = current - current_mean

    variance_lagged = (centred_lagged**2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = (centred_lagged * centred_current).sum(axis=0) / variance_lagged
    b = np.where(variance_lagged > 0, b, np.nan)
    a = current_mean - b * lagged_mean

    zeta = current - (a + b * lagged)
    # Two parameters are estimated, hence n_pairs - 2 degrees of freedom.
    var_zeta = (zeta**2).sum(axis=0) / (n_pairs - 2)

    if bias_correction:
        # Kendall (1954): E[b_hat] - b ~= -(1 + 3b) / T for an AR(1) with an
        # estimated intercept. Inverting to first order and capping below 1 so
        # the correction cannot manufacture an explosive process.
        b = np.minimum(b + (1.0 + 3.0 * b) / n_pairs, 0.9999)
        a = current_mean - b * lagged_mean

    # Validity domain. b <= 0 is an oscillating AR(1), not a discretised O-U:
    # it would yield an enormous kappa and slip through the speed filter.
    is_valid = np.isfinite(b) & (b > 0.0) & (b < max_ar1_coefficient)

    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(is_valid, -np.log(np.where(is_valid, b, np.nan)) / dt, np.nan)
        m_raw = np.where(is_valid, a / (1.0 - b), np.nan)
        variance_equilibrium = np.where(is_valid, var_zeta / (1.0 - b**2), np.nan)

    sigma_eq = np.sqrt(variance_equilibrium)
    sigma = sigma_eq * np.sqrt(2.0 * kappa)
    half_life = np.log(2.0) / kappa

    parameters = pd.DataFrame(
        {
            "a": a,
            "b": b,
            "var_zeta": var_zeta,
            "kappa": kappa,
            "m_raw": m_raw,
            "sigma": sigma,
            "sigma_eq": sigma_eq,
            "half_life": half_life,
            "half_life_days": half_life * TRADING_DAYS_PER_YEAR,
        },
        index=cumulative_residuals.columns,
    )

    # Equation (18): in aggregate stocks are fairly priced, so a non-zero
    # average equilibrium level is model bias rather than signal.
    if center_means:
        cross_sectional_mean = parameters.loc[is_valid, "m_raw"].mean()
        parameters["m"] = parameters["m_raw"] - cross_sectional_mean
    else:
        parameters["m"] = parameters["m_raw"]

    parameters["is_tradeable"] = (
        is_valid
        & (parameters["kappa"].to_numpy() > min_mean_reversion_speed)
        & np.isfinite(parameters["sigma_eq"].to_numpy())
        & (parameters["sigma_eq"].to_numpy() > 0)
    )

    return parameters


def simulate_ou(
    kappa: float,
    m: float,
    sigma: float,
    n_steps: int,
    dt: float = 1.0 / TRADING_DAYS_PER_YEAR,
    x0: float | None = None,
    random_state: int | None = None,
) -> np.ndarray:
    """Exact simulation of an O-U path, used to validate the estimator.

    Uses the closed-form transition of equation (13) rather than an Euler
    scheme, so the simulated process has exactly the AR(1) law the estimator
    assumes and any bias measured against it is the estimator's own.

    Parameters:
        kappa (float): Mean-reversion speed, per year.
        m (float): Equilibrium level.
        sigma (float): Diffusion volatility.
        n_steps (int): Number of points to generate.
        dt (float): Time step in years.
        x0 (float | None): Starting value; defaults to a draw from the
            stationary distribution.
        random_state (int | None): Seed for reproducibility.

    Returns:
        np.ndarray: Simulated path of length ``n_steps``.
    """
    generator = np.random.default_rng(random_state)

    b = np.exp(-kappa * dt)
    stationary_std = sigma / np.sqrt(2.0 * kappa)
    innovation_std = stationary_std * np.sqrt(1.0 - b**2)

    path = np.empty(n_steps)
    path[0] = m + stationary_std * generator.standard_normal() if x0 is None else x0
    for step in range(1, n_steps):
        path[step] = (
            m + b * (path[step - 1] - m) + innovation_std * generator.standard_normal()
        )

    return path


def false_positive_rate(
    n_observations: int,
    min_mean_reversion_speed: float = TRADING_DAYS_PER_YEAR / 30,
    n_paths: int = 2000,
    bias_correction: bool = False,
    random_state: int | None = 0,
) -> float:
    """Share of pure random walks that wrongly pass the mean-reversion filter.

    A random walk has kappa = 0 and should never be traded. Because the OLS
    estimator of b is biased towards zero in short samples, a large fraction of
    them look like fast mean-reverters on a 60-day window. Measuring that rate
    is the honest way to decide how much the filter is worth.

    Parameters:
        n_observations (int): Length of the estimation window.
        min_mean_reversion_speed (float): The kappa cutoff under test.
        n_paths (int): Number of simulated random walks.
        bias_correction (bool): Whether to apply Kendall's correction.
        random_state (int | None): Seed.

    Returns:
        float: Fraction of paths passing the filter; the filter's false
            positive rate under the null of no mean reversion.
    """
    generator = np.random.default_rng(random_state)
    walks = pd.DataFrame(
        generator.standard_normal((n_observations, n_paths)).cumsum(axis=0) * 0.01
    )
    parameters = estimate_ou_parameters(
        walks,
        min_mean_reversion_speed=min_mean_reversion_speed,
        center_means=False,
        bias_correction=bias_correction,
    )
    return float(parameters["is_tradeable"].mean())
