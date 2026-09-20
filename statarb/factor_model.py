"""Statistical factor extraction on a rolling estimation window.

Step 2.a of the strategy: on each rebalance date, diagonalise the sample
correlation matrix of the trailing window and build the eigenportfolio return
series that will act as risk factors in the multi-factor regression.

Avellaneda & Lee standardise returns before forming the correlation matrix
(section 2.1). Note that standardising by the mean is immaterial for the
residuals: R_i,k = sigma_i Y_i,k + mean_i, hence

    F_j,k = sum_i (v_i / sigma_i) R_i,k = (Y v)_k + constant,

and the constant is absorbed by the intercept of the factor regression. The raw
returns are used here so that the factors are genuine tradeable portfolio
returns, as in equation (9).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from statarb.pca import (
    eigenportfolio_weights,
    n_factors_for_target_variance,
    principal_component_analysis,
)


@dataclass(frozen=True)
class FactorDecomposition:
    """Outcome of the PCA step on a single estimation window."""

    eigenvalues: np.ndarray
    """Full eigenvalue spectrum of the correlation matrix, decreasing order."""

    eigenvectors: pd.DataFrame
    """All eigenvectors, assets as index, ``PC{j}`` as columns."""

    eigenportfolios: pd.DataFrame
    """Retained eigenportfolio weights Q_i = v_i / sigma_i, assets as index."""

    factors: pd.DataFrame
    """Retained eigenportfolio return series, dates as index."""

    volatilities: pd.Series
    """Per-asset return standard deviation over the window."""

    n_factors: int
    """Number of retained components."""

    metadata: dict = field(default_factory=dict)
    """Diagnostics: explained variance per component and cumulative share."""

    @property
    def explained_variance(self) -> np.ndarray:
        """Variance share of each *retained* component."""
        return self.eigenvalues[: self.n_factors] / self.eigenvalues.sum()

    @property
    def total_explained_variance(self) -> float:
        """Variance share captured by the retained components."""
        return float(self.explained_variance.sum())


def extract_factors(
    returns: pd.DataFrame,
    n_factors: int | None = None,
    target_variance: float | None = None,
    min_variance: float = 1e-12,
) -> FactorDecomposition:
    """Extract eigenportfolio factors from a window of asset returns.

    Exactly one of ``n_factors`` (fixed count) or ``target_variance`` (variable
    count meeting an explained-variance target) must be supplied.

    Parameters:
        returns (pd.DataFrame): Return window, dates as index, assets as
            columns. Must be free of missing values.
        n_factors (int | None): Fixed number of components to retain.
        target_variance (float | None): Retain the smallest number of
            components whose cumulative variance share reaches this level.
        min_variance (float): Assets whose return variance falls below this
            threshold are dropped, since their correlation row is undefined and
            their eigenportfolio weight would diverge.

    Returns:
        FactorDecomposition: Eigen-spectrum, eigenportfolios and factor returns.
    """
    if (n_factors is None) == (target_variance is None):
        raise ValueError(
            "supply exactly one of n_factors or target_variance"
        )

    if not isinstance(returns, pd.DataFrame):
        raise TypeError(f"returns must be a DataFrame, got {type(returns)}")

    if returns.empty:
        raise ValueError("returns is empty")

    if returns.isna().any().any():
        raise ValueError(
            "returns contains missing values; build the estimation window "
            "first so that the coverage filter and imputation are explicit"
        )

    volatilities = returns.std(ddof=1)
    # A zero-variance column (a stock imputed to a constant, or halted for the
    # whole window) makes the correlation matrix singular and Q_i = v_i/sigma_i
    # infinite. Drop it here rather than letting NaNs propagate silently.
    tradeable = volatilities.index[volatilities**2 > min_variance]
    dropped = [asset for asset in returns.columns if asset not in tradeable]

    window = returns.loc[:, tradeable]
    volatilities = volatilities.loc[tradeable]

    n_periods, n_assets = window.shape
    if n_periods < n_assets:
        raise ValueError(
            f"insufficient data: {n_periods} observations for {n_assets} "
            f"assets; the sample correlation matrix would be singular"
        )

    correlation = window.corr()
    eigenvalues, eigenvectors = principal_component_analysis(correlation.values)

    if target_variance is not None:
        n_factors = n_factors_for_target_variance(eigenvalues, target_variance)

    if not 1 <= n_factors <= n_assets:
        raise ValueError(
            f"n_factors must be in [1, {n_assets}], got {n_factors}"
        )

    component_labels = [f"PC{j + 1}" for j in range(n_assets)]
    eigenvectors_df = pd.DataFrame(
        eigenvectors, index=tradeable, columns=component_labels
    )

    portfolio_weights = eigenportfolio_weights(
        eigenvectors[:, :n_factors], volatilities.values
    )
    eigenportfolios = pd.DataFrame(
        portfolio_weights, index=tradeable, columns=component_labels[:n_factors]
    )

    # F = R Q: the factor returns are the returns of the eigenportfolios.
    factors = window.to_numpy() @ portfolio_weights
    factors_df = pd.DataFrame(
        factors, index=window.index, columns=eigenportfolios.columns
    )

    total_variance = eigenvalues.sum()
    metadata = {
        "explained_variance_per_component": eigenvalues / total_variance,
        "cumulative_explained_variance": np.cumsum(eigenvalues) / total_variance,
        "dropped_assets": dropped,
        "selection_rule": (
            f"target_variance={target_variance:.0%}"
            if target_variance is not None
            else f"fixed n_factors={n_factors}"
        ),
    }

    return FactorDecomposition(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors_df,
        eigenportfolios=eigenportfolios,
        factors=factors_df,
        volatilities=volatilities,
        n_factors=n_factors,
        metadata=metadata,
    )


@dataclass(frozen=True)
class FactorRegression:
    """Output of regressing asset returns on the eigenportfolio factors."""

    alphas: pd.Series
    """Daily intercepts of the regression, one per asset."""

    betas: pd.DataFrame
    """Factor loadings, assets as index, components as columns."""

    residuals: pd.DataFrame
    """Idiosyncratic returns, dates as index, assets as columns."""

    r_squared: pd.Series
    """Share of each asset's variance explained by the factors."""

    @property
    def cumulative_residuals(self) -> pd.DataFrame:
        """The auxiliary process X_k = sum_{j<=k} epsilon_j of the appendix.

        Because the regression forces the residuals to average to zero over the
        estimation window, the last value X_T is identically zero. That is an
        artefact of estimating betas and residuals on the same sample, and it is
        what turns the s-score into s = -m / sigma_eq (equation 22).
        """
        return self.residuals.cumsum()


def fit_factor_regression(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
) -> FactorRegression:
    """Regress every asset on the factors in a single least-squares solve.

    This is equation (5) of the paper, R_i = alpha_i + sum_j beta_ij F_j + eps_i.

    All assets share the same design matrix [1, F], so instead of looping over
    columns the whole system is solved at once: ``lstsq`` on an (T x (1+m))
    design against a (T x N) target returns the (1+m) x N coefficient matrix.
    On this universe that turns roughly 120 000 small regressions into 2 400
    matrix solves over a full backtest.

    Parameters:
        returns (pd.DataFrame): Returns over the regression window (T x N).
        factors (pd.DataFrame): Factor returns over the same dates (T x m).

    Returns:
        FactorRegression: Intercepts, loadings, residuals and R-squared.
    """
    if not returns.index.equals(factors.index):
        raise ValueError("returns and factors must share the same dates")

    n_periods = len(returns)
    if n_periods <= factors.shape[1] + 1:
        raise ValueError(
            f"{n_periods} observations is not enough to fit an intercept and "
            f"{factors.shape[1]} factors"
        )

    target = returns.to_numpy(dtype=float)
    design = np.column_stack(
        [np.ones(n_periods), factors.to_numpy(dtype=float)]
    )

    # rcond=None uses the machine-precision cutoff for rank determination.
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    fitted = design @ coefficients
    residuals = target - fitted

    total_sum_of_squares = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    residual_sum_of_squares = (residuals**2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_squared = np.where(
            total_sum_of_squares > 0,
            1.0 - residual_sum_of_squares / total_sum_of_squares,
            np.nan,
        )

    return FactorRegression(
        alphas=pd.Series(coefficients[0], index=returns.columns),
        betas=pd.DataFrame(
            coefficients[1:].T, index=returns.columns, columns=factors.columns
        ),
        residuals=pd.DataFrame(
            residuals, index=returns.index, columns=returns.columns
        ),
        r_squared=pd.Series(r_squared, index=returns.columns),
    )
