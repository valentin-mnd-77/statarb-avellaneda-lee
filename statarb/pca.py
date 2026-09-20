"""Principal component analysis of the return correlation matrix.

Implements the factor-extraction step of Avellaneda & Lee (2008), section 2.1.

The correlation matrix is diagonalised and the retained eigenvectors are turned
into *eigenportfolios* whose dollar holdings are, following equation (9),

    Q_i^(j) = v_i^(j) / sigma_i

so that the factor returns are the returns of those portfolios,

    F_j,k = sum_i Q_i^(j) R_i,k .

Two facts that the implementation relies on:

1.  The eigenportfolio returns are exactly uncorrelated. With D = diag(sigma)
    and Sigma = D rho D the sample covariance matrix,
        Cov(F_j, F_j') = (v_j / sigma)' Sigma (v_j' / sigma) = v_j' rho v_j'
                       = lambda_j delta_{j,j'} .
    So Var(F_j) = lambda_j and the design matrix of the factor regression is
    orthogonal by construction.

2.  Rescaling an eigenportfolio, Q -> c Q, rescales the factor by c and the
    regression beta by 1/c, leaving the residuals unchanged. The normalisation
    chosen below is therefore purely cosmetic and does not affect the strategy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Eigenvalues of a correlation matrix are non-negative in theory; sample
# estimates can return tiny negative values through round-off.
_EIGENVALUE_FLOOR = 0.0


def principal_component_analysis(
    matrix: np.ndarray,
    n_components: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalise a symmetric matrix, eigenvalues sorted in decreasing order.

    Parameters:
        matrix (np.ndarray): Symmetric (N x N) matrix, typically a correlation
            or covariance matrix.
        n_components (int | None): If given, return only the leading
            ``n_components`` eigenvalues/eigenvectors. The full spectrum is
            always needed to compute explained-variance ratios, so callers that
            select factors by target variance should keep this at None.

    Returns:
        tuple[np.ndarray, np.ndarray]: Eigenvalues (N,) in decreasing order and
            the matching eigenvectors as columns of an (N x N) array.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square, got shape {matrix.shape}")

    if not np.isfinite(matrix).all():
        raise ValueError("matrix contains NaN or Inf values")

    if not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ValueError("matrix must be symmetric")

    # eigh exploits symmetry: real eigenvalues, orthonormal eigenvectors.
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # eigh returns ascending order; the strategy wants the dominant modes first.
    descending = eigenvalues.argsort()[::-1]
    eigenvalues = np.maximum(eigenvalues[descending], _EIGENVALUE_FLOOR)
    eigenvectors = eigenvectors[:, descending]

    if n_components is not None:
        eigenvalues = eigenvalues[:n_components]
        eigenvectors = eigenvectors[:, :n_components]

    return eigenvalues, eigenvectors


def n_factors_for_target_variance(
    eigenvalues: np.ndarray,
    target_variance: float,
) -> int:
    """Smallest number of leading eigenvalues explaining ``target_variance``.

    Parameters:
        eigenvalues (np.ndarray): Full eigenvalue spectrum, decreasing order.
        target_variance (float): Target fraction of total variance, in (0, 1].

    Returns:
        int: Number of factors to retain, at least one.
    """
    if not 0 < target_variance <= 1:
        raise ValueError(
            f"target_variance must be in (0, 1], got {target_variance}"
        )

    total = eigenvalues.sum()
    if total <= 0:
        raise ValueError("eigenvalues must have a strictly positive sum")

    explained = np.cumsum(eigenvalues) / total
    # searchsorted gives the first index whose cumulative share reaches the
    # target; +1 converts a 0-based index into a count.
    return int(np.searchsorted(explained, target_variance) + 1)


def eigenportfolio_weights(
    eigenvectors: np.ndarray,
    volatilities: np.ndarray,
    normalise: bool = True,
) -> np.ndarray:
    """Convert eigenvectors into eigenportfolio dollar weights, Q_i = v_i / sigma_i.

    Parameters:
        eigenvectors (np.ndarray): Eigenvectors as columns, (N x m).
        volatilities (np.ndarray): Per-asset return standard deviations, (N,).
        normalise (bool): Scale each portfolio to unit gross exposure. Purely
            cosmetic: it leaves the factor-model residuals unchanged.

    Returns:
        np.ndarray: Eigenportfolio weights, (N x m).
    """
    volatilities = np.asarray(volatilities, dtype=float)

    if (volatilities <= 0).any() or not np.isfinite(volatilities).all():
        raise ValueError(
            "volatilities must be strictly positive and finite; drop constant "
            "or all-missing assets from the estimation window first"
        )

    weights = eigenvectors / volatilities[:, np.newaxis]

    if normalise:
        gross = np.abs(weights).sum(axis=0)
        weights = weights / gross[np.newaxis, :]

    return weights


def align_eigenvector_signs(
    eigenvectors: pd.DataFrame,
    previous: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fix the arbitrary sign of each eigenvector for a readable time series.

    An eigenvector is only defined up to sign, so consecutive estimation dates
    can flip it and make loading time series unreadable. The first component is
    anchored to a positive sum (the "market" mode, positive by Krein's theorem
    when correlations are non-negative); the others are aligned to the previous
    date by maximising the dot product on the shared universe.

    This is presentation only: flipping a column of the design matrix flips the
    matching beta and leaves the residuals untouched.

    Parameters:
        eigenvectors (pd.DataFrame): Current eigenvectors, assets as index,
            components as columns.
        previous (pd.DataFrame | None): Eigenvectors from the previous date.

    Returns:
        pd.DataFrame: Sign-aligned copy of ``eigenvectors``.
    """
    aligned = eigenvectors.copy()

    if aligned.iloc[:, 0].sum() < 0:
        aligned.isetitem(0, -aligned.iloc[:, 0])

    if previous is None:
        return aligned

    shared_assets = aligned.index.intersection(previous.index)
    n_common = min(aligned.shape[1], previous.shape[1])

    # Start at 1: the first component is already anchored by its sign above.
    for component in range(1, n_common):
        overlap = float(
            aligned.iloc[:, component].reindex(shared_assets)
            @ previous.iloc[:, component].reindex(shared_assets)
        )
        if overlap < 0:
            aligned.isetitem(component, -aligned.iloc[:, component])

    return aligned
