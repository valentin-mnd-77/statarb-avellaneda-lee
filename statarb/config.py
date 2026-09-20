"""Single source of truth for every parameter of the strategy.

Keeping the parameters in one frozen dataclass rather than scattered across a
notebook means a backtest is fully described by one object: it can be printed,
hashed, serialised next to its results, and varied in a sensitivity study
without touching the code that consumes it.

Defaults follow Avellaneda & Lee (2008). Where the Politecnico assignment
deviates from the paper, the paper wins and the difference is noted in the
field documentation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class StrategyConfig:
    """Parameters of the PCA statistical arbitrage strategy."""

    # ------------------------------------------------------------------
    # Estimation windows
    # ------------------------------------------------------------------
    estimation_window: int = 252
    """Trading days used for the PCA on the correlation matrix (paper: 1 year)."""

    ou_window: int = 60
    """Trading days used for the factor regression and the AR(1) fit.

    The paper picks 60 days because it spans roughly one earnings cycle
    (section 3). The factor *loadings* are estimated on this shorter window,
    while the factors themselves come from the 252-day PCA.
    """

    min_coverage: float = 0.95
    """Minimum fraction of non-missing returns for an asset to stay in a window."""

    # ------------------------------------------------------------------
    # Factor selection
    # ------------------------------------------------------------------
    n_factors: int | None = 4
    """Fixed number of principal components. Mutually exclusive with target_variance.

    The paper uses 15 on a ~1400-stock US universe, which corresponds to
    roughly 50% of explained variance. On the 50-name Euro Stoxx 50 universe,
    4 components explain ~60% on average, which is the range the paper's
    conclusion identifies as optimal. See README for the measurement.
    """

    target_variance: float | None = None
    """Variable factor count: retain the smallest m reaching this variance share."""

    # ------------------------------------------------------------------
    # Ornstein-Uhlenbeck estimation and filtering
    # ------------------------------------------------------------------
    min_mean_reversion_speed: float = TRADING_DAYS_PER_YEAR / 30
    """Reject assets reverting more slowly than kappa = 8.4 (section 4).

    kappa > 252/30 means a mean-reversion time under half the 60-day estimation
    window, which is what makes the constant-parameter assumption defensible.
    """

    max_ar1_coefficient: float = 0.9672
    """Upper bound on the AR(1) coefficient b, from the appendix.

    exp(-8.4/252) = 0.9672, so this is the kappa filter restated on b. The lower
    bound b > 0 is enforced unconditionally: a negative b describes an
    oscillating series, not a discretised O-U process, and would produce a
    spuriously huge kappa that sails through the speed filter.
    """

    bias_correction: bool = False
    """Apply Kendall's small-sample correction to the AR(1) coefficient.

    Off by default because the paper does not do it, but measurably important:
    on a 60-day window roughly three quarters of pure random walks pass the
    kappa > 8.4 filter without it. See ``statarb.ou.false_positive_rate``.
    """

    center_ou_means: bool = True
    """Subtract the cross-sectional average of m (equation 18).

    The paper adopts this for every strategy it reports: in aggregate stocks are
    correctly priced, so a non-zero average equilibrium level is model bias.
    The assignment asks to skip it; we follow the paper.
    """

    # ------------------------------------------------------------------
    # Trading signal
    # ------------------------------------------------------------------
    use_modified_s_score: bool = False
    """Use s_mod = s - alpha/(kappa sigma_eq) instead of the pure s-score.

    The paper defines the modified score (equation 17) but back-tests the pure
    one, judging the drift correction minor at these horizons (section 4.2).
    """

    s_open_long: float = 1.25
    """Open a long when s < -s_open_long (paper: s_bo = 1.25)."""

    s_open_short: float = 1.25
    """Open a short when s > +s_open_short (paper: s_so = 1.25)."""

    s_close_long: float = 0.50
    """Close a long when s > -s_close_long (paper: 0.50).

    Careful with naming: the paper calls the short-closing cutoff s_bc = 0.75
    and the long-closing cutoff s_sc = 0.50. The assignment's skeleton code uses
    the opposite convention. These fields are named after what they *do*.
    """

    s_close_short: float = 0.75
    """Close a short when s < +s_close_short (paper: 0.75).

    Asymmetric on purpose: the paper found closing shorts earlier works better.
    """

    # ------------------------------------------------------------------
    # Portfolio construction
    # ------------------------------------------------------------------
    hedge_factor_exposure: bool = True
    """Short beta_ij units of eigenportfolio j against each dollar of stock i.

    This is what makes the book market-neutral in the sense of equations (6)-(7).
    The assignment drops it and takes raw equal weights, which leaves a
    substantial residual factor exposure.
    """

    weighting: Literal["fixed_notional", "equal_gross"] = "fixed_notional"
    """How signals become dollar positions.

    - ``fixed_notional``: every open position gets the same notional Lambda,
      so gross exposure floats with the number of active signals. This is the
      paper's scheme (section 5, footnote 8) and it avoids charging transaction
      costs for rescaling positions that did not actually change.
    - ``equal_gross``: rescale daily to 100% gross, as the assignment asks.
      Simpler to read, but every change in the number of signals reprices the
      whole book and inflates measured turnover.
    """

    notional_per_position: float = 0.05
    """Lambda: fraction of equity per open position under ``fixed_notional``.

    A pure leverage constant. 5% per name means 100% gross at 20 simultaneous
    positions, close to the median observed on this universe.
    """

    # ------------------------------------------------------------------
    # Execution and costs
    # ------------------------------------------------------------------
    execution_lag: int = 1
    """Days between the signal date and the first day the position earns a return.

    1 means trading at the close of the signal day, as in the paper. The
    assignment specifies execution at the close of the *following* day, which is
    lag 2 and is available as a sensitivity.
    """

    transaction_costs: float = 5e-4
    """All-in one-way cost applied to turnover: 5 bps per trade, 10 bps round trip."""

    no_trade_band: float = 0.0
    """Minimum weight change worth executing, as a fraction of equity.

    The factor hedge is re-estimated daily, so every weight drifts a little even
    on days when no signal changed. Without a band the strategy pays the spread
    on that noise: on this universe the unfiltered book turns over 77 times a
    year, of which a large part is hedge maintenance rather than position
    changes. A band is the cheapest form of cost-aware rebalancing; closing
    trades are always executed regardless of it.
    """

    # ------------------------------------------------------------------
    # Trading-time extension (section 6)
    # ------------------------------------------------------------------
    trading_time: bool = False
    """Estimate signals on volume-adjusted returns rather than calendar returns."""

    volume_window: int = 60
    """Trailing window for the average daily volume used in the adjustment."""

    max_volume_adjustment: float = 10.0
    """Cap on the ratio <dV>/V_t, which explodes on near-zero volume days."""

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if (self.n_factors is None) == (self.target_variance is None):
            raise ValueError(
                "set exactly one of n_factors or target_variance"
            )
        if self.ou_window > self.estimation_window:
            raise ValueError("ou_window cannot exceed estimation_window")
        if self.execution_lag < 1:
            raise ValueError("execution_lag must be at least 1 to avoid look-ahead")
        if not 0 < self.min_coverage <= 1:
            raise ValueError("min_coverage must lie in (0, 1]")

    @property
    def dt(self) -> float:
        """Time step in years, used to annualise the O-U parameters."""
        return 1.0 / TRADING_DAYS_PER_YEAR

    def variant(self, **overrides) -> "StrategyConfig":
        """Return a copy with some fields replaced, for sensitivity runs."""
        return replace(self, **overrides)

    def to_dict(self) -> dict:
        """Flat dictionary of the configuration, for logging next to results."""
        return asdict(self)
