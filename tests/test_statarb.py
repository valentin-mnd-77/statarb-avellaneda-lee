"""Unit tests.

Each test checks a property whose correct answer is known independently of the
implementation: an identity that must hold algebraically, a parameter recovered
from a simulated process, or a hand-computable example. Tests that merely assert
the code reproduces its own output would catch nothing.

Run with:  pytest -q
"""

from __future__ import annotations

import sys
import types
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from statarb.backtest import align_weights, run_backtest
from statarb.config import StrategyConfig
from statarb.data import compute_returns, prepare_estimation_window
from statarb.factor_model import extract_factors, fit_factor_regression
from statarb.metrics import performance_statistics
from statarb.ou import (
    estimate_ou_parameters,
    false_positive_rate,
    simulate_ou,
)
from statarb.pca import (
    align_eigenvector_signs,
    eigenportfolio_weights,
    n_factors_for_target_variance,
    principal_component_analysis,
)
from statarb.portfolio import (
    apply_no_trade_band,
    hedge_factor_exposure,
    residual_factor_exposure,
    size_positions,
)
from statarb.public_data import EURO_STOXX_50_YAHOO, download_public_data
from statarb.signals import SignalThresholds, compute_s_scores, update_positions


@pytest.fixture
def correlated_returns() -> pd.DataFrame:
    """A 400 x 12 return panel with a strong common factor plus noise."""
    generator = np.random.default_rng(0)
    n_periods, n_assets = 400, 12

    market = generator.standard_normal(n_periods)
    loadings = generator.uniform(0.5, 1.5, n_assets)
    idiosyncratic = generator.standard_normal((n_periods, n_assets))
    volatilities = generator.uniform(0.8, 2.0, n_assets)

    panel = (
        np.outer(market, loadings) + idiosyncratic
    ) * volatilities * 0.01

    return pd.DataFrame(
        panel,
        index=pd.bdate_range("2020-01-01", periods=n_periods),
        columns=[f"S{i:02d}" for i in range(n_assets)],
    )


# ----------------------------------------------------------------------
# PCA
# ----------------------------------------------------------------------
class TestPCA:
    def test_eigenvalues_are_sorted_and_sum_to_n(self, correlated_returns):
        """The trace of a correlation matrix equals the number of assets."""
        correlation = correlated_returns.corr().to_numpy()
        eigenvalues, _ = principal_component_analysis(correlation)

        assert np.all(np.diff(eigenvalues) <= 1e-12)
        assert eigenvalues.sum() == pytest.approx(len(correlation))

    def test_eigenvectors_are_orthonormal(self, correlated_returns):
        correlation = correlated_returns.corr().to_numpy()
        _, eigenvectors = principal_component_analysis(correlation)

        identity = eigenvectors.T @ eigenvectors
        assert np.allclose(identity, np.eye(len(correlation)), atol=1e-10)

    def test_reconstruction(self, correlated_returns):
        """V diag(lambda) V' must rebuild the original matrix."""
        correlation = correlated_returns.corr().to_numpy()
        eigenvalues, eigenvectors = principal_component_analysis(correlation)

        rebuilt = (eigenvectors * eigenvalues) @ eigenvectors.T
        assert np.allclose(rebuilt, correlation, atol=1e-10)

    def test_rejects_asymmetric_input(self):
        with pytest.raises(ValueError, match="symmetric"):
            principal_component_analysis(np.array([[1.0, 0.5], [0.2, 1.0]]))

    def test_target_variance_selection(self):
        eigenvalues = np.array([5.0, 3.0, 1.5, 0.5])  # shares .5, .3, .15, .05
        assert n_factors_for_target_variance(eigenvalues, 0.50) == 1
        assert n_factors_for_target_variance(eigenvalues, 0.55) == 2
        assert n_factors_for_target_variance(eigenvalues, 0.95) == 3
        assert n_factors_for_target_variance(eigenvalues, 1.00) == 4

    def test_eigenportfolio_is_volatility_weighted(self):
        eigenvectors = np.array([[0.6], [0.8]])
        volatilities = np.array([0.01, 0.02])

        weights = eigenportfolio_weights(
            eigenvectors, volatilities, normalise=False
        )
        assert weights[0, 0] == pytest.approx(60.0)
        assert weights[1, 0] == pytest.approx(40.0)

    def test_eigenportfolio_rejects_zero_volatility(self):
        with pytest.raises(ValueError, match="strictly positive"):
            eigenportfolio_weights(np.array([[1.0]]), np.array([0.0]))

    def test_sign_alignment_follows_previous_date(self):
        current = pd.DataFrame(
            [[-0.5, 0.4], [-0.5, -0.6], [-0.5, 0.3]],
            index=["A", "B", "C"],
            columns=["PC1", "PC2"],
        )
        previous = current.copy()
        previous["PC2"] *= -1

        aligned = align_eigenvector_signs(current, previous)
        assert aligned["PC1"].sum() > 0  # market mode anchored positive
        assert np.allclose(aligned["PC2"], -current["PC2"])


# ----------------------------------------------------------------------
# Factor model
# ----------------------------------------------------------------------
class TestFactorModel:
    def test_factors_are_mutually_uncorrelated(self, correlated_returns):
        """Eigenportfolio returns are orthogonal by construction."""
        decomposition = extract_factors(correlated_returns, n_factors=4)
        correlation = decomposition.factors.corr().to_numpy()

        assert np.allclose(correlation, np.eye(4), atol=1e-10)

    def test_first_eigenportfolio_is_long_only(self, correlated_returns):
        """With uniformly positive correlations, PC1 is the market mode."""
        decomposition = extract_factors(correlated_returns, n_factors=1)
        assert (decomposition.eigenportfolios["PC1"] > 0).all()

    def test_residuals_are_orthogonal_to_factors(self, correlated_returns):
        """OLS residuals must be uncorrelated with every regressor."""
        decomposition = extract_factors(correlated_returns, n_factors=3)
        regression = fit_factor_regression(
            correlated_returns, decomposition.factors
        )

        cross = decomposition.factors.T.to_numpy() @ regression.residuals.to_numpy()
        assert np.abs(cross).max() < 1e-8
        assert np.abs(regression.residuals.mean()).max() < 1e-12

    def test_cumulative_residuals_end_at_zero(self, correlated_returns):
        """X_T = 0 is the artefact the appendix relies on for equation (22)."""
        decomposition = extract_factors(correlated_returns, n_factors=3)
        regression = fit_factor_regression(
            correlated_returns, decomposition.factors
        )
        assert np.abs(regression.cumulative_residuals.iloc[-1]).max() < 1e-10

    def test_recovers_known_loadings(self):
        """With a single factor and no noise, the beta must be exact."""
        dates = pd.bdate_range("2021-01-01", periods=200)
        factor = pd.DataFrame(
            {"PC1": np.random.default_rng(1).standard_normal(200) * 0.01},
            index=dates,
        )
        returns = pd.DataFrame(
            {"A": 0.0002 + 1.7 * factor["PC1"], "B": -0.0001 + 0.3 * factor["PC1"]},
            index=dates,
        )

        regression = fit_factor_regression(returns, factor)
        assert regression.betas.loc["A", "PC1"] == pytest.approx(1.7, abs=1e-10)
        assert regression.betas.loc["B", "PC1"] == pytest.approx(0.3, abs=1e-10)
        assert regression.alphas["A"] == pytest.approx(0.0002, abs=1e-10)
        assert regression.r_squared["A"] == pytest.approx(1.0, abs=1e-10)

    def test_residuals_invariant_to_eigenportfolio_scaling(self, correlated_returns):
        """Rescaling Q rescales beta inversely and leaves residuals untouched."""
        decomposition = extract_factors(correlated_returns, n_factors=3)
        scaled = decomposition.factors * 7.0

        original = fit_factor_regression(correlated_returns, decomposition.factors)
        rescaled = fit_factor_regression(correlated_returns, scaled)

        assert np.allclose(
            original.residuals.to_numpy(), rescaled.residuals.to_numpy(), atol=1e-12
        )

    def test_rejects_missing_values(self, correlated_returns):
        broken = correlated_returns.copy()
        broken.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="missing values"):
            extract_factors(broken, n_factors=2)

    def test_requires_exactly_one_selection_rule(self, correlated_returns):
        with pytest.raises(ValueError, match="exactly one"):
            extract_factors(correlated_returns, n_factors=2, target_variance=0.5)


# ----------------------------------------------------------------------
# Ornstein-Uhlenbeck
# ----------------------------------------------------------------------
class TestOrnsteinUhlenbeck:
    def test_recovers_parameters_from_a_simulated_process(self):
        """On a long exact simulation the estimator must land on the truth."""
        true_kappa, true_m, true_sigma = 12.0, 0.03, 0.25
        path = simulate_ou(
            true_kappa, true_m, true_sigma, n_steps=200_000, random_state=7
        )
        frame = pd.DataFrame({"X": path})

        estimated = estimate_ou_parameters(frame, center_means=False).loc["X"]

        assert estimated["kappa"] == pytest.approx(true_kappa, rel=0.05)
        assert estimated["m"] == pytest.approx(true_m, abs=0.005)
        assert estimated["sigma"] == pytest.approx(true_sigma, rel=0.05)
        assert estimated["sigma_eq"] == pytest.approx(
            true_sigma / np.sqrt(2 * true_kappa), rel=0.05
        )

    def test_half_life_identity(self):
        path = simulate_ou(20.0, 0.0, 0.3, n_steps=50_000, random_state=3)
        estimated = estimate_ou_parameters(
            pd.DataFrame({"X": path}), center_means=False
        ).loc["X"]

        assert estimated["half_life"] == pytest.approx(
            np.log(2) / estimated["kappa"], rel=1e-12
        )
        assert estimated["sigma_eq"] == pytest.approx(
            estimated["sigma"] / np.sqrt(2 * estimated["kappa"]), rel=1e-10
        )

    def test_rejects_negative_ar1_coefficient(self):
        """An oscillating series is not a discretised O-U process."""
        alternating = pd.DataFrame({"X": [(-1.0) ** i for i in range(60)]})
        estimated = estimate_ou_parameters(alternating, center_means=False).loc["X"]

        assert estimated["b"] < 0
        assert not estimated["is_tradeable"]
        assert np.isnan(estimated["kappa"])

    def test_rejects_slow_mean_reversion(self):
        """A near-random walk has b close to 1 and must be filtered out."""
        slow = simulate_ou(0.5, 0.0, 0.2, n_steps=2000, random_state=11)
        estimated = estimate_ou_parameters(
            pd.DataFrame({"X": slow}), center_means=False
        ).loc["X"]

        assert not estimated["is_tradeable"]

    def test_short_window_filter_is_badly_oversized(self):
        """Documents a real weakness rather than asserting the code is fine.

        OLS on a near-unit-root AR(1) is biased towards zero by roughly
        (1 + 3b) / T, so on the paper's 60-day window most random walks look
        like week-long mean-reverters. The kappa > 8.4 filter therefore rejects
        far less than it appears to.
        """
        uncorrected = false_positive_rate(60, n_paths=1000)
        corrected = false_positive_rate(60, n_paths=1000, bias_correction=True)

        assert uncorrected > 0.60
        assert corrected < uncorrected / 2
        # Lengthening the window is the other lever, and it works.
        assert false_positive_rate(504, n_paths=1000) < 0.10

    def test_centring_removes_the_cross_sectional_mean(self):
        paths = {
            f"S{i}": simulate_ou(15.0, 0.02, 0.2, 500, random_state=i)
            for i in range(20)
        }
        frame = pd.DataFrame(paths)

        centred = estimate_ou_parameters(frame, center_means=True)
        tradeable = centred["is_tradeable"]

        assert centred.loc[tradeable, "m"].mean() == pytest.approx(0.0, abs=1e-12)
        assert centred.loc[tradeable, "m_raw"].mean() != pytest.approx(0.0, abs=1e-6)

    def test_matches_a_loop_of_scalar_regressions(self):
        """The vectorised estimator must equal an explicit per-column fit."""
        generator = np.random.default_rng(2)
        frame = pd.DataFrame(
            generator.standard_normal((60, 8)).cumsum(axis=0) * 0.01,
            columns=[f"S{i}" for i in range(8)],
        )
        vectorised = estimate_ou_parameters(frame, center_means=False)

        for column in frame.columns:
            values = frame[column].to_numpy()
            design = np.column_stack([np.ones(59), values[:-1]])
            intercept, slope = np.linalg.lstsq(design, values[1:], rcond=None)[0]

            assert vectorised.loc[column, "a"] == pytest.approx(intercept, rel=1e-10)
            assert vectorised.loc[column, "b"] == pytest.approx(slope, rel=1e-10)


# ----------------------------------------------------------------------
# Signals
# ----------------------------------------------------------------------
class TestSignals:
    @staticmethod
    def _parameters(**overrides):
        base = {
            "m": 0.0,
            "sigma_eq": 1.0,
            "kappa": 20.0,
            "is_tradeable": True,
            "a": 0.0,
            "b": 0.9,
            "var_zeta": 1.0,
            "m_raw": 0.0,
            "sigma": 1.0,
            "half_life": 0.03,
            "half_life_days": 8.0,
        }
        base.update(overrides)
        return pd.DataFrame({"A": base}).T

    def test_s_score_is_the_standardised_dislocation(self):
        residuals = pd.DataFrame({"A": [0.0, 0.0, 0.6]})
        parameters = self._parameters(m=0.1, sigma_eq=0.25)

        score = compute_s_scores(residuals, parameters)
        assert score["A"] == pytest.approx((0.6 - 0.1) / 0.25)

    def test_modified_s_score_subtracts_the_annualised_drift(self):
        residuals = pd.DataFrame({"A": [0.0, 0.0, 0.5]})
        parameters = self._parameters(m=0.0, sigma_eq=0.25, kappa=20.0)
        alphas = pd.Series({"A": 0.0004})  # daily intercept

        pure = compute_s_scores(residuals, parameters)
        modified = compute_s_scores(residuals, parameters, alphas, modified=True)

        expected_shift = (0.0004 * 252) / (20.0 * 0.25)
        assert (pure["A"] - modified["A"]) == pytest.approx(expected_shift)

    def test_modified_s_score_requires_alphas(self):
        residuals = pd.DataFrame({"A": [0.0, 0.1]})
        with pytest.raises(ValueError, match="alphas"):
            compute_s_scores(residuals, self._parameters(), modified=True)

    def test_untradeable_assets_get_no_score(self):
        residuals = pd.DataFrame({"A": [0.0, 0.5]})
        score = compute_s_scores(residuals, self._parameters(is_tradeable=False))
        assert np.isnan(score["A"])

    def test_entry_and_exit_state_machine(self):
        thresholds = SignalThresholds(1.25, 1.25, 0.50, 0.75)

        # Flat: only a score beyond the entry cutoff opens a position.
        assert update_positions({}, pd.Series({"A": -1.30}), thresholds) == {"A": 1.0}
        assert update_positions({}, pd.Series({"A": 1.30}), thresholds) == {"A": -1.0}
        assert update_positions({}, pd.Series({"A": -1.20}), thresholds) == {}

        # Long: held while s <= -0.50, closed above it.
        held = update_positions({"A": 1.0}, pd.Series({"A": -0.80}), thresholds)
        assert held == {"A": 1.0}
        assert update_positions({"A": 1.0}, pd.Series({"A": -0.40}), thresholds) == {}

        # Short: exits earlier, at 0.75, which is the paper's asymmetry.
        still_short = update_positions({"A": -1.0}, pd.Series({"A": 0.80}), thresholds)
        assert still_short == {"A": -1.0}
        assert update_positions({"A": -1.0}, pd.Series({"A": 0.60}), thresholds) == {}

    def test_position_survives_a_score_that_stays_extreme(self):
        """Positions are path dependent: no re-entry churn while held."""
        thresholds = SignalThresholds(1.25, 1.25, 0.50, 0.75)
        positions = {}
        for score in [-2.0, -1.8, -1.9, -1.0]:
            positions = update_positions(positions, pd.Series({"A": score}), thresholds)
        assert positions == {"A": 1.0}

    def test_missing_score_closes_the_position(self):
        thresholds = SignalThresholds(1.25, 1.25, 0.50, 0.75)
        assert update_positions({"A": 1.0}, pd.Series({"A": np.nan}), thresholds) == {}


# ----------------------------------------------------------------------
# Portfolio construction
# ----------------------------------------------------------------------
class TestPortfolio:
    def test_fixed_notional_sizing(self):
        weights = size_positions(
            {"A": 1.0, "B": -1.0}, "fixed_notional", notional_per_position=0.05
        )
        assert weights["A"] == pytest.approx(0.05)
        assert weights["B"] == pytest.approx(-0.05)

    def test_equal_gross_sizing_sums_to_one(self):
        weights = size_positions({"A": 1.0, "B": -1.0, "C": 1.0}, "equal_gross")
        assert weights.abs().sum() == pytest.approx(1.0)

    def test_hedge_removes_factor_exposure(self, correlated_returns):
        """B'P = I holds whenever factors and loadings share a sample, so the
        paper's direct hedge is an exact projection."""
        decomposition = extract_factors(correlated_returns, n_factors=3)
        regression = fit_factor_regression(
            correlated_returns, decomposition.factors
        )

        raw = size_positions(
            {"S00": 1.0, "S03": -1.0, "S07": 1.0}, "fixed_notional", 0.05
        )
        before = residual_factor_exposure(raw, regression.betas)
        hedged = hedge_factor_exposure(
            raw, regression.betas, decomposition.eigenportfolios
        )
        after = residual_factor_exposure(hedged, regression.betas)

        assert np.abs(before).max() > 1e-3
        assert np.abs(after).max() < 1e-10

    def test_no_trade_band_blocks_small_moves_but_not_exits(self):
        targets = pd.DataFrame(
            {"A": [0.050, 0.051, 0.070, 0.000]},
            index=pd.bdate_range("2022-01-03", periods=4),
        )
        held = apply_no_trade_band(targets, band=0.01)

        assert held["A"].tolist() == [0.050, 0.050, 0.070, 0.000]

    def test_zero_band_is_a_passthrough(self):
        targets = pd.DataFrame({"A": [0.01, 0.02, 0.03]})
        assert apply_no_trade_band(targets, 0.0).equals(targets)


# ----------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------
class TestBacktest:
    @pytest.fixture
    def simple_case(self):
        dates = pd.bdate_range("2022-01-03", periods=5)
        weights = pd.DataFrame({"A": [1.0, 1.0, 0.0, 0.0, 0.0]}, index=dates)
        returns = pd.DataFrame({"A": [0.01, 0.02, 0.03, 0.04, 0.05]}, index=dates)
        return weights, returns

    def test_execution_lag_shifts_the_earned_return(self, simple_case):
        weights, returns = simple_case

        lag_one, _ = align_weights(weights, returns, execution_lag=1)
        lag_two, _ = align_weights(weights, returns, execution_lag=2)

        # The signal of day 0 earns day 1's return at lag 1, day 2's at lag 2.
        assert lag_one["A"].reindex(returns.index).fillna(0.0).tolist() == [
            0.0, 1.0, 1.0, 0.0, 0.0
        ]
        assert lag_two["A"].reindex(returns.index).fillna(0.0).tolist() == [
            0.0, 0.0, 1.0, 1.0, 0.0
        ]

    def test_gross_pnl_is_the_weighted_return(self, simple_case):
        weights, returns = simple_case
        result = run_backtest(weights, returns, transaction_costs=0.0)

        assert result.gross_returns.loc[returns.index[1]] == pytest.approx(0.02)
        assert result.gross_returns.loc[returns.index[3]] == pytest.approx(0.0)

    def test_turnover_counts_the_initial_build_and_the_exit(self, simple_case):
        weights, returns = simple_case
        result = run_backtest(weights, returns, transaction_costs=0.0)

        # One unit in, one unit out: two units of one-way turnover in total.
        assert result.turnover.sum() == pytest.approx(2.0)

    def test_costs_reduce_returns_by_rate_times_turnover(self, simple_case):
        weights, returns = simple_case
        rate = 5e-4
        free = run_backtest(weights, returns, transaction_costs=0.0)
        charged = run_backtest(weights, returns, transaction_costs=rate)

        difference = (free.net_returns - charged.net_returns).sum()
        assert difference == pytest.approx(rate * free.turnover.sum())

    def test_rejects_exposure_without_a_return(self):
        dates = pd.bdate_range("2022-01-03", periods=3)
        weights = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=dates)
        returns = pd.DataFrame({"A": [0.01, np.nan, 0.02]}, index=dates)

        with pytest.raises(ValueError, match="no return"):
            run_backtest(weights, returns)

    def test_rejects_zero_execution_lag(self):
        with pytest.raises(ValueError, match="look-ahead"):
            StrategyConfig(execution_lag=0)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
class TestMetrics:
    def test_statistics_on_a_constant_return_series(self):
        returns = pd.Series(
            [0.001] * 252, index=pd.bdate_range("2022-01-03", periods=252)
        )
        statistics = performance_statistics(returns)

        assert statistics["annualised_volatility"] == pytest.approx(0.0)
        assert statistics["max_drawdown"] == pytest.approx(0.0)
        assert statistics["hit_rate"] == pytest.approx(1.0)
        assert statistics["total_return"] == pytest.approx(1.001**252 - 1)

    def test_drawdown_is_measured_from_inception(self):
        returns = pd.Series(
            [-0.10, 0.05], index=pd.bdate_range("2022-01-03", periods=2)
        )
        statistics = performance_statistics(returns)

        # A first-day loss is a drawdown even though it precedes any peak.
        assert statistics["max_drawdown"] == pytest.approx(-0.10)

    def test_sharpe_uses_arithmetic_annualisation(self):
        generator = np.random.default_rng(4)
        returns = pd.Series(
            generator.standard_normal(2520) * 0.01 + 0.0004,
            index=pd.bdate_range("2013-01-01", periods=2520),
        )
        statistics = performance_statistics(returns)

        expected = (returns.mean() * 252) / (returns.std(ddof=1) * np.sqrt(252))
        assert statistics["sharpe_ratio"] == pytest.approx(expected)


# ----------------------------------------------------------------------
# Public-data loader
# ----------------------------------------------------------------------
class TestPublicDataLoader:
    """The loader is exercised against a stubbed yfinance.

    Hitting the live API in a test suite would make it slow, flaky and dependent
    on someone else's uptime. What is worth testing is the reshaping: pulling one
    field out of a multi-indexed frame, renaming Yahoo symbols back to the
    repository's tickers, stripping the timezone, and dropping series the API
    returned empty.
    """

    @staticmethod
    def _fake_download(n_days: int = 300, empty_symbol: str | None = None):
        symbols = list(EURO_STOXX_50_YAHOO.values())
        dates = pd.date_range("2013-01-02", periods=n_days, freq="B", tz="UTC")
        generator = np.random.default_rng(0)

        columns = pd.MultiIndex.from_product(
            [["Close", "High", "Low", "Open", "Volume"], symbols]
        )
        values = np.empty((n_days, len(columns)))
        for position, (field, _) in enumerate(columns):
            values[:, position] = (
                generator.integers(1e5, 1e7, n_days)
                if field == "Volume"
                else 100 * np.exp(np.cumsum(generator.standard_normal(n_days) * 0.01))
            )

        frame = pd.DataFrame(values, index=dates, columns=columns)
        if empty_symbol is not None:
            frame[("Close", empty_symbol)] = np.nan
            frame[("Volume", empty_symbol)] = np.nan
        return frame

    def _patched(self, frame):
        stub = types.SimpleNamespace(
            __version__="stub", download=lambda **kwargs: frame
        )
        return mock.patch.dict(sys.modules, {"yfinance": stub})

    def test_returns_repository_tickers_not_yahoo_symbols(self):
        with self._patched(self._fake_download()):
            prices, volumes = download_public_data()

        assert "ADS.DE" not in prices.columns  # Yahoo symbol
        assert "ADSGn.DE" in prices.columns  # repository ticker
        assert prices.columns.equals(volumes.columns)

    def test_index_is_naive_and_sorted(self):
        with self._patched(self._fake_download()):
            prices, _ = download_public_data()

        assert prices.index.tz is None
        assert prices.index.is_monotonic_increasing
        assert prices.index.name == "Date"

    def test_drops_symbols_the_api_returned_empty(self):
        symbols = list(EURO_STOXX_50_YAHOO.values())
        frame = self._fake_download(empty_symbol=symbols[3])

        with self._patched(frame):
            prices, volumes = download_public_data()

        missing = [
            ticker
            for ticker, symbol in EURO_STOXX_50_YAHOO.items()
            if symbol == symbols[3]
        ][0]
        assert missing not in prices.columns
        assert missing not in volumes.columns

    def test_output_feeds_the_estimation_pipeline(self):
        """The whole point: the loader's output must be a drop-in substitute."""
        with self._patched(self._fake_download()):
            prices, _ = download_public_data()

        returns = compute_returns(prices.ffill())
        window, _ = prepare_estimation_window(returns, returns.index[-1], 252, 0.95)
        decomposition = extract_factors(window, n_factors=4)

        assert window.shape[0] == 252
        assert decomposition.factors.shape[1] == 4

    def test_symbol_map_has_no_duplicates(self):
        symbols = list(EURO_STOXX_50_YAHOO.values())
        assert len(symbols) == len(set(symbols))
        assert len(EURO_STOXX_50_YAHOO) == 48

    def test_rejects_an_empty_download(self):
        with self._patched(pd.DataFrame()), pytest.raises(ValueError, match="no data"):
            download_public_data()
