"""Optional public-data loader, so the pipeline runs without licensed files.

The backtests in this repository use total-return index levels and traded volume
from a market data terminal, which cannot be redistributed. This module
reconstructs an approximate substitute from Yahoo Finance, letting anyone clone
the repository and run the full pipeline end to end.

The substitute is not the same dataset, and results will differ. Three
differences matter:

* **Prices, not total-return indices.** ``auto_adjust=True`` gives prices
  adjusted for splits and dividends, which is close to a total-return series but
  not identical in its treatment of withholding tax on dividends.
* **Survivorship, again.** The ticker list below is the index membership on a
  single date, held fixed. The licensed dataset has the same flaw, so on this
  axis the two are comparable, but neither is point-in-time.
* **Currency.** Most constituents trade in EUR, but a few historically quoted
  elsewhere. Yahoo returns each series in its listing currency, so a strict
  replication would convert them; the strategy is cross-sectional and
  dollar-neutral, so the effect is second order and is not corrected here.

Treat this as a way to exercise the code, not as a way to reproduce the numbers
in the README.

Requires the optional dependency::

    pip install "statarb-avellaneda-lee[public-data]"
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Euro Stoxx 50 membership, Yahoo Finance symbology. Kept alongside the RIC
# codes used by the licensed files so the two universes can be compared.
# Yahoo suffixes: .PA Paris, .DE Xetra, .AS Amsterdam, .MC Madrid, .MI Milan,
# .BR Brussels, .HE Helsinki, .IR Dublin.
EURO_STOXX_50_YAHOO: dict[str, str] = {
    "ABI.BR": "ABI.BR",
    "AD.AS": "AD.AS",
    "ADSGn.DE": "ADS.DE",
    "ADYEN.AS": "ADYEN.AS",
    "AIR.PA": "AIR.PA",
    "AIRP.PA": "AI.PA",
    "ALVG.DE": "ALV.DE",
    "ASML.AS": "ASML.AS",
    "AXAF.PA": "CS.PA",
    "BASFn.DE": "BAS.DE",
    "BAYGn.DE": "BAYN.DE",
    "BBVA.MC": "BBVA.MC",
    "BMWG.DE": "BMW.DE",
    "BNPP.PA": "BNP.PA",
    "CRH.I": "CRH.L",
    "DANO.PA": "BN.PA",
    "DB1Gn.DE": "DB1.DE",
    "DPWGn.DE": "DHL.DE",
    "DTEGn.DE": "DTE.DE",
    "ENEI.MI": "ENEL.MI",
    "ENI.MI": "ENI.MI",
    "ESLX.PA": "EL.PA",
    "FLTRF.I": "FLTR.L",
    "HRMS.PA": "RMS.PA",
    "IBE.MC": "IBE.MC",
    "IFXGn.DE": "IFX.DE",
    "INGA.AS": "INGA.AS",
    "ISP.MI": "ISP.MI",
    "ITX.MC": "ITX.MC",
    "LINI.DE": "LIN.DE",
    "LVMH.PA": "MC.PA",
    "MBGn.DE": "MBG.DE",
    "MUVGn.DE": "MUV2.DE",
    "NDAFI.HE": "NDA-FI.HE",
    "NOKIA.HE": "NOKIA.HE",
    "OREP.PA": "OR.PA",
    "PERP.PA": "RI.PA",
    "PRTP.PA": "CFR.SW",
    "PRX.AS": "PRX.AS",
    "SAF.PA": "SAF.PA",
    "SAN.MC": "SAN.MC",
    "SAPG.DE": "SAP.DE",
    "SASY.PA": "SAN.PA",
    "SCHN.PA": "SU.PA",
    "SGEF.PA": "DG.PA",
    "SIEGn.DE": "SIE.DE",
    "STLAM.MI": "STLAM.MI",
    "TTEF.PA": "TTE.PA",
    "VNAn.DE": "VNA.DE",
    "VOW": "VOW3.DE",
}

DEFAULT_START = "2013-01-02"
DEFAULT_END = "2023-02-17"


def download_public_data(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    symbols: dict[str, str] | None = None,
    min_coverage: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download adjusted prices and volumes from Yahoo Finance.

    Parameters:
        start (str): First date, inclusive, as ``YYYY-MM-DD``.
        end (str): Last date, exclusive of the following day.
        symbols (dict[str, str] | None): Mapping from the ticker names used
            throughout the repository to Yahoo symbols. Defaults to
            :data:`EURO_STOXX_50_YAHOO`.
        min_coverage (float): Drop any name observed on less than this share of
            the sample. Yahoo occasionally returns an empty series for a symbol
            that has been renamed or delisted, and a column of missing values
            would otherwise propagate into the estimation window.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Prices and volumes, dates as index
            and repository ticker names as columns.

    Raises:
        ImportError: If yfinance is not installed.
        ValueError: If the download returns nothing usable.
    """
    try:
        import yfinance
    except ImportError as error:  # pragma: no cover - depends on environment
        raise ImportError(
            'yfinance is required for the public-data loader; install it with '
            'pip install "statarb-avellaneda-lee[public-data]"'
        ) from error

    symbols = symbols or EURO_STOXX_50_YAHOO

    raw = yfinance.download(
        tickers=list(symbols.values()),
        start=start,
        end=end,
        auto_adjust=True,      # splits and dividends folded into the price
        group_by="column",     # outer level is the field, inner the ticker
        progress=False,
        threads=True,
    )

    if raw is None or raw.empty:
        raise ValueError(
            "Yahoo Finance returned no data; check connectivity and the date range"
        )

    prices = _extract_field(raw, "Close", symbols)
    volumes = _extract_field(raw, "Volume", symbols)

    return _drop_sparse_columns(prices, volumes, min_coverage)


def _extract_field(
    raw: pd.DataFrame,
    field: str,
    symbols: dict[str, str],
) -> pd.DataFrame:
    """Pull one field out of the multi-indexed frame and rename to repo tickers.

    Parameters:
        raw (pd.DataFrame): Frame returned by ``yfinance.download``.
        field (str): ``Close`` or ``Volume``.
        symbols (dict[str, str]): Repository ticker to Yahoo symbol.

    Returns:
        pd.DataFrame: One column per successfully downloaded ticker.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        if field not in raw.columns.get_level_values(0):
            raise ValueError(f"field {field!r} missing from the download")
        frame = raw[field].copy()
    else:
        # A single ticker comes back with flat columns.
        frame = raw[[field]].copy()
        frame.columns = list(symbols.values())

    reverse = {yahoo: repository for repository, yahoo in symbols.items()}
    available = [column for column in frame.columns if column in reverse]
    frame = frame.loc[:, available].rename(columns=reverse)

    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    frame.index.name = "Date"

    return frame.sort_index().loc[:, sorted(frame.columns)]


def _drop_sparse_columns(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    min_coverage: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep names observed often enough, on the intersection of both frames.

    Parameters:
        prices (pd.DataFrame): Adjusted close prices.
        volumes (pd.DataFrame): Traded volumes.
        min_coverage (float): Minimum share of non-missing prices.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Filtered prices and volumes.
    """
    shared = prices.columns.intersection(volumes.columns)
    prices = prices.loc[:, shared]
    volumes = volumes.loc[:, shared]

    coverage = prices.notna().mean()
    retained = coverage.index[coverage >= min_coverage]

    dropped = sorted(set(shared) - set(retained))
    if dropped:
        print(f"dropped {len(dropped)} sparse tickers: {', '.join(dropped)}")

    if len(retained) < 10:
        raise ValueError(
            f"only {len(retained)} usable tickers; the strategy needs a "
            f"cross-section, check the symbol mapping"
        )

    return prices.loc[:, retained], volumes.loc[:, retained]


def write_public_data(
    destination: Path | str = "data",
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> tuple[Path, Path]:
    """Download and write CSVs in the layout the rest of the package expects.

    Files are written under the ``public_`` prefix so they can never be mistaken
    for, or silently overwrite, the licensed dataset.

    Parameters:
        destination (Path | str): Directory to write into.
        start (str): First date.
        end (str): Last date.

    Returns:
        tuple[Path, Path]: Paths of the price and volume files.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    prices, volumes = download_public_data(start=start, end=end)

    price_path = destination / "public_sx5e_underlyings.csv"
    volume_path = destination / "public_volume.csv"
    prices.to_csv(price_path)
    volumes.to_csv(volume_path)

    print(
        f"wrote {prices.shape[1]} tickers x {prices.shape[0]} days "
        f"({prices.index[0].date()} to {prices.index[-1].date()})"
    )
    return price_path, volume_path


if __name__ == "__main__":
    write_public_data()
