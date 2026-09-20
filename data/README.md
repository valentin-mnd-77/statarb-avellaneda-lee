# Data

Three daily files covering the Euro Stoxx 50 constituents, 2013-01-02 to
2023-02-17:

| File | Contents |
|---|---|
| `sx5e_underlyings.csv` | Total-return index levels in EUR, one column per ticker |
| `volume.csv` | Traded share volume, same tickers; gaps encoded as `#N/A N/A` |
| `ticker_details.csv` | Name, Bloomberg code and GICS sector per ticker |

**Universe caveat.** Membership is the *current* index composition held fixed
over the whole period. Names that left the index are absent, which is a
survivorship bias that inflates every backtest result in this repository. A
production system needs point-in-time membership.

**Calendar caveat.** The universe spans several exchanges with different holiday
calendars, so prices are forward-filled and the resulting artificial zero returns
are reported by `statarb.data.coverage_report`.

If you cannot redistribute your own market data, delete these files, add
`data/*.csv` to `.gitignore`, and keep this note so the expected schema is clear.
