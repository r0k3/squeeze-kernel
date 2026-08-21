"""Build the materialised equity returns panel shipped with the examples.

Provenance script — documents exactly how ``equity_panel_returns.parquet``
was produced. Running it requires a local market-data lake (ArcticDB with
Yahoo-sourced EOD bars); end users do NOT need to run it — the parquet is
committed so the backtest example reproduces from the repo alone.

Construction:
  1. Candidates: US common stocks in the lake with dividend-adjusted daily
     bars starting on/before 2000-01-03 and current through 2026-08.
  2. Liquidity rank: median daily dollar volume (close x volume) over the
     full sample.
  3. Calendar: trading days on which at least 90% of candidates have a bar.
  4. Prices forward-filled over gaps of at most 5 trading days; symbols with
     less than 99.5% coverage or any longer gap are dropped.
  5. Daily simple returns from dividend-adjusted closes (total-return style).
     Symbols with any absolute daily return above 75% are dropped (these are
     data glitches at this liquidity tier, not market moves).
  6. The 300 most liquid survivors are kept, columns alphabetical,
     stored float32 / zstd.

Known caveat, disclosed in the example: the universe is selected with
hindsight (names that survived to 2026), so absolute performance is
optimistic. Both covariance estimators run on the identical panel, so the
comparison between them is unaffected.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from datastore_yahoo import YahooLake, LIB_ADJ, LIB_RAW

UNIVERSE_STATUS = Path.home() / "projects/investment-universe/universe/ohlcv_status.jsonl"
OUT_PATH = Path(__file__).resolve().parent / "equity_panel_returns.parquet"

START = "2000-01-01"
FIRST_BAR_ON_OR_BEFORE = "2000-01-03"
N_TARGET = 300
CALENDAR_QUORUM = 0.90
MAX_FFILL_GAP = 5
MIN_COVERAGE = 0.995
MAX_ABS_RETURN = 0.75


def candidates() -> list[str]:
    rows = [json.loads(line) for line in UNIVERSE_STATUS.read_text().splitlines() if line.strip()]
    return sorted(
        r["symbol"]
        for r in rows
        if r.get("status") == "ok"
        and r.get("kind") == "stock"
        and (r.get("start") or "9999") <= FIRST_BAR_ON_OR_BEFORE
        and (r.get("end") or "0") >= "2026-07-01"
    )


def main() -> None:
    lake = YahooLake()
    syms = candidates()
    print(f"{len(syms)} candidate symbols with history since {FIRST_BAR_ON_OR_BEFORE}")

    # Liquidity rank + adjusted closes in one pass over the lake.
    liquidity: dict[str, float] = {}
    closes: dict[str, pd.Series] = {}
    for i, sym in enumerate(syms):
        raw = lake.read_ohlcv(LIB_RAW, sym)
        adj = lake.read_ohlcv(LIB_ADJ, sym)
        if raw.empty or adj.empty:
            continue
        raw = raw.loc[START:]
        adj = adj.loc[START:]
        liquidity[sym] = float((raw["close"] * raw["volume"]).median())
        closes[sym] = adj["close"]
        if (i + 1) % 200 == 0:
            print(f"  read {i + 1}/{len(syms)}")

    prices = pd.DataFrame(closes).sort_index()
    # Trading calendar: days where a quorum of candidates has a bar.
    quorum = prices.notna().mean(axis=1) >= CALENDAR_QUORUM
    prices = prices.loc[quorum]
    print(f"calendar: {len(prices)} trading days {prices.index[0].date()} -> {prices.index[-1].date()}")

    # Coverage screen on the quorum calendar, then bounded forward-fill.
    coverage = prices.notna().mean()
    gap = prices.isna().astype(int)
    max_gap = gap.apply(lambda c: (c.groupby((c == 0).cumsum()).cumsum()).max())
    keep = coverage.index[(coverage >= MIN_COVERAGE) & (max_gap <= MAX_FFILL_GAP)]
    prices = prices[keep].ffill(limit=MAX_FFILL_GAP)
    prices = prices.dropna(axis=1)  # anything still NaN (leading edge) goes
    print(f"{prices.shape[1]} symbols pass coverage/gap screen")

    rets = prices.pct_change().iloc[1:]
    glitch = rets.abs().max() > MAX_ABS_RETURN
    if glitch.any():
        print(f"dropping {int(glitch.sum())} symbols on the glitch screen: "
              f"{sorted(rets.columns[glitch])}")
    rets = rets.loc[:, ~glitch]

    top = sorted(sorted(rets.columns, key=lambda s: -liquidity[s])[:N_TARGET])
    panel = rets[top].astype(np.float32)
    panel.index.name = "date"
    print(f"final panel: {panel.shape[0]} days x {panel.shape[1]} symbols")

    panel.to_parquet(OUT_PATH, compression="zstd")
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"wrote {OUT_PATH.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
