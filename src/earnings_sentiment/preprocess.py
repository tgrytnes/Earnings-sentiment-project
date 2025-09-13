from __future__ import annotations
import pandas as pd
from pathlib import Path
from .config import load_config
from .utils import ensure_dir

def align_events(prices: pd.DataFrame, earnings: pd.DataFrame) -> pd.DataFrame:
    if earnings.empty or prices.empty:
        return pd.DataFrame()
    p = prices.copy()
    p['date'] = pd.to_datetime(p['date']).dt.tz_localize(None)

    e = earnings.copy()
    e['announce_datetime'] = pd.to_datetime(e['announce_datetime']).dt.tz_localize(None)
    e['t0_date'] = e.apply(
        lambda r: r['announce_datetime'].date() if r['bmo_amc']=='BMO' else (r['announce_datetime'] + pd.Timedelta(days=1)).date(),
        axis=1
    )
    e['t0_date'] = pd.to_datetime(e['t0_date'])
    # Attach close_t0 and close_t1 for labeling convenience
    m = e.merge(p[['ticker','date','close']].rename(columns={'date':'t0_date','close':'close_t0'}),
                on=['ticker','t0_date'], how='left')
    p = p.sort_values(['ticker','date'])
    p['close_t1'] = p.groupby('ticker')['close'].shift(-1)
    m = m.merge(p[['ticker','date','close_t1']].rename(columns={'date':'t0_date'}),
                on=['ticker','t0_date'], how='left')
    return m

def main(config_path: str = "configs/exp_baseline.yaml"):
    cfg = load_config(config_path)
    from .ingest import load_prices, load_earnings
    prices = load_prices(Path(cfg.paths['raw_prices']))
    earnings = load_earnings(Path(cfg.paths['raw_earnings']))
    events = align_events(prices, earnings)
    out = Path(cfg.paths['artifacts']) / "events.csv"
    ensure_dir(out.parent)
    events.to_csv(out, index=False)
    print(f"Saved aligned events to {out} (rows={len(events)})")

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else "configs/exp_baseline.yaml")
