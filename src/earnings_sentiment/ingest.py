from __future__ import annotations
import pandas as pd
from pathlib import Path

def load_prices(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=['ticker','date','open','high','low','close','volume'])
    return pd.read_csv(path, parse_dates=['date'])

def load_earnings(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=['ticker','announce_datetime','bmo_amc','eps_actual','eps_estimate'])
    return pd.read_csv(path, parse_dates=['announce_datetime'])
