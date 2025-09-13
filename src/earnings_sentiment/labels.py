import pandas as pd

def make_day_ahead_label(events: pd.DataFrame) -> pd.DataFrame:
    ev = events.copy()
    if 'close_t0' in ev and 'close_t1' in ev:
        ev['y_d1'] = (ev['close_t1'] > ev['close_t0']).astype('Int64')
    else:
        ev['y_d1'] = pd.NA
    return ev
