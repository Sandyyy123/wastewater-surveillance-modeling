"""Smoothed levels and trend categories from normalized wastewater signals.

`daily_signal` puts a site's sampled log10 series on a daily grid (last
sample carried forward, outliers removed) and smooths it with a trailing
7-day mean, so no future sample leaks into a given day.

`trend_category` fits a least-squares line to the last `window` days of the
smoothed log10 series and converts the slope into a percent change over the
window, then labels it. Thresholds are configurable; the defaults are a
reasonable starting point to agree with the surveillance team, not a standard.
"""
import numpy as np
import pandas as pd

DEFAULT_CUTS = ((-10.0, 'decreasing'), (10.0, 'stable'), (100.0, 'increasing'))


def daily_signal(norm, site_id, column='log10_ratio_pmmov', smooth_days=7):
    g = norm[(norm['site_id'] == site_id) & ~norm['outlier']].set_index('date')[column]
    full = pd.date_range(g.index.min(), g.index.max(), freq='D')
    # Carry the last sample forward: interpolating towards the NEXT sample
    # would use a result that is not yet available on that day.
    daily = g.reindex(full).ffill()
    return daily.rolling(smooth_days, min_periods=1).mean().rename(column)


def window_percent_change(log10_series, window=15):
    y = np.asarray(log10_series, dtype=float)[-window:]
    if len(y) < 3 or np.isnan(y).any():
        return np.nan
    x = np.arange(len(y), dtype=float)
    slope = np.polyfit(x, y, 1)[0]  # log10 units per day
    return float((10 ** (slope * (len(y) - 1)) - 1) * 100)


def trend_category(pct, cuts=DEFAULT_CUTS):
    if pct is None or np.isnan(pct):
        return 'insufficient data'
    for upper, label in cuts:
        if pct < upper:
            return label
    return 'sharply increasing'


def latest_trends(norm, sites, column='log10_ratio_pmmov', window=15, max_non_detect=0.5):
    """Latest 15-day trend per site.

    If more than `max_non_detect` of the samples in the window were below the
    limit of detection, the slope mostly reflects the substituted value, so
    the site is reported as 'mostly below detection' instead of a trend.
    """
    rows = []
    for sid in sites:
        s = daily_signal(norm, sid, column)
        g = norm[norm['site_id'] == sid]
        recent = g[g['date'] > s.index.max() - pd.Timedelta(days=window)]
        nd_share = float(recent['non_detect'].mean()) if len(recent) else 1.0
        pct = window_percent_change(s.to_numpy(), window)
        category = 'mostly below detection' if nd_share > max_non_detect else trend_category(pct)
        rows.append({'site_id': sid, 'as_of': s.index.max().date().isoformat(),
                     'pct_change_15d': None if np.isnan(pct) else round(pct, 1),
                     'non_detect_share_15d': round(nd_share * 100, 1),
                     'category': category})
    return pd.DataFrame(rows)
