"""Forecast reported cases from wastewater signals, validated out of sample.

Target: log10(1 + 7-day mean reported cases per 100k), `horizon` days ahead.
Features available on the forecast date only (no look-ahead):
  - smoothed log10 wastewater signal today, 7 days ago and 14 days ago
  - its 7-day change
  - the current log10 case level (so the model adds wastewater information
    on top of what is already known from case reports)
Model: ridge regression in closed form (numpy only), standardized features.
Validation: rolling-origin (expanding window) evaluation. Each fold trains on
dates strictly before the fold's test block, with a gap equal to the horizon
so training targets never overlap the test period.
Baseline: persistence (the forecast equals today's case level).
Prediction intervals: 5th-95th percentiles of out-of-sample residuals on a
calibration block at the end of each training window (in-sample residuals
are too optimistic and under-cover).
"""
import numpy as np
import pandas as pd

from .trend import daily_signal


def build_features(norm, cases, site_id, horizon=7, column='log10_ratio_pmmov'):
    s = daily_signal(norm, site_id, column)
    c = cases[cases['site_id'] == site_id].set_index('date').sort_index()
    pop = float(c['population'].iloc[0])
    case7 = c['cases'].rolling(7, min_periods=7).mean() / pop * 1e5
    df = pd.DataFrame({'ww': s}).join(np.log10(1 + case7).rename('log_cases'), how='inner')
    df['ww_lag7'] = df['ww'].shift(7)
    df['ww_lag14'] = df['ww'].shift(14)
    df['ww_change7'] = df['ww'] - df['ww_lag7']
    df['target'] = df['log_cases'].shift(-horizon)
    df['site_id'] = site_id
    return df.dropna(subset=['ww', 'ww_lag7', 'ww_lag14', 'log_cases'])


FEATURES = ('ww', 'ww_lag7', 'ww_lag14', 'ww_change7', 'log_cases')


class Ridge:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd == 0] = 1.0
        Z = (X - self.mu) / self.sd
        self.b0 = y.mean()
        A = Z.T @ Z + self.alpha * np.eye(Z.shape[1])
        self.coef = np.linalg.solve(A, Z.T @ (y - self.b0))
        resid = y - self.predict(X)
        self.q_lo, self.q_hi = np.quantile(resid, [0.05, 0.95])
        return self

    def predict(self, X):
        Z = (np.asarray(X, float) - self.mu) / self.sd
        return self.b0 + Z @ self.coef


def rolling_origin(df, horizon=7, n_folds=6, test_days=21, min_train=60, alpha=1.0, cal_days=28,
                  features=FEATURES):
    """Return per-prediction rows for all folds (all sites pooled per date).

    `features` lets the same evaluation run a cases-only model, which shows
    whether the wastewater signal adds skill beyond case reports.
    """
    data = df.dropna(subset=['target']).sort_index()
    dates = np.array(sorted(data.index.unique()))
    last_start = len(dates) - test_days
    starts = np.linspace(min_train + horizon, last_start, n_folds).astype(int)
    rows = []
    for fold, start in enumerate(starts):
        test_dates = dates[start:start + test_days]
        cutoff = dates[start - horizon]  # train targets end before the test block
        train = data[data.index < cutoff]
        test = data[data.index.isin(test_dates)]
        if len(train) < min_train or test.empty:
            continue
        # Interval calibration on out-of-sample residuals: fit on the earlier
        # part of the training data, score the last `cal_days`, then refit on
        # all training data for the point forecast.
        tdates = np.array(sorted(train.index.unique()))
        cal_start = tdates[-cal_days]
        early = train[train.index < tdates[-cal_days - horizon]]
        cal = train[train.index >= cal_start]
        m_cal = Ridge(alpha).fit(early[list(features)], early['target'])
        cal_resid = cal['target'].to_numpy() - m_cal.predict(cal[list(features)])
        q_lo, q_hi = np.quantile(cal_resid, [0.05, 0.95])
        m = Ridge(alpha).fit(train[list(features)], train['target'])
        pred = m.predict(test[list(features)])
        for (d, r), p in zip(test.iterrows(), pred):
            rows.append({'fold': fold, 'date': d, 'site_id': r['site_id'],
                         'train_end': cutoff, 'y_true': r['target'], 'y_pred': p,
                         'y_persist': r['log_cases'], 'lo': p + q_lo, 'hi': p + q_hi})
    return pd.DataFrame(rows)
