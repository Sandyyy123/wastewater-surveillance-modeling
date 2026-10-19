import numpy as np
import pandas as pd
import pytest

from main import run
from wws.model import Ridge, build_features, rolling_origin
from wws.normalize import hampel_flags, normalize, substitute_non_detects
from wws.simulate import LOD_GC_PER_L, simulate
from wws.trend import trend_category, window_percent_change


@pytest.fixture(scope='module')
def data():
    samples, cases = simulate(seed=7)
    return samples, cases, normalize(samples)


def test_non_detects_substituted_with_half_lod():
    df = pd.DataFrame({'target_gc_per_l': [np.nan, 5000.0], 'below_lod': [True, False]})
    out = substitute_non_detects(df)
    assert out['target_filled'].tolist() == [LOD_GC_PER_L / 2, 5000.0]
    assert out['non_detect'].tolist() == [True, False]


def test_flow_population_normalization_formula():
    df = pd.DataFrame({'site_id': ['S'], 'date': [pd.Timestamp('2025-01-01')],
                       'target_gc_per_l': [10000.0], 'below_lod': [False],
                       'pmmov_gc_per_l': [1e6], 'flow_mld': [50.0], 'population': [100000]})
    n = normalize(df).iloc[0]
    # 10,000 gc/L * 50e6 L/day / 100,000 people * 1e5 = 5e11 gc/day/100k
    assert n['gc_per_day_per_100k'] == pytest.approx(5e11)
    assert n['target_to_pmmov'] == pytest.approx(0.01)


def test_hampel_flags_spike_but_not_smooth_trend():
    values = np.linspace(3.0, 4.0, 30)
    values[20] += 1.5
    flags = hampel_flags(values)
    assert flags[20] and flags.sum() == 1


def test_injected_outliers_are_flagged(data):
    _, _, norm = data
    assert norm.groupby('site_id')['outlier'].sum().min() >= 1


def test_trend_categories():
    rising = np.log10(np.linspace(100, 300, 15))
    falling = rising[::-1]
    assert trend_category(window_percent_change(rising)) in ('increasing', 'sharply increasing')
    assert trend_category(window_percent_change(falling)) == 'decreasing'
    assert trend_category(float('nan')) == 'insufficient data'


def test_ridge_recovers_linear_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = 1.0 + 2.0 * X[:, 0] - 1.0 * X[:, 1]
    m = Ridge(alpha=1e-6).fit(X, y)
    assert np.allclose(m.predict(X), y, atol=1e-3)


def test_rolling_origin_has_no_temporal_leakage(data):
    _, cases, norm = data
    feats = build_features(norm, cases, 'SITE_A')
    preds = rolling_origin(feats)
    assert not preds.empty
    # Training data ends at least `horizon` days before every test date.
    gap = (preds['date'] - preds['train_end']).dt.days
    assert gap.min() >= 7


def test_features_do_not_use_future_samples(data):
    samples, cases, _ = data
    cut = pd.Timestamp('2025-06-01')
    full = build_features(normalize(samples), cases, 'SITE_B')
    early = build_features(normalize(samples[samples['date'] <= cut]), cases, 'SITE_B')
    common = early.index[early.index <= cut]
    cols = ['ww', 'ww_lag7', 'ww_lag14', 'ww_change7']
    pd.testing.assert_frame_equal(full.loc[common, cols], early.loc[common, cols])


def test_end_to_end_wastewater_beats_persistence(tmp_path):
    report = run(out=tmp_path)
    assert report['leakage_check'] is True
    assert report['pmmov_normalized']['skill_vs_persistence_pct'] > 0
    assert (tmp_path / 'report.json').is_file()
