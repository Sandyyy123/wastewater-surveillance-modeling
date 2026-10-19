"""Run the full demo pipeline on synthetic data and write outputs/.

Usage: python main.py [--horizon 7] [--seed 7] [--out outputs]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from wws.model import CASE_FEATURES, COMBINED_FEATURES, build_features, rolling_origin
from wws.normalize import normalize, qc_summary
from wws.simulate import simulate
from wws.trend import latest_trends
from wws.validate import metrics, metrics_by_site


def run(horizon=7, seed=7, out='outputs'):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    samples, cases = simulate(seed=seed)
    norm = normalize(samples)
    sites = sorted(norm['site_id'].unique())
    feats = pd.concat([build_features(norm, cases, s, horizon) for s in sites])
    preds = rolling_origin(feats, horizon=horizon)
    # Fair comparison on identical rows: case trend alone, and both combined.
    preds_cases = rolling_origin(feats, horizon=horizon, features=CASE_FEATURES)
    preds_comb = rolling_origin(feats, horizon=horizon, features=COMBINED_FEATURES)
    # Same evaluation with flow/population normalization for comparison.
    feats_flow = pd.concat([build_features(norm, cases, s, horizon, 'log10_load_100k') for s in sites])
    preds_flow = rolling_origin(feats_flow, horizon=horizon)
    report = {
        'data': 'synthetic, illustrative only',
        'seed': seed, 'horizon_days': horizon,
        'sites': len(sites), 'samples': int(len(samples)),
        'qc': qc_summary(norm).to_dict(orient='records'),
        'folds': int(preds['fold'].nunique()),
        'pmmov_normalized': metrics(preds),
        'flow_population_normalized': metrics(preds_flow),
        'case_trend_only': metrics(preds_cases),
        'combined_wastewater_and_case_trend': metrics(preds_comb),
        'by_site_pmmov': metrics_by_site(preds),
        'latest_trends': latest_trends(norm, sites).to_dict(orient='records'),
        'min_train_test_gap_days': int((preds['date'] - preds['train_end']).dt.days.min()),
    }
    norm.to_csv(out / 'normalized_samples.csv', index=False)
    preds.to_csv(out / 'forecasts_pmmov.csv', index=False)
    (out / 'report.json').write_text(json.dumps(report, indent=2, default=str))
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--horizon', type=int, default=7)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default='outputs')
    a = ap.parse_args()
    r = run(a.horizon, a.seed, a.out)
    print(json.dumps({k: r[k] for k in ('samples', 'folds', 'pmmov_normalized',
                                        'flow_population_normalized', 'case_trend_only',
                                        'combined_wastewater_and_case_trend',
                                        'min_train_test_gap_days')}, indent=2))
