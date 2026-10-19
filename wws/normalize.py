"""Quality control and normalization of wastewater measurements.

Steps, in order:
1. Non-detects: values below the limit of detection (LOD) are substituted with
   LOD/2 and flagged, so they stay visible instead of silently dropping out.
2. Outlier screen: a Hampel filter on log10 concentration, per site, flags
   points more than `k` scaled median absolute deviations from a
   rolling median (trailing by default). Flagged points are kept in the table but excluded from
   modelling.
3. Normalization, two complementary forms:
   - flow/population: gene copies per day per 100,000 people
     = concentration (gc/L) * flow (L/day) / population * 1e5
   - fecal-strength: target / PMMoV ratio, which also cancels dilution and
     shared laboratory batch effects.
4. log10 transform of each normalized series for modelling.
"""
import numpy as np
import pandas as pd

from .simulate import LOD_GC_PER_L


def substitute_non_detects(df, lod=LOD_GC_PER_L):
    out = df.copy()
    out['non_detect'] = out['target_gc_per_l'].isna() | out.get('below_lod', False)
    out['target_filled'] = out['target_gc_per_l'].where(~out['non_detect'], lod / 2.0)
    return out


def hampel_flags(values, window=7, k=3.5, center=False):
    """Boolean array: True where a value is an outlier versus its neighbours.

    The default trailing window uses only earlier samples, so a flag never
    depends on results that arrive later (safe for forecasting). Use
    center=True for retrospective cleaning.
    """
    s = pd.Series(np.asarray(values, dtype=float))
    med = s.rolling(window, center=center, min_periods=3).median()
    mad = (s - med).abs().rolling(window, center=center, min_periods=3).median()
    scale = 1.4826 * mad
    flags = (s - med).abs() > k * scale.replace(0, np.nan)
    return flags.fillna(False).to_numpy()


def normalize(samples, lod=LOD_GC_PER_L):
    df = substitute_non_detects(samples, lod)
    df = df.sort_values(['site_id', 'date']).reset_index(drop=True)
    df['log10_target'] = np.log10(df['target_filled'])
    df['outlier'] = False
    for sid, idx in df.groupby('site_id').groups.items():
        df.loc[idx, 'outlier'] = hampel_flags(df.loc[idx, 'log10_target'].to_numpy())
    litres_per_day = df['flow_mld'] * 1e6
    df['gc_per_day_per_100k'] = df['target_filled'] * litres_per_day / df['population'] * 1e5
    df['target_to_pmmov'] = df['target_filled'] / df['pmmov_gc_per_l']
    df['log10_load_100k'] = np.log10(df['gc_per_day_per_100k'])
    df['log10_ratio_pmmov'] = np.log10(df['target_to_pmmov'])
    return df


def qc_summary(df):
    rows = []
    for sid, g in df.groupby('site_id'):
        rows.append({'site_id': sid, 'samples': int(len(g)),
                     'non_detects': int(g['non_detect'].sum()),
                     'outliers_flagged': int(g['outlier'].sum())})
    return pd.DataFrame(rows)
