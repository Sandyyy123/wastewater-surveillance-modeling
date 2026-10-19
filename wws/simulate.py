"""Synthetic wastewater surveillance data (illustrative only, no real sites).

Each site has a population, a daily influent flow series and a latent daily
infection curve made of two epidemic waves. Wastewater samples are taken on
three weekdays. The measured target concentration (gene copies per litre)
depends on shedding by infected people, dilution by flow, a fecal-strength
factor that the PMMoV marker also tracks, laboratory batch effects,
log-normal measurement noise and a limit of detection (LOD). Reported clinical
cases follow infections with a delay and an under-ascertainment fraction.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

LOD_GC_PER_L = 800.0


@dataclass(frozen=True)
class Site:
    site_id: str
    population: int
    base_flow_mld: float  # megalitres per day (MLD)
    wave_peaks: tuple  # day index of each wave peak
    wave_heights: tuple  # peak daily infections per 100k


SITES = (
    Site('SITE_A', 250_000, 95.0, (70, 230), (180.0, 120.0)),
    Site('SITE_B', 60_000, 21.0, (85, 245), (220.0, 90.0)),
    Site('SITE_C', 140_000, 48.0, (60, 215), (150.0, 160.0)),
)


def _infections_per_100k(days, peaks, heights, width=22.0, floor=4.0):
    curve = np.full(days.shape, floor, dtype=float)
    for peak, height in zip(peaks, heights):
        curve += height * np.exp(-0.5 * ((days - peak) / width) ** 2)
    return curve


def simulate(n_days=300, seed=7, start='2025-01-06'):
    """Return (samples, cases) data frames for all synthetic sites."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq='D')
    days = np.arange(n_days, dtype=float)
    sample_rows, case_rows = [], []
    for site in SITES:
        inf100k = _infections_per_100k(days, site.wave_peaks, site.wave_heights)
        infections = inf100k * site.population / 1e5
        # Shedding persists for several days after infection: 7-day kernel.
        kernel = np.array([0.05, 0.15, 0.25, 0.2, 0.15, 0.12, 0.08])
        shedders = np.convolve(infections, kernel, mode='full')[:n_days]
        rain = rng.random(n_days) < 0.12
        flow = site.base_flow_mld * np.exp(0.06 * rng.standard_normal(n_days)) * np.where(rain, 1.8, 1.0)
        fecal = np.exp(0.25 * rng.standard_normal(n_days))  # daily fecal strength
        pmmov = 3.0e7 * fecal * site.population / (flow * 1e6) * 4.0  # gc/L
        load = 4.0e9 * shedders * fecal  # gene copies per day reaching the plant
        conc = load / (flow * 1e6)  # gc/L
        batch = np.where((days // 28) % 3 == 1, 0.7, 1.0)  # periodic lab batch shift
        measured = conc * batch * np.exp(0.35 * rng.standard_normal(n_days))
        pmmov_measured = pmmov * batch * np.exp(0.2 * rng.standard_normal(n_days))
        # Reported cases: 5-day reporting delay, 35% ascertainment, Poisson noise.
        expected_cases = 0.35 * np.roll(infections, 5)
        expected_cases[:5] = 0.35 * infections[0]
        cases = rng.poisson(expected_cases)
        for i, d in enumerate(dates):
            case_rows.append({'site_id': site.site_id, 'date': d, 'cases': int(cases[i]),
                              'population': site.population})
            if d.dayofweek in (0, 2, 4):
                value = measured[i]
                sample_rows.append({
                    'site_id': site.site_id, 'date': d,
                    'target_gc_per_l': np.nan if value < LOD_GC_PER_L else round(float(value), 1),
                    'below_lod': bool(value < LOD_GC_PER_L),
                    'pmmov_gc_per_l': round(float(pmmov_measured[i]), 1),
                    'flow_mld': round(float(flow[i]), 3),
                    'population': site.population,
                    'lab_batch': int(days[i] // 28),
                })
    samples = pd.DataFrame(sample_rows)
    cases = pd.DataFrame(case_rows)
    # One gross outlier per site to exercise the outlier screen.
    for sid in samples['site_id'].unique():
        idx = samples.index[(samples['site_id'] == sid) & samples['target_gc_per_l'].notna()][40]
        samples.loc[idx, 'target_gc_per_l'] *= 25
    return samples, cases
