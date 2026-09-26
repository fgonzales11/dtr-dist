"""
Distribution Transformer Loss-of-Life Analysis (single transformer)

Streamlit app that analyzes the thermal impact of added load - typically EV
charging - on ONE overhead-type, liquid-immersed, self-cooled distribution
transformer, including insulation loss of life.

Standards basis
* IEEE Std C57.12.20-2017, Overhead-Type Distribution Transformers 500 kVA and
  Smaller (IEEE-Std-C57-12-20-2017.pdf in this project):
    - Clause 1.1 scope: single-/three-phase, 60 Hz, liquid-immersed, self-cooled,
      500 kVA and smaller, HV 34 500 V and below, LV 7970/13 800Y V and below
    - Clause 4.1 thermal basis of the kVA rating: 65 °C average winding rise or
      80 °C hottest-spot rise, top-liquid rise not above 65 °C, under the usual
      service conditions of IEEE Std C57.12.00
    - Table 1 kVA ratings, Tables 2-4 voltage ratings / BIL / minimum kVA,
      Table 12 minimum percent impedance
* IEEE Std C57.91 (normative reference in C57.12.20 Clause 2; NOT included in
  this project): top-oil / hottest-spot temperature model and the insulation
  aging (loss-of-life) equations. Check the equations and constants against your
  copy of C57.91 before relying on the results.

Defaults marked "typical" or "illustrative" are placeholders - replace them with
the transformer's certified test report and your own load data.

Run with:  python -m streamlit run dist_dtr_app.py
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


# =============================================================================
# IEEE Std C57.12.20-2017 data
# =============================================================================

PHASES = ('Single-phase', 'Three-phase')

# Table 1 - Kilovolt-ampere ratings
KVA_RATINGS = {
    'Single-phase': [10, 15, 25, 37.5, 50, 75, 100, 167, 250, 333, 500],
    'Three-phase': [15, 30, 45, 75, 112.5, 150, 225, 300, 500],
}

# Table 12 - Minimum percent impedance voltage (low-voltage rating of 600 V and below)
MIN_IMPEDANCE_PCT = {
    'Single-phase': {10: 1.0, 15: 1.0, 25: 1.0, 37.5: 1.5, 50: 1.5, 75: 1.5,
                     100: 2.0, 167: 2.0, 250: 2.5, 333: 2.5, 500: 3.5},
    'Three-phase': {15: 1.0, 30: 1.0, 45: 1.0, 75: 1.0, 112.5: 1.5, 150: 1.5,
                    225: 1.5, 300: 2.0, 500: 2.0},
}

# Low-voltage column groups used by the minimum-kVA columns of Tables 2-4
LV_COLUMNS = {
    'Single-phase': {'A': '120/240, 277, 240/480, 346 or 600 V',
                     'B': '2400 or 4800 V',
                     'C': '7200, 7620 or 7970 V'},
    'Three-phase': {'A': '208Y/120, 480Y/277 or 600Y/346 V',
                    'B': '240Δ, 480Δ, 240Δ × 480Δ or 600Δ V',
                    'C': '2400Δ or 4800Δ V',
                    'D': '4160Y/2400 or 8320Y/4800 V'},
}


@dataclass(frozen=True)
class LowVoltageRating:
    label: str
    column: str  # Column group in Tables 2-4
    current_basis_v: float  # Full-winding (single-phase) or line-to-line (three-phase) volts


@dataclass(frozen=True)
class HighVoltageRating:
    label: str
    table: str
    bil_kv: str
    min_kva: Tuple[Optional[float], ...]  # Per LV column group (A, B, ...); None = not listed
    current_basis_v: float  # Winding volts (single-phase) or line-to-line volts (three-phase)


LV_RATINGS = {
    'Single-phase': [
        LowVoltageRating('120/240', 'A', 240), LowVoltageRating('277', 'A', 277),
        LowVoltageRating('240/480', 'A', 480), LowVoltageRating('346', 'A', 346),
        LowVoltageRating('600', 'A', 600), LowVoltageRating('2400', 'B', 2400),
        LowVoltageRating('4800', 'B', 4800), LowVoltageRating('7200', 'C', 7200),
        LowVoltageRating('7620', 'C', 7620), LowVoltageRating('7970', 'C', 7970),
    ],
    'Three-phase': [
        LowVoltageRating('208Y/120', 'A', 208), LowVoltageRating('480Y/277', 'A', 480),
        LowVoltageRating('600Y/346', 'A', 600), LowVoltageRating('240Δ', 'B', 240),
        LowVoltageRating('480Δ', 'B', 480),
        LowVoltageRating('240Δ × 480Δ (480 V connection)', 'B', 480),
        LowVoltageRating('600Δ', 'B', 600), LowVoltageRating('2400Δ', 'C', 2400),
        LowVoltageRating('4800Δ', 'C', 4800), LowVoltageRating('4160Y/2400', 'D', 4160),
        LowVoltageRating('8320Y/4800', 'D', 8320),
    ],
}


def _hv(label, table, bil, min_kva, volts):
    return HighVoltageRating(label, table, bil, tuple(min_kva), volts)


HV_RATINGS = {
    'Single-phase': [
        # Table 2 - single ratio: minimum kVA for LV columns (A, B, C)
        _hv('12470GrdY/7200', 'Table 2', '95', (10, 50, None), 7200),
        _hv('13200GrdY/7620', 'Table 2', '95', (10, 50, None), 7620),
        _hv('13800GrdY/7970', 'Table 2', '95', (10, 50, None), 7970),
        _hv('24940GrdY/14400', 'Table 2', '125', (10, 50, 50), 14400),
        _hv('34500GrdY/19920 (125 kV BIL)', 'Table 2', '125', (10, 50, 50), 19920),
        _hv('34500GrdY/19920 (150 kV BIL)', 'Table 2', '150', (10, 50, 50), 19920),
        _hv('2400/4160Y', 'Table 2', '60', (10, None, None), 2400),
        _hv('4800/8320Y', 'Table 2', '75', (10, None, None), 4800),
        _hv('7200/12470Y', 'Table 2', '95', (10, 50, None), 7200),
        _hv('7620/13200Y', 'Table 2', '95', (10, 50, None), 7620),
        _hv('13200/22860Y', 'Table 2', '125', (10, 50, 50), 13200),
        _hv('13800/23900Y', 'Table 2', '125', (10, 50, 50), 13800),
        _hv('14400/24940Y', 'Table 2', '125', (10, 50, 50), 14400),
        _hv('12000', 'Table 2', '95', (10, 50, 50), 12000),
        _hv('13200', 'Table 2', '95', (10, 50, 50), 13200),
        _hv('13800', 'Table 2', '95', (10, 50, 50), 13800),
        _hv('16340', 'Table 2', '95', (10, 50, 50), 16340),
        _hv('34500', 'Table 2', '200', (25, 50, 50), 34500),
        # Table 3 - series-multiple: LV columns (A, B); current on the higher-voltage connection
        _hv('4160GrdY/2400 × 12470GrdY/7200', 'Table 3', '60 × 95', (10, None, None), 7200),
        _hv('4160GrdY/2400 × 13200GrdY/7620', 'Table 3', '60 × 95', (10, None, None), 7620),
        _hv('8320GrdY/4800 × 12470GrdY/7200', 'Table 3', '75 × 95', (10, None, None), 7200),
        _hv('8320GrdY/4800 × 13200GrdY/7620', 'Table 3', '75 × 95', (10, None, None), 7620),
        _hv('12470GrdY/7200 × 24940GrdY/14400', 'Table 3', '95 × 125', (10, None, None), 14400),
        _hv('13200GrdY/7620 × 24940GrdY/14400', 'Table 3', '95 × 125', (10, None, None), 14400),
        _hv('2400/4160Y × 7200/12470Y', 'Table 3', '60 × 95', (10, 50, None), 7200),
        _hv('2400/4160Y × 7620/13200Y', 'Table 3', '60 × 95', (10, 50, None), 7620),
        _hv('4800/8320Y × 7200/12470Y', 'Table 3', '75 × 95', (10, 50, None), 7200),
        _hv('4800/8320Y × 7620/13200Y', 'Table 3', '75 × 95', (10, 50, None), 7620),
        _hv('7200/12470Y × 14400/24940Y', 'Table 3', '95 × 125', (10, 50, None), 14400),
        _hv('7620/13200Y × 14400/24940Y', 'Table 3', '95 × 125', (10, 50, None), 14400),
    ],
    'Three-phase': [
        # Table 4: minimum kVA for LV columns (A, B, C, D)
        _hv('2400Δ', 'Table 4', '45', (15, 15, None, None), 2400),
        _hv('4160Δ', 'Table 4', '60', (15, 15, None, None), 4160),
        _hv('4800Δ', 'Table 4', '60', (15, 15, None, None), 4800),
        _hv('7200Δ', 'Table 4', '75', (15, 15, 150, 150), 7200),
        _hv('12000Δ', 'Table 4', '95', (15, 15, 150, 150), 12000),
        _hv('13200Δ', 'Table 4', '95', (15, 15, 150, 150), 13200),
        _hv('13800Δ', 'Table 4', '95', (15, 15, 150, 150), 13800),
        _hv('4160Y', 'Table 4', '60', (None, 15, None, None), 4160),
        _hv('8320Y', 'Table 4', '75', (None, 15, None, None), 8320),
        _hv('12470Y', 'Table 4', '95', (None, 15, None, None), 12470),
        _hv('13200Y', 'Table 4', '95', (None, 15, None, None), 13200),
        _hv('4160Y/2400', 'Table 4', '60', (15, None, None, None), 4160),
        _hv('8320Y/4800', 'Table 4', '75', (15, None, None, None), 8320),
        _hv('12470Y/7200', 'Table 4', '95', (15, None, None, None), 12470),
        _hv('13200Y/7620', 'Table 4', '95', (15, None, None, None), 13200),
        _hv('13800GrdY/7970', 'Table 4', '95', (15, None, None, None), 13800),
        _hv('24940GrdY/14400', 'Table 4', '125', (15, None, None, None), 24940),
        _hv('34500GrdY/19920 (125 kV BIL)', 'Table 4', '125', (15, None, None, None), 34500),
        _hv('34500GrdY/19920 (150 kV BIL)', 'Table 4', '150', (15, None, None, None), 34500),
    ],
}

# Thermal basis of the kVA rating (C57.12.20 Clause 4.1)
RATED_HOT_SPOT_RISE_C = 80.0
MAX_TOP_OIL_RISE_C = 65.0

# Usual service conditions of IEEE Std C57.12.00 (referenced by C57.12.20 Clause 4.1)
USUAL_AMBIENT_AVG_C = 30.0
USUAL_AMBIENT_MAX_C = 40.0


def minimum_kva(hv: HighVoltageRating, lv: LowVoltageRating) -> Optional[float]:
    """Minimum kVA listed in Tables 2-4 for this HV/LV combination (None = not listed)"""
    index = 'ABCD'.index(lv.column)
    return hv.min_kva[index] if index < len(hv.min_kva) else None


def table12_min_impedance(phases: str, kva: float, lv: LowVoltageRating) -> Optional[float]:
    """Table 12 minimum %Z, which applies to low-voltage ratings of 600 V and below"""
    if lv.current_basis_v > 600:
        return None
    return MIN_IMPEDANCE_PCT[phases].get(kva)


# =============================================================================
# Transformer thermal model and insulation aging (IEEE C57.91)
# =============================================================================

# Aging acceleration factor constants: F_AA = 1 at a 110 °C hottest spot
AGING_B = 15000.0
AGING_REFERENCE_HOT_SPOT_C = 110.0

# Normal insulation life options (IEEE C57.91 Table 2)
NORMAL_LIFE_HOURS = {
    'Distribution transformer functional life test (180,000 h)': 180000.0,
    '200 retained degree of polymerization (150,000 h)': 150000.0,
    '25% retained tensile strength (135,000 h)': 135000.0,
    '50% retained tensile strength (65,000 h)': 65000.0,
}


@dataclass
class DistributionTransformer:
    """Nameplate, test-report and thermal data for one distribution transformer"""
    phases: str
    kva: float
    hv: HighVoltageRating
    lv: LowVoltageRating
    impedance_pct: float
    no_load_loss_w: float
    load_loss_w: float  # At rated kVA
    top_oil_rise_rated: float = 55.0  # °C over ambient at rated load
    hot_spot_rise_rated: float = RATED_HOT_SPOT_RISE_C  # °C over ambient at rated load
    oil_time_constant_min: float = 180.0
    winding_time_constant_min: float = 4.0
    oil_exponent: float = 0.8  # n, self-cooled (ONAN)
    winding_exponent: float = 0.8  # m, self-cooled (ONAN)

    def __post_init__(self):
        if self.no_load_loss_w <= 0 or self.load_loss_w <= 0:
            raise ValueError("No-load and load losses must be greater than zero")
        if self.hot_spot_rise_rated <= self.top_oil_rise_rated:
            raise ValueError("Hottest-spot rise must be greater than the top-oil rise")

    @property
    def loss_ratio(self) -> float:
        """R: load loss at rated load / no-load loss"""
        return self.load_loss_w / self.no_load_loss_w

    @property
    def hot_spot_gradient_rated(self) -> float:
        """Hottest-spot rise over top oil at rated load (°C)"""
        return self.hot_spot_rise_rated - self.top_oil_rise_rated

    @property
    def percent_resistance(self) -> float:
        return self.load_loss_w / (self.kva * 1000) * 100

    @property
    def percent_reactance(self) -> float:
        return math.sqrt(max(self.impedance_pct ** 2 - self.percent_resistance ** 2, 0.0))

    def _current(self, kva, volts):
        phase_factor = math.sqrt(3) if self.phases == 'Three-phase' else 1.0
        return kva * 1000 / (phase_factor * volts)

    def lv_current(self, kva):
        """Low-voltage current (A) for a load in kVA"""
        return self._current(kva, self.lv.current_basis_v)

    def hv_current(self, kva):
        """High-voltage current (A) for a load in kVA"""
        return self._current(kva, self.hv.current_basis_v)


def _first_order_response(ultimate: np.ndarray, tau_min: float, dt_min: float) -> np.ndarray:
    """Exponential response to a value held over each time step, starting in steady state

    x[k] = u[k] + (x[k-1] - u[k]) * exp(-dt / tau), applied along the last axis.
    """
    decay = math.exp(-dt_min / tau_min)
    response = np.empty_like(ultimate)
    state = ultimate[..., 0]
    for k in range(ultimate.shape[-1]):
        state = ultimate[..., k] + (state - ultimate[..., k]) * decay
        response[..., k] = state
    return response


def simulate_temperatures(xfmr: DistributionTransformer, load_pu: np.ndarray,
                          ambient_c: np.ndarray, dt_min: float) -> Dict[str, np.ndarray]:
    """Top-oil and hottest-spot temperatures for a per-unit load profile

    Ultimate top-oil rise   = rated rise * ((K^2 R + 1) / (R + 1))^n
    Ultimate hot-spot rise over top oil = rated gradient * K^(2m)
    Each rise approaches its ultimate value exponentially with the oil or winding
    time constant, and hottest spot = ambient + top-oil rise + hot-spot gradient.
    load_pu may be 1-D (time) or 2-D (scenarios x time).
    """
    load_pu = np.asarray(load_pu, dtype=float)
    ambient_c = np.broadcast_to(np.asarray(ambient_c, dtype=float), load_pu.shape)
    R = xfmr.loss_ratio

    top_oil_rise_u = xfmr.top_oil_rise_rated * ((load_pu ** 2 * R + 1) / (R + 1)) ** xfmr.oil_exponent
    gradient_u = xfmr.hot_spot_gradient_rated * load_pu ** (2 * xfmr.winding_exponent)

    top_oil_rise = _first_order_response(top_oil_rise_u, xfmr.oil_time_constant_min, dt_min)
    gradient = _first_order_response(gradient_u, xfmr.winding_time_constant_min, dt_min)

    top_oil = ambient_c + top_oil_rise
    return {'top_oil': top_oil, 'hot_spot': top_oil + gradient}


def aging_acceleration_factor(hot_spot_c):
    """F_AA = exp(15000/383 - 15000/(hot spot + 273)); 1.0 at 110 °C"""
    return np.exp(AGING_B / (AGING_REFERENCE_HOT_SPOT_C + 273) - AGING_B / (np.asarray(hot_spot_c) + 273))


def loss_of_life(hot_spot_c: np.ndarray, dt_min: float, normal_life_hours: float) -> Dict:
    """Equivalent aging and insulation loss of life over the profile (per row if 2-D)"""
    faa = aging_acceleration_factor(hot_spot_c)
    feqa = faa.mean(axis=-1)  # Equal time steps
    hours = hot_spot_c.shape[-1] * dt_min / 60
    lol_hours = feqa * hours
    lol_pct = lol_hours / normal_life_hours * 100
    return {
        'faa': faa,
        'feqa': feqa,
        'hours': hours,
        'lol_hours': lol_hours,
        'lol_pct': lol_pct,
        'lol_pct_per_day': lol_pct / (hours / 24),
        'life_years': normal_life_hours / (feqa * 8760)
    }


def voltage_regulation_pct(xfmr: DistributionTransformer, load_pu, power_factor):
    """Approximate regulation (%) from %R (load loss) and %X (from %Z), lagging power factor"""
    r, x = xfmr.percent_resistance, xfmr.percent_reactance
    pf = np.asarray(power_factor)
    sin_phi = np.sqrt(np.clip(1 - pf ** 2, 0, None))
    return load_pu * (r * pf + x * sin_phi) + (load_pu * (x * pf - r * sin_phi)) ** 2 / 200


# =============================================================================
# Load, EV charging and ambient profiles
# =============================================================================

# Illustrative summer residential shape (fraction of daily peak at the start of each
# hour 0-23) - replace with interval (AMI) data for the transformer where available
RESIDENTIAL_SHAPE = np.array([
    0.45, 0.40, 0.37, 0.35, 0.35, 0.38, 0.48, 0.58, 0.60, 0.57, 0.55, 0.55,
    0.57, 0.60, 0.65, 0.72, 0.82, 0.92, 1.00, 0.98, 0.93, 0.83, 0.68, 0.55
])


def steps_per_day(dt_min: float) -> int:
    return int(round(24 * 60 / dt_min))


def synthetic_base_load_kw(days: int, dt_min: float, peak_kw: float,
                           rng: np.random.Generator, variability: float = 0.05) -> np.ndarray:
    """Residential load built from RESIDENTIAL_SHAPE with day-to-day and interval
    variability, scaled so the highest interval equals peak_kw"""
    spd = steps_per_day(dt_min)
    hours = np.arange(days * spd) * dt_min / 60
    shape = np.interp(hours % 24, np.arange(25), np.append(RESIDENTIAL_SHAPE, RESIDENTIAL_SHAPE[0]))
    daily_scale = np.repeat(rng.normal(1.0, variability, days), spd)
    noise = rng.normal(0.0, variability, days * spd)
    load = np.clip(shape * daily_scale * (1 + noise), 0, None)
    return peak_kw * load / load.max()


def base_load_from_csv(df: pd.DataFrame, dt_min: float) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Interval data (columns 'timestamp' and 'kW') resampled to whole days at dt_min"""
    columns = {c.lower().strip(): c for c in df.columns}
    if 'timestamp' not in columns or 'kw' not in columns:
        raise ValueError("The CSV needs a 'timestamp' column and a 'kW' column")

    series = pd.Series(
        pd.to_numeric(df[columns['kw']], errors='coerce').values,
        index=pd.to_datetime(df[columns['timestamp']], errors='coerce')
    )
    series = series[series.index.notna()].sort_index()
    series = series.resample(f'{int(dt_min)}min').mean().interpolate(limit_direction='both')

    spd = steps_per_day(dt_min)
    full_days = len(series) // spd
    if full_days < 1 or series.isna().all():
        raise ValueError("The CSV must contain at least one full day of numeric kW readings")
    series = series.iloc[:full_days * spd]
    return np.clip(series.values, 0, None), series.index


@dataclass
class EVCharging:
    """Added EV charging load (planning assumptions)"""
    count: int = 3
    charger_kw: float = 7.2
    daily_energy_kwh: float = 10.0  # Average energy per EV per day
    arrival_mean_h: float = 18.0  # Hour the EV is plugged in
    arrival_sd_h: float = 1.5
    delayed: bool = False  # True: hold charging until delayed_start_h (off-peak)
    delayed_start_h: float = 23.0  # Hours before 12 mean after midnight
    start_window_h: float = 2.0  # Delayed starts are spread randomly over this window
    power_factor: float = 1.0


def ev_charging_kw(ev: EVCharging, num_evs: int, days: int, dt_min: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Charging power (kW) per EV (rows) per time step; each EV charges once a day at
    full charger power until its energy is delivered"""
    spd = steps_per_day(dt_min)
    total_steps = days * spd
    power = np.zeros((num_evs, total_steps + spd))  # Extra day absorbs sessions past the end

    arrival_h = rng.normal(ev.arrival_mean_h, ev.arrival_sd_h, (num_evs, days))
    energy_kwh = np.clip(rng.normal(ev.daily_energy_kwh, 0.25 * ev.daily_energy_kwh,
                                    (num_evs, days)), 0.5, None)
    start_offset_h = rng.uniform(0, ev.start_window_h, (num_evs, days))

    if ev.delayed:
        delayed_start = ev.delayed_start_h + (24 if ev.delayed_start_h < 12 else 0)
        start_h = np.maximum(delayed_start + start_offset_h, arrival_h)
    else:
        start_h = arrival_h

    steps_per_hour = 60 / dt_min
    start = np.clip((np.arange(days) * 24 + start_h) * steps_per_hour, 0, None)
    end = start + energy_kwh / ev.charger_kw * steps_per_hour

    for i in range(num_evs):
        for d in range(days):
            s, e = start[i, d], end[i, d]
            for k in range(int(s), min(int(math.ceil(e)), power.shape[1])):
                power[i, k] += ev.charger_kw * (min(e, k + 1) - max(s, k))

    return power[:, :total_steps]


def ambient_profile(num_steps: int, dt_min: float, average_c: float,
                    swing_c: float) -> np.ndarray:
    """Daily sinusoid, coolest at 03:00 and warmest at 15:00 (swing is peak-to-peak)"""
    hours = np.arange(num_steps) * dt_min / 60
    return average_c + swing_c / 2 * np.sin(2 * np.pi * (hours - 9) / 24)


# =============================================================================
# Analysis
# =============================================================================

MAX_SCAN_ELEMENTS = 1_000_000  # Caps memory for the EV hosting-capacity scan


def _power_factor_q(p_kw, power_factor):
    return p_kw * math.tan(math.acos(power_factor))


def run_analysis(xfmr: DistributionTransformer, base_kw: np.ndarray, base_pf: float,
                 ev: EVCharging, ambient_c: np.ndarray, dt_min: float,
                 normal_life_hours: float, hot_spot_limit_c: float,
                 top_oil_limit_c: float, seed: int) -> Dict:
    """Thermal and loss-of-life results with and without the added EVs

    base_kw and ambient_c include a leading warm-up day that settles the thermal
    state and carries overnight charging into day 1; it is excluded from all results.
    Every EV count from 0 upward is simulated with the same random charging sessions
    to find the EV hosting capacity.
    """
    spd = steps_per_day(dt_min)
    total_steps = len(base_kw)
    days = total_steps // spd

    scan_max = max(10, math.ceil(3 * xfmr.kva / ev.charger_kw), 2 * ev.count)
    scan_max = min(scan_max, 200, MAX_SCAN_ELEMENTS // total_steps - 1)
    scan_max = max(scan_max, ev.count)

    ev_each_kw = ev_charging_kw(ev, scan_max, days, dt_min, np.random.default_rng([seed, 1]))
    ev_kw = np.vstack([np.zeros(total_steps), np.cumsum(ev_each_kw, axis=0)])  # Row n = n EVs

    p_kw = base_kw + ev_kw
    q_kvar = _power_factor_q(base_kw, base_pf) + _power_factor_q(ev_kw, ev.power_factor)
    load_kva = np.hypot(p_kw, q_kvar)
    load_pu = load_kva / xfmr.kva
    temps = simulate_temperatures(xfmr, load_pu, ambient_c, dt_min)

    # Drop the warm-up day
    window = slice(spd, None)
    hot_spot = temps['hot_spot'][:, window]
    top_oil = temps['top_oil'][:, window]
    aging = loss_of_life(hot_spot, dt_min, normal_life_hours)

    scan = pd.DataFrame({
        'ev_count': np.arange(scan_max + 1),
        'feqa': aging['feqa'],
        'lol_pct_per_day': aging['lol_pct_per_day'],
        'peak_hot_spot_c': hot_spot.max(axis=1),
        'peak_top_oil_c': top_oil.max(axis=1),
        'peak_load_pct': load_pu[:, window].max(axis=1) * 100
    })
    scan['meets_criteria'] = ((scan['feqa'] <= 1.0)
                              & (scan['peak_hot_spot_c'] <= hot_spot_limit_c)
                              & (scan['peak_top_oil_c'] <= top_oil_limit_c))
    failing = scan.index[~scan['meets_criteria']]
    hosting_capacity = int(failing[0]) - 1 if len(failing) else scan_max

    def scenario(n):
        pf = np.divide(p_kw[n], load_kva[n], out=np.ones(total_steps), where=load_kva[n] > 0)
        loss_kw = (xfmr.no_load_loss_w + load_pu[n] ** 2 * xfmr.load_loss_w) / 1000
        return {
            'ev_count': n,
            'ev_kw': ev_kw[n, window],
            'load_kva': load_kva[n, window],
            'load_pu': load_pu[n, window],
            'power_factor': pf[window],
            'top_oil': top_oil[n],
            'hot_spot': hot_spot[n],
            'faa': aging['faa'][n],
            'feqa': aging['feqa'][n],
            'lol_hours': aging['lol_hours'][n],
            'lol_pct': aging['lol_pct'][n],
            'lol_pct_per_day': aging['lol_pct_per_day'][n],
            'life_years': aging['life_years'][n],
            'loss_kwh_per_day': loss_kw[window].mean() * 24,
            'voltage_regulation_pct': voltage_regulation_pct(xfmr, load_pu[n, window], pf[window])
        }

    return {
        'dt_min': dt_min,
        'days': days - 1,
        'hours': np.arange(total_steps - spd) * dt_min / 60,
        'base_kw': base_kw[window],
        'ambient': np.asarray(ambient_c)[window],
        'without_ev': scenario(0),
        'with_ev': scenario(ev.count),
        'scan': scan,
        'hosting_capacity': hosting_capacity,
        'scan_max': scan_max
    }


def daily_summary(results: Dict, key: str, day_labels: List[str]) -> pd.DataFrame:
    """Per-day peaks and loss of life for one scenario"""
    s = results[key]
    spd = steps_per_day(results['dt_min'])
    day = np.arange(len(s['hot_spot'])) // spd
    frame = pd.DataFrame({
        'day': day,
        'load_kva': s['load_kva'],
        'hot_spot': s['hot_spot'],
        'top_oil': s['top_oil'],
        'faa': s['faa']
    })
    grouped = frame.groupby('day').agg(**{
        'Peak load (kVA)': ('load_kva', 'max'),
        'Peak hottest spot (°C)': ('hot_spot', 'max'),
        'Peak top oil (°C)': ('top_oil', 'max'),
        'F_EQA': ('faa', 'mean')
    })
    grouped['Insulation life used (h)'] = grouped['F_EQA'] * 24
    grouped.insert(0, 'Day', day_labels[:len(grouped)])
    return grouped.reset_index(drop=True)


# =============================================================================
# Charts
# =============================================================================

# Categorical slots (validated palette): scenario identity is the same in every chart
COLOR_WITHOUT_EV = '#2a78d6'  # Slot 1 blue - existing load only
COLOR_WITH_EV = '#eb6834'  # Slot 2 orange - with added EVs
COLOR_TOP_OIL = '#1baf7a'  # Slot 3 aqua
COLOR_CONTEXT = '#898781'  # Muted ink: ambient, reference lines
WASH_WITHOUT_EV = 'rgba(42, 120, 214, 0.10)'
WASH_WITH_EV = 'rgba(235, 104, 52, 0.10)'


def _style(fig: go.Figure, y_title: str, height: int = 340) -> go.Figure:
    """Shared layout for time-series charts (x = hours, ticks at day boundaries)"""
    hours = fig.data[0].x
    days = max(1, int(round((hours[-1] - hours[0]) / 24)))
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,
                    traceorder='normal')
    )
    fig.update_yaxes(title_text=y_title)
    fig.update_xaxes(title_text="Hours from start of analysis", tick0=0,
                     dtick=24 * math.ceil(days / 10))
    return fig


def _reference_line(fig, y, text, position='top left', **kwargs):
    fig.add_hline(y=y, line_dash='dash', line_width=1, line_color=COLOR_CONTEXT,
                  annotation_text=text, annotation_position=position,
                  annotation_font_color='#52514e', **kwargs)


def load_chart(results: Dict, xfmr: DistributionTransformer, ev_label: str) -> go.Figure:
    hours = results['hours']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours, y=results['without_ev']['load_kva'], name='Existing load',
        line=dict(color=COLOR_WITHOUT_EV, width=2), fill='tozeroy', fillcolor=WASH_WITHOUT_EV,
        hovertemplate='%{y:.1f} kVA'
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=results['with_ev']['load_kva'], name=f'With {ev_label}',
        line=dict(color=COLOR_WITH_EV, width=2), fill='tonexty', fillcolor=WASH_WITH_EV,
        hovertemplate='%{y:.1f} kVA'
    ))
    _reference_line(fig, xfmr.kva, f"Nameplate {xfmr.kva:g} kVA")
    return _style(fig, "Load (kVA)")


def temperature_chart(results: Dict, ev_label: str, hot_spot_limit_c: float) -> go.Figure:
    hours = results['hours']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours, y=results['without_ev']['hot_spot'], name='Hottest spot - existing load',
        line=dict(color=COLOR_WITHOUT_EV, width=2), hovertemplate='%{y:.1f} °C'
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=results['with_ev']['hot_spot'], name=f'Hottest spot - with {ev_label}',
        line=dict(color=COLOR_WITH_EV, width=2), hovertemplate='%{y:.1f} °C'
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=results['with_ev']['top_oil'], name=f'Top oil - with {ev_label}',
        line=dict(color=COLOR_TOP_OIL, width=2), hovertemplate='%{y:.1f} °C'
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=results['ambient'], name='Ambient',
        line=dict(color=COLOR_CONTEXT, width=1), hovertemplate='%{y:.1f} °C'
    ))
    _reference_line(fig, AGING_REFERENCE_HOT_SPOT_C, "110 °C: normal aging rate (F_AA = 1)",
                    position='bottom left')
    _reference_line(fig, hot_spot_limit_c, f"Hottest-spot limit {hot_spot_limit_c:g} °C")
    return _style(fig, "Temperature (°C)", height=380)


def aging_chart(results: Dict, ev_label: str) -> go.Figure:
    hours = results['hours']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours, y=results['without_ev']['faa'], name='Existing load',
        line=dict(color=COLOR_WITHOUT_EV, width=2), hovertemplate='%{y:.3g}×'
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=results['with_ev']['faa'], name=f'With {ev_label}',
        line=dict(color=COLOR_WITH_EV, width=2), hovertemplate='%{y:.3g}×'
    ))
    _reference_line(fig, 1.0, "F_AA = 1 (110 °C)")
    fig.update_yaxes(type='log')
    return _style(fig, "Aging acceleration factor F_AA (log)")


def cumulative_lol_chart(results: Dict, ev_label: str) -> go.Figure:
    hours = results['hours']
    dt_h = results['dt_min'] / 60
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours, y=np.cumsum(results['without_ev']['faa']) * dt_h, name='Existing load',
        line=dict(color=COLOR_WITHOUT_EV, width=2), hovertemplate='%{y:.2f} h'
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=np.cumsum(results['with_ev']['faa']) * dt_h, name=f'With {ev_label}',
        line=dict(color=COLOR_WITH_EV, width=2), hovertemplate='%{y:.2f} h'
    ))
    return _style(fig, "Cumulative insulation life used (h)")


def hosting_capacity_chart(results: Dict, ev_count: int, hot_spot_limit_c: float) -> go.Figure:
    scan = results['scan']
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.1, subplot_titles=(
        "F_EQA (log scale)", "Peak hottest spot, °C", "Peak load, % of rating"))

    for col, column, fmt in ((1, 'feqa', '%{y:.2f}'), (2, 'peak_hot_spot_c', '%{y:.1f} °C'),
                             (3, 'peak_load_pct', '%{y:.0f}%')):
        fig.add_trace(go.Scatter(
            x=scan['ev_count'], y=scan[column], mode='lines', showlegend=False,
            line=dict(color=COLOR_WITHOUT_EV, width=2),
            hovertemplate=f'%{{x}} EVs: {fmt}<extra></extra>'
        ), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=[ev_count], y=[scan.loc[ev_count, column]], mode='markers', showlegend=False,
            marker=dict(color=COLOR_WITH_EV, size=10, line=dict(color='#fcfcfb', width=2)),
            hovertemplate=f'Configured ({ev_count} EVs): {fmt}<extra></extra>'
        ), row=1, col=col)

    for col, y, text in ((1, 1.0, "Normal aging"), (2, hot_spot_limit_c, "Limit"),
                         (3, 100.0, "Nameplate")):
        fig.add_hline(y=y, line_dash='dash', line_width=1, line_color=COLOR_CONTEXT,
                      annotation_text=text, annotation_position='top left',
                      annotation_font_color='#52514e', row=1, col=col)

    fig.update_yaxes(type='log', row=1, col=1)
    fig.update_xaxes(title_text="Added EVs")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10), hovermode='closest')
    return fig


# =============================================================================
# Streamlit UI
# =============================================================================

DEFAULT_KVA = {'Single-phase': 50, 'Three-phase': 75}
DEFAULT_HV = {'Single-phase': '12470GrdY/7200', 'Three-phase': '12470Y/7200'}
DEFAULT_LV = {'Single-phase': '120/240', 'Three-phase': '208Y/120'}
CHARGING_STRATEGIES = ("Uncontrolled (charge on arrival)", "Delayed (off-peak start)")


def sidebar_inputs() -> Dict:
    """Collect the configuration from the sidebar"""
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("Transformer (IEEE C57.12.20)")
        phases = st.radio("Phases", PHASES, horizontal=True)
        ratings = KVA_RATINGS[phases]
        kva = st.selectbox("Rating (Table 1)", ratings, index=ratings.index(DEFAULT_KVA[phases]),
                           format_func=lambda v: f"{v:g} kVA")

        hv_options = HV_RATINGS[phases]
        hv_labels = [h.label for h in hv_options]
        hv = hv_options[hv_labels.index(st.selectbox(
            "High-voltage rating (Tables 2–4)", hv_labels,
            index=hv_labels.index(DEFAULT_HV[phases])
        ))]
        lv_options = LV_RATINGS[phases]
        lv_labels = [v.label for v in lv_options]
        lv = lv_options[lv_labels.index(st.selectbox(
            "Low-voltage rating (V)", lv_labels, index=lv_labels.index(DEFAULT_LV[phases])
        ))]

        min_z = table12_min_impedance(phases, kva, lv)
        impedance_pct = st.number_input(
            "Impedance (%Z)", value=(min_z + 0.5) if min_z else 2.5,
            min_value=0.5, max_value=10.0, step=0.1,
            help="Nameplate impedance. Table 12 minimum for this rating: "
                 + (f"{min_z:g}%" if min_z else "none (LV above 600 V)")
        )

        with st.expander("Losses & thermal data (test report)"):
            no_load_loss_w = st.number_input(
                "No-load loss (W)", value=float(round(kva * 2.0)), min_value=1.0, step=5.0,
                help="Illustrative default (0.2% of rating) - use the certified test report"
            )
            load_loss_w = st.number_input(
                "Load loss at rated kVA (W)", value=float(round(kva * 10.0)), min_value=1.0,
                step=10.0, help="Illustrative default (1.0% of rating) - use the certified test report"
            )
            top_oil_rise = st.number_input(
                "Top-oil rise at rated load (°C)", value=55.0, min_value=20.0, max_value=80.0,
                step=1.0, help=f"C57.12.20 Clause 4.1 limit: {MAX_TOP_OIL_RISE_C:g} °C. "
                               "55 °C is a typical value - use the heat-run test"
            )
            hot_spot_rise = st.number_input(
                "Hottest-spot rise at rated load (°C)", value=RATED_HOT_SPOT_RISE_C,
                min_value=30.0, max_value=100.0, step=1.0,
                help="Basis of the kVA rating per C57.12.20 Clause 4.1: 80 °C"
            )
            tau_oil = st.number_input(
                "Top-oil time constant (min)", value=180.0, min_value=10.0, max_value=600.0,
                step=10.0, help="Typical for ONAN distribution transformers - use manufacturer data"
            )
            tau_winding = st.number_input(
                "Winding time constant (min)", value=4.0, min_value=1.0, max_value=30.0,
                step=1.0, help="Typical for ONAN distribution transformers - use manufacturer data"
            )

        st.subheader("Service conditions")
        ambient_avg = st.number_input(
            "Average ambient (°C)", value=USUAL_AMBIENT_AVG_C, min_value=-30.0, max_value=50.0,
            step=1.0, help="The kVA rating assumes the usual service conditions of C57.12.00: "
                           "24-hour average not above 30 °C and maximum not above 40 °C"
        )
        ambient_swing = st.number_input(
            "Daily ambient swing, peak-to-peak (°C)", value=10.0, min_value=0.0,
            max_value=30.0, step=1.0
        )

        st.subheader("Existing load")
        load_source = st.radio("Source", ("Synthetic residential profile",
                                          "Upload interval data (CSV)"))
        uploaded = None
        base_peak_kw = None
        if load_source == "Synthetic residential profile":
            base_peak_kw = st.number_input(
                "Peak existing load (kW)", value=float(round(0.8 * kva, 1)), min_value=0.1,
                max_value=5.0 * kva, step=1.0,
                help="Highest interval of the illustrative residential profile"
            )
        else:
            uploaded = st.file_uploader(
                "Interval data CSV", type=['csv'],
                help="Columns 'timestamp' and 'kW' - existing load on this transformer"
            )
        base_pf = st.number_input("Existing load power factor", value=0.95, min_value=0.5,
                                  max_value=1.0, step=0.01)

        st.subheader("EV charging (added load)")
        ev_count = st.number_input("Number of EVs", value=3, min_value=0, max_value=200, step=1)
        charger_kw = st.number_input("Charger power (kW)", value=7.2, min_value=1.0,
                                     max_value=50.0, step=0.1,
                                     help="e.g. 1.4 kW Level 1, 7.2-11.5 kW Level 2")
        daily_energy = st.number_input(
            "Energy per EV per day (kWh)", value=10.0, min_value=1.0, max_value=100.0,
            step=1.0, help="Planning assumption - average daily charging energy per EV"
        )
        arrival_mean = st.slider("Typical plug-in time (hour)", 0.0, 23.5, 18.0, 0.5)
        arrival_sd = st.slider("Plug-in time spread, std. dev. (h)", 0.0, 4.0, 1.5, 0.25)
        strategy = st.radio("Charging strategy", CHARGING_STRATEGIES)
        delayed_start, start_window = 23.0, 2.0
        if strategy == CHARGING_STRATEGIES[1]:
            delayed_start = st.slider("Off-peak start (hour)", 0.0, 23.5, 23.0, 0.5,
                                      help="Hours before 12:00 are treated as after midnight")
            start_window = st.slider("Randomized start window (h)", 0.0, 6.0, 2.0, 0.5,
                                     help="Spreads start times so chargers don't all switch on together")
        ev_pf = st.number_input("EV charger power factor", value=1.0, min_value=0.5,
                                max_value=1.0, step=0.01)

        st.subheader("Analysis")
        days = st.selectbox("Days to simulate", [1, 7, 14, 30], index=1,
                            help="Synthetic profile only; uploaded data sets its own length")
        dt_min = st.selectbox("Time step (min)", [5, 15, 30, 60], index=1)
        normal_life_label = st.selectbox("Normal insulation life (IEEE C57.91)",
                                         list(NORMAL_LIFE_HOURS))
        hot_spot_limit = st.number_input(
            "Hottest-spot screening limit (°C)", value=120.0, min_value=90.0, max_value=200.0,
            step=5.0, help="Placeholder - set from your utility's loading guide"
        )
        top_oil_limit = st.number_input(
            "Top-oil screening limit (°C)", value=105.0, min_value=70.0, max_value=130.0,
            step=5.0, help="Placeholder - set from your utility's loading guide"
        )
        seed = st.number_input("Random seed", value=1, min_value=0, step=1,
                               help="Change to see another random realization of load and charging")

    return {
        'phases': phases, 'kva': kva, 'hv': hv, 'lv': lv, 'impedance_pct': impedance_pct,
        'no_load_loss_w': no_load_loss_w, 'load_loss_w': load_loss_w,
        'top_oil_rise': top_oil_rise, 'hot_spot_rise': hot_spot_rise,
        'tau_oil': tau_oil, 'tau_winding': tau_winding,
        'ambient_avg': ambient_avg, 'ambient_swing': ambient_swing,
        'load_source': load_source, 'uploaded': uploaded, 'base_peak_kw': base_peak_kw,
        'base_pf': base_pf,
        'ev': EVCharging(
            count=int(ev_count), charger_kw=charger_kw, daily_energy_kwh=daily_energy,
            arrival_mean_h=arrival_mean, arrival_sd_h=arrival_sd,
            delayed=strategy == CHARGING_STRATEGIES[1], delayed_start_h=delayed_start,
            start_window_h=start_window, power_factor=ev_pf
        ),
        'days': days, 'dt_min': dt_min,
        'normal_life_hours': NORMAL_LIFE_HOURS[normal_life_label],
        'hot_spot_limit': hot_spot_limit, 'top_oil_limit': top_oil_limit, 'seed': int(seed)
    }


def standards_checks(xfmr: DistributionTransformer, cfg: Dict) -> pd.DataFrame:
    """C57.12.20 checks for the configured transformer"""
    rows = []

    def add(item, reference, requirement, value, ok):
        status = '✅ Meets' if ok is True else ('⚠️ Check' if ok is False else 'ℹ️ Info')
        rows.append({'Check': item, 'Reference': reference, 'Requirement': requirement,
                     'This transformer': value, 'Status': status})

    add("kVA rating", "Table 1", f"Standard {xfmr.phases.lower()} rating",
        f"{xfmr.kva:g} kVA", True)

    min_kva = minimum_kva(xfmr.hv, xfmr.lv)
    lv_group = LV_COLUMNS[xfmr.phases][xfmr.lv.column]
    if min_kva is None:
        add("Voltage combination", xfmr.hv.table, f"{xfmr.hv.label} with LV {lv_group}",
            "Not listed", False)
    else:
        add("Voltage combination", xfmr.hv.table,
            f"≥ {min_kva:g} kVA for {xfmr.hv.label} with LV {lv_group}",
            f"{xfmr.kva:g} kVA", xfmr.kva >= min_kva)
    add("Basic lightning impulse insulation level", xfmr.hv.table, "Listed BIL",
        f"{xfmr.hv.bil_kv} kV", None)

    min_z = table12_min_impedance(xfmr.phases, xfmr.kva, xfmr.lv)
    if min_z is None:
        add("Minimum impedance", "Table 12", "Applies to LV 600 V and below",
            f"{xfmr.impedance_pct:g}%", None)
    else:
        add("Minimum impedance", "Table 12", f"≥ {min_z:g}%", f"{xfmr.impedance_pct:g}%",
            xfmr.impedance_pct >= min_z)

    add("Top-liquid rise at rated load", "Clause 4.1", f"≤ {MAX_TOP_OIL_RISE_C:g} °C",
        f"{xfmr.top_oil_rise_rated:g} °C", xfmr.top_oil_rise_rated <= MAX_TOP_OIL_RISE_C)
    add("Hottest-spot rise at rated load", "Clause 4.1", f"≤ {RATED_HOT_SPOT_RISE_C:g} °C",
        f"{xfmr.hot_spot_rise_rated:g} °C", xfmr.hot_spot_rise_rated <= RATED_HOT_SPOT_RISE_C)

    ambient_max = cfg['ambient_avg'] + cfg['ambient_swing'] / 2
    add("Ambient vs rating basis", "Clause 4.1 → C57.12.00 usual service",
        f"Average ≤ {USUAL_AMBIENT_AVG_C:g} °C, maximum ≤ {USUAL_AMBIENT_MAX_C:g} °C",
        f"Average {cfg['ambient_avg']:g} °C, maximum {ambient_max:g} °C",
        cfg['ambient_avg'] <= USUAL_AMBIENT_AVG_C and ambient_max <= USUAL_AMBIENT_MAX_C)
    return pd.DataFrame(rows)


def _format_life(years: float) -> str:
    return "> 200 yr" if years > 200 else f"{years:.1f} yr"


def render_kpis(results: Dict, xfmr: DistributionTransformer, cfg: Dict):
    w, wo = results['with_ev'], results['without_ev']
    ev_count = cfg['ev'].count

    cols = st.columns(5)
    cols[0].metric(
        "Peak load (% of nameplate)", f"{w['load_pu'].max() * 100:.0f}%",
        f"{(w['load_pu'].max() - wo['load_pu'].max()) * 100:+.0f} pts vs no EVs",
        delta_color='inverse'
    )
    cols[1].metric(
        "Peak hottest spot", f"{w['hot_spot'].max():.0f} °C",
        f"{w['hot_spot'].max() - wo['hot_spot'].max():+.1f} °C vs no EVs", delta_color='inverse'
    )
    cols[2].metric(
        "Loss of life per day", f"{w['lol_pct_per_day']:.4f}%",
        f"{w['lol_pct_per_day'] / wo['lol_pct_per_day']:.1f}× the no-EV rate",
        delta_color='inverse',
        help=f"Normal aging (F_EQA = 1) uses {24 / cfg['normal_life_hours'] * 100:.4f}% per day"
    )
    cols[3].metric(
        "Equivalent aging F_EQA", f"{w['feqa']:.2f}",
        f"{w['feqa'] - wo['feqa']:+.2f} vs no EVs", delta_color='inverse',
        help="Average aging acceleration factor; 1.0 = normal aging (110 °C hottest spot)"
    )
    life_delta = None
    if w['life_years'] <= 200 and wo['life_years'] <= 200:
        life_delta = f"{w['life_years'] - wo['life_years']:+.1f} yr vs no EVs"
    cols[4].metric(
        "Projected insulation life", _format_life(w['life_years']), life_delta,
        help="Normal insulation life ÷ F_EQA, if the simulated conditions lasted all year"
    )

    cols = st.columns(5)
    hosting = results['hosting_capacity']
    hosting_text = (f"≥ {hosting} EVs" if hosting >= results['scan_max']
                    else ("None" if hosting < 0 else f"{hosting} EVs"))
    cols[0].metric(
        "EV hosting capacity", hosting_text,
        help="Most added EVs (same charging assumptions) with F_EQA ≤ 1 and peak hottest-spot "
             "and top-oil temperatures within the screening limits"
    )
    peak_kva = w['load_kva'].max()
    rated_lv = xfmr.lv_current(xfmr.kva)
    cols[1].metric(
        "Peak LV current", f"{xfmr.lv_current(peak_kva):,.0f} A",
        help=f"Rated LV current {rated_lv:,.0f} A at {xfmr.lv.current_basis_v:g} V; "
             f"rated HV current {xfmr.hv_current(xfmr.kva):,.2f} A"
    )
    cols[2].metric(
        "Voltage regulation at peak", f"{w['voltage_regulation_pct'].max():.1f}%",
        help=f"Approximation from %Z = {xfmr.impedance_pct:g}% and %R = "
             f"{xfmr.percent_resistance:.2f}% (load loss); transformer only"
    )
    cols[3].metric(
        "Transformer losses per day", f"{w['loss_kwh_per_day']:.1f} kWh",
        f"{w['loss_kwh_per_day'] - wo['loss_kwh_per_day']:+.1f} vs no EVs", delta_color='inverse'
    )
    hours_over = (w['load_pu'] > 1).sum() * results['dt_min'] / 60 / results['days']
    cols[4].metric("Time above nameplate per day", f"{hours_over:.1f} h",
                   help=f"With {ev_count} EVs, averaged over the analysis window")


def time_series_frame(results: Dict) -> pd.DataFrame:
    w, wo = results['with_ev'], results['without_ev']
    dt_h = results['dt_min'] / 60
    return pd.DataFrame({
        'hour': results['hours'],
        'ambient_c': results['ambient'],
        'existing_load_kw': results['base_kw'],
        'ev_load_kw': w['ev_kw'],
        'load_kva_without_ev': wo['load_kva'],
        'load_kva_with_ev': w['load_kva'],
        'load_pu_with_ev': w['load_pu'],
        'top_oil_c_without_ev': wo['top_oil'],
        'top_oil_c_with_ev': w['top_oil'],
        'hot_spot_c_without_ev': wo['hot_spot'],
        'hot_spot_c_with_ev': w['hot_spot'],
        'faa_without_ev': wo['faa'],
        'faa_with_ev': w['faa'],
        'life_used_h_without_ev': np.cumsum(wo['faa']) * dt_h,
        'life_used_h_with_ev': np.cumsum(w['faa']) * dt_h,
        'voltage_regulation_pct_with_ev': w['voltage_regulation_pct']
    })


def render_method():
    st.markdown(r"""
**Scope and ratings - IEEE Std C57.12.20-2017.** Single- and three-phase, 60 Hz,
liquid-immersed, self-cooled overhead-type distribution transformers of 500 kVA and
smaller. The kVA rating is a continuous rating based on a 65 °C average winding rise
or an 80 °C hottest-spot rise, with a top-liquid rise of no more than 65 °C, under the
usual service conditions of IEEE Std C57.12.00 (Clause 4.1). Ratings, voltage
combinations, BIL and minimum impedance come from Tables 1–4 and 12.

**Temperatures - IEEE C57.91 top-oil / hottest-spot method.** With per-unit load $K$
and loss ratio $R$ (load loss ÷ no-load loss):

$$\Delta\theta_{TO,U} = \Delta\theta_{TO,R}\left[\frac{K^2R+1}{R+1}\right]^n \qquad
\Delta\theta_{H,U} = \Delta\theta_{H,R}\,K^{2m}$$

Each rise moves exponentially toward its ultimate value with the top-oil time
constant $\tau_{TO}$ or winding time constant $\tau_W$, and
$\theta_H = \theta_A + \Delta\theta_{TO} + \Delta\theta_H$. Here
$\Delta\theta_{H,R}$ = hottest-spot rise − top-oil rise, and $n = m = 0.8$ for
self-cooled units. Load is held constant within each time step and the rated time
constants are used throughout (the load-dependent time-constant correction is not applied).

**Insulation aging - IEEE C57.91.**

$$F_{AA} = \exp\left(\frac{15000}{383} - \frac{15000}{\theta_H + 273}\right) \qquad
F_{EQA} = \frac{\sum F_{AA,n}\,\Delta t_n}{\sum \Delta t_n}$$

$$\%\,\text{Loss of life} = \frac{F_{EQA} \times t \times 100}{\text{normal insulation life}}$$

$F_{AA} = 1$ at a 110 °C hottest spot, which is the 80 °C rated rise over a 30 °C
average ambient. The projected life assumes the simulated days repeat all year, so a
summer design day gives a conservative (short) projection.

**Load model.** The existing load is an illustrative residential shape (or your
interval data), and each EV charges once a day at full charger power from a random
plug-in time until its daily energy is delivered. Hosting capacity re-runs every EV
count with the same random sessions. The first simulated day is a warm-up and is
excluded from all results.

**Before relying on the results:** check the C57.91 equations and constants against
your copy of the guide (it isn't in this project), and replace the illustrative
losses, temperature rises, time constants and load assumptions with the
transformer's test report and measured load.
""")


def main():
    st.set_page_config(page_title="Distribution Transformer Loss of Life", page_icon="⚡",
                       layout="wide")
    st.title("⚡ Distribution Transformer Loss-of-Life Analysis")
    st.caption("One overhead-type, liquid-immersed, self-cooled distribution transformer · "
               "Ratings per IEEE Std C57.12.20-2017 · Thermal model and insulation aging "
               "per IEEE Std C57.91")

    cfg = sidebar_inputs()

    try:
        xfmr = DistributionTransformer(
            phases=cfg['phases'], kva=cfg['kva'], hv=cfg['hv'], lv=cfg['lv'],
            impedance_pct=cfg['impedance_pct'], no_load_loss_w=cfg['no_load_loss_w'],
            load_loss_w=cfg['load_loss_w'], top_oil_rise_rated=cfg['top_oil_rise'],
            hot_spot_rise_rated=cfg['hot_spot_rise'], oil_time_constant_min=cfg['tau_oil'],
            winding_time_constant_min=cfg['tau_winding']
        )
    except ValueError as e:
        st.error(f"❌ {e}")
        st.stop()

    dt_min = cfg['dt_min']
    spd = steps_per_day(dt_min)

    # Existing load with a leading warm-up day
    if cfg['load_source'] == "Synthetic residential profile":
        days = cfg['days']
        base_kw = synthetic_base_load_kw(days + 1, dt_min, cfg['base_peak_kw'],
                                         np.random.default_rng([cfg['seed'], 0]))
        day_labels = [f"Day {d + 1}" for d in range(days)]
    else:
        if cfg['uploaded'] is None:
            st.info("⬅️ Upload interval data for the existing load: a CSV with a "
                    "`timestamp` column and a `kW` column (any interval; it is resampled "
                    f"to {dt_min} minutes). Only whole days are used.")
            st.stop()
        try:
            measured_kw, index = base_load_from_csv(pd.read_csv(cfg['uploaded']), dt_min)
        except (ValueError, pd.errors.ParserError) as e:
            st.error(f"❌ Could not use the CSV: {e}")
            st.stop()
        base_kw = np.concatenate([measured_kw[:spd], measured_kw])  # Repeat day 1 as warm-up
        day_labels = [str(d.date()) for d in index[::spd]]

    ambient = ambient_profile(len(base_kw), dt_min, cfg['ambient_avg'], cfg['ambient_swing'])
    results = run_analysis(xfmr, base_kw, cfg['base_pf'], cfg['ev'], ambient, dt_min,
                           cfg['normal_life_hours'], cfg['hot_spot_limit'],
                           cfg['top_oil_limit'], cfg['seed'])

    ev = cfg['ev']
    ev_label = f"{ev.count} EV{'s' if ev.count != 1 else ''}"

    st.markdown(
        f"**{xfmr.kva:g} kVA {xfmr.phases.lower()}**, {xfmr.hv.label} – {xfmr.lv.label} V, "
        f"{xfmr.impedance_pct:g}% Z · existing load plus **{ev_label}** at {ev.charger_kw:g} kW "
        f"({'delayed start' if ev.delayed else 'uncontrolled'}) · {results['days']} day(s), "
        f"{dt_min}-minute steps"
    )

    checks = standards_checks(xfmr, cfg)
    failed = checks['Status'].str.startswith('⚠️').sum()
    with st.expander(f"IEEE C57.12.20-2017 checks — "
                     f"{'all met' if failed == 0 else f'{failed} to check'}",
                     expanded=failed > 0):
        st.dataframe(checks, hide_index=True)

    render_kpis(results, xfmr, cfg)

    tab_load, tab_life, tab_hosting, tab_daily, tab_data, tab_method = st.tabs([
        "🌡️ Load & temperature", "⏳ Loss of life", "🚗 EV hosting capacity",
        "📅 Daily summary", "📋 Data & export", "📐 Method & references"
    ])

    with tab_load:
        st.plotly_chart(load_chart(results, xfmr, ev_label))
        st.plotly_chart(temperature_chart(results, ev_label, cfg['hot_spot_limit']))

    with tab_life:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(aging_chart(results, ev_label))
        with col2:
            st.plotly_chart(cumulative_lol_chart(results, ev_label))
            st.caption(f"At the normal aging rate (F_AA = 1) the insulation would use "
                       f"{results['days'] * 24} h of life over this period.")

        w, wo = results['with_ev'], results['without_ev']
        life_table = pd.DataFrame({
            'Metric': ['Equivalent aging F_EQA', 'Insulation life used (h)',
                       'Loss of life over the period (%)', 'Loss of life per day (%)',
                       'Peak aging acceleration F_AA', 'Projected insulation life'],
            'Existing load': [f"{wo['feqa']:.3f}", f"{wo['lol_hours']:.2f}",
                              f"{wo['lol_pct']:.4f}", f"{wo['lol_pct_per_day']:.4f}",
                              f"{wo['faa'].max():.2f}", _format_life(wo['life_years'])],
            f'With {ev_label}': [f"{w['feqa']:.3f}", f"{w['lol_hours']:.2f}",
                                 f"{w['lol_pct']:.4f}", f"{w['lol_pct_per_day']:.4f}",
                                 f"{w['faa'].max():.2f}", _format_life(w['life_years'])]
        })
        st.dataframe(life_table, hide_index=True)
        st.caption(f"Normal insulation life: {cfg['normal_life_hours']:,.0f} h. "
                   f"Analysis period: {results['days']} day(s) = {results['days'] * 24} h.")

    with tab_hosting:
        st.plotly_chart(hosting_capacity_chart(results, ev.count, cfg['hot_spot_limit']))
        scan = results['scan']
        hosting = results['hosting_capacity']
        if hosting < 0:
            st.warning("The existing load alone does not meet the criteria "
                       "(F_EQA ≤ 1 and temperatures within the screening limits).")
        elif hosting < results['scan_max']:
            first_fail = scan.loc[hosting + 1]
            reasons = []
            if first_fail['feqa'] > 1:
                reasons.append(f"F_EQA reaches {first_fail['feqa']:.2f}")
            if first_fail['peak_hot_spot_c'] > cfg['hot_spot_limit']:
                reasons.append(f"hottest spot reaches {first_fail['peak_hot_spot_c']:.0f} °C")
            if first_fail['peak_top_oil_c'] > cfg['top_oil_limit']:
                reasons.append(f"top oil reaches {first_fail['peak_top_oil_c']:.0f} °C")
            st.info(f"Up to **{hosting} EVs** meet the criteria. At {hosting + 1}, "
                    + " and ".join(reasons) + ".")
        else:
            st.info(f"All {results['scan_max']} EV counts scanned meet the criteria.")
        st.dataframe(pd.DataFrame({
            'Added EVs': scan['ev_count'],
            'F_EQA': scan['feqa'].round(4),
            'Loss of life per day (%)': scan['lol_pct_per_day'].round(5),
            'Peak hottest spot (°C)': scan['peak_hot_spot_c'].round(1),
            'Peak top oil (°C)': scan['peak_top_oil_c'].round(1),
            'Peak load (% of nameplate)': scan['peak_load_pct'].round(0),
            'Meets criteria': scan['meets_criteria']
        }), hide_index=True)

    with tab_daily:
        st.markdown(f"**With {ev_label}**")
        st.dataframe(daily_summary(results, 'with_ev', day_labels).round(3), hide_index=True)
        st.markdown("**Existing load only**")
        st.dataframe(daily_summary(results, 'without_ev', day_labels).round(3), hide_index=True)

    with tab_data:
        frame = time_series_frame(results)
        st.dataframe(frame.round(3), hide_index=True)
        st.download_button("📥 Download time series (CSV)", frame.to_csv(index=False),
                           file_name="distribution_transformer_lol.csv", mime="text/csv")

    with tab_method:
        render_method()


if __name__ == "__main__":
    main()
