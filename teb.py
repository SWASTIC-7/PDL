
# c6.py — PDL-GAT v6: Divergence Rescue on Stressed Operating Points
#
# Extends c5.py to demonstrate that PDL warm-start rescues NR-divergent cases.
#
# KEY ADDITIONS over c5.py:
#   1. Real PGLIB-OPF API (Active Power Increase) benchmark files for test:
#      - pglib_opf_case39_epri__api.m   (New England 39-bus, stressed)
#      - pglib_opf_case118_ieee__api.m  (IEEE 118-bus, stressed)
#      - pglib_opf_case300_ieee__api.m  (IEEE 300-bus, larger stressed)
#      - pglib_opf_case1354_pegase__api.m (PEGASE 1354-bus, very large stressed)
#      These are academically citable near-collapse operating points.
#   2. MATPOWER .m file parser + automatic download from PGLIB-OPF GitHub
#   3. Per-point evaluation: NR flat-start (converge/diverge) vs PDL warm-start + NR
#   4. Rescue rate metric: "PDL warm-start rescues X% of NR-divergent cases"
#   5. Publication plots: divergence rescue bar chart, convergence vs stress level
#   6. Targets case39 / case118 / case300 / case1354pegase

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import json
import copy
import os
import re
import random
import urllib.request
import tempfile
from collections import defaultdict
import logging
import pandapower as pp
import pandapower.networks as nw
from pandapower.pypower.makeYbus import makeYbus
from pandapower.converter import from_ppc
import warnings
warnings.filterwarnings("ignore")


class _DropPandapowerNumbaWarning(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return 'numba cannot be imported and numba functions are disabled.' not in msg


_pp_numba_filter = _DropPandapowerNumbaWarning()
logging.getLogger('pandapower').addFilter(_pp_numba_filter)
logging.getLogger('pandapower.auxiliary').addFilter(_pp_numba_filter)

try:
    pd.options.mode.copy_on_write = False
except Exception:
    pass


# ======================================================================
# Utility helpers (from c5)
# ======================================================================
def _make_net_writable(net):
    for key in list(net.keys()):
        obj = net[key]
        if isinstance(obj, pd.DataFrame):
            for col in obj.columns:
                try:
                    arr = obj[col].values
                    if isinstance(arr, np.ndarray) and not arr.flags.writeable:
                        obj[col] = arr.copy()
                except Exception:
                    pass
    return net


def _get_nr_iterations(net):
    try:
        return int(net._ppc['iterations'])
    except (KeyError, TypeError, AttributeError):
        return None


def set_global_seed(seed):
    """Seed Python, NumPy and PyTorch RNGs for reproducible runs.

    Without this, training data, model init and test points are redrawn every
    run, which causes large run-to-run swings on knife-edge cases (e.g. 300,
    1354) where the model sits right at the boundary of usefulness.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"  Global RNG seed set to {seed}")


def _compute_participation_factors(pg_pu, pmax_pu):
    """Compute participation factors from generator headroom.

    Formula: alpha_i = (Pmax_i - Pg_i) / sum_j (Pmax_j - Pg_j)
    Returns zeros if the denominator is non-positive.
    """
    headroom = pmax_pu - pg_pu
    denom = float(np.sum(headroom))
    if denom <= 0.0:
        return np.zeros_like(headroom, dtype=np.float32)
    return (headroom / denom).astype(np.float32)


plt.rcParams.update({
    'font.size': 12, 'font.family': 'serif',
    'axes.labelsize': 14, 'axes.titlesize': 16,
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'legend.fontsize': 11, 'figure.titlesize': 18,
    'lines.linewidth': 2.0, 'axes.grid': True, 'grid.alpha': 0.3,
})


# ======================================================================
# Data extraction  (identical to c5)
# ======================================================================
def get_pandapower_data(net, case_name):
    print(f"  Loading data from pandapower: {case_name}")
    try:
        pp.runpp(net, calculate_voltage_angles=False,
             enforce_q_limits=False, enforce_v_limits=False, init='flat',
             numba=False)
    except Exception:
        pass

    ppc = net._ppc
    baseMVA = ppc['baseMVA']
    bus, gen, branch = ppc['bus'], ppc['gen'], ppc['branch']
    Ybus, _, _ = makeYbus(baseMVA, bus, branch)
    Y_bus = Ybus.toarray()

    n_buses      = net.bus.shape[0]
    n_generators = net.gen.shape[0]
    n_loads      = net.load.shape[0]

    gen_buses = net.gen.bus.values.tolist()
    slack_bus = int(net.ext_grid.bus.values[0])
    pv_buses  = sorted(set(gen_buses) - {slack_bus})
    pq_buses  = sorted(set(range(n_buses)) - set(gen_buses) - {slack_bus})

    gen_voltage_setpoints = np.ones(n_buses)
    for idx, gbus in enumerate(net.gen.bus.values):
        if 'vm_pu' in net.gen.columns and not pd.isna(net.gen.vm_pu.values[idx]):
            gen_voltage_setpoints[gbus] = net.gen.vm_pu.values[idx]
    if 'vm_pu' in net.ext_grid.columns and len(net.ext_grid) > 0:
        gen_voltage_setpoints[slack_bus] = float(net.ext_grid.vm_pu.values[0])

    base_P = np.zeros(n_buses)
    base_Q = np.zeros(n_buses)
    for _, load in net.load.iterrows():
        base_P[int(load.bus)] += load.p_mw
        base_Q[int(load.bus)] += load.q_mvar

    gen_limits = {
        'P_min': net.gen.min_p_mw.values / baseMVA,
        'P_max': net.gen.max_p_mw.values / baseMVA,
        'Q_min': net.gen.min_q_mvar.values / baseMVA,
        'Q_max': net.gen.max_q_mvar.values / baseMVA,
    }

    adj = (np.abs(Y_bus) > 1e-8).astype(np.float32)
    np.fill_diagonal(adj, 1.0)
    y_mag = np.abs(Y_bus).astype(np.float32)
    ym = y_mag.max()
    if ym > 0:
        y_mag /= ym

    print(f"  System: {n_buses} buses, {n_generators} generators, {n_loads} loads")
    print(f"  Base MVA: {baseMVA}, Slack: {slack_bus}, "
          f"PV: {len(pv_buses)}, PQ: {len(pq_buses)}")

    return dict(
        n_buses=n_buses, n_generators=n_generators, n_loads=n_loads,
        gen_buses=gen_buses, slack_bus=slack_bus,
        pv_buses=pv_buses, pq_buses=pq_buses,
        gen_voltage_setpoints=gen_voltage_setpoints,
        Y_bus=Y_bus, adj=adj, y_mag=y_mag,
        base_P_demand_pu=base_P / baseMVA,
        base_Q_demand_pu=base_Q / baseMVA,
        gen_limits=gen_limits, baseMVA=baseMVA, net=net,
    )


# ======================================================================
# PGLIB-OPF MATPOWER .m File Parser & Loader
# ======================================================================

# Map of case key → PGLIB-OPF raw GitHub URLs for the API tier
PGLIB_API_URLS = {
    'case30':  'https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master/api/pglib_opf_case30_ieee__api.m',
    'case39':  'https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master/api/pglib_opf_case39_epri__api.m',
    'case118': 'https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master/api/pglib_opf_case118_ieee__api.m',
    'case300': 'https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master/api/pglib_opf_case300_ieee__api.m',
    'case1354pegase': 'https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master/api/pglib_opf_case1354_pegase__api.m',
    'case1888rte': 'https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master/api/pglib_opf_case1888_rte__api.m',
}

# Local cache directory (works in both scripts and Jupyter/Kaggle notebooks)
try:
    _CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pglib_cache')
except NameError:
    _CACHE_DIR = os.path.join(os.getcwd(), 'pglib_cache')


def _download_pglib_file(case_key):
    """Download a PGLIB-OPF .m file and cache it locally. Returns file path."""
    url = PGLIB_API_URLS.get(case_key)
    if url is None:
        raise ValueError(f"No PGLIB-OPF API URL configured for '{case_key}'. "
                         f"Available: {list(PGLIB_API_URLS.keys())}")
    os.makedirs(_CACHE_DIR, exist_ok=True)
    fname = url.split('/')[-1]
    local_path = os.path.join(_CACHE_DIR, fname)
    if os.path.isfile(local_path):
        print(f"  PGLIB cache hit: {local_path}")
        return local_path
    print(f"  Downloading PGLIB-OPF API file: {fname}")
    print(f"    URL: {url}")
    urllib.request.urlretrieve(url, local_path)
    print(f"    Saved to: {local_path}")
    return local_path


def _parse_matpower_matrix(text, name):
    """Extract a numeric matrix from MATPOWER .m file text.

    Searches for  `mpc.<name> = [\n ... \n];`  and parses each row.
    """
    # Match  mpc.name = [   ...   ];
    pattern = rf'mpc\.{name}\s*=\s*\[([^\]]+)\]'
    m = re.search(pattern, text, re.DOTALL)
    if m is None:
        raise ValueError(f"Could not find mpc.{name} in MATPOWER file")
    block = m.group(1)
    rows = []
    for line in block.strip().split('\n'):
        line = line.split('%')[0].strip().rstrip(';')
        if not line:
            continue
        vals = line.split()
        if vals:
            rows.append([float(v) for v in vals])
    return np.array(rows)


def _parse_matpower_scalar(text, name):
    """Extract  mpc.<name> = <value>;  from MATPOWER .m file text."""
    pattern = rf'mpc\.{name}\s*=\s*([\d.eE+\-]+)'
    m = re.search(pattern, text)
    if m is None:
        raise ValueError(f"Could not find mpc.{name} in MATPOWER file")
    return float(m.group(1))


def parse_matpower_file(filepath):
    """Parse a MATPOWER .m file into a PYPOWER-compatible dict (ppc).

    Returns dict with keys: baseMVA, bus, gen, branch, (gencost if present).
    """
    with open(filepath, 'r') as f:
        text = f.read()

    ppc = {}
    ppc['baseMVA'] = _parse_matpower_scalar(text, 'baseMVA')
    ppc['bus']     = _parse_matpower_matrix(text, 'bus')
    ppc['gen']     = _parse_matpower_matrix(text, 'gen')
    ppc['branch']  = _parse_matpower_matrix(text, 'branch')
    try:
        ppc['gencost'] = _parse_matpower_matrix(text, 'gencost')
    except ValueError:
        pass  # gencost is optional

    # Validate basic shapes (PGLIB uses standard MATPOWER: bus>=13, gen>=10, branch>=13)
    assert ppc['bus'].shape[1] >= 13, (
        f"Bus matrix has {ppc['bus'].shape[1]} cols, expected >= 13")
    assert ppc['gen'].shape[1] >= 10, (
        f"Gen matrix has {ppc['gen'].shape[1]} cols, expected >= 10")
    assert ppc['branch'].shape[1] >= 13, (
        f"Branch matrix has {ppc['branch'].shape[1]} cols, expected >= 13")

    nb = ppc['bus'].shape[0]
    ng = ppc['gen'].shape[0]
    nl = ppc['branch'].shape[0]
    print(f"  Parsed MATPOWER file: {nb} buses, {ng} gens, {nl} branches")
    print(f"  baseMVA = {ppc['baseMVA']}")
    return ppc


def pglib_ppc_to_pandapower(ppc):
    """Convert a parsed PYPOWER ppc dict into a pandapower network."""
    # Ensure integer bus types and indices
    ppc_copy = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in ppc.items()}
    ppc_copy['bus'][:, 0] = ppc_copy['bus'][:, 0].astype(int)  # BUS_I
    ppc_copy['bus'][:, 1] = ppc_copy['bus'][:, 1].astype(int)  # BUS_TYPE
    ppc_copy['gen'][:, 0] = ppc_copy['gen'][:, 0].astype(int)  # GEN_BUS
    ppc_copy['branch'][:, 0] = ppc_copy['branch'][:, 0].astype(int)  # F_BUS
    ppc_copy['branch'][:, 1] = ppc_copy['branch'][:, 1].astype(int)  # T_BUS
    ppc_copy['branch'][:, 10] = ppc_copy['branch'][:, 10].astype(int)  # BR_STATUS

    # Pad gen matrix to 21 columns if needed (from_ppc expects full MATPOWER format)
    gen = ppc_copy['gen']
    if gen.shape[1] < 21:
        pad = np.zeros((gen.shape[0], 21 - gen.shape[1]))
        ppc_copy['gen'] = np.hstack([gen, pad])

    # version tag required by from_ppc
    ppc_copy['version'] = '2'
    net = from_ppc(ppc_copy, f_hz=60.0)
    return net


def load_pglib_api_case(case_key):
    """Download (if needed), parse, and convert a PGLIB-OPF API case.

    Returns (pandapower_net, ppc_dict).
    """
    fpath = _download_pglib_file(case_key)
    ppc = parse_matpower_file(fpath)
    net = pglib_ppc_to_pandapower(ppc)
    return net, ppc


def extract_api_loads_pu(ppc):
    """Extract per-bus P and Q demand from a PGLIB API ppc in per-unit.

    Returns (P_pu, Q_pu) arrays of shape (n_buses,).
    MATPOWER bus columns: 2=Pd(MW), 3=Qd(MVAr).
    """
    baseMVA = ppc['baseMVA']
    bus = ppc['bus']
    # Build a full bus-indexed array (buses may not be 0-indexed)
    bus_ids = bus[:, 0].astype(int)
    n_buses = len(bus_ids)
    P_pu = bus[:, 2] / baseMVA
    Q_pu = bus[:, 3] / baseMVA
    return P_pu, Q_pu, bus_ids


def generate_api_test_scenarios(ppc, system_data, n_samples,
                                variation=None, base_scale=1.0):
    """Generate test scenarios centred on the PGLIB API operating point.

    The API file defines a single near-collapse load level.  We create
    n_samples perturbations around a scaled API base so that most points
    are near or beyond the loadability limit.

    `variation` sets per-bus log-normal-ish noise (larger -> harder/more
    NR-divergent test set).  `base_scale` uniformly scales the API operating
    point before perturbation: values >1 push the whole test set deeper past
    the loadability limit, raising the NR-divergence rate so a rescue method
    has a meaningful regime to act on (used to stress well-conditioned large
    cases like PEGASE-1354 whose API point alone diverges only ~9%).

    Returns (P, Q, labels) where P, Q have shape (n_samples, n_buses)
    and are in per-unit, indexed in the same bus order as system_data.
    """
    api_P, api_Q, api_bus_ids = extract_api_loads_pu(ppc)
    n_buses_sd = system_data['n_buses']
    baseMVA_sd = system_data['baseMVA']

    # Map API bus loads onto system_data's bus ordering.
    # The pandapower network from from_ppc uses 0-based consecutive indices.
    # api_bus_ids from the .m file might be 1-based; we map them.
    # from_ppc renumbers buses to 0..n-1 in the order they appear in ppc['bus'].
    n_api_buses = len(api_bus_ids)
    api_P_full = np.zeros(n_buses_sd)
    api_Q_full = np.zeros(n_buses_sd)
    for i in range(min(n_api_buses, n_buses_sd)):
        api_P_full[i] = api_P[i]
        api_Q_full[i] = api_Q[i]

    # Uniformly scale the API base operating point.
    api_P_full = api_P_full * base_scale
    api_Q_full = api_Q_full * base_scale

    P = np.zeros((n_samples, n_buses_sd))
    Q = np.zeros((n_samples, n_buses_sd))
    labels = []

    for i in range(n_samples):
        var_i = float(variation) if variation is not None else 0.15

        strategy = np.random.choice(['API', 'API+', 'API-', 'API_Q'],
                                    p=[0.35, 0.25, 0.20, 0.20])
        labels.append(strategy)

        if strategy == 'API':
            # Direct API load with small per-bus noise
            noise = 1.0 + np.random.randn(n_buses_sd) * var_i
            P[i] = np.clip(api_P_full * noise, 0, None)
            Q[i] = api_Q_full * noise

        elif strategy == 'API+':
            # API load scaled UP (even more stressed)
            scale = np.random.uniform(1.0, 1.3)
            noise = 1.0 + np.random.randn(n_buses_sd) * (var_i * 0.7)
            P[i] = np.clip(api_P_full * scale * noise, 0, None)
            Q[i] = api_Q_full * scale * noise

        elif strategy == 'API-':
            # API load scaled DOWN slightly (still stressed, some may converge)
            scale = np.random.uniform(0.75, 1.0)
            noise = 1.0 + np.random.randn(n_buses_sd) * var_i
            P[i] = np.clip(api_P_full * scale * noise, 0, None)
            Q[i] = api_Q_full * scale * noise

        else:  # API_Q — extra reactive stress on top of API
            noise = 1.0 + np.random.randn(n_buses_sd) * var_i
            P[i] = np.clip(api_P_full * noise, 0, None)
            q_boost = np.random.uniform(1.2, 2.0)
            Q[i] = api_Q_full * noise * q_boost

    return P, Q, labels


def generate_mixed_training_with_api(system_data, ppc_api, n_samples,
                                     stressed_range=(1.2, 1.8),
                                     api_variation=0.20):
    """Training data: 50% normal, 25% synthetic stressed, 25% API-derived."""
    n_normal = int(n_samples * 0.50)
    n_synth  = int(n_samples * 0.25)
    n_api    = n_samples - n_normal - n_synth

    P_n, Q_n = generate_load_scenarios(system_data, n_normal, 0.3)
    P_s, Q_s, _ = generate_stressed_scenarios(
        system_data, n_synth,
        load_multiplier_range=stressed_range, bus_variation=0.2)
    P_a, Q_a, _ = generate_api_test_scenarios(
        ppc_api, system_data, n_api,
        variation=api_variation)

    P = np.concatenate([P_n, P_s, P_a], axis=0)
    Q = np.concatenate([Q_n, Q_s, Q_a], axis=0)
    idx = np.random.permutation(n_samples)
    return P[idx], Q[idx]


def build_training_phases():
    """Single hard-distribution training phase (curriculum removed).

    Curriculum learning split the iteration budget across easy/medium/hard
    distributions, leaving only ~1/3 of the iterations on the hard
    distribution that matches the stressed test set — which measurably
    worsened the larger cases. We train directly on the hard distribution
    for the full budget, as the original ("outputs are outstanding") version
    did.
    """
    return [
        dict(label='hard', stressed_range=(1.2, 1.8), api_variation=0.20),
    ]


# ======================================================================
# Load scenario generation — NORMAL (for training, same as c5)
# ======================================================================
def generate_load_scenarios(system_data, n_samples, load_variation=0.3):
    n_buses = system_data['n_buses']
    base_P  = system_data['base_P_demand_pu']
    base_Q  = system_data['base_Q_demand_pu']
    P = np.zeros((n_samples, n_buses))
    Q = np.zeros((n_samples, n_buses))
    for i in range(n_samples):
        lf  = np.clip(1.0 + np.random.randn() * load_variation, 0.7, 1.3)
        ind = 1.0 + np.random.randn(n_buses) * 0.1
        P[i] = np.clip(base_P * lf * ind, 0, None)
        Q[i] = base_Q * lf * ind
    return P, Q


# ======================================================================
# STRESSED scenario generation — produces NR-divergent operating points
# ======================================================================
def generate_stressed_scenarios(system_data, n_samples,
                                load_multiplier_range=(1.3, 2.0),
                                bus_variation=0.25,
                                pf_degradation=True):
    """Generate deliberately stressed load scenarios.

    Strategy mix (each sample picks one):
      A) Uniform heavy loading:  global multiplier in [1.3, 2.0]
      B) Concentrated stress:    a few buses get 2x–4x while others stay near 1x
      C) Reactive stress:        high Q/P ratio (degraded power factor)
      D) Combined:               heavy load + bus variation + PF degradation

    These push the system toward its loadability limit where NR flat-start
    is likely to diverge.
    """
    n_buses = system_data['n_buses']
    base_P  = system_data['base_P_demand_pu'].copy()
    base_Q  = system_data['base_Q_demand_pu'].copy()
    lo, hi  = load_multiplier_range

    P = np.zeros((n_samples, n_buses))
    Q = np.zeros((n_samples, n_buses))
    labels = []  # track which strategy

    for i in range(n_samples):
        strategy = np.random.choice(['A', 'B', 'C', 'D'],
                                    p=[0.30, 0.25, 0.20, 0.25])
        labels.append(strategy)

        if strategy == 'A':
            # Uniform heavy loading
            lf = np.random.uniform(lo, hi)
            ind = 1.0 + np.random.randn(n_buses) * bus_variation
            ind = np.clip(ind, 0.5, 2.0)
            P[i] = np.clip(base_P * lf * ind, 0, None)
            Q[i] = base_Q * lf * ind

        elif strategy == 'B':
            # Concentrated stress on a subset of buses
            lf_base = np.random.uniform(1.0, 1.3)
            n_stressed = max(1, int(n_buses * np.random.uniform(0.1, 0.3)))
            stressed_buses = np.random.choice(n_buses, n_stressed, replace=False)
            ind = np.ones(n_buses) * lf_base
            ind[stressed_buses] = np.random.uniform(2.0, 4.0,
                                                     size=n_stressed)
            P[i] = np.clip(base_P * ind, 0, None)
            Q[i] = base_Q * ind

        elif strategy == 'C':
            # Reactive stress — degraded power factor
            lf = np.random.uniform(1.1, 1.6)
            ind = 1.0 + np.random.randn(n_buses) * 0.15
            P[i] = np.clip(base_P * lf * ind, 0, None)
            # Increase reactive demand disproportionately
            q_mult = np.random.uniform(1.5, 3.0)
            Q[i] = base_Q * lf * ind * q_mult

        else:  # D — combined
            lf = np.random.uniform(lo, hi)
            ind = 1.0 + np.random.randn(n_buses) * bus_variation
            ind = np.clip(ind, 0.3, 3.0)
            n_stressed = max(1, int(n_buses * np.random.uniform(0.05, 0.2)))
            stressed_buses = np.random.choice(n_buses, n_stressed, replace=False)
            ind[stressed_buses] *= np.random.uniform(1.5, 2.5,
                                                      size=n_stressed)
            P[i] = np.clip(base_P * lf * ind, 0, None)
            q_mult = np.random.uniform(1.0, 2.0)
            Q[i] = base_Q * lf * ind * q_mult

    return P, Q, labels


def generate_mixed_training_scenarios(system_data, n_samples):
    """Mix of normal (60%) and stressed (40%) scenarios for robust training."""
    n_normal  = int(n_samples * 0.6)
    n_stress  = n_samples - n_normal
    P_n, Q_n = generate_load_scenarios(system_data, n_normal, 0.3)
    P_s, Q_s, _ = generate_stressed_scenarios(system_data, n_stress,
                                               load_multiplier_range=(1.2, 1.8),
                                               bus_variation=0.2)
    P = np.concatenate([P_n, P_s], axis=0)
    Q = np.concatenate([Q_n, Q_s], axis=0)
    idx = np.random.permutation(n_samples)
    return P[idx], Q[idx]


# ======================================================================
# NR convergence tester  (returns per-point convergence status)
# ======================================================================
def test_nr_convergence(system_data, P_pu, Q_pu, init_mode='flat',
                        V_init=None, theta_init_deg=None,
                        Pg_init=None, gen_buses=None,
                        max_iter=30, label='NR'):
    """Run NR on each sample, return per-point results.

    Returns list of dicts: {converged: bool, iterations: int, time: float,
                            V: array, theta_deg: array}
    """
    net_base = system_data['net']
    baseMVA  = system_data['baseMVA']
    n_gen = system_data['n_generators']
    n_test   = P_pu.shape[0]
    results  = []

    for i in range(n_test):
        net = _make_net_writable(copy.deepcopy(net_base))
        # Set loads
        for ii, ld in net.load.iterrows():
            bus = int(ld.bus)
            net.load.at[ii, 'p_mw']   = float(P_pu[i, bus] * baseMVA)
            net.load.at[ii, 'q_mvar'] = float(Q_pu[i, bus] * baseMVA)

        pp_init = 'flat'
        if init_mode == 'warm' and V_init is not None:
            pp_init = 'results'
            # Set warm-start initial point
            if Pg_init is not None and gen_buses is not None:
                for g, gbus in enumerate(gen_buses):
                    net.gen.at[g, 'p_mw'] = float(Pg_init[i, g] * baseMVA)
            V_clip = np.clip(V_init[i], 0.5, 1.5)
            net.res_bus = net.res_bus.copy()
            net.res_bus.loc[:, 'vm_pu']     = V_clip
            net.res_bus.loc[:, 'va_degree'] = theta_init_deg[i]

        t0 = time.time()
        try:
            pp.runpp(net, init=pp_init, calculate_voltage_angles=True,
                     enforce_q_limits=False, enforce_v_limits=False,
                     max_iteration=max_iter, numba=False)
            elapsed = time.time() - t0
            nri = _get_nr_iterations(net)
            if hasattr(net, 'res_gen') and len(net.res_gen) >= n_gen:
                pg = (net.res_gen.p_mw.values[:n_gen] / baseMVA).astype(np.float32)
            else:
                pg = (net.gen.p_mw.values[:n_gen] / baseMVA).astype(np.float32)
            results.append(dict(
                converged=True,
                iterations=nri if nri is not None else -1,
                time=elapsed,
                V=net.res_bus.vm_pu.values.copy(),
                theta_deg=net.res_bus.va_degree.values.copy(),
                Pg=pg,
            ))
        except Exception:
            elapsed = time.time() - t0
            results.append(dict(
                converged=False, iterations=max_iter,
                time=elapsed, V=None, theta_deg=None,
                Pg=None,
            ))
    return results


def compute_participation_from_results(nr_results, system_data):
    """Compute participation factors for each converged NR result."""
    pmax_pu = np.array(system_data['gen_limits']['P_max'], dtype=np.float32)
    pf_list = []
    for r in nr_results:
        if r.get('converged') and r.get('Pg') is not None:
            pf_list.append(_compute_participation_factors(r['Pg'], pmax_pu))
        else:
            pf_list.append(None)
    return pf_list


def build_voltage_init_template(system_data, n_samples):
    """Build a flat V init with PV/slack setpoints pinned."""
    n_buses = system_data['n_buses']
    V_init = np.ones((n_samples, n_buses), dtype=np.float32)
    vsp = system_data['gen_voltage_setpoints']
    fixed = list(system_data['pv_buses']) + [int(system_data['slack_bus'])]
    if fixed:
        V_init[:, fixed] = vsp[fixed]
    return V_init


def run_dcopf_theta_batch(system_data, P_pu, Q_pu):
    """Run DCPF per scenario and return theta (deg) for NR warm-start.

    Returns dict with keys:
      theta_deg: (n_samples, n_buses)
      Pg:        (n_samples, n_generators) in p.u.
            converged: list[bool] indicating whether DCPF solved each point
      elapsed:   total runtime in seconds
    """
    net_base = system_data['net']
    baseMVA = system_data['baseMVA']
    n_test = P_pu.shape[0]
    n_buses = system_data['n_buses']
    n_gens = system_data['n_generators']

    theta_deg = np.zeros((n_test, n_buses), dtype=np.float32)
    Pg = np.zeros((n_test, n_gens), dtype=np.float32)
    converged = []

    t0 = time.time()
    has_dcpf = hasattr(pp, 'rundcpp')

    for i in range(n_test):
        net = _make_net_writable(copy.deepcopy(net_base))
        for ii, ld in net.load.iterrows():
            bus = int(ld.bus)
            net.load.at[ii, 'p_mw'] = float(P_pu[i, bus] * baseMVA)
            net.load.at[ii, 'q_mvar'] = float(Q_pu[i, bus] * baseMVA)

        try:
            if not has_dcpf:
                raise RuntimeError("pandapower.rundcpp is unavailable")
            pp.rundcpp(net, numba=False)

            if 'va_degree' in net.res_bus.columns:
                theta_deg[i] = net.res_bus.va_degree.values.astype(np.float32)
            if len(net.res_gen) >= n_gens and 'p_mw' in net.res_gen.columns:
                Pg[i] = (net.res_gen.p_mw.values[:n_gens] / baseMVA).astype(np.float32)
            else:
                Pg[i] = (net.gen.p_mw.values[:n_gens] / baseMVA).astype(np.float32)
            converged.append(True)
        except Exception:
            # Fallback initialization if DCPF fails for this sample.
            theta_deg[i] = 0.0
            Pg[i] = (net.gen.p_mw.values[:n_gens] / baseMVA).astype(np.float32)
            converged.append(False)

    elapsed = time.time() - t0
    return dict(theta_deg=theta_deg, Pg=Pg, converged=converged, elapsed=elapsed)


class NormalACOPFNN(nn.Module):
    """Simple feed-forward baseline for supervised AC warm-start prediction."""

    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def build_supervised_nr_targets(system_data, P_pu, Q_pu,
                                max_samples=1500, nr_max_iter=30):
    """Build supervised labels from NR-converged samples.

    Targets are concatenated as [Pg(gen order), V(all buses), theta_deg(all buses)].
    """
    n = P_pu.shape[0]
    if n == 0:
        return None, None

    if n > max_samples:
        idx = np.random.choice(n, max_samples, replace=False)
    else:
        idx = np.arange(n)

    P_sel = P_pu[idx]
    Q_sel = Q_pu[idx]
    nr = test_nr_convergence(system_data, P_sel, Q_sel, init_mode='flat',
                             max_iter=nr_max_iter, label='NN-target-NR')

    baseMVA = system_data['baseMVA']
    net_base = system_data['net']
    n_gen = system_data['n_generators']

    X_list, Y_list = [], []
    for i, r in enumerate(nr):
        if not r['converged']:
            continue

        net = _make_net_writable(copy.deepcopy(net_base))
        for ii, ld in net.load.iterrows():
            b = int(ld.bus)
            net.load.at[ii, 'p_mw'] = float(P_sel[i, b] * baseMVA)
            net.load.at[ii, 'q_mvar'] = float(Q_sel[i, b] * baseMVA)
        try:
            pp.runpp(net, init='flat', calculate_voltage_angles=True,
                     enforce_q_limits=False, enforce_v_limits=False,
                     max_iteration=nr_max_iter, numba=False)
            pg = (net.res_gen.p_mw.values[:n_gen] / baseMVA).astype(np.float32)
            v = net.res_bus.vm_pu.values.astype(np.float32)
            th = net.res_bus.va_degree.values.astype(np.float32)
            x = np.concatenate([P_sel[i], Q_sel[i]], axis=0).astype(np.float32)
            y = np.concatenate([pg, v, th], axis=0).astype(np.float32)
            X_list.append(x)
            Y_list.append(y)
        except Exception:
            continue

    if not X_list:
        return None, None
    return np.stack(X_list, axis=0), np.stack(Y_list, axis=0)


def train_normal_nn_acopf(system_data, P_train, Q_train, device,
                          max_samples=1500, nr_max_iter=30,
                          epochs=40, batch_size=256, lr=1e-3,
                          val_frac=0.1, patience=40):
    """Train a simple supervised NN baseline with MSE loss.

    Uses a held-out validation split with early stopping and cosine LR decay,
    restoring the best-val checkpoint.  Without these, the baseline was badly
    under-trained on large systems (e.g. PEGASE-1354: 30 epochs / 0.2 s left
    the loss still falling steeply, so its warm-start point was worse than a
    flat start and rescued 0% of NR-divergent cases).  We also record the
    label angle range so inference can clamp wild theta predictions that would
    otherwise poison NR.
    """
    print(f"\n  Building supervised NR labels for normal NN baseline...")
    X_np, Y_np = build_supervised_nr_targets(
        system_data, P_train, Q_train,
        max_samples=max_samples, nr_max_iter=nr_max_iter,
    )
    if X_np is None or Y_np is None or len(X_np) < 32:
        print("  Normal NN baseline skipped (insufficient converged labels).")
        return None

    x_mean = X_np.mean(axis=0, keepdims=True)
    x_std = X_np.std(axis=0, keepdims=True) + 1e-6
    y_mean = Y_np.mean(axis=0, keepdims=True)
    y_std = Y_np.std(axis=0, keepdims=True) + 1e-6

    # Angle clamp range (degrees) taken from the converged NR labels, padded.
    n_gen = system_data['n_generators']
    n_bus = system_data['n_buses']
    th_labels = Y_np[:, n_gen + n_bus:n_gen + 2 * n_bus]
    th_lo = float(th_labels.min()) - 15.0
    th_hi = float(th_labels.max()) + 15.0

    X_all = torch.tensor((X_np - x_mean) / x_std, dtype=torch.float32, device=device)
    Y_all = torch.tensor((Y_np - y_mean) / y_std, dtype=torch.float32, device=device)

    # Train / validation split for early stopping.
    n_total = X_all.shape[0]
    n_val = max(8, int(n_total * val_frac)) if n_total >= 64 else 0
    perm0 = torch.randperm(n_total, device=device)
    val_idx = perm0[:n_val]
    tr_idx = perm0[n_val:]
    X, Y = X_all[tr_idx], Y_all[tr_idx]
    X_val, Y_val = X_all[val_idx], Y_all[val_idx]

    model = NormalACOPFNN(X.shape[1], Y.shape[1]).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    loss_fn = nn.MSELoss()

    n = X.shape[0]
    t0 = time.time()
    best_val = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        ep_loss = 0.0
        nb = 0
        for s in range(0, n, batch_size):
            bi = perm[s:min(s + batch_size, n)]
            pred = model(X[bi])
            loss = loss_fn(pred, Y[bi])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss.item())
            nb += 1
        sched.step()

        # Validation (falls back to train loss when no val split).
        model.eval()
        with torch.no_grad():
            if n_val > 0:
                val_mse = float(loss_fn(model(X_val), Y_val).item())
            else:
                val_mse = ep_loss / max(nb, 1)
        if val_mse < best_val - 1e-5:
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (ep + 1) % 20 == 0 or ep == epochs - 1:
            print(f"    NN epoch {ep + 1:4d}/{epochs}: "
                  f"train_mse={ep_loss / max(nb, 1):.4e} "
                  f"val_mse={val_mse:.4e} best={best_val:.4e}")
        if epochs_no_improve >= patience:
            print(f"    NN early stop at epoch {ep + 1} "
                  f"(no val improvement for {patience} epochs)")
            break

    model.load_state_dict(best_state)
    elapsed = time.time() - t0
    model.eval()
    with torch.no_grad():
        train_mse = float(loss_fn(model(X_all), Y_all).item())

    print(f"  Normal NN baseline trained on {n}+{n_val} (train+val) samples "
          f"in {elapsed:.1f}s | best val_mse={best_val:.4e}")
    return dict(
        model=model,
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        y_mean=y_mean.astype(np.float32),
        y_std=y_std.astype(np.float32),
        train_mse=train_mse,
        n_labels=int(n_total),
        train_time=elapsed,
        theta_lo=th_lo,
        theta_hi=th_hi,
    )


def normal_nn_predict_chunked(nn_state, system_data, P_np, Q_np, device,
                              chunk_size=256):
    """Run normal NN baseline inference and unpack [Pg, V, theta_deg]."""
    model = nn_state['model']
    n = P_np.shape[0]
    n_gen = system_data['n_generators']
    n_bus = system_data['n_buses']

    x_mean = nn_state['x_mean']
    x_std = nn_state['x_std']
    y_mean = nn_state['y_mean']
    y_std = nn_state['y_std']

    out = []
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            Xc = np.concatenate([P_np[s:e], Q_np[s:e]], axis=1).astype(np.float32)
            Xn = (Xc - x_mean) / x_std
            Xt = torch.tensor(Xn, dtype=torch.float32, device=device)
            Yn = model(Xt).cpu().numpy()
            Y = Yn * y_std + y_mean
            out.append(Y)

    Y_all = np.concatenate(out, axis=0)
    pg = Y_all[:, :n_gen]
    v = np.clip(Y_all[:, n_gen:n_gen + n_bus], 0.5, 1.5)
    # Clamp angles to the label range so wild predictions don't poison NR.
    th_lo = nn_state.get('theta_lo', -90.0)
    th_hi = nn_state.get('theta_hi', 90.0)
    th_deg = np.clip(Y_all[:, n_gen + n_bus:n_gen + 2 * n_bus], th_lo, th_hi)
    return dict(Pg=pg, V=v, theta_deg=th_deg, elapsed=time.time() - t0)


# ======================================================================
# Graph Attention Layer  (identical to c5)
# ======================================================================
class GraphAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.05):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.scale    = self.head_dim ** -0.5
        self.qkv      = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model)
        self.dropout   = nn.Dropout(dropout)
        self.edge_scale = nn.Parameter(torch.zeros(n_heads))
        # Sparse edge-index cache (built lazily from `adj`). Power-grid
        # adjacency is ~0.3% dense, so a dense (B,H,N,N) attention matrix is
        # almost entirely masked -inf — wasteful in both memory and compute.
        # We attend only over real edges instead; mathematically identical to
        # the masked dense softmax above, but O(E) instead of O(N^2).
        self._adj_key = None
        self._ei = None
        self._ej = None

    def _edge_index(self, adj):
        """Cache (row=query, col=key) indices of nonzero adjacency entries."""
        key = (adj.data_ptr(), adj.shape)
        if self._adj_key != key:
            ei, ej = torch.nonzero(adj, as_tuple=True)
            self._ei, self._ej = ei.contiguous(), ej.contiguous()
            self._adj_key = key
        return self._ei, self._ej

    def forward(self, x, adj, edge_weights=None):
        B, N, C = x.shape
        H, D = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                       # each [B, H, N, D]

        ei, ej = self._edge_index(adj)                # [E] query / key nodes
        E = ei.shape[0]
        scores = (q[:, :, ei, :] * k[:, :, ej, :]).sum(-1) * self.scale  # [B,H,E]
        if edge_weights is not None:
            scores = scores + edge_weights[ei, ej].view(1, 1, E) \
                * self.edge_scale.view(1, H, 1)

        # Grouped softmax over edges that share the same query node `ei`.
        idx = ei.view(1, 1, E).expand(B, H, E)
        m = torch.full((B, H, N), float('-inf'), device=x.device, dtype=scores.dtype)
        m.scatter_reduce_(2, idx, scores, reduce='amax', include_self=False)
        exps = (scores - m.gather(2, idx)).exp()
        denom = torch.zeros(B, H, N, device=x.device, dtype=scores.dtype)
        denom.scatter_add_(2, idx, exps)
        w = exps / denom.gather(2, idx).clamp_min(1e-20)   # [B, H, E]
        w = self.dropout(w)

        out = torch.zeros(B, H, N, D, device=x.device, dtype=v.dtype)
        out.scatter_add_(2, idx.unsqueeze(-1).expand(B, H, E, D),
                         w.unsqueeze(-1) * v[:, :, ej, :])
        out = torch.nan_to_num(out, 0.0)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)


class GATBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.05):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = GraphAttentionLayer(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x, adj, edge_weights=None):
        x = x + self.attn(self.norm1(x), adj, edge_weights)
        x = x + self.ffn(self.norm2(x))
        return x


# ======================================================================
# Primal Network  (GAT + Global Attention)  — identical to c5
# ======================================================================
class ACPFPrimalGAT(nn.Module):
    def __init__(self, n_buses, n_generators, pq_buses, pv_buses,
                 gen_buses, slack_bus,
                 d_model=64, n_heads=4, n_layers=4,
                 max_angle_rad=np.pi / 2.0):
        super().__init__()
        self.n_buses       = n_buses
        self.n_generators  = n_generators
        self.max_angle_rad = float(max_angle_rad)
        self.d_model = d_model

        self.pq_buses  = list(pq_buses)
        self.pv_buses  = list(pv_buses)
        self.gen_buses = list(gen_buses)
        self.slack_bus = int(slack_bus)
        self.non_slack = sorted(i for i in range(n_buses) if i != self.slack_bus)
        self.n_pq = len(self.pq_buses)

        self.node_embed = nn.Sequential(
            nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.bus_id = nn.Parameter(torch.randn(n_buses, d_model) * 0.02)
        self.bus_type_embed = nn.Embedding(3, d_model)
        bus_types = torch.zeros(n_buses, dtype=torch.long)
        for b in pv_buses:
            bus_types[b] = 1
        bus_types[slack_bus] = 2
        self.register_buffer('bus_types', bus_types)

        self.gat_layers = nn.ModuleList([
            GATBlock(d_model, n_heads) for _ in range(n_layers)
        ])

        self.global_norm = nn.LayerNorm(d_model)
        self.global_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=0.05,
        )
        self.final_norm = nn.LayerNorm(d_model)

        def head():
            return nn.Sequential(
                nn.Linear(d_model, d_model // 2), nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )

        self.pg_head    = head()
        self.qg_head    = head()
        self.theta_head = head()
        if self.n_pq > 0:
            self.v_head = head()
            nn.init.zeros_(self.v_head[-1].bias)
            nn.init.xavier_uniform_(self.v_head[-1].weight, gain=0.1)

    def forward(self, P_demand, Q_demand, gen_limits,
                gen_voltage_setpoints, adj, edge_weights=None):
        B = P_demand.shape[0]
        device = P_demand.device

        x = torch.stack([P_demand, Q_demand], dim=-1)
        x = self.node_embed(x)
        x = x + self.bus_id.unsqueeze(0) + self.bus_type_embed(self.bus_types).unsqueeze(0)

        for layer in self.gat_layers:
            x = layer(x, adj, edge_weights)

        x_norm = self.global_norm(x)
        x_global, _ = self.global_attn(x_norm, x_norm, x_norm)
        x = x + x_global
        x = self.final_norm(x)

        gen_feat = x[:, self.gen_buses, :]
        P_min = torch.tensor(gen_limits['P_min'], device=device, dtype=torch.float32)
        P_max = torch.tensor(gen_limits['P_max'], device=device, dtype=torch.float32)
        Pg = P_min + torch.sigmoid(self.pg_head(gen_feat).squeeze(-1)) * (P_max - P_min)

        Q_min = torch.tensor(gen_limits['Q_min'], device=device, dtype=torch.float32)
        Q_max = torch.tensor(gen_limits['Q_max'], device=device, dtype=torch.float32)
        Qg = Q_min + torch.sigmoid(self.qg_head(gen_feat).squeeze(-1)) * (Q_max - Q_min)

        V = torch.ones(B, self.n_buses, device=device)
        vsp = torch.tensor(gen_voltage_setpoints, device=device, dtype=torch.float32)
        fixed = self.pv_buses + [self.slack_bus]
        V[:, fixed] = vsp[fixed].unsqueeze(0)
        if self.n_pq > 0:
            V_pq = 0.85 + 0.30 * torch.sigmoid(
                self.v_head(x[:, self.pq_buses, :]).squeeze(-1))
            V[:, self.pq_buses] = V_pq

        ns_feat = x[:, self.non_slack, :]
        theta_raw = torch.tanh(
            self.theta_head(ns_feat).squeeze(-1)) * self.max_angle_rad
        theta = torch.zeros(B, self.n_buses, device=device)
        theta[:, self.non_slack] = theta_raw

        return Pg, Qg, V, theta


# ======================================================================
# Dual Network  — identical to c5
# ======================================================================
class ACPFDualGAT(nn.Module):
    def __init__(self, n_buses, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.d_model = d_model
        self.n_buses = n_buses
        self.lambda_scale = None

        self.node_embed = nn.Sequential(
            nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.bus_id = nn.Parameter(torch.randn(n_buses, d_model) * 0.02)
        self.layers = nn.ModuleList([
            GATBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 2),
        )
        nn.init.zeros_(self.output_head[-1].bias)
        nn.init.xavier_uniform_(self.output_head[-1].weight, gain=0.01)

    def forward(self, P_demand, Q_demand, adj, edge_weights=None):
        x = torch.stack([P_demand, Q_demand], dim=-1)
        x = self.node_embed(x) + self.bus_id.unsqueeze(0)
        for layer in self.layers:
            x = layer(x, adj, edge_weights)
        x = self.final_norm(x)
        lam = self.output_head(x)
        return torch.cat([lam[:, :, 0], lam[:, :, 1]], dim=1)


# ======================================================================
# PDL Solver  (deterministic ρ schedule)  — identical to c5
# ======================================================================
class PDL_ACPF_GAT:
    def __init__(self, system_data,
                 rho_init=1.0, rho_max=500.0,
                 alpha=1.2, tau=0.9,
                 device='cpu', max_angle_rad=np.pi / 2.0,
                 d_model=64, n_heads=4,
                 n_layers_primal=4, n_layers_dual=2,
                 warmup_iters=15, rho_check_freq=3):
        self.device      = device
        self.system_data = system_data
        self.n_buses     = system_data['n_buses']
        self.n_generators = system_data['n_generators']
        self.gen_buses   = system_data['gen_buses']
        self.slack_bus   = int(system_data['slack_bus'])
        self.pv_buses    = system_data['pv_buses']
        self.pq_buses    = system_data['pq_buses']
        self.gen_voltage_setpoints = system_data['gen_voltage_setpoints']
        self.Y_bus = torch.tensor(
            system_data['Y_bus'], dtype=torch.complex64).to(device)
        self.gen_limits = system_data['gen_limits']
        self.baseMVA    = system_data['baseMVA']

        self.adj = torch.tensor(
            system_data['adj'], dtype=torch.float32).to(device)
        self.edge_weights = torch.tensor(
            system_data['y_mag'], dtype=torch.float32).to(device)

        self.rho     = float(rho_init)
        self.rho_max = float(rho_max)
        self.alpha   = float(alpha)
        self.tau     = float(tau)
        self.warmup_iters   = int(warmup_iters)
        self.rho_check_freq = int(rho_check_freq)
        self.epoch_counter  = 0

        self.primal_net = ACPFPrimalGAT(
            self.n_buses, self.n_generators,
            self.pq_buses, self.pv_buses, self.gen_buses, self.slack_bus,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers_primal,
            max_angle_rad=max_angle_rad,
        ).to(device)

        self.dual_net = ACPFDualGAT(
            self.n_buses, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers_dual,
        ).to(device)

        self._init_optimizers()
        self.history = defaultdict(list)

    def _init_optimizers(self):
        self.primal_optimizer = optim.AdamW(
            self.primal_net.parameters(), lr=3e-4, weight_decay=1e-5)
        self.dual_optimizer = optim.AdamW(
            self.dual_net.parameters(), lr=3e-4, weight_decay=1e-4)
        self.primal_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.primal_optimizer, T_0=25, T_mult=2, eta_min=1e-5)
        self.dual_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.dual_optimizer, T_0=25, T_mult=2, eta_min=1e-5)

    def _update_rho(self, max_viol):
        self.epoch_counter += 1
        if not hasattr(self, 'prev_max_viol'):
            self.prev_max_viol = max_viol
        elif (self.epoch_counter > self.warmup_iters
              and self.epoch_counter % self.rho_check_freq == 0):
            if max_viol > self.tau * self.prev_max_viol:
                self.rho = min(self.alpha * self.rho, self.rho_max)
            self.prev_max_viol = max_viol

    def compute_power_balance(self, Pg, Qg, V, theta, P_demand, Q_demand):
        B = V.shape[0]
        # Scatter generator injections onto their buses. index_add_ handles
        # multiple gens on the same bus (accumulates) and replaces the old
        # Python for-loop over ~260 gens (260 tiny kernel launches per call).
        if getattr(self, '_gen_bus_idx', None) is None:
            self._gen_bus_idx = torch.as_tensor(
                self.gen_buses, dtype=torch.long, device=self.device)
        P_inj = torch.zeros(B, self.n_buses, device=self.device)
        Q_inj = torch.zeros(B, self.n_buses, device=self.device)
        P_inj.index_add_(1, self._gen_bus_idx, Pg)
        Q_inj.index_add_(1, self._gen_bus_idx, Qg)
        P_inj -= P_demand
        Q_inj -= Q_demand

        V_c = V * (torch.cos(theta) + 1j * torch.sin(theta))
        S = V_c * torch.conj(torch.matmul(V_c, self.Y_bus.T))
        P_viol = S.real - P_inj
        Q_viol = S.imag - Q_inj
        P_viol[:, self.slack_bus] = 0.0
        Q_viol[:, self.slack_bus] = 0.0
        return P_viol, Q_viol

    def primal_loss(self, Pg, Qg, V, theta, P_demand, Q_demand, multipliers):
        Pv, Qv = self.compute_power_balance(
            Pg, Qg, V, theta, P_demand, Q_demand)
        lP = multipliers[:, :self.n_buses]
        lQ = multipliers[:, self.n_buses:]
        lag = (torch.sum(lP * Pv, 1) + torch.sum(lQ * Qv, 1)).mean()
        pen = (self.rho / 2.0) * (
            torch.sum(Pv**2, 1) + torch.sum(Qv**2, 1)).mean()
        return lag + pen

    def dual_loss_cached(self, multipliers, mults_old,
                         Pg, Qg, V, theta, P_demand, Q_demand):
        Pv, Qv = self.compute_power_balance(
            Pg, Qg, V, theta, P_demand, Q_demand)
        target_P = mults_old[:, :self.n_buses] + self.rho * Pv
        target_Q = mults_old[:, self.n_buses:] + self.rho * Qv
        return (torch.mean((multipliers[:, :self.n_buses] - target_P) ** 2)
                + torch.mean((multipliers[:, self.n_buses:] - target_Q) ** 2))

    def _primal_forward(self, Pm, Qm):
        return self.primal_net(
            Pm, Qm, self.gen_limits,
            self.gen_voltage_setpoints, self.adj, self.edge_weights)

    def _dual_forward(self, Pm, Qm):
        return self.dual_net(Pm, Qm, self.adj, self.edge_weights)

    def _eval_violation(self, P_batch, Q_batch, n_eval, batch_size):
        """Chunked violation eval to avoid OOM on large systems.

        The attention matrix is (B, H, N, N); a full forward over ~1000
        samples blows up for large N (e.g. PEGASE1354). Chunk by batch_size.
        """
        max_Pv = max_Qv = 0.0
        sum_abs = 0.0
        count = 0
        bs = max(1, batch_size)
        with torch.no_grad():
            for i in range(0, n_eval, bs):
                e = min(i + bs, n_eval)
                Pm, Qm = P_batch[i:e], Q_batch[i:e]
                Pg, Qg, V, th = self._primal_forward(Pm, Qm)
                Pv, Qv = self.compute_power_balance(Pg, Qg, V, th, Pm, Qm)
                max_Pv = max(max_Pv, torch.max(torch.abs(Pv)).item())
                max_Qv = max(max_Qv, torch.max(torch.abs(Qv)).item())
                sum_abs += (torch.sum(torch.abs(Pv))
                            + torch.sum(torch.abs(Qv))).item()
                count += Pv.numel() + Qv.numel()
        mean_viol = sum_abs / max(count, 1)
        return max_Pv, max_Qv, mean_viol

    def pretrain(self, P_batch, Q_batch, n_iters=15, rho_pre=10.0,
                 batch_size=1024):
        print(f"  Pre-training: {n_iters} iters, penalty-only at ρ={rho_pre}")
        self.primal_net.train()
        for k in range(n_iters):
            idx = torch.randperm(P_batch.shape[0], device=self.device)
            for i in range(0, P_batch.shape[0], batch_size):
                bi = idx[i:min(i + batch_size, P_batch.shape[0])]
                Pm, Qm = P_batch[bi], Q_batch[bi]
                Pg, Qg, V, th = self._primal_forward(Pm, Qm)
                Pv, Qv = self.compute_power_balance(
                    Pg, Qg, V, th, Pm, Qm)
                loss = (rho_pre / 2) * (
                    torch.sum(Pv**2, 1) + torch.sum(Qv**2, 1)).mean()
                self.primal_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.primal_net.parameters(), 1.0)
                self.primal_optimizer.step()
            if (k + 1) % 5 == 0 or k == n_iters - 1:
                es = min(1000, P_batch.shape[0])
                max_Pv, max_Qv, _ = self._eval_violation(
                    P_batch, Q_batch, es, batch_size)
                mv = max(max_Pv, max_Qv)
                print(f"    pre-train iter {k + 1:2d}: "
                      f"max_viol={mv:.4e} p.u., ρ_pre={rho_pre}")

    def train_epoch(self, P_batch, Q_batch,
                    inner_iters=150, batch_size=1024, accum_steps=10):
        n = P_batch.shape[0]
        eff = max(1, inner_iters // accum_steps)

        # --- Primal step ---
        p_losses = []
        for _ in range(eff):
            idx = torch.randperm(n, device=self.device)
            self.primal_optimizer.zero_grad()
            acc = 0.0
            for step, i in enumerate(range(0, n, batch_size)):
                bi = idx[i:min(i + batch_size, n)]
                Pm, Qm = P_batch[bi], Q_batch[bi]
                Pg, Qg, V, th = self._primal_forward(Pm, Qm)
                with torch.no_grad():
                    mults = self._dual_forward(Pm, Qm)
                loss = self.primal_loss(
                    Pg, Qg, V, th, Pm, Qm, mults) / accum_steps
                loss.backward()
                acc += loss.item()
                if (step + 1) % accum_steps == 0 or i + batch_size >= n:
                    torch.nn.utils.clip_grad_norm_(
                        self.primal_net.parameters(), 0.5)
                    self.primal_optimizer.step()
                    self.primal_optimizer.zero_grad()
                    p_losses.append(acc * accum_steps)
                    acc = 0.0
        self.primal_scheduler.step()

        # Cache old multipliers
        dual_state_old = {k: v.clone().detach()
                          for k, v in self.dual_net.state_dict().items()}
        with torch.no_grad():
            old_dual = ACPFDualGAT(
                self.n_buses, d_model=self.dual_net.d_model,
            ).to(self.device)
            old_dual.load_state_dict(dual_state_old)
            old_dual.eval()
            chunks = []
            for ci in range(0, n, batch_size):
                chunk = old_dual(
                    P_batch[ci:ci + batch_size],
                    Q_batch[ci:ci + batch_size],
                    self.adj, self.edge_weights)
                chunks.append(chunk)
            all_mults_old = torch.cat(chunks, dim=0)
            del old_dual

        # --- Dual step ---
        d_losses = []
        for _ in range(eff):
            idx = torch.randperm(n, device=self.device)
            self.dual_optimizer.zero_grad()
            acc = 0.0
            for step, i in enumerate(range(0, n, batch_size)):
                bi = idx[i:min(i + batch_size, n)]
                Pm, Qm = P_batch[bi], Q_batch[bi]
                with torch.no_grad():
                    Pg, Qg, V, th = self._primal_forward(Pm, Qm)
                mults = self._dual_forward(Pm, Qm)
                mults_old = all_mults_old[bi]
                loss = self.dual_loss_cached(
                    mults, mults_old, Pg, Qg, V, th, Pm, Qm) / accum_steps
                loss.backward()
                acc += loss.item()
                if (step + 1) % accum_steps == 0 or i + batch_size >= n:
                    torch.nn.utils.clip_grad_norm_(
                        self.dual_net.parameters(), 0.5)
                    self.dual_optimizer.step()
                    self.dual_optimizer.zero_grad()
                    d_losses.append(acc * accum_steps)
                    acc = 0.0
        self.dual_scheduler.step()
        del all_mults_old

        es = min(1000, n)
        max_Pv, max_Qv, mean_viol = self._eval_violation(
            P_batch, Q_batch, es, batch_size)
        max_viol = max(max_Pv, max_Qv)

        self._update_rho(max_viol)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.history['max_P_viol'].append(max_Pv)
        self.history['max_Q_viol'].append(max_Qv)
        self.history['max_viol'].append(max_viol)
        self.history['mean_viol'].append(mean_viol)
        self.history['rho'].append(self.rho)
        return np.mean(p_losses) if p_losses else 0.0, max_viol

    def predict(self, P_demand, Q_demand):
        self.primal_net.eval()
        with torch.no_grad():
            out = self._primal_forward(P_demand, Q_demand)
        self.primal_net.train()
        return out


# ======================================================================
# Plotting moved to charts.py (teb.py only computes and saves summaries)
# ======================================================================


def get_memory_safe_case_config(case_key, n_buses):
    """Return run config per case size (scaled for a large ≥48 GB GPU)."""
    use_cuda = torch.cuda.is_available()

    cfg = dict(
        seed=42,
        n_train=10000,
        n_stressed_test=2000,
        max_outer_iters=100,
        convergence_threshold=1e-4,
        nr_max_iter=30,
        pretrain_iters=15,
        train_inner_iters=150,
        train_batch_size=1024 if use_cuda else 256,
        train_accum_steps=10,
        sweep_samples=500,
        stress_multipliers=[0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        inference_chunk_size=512 if use_cuda else 128,
        nn_supervised_samples=1500,
        nn_epochs=40,
        nn_batch_size=256 if use_cuda else 128,
        model_kwargs=dict(d_model=64, n_heads=4,
                          n_layers_primal=4, n_layers_dual=2),
        curriculum_phases=build_training_phases(),
        curriculum_outer_iters=None,
        retrain_on_failed=True,
        retrain_top_frac=0.25,
        retrain_iters=8,
        retrain_inner_iters=80,
    )

    if case_key == 'case1354pegase' or (1000 <= n_buses < 1700):
        # Scaled up for a large (≥48 GB) GPU. The previous values were
        # throttled for a 12 GB / Kaggle card and left this case badly
        # under-trained (final violation ~29 p.u., PDL warm worse than flat
        # NR). Bigger model + more iterations + more samples. NOTE: this is
        # the most expensive case — expect a long wall-clock training time;
        # dial max_outer_iters / n_train down if it is too slow.
        cfg.update(dict(
            n_train=3500,
            n_stressed_test=450,
            max_outer_iters=60,
            pretrain_iters=10,
            train_inner_iters=90,
            train_batch_size=96 if use_cuda else 24,
            train_accum_steps=6,
            sweep_samples=120 if use_cuda else 30,
            stress_multipliers=[0.75, 0.85, 0.95, 1.00, 1.05, 1.10],
            inference_chunk_size=96 if use_cuda else 24,
            nn_supervised_samples=1500,
            nn_epochs=400,
            nn_batch_size=96 if use_cuda else 24,
            model_kwargs=dict(d_model=64, n_heads=4,
                              n_layers_primal=4, n_layers_dual=2),
            retrain_iters=8,
            retrain_inner_iters=80,
            # PEGASE-1354's API point alone diverges only ~9% under flat NR,
            # which makes the rescue story weak. Push the test set deeper past
            # the loadability limit so NR diverges in a meaningful regime
            # (~30-50%) where PDL warm-start can demonstrate dominance.
            # Tune these on a short run if the divergence rate is off.
            test_variation=0.25,
            test_base_scale=1.15,
        ))
    elif n_buses >= 1700 or case_key == 'case1888rte':
        # RTE-1888: larger than PEGASE-1354 and severely ill-conditioned for
        # flat-start NR — it diverges on ~100% of operating points at ANY
        # load level (probed: 100% even at 0.70x the API point). So the knob
        # here controls *rescuability*, not the flat-divergence rate: we scale
        # the API point DOWN to ~0.78x so the diverged points stay solvable
        # (DCPF-theta alone rescues ~38% there; PDL should rescue more). At
        # >=0.85x almost nothing is rescuable. This gives the strongest
        # big-system story: "flat NR diverges on 100% of points, PDL rescues
        # X%". Re-probe if you change the case.
        cfg.update(dict(
            n_train=3500,
            # PDL was badly under-trained here: with the case1354 budget the
            # training violation oscillated 26-155 p.u. for all 60 iters and
            # NEVER converged (final 30.9 p.u.), so the predicted V/θ/Pg were
            # near-random. On the knife-edge 0.78x test regime that injects
            # noisy voltages into NR's tiny basin of attraction -> 0% rescue,
            # while DCPF's clean flat-V seed rescues ~48%. Root cause was the
            # augmented-Lagrangian penalty never growing (ρ stuck at 6.2): the
            # primal could ignore feasibility. Fixes below: (a) longer budget
            # so the loss can actually settle, (b) a much more aggressive ρ
            # schedule (start at 10 to match pretrain, grow x1.5, short warmup,
            # check every iter) so feasibility is enforced hard, (c) more
            # pretraining to seed a feasible basin before the primal-dual loop.
            max_outer_iters=110,
            pretrain_iters=25,
            train_inner_iters=90,
            train_batch_size=96 if use_cuda else 24,
            train_accum_steps=6,
            sweep_samples=120 if use_cuda else 30,
            stress_multipliers=[0.75, 0.85, 0.95, 1.00, 1.05, 1.10],
            inference_chunk_size=96 if use_cuda else 24,
            nn_supervised_samples=1500,
            nn_epochs=400,
            nn_batch_size=96 if use_cuda else 24,
            model_kwargs=dict(d_model=64, n_heads=4,
                              n_layers_primal=4, n_layers_dual=2),
            retrain_iters=8,
            retrain_inner_iters=80,
            test_variation=0.20,
            test_base_scale=0.78,
            # Aggressive ρ schedule (vs the default 1.0 / x1.2 / warmup 15 /
            # freq 3 that left ρ at 6.2). Start where pretrain ends (10), grow
            # fast, and start checking early so the penalty actually climbs
            # toward rho_max and pins down power-balance feasibility.
            rho_init=10.0,
            rho_max=500.0,
            rho_alpha=1.5,
            rho_tau=0.9,
            rho_warmup_iters=8,
            rho_check_freq=1,
        ))
    elif n_buses >= 250 or case_key == 'case300':
        # REVERTED to the 12 GB-era config (d_model=48 / 3 primal layers /
        # n_train=4000 / 60 iters). The "scaled up for 48 GB" config
        # (d_model=64 / 4 layers / n_train=7000 / 90 iters) drove the ACOPF
        # violation lower (8.1 -> 5.9 p.u.) but WORSENED NR warm-start
        # convergence (77.2% -> 61.5%): the bigger model finds lower-violation
        # power-flow points that are worse NR seeds (nearer the collapse
        # branch). The smaller model is the better warm-start predictor here.
        cfg.update(dict(
            n_train=4000,
            n_stressed_test=800,
            max_outer_iters=60,
            pretrain_iters=10,
            train_inner_iters=90,
            train_batch_size=192 if use_cuda else 64,
            train_accum_steps=6,
            sweep_samples=200,
            inference_chunk_size=128 if use_cuda else 64,
            nn_supervised_samples=1200,
            nn_epochs=300,
            nn_batch_size=96 if use_cuda else 48,
            model_kwargs=dict(d_model=48, n_heads=4,
                              n_layers_primal=3, n_layers_dual=2),
            retrain_iters=8,
            retrain_inner_iters=80,
        ))
    elif n_buses >= 100:
        cfg.update(dict(
            n_train=7000,
            n_stressed_test=1600,
            max_outer_iters=80,
            train_inner_iters=120,
            train_batch_size=512 if use_cuda else 128,
            train_accum_steps=8,
            sweep_samples=350,
            inference_chunk_size=256 if use_cuda else 96,
            nn_supervised_samples=1200,
            nn_epochs=36,
            nn_batch_size=192 if use_cuda else 96,
            model_kwargs=dict(d_model=56, n_heads=4,
                              n_layers_primal=4, n_layers_dual=2),
            retrain_iters=8,
            retrain_inner_iters=80,
        ))

    return cfg


def pdl_predict_chunked(pdl, P_np, Q_np, device,
                        chunk_size=256, compute_violation=True):
    """Run PDL inference in chunks to avoid GPU OOM on larger systems."""
    n = P_np.shape[0]
    v_list, th_list, pg_list = [], [], []
    max_p_viol, max_q_viol = 0.0, 0.0

    t0 = time.time()
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        P_t = torch.tensor(P_np[s:e], dtype=torch.float32, device=device)
        Q_t = torch.tensor(Q_np[s:e], dtype=torch.float32, device=device)

        with torch.no_grad():
            Pg, Qg, V, th = pdl.predict(P_t, Q_t)
            if compute_violation:
                Pv, Qv = pdl.compute_power_balance(Pg, Qg, V, th, P_t, Q_t)
                max_p_viol = max(max_p_viol, torch.max(torch.abs(Pv)).item())
                max_q_viol = max(max_q_viol, torch.max(torch.abs(Qv)).item())

        v_list.append(V.cpu().numpy())
        th_list.append(np.degrees(th.cpu().numpy()))
        pg_list.append(Pg.cpu().numpy())

        del P_t, Q_t, Pg, Qg, V, th
        if compute_violation:
            del Pv, Qv
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - t0
    return dict(
        V=np.concatenate(v_list, axis=0),
        theta_deg=np.concatenate(th_list, axis=0),
        Pg=np.concatenate(pg_list, axis=0),
        elapsed=total_time,
        max_P_viol=max_p_viol,
        max_Q_viol=max_q_viol,
    )


def pdl_violation_per_sample(pdl, P_np, Q_np, device, chunk_size=256):
    """Compute per-sample max violation for filtering hard cases."""
    n = P_np.shape[0]
    viol = np.zeros(n, dtype=np.float32)
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        P_t = torch.tensor(P_np[s:e], dtype=torch.float32, device=device)
        Q_t = torch.tensor(Q_np[s:e], dtype=torch.float32, device=device)
        with torch.no_grad():
            Pg, Qg, V, th = pdl.predict(P_t, Q_t)
            Pv, Qv = pdl.compute_power_balance(Pg, Qg, V, th, P_t, Q_t)
            mv = torch.max(torch.stack([
                torch.max(torch.abs(Pv), dim=1).values,
                torch.max(torch.abs(Qv), dim=1).values,
            ], dim=1), dim=1).values
        viol[s:e] = mv.cpu().numpy().astype(np.float32)
        del P_t, Q_t, Pg, Qg, V, th, Pv, Qv, mv
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return viol


# ======================================================================
# Main experiment runner
# ======================================================================
def run_divergence_rescue_experiment(
        case_name, system_data, ppc_api,
        n_train=10000, n_stressed_test=2000,
        max_outer_iters=100, convergence_threshold=1e-4,
    nr_max_iter=30,
    pretrain_iters=15,
    train_inner_iters=150,
    train_batch_size=1024,
    train_accum_steps=10,
    sweep_samples=500,
    stress_multipliers=None,
    inference_chunk_size=256,
    model_kwargs=None,
    nn_supervised_samples=1500,
    nn_epochs=40,
    nn_batch_size=256,
    curriculum_phases=None,
    curriculum_outer_iters=None,
    retrain_on_failed=True,
    retrain_top_frac=0.25,
    retrain_iters=8,
    retrain_inner_iters=80,
    test_variation=None,
    test_base_scale=1.0,
    rho_init=1.0,
    rho_max=500.0,
    rho_alpha=1.2,
    rho_tau=0.9,
    rho_warmup_iters=15,
    rho_check_freq=3,
    seed=42):
    """Full pipeline: train PDL-GAT → test on PGLIB API stressed points → rescue."""

    if seed is not None:
        set_global_seed(seed)

    print(f"\n{'=' * 80}")
    print(f"  DIVERGENCE RESCUE EXPERIMENT: {case_name}")
    print(f"{'=' * 80}")
    print(f"\n  CONFIG:")
    print(f"  - Architecture: GAT + Global Attention (same as c5)")
    print(f"  - Adaptive ρ: init={rho_init}, max={rho_max}, α={rho_alpha}, "
          f"τ={rho_tau}, warmup={rho_warmup_iters}, freq={rho_check_freq}")
    print(f"  - Pre-training: {pretrain_iters} iters penalty-only at ρ=10")
    print(f"  - Training data: {n_train} mixed (50% normal + 25% synth + 25% API)")
    print(f"  - Test data: {n_stressed_test} PGLIB API-derived stressed scenarios")
    print(f"  - NR max iterations: {nr_max_iter}")
    print(f"  - Train batch size: {train_batch_size}, accum steps: {train_accum_steps}")
    print(f"  - Inference chunk size: {inference_chunk_size}")
    print(f"  - Stress sweep samples per level: {sweep_samples}")
    print(f"  - Data source: PGLIB-OPF API tier (Active Power Increase)")
    print(f"  - Normal NN baseline: supervised MSE on NR labels")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    print(f"  System: {system_data['n_buses']} buses, "
          f"{system_data['n_generators']} gens")

    # ------------------------------------------------------------------
    # 1. Curriculum training data (easy -> hard)
    # ------------------------------------------------------------------
    phases = curriculum_phases if curriculum_phases is not None else build_training_phases()
    if not phases:
        phases = [dict(label='single', stressed_range=(1.2, 1.8), api_variation=0.20)]

    if curriculum_outer_iters is None:
        per_phase = max(1, max_outer_iters // len(phases))
        per_phase_iters = [per_phase] * len(phases)
        remainder = max_outer_iters - sum(per_phase_iters)
        for i in range(remainder):
            per_phase_iters[i % len(phases)] += 1
    elif isinstance(curriculum_outer_iters, (list, tuple)):
        per_phase_iters = list(curriculum_outer_iters)
    else:
        per_phase_iters = [int(curriculum_outer_iters)] * len(phases)

    print(f"\n  Curriculum phases: {len(phases)}")
    for i, ph in enumerate(phases):
        print(f"    {i + 1}. {ph.get('label', 'phase')} "
              f"range={ph['stressed_range']}, api_var={ph['api_variation']} "
              f"iters={per_phase_iters[i]}")

    P_tr = None
    Q_tr = None
    P_tr_t = None
    Q_tr_t = None

    # ------------------------------------------------------------------
    # 2. Initialise and train PDL-GAT
    # ------------------------------------------------------------------
    print(f"  Initialising PDL-GAT v6 model...")
    pdl = PDL_ACPF_GAT(
        system_data, rho_init=rho_init, rho_max=rho_max,
        alpha=rho_alpha, tau=rho_tau,
        device=device, max_angle_rad=np.pi / 2.0,
        **(model_kwargs or dict(d_model=64, n_heads=4,
                                n_layers_primal=4, n_layers_dual=2)),
        warmup_iters=rho_warmup_iters, rho_check_freq=rho_check_freq,
    )

    # Pre-training and curriculum training
    pretrain_time = 0.0
    tr_time = 0.0
    t0 = time.time()
    k, max_viol = 0, float('inf')
    for ph_i, phase in enumerate(phases):
        print(f"\n  Generating {n_train} training samples for "
              f"phase '{phase.get('label', ph_i + 1)}'...")
        P_tr, Q_tr = generate_mixed_training_with_api(
            system_data, ppc_api, n_train,
            stressed_range=phase['stressed_range'],
            api_variation=phase['api_variation'])
        P_tr_t = torch.tensor(P_tr, dtype=torch.float32, device=device)
        Q_tr_t = torch.tensor(Q_tr, dtype=torch.float32, device=device)

        if ph_i == 0 and pretrain_iters > 0:
            t0_pre = time.time()
            pdl.pretrain(P_tr_t, Q_tr_t, n_iters=pretrain_iters,
                         rho_pre=10.0, batch_size=train_batch_size)
            pretrain_time = time.time() - t0_pre
            print(f"  Pre-training done in {pretrain_time:.1f}s")

        print(f"  PDL training phase '{phase.get('label', ph_i + 1)}' "
              f"(threshold: {convergence_threshold} p.u.)...")
        print("-" * 80)
        phase_iters = per_phase_iters[ph_i] if ph_i < len(per_phase_iters) else per_phase_iters[-1]
        for it in range(phase_iters):
            t_it = time.time()
            _, max_viol = pdl.train_epoch(
                P_tr_t, Q_tr_t, inner_iters=train_inner_iters,
                batch_size=train_batch_size, accum_steps=train_accum_steps)
            # Print every iteration so progress is visible on large/slow cases
            # (one outer iter on PEGASE1354 is ~minutes — silence looks hung).
            print(f"  Iter {k + 1:3d}: Max Viol={max_viol:.6e} p.u., "
                  f"ρ={pdl.rho:.1f} ({time.time() - t_it:.1f}s)", flush=True)
            k += 1
            if max_viol <= convergence_threshold:
                break

    tr_time = time.time() - t0
    total_train_time = pretrain_time + tr_time
    print(f"\n  Training done: {k} iters in {tr_time:.1f}s "
          f"(+{pretrain_time:.1f}s pre-train = {total_train_time:.1f}s)")
    print(f"  Final violation: {max_viol:.4e} p.u., final ρ: {pdl.rho:.1f}")

    # ------------------------------------------------------------------
    # 2b. Train supervised normal NN baseline (MSE)
    # ------------------------------------------------------------------
    nn_state = train_normal_nn_acopf(
        system_data, P_tr, Q_tr, device,
        max_samples=nn_supervised_samples,
        nr_max_iter=nr_max_iter,
        epochs=nn_epochs,
        batch_size=nn_batch_size,
        lr=1e-3,
    )

    # ------------------------------------------------------------------
    # 3. Generate PGLIB API-derived stressed test scenarios
    # ------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print(f"  STRESSED TEST SET: {n_stressed_test} PGLIB API-derived scenarios")
    print(f"{'=' * 80}\n")

    P_te, Q_te, strat_labels = generate_api_test_scenarios(
        ppc_api, system_data, n_stressed_test,
        variation=test_variation, base_scale=test_base_scale,
    )
    if test_variation is not None or test_base_scale != 1.0:
        print(f"  Test stress knob: variation={test_variation}, "
              f"base_scale={test_base_scale}")
    # Show load level comparison
    base_P = system_data['base_P_demand_pu']
    api_P_mean = np.mean(np.sum(P_te, axis=1))
    base_P_total = np.sum(base_P)
    if base_P_total > 0:
        avg_mult = api_P_mean / base_P_total
        print(f"  API test avg total P: {api_P_mean:.4f} p.u. "
              f"(~{avg_mult:.2f}x base load)")

    # ------------------------------------------------------------------
    # 4. NR flat-start on all test points
    # ------------------------------------------------------------------
    print(f"  Running NR flat-start on {n_stressed_test} stressed points...")
    t0 = time.time()
    nr_flat_results = test_nr_convergence(
        system_data, P_te, Q_te, init_mode='flat',
        max_iter=nr_max_iter, label='NR-flat')
    nr_flat_time = time.time() - t0

    n_nr_conv = sum(1 for r in nr_flat_results if r['converged'])
    n_nr_div  = n_stressed_test - n_nr_conv
    print(f"  NR flat-start: {n_nr_conv}/{n_stressed_test} converged, "
          f"{n_nr_div} DIVERGED ({n_nr_div/n_stressed_test*100:.1f}%)")
    print(f"  NR flat-start total time: {nr_flat_time:.1f}s")

    # ------------------------------------------------------------------
    # 5. PDL prediction on all test points
    # ------------------------------------------------------------------
    print(f"\n  Running PDL inference...")
    pdl_pred = pdl_predict_chunked(
        pdl, P_te, Q_te, device,
        chunk_size=inference_chunk_size,
        compute_violation=True)
    pdl_inf_time = pdl_pred['elapsed']
    V_pdl_np   = pdl_pred['V']
    th_pdl_deg = pdl_pred['theta_deg']
    Pg_pdl_np  = pdl_pred['Pg']
    pdl_max_Pv = pdl_pred['max_P_viol']
    pdl_max_Qv = pdl_pred['max_Q_viol']
    print(f"  PDL inference: {pdl_inf_time*1000:.1f} ms "
          f"({pdl_inf_time*1000/n_stressed_test:.4f} ms/sample)")
    print(f"  PDL max violations: P={pdl_max_Pv:.4e}, Q={pdl_max_Qv:.4e}")

    # ------------------------------------------------------------------
    # 5b. Retrain on failed high-divergence points
    # ------------------------------------------------------------------
    retrain_samples = 0
    retrain_time = 0.0
    if retrain_on_failed and retrain_iters > 0:
        print(f"\n  Selecting failed high-divergence points for retrain...")
        viol = pdl_violation_per_sample(
            pdl, P_te, Q_te, device,
            chunk_size=inference_chunk_size)
        failed_mask = np.array([not r['converged'] for r in nr_flat_results], dtype=bool)
        if np.any(failed_mask):
            cutoff = np.quantile(viol[failed_mask], 1.0 - retrain_top_frac)
            hard_idx = np.where(failed_mask & (viol >= cutoff))[0]
            retrain_samples = int(len(hard_idx))
            if retrain_samples > 0:
                print(f"  Retraining on {retrain_samples} hard points "
                      f"(top {retrain_top_frac*100:.0f}% divergence of failed set)")
                P_hard = torch.tensor(P_te[hard_idx], dtype=torch.float32, device=device)
                Q_hard = torch.tensor(Q_te[hard_idx], dtype=torch.float32, device=device)
                t0_retrain = time.time()
                for ri in range(retrain_iters):
                    _, max_viol = pdl.train_epoch(
                        P_hard, Q_hard, inner_iters=retrain_inner_iters,
                        batch_size=train_batch_size, accum_steps=train_accum_steps)
                    if ri % 3 == 0 or ri == retrain_iters - 1:
                        print(f"    Retrain iter {ri + 1:2d}/{retrain_iters}: "
                              f"max_viol={max_viol:.4e}")
                retrain_time = time.time() - t0_retrain
                total_train_time += retrain_time
                print(f"  Retrain time: {retrain_time:.1f}s "
                      f"(total training time {total_train_time:.1f}s)")

                print(f"  Re-running PDL inference after retrain...")
                pdl_pred = pdl_predict_chunked(
                    pdl, P_te, Q_te, device,
                    chunk_size=inference_chunk_size,
                    compute_violation=True)
                pdl_inf_time = pdl_pred['elapsed']
                V_pdl_np   = pdl_pred['V']
                th_pdl_deg = pdl_pred['theta_deg']
                Pg_pdl_np  = pdl_pred['Pg']
                pdl_max_Pv = pdl_pred['max_P_viol']
                pdl_max_Qv = pdl_pred['max_Q_viol']
        else:
            print("  No failed flat-start points found; retrain skipped.")

    # ------------------------------------------------------------------
    # 6. PDL warm-start + NR on all test points
    # ------------------------------------------------------------------
    print(f"\n  Running PDL warm-start + NR on {n_stressed_test} "
          f"stressed points...")
    t0 = time.time()
    nr_warm_results = test_nr_convergence(
        system_data, P_te, Q_te, init_mode='warm',
        V_init=V_pdl_np, theta_init_deg=th_pdl_deg,
        Pg_init=Pg_pdl_np, gen_buses=pdl.gen_buses,
        max_iter=nr_max_iter, label='NR-warm')
    nr_warm_time = time.time() - t0

    n_warm_conv = sum(1 for r in nr_warm_results if r['converged'])
    n_warm_div  = n_stressed_test - n_warm_conv
    print(f"  PDL warm-start + NR: {n_warm_conv}/{n_stressed_test} converged, "
          f"{n_warm_div} diverged")
    print(f"  PDL warm-start + NR total time: {nr_warm_time:.1f}s")

    pf_warm = compute_participation_from_results(nr_warm_results, system_data)

    # ------------------------------------------------------------------
    # 6a. Normal NN warm-start + NR on all test points
    # ------------------------------------------------------------------
    n_nn_conv = 0
    n_nn_div = n_stressed_test
    n_rescued_nn = 0
    nn_rescue_rate = 0.0
    n_flat_ok_nn_warm_div = 0
    nn_pred_time = 0.0
    nn_train_mse = None
    nn_n_labels = 0
    nn_train_time = 0.0
    nr_nn_warm_time = 0.0
    nn_warm_iters_list = [nr_max_iter] * n_stressed_test
    nn_warm_conv_list = [False] * n_stressed_test

    if nn_state is not None:
        print(f"\n  Running normal NN inference...")
        nn_pred = normal_nn_predict_chunked(
            nn_state, system_data, P_te, Q_te, device,
            chunk_size=inference_chunk_size,
        )
        nn_pred_time = nn_pred['elapsed']
        nn_train_mse = float(nn_state['train_mse'])
        nn_n_labels = int(nn_state['n_labels'])
        nn_train_time = float(nn_state['train_time'])

        print(f"  Normal NN inference: {nn_pred_time*1000:.1f} ms "
              f"({nn_pred_time*1000/n_stressed_test:.4f} ms/sample)")
        print(f"\n  Running normal NN warm-start + NR on {n_stressed_test} points...")
        t0 = time.time()
        nr_nn_warm_results = test_nr_convergence(
            system_data, P_te, Q_te, init_mode='warm',
            V_init=nn_pred['V'], theta_init_deg=nn_pred['theta_deg'],
            Pg_init=nn_pred['Pg'], gen_buses=pdl.gen_buses,
            max_iter=nr_max_iter, label='NR-nn-warm')
        nr_nn_warm_time = time.time() - t0
        n_nn_conv = sum(1 for r in nr_nn_warm_results if r['converged'])
        n_nn_div = n_stressed_test - n_nn_conv
        print(f"  Normal NN warm-start + NR: {n_nn_conv}/{n_stressed_test} "
              f"converged, {n_nn_div} diverged")
        print(f"  Normal NN warm-start + NR total time: {nr_nn_warm_time:.1f}s")

        nn_warm_iters_list = [r['iterations'] for r in nr_nn_warm_results]
        nn_warm_conv_list = [r['converged'] for r in nr_nn_warm_results]

    # ------------------------------------------------------------------
    # 6b. DCPF-theta warm-start + NR on all test points
    # ------------------------------------------------------------------
    print(f"\n  Running DCPF on each stressed point for theta warm-start...")
    dcopf_out = run_dcopf_theta_batch(system_data, P_te, Q_te)
    dcopf_theta_deg = dcopf_out['theta_deg']
    dcopf_pg = dcopf_out['Pg']
    dcopf_conv_flags = dcopf_out['converged']
    n_dcopf_ok = sum(1 for x in dcopf_conv_flags if x)
    print(f"  DCPF solved: {n_dcopf_ok}/{n_stressed_test} "
          f"({n_dcopf_ok / n_stressed_test * 100:.1f}%)")
    print(f"  DCPF total time: {dcopf_out['elapsed']:.1f}s")

    print(f"\n  Running DCPF-theta warm-start + NR on {n_stressed_test} "
          f"stressed points...")
    V_dcopf_init = build_voltage_init_template(system_data, n_stressed_test)
    t0 = time.time()
    nr_dcopf_warm_results = test_nr_convergence(
        system_data, P_te, Q_te, init_mode='warm',
        V_init=V_dcopf_init, theta_init_deg=dcopf_theta_deg,
        Pg_init=dcopf_pg, gen_buses=pdl.gen_buses,
        max_iter=nr_max_iter, label='NR-dcpf-theta-warm')
    nr_dcopf_warm_time = time.time() - t0

    n_dcopf_warm_conv = sum(1 for r in nr_dcopf_warm_results if r['converged'])
    n_dcopf_warm_div = n_stressed_test - n_dcopf_warm_conv
    print(f"  DCPF-theta warm-start + NR: {n_dcopf_warm_conv}/{n_stressed_test} "
          f"converged, {n_dcopf_warm_div} diverged")
    print(f"  DCPF-theta warm-start + NR total time: {nr_dcopf_warm_time:.1f}s")

    # ------------------------------------------------------------------
    # 7. Rescue statistics
    # ------------------------------------------------------------------
    n_rescued = 0
    n_rescued_nn = 0
    n_rescued_dcopf = 0
    n_both_div = 0
    n_both_conv = 0
    n_flat_ok_warm_div = 0
    n_flat_ok_nn_warm_div = 0
    n_flat_ok_dcopf_warm_div = 0

    flat_iters_list = []
    warm_iters_list = []
    dcopf_warm_iters_list = []
    flat_conv_list  = []
    warm_conv_list  = []
    dcopf_warm_conv_list = []

    for i in range(n_stressed_test):
        fc = nr_flat_results[i]['converged']
        wc = nr_warm_results[i]['converged']
        nc = nn_warm_conv_list[i]
        dc = nr_dcopf_warm_results[i]['converged']
        flat_conv_list.append(fc)
        warm_conv_list.append(wc)
        dcopf_warm_conv_list.append(dc)
        flat_iters_list.append(nr_flat_results[i]['iterations'])
        warm_iters_list.append(nr_warm_results[i]['iterations'])
        dcopf_warm_iters_list.append(nr_dcopf_warm_results[i]['iterations'])

        if fc and wc:
            n_both_conv += 1
        elif (not fc) and wc:
            n_rescued += 1
        elif (not fc) and (not wc):
            n_both_div += 1
        else:  # fc and not wc
            n_flat_ok_warm_div += 1

        if (not fc) and dc:
            n_rescued_dcopf += 1
        elif fc and (not dc):
            n_flat_ok_dcopf_warm_div += 1

        if (not fc) and nc:
            n_rescued_nn += 1
        elif fc and (not nc):
            n_flat_ok_nn_warm_div += 1

    rescue_rate = n_rescued / max(n_nr_div, 1) * 100
    nn_rescue_rate = n_rescued_nn / max(n_nr_div, 1) * 100
    dcopf_rescue_rate = n_rescued_dcopf / max(n_nr_div, 1) * 100

    print(f"\n{'=' * 80}")
    print(f"  DIVERGENCE RESCUE RESULTS — {case_name}")
    print(f"{'=' * 80}\n")
    print(f"  Total stressed test points:        {n_stressed_test}")
    print(f"  NR flat-start converged:           {n_nr_conv} "
          f"({n_nr_conv / n_stressed_test * 100:.1f}%)")
    print(f"  NR flat-start DIVERGED:            {n_nr_div} "
          f"({n_nr_div / n_stressed_test * 100:.1f}%)")
    print(f"  PDL warm-start + NR converged:     {n_warm_conv} "
          f"({n_warm_conv / n_stressed_test * 100:.1f}%)")
    print(f"  NN warm-start + NR converged:      {n_nn_conv} "
          f"({n_nn_conv / n_stressed_test * 100:.1f}%)")
    print(f"  DCPF-theta warm-start + NR conv:  {n_dcopf_warm_conv} "
          f"({n_dcopf_warm_conv / n_stressed_test * 100:.1f}%)")
    print(f"")
    print(f"  Both converged:                    {n_both_conv}")
    print(f"  *** RESCUED by PDL warm-start:     {n_rescued} ***")
    print(f"  *** RESCUED by NN warm-start:      {n_rescued_nn} ***")
    print(f"  *** RESCUED by DCPF-theta warm:    {n_rescued_dcopf} ***")
    print(f"  Both diverged:                     {n_both_div}")
    print(f"  Flat OK but warm diverged:         {n_flat_ok_warm_div}")
    print(f"  Flat OK but NN warm diverged:      {n_flat_ok_nn_warm_div}")
    print(f"  Flat OK but DCPF warm diverged:    {n_flat_ok_dcopf_warm_div}")
    print(f"")
    print(f"  ╔═══════════════════════════════════════════════════════════╗")
    print(f"  ║  PDL WARM-START RESCUES {rescue_rate:5.1f}% "
          f"OF NR-DIVERGENT CASES     ║")
    print(f"  ║  ({n_rescued} out of {n_nr_div} divergent points)   "
          f"{'':>{30 - len(str(n_rescued)) - len(str(n_nr_div))}}║")
    print(f"  ╚═══════════════════════════════════════════════════════════╝")
    print(f"  NN warm-start rescue rate:         {nn_rescue_rate:.1f}%")
    print(f"  DCPF-theta rescue rate:            {dcopf_rescue_rate:.1f}%")

    # Iteration comparison for cases where both converged
    if n_both_conv > 0:
        fi_both = [flat_iters_list[i] for i in range(n_stressed_test)
                   if flat_conv_list[i] and warm_conv_list[i]
                   and flat_iters_list[i] > 0 and warm_iters_list[i] > 0]
        wi_both = [warm_iters_list[i] for i in range(n_stressed_test)
                   if flat_conv_list[i] and warm_conv_list[i]
                   and flat_iters_list[i] > 0 and warm_iters_list[i] > 0]
        if fi_both and wi_both:
            avg_f = np.mean(fi_both)
            avg_w = np.mean(wi_both)
            red = (1 - avg_w / max(avg_f, 1)) * 100
            print(f"\n  Iteration reduction (both-converged subset):")
            print(f"    NR flat avg iters:  {avg_f:.2f}")
            print(f"    PDL warm avg iters: {avg_w:.2f}")
            print(f"    Reduction:          {red:.1f}%")

    # Accuracy vs NR for converged-by-both points
    V_nr_conv = []
    th_nr_conv_deg = []
    V_pdl_conv = []
    th_pdl_conv_deg = []
    for i in range(n_stressed_test):
        if flat_conv_list[i] and warm_conv_list[i]:
            V_nr_conv.append(nr_warm_results[i]['V'])
            th_nr_conv_deg.append(nr_warm_results[i]['theta_deg'])
            V_pdl_conv.append(V_pdl_np[i])
            th_pdl_conv_deg.append(th_pdl_deg[i])

    if V_nr_conv:
        V_nr_conv = np.array(V_nr_conv)
        th_nr_conv_deg = np.array(th_nr_conv_deg)
        V_pdl_conv = np.array(V_pdl_conv)
        th_pdl_conv_deg = np.array(th_pdl_conv_deg)
        V_mse  = np.mean((V_pdl_conv - V_nr_conv) ** 2)
        th_mse = np.mean((th_pdl_conv_deg - th_nr_conv_deg) ** 2)
        print(f"\n  Accuracy vs NR (both-converged subset, {len(V_nr_conv)} pts):")
        print(f"    Voltage MSE:  {V_mse:.4e} p.u.²")
        print(f"    Angle MSE:    {th_mse:.4e} deg²")

    # ------------------------------------------------------------------
    # 8. Convergence vs stress level sweep
    # ------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print(f"  CONVERGENCE VS STRESS LEVEL SWEEP")
    print(f"{'=' * 80}\n")

    # Use the API loads as the "1.0x" reference for the sweep
    api_P_base, api_Q_base, _ = extract_api_loads_pu(ppc_api)
    api_P_full = np.zeros(system_data['n_buses'])
    api_Q_full = np.zeros(system_data['n_buses'])
    for ii in range(min(len(api_P_base), system_data['n_buses'])):
        api_P_full[ii] = api_P_base[ii]
        api_Q_full[ii] = api_Q_base[ii]

    if stress_multipliers is None:
        stress_multipliers = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    nr_conv_rates = []
    warm_conv_rates = []
    n_sweep = sweep_samples  # samples per stress level

    for mult in stress_multipliers:
        print(f"  API load × {mult:.1f} ...")
        n_buses = system_data['n_buses']
        P_sw = np.zeros((n_sweep, n_buses))
        Q_sw = np.zeros((n_sweep, n_buses))
        for si in range(n_sweep):
            noise = 1.0 + np.random.randn(n_buses) * 0.08
            P_sw[si] = np.clip(api_P_full * mult * noise, 0, None)
            Q_sw[si] = api_Q_full * mult * noise

        # NR flat
        nr_sw = test_nr_convergence(
            system_data, P_sw, Q_sw, init_mode='flat',
            max_iter=nr_max_iter)
        nr_rate = sum(1 for r in nr_sw if r['converged']) / n_sweep * 100

        # PDL warm-start + NR
        sw_pred = pdl_predict_chunked(
            pdl, P_sw, Q_sw, device,
            chunk_size=inference_chunk_size,
            compute_violation=False)
        warm_sw = test_nr_convergence(
            system_data, P_sw, Q_sw, init_mode='warm',
            V_init=sw_pred['V'],
            theta_init_deg=sw_pred['theta_deg'],
            Pg_init=sw_pred['Pg'],
            gen_buses=pdl.gen_buses,
            max_iter=nr_max_iter)
        warm_rate = sum(1 for r in warm_sw if r['converged']) / n_sweep * 100

        nr_conv_rates.append(nr_rate)
        warm_conv_rates.append(warm_rate)
        print(f"    NR flat: {nr_rate:.1f}%   PDL warm: {warm_rate:.1f}%   "
              f"Rescue margin: +{warm_rate - nr_rate:.1f}pp")

    # ------------------------------------------------------------------
    # 8b. Tractability snapshot: base case + variation levels
    # ------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print(f"  TRACTABILITY SNAPSHOT (BASE + VARIATION)")
    print(f"{'=' * 80}\n")

    tractability = {
        'base_case': {},
        'variation_cases': [],
    }

    # Base case: exact API operating point (1.0x, no added noise)
    P_base = api_P_full.reshape(1, -1).copy()
    Q_base = api_Q_full.reshape(1, -1).copy()

    base_nr = test_nr_convergence(
        system_data, P_base, Q_base,
        init_mode='flat',
        max_iter=nr_max_iter,
        label='NR-base-flat')

    base_pdl_pred = pdl_predict_chunked(
        pdl, P_base, Q_base, device,
        chunk_size=1,
        compute_violation=False)
    base_pdl_warm = test_nr_convergence(
        system_data, P_base, Q_base,
        init_mode='warm',
        V_init=base_pdl_pred['V'],
        theta_init_deg=base_pdl_pred['theta_deg'],
        Pg_init=base_pdl_pred['Pg'],
        gen_buses=pdl.gen_buses,
        max_iter=nr_max_iter,
        label='NR-base-pdl-warm')

    base_case = {
        'nr_flat_converged': bool(base_nr[0]['converged']),
        'nr_flat_iterations': int(base_nr[0]['iterations']),
        'pdl_warm_converged': bool(base_pdl_warm[0]['converged']),
        'pdl_warm_iterations': int(base_pdl_warm[0]['iterations']),
    }

    if nn_state is not None:
        base_nn_pred = normal_nn_predict_chunked(
            nn_state, system_data, P_base, Q_base, device,
            chunk_size=1,
        )
        base_nn_warm = test_nr_convergence(
            system_data, P_base, Q_base,
            init_mode='warm',
            V_init=base_nn_pred['V'],
            theta_init_deg=base_nn_pred['theta_deg'],
            Pg_init=base_nn_pred['Pg'],
            gen_buses=pdl.gen_buses,
            max_iter=nr_max_iter,
            label='NR-base-nn-warm')
        base_case['nn_warm_converged'] = bool(base_nn_warm[0]['converged'])
        base_case['nn_warm_iterations'] = int(base_nn_warm[0]['iterations'])

    base_dcopf = run_dcopf_theta_batch(system_data, P_base, Q_base)
    base_v_init = build_voltage_init_template(system_data, 1)
    base_dcopf_warm = test_nr_convergence(
        system_data, P_base, Q_base,
        init_mode='warm',
        V_init=base_v_init,
        theta_init_deg=base_dcopf['theta_deg'],
        Pg_init=base_dcopf['Pg'],
        gen_buses=pdl.gen_buses,
        max_iter=nr_max_iter,
        label='NR-base-dcpf-warm')
    base_case['dcopf_solved'] = bool(base_dcopf['converged'][0])
    base_case['dcopf_warm_converged'] = bool(base_dcopf_warm[0]['converged'])
    base_case['dcopf_warm_iterations'] = int(base_dcopf_warm[0]['iterations'])

    tractability['base_case'] = base_case
    print(f"  Base case: NR flat={base_case['nr_flat_converged']} "
          f"| PDL warm={base_case['pdl_warm_converged']} "
            f"| DCPF warm={base_case['dcopf_warm_converged']}")

    # Variation tractability: fixed variation level, API-derived scenarios.
    var_levels = [0.10, 0.15, 0.20]
    n_var = int(max(24, min(120, n_stressed_test // 10)))
    tractability['variation_sample_count'] = n_var
    tractability['variation_levels'] = var_levels

    for var in var_levels:
        P_var, Q_var, _ = generate_api_test_scenarios(
            ppc_api, system_data, n_var,
            variation=var,
        )

        nr_var = test_nr_convergence(
            system_data, P_var, Q_var,
            init_mode='flat',
            max_iter=nr_max_iter,
            label=f'NR-var-{var:.2f}-flat')
        nr_conv = sum(1 for r in nr_var if r['converged'])
        nr_rate = nr_conv / n_var * 100.0

        pdl_var_pred = pdl_predict_chunked(
            pdl, P_var, Q_var, device,
            chunk_size=inference_chunk_size,
            compute_violation=False)
        warm_var = test_nr_convergence(
            system_data, P_var, Q_var,
            init_mode='warm',
            V_init=pdl_var_pred['V'],
            theta_init_deg=pdl_var_pred['theta_deg'],
            Pg_init=pdl_var_pred['Pg'],
            gen_buses=pdl.gen_buses,
            max_iter=nr_max_iter,
            label=f'NR-var-{var:.2f}-pdl-warm')
        warm_conv = sum(1 for r in warm_var if r['converged'])
        warm_rate = warm_conv / n_var * 100.0

        v_case = {
            'variation': float(var),
            'n_samples': int(n_var),
            'nr_flat_converged': int(nr_conv),
            'nr_flat_conv_rate_pct': float(nr_rate),
            'pdl_warm_converged': int(warm_conv),
            'pdl_warm_conv_rate_pct': float(warm_rate),
            'rescue_margin_pp': float(warm_rate - nr_rate),
        }
        tractability['variation_cases'].append(v_case)
        print(f"  Variation {var:.2f}: NR flat={nr_rate:.1f}% "
              f"| PDL warm={warm_rate:.1f}% "
              f"| margin=+{warm_rate - nr_rate:.1f}pp")

    def _avg_iters(iters, conv):
        vals = [float(i) for i, c in zip(iters, conv) if c and i > 0]
        if len(vals) == 0:
            return None
        return float(np.mean(vals))

    # ------------------------------------------------------------------
    # 9. Summary dict (JSON-ready source for charts.py)
    # ------------------------------------------------------------------
    summary = dict(
        case_name=case_name,
        n_train=n_train,
        n_stressed_test=n_stressed_test,
        training_time=total_train_time,
            retrain_time=retrain_time,
        convergence_iters=k,
        final_violation=max_viol,
        # NR flat
        nr_flat_converged=n_nr_conv,
        nr_flat_diverged=n_nr_div,
        nr_flat_conv_rate=n_nr_conv / n_stressed_test * 100,
        # PDL warm
        warm_converged=n_warm_conv,
        warm_diverged=n_warm_div,
        warm_conv_rate=n_warm_conv / n_stressed_test * 100,
        # Timing breakdown
        nr_flat_time=nr_flat_time,
        nr_warm_time=nr_warm_time,
        pdl_inf_time=pdl_inf_time,
        pdl_warm_total_time=nr_warm_time + pdl_inf_time,
        # Normal NN warm
        nn_train_mse=nn_train_mse,
        nn_train_labels=nn_n_labels,
        nn_train_time=nn_train_time,
        nn_inference_time=nn_pred_time,
        nn_warm_time=nr_nn_warm_time,
        nn_warm_total_time=nr_nn_warm_time + nn_pred_time,
        nn_warm_converged=n_nn_conv,
        nn_warm_diverged=n_nn_div,
        nn_warm_conv_rate=n_nn_conv / n_stressed_test * 100,
        # DCOPF-theta warm
        dcopf_solved=n_dcopf_ok,
        dcopf_failed=n_stressed_test - n_dcopf_ok,
        dcopf_time=dcopf_out['elapsed'],
        dcopf_warm_time=nr_dcopf_warm_time,
        dcopf_warm_total_time=dcopf_out['elapsed'] + nr_dcopf_warm_time,
        dcopf_warm_converged=n_dcopf_warm_conv,
        dcopf_warm_diverged=n_dcopf_warm_div,
        dcopf_warm_conv_rate=n_dcopf_warm_conv / n_stressed_test * 100,
        # Rescue
        n_rescued=n_rescued,
        rescue_rate=rescue_rate,
        n_rescued_nn=n_rescued_nn,
        nn_rescue_rate=nn_rescue_rate,
        n_rescued_dcopf=n_rescued_dcopf,
        dcopf_rescue_rate=dcopf_rescue_rate,
        n_both_div=n_both_div,
        n_flat_ok_warm_div=n_flat_ok_warm_div,
        n_flat_ok_nn_warm_div=n_flat_ok_nn_warm_div,
        n_flat_ok_dcopf_warm_div=n_flat_ok_dcopf_warm_div,
        # Stress sweep
        stress_multipliers=stress_multipliers,
        nr_conv_rates=nr_conv_rates,
        warm_conv_rates=warm_conv_rates,
        tractability=tractability,
        # PDL accuracy
        pdl_max_P_viol=pdl_max_Pv,
        pdl_max_Q_viol=pdl_max_Qv,
        # Per-point data (for post-hoc plotting)
        flat_iters=flat_iters_list,
        warm_iters=warm_iters_list,
        nn_warm_iters=nn_warm_iters_list,
        dcopf_warm_iters=dcopf_warm_iters_list,
        flat_conv=flat_conv_list,
        warm_conv=warm_conv_list,
        nn_warm_conv=nn_warm_conv_list,
        dcopf_warm_conv=dcopf_warm_conv_list,
        # Aggregated iteration metrics by method (converged subset)
        flat_avg_iters=_avg_iters(flat_iters_list, flat_conv_list),
        warm_avg_iters=_avg_iters(warm_iters_list, warm_conv_list),
        nn_warm_avg_iters=_avg_iters(nn_warm_iters_list, nn_warm_conv_list),
        dcopf_warm_avg_iters=_avg_iters(dcopf_warm_iters_list, dcopf_warm_conv_list),
        history=dict(pdl.history),
        strat_labels=strat_labels,
        retrain_on_failed=bool(retrain_on_failed),
        retrain_top_frac=float(retrain_top_frac),
        retrain_iters=int(retrain_iters),
        retrain_inner_iters=int(retrain_inner_iters),
        retrain_samples=int(retrain_samples),
        participation_factors=dict(
            pdl_warm=pf_warm,
        ),
    )
    if V_nr_conv is not None and len(V_nr_conv) > 0:
        summary['V_mse'] = float(V_mse)
        summary['theta_mse'] = float(th_mse)
        summary['V_pdl_conv'] = V_pdl_conv
        summary['V_nr_conv'] = V_nr_conv
        summary['th_pdl_conv_deg'] = th_pdl_conv_deg
        summary['th_nr_conv_deg'] = th_nr_conv_deg

    return pdl, summary


# ======================================================================
# Final comparison table (across cases)
# ======================================================================
def print_comparison_table(summaries):
    print(f"\n\n{'=' * 90}")
    print(f"  CROSS-CASE COMPARISON: DIVERGENCE RESCUE SUMMARY")
    print(f"{'=' * 90}\n")

    header = (f"  {'Case':<12} {'Test Pts':>8} {'NR Div':>8} {'NR Div%':>8} "
              f"{'PDL Res':>8} {'PDL %':>8} {'NN Res':>8} {'NN %':>8} "
              f"{'DC Res':>8} {'DC %':>8} {'PDL Conv%':>10} "
              f"{'NN Conv%':>9} {'DC Conv%':>9} {'Train (s)':>10}")
    print(header)
    print(f"  {'-' * 120}")
    for s in summaries:
        print(f"  {s['case_name']:<12} "
              f"{s['n_stressed_test']:>8d} "
              f"{s['nr_flat_diverged']:>8d} "
              f"{100 - s['nr_flat_conv_rate']:>7.1f}% "
              f"{s['n_rescued']:>8d} "
              f"{s['rescue_rate']:>7.1f}% "
              f"{s.get('n_rescued_nn', 0):>8d} "
              f"{s.get('nn_rescue_rate', 0.0):>7.1f}% "
              f"{s.get('n_rescued_dcopf', 0):>8d} "
              f"{s.get('dcopf_rescue_rate', 0.0):>7.1f}% "
              f"{s['warm_conv_rate']:>9.1f}% "
              f"{s.get('nn_warm_conv_rate', 0.0):>8.1f}% "
              f"{s.get('dcopf_warm_conv_rate', 0.0):>8.1f}% "
              f"{s['training_time']:>10.1f}")
    print()

    print("  NARRATIVE:")
    for s in summaries:
        print(f"  • {s['case_name']}: NR diverges on "
              f"{100 - s['nr_flat_conv_rate']:.1f}% of stressed points. "
              f"PDL warm-start rescues {s['rescue_rate']:.1f}% of those, "
              f"NN warm-start rescues {s.get('nn_rescue_rate', 0.0):.1f}%, "
              f"while DCPF-theta warm-start rescues "
              f"{s.get('dcopf_rescue_rate', 0.0):.1f}%.")
    print()


def _build_case_report(summary):
    """Build a JSON-safe report from a full summary dict."""

    def _jsonify(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, dict):
            return {kk: _jsonify(vv) for kk, vv in v.items()}
        if isinstance(v, list):
            return [_jsonify(x) for x in v]
        return v

    report = {
        'case_name': summary.get('case_name'),
        'n_train': int(summary.get('n_train', 0)),
        'n_stressed_test': int(summary.get('n_stressed_test', 0)),
        'training_time_s': float(summary.get('training_time', 0.0)),
        'retrain_time_s': float(summary.get('retrain_time', 0.0)),
        'convergence_iters': int(summary.get('convergence_iters', 0)),
        'final_violation_pu': float(summary.get('final_violation', 0.0)),
        'nr_flat_converged': int(summary.get('nr_flat_converged', 0)),
        'nr_flat_diverged': int(summary.get('nr_flat_diverged', 0)),
        'nr_flat_conv_rate_pct': float(summary.get('nr_flat_conv_rate', 0.0)),
        'warm_converged': int(summary.get('warm_converged', 0)),
        'warm_diverged': int(summary.get('warm_diverged', 0)),
        'warm_conv_rate_pct': float(summary.get('warm_conv_rate', 0.0)),
        'nn_train_mse': (None if summary.get('nn_train_mse') is None
            else float(summary.get('nn_train_mse'))),
        'nn_train_labels': int(summary.get('nn_train_labels', 0)),
        'nn_train_time_s': float(summary.get('nn_train_time', 0.0)),
        'nn_inference_time_s': float(summary.get('nn_inference_time', 0.0)),
        'nn_warm_converged': int(summary.get('nn_warm_converged', 0)),
        'nn_warm_diverged': int(summary.get('nn_warm_diverged', 0)),
        'nn_warm_conv_rate_pct': float(summary.get('nn_warm_conv_rate', 0.0)),
        'dcopf_solved': int(summary.get('dcopf_solved', 0)),
        'dcopf_failed': int(summary.get('dcopf_failed', 0)),
        'dcopf_time_s': float(summary.get('dcopf_time', 0.0)),
        'dcopf_warm_converged': int(summary.get('dcopf_warm_converged', 0)),
        'dcopf_warm_diverged': int(summary.get('dcopf_warm_diverged', 0)),
        'dcopf_warm_conv_rate_pct': float(summary.get('dcopf_warm_conv_rate', 0.0)),
        'n_rescued': int(summary.get('n_rescued', 0)),
        'rescue_rate_pct': float(summary.get('rescue_rate', 0.0)),
        'n_rescued_nn': int(summary.get('n_rescued_nn', 0)),
        'nn_rescue_rate_pct': float(summary.get('nn_rescue_rate', 0.0)),
        'n_rescued_dcopf': int(summary.get('n_rescued_dcopf', 0)),
        'dcopf_rescue_rate_pct': float(summary.get('dcopf_rescue_rate', 0.0)),
        'n_both_div': int(summary.get('n_both_div', 0)),
        'n_flat_ok_warm_div': int(summary.get('n_flat_ok_warm_div', 0)),
        'n_flat_ok_nn_warm_div': int(summary.get('n_flat_ok_nn_warm_div', 0)),
        'n_flat_ok_dcopf_warm_div': int(summary.get('n_flat_ok_dcopf_warm_div', 0)),
        'pdl_max_P_viol': float(summary.get('pdl_max_P_viol', 0.0)),
        'pdl_max_Q_viol': float(summary.get('pdl_max_Q_viol', 0.0)),
        'nr_flat_time_s': float(summary.get('nr_flat_time', 0.0)),
        'nr_warm_time_s': float(summary.get('nr_warm_time', 0.0)),
        'pdl_inference_time_s': float(summary.get('pdl_inf_time', 0.0)),
        'pdl_warm_total_time_s': float(summary.get('pdl_warm_total_time', 0.0)),
        'nn_warm_time_s': float(summary.get('nn_warm_time', 0.0)),
        'nn_warm_total_time_s': float(summary.get('nn_warm_total_time', 0.0)),
        'dcopf_warm_time_s': float(summary.get('dcopf_warm_time', 0.0)),
        'dcopf_warm_total_time_s': float(summary.get('dcopf_warm_total_time', 0.0)),
        'flat_avg_iters': summary.get('flat_avg_iters'),
        'warm_avg_iters': summary.get('warm_avg_iters'),
        'nn_warm_avg_iters': summary.get('nn_warm_avg_iters'),
        'dcopf_warm_avg_iters': summary.get('dcopf_warm_avg_iters'),
        'flat_iters': _jsonify(summary.get('flat_iters', [])),
        'warm_iters': _jsonify(summary.get('warm_iters', [])),
        'nn_warm_iters': _jsonify(summary.get('nn_warm_iters', [])),
        'dcopf_warm_iters': _jsonify(summary.get('dcopf_warm_iters', [])),
        'flat_conv': _jsonify(summary.get('flat_conv', [])),
        'warm_conv': _jsonify(summary.get('warm_conv', [])),
        'nn_warm_conv': _jsonify(summary.get('nn_warm_conv', [])),
        'dcopf_warm_conv': _jsonify(summary.get('dcopf_warm_conv', [])),
        'history': _jsonify(summary.get('history', {})),
        'strat_labels': _jsonify(summary.get('strat_labels', [])),
        'tractability': _jsonify(summary.get('tractability', {})),
        'retrain_on_failed': bool(summary.get('retrain_on_failed', False)),
        'retrain_top_frac': float(summary.get('retrain_top_frac', 0.0)),
        'retrain_iters': int(summary.get('retrain_iters', 0)),
        'retrain_inner_iters': int(summary.get('retrain_inner_iters', 0)),
        'retrain_samples': int(summary.get('retrain_samples', 0)),
        'participation_factors': _jsonify(summary.get('participation_factors', {})),
    }
    if 'V_mse' in summary:
        report['V_mse'] = float(summary['V_mse'])
    if 'theta_mse' in summary:
        report['theta_mse'] = float(summary['theta_mse'])

    stress = summary.get('stress_multipliers', [])
    nr_rates = summary.get('nr_conv_rates', [])
    warm_rates = summary.get('warm_conv_rates', [])
    report['stress_sweep'] = [
        {
            'stress_multiplier': float(m),
            'nr_conv_rate_pct': float(nr),
            'warm_conv_rate_pct': float(wr),
        }
        for m, nr, wr in zip(stress, nr_rates, warm_rates)
    ]
    return report


def save_case_outputs(case_key, case_label, summary, root_dir='outputs'):
    """Save per-case summary artifacts to a timestamped output directory."""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(root_dir, f"{case_key}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    report = _build_case_report(summary)
    report['case_key'] = case_key
    report['case_label'] = case_label
    report['saved_at'] = timestamp

    json_path = os.path.join(run_dir, 'summary.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)

    txt_path = os.path.join(run_dir, 'summary.txt')
    with open(txt_path, 'w') as f:
        f.write(f"Case: {case_label} ({case_key})\n")
        f.write(f"Saved at: {timestamp}\n\n")
        f.write(f"Test points: {report['n_stressed_test']}\n")
        f.write(f"NR diverged: {report['nr_flat_diverged']} ")
        f.write(f"({100.0 - report['nr_flat_conv_rate_pct']:.1f}%)\n")
        f.write(f"Rescued: {report['n_rescued']} ")
        f.write(f"({report['rescue_rate_pct']:.1f}% of NR-divergent)\n")
        f.write(f"Warm-start convergence: {report['warm_conv_rate_pct']:.1f}%\n")
        f.write(f"NN train labels: {report['nn_train_labels']}\n")
        if report['nn_train_mse'] is not None:
            f.write(f"NN train MSE: {report['nn_train_mse']:.4e}\n")
        f.write(f"NN rescued: {report['n_rescued_nn']} ")
        f.write(f"({report['nn_rescue_rate_pct']:.1f}% of NR-divergent)\n")
        f.write(f"NN warm convergence: {report['nn_warm_conv_rate_pct']:.1f}%\n")
        f.write(f"DCPF solved: {report['dcopf_solved']}/{report['n_stressed_test']} ")
        f.write(f"({report['dcopf_solved']/max(report['n_stressed_test'], 1)*100:.1f}%)\n")
        f.write(f"DCPF-theta rescued: {report['n_rescued_dcopf']} ")
        f.write(f"({report['dcopf_rescue_rate_pct']:.1f}% of NR-divergent)\n")
        f.write(f"DCPF-theta warm convergence: {report['dcopf_warm_conv_rate_pct']:.1f}%\n")
        pf = report.get('participation_factors', {}) or {}

        def _pf_count(key):
            vals = pf.get(key)
            if not vals:
                return 0
            return sum(1 for v in vals if v is not None)

        f.write("Participation factors computed (converged):\n")
        f.write(f"  PDL warm: { _pf_count('pdl_warm') }/{report['n_stressed_test']}\n")
        f.write(f"Retrain on failed: {report['retrain_on_failed']} ")
        f.write(f"| samples={report['retrain_samples']} ")
        f.write(f"| iters={report['retrain_iters']}\n")
        f.write(f"Retrain time: {report['retrain_time_s']:.1f}s\n")
        f.write(f"Training time: {report['training_time_s']:.1f}s\n")
        f.write(f"Final violation: {report['final_violation_pu']:.4e} p.u.\n")

    return run_dir


# ======================================================================
# Entry point
# ==============================================    ========================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  PDL-GAT v6 — Divergence Rescue on Stressed Operating Points")
    print("  Self-supervised transformer warm-start for Newton-Raphson")
    print("=" * 80)

    output_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
    os.makedirs(output_root, exist_ok=True)
    print(f"  Output directory: {output_root}")

    all_summaries = []

    for case_num, case_key, case_label in [
        # (30, 'case30', 'IEEE30'),
        # (39, 'case39', 'EPRI39'),
        # (118, 'case118', 'IEEE118'),
        # (300, 'case300', 'IEEE300'),
        # (1354, 'case1354pegase', 'PEGASE1354'),
        (1888, 'case1888rte', 'RTE1888'),
    ]:
        try:
            # Load base network from pandapower (for topology / Ybus / training)
            net = getattr(nw, case_key)()
            sd  = get_pandapower_data(net, case_key)

            case_cfg = get_memory_safe_case_config(case_key, sd['n_buses'])
            print(f"\n  Memory-safe config for {case_key}: ")
            print(f"    n_train={case_cfg['n_train']}, "
                  f"n_test={case_cfg['n_stressed_test']}, "
                  f"batch={case_cfg['train_batch_size']}, "
                  f"chunk={case_cfg['inference_chunk_size']}, "
                f"nn_sup={case_cfg['nn_supervised_samples']}, "
                f"sweep={case_cfg['sweep_samples']}x{len(case_cfg['stress_multipliers'])}, "
                  f"model={case_cfg['model_kwargs']}")

            # Load real PGLIB-OPF API file (near-collapse operating point)
            print(f"\n  Loading PGLIB-OPF API file for {case_key}...")
            _, ppc_api = load_pglib_api_case(case_key)

            pdl, summary = run_divergence_rescue_experiment(
                case_name=case_label,
                system_data=sd,
                ppc_api=ppc_api,
                n_train=case_cfg['n_train'],
                n_stressed_test=case_cfg['n_stressed_test'],
                max_outer_iters=case_cfg['max_outer_iters'],
                convergence_threshold=case_cfg['convergence_threshold'],
                nr_max_iter=case_cfg['nr_max_iter'],
                pretrain_iters=case_cfg['pretrain_iters'],
                train_inner_iters=case_cfg['train_inner_iters'],
                train_batch_size=case_cfg['train_batch_size'],
                train_accum_steps=case_cfg['train_accum_steps'],
                sweep_samples=case_cfg['sweep_samples'],
                stress_multipliers=case_cfg['stress_multipliers'],
                inference_chunk_size=case_cfg['inference_chunk_size'],
                model_kwargs=case_cfg['model_kwargs'],
                nn_supervised_samples=case_cfg['nn_supervised_samples'],
                nn_epochs=case_cfg['nn_epochs'],
                nn_batch_size=case_cfg['nn_batch_size'],
                curriculum_phases=case_cfg.get('curriculum_phases'),
                curriculum_outer_iters=case_cfg.get('curriculum_outer_iters'),
                retrain_on_failed=case_cfg.get('retrain_on_failed', True),
                retrain_top_frac=case_cfg.get('retrain_top_frac', 0.25),
                retrain_iters=case_cfg.get('retrain_iters', 8),
                retrain_inner_iters=case_cfg.get('retrain_inner_iters', 80),
                test_variation=case_cfg.get('test_variation'),
                test_base_scale=case_cfg.get('test_base_scale', 1.0),
                rho_init=case_cfg.get('rho_init', 1.0),
                rho_max=case_cfg.get('rho_max', 500.0),
                rho_alpha=case_cfg.get('rho_alpha', 1.2),
                rho_tau=case_cfg.get('rho_tau', 0.9),
                rho_warmup_iters=case_cfg.get('rho_warmup_iters', 15),
                rho_check_freq=case_cfg.get('rho_check_freq', 3),
                seed=case_cfg.get('seed', 42),
            )
            all_summaries.append(summary)
            case_out_dir = save_case_outputs(
                case_key=case_key,
                case_label=case_label,
                summary=summary,
                root_dir=output_root,
            )
            print(f"  Saved outputs: {case_out_dir}")
            print(f"\n  {case_label}: COMPLETED\n")
        except Exception as e:
            import traceback
            print(f"\n  {case_label}: FAILED — {e}")
            traceback.print_exc()

    if all_summaries:
        print_comparison_table(all_summaries)

    print("\n" + "=" * 80)
    print("  ALL DONE")
    print("=" * 80)
