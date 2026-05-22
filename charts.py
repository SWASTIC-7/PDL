import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import os
import json
import glob
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colormaps

# ── Try to load optional deps for extra polish ──────────────────────────────
try:
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ═══════════════════════════════════════════════════════════════════
# 1.  GLOBAL STYLE  (machinelearningplus-inspired)
# ═══════════════════════════════════════════════════════════════════
_BG       = '#FAFAFA'
_GRID     = '#E0E0E0'
_TEXT     = '#2D2D2D'
_ACCENT1  = '#1D6FD6'   # dashboard blue
_ACCENT2  = '#18C4D9'   # dashboard cyan
_ACCENT3  = '#0C5DB1'   # deep blue
_ACCENT4  = '#59D3E8'   # light cyan
_ACCENT5  = '#6F8EB8'   # muted blue gray
_ACCENT6  = '#9AAFC8'   # slate blue gray

_PALETTE = [_ACCENT1, _ACCENT2, _ACCENT3, _ACCENT4, _ACCENT5, _ACCENT6]

plt.rcdefaults()
plt.rcParams.update({
    # figure
    'figure.facecolor':   _BG,
    'figure.edgecolor':   'none',
    'figure.dpi':         150,
    # axes
    'axes.facecolor':     '#FFFFFF',
    'axes.edgecolor':     '#CCCCCC',
    'axes.linewidth':     0.8,
    'axes.grid':          True,
    'axes.titlesize':     16,
    'axes.titleweight':   'bold',
    'axes.titlepad':      14,
    'axes.labelsize':     13,
    'axes.labelweight':   'medium',
    'axes.labelpad':      8,
    'axes.prop_cycle':    plt.cycler(color=_PALETTE),
    # grid
    'grid.color':         _GRID,
    'grid.linewidth':     0.6,
    'grid.linestyle':     '--',
    'grid.alpha':         0.7,
    # ticks
    'xtick.labelsize':    11,
    'ytick.labelsize':    11,
    'xtick.direction':    'out',
    'ytick.direction':    'out',
    'xtick.major.size':   4,
    'ytick.major.size':   4,
    'xtick.major.pad':    5,
    'ytick.major.pad':    5,
    'xtick.color':        _TEXT,
    'ytick.color':        _TEXT,
    # text & font
    'text.color':         _TEXT,
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Segoe UI', 'Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size':          12,
    # legend
    'legend.frameon':     True,
    'legend.framealpha':  0.85,
    'legend.edgecolor':   '#CCCCCC',
    'legend.fontsize':    11,
    'legend.title_fontsize': 12,
    'legend.borderpad':   0.6,
    'legend.handlelength': 1.8,
    # lines
    'lines.linewidth':    2.2,
    'lines.markersize':   7,
    # savefig
    'savefig.dpi':        200,
    'savefig.bbox':       'tight',
    'savefig.facecolor':  _BG,
})


def _subtitle(ax, text, y=-0.14, fontsize=10):
    """Add a light subtitle below an axes."""
    ax.text(0.5, y, text, transform=ax.transAxes,
            ha='center', va='top', fontsize=fontsize,
            color='#888888', style='italic')


def _glow_line(ax, x, y, color, **kw):
    """Plot a line with a soft glow behind it."""
    ax.plot(x, y, color=color, alpha=0.15, linewidth=6,
            solid_capstyle='round')
    return ax.plot(x, y, color=color, **kw)


def _rounded_box(ax):
    """Give axes a subtle rounded-corner look via spines."""
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.7)
        sp.set_color('#CCCCCC')


def _safe_float(v, default=np.nan):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _avg_iters_from_summary(s, it_key, conv_key):
    """Average iterations over converged points when per-point arrays exist."""
    iters = s.get(it_key)
    conv = s.get(conv_key)
    if iters is None or conv is None:
        return np.nan
    iters = np.array(iters)
    conv = np.array(conv, dtype=bool)
    if len(iters) == 0 or len(conv) == 0:
        return np.nan
    mask = conv & (iters > 0)
    if np.sum(mask) == 0:
        return np.nan
    return float(np.mean(iters[mask]))


def _first_present(d, keys, default=None):
    """Return first present key from a dict-like object."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _normalize_summary_keys(summary):
    """Normalize compact output summary.json keys to plotting schema.

    The plotting functions were originally written against full in-memory
    summaries from training runs. Saved output summaries are compact and use
    slightly different key names (for example *_pct, *_time_s).
    """
    s = dict(summary)

    # Canonical aliases used by existing plotting code.
    aliases = {
        'case_name': ['case_name', 'case_label', 'case_key'],
        'training_time': ['training_time', 'training_time_s'],
        'final_violation': ['final_violation', 'final_violation_pu'],
        'nr_flat_conv_rate': ['nr_flat_conv_rate', 'nr_flat_conv_rate_pct'],
        'warm_conv_rate': ['warm_conv_rate', 'warm_conv_rate_pct'],
        'rescue_rate': ['rescue_rate', 'rescue_rate_pct'],
        'nn_rescue_rate': ['nn_rescue_rate', 'nn_rescue_rate_pct'],
        'dcopf_rescue_rate': ['dcopf_rescue_rate', 'dcopf_rescue_rate_pct'],
        'nn_warm_conv_rate': ['nn_warm_conv_rate', 'nn_warm_conv_rate_pct'],
        'dcopf_warm_conv_rate': ['dcopf_warm_conv_rate', 'dcopf_warm_conv_rate_pct'],
        'nn_inference_time': ['nn_inference_time', 'nn_inference_time_s', 'pdl_inference_time_s'],
        'nr_flat_time_s': ['nr_flat_time_s', 'nr_flat_time'],
        'nr_warm_time_s': ['nr_warm_time_s', 'nr_warm_time'],
        'pdl_inference_time_s': ['pdl_inference_time_s', 'pdl_inf_time', 'nn_inference_time_s'],
        'pdl_warm_total_time_s': ['pdl_warm_total_time_s', 'pdl_warm_total_time'],
        'nn_warm_total_time_s': ['nn_warm_total_time_s', 'nn_warm_total_time'],
        'dcopf_warm_total_time_s': ['dcopf_warm_total_time_s', 'dcopf_warm_total_time'],
    }

    for dst, keys in aliases.items():
        if dst not in s or s.get(dst) is None:
            s[dst] = _first_present(s, keys)

    # Stress sweep may be stored as a list of dicts in compact summaries.
    if ('stress_multipliers' not in s or 'nr_conv_rates' not in s or 'warm_conv_rates' not in s):
        sweep = s.get('stress_sweep', [])
        if isinstance(sweep, list) and len(sweep) > 0 and isinstance(sweep[0], dict):
            s.setdefault('stress_multipliers', [r.get('stress_multiplier', np.nan) for r in sweep])
            s.setdefault('nr_conv_rates', [r.get('nr_conv_rate_pct', np.nan) for r in sweep])
            s.setdefault('warm_conv_rates', [r.get('warm_conv_rate_pct', np.nan) for r in sweep])

    return s


def load_summaries_from_output_dir(output_dir='outputs', latest_per_case=True):
    """Load summary.json files from output subdirectories."""
    pattern = os.path.join(output_dir, '*', 'summary.json')
    files = glob.glob(pattern)
    if not files:
        return []

    records = []
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                s = json.load(f)
            s = _normalize_summary_keys(s)
            s['_summary_path'] = fpath
            s['_summary_mtime'] = os.path.getmtime(fpath)
            records.append(s)
        except Exception:
            continue

    if not latest_per_case:
        return sorted(records, key=lambda x: x.get('_summary_mtime', 0.0))

    latest = {}
    for s in records:
        case_id = s.get('case_key', s.get('case_name', 'unknown'))
        if case_id not in latest or s.get('_summary_mtime', 0) > latest[case_id].get('_summary_mtime', 0):
            latest[case_id] = s

    out = list(latest.values())
    out.sort(key=lambda x: x.get('case_name', ''))
    return out


def _sanitize_filename(name):
    """Return a filesystem-safe filename stem."""
    safe = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in str(name))
    safe = safe.strip('_')
    return safe or 'figure'


def _save_fig(fig, out_path):
    """Save a matplotlib figure and close it to release memory."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# 2.  PLOT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

# ── 2A.  Training Convergence ────────────────────────────────────
def beautiful_convergence(s):
    """Training convergence & ρ schedule — dual-panel."""
    h = s.get('history')
    if not isinstance(h, dict):
        return None
    needed = ('max_viol', 'mean_viol', 'rho')
    if any(k not in h for k in needed):
        return None
    n = len(h['max_viol'])
    if n == 0:
        return None
    x = np.arange(1, n + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle(f'PDL-GAT Training Dynamics  —  {s["case_name"]}',
                 fontsize=18, fontweight='bold', y=1.02)

    # left: violations
    _glow_line(ax1, x, h['max_viol'], _ACCENT2)
    ax1.semilogy(x, h['max_viol'], 'o-', color=_ACCENT2,
                 markersize=4, label='Max violation')
    _glow_line(ax1, x, h['mean_viol'], _ACCENT1)
    ax1.semilogy(x, h['mean_viol'], 's--', color=_ACCENT1,
                 markersize=3, alpha=0.85, label='Mean violation')
    ax1.axhline(1e-4, color=_ACCENT3, linestyle=':', linewidth=1.5,
                alpha=0.7, label='Target 1e-4')
    ax1.set_xlabel('Outer Iteration')
    ax1.set_ylabel('Power-Balance Violation (p.u.)')
    ax1.set_title('Constraint Satisfaction')
    ax1.legend(loc='upper right')
    _subtitle(ax1, f'Converged in {s["convergence_iters"]} iterations  |  '
              f'Final: {s["final_violation"]:.2e} p.u.')
    _rounded_box(ax1)

    # right: rho
    rho = np.array(h['rho'])
    color_rho = _ACCENT3
    _glow_line(ax2, x, rho, color_rho)
    ax2.semilogy(x, rho, '-', color=color_rho, linewidth=2.5)
    ax2.fill_between(x, rho[0], rho, color=color_rho, alpha=0.08)
    ax2.set_xlabel('Outer Iteration')
    ax2.set_ylabel('ρ  (penalty weight)')
    ax2.set_title('Deterministic ρ Schedule')
    _subtitle(ax2, f'ρ: {rho[0]:.0f}  →  {rho[-1]:.0f}  (exponential)')
    _rounded_box(ax2)

    plt.tight_layout()
    return fig


# ── 2B.  Error Histograms ───────────────────────────────────────
def beautiful_error_histograms(s):
    """Voltage & angle error distributions with KDE-like overlay."""
    if 'V_pdl_conv' not in s:
        return None
    V_pdl = np.array(s['V_pdl_conv'])
    V_nr = np.array(s['V_nr_conv'])
    th_pdl = np.array(s['th_pdl_conv_deg'])
    th_nr = np.array(s['th_nr_conv_deg'])
    if V_pdl.size == 0 or V_nr.size == 0 or th_pdl.size == 0 or th_nr.size == 0:
        return None
    V_err  = (V_pdl - V_nr).flatten()
    th_err = (th_pdl - th_nr).flatten()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'PDL Prediction Error Distribution  —  {s["case_name"]}',
                 fontsize=18, fontweight='bold', y=1.02)

    for ax, err, xl, col, unit in [
        (ax1, V_err,  'Voltage Magnitude Error', _ACCENT1, 'p.u.'),
        (ax2, th_err, 'Voltage Angle Error',     _ACCENT4, 'deg'),
    ]:
        n_pts = len(err)
        bins = min(80, max(30, n_pts // 40))
        counts, edges, bars = ax.hist(
            err, bins=bins, color=col, alpha=0.55, edgecolor='white',
            linewidth=0.5, density=True, label='Histogram')

        # Smooth envelope
        if HAS_SCIPY:
            centers = (edges[:-1] + edges[1:]) / 2
            smooth  = gaussian_filter1d(counts.astype(float), sigma=1.5)
            ax.plot(centers, smooth, color=col, linewidth=2.5,
                    label='Smoothed density')

        ax.axvline(0, color=_ACCENT2, linestyle='--', linewidth=1.8,
                   alpha=0.8, label='Zero')
        ax.axvline(np.mean(err), color=_ACCENT3, linestyle=':',
                   linewidth=1.5, alpha=0.7,
                   label=f'Mean = {np.mean(err):.2e}')

        ax.set_xlabel(f'{xl} ({unit})')
        ax.set_ylabel('Density')
        ax.set_title(xl)
        ax.legend(fontsize=10)
        _subtitle(ax, f'Std = {np.std(err):.2e} {unit}  |  '
                  f'Max |err| = {np.max(np.abs(err)):.2e} {unit}')
        _rounded_box(ax)

    plt.tight_layout()
    return fig


# ── 2C.  Rescue Bar Chart  (stacked, gradient-fill style) ───────
def beautiful_rescue_bar(s):
    """Horizontal butterfly bar chart: NR flat vs PDL warm-start."""
    n_total = s['n_stressed_test']
    n_nr_div = s['nr_flat_diverged']
    n_nr_conv = n_total - n_nr_div
    n_rescued = s['n_rescued']
    n_warm_conv = s['warm_converged']
    n_warm_div = n_total - n_warm_conv

    fig, ax = plt.subplots(figsize=(12, 5.5))

    cats = ['NR Flat-Start', 'PDL Warm + NR']
    conv_vals = [n_nr_conv, n_warm_conv]
    div_vals  = [n_nr_div,  n_warm_div]

    y = np.arange(len(cats))
    h = 0.52

    # Converged bars
    bars_c = ax.barh(y, conv_vals, h, color=_ACCENT3, alpha=0.85,
                     edgecolor='white', linewidth=1.2, label='Converged',
                     zorder=3)
    # Diverged bars
    bars_d = ax.barh(y, div_vals, h, left=conv_vals, color=_ACCENT2,
                     alpha=0.75, edgecolor='white', linewidth=1.2,
                     label='Diverged', zorder=3)

    # Labels inside bars
    for i in range(len(cats)):
        if conv_vals[i] > 0:
            ax.text(conv_vals[i] / 2, y[i], f'{conv_vals[i]}',
                    ha='center', va='center', fontweight='bold',
                    fontsize=15, color='white',
                    path_effects=[pe.withStroke(linewidth=2, foreground='black')])
        if div_vals[i] > 0:
            ax.text(conv_vals[i] + div_vals[i] / 2, y[i], f'{div_vals[i]}',
                    ha='center', va='center', fontweight='bold',
                    fontsize=15, color='white',
                    path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    # Rescue annotation arrow
    if n_rescued > 0:
        mid_y = 0.5
        ax.annotate(
            f'  {n_rescued} rescued\n  ({s["rescue_rate"]:.1f}%)',
            xy=(n_nr_conv + n_rescued / 2, 1),
            xytext=(n_total * 0.75, mid_y),
            fontsize=14, fontweight='bold', color=_ACCENT1,
            arrowprops=dict(arrowstyle='->', color=_ACCENT1,
                            linewidth=2, connectionstyle='arc3,rad=0.2'),
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F4FD',
                      edgecolor=_ACCENT1, linewidth=1.5),
        )

    ax.set_yticks(y)
    ax.set_yticklabels(cats, fontsize=14, fontweight='medium')
    ax.set_xlabel('Number of Test Points', fontsize=13)
    ax.set_title(f'NR Divergence Rescue  —  {s["case_name"]}',
                 fontsize=17, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12,
              framealpha=0.9, edgecolor='#CCC')
    ax.set_xlim(0, n_total * 1.12)
    ax.invert_yaxis()
    _rounded_box(ax)

    # Bottom text
    ax.text(0.5, -0.10,
            f'Standard NR diverges on {100 - s["nr_flat_conv_rate"]:.1f}% '
            f'of stressed points  |  '
            f'PDL warm-start rescues {s["rescue_rate"]:.1f}% of those',
            transform=ax.transAxes, ha='center', fontsize=11,
            color='#666666', style='italic')

    plt.tight_layout()
    return fig


# ── 2D.  Convergence Rate vs Stress Level ────────────────────────
def beautiful_conv_vs_stress(s):
    """Dual-line with shaded rescue margin and annotation callouts."""
    mults    = s.get('stress_multipliers')
    nr_rates = s.get('nr_conv_rates')
    w_rates  = s.get('warm_conv_rates')
    if mults is None or nr_rates is None or w_rates is None:
        return None
    if len(mults) == 0 or len(nr_rates) == 0 or len(w_rates) == 0:
        return None

    fig, ax = plt.subplots(figsize=(11, 6))

    # Shaded rescue margin
    ax.fill_between(mults, nr_rates, w_rates,
                    color=_ACCENT1, alpha=0.10, label='Rescue margin',
                    zorder=1)

    # NR flat (with glow)
    _glow_line(ax, mults, nr_rates, _ACCENT2)
    ax.plot(mults, nr_rates, 'o-', color=_ACCENT2, markersize=9,
            markeredgecolor='white', markeredgewidth=1.5,
            label='NR Flat-Start', zorder=4)

    # PDL warm (with glow)
    _glow_line(ax, mults, w_rates, _ACCENT1)
    ax.plot(mults, w_rates, 's-', color=_ACCENT1, markersize=9,
            markeredgecolor='white', markeredgewidth=1.5,
            label='PDL Warm-Start + NR', zorder=4)

    # Annotate each point with the gap
    for i, m in enumerate(mults):
        gap = w_rates[i] - nr_rates[i]
        if gap > 2:
            mid = (nr_rates[i] + w_rates[i]) / 2
            ax.annotate(f'+{gap:.0f}pp', xy=(m, mid),
                        fontsize=9, fontweight='bold', color=_ACCENT1,
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor='white', edgecolor=_ACCENT1,
                                  alpha=0.85, linewidth=0.8))

    # Danger zone shading
    ax.axhspan(0, 50, color=_ACCENT2, alpha=0.03, zorder=0)
    ax.axhline(50, color=_ACCENT2, linestyle=':', alpha=0.3, linewidth=1)
    ax.text(mults[0] + 0.02, 47, 'Below 50% — danger zone', fontsize=9,
            color=_ACCENT2, alpha=0.5, style='italic')

    ax.set_xlabel('API Load Multiplier (1.0 = PGLIB API base)')
    ax.set_ylabel('Convergence Rate (%)')
    ax.set_title(f'Convergence Rate vs API Load Level  —  {s["case_name"]}')
    ax.set_ylim(-3, 108)
    ax.set_xlim(mults[0] - 0.05, mults[-1] + 0.05)
    ax.legend(loc='lower left', fontsize=12)
    _rounded_box(ax)
    _subtitle(ax, 'PGLIB-OPF API tier: 1.0× = near-collapse base; '
              'PDL warm-start maintains convergence')
    plt.tight_layout()
    return fig


# ── 2E.  Per-Point Iteration Scatter (beautiful version) ────────
def beautiful_iter_scatter(s, max_iter=30):
    """Classified scatter with custom markers and marginal histograms."""
    required = ('flat_iters', 'flat_conv', 'warm_iters', 'warm_conv')
    if any(k not in s for k in required):
        return None
    fi = np.array([r if c else max_iter for r, c in
                   zip(s['flat_iters'], s['flat_conv'])], dtype=float)
    wi = np.array([r if c else max_iter for r, c in
                   zip(s['warm_iters'], s['warm_conv'])], dtype=float)
    fc = np.array(s['flat_conv'])
    wc = np.array(s['warm_conv'])

    both   = fc & wc
    rescue = (~fc) & wc
    both_d = (~fc) & (~wc)
    fw_div = fc & (~wc)

    fig = plt.figure(figsize=(9, 9))
    gs = GridSpec(4, 4, hspace=0.05, wspace=0.05)
    ax_main = fig.add_subplot(gs[1:, :-1])
    ax_top  = fig.add_subplot(gs[0, :-1], sharex=ax_main)
    ax_rt   = fig.add_subplot(gs[1:, -1], sharey=ax_main)

    jitter = lambda n: np.random.uniform(-0.35, 0.35, n)

    # Main scatter
    if np.any(both):
        ax_main.scatter(fi[both] + jitter(np.sum(both)),
                        wi[both] + jitter(np.sum(both)),
                        c=_ACCENT3, alpha=0.35, s=18, zorder=2,
                        label=f'Both converged ({np.sum(both)})')
    if np.any(rescue):
        ax_main.scatter(fi[rescue] + jitter(np.sum(rescue)),
                        wi[rescue] + jitter(np.sum(rescue)),
                        c=_ACCENT1, alpha=0.75, s=55, marker='^',
                        edgecolors='white', linewidths=0.5, zorder=4,
                        label=f'RESCUED ({np.sum(rescue)})')
    if np.any(both_d):
        ax_main.scatter(fi[both_d] + jitter(np.sum(both_d)),
                        wi[both_d] + jitter(np.sum(both_d)),
                        c=_ACCENT2, alpha=0.4, s=18, marker='x',
                        linewidths=1, zorder=2,
                        label=f'Both diverged ({np.sum(both_d)})')
    if np.any(fw_div):
        ax_main.scatter(fi[fw_div] + jitter(np.sum(fw_div)),
                        wi[fw_div] + jitter(np.sum(fw_div)),
                        c=_ACCENT5, alpha=0.6, s=40, marker='v',
                        edgecolors='white', linewidths=0.5, zorder=3,
                        label=f'Flat OK / warm div ({np.sum(fw_div)})')

    ax_main.plot([0, max_iter + 1], [0, max_iter + 1], color='#AAAAAA',
                 linestyle='--', linewidth=1, alpha=0.6, zorder=1)
    ax_main.axhline(max_iter, color='#DDDDDD', linewidth=0.8, zorder=0)
    ax_main.axvline(max_iter, color='#DDDDDD', linewidth=0.8, zorder=0)

    # Divergence zone shading
    ax_main.axvspan(max_iter - 0.5, max_iter + 2, color=_ACCENT2,
                    alpha=0.04, zorder=0)
    ax_main.axhspan(max_iter - 0.5, max_iter + 2, color=_ACCENT2,
                    alpha=0.04, zorder=0)

    ax_main.set_xlabel('NR Flat-Start Iterations')
    ax_main.set_ylabel('PDL Warm-Start + NR Iterations')
    ax_main.set_xlim(-1, max_iter + 2)
    ax_main.set_ylim(-1, max_iter + 2)
    ax_main.legend(loc='upper left', fontsize=9.5, framealpha=0.9)
    _rounded_box(ax_main)

    # Marginal histograms
    bins = np.arange(-0.5, max_iter + 2.5, 1)
    ax_top.hist(fi[fc], bins=bins, color=_ACCENT3, alpha=0.5,
                edgecolor='white', linewidth=0.5)
    ax_top.hist(fi[~fc], bins=bins, color=_ACCENT2, alpha=0.5,
                edgecolor='white', linewidth=0.5)
    ax_top.set_ylabel('Count')
    ax_top.set_title(f'Per-Point Iteration Map  —  {s["case_name"]}',
                     fontsize=15, fontweight='bold')
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.spines['bottom'].set_visible(False)

    ax_rt.hist(wi[wc], bins=bins, orientation='horizontal',
               color=_ACCENT1, alpha=0.5, edgecolor='white', linewidth=0.5)
    ax_rt.hist(wi[~wc], bins=bins, orientation='horizontal',
               color=_ACCENT2, alpha=0.5, edgecolor='white', linewidth=0.5)
    ax_rt.set_xlabel('Count')
    plt.setp(ax_rt.get_yticklabels(), visible=False)
    ax_rt.spines['left'].set_visible(False)

    plt.tight_layout()
    return fig


# ── 2E2.  Participation Factor Dot Plot ───────────────────────
def beautiful_participation_dots(s):
    """Dot plot of PDL warm-start participation factors across generators."""
    pf = s.get('participation_factors', {})
    pdl_pf = pf.get('pdl_warm') if isinstance(pf, dict) else None
    if not pdl_pf:
        return None

    rows = [np.array(v, dtype=float) for v in pdl_pf if v is not None]
    if len(rows) == 0:
        return None

    mat = np.vstack(rows)
    n_gen = mat.shape[1]
    x = np.arange(1, n_gen + 1)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    jitter = np.random.uniform(-0.18, 0.18, size=mat.shape)
    for i in range(n_gen):
        ax.scatter(
            x[i] + jitter[:, i],
            mat[:, i],
            s=14,
            alpha=0.35,
            color=_ACCENT1,
            edgecolors='none',
            zorder=2,
        )

    mean_pf = np.mean(mat, axis=0)
    ax.plot(x, mean_pf, 'o-', color=_ACCENT3, linewidth=2.0,
            markersize=5, markeredgecolor='white', markeredgewidth=0.8,
            label='Mean')

    ax.set_xlabel('Generator Index')
    ax.set_ylabel('Participation Factor')
    ax.set_title(f'PDL Warm-Start Participation Factors  —  {s["case_name"]}')
    ax.set_xlim(0.5, n_gen + 0.5)
    ax.set_ylim(0.0, max(0.05, float(np.max(mat)) * 1.15))
    ax.legend(loc='upper right', fontsize=10)
    _rounded_box(ax)
    _subtitle(ax, f'{mat.shape[0]} converged points; dots show per-point factors')
    plt.tight_layout()
    return fig


# ── 2F.  Rescue Summary Donut Chart ─────────────────────────────
def beautiful_donut(s):
    """Donut / pie showing the 4-way classification of test points."""
    vals = [
        s['nr_flat_converged'] - s.get('n_flat_ok_warm_div', 0),
        s['n_rescued'],
        s['n_both_div'],
        s.get('n_flat_ok_warm_div', 0),
    ]
    labels = [
        'Both converged',
        'RESCUED by PDL',
        'Both diverged',
        'Flat OK / warm div',
    ]
    colors = [_ACCENT3, _ACCENT1, _ACCENT2, _ACCENT5]

    # Remove zero slices
    keep = [i for i, v in enumerate(vals) if v > 0]
    vals   = [vals[i]   for i in keep]
    labels = [labels[i] for i in keep]
    colors = [colors[i] for i in keep]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        vals, labels=None, colors=colors, autopct='%1.1f%%',
        startangle=90, pctdistance=0.78, wedgeprops=dict(width=0.42,
        edgecolor='white', linewidth=2.5),
        textprops=dict(fontsize=12, fontweight='bold', color='white'))

    for at in autotexts:
        at.set_path_effects([pe.withStroke(linewidth=2, foreground='black')])

    # Centre text
    ax.text(0, 0.06, f'{s["case_name"]}', ha='center', va='center',
            fontsize=18, fontweight='bold', color=_TEXT)
    ax.text(0, -0.08, f'{s["n_stressed_test"]} test pts', ha='center',
            va='center', fontsize=12, color='#888888')

    ax.legend(wedges, [f'{l}  ({v})' for l, v in zip(labels, vals)],
              loc='lower center', ncol=2, fontsize=11,
              bbox_to_anchor=(0.5, -0.05), frameon=True, framealpha=0.9,
              edgecolor='#CCC')
    ax.set_title('Test Point Classification', fontsize=16, fontweight='bold',
                 pad=20)
    plt.tight_layout()
    return fig


def beautiful_convergence_count(s):
    """Stacked convergence/divergence counts across warm-start methods."""
    n_total = int(s.get('n_stressed_test', 0))
    if n_total <= 0:
        return None

    names = ['NR Flat', 'PDL Warm + NR', 'NN Warm + NR', 'DCOPF Warm + NR']
    conv = [
        int(s.get('nr_flat_converged', 0)),
        int(s.get('warm_converged', 0)),
        int(s.get('nn_warm_converged', 0)),
        int(s.get('dcopf_warm_converged', 0)),
    ]
    div = [max(n_total - c, 0) for c in conv]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x, conv, 0.6, color=_ACCENT3, edgecolor='white', linewidth=1,
           label='Converged', zorder=3)
    ax.bar(x, div, 0.6, bottom=conv, color=_ACCENT2, edgecolor='white',
           linewidth=1, label='Diverged', zorder=3)

    for i in range(len(names)):
        if conv[i] > 0:
            ax.text(x[i], conv[i] / 2.0, f'{conv[i]}', ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')
        if div[i] > 0:
            ax.text(x[i], conv[i] + div[i] / 2.0, f'{div[i]}', ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel('Test points')
    ax.set_title(f'Convergence Count by Method  —  {s["case_name"]}')
    ax.legend(loc='upper right', fontsize=10)
    _rounded_box(ax)
    plt.tight_layout()
    return fig


def beautiful_method_time_breakdown(s):
    """Per-case timing breakdown for flat/warm-start methods."""
    labels = [
        'NR Flat', 'PDL Infer', 'PDL Warm+NR',
        'NN Infer', 'NN Warm+NR', 'DCOPF', 'DCOPF Warm+NR'
    ]
    vals = [
        _safe_float(s.get('nr_flat_time_s', np.nan)),
        _safe_float(s.get('pdl_inference_time_s', np.nan)),
        _safe_float(s.get('nr_warm_time_s', np.nan)),
        _safe_float(s.get('nn_inference_time_s', np.nan)),
        _safe_float(s.get('nn_warm_time_s', np.nan)),
        _safe_float(s.get('dcopf_time_s', np.nan)),
        _safe_float(s.get('dcopf_warm_time_s', np.nan)),
    ]
    if np.all(np.isnan(vals)):
        return None

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    ax.bar(x, np.nan_to_num(vals, nan=0.0), color=_ACCENT1, alpha=0.82,
           edgecolor='white', linewidth=1, zorder=3)

    for i, v in enumerate(vals):
        txt = 'NA' if np.isnan(v) else f'{v:.2f}'
        y = 0.15 if np.isnan(v) else v + max(0.05, v * 0.02)
        ax.text(i, y, txt, ha='center', va='bottom', fontsize=9,
                fontweight='bold' if not np.isnan(v) else 'normal',
                color=_TEXT if not np.isnan(v) else '#777777')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha='right')
    ax.set_ylabel('Time (seconds)')
    ax.set_title(f'Timing Breakdown by Method  —  {s["case_name"]}')
    _rounded_box(ax)
    plt.tight_layout()
    return fig


def beautiful_method_iteration_compare(s):
    """Per-case average NR iterations across methods."""
    labels = ['NR Flat', 'PDL Warm+NR', 'NN Warm+NR', 'DCOPF Warm+NR']
    vals = [
        _safe_float(s.get('flat_avg_iters', np.nan)),
        _safe_float(s.get('warm_avg_iters', np.nan)),
        _safe_float(s.get('nn_warm_avg_iters', np.nan)),
        _safe_float(s.get('dcopf_warm_avg_iters', np.nan)),
    ]

    # Fallback to per-point arrays where available.
    if np.isnan(vals[0]):
        vals[0] = _avg_iters_from_summary(s, 'flat_iters', 'flat_conv')
    if np.isnan(vals[1]):
        vals[1] = _avg_iters_from_summary(s, 'warm_iters', 'warm_conv')
    if np.isnan(vals[2]):
        vals[2] = _avg_iters_from_summary(s, 'nn_warm_iters', 'nn_warm_conv')
    if np.isnan(vals[3]):
        vals[3] = _avg_iters_from_summary(s, 'dcopf_warm_iters', 'dcopf_warm_conv')

    if np.all(np.isnan(vals)):
        return None

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x, np.nan_to_num(vals, nan=0.0), color=_ACCENT4, alpha=0.88,
           edgecolor='white', linewidth=1, zorder=3)

    for i, v in enumerate(vals):
        txt = 'NA' if np.isnan(v) else f'{v:.2f}'
        y = 0.12 if np.isnan(v) else v + 0.15
        ax.text(i, y, txt, ha='center', va='bottom', fontsize=10,
                color=_TEXT if not np.isnan(v) else '#777777')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Average NR iterations (converged points)')
    ax.set_title(f'Iteration Count by Method  —  {s["case_name"]}')
    _rounded_box(ax)
    plt.tight_layout()
    return fig


# ── 2G.  Cross-Case Comparison (grouped bar) ────────────────────
def beautiful_cross_case(summaries):
    """Side-by-side grouped bars comparing all cases."""
    if len(summaries) < 1:
        return None

    names = [s['case_name'] for s in summaries]
    nr_div_pct  = [100 - s['nr_flat_conv_rate'] for s in summaries]
    rescue_pct  = [s['rescue_rate'] for s in summaries]
    warm_conv   = [s['warm_conv_rate'] for s in summaries]

    x = np.arange(len(names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    b1 = ax.bar(x - w, nr_div_pct, w, color=_ACCENT2, alpha=0.85,
                edgecolor='white', linewidth=1, label='NR Diverged %',
                zorder=3)
    b2 = ax.bar(x, rescue_pct, w, color=_ACCENT1, alpha=0.85,
                edgecolor='white', linewidth=1, label='Rescue Rate %',
                zorder=3)
    b3 = ax.bar(x + w, warm_conv, w, color=_ACCENT3, alpha=0.85,
                edgecolor='white', linewidth=1, label='Warm Conv %',
                zorder=3)

    # Value labels
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.2,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=11,
                    fontweight='bold', color=_TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=14, fontweight='medium')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Cross-Case Divergence Rescue Comparison',
                 fontsize=17, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.set_ylim(0, max(max(nr_div_pct), max(warm_conv)) * 1.18)
    _rounded_box(ax)

    _subtitle(ax, 'PDL warm-start consistently rescues a large fraction '
              'of NR-divergent stressed operating points')
    plt.tight_layout()
    return fig


def beautiful_inference_time_compare(summaries):
    """Compare flat-start vs warm-start total runtime by case."""
    if len(summaries) < 1:
        return None

    names = [s.get('case_name', 'case') for s in summaries]
    flat_t = []
    warm_t = []

    for s in summaries:
        ft = _safe_float(s.get('nr_flat_time_s', s.get('nr_flat_time', np.nan)))
        wt_nr = _safe_float(s.get('nr_warm_time_s', s.get('nr_warm_time', np.nan)))
        wt_inf = _safe_float(s.get('pdl_inference_time_s', s.get('pdl_inf_time', np.nan)))
        if np.isnan(wt_nr) and np.isnan(wt_inf):
            wt = np.nan
        elif np.isnan(wt_inf):
            wt = wt_nr
        elif np.isnan(wt_nr):
            wt = wt_inf
        else:
            wt = wt_nr + wt_inf
        flat_t.append(ft)
        warm_t.append(wt)

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(x - w / 2, np.nan_to_num(flat_t, nan=0.0), w,
           color=_ACCENT3, alpha=0.85, edgecolor='white', linewidth=1,
           label='Flat-start total time (s)')
    ax.bar(x + w / 2, np.nan_to_num(warm_t, nan=0.0), w,
           color=_ACCENT2, alpha=0.85, edgecolor='white', linewidth=1,
           label='Warm-start total time (s)')

    for i in range(len(names)):
        if not np.isnan(flat_t[i]):
            ax.text(x[i] - w / 2, flat_t[i] + max(0.1, flat_t[i] * 0.02),
                    f'{flat_t[i]:.1f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
        else:
            ax.text(x[i] - w / 2, 0.2, 'NA', ha='center', va='bottom',
                    fontsize=9, color='#777777')

        if not np.isnan(warm_t[i]):
            ax.text(x[i] + w / 2, warm_t[i] + max(0.1, warm_t[i] * 0.02),
                    f'{warm_t[i]:.1f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
        else:
            ax.text(x[i] + w / 2, 0.2, 'NA', ha='center', va='bottom',
                    fontsize=9, color='#777777')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12, fontweight='medium')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Inference Time Comparison: Flat vs Warm Start',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ymax = max(np.nanmax(np.nan_to_num(flat_t, nan=0.0)),
               np.nanmax(np.nan_to_num(warm_t, nan=0.0)), 1.0)
    ax.set_ylim(0, ymax * 1.2)
    _rounded_box(ax)
    _subtitle(ax, 'Loaded from outputs/summary.json. NA indicates missing metric.')
    plt.tight_layout()
    return fig


def beautiful_avg_iterations_compare(summaries):
    """Compare average NR iterations for flat vs warm starts across cases."""
    if len(summaries) < 1:
        return None

    names = [s.get('case_name', 'case') for s in summaries]
    flat_avg = []
    warm_avg = []

    for s in summaries:
        fi = _safe_float(s.get('flat_avg_iters', np.nan))
        wi = _safe_float(s.get('warm_avg_iters', np.nan))
        if np.isnan(fi):
            fi = _avg_iters_from_summary(s, 'flat_iters', 'flat_conv')
        if np.isnan(wi):
            wi = _avg_iters_from_summary(s, 'warm_iters', 'warm_conv')
        flat_avg.append(fi)
        warm_avg.append(wi)

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(x - w / 2, np.nan_to_num(flat_avg, nan=0.0), w,
           color=_ACCENT3, alpha=0.85, edgecolor='white', linewidth=1,
           label='Flat-start avg iterations')
    ax.bar(x + w / 2, np.nan_to_num(warm_avg, nan=0.0), w,
           color=_ACCENT2, alpha=0.85, edgecolor='white', linewidth=1,
           label='Warm-start avg iterations')

    for i in range(len(names)):
        if not np.isnan(flat_avg[i]):
            ax.text(x[i] - w / 2, flat_avg[i] + 0.2, f'{flat_avg[i]:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax.text(x[i] - w / 2, 0.2, 'NA', ha='center', va='bottom',
                    fontsize=9, color='#777777')

        if not np.isnan(warm_avg[i]):
            ax.text(x[i] + w / 2, warm_avg[i] + 0.2, f'{warm_avg[i]:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax.text(x[i] + w / 2, 0.2, 'NA', ha='center', va='bottom',
                    fontsize=9, color='#777777')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12, fontweight='medium')
    ax.set_ylabel('Average NR Iterations')
    ax.set_title('Average Iteration Count: Flat vs Warm Start',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ymax = max(np.nanmax(np.nan_to_num(flat_avg, nan=0.0)),
               np.nanmax(np.nan_to_num(warm_avg, nan=0.0)), 1.0)
    ax.set_ylim(0, ymax * 1.25)
    _rounded_box(ax)
    _subtitle(ax, 'Uses iteration arrays when available in output summaries.')
    plt.tight_layout()
    return fig


def beautiful_summary_table(summaries):
    """Cross-case table summary for key rescue, timing, and iteration metrics."""
    if len(summaries) < 1:
        return None

    cols = [
        'Case', 'NR Div %', 'PDL Rescue %', 'PDL Conv %',
        'PDL Warm Total (s)', 'NN Warm Total (s)', 'DCOPF Warm Total (s)',
        'Flat Avg It', 'PDL Avg It'
    ]
    rows = []
    for s in summaries:
        rows.append([
            s.get('case_name', 'case'),
            f"{100 - _safe_float(s.get('nr_flat_conv_rate', np.nan), 0.0):.1f}",
            f"{_safe_float(s.get('rescue_rate', np.nan), 0.0):.1f}",
            f"{_safe_float(s.get('warm_conv_rate', np.nan), 0.0):.1f}",
            f"{_safe_float(s.get('pdl_warm_total_time_s', np.nan), 0.0):.2f}",
            f"{_safe_float(s.get('nn_warm_total_time_s', np.nan), 0.0):.2f}",
            f"{_safe_float(s.get('dcopf_warm_total_time_s', np.nan), 0.0):.2f}",
            f"{_safe_float(s.get('flat_avg_iters', np.nan), 0.0):.2f}",
            f"{_safe_float(s.get('warm_avg_iters', np.nan), 0.0):.2f}",
        ])

    fig_h = max(3.0, 1.2 + 0.45 * len(rows))
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis('off')
    tab = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.scale(1.0, 1.4)

    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor('#DDDDDD')
        if r == 0:
            cell.set_facecolor('#E8F4FD')
            cell.set_text_props(weight='bold', color=_TEXT)
        else:
            cell.set_facecolor('#FFFFFF' if r % 2 else '#F8FBFF')

    ax.set_title('Cross-Case Summary Table', pad=14, fontsize=15, fontweight='bold')
    plt.tight_layout()
    return fig


# ── 2H.  Strategy Breakdown (violin-style grouped) ──────────────
def beautiful_strategy_breakdown(s, max_iter=30):
    """Show convergence rate by stress strategy (API/API+/API-/API_Q)."""
    if 'strat_labels' not in s:
        return None
    labels = s['strat_labels']
    fc = np.array(s['flat_conv'])
    wc = np.array(s['warm_conv'])

    strategies = sorted(set(labels))
    strat_names = {
        'API':  'API Base\n(±noise)',
        'API+': 'API Scaled\nUP (1–1.3×)',
        'API-': 'API Scaled\nDOWN (0.75–1×)',
        'API_Q': 'API + Extra\nReactive',
        # Legacy synthetic labels (fallback)
        'A': 'Uniform\nHeavy Load',
        'B': 'Concentrated\nStress',
        'C': 'Reactive\nStress',
        'D': 'Combined',
    }

    nr_rates, warm_rates = [], []
    for st in strategies:
        mask = np.array([l == st for l in labels])
        nr_rates.append(np.mean(fc[mask]) * 100)
        warm_rates.append(np.mean(wc[mask]) * 100)

    x = np.arange(len(strategies))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w / 2, nr_rates, w, color=_ACCENT2, alpha=0.8,
                edgecolor='white', linewidth=1, label='NR Flat-Start',
                zorder=3)
    b2 = ax.bar(x + w / 2, warm_rates, w, color=_ACCENT1, alpha=0.8,
                edgecolor='white', linewidth=1, label='PDL Warm + NR',
                zorder=3)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
                    f'{h:.0f}%', ha='center', va='bottom', fontsize=11,
                    fontweight='bold', color=_TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels([strat_names.get(st, st) for st in strategies],
                       fontsize=12)
    ax.set_ylabel('Convergence Rate (%)')
    ax.set_title(f'Convergence by Stress Strategy  —  {s["case_name"]}',
                 fontsize=16, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.legend(fontsize=12)
    _rounded_box(ax)
    _subtitle(ax, 'PGLIB-OPF API tier: base ± noise | scaled up/down | '
              'extra reactive stress')
    plt.tight_layout()
    return fig


# ── 2I.  Grand Summary Dashboard (single figure) ────────────────
def beautiful_dashboard(s):
    """All key metrics in one 2×3 dashboard figure."""
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'PDL-GAT Divergence Rescue Dashboard  —  {s["case_name"]}',
                 fontsize=22, fontweight='bold', y=0.98)

    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    # ── Panel 1: convergence ──
    ax1 = fig.add_subplot(gs[0, 0])
    h = s.get('history')
    if isinstance(h, dict) and all(k in h for k in ('max_viol', 'mean_viol', 'rho')) and len(h['max_viol']) > 0:
        n = len(h['max_viol'])
        x = np.arange(1, n + 1)
        ax1.semilogy(x, h['max_viol'], 'o-', color=_ACCENT2, markersize=3,
                     label='Max viol')
        ax1.semilogy(x, h['mean_viol'], 's--', color=_ACCENT1, markersize=2,
                     alpha=0.7, label='Mean viol')
        ax1.axhline(1e-4, color=_ACCENT3, linestyle=':', alpha=0.6)
        ax1.legend(fontsize=9)
    else:
        ax1.text(0.5, 0.5, 'No training history\nfound in summary.json',
                 ha='center', va='center', fontsize=12, color='#AAAAAA',
                 transform=ax1.transAxes)
    ax1.set_title('Training Convergence', fontsize=13)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Violation (p.u.)')
    _rounded_box(ax1)

    # ── Panel 2: rescue donut ──
    ax2 = fig.add_subplot(gs[0, 1])
    vals = [
        s['nr_flat_converged'] - s.get('n_flat_ok_warm_div', 0),
        s['n_rescued'],
        s['n_both_div'],
        s.get('n_flat_ok_warm_div', 0),
    ]
    colors = [_ACCENT3, _ACCENT1, _ACCENT2, _ACCENT5]
    labs = ['Both conv', 'Rescued', 'Both div', 'Flat ok/warm div']
    keep = [i for i, v in enumerate(vals) if v > 0]
    ax2.pie([vals[i] for i in keep],
            labels=[labs[i] for i in keep],
            colors=[colors[i] for i in keep],
            autopct='%1.1f%%', startangle=90,
            wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2),
            textprops=dict(fontsize=9))
    ax2.set_title('Test Point Classification', fontsize=13)

    # ── Panel 3: stress sweep ──
    ax3 = fig.add_subplot(gs[0, 2])
    mults = s.get('stress_multipliers', [])
    nr_rates = s.get('nr_conv_rates', [])
    warm_rates = s.get('warm_conv_rates', [])
    if len(mults) > 0 and len(nr_rates) > 0 and len(warm_rates) > 0:
        ax3.plot(mults, nr_rates, 'o-', color=_ACCENT2, markersize=6,
                 label='NR Flat')
        ax3.plot(mults, warm_rates, 's-', color=_ACCENT1, markersize=6,
                 label='PDL Warm')
        ax3.fill_between(mults, nr_rates, warm_rates,
                         color=_ACCENT1, alpha=0.1)
        ax3.legend(fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'No stress sweep\nfound in summary.json',
                 ha='center', va='center', fontsize=12, color='#AAAAAA',
                 transform=ax3.transAxes)
    ax3.set_title('Conv Rate vs Stress', fontsize=13)
    ax3.set_xlabel('API Load Multiplier')
    ax3.set_ylabel('Convergence %')
    ax3.set_ylim(-3, 108)
    _rounded_box(ax3)

    # ── Panel 4: convergence count (stacked bar) ──
    ax4 = fig.add_subplot(gs[1, 0])
    cats = ['NR Flat', 'PDL Warm']
    conv_v = [s['nr_flat_converged'], s['warm_converged']]
    div_v  = [s['nr_flat_diverged'], s['warm_diverged']]
    xx = np.arange(2)
    ax4.bar(xx, conv_v, 0.55, color=_ACCENT3, edgecolor='white', label='Converged')
    ax4.bar(xx, div_v, 0.55, bottom=conv_v, color=_ACCENT2,
            edgecolor='white', label='Diverged')
    for i in range(len(cats)):
        if conv_v[i] > 0:
            ax4.text(xx[i], conv_v[i] / 2.0, f'{int(conv_v[i])}',
                     ha='center', va='center', fontsize=10,
                     color='white', fontweight='bold')
        if div_v[i] > 0:
            ax4.text(xx[i], conv_v[i] + div_v[i] / 2.0, f'{int(div_v[i])}',
                     ha='center', va='center', fontsize=10,
                     color='white', fontweight='bold')
    ax4.set_xticks(xx)
    ax4.set_xticklabels(cats, fontsize=11)
    ax4.set_ylabel('Test points')
    ax4.set_title('Convergence Count', fontsize=13)
    ax4.legend(fontsize=9, loc='lower right')
    _rounded_box(ax4)

    # ── Panel 5: error histogram (if available) ──
    ax5 = fig.add_subplot(gs[1, 1])
    if 'V_pdl_conv' in s:
        err = (s['V_pdl_conv'] - s['V_nr_conv']).flatten()
        ax5.hist(err, bins=50, color=_ACCENT1, alpha=0.6, edgecolor='white')
        ax5.axvline(0, color=_ACCENT2, linestyle='--', linewidth=1.5)
        ax5.set_title('Voltage Error Dist.', fontsize=13)
        ax5.set_xlabel('Error (p.u.)')
    else:
        ax5.text(0.5, 0.5, 'No converged\nsubset data', ha='center',
                 va='center', fontsize=12, color='#AAAAAA',
                 transform=ax5.transAxes)
        ax5.set_title('Voltage Error Dist.', fontsize=13)
    _rounded_box(ax5)

    # ── Panel 6: big number cards ──
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    cards = [
        (f'{s["rescue_rate"]:.1f}%', 'Rescue Rate', _ACCENT1),
        (f'{s["nr_flat_diverged"]}', 'NR Diverged', _ACCENT2),
        (f'{s["n_rescued"]}', 'Points Rescued', _ACCENT3),
        (f'{s["training_time"]:.0f}s', 'Train Time', _ACCENT4),
    ]
    for idx, (val, label, col) in enumerate(cards):
        row, ccc = divmod(idx, 2)
        cx = 0.25 + ccc * 0.50
        cy = 0.75 - row * 0.50
        ax6.text(cx, cy, val, ha='center', va='center',
                 fontsize=28, fontweight='bold', color=col,
                 transform=ax6.transAxes,
                 path_effects=[pe.withStroke(linewidth=1, foreground='white')])
        ax6.text(cx, cy - 0.12, label, ha='center', va='center',
                 fontsize=11, color='#888888', transform=ax6.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ═══════════════════════════════════════════════════════════════════
# 3.  MAIN — Generate all beautiful plots from `all_summaries`
# ═══════════════════════════════════════════════════════════════════
def generate_all_beautiful_plots(all_summaries=None, output_dir='outputs', save_to_output=True):
    """Generate figures and optionally save them under output directories."""
    if all_summaries is None:
        all_summaries = load_summaries_from_output_dir(output_dir=output_dir)
        print(f"\n  Loaded {len(all_summaries)} summaries from {output_dir}")

    if len(all_summaries) == 0:
        print("\n  No summaries found. Run teb.py first to populate outputs/.")
        return {}

    all_figs = {}
    saved_files = []

    for s0 in all_summaries:
        s = _normalize_summary_keys(s0)
        name = s['case_name']
        print(f"\n  Generating beautiful plots for {name} ...")

        all_figs[f'{name}_convergence']  = beautiful_convergence(s)
        all_figs[f'{name}_errors']       = beautiful_error_histograms(s)
        all_figs[f'{name}_rescue_bar']   = beautiful_rescue_bar(s)
        all_figs[f'{name}_convergence_count'] = beautiful_convergence_count(s)
        all_figs[f'{name}_stress_sweep'] = beautiful_conv_vs_stress(s)
        all_figs[f'{name}_iter_scatter'] = beautiful_iter_scatter(s)
        all_figs[f'{name}_participation_dots'] = beautiful_participation_dots(s)
        all_figs[f'{name}_method_times'] = beautiful_method_time_breakdown(s)
        all_figs[f'{name}_method_iterations'] = beautiful_method_iteration_compare(s)
        all_figs[f'{name}_donut']        = beautiful_donut(s)
        all_figs[f'{name}_strategy']     = beautiful_strategy_breakdown(s)
        all_figs[f'{name}_dashboard']    = beautiful_dashboard(s)

        if save_to_output:
            case_out_dir = os.path.dirname(s.get('_summary_path', ''))
            if not case_out_dir:
                case_out_dir = os.path.join(output_dir, _sanitize_filename(name))

            for key in [
                f'{name}_convergence', f'{name}_errors', f'{name}_rescue_bar',
                f'{name}_convergence_count', f'{name}_stress_sweep',
                f'{name}_iter_scatter', f'{name}_participation_dots', f'{name}_method_times',
                f'{name}_method_iterations', f'{name}_donut',
                f'{name}_strategy', f'{name}_dashboard'
            ]:
                fig = all_figs.get(key)
                if fig is None:
                    continue
                suffix = key[len(name) + 1:] if key.startswith(name + '_') else key
                out_png = os.path.join(case_out_dir, f'chart_{_sanitize_filename(suffix)}.png')
                _save_fig(fig, out_png)
                saved_files.append(out_png)

    # Cross-case comparison
    if len(all_summaries) > 1:
        print("  Generating cross-case comparison ...")
        all_figs['cross_case'] = beautiful_cross_case(all_summaries)
        all_figs['cross_case_inference_time'] = beautiful_inference_time_compare(all_summaries)
        all_figs['cross_case_avg_iterations'] = beautiful_avg_iterations_compare(all_summaries)
        all_figs['cross_case_summary_table'] = beautiful_summary_table(all_summaries)
        if save_to_output:
            cross_dir = os.path.join(output_dir, 'cross_case_latest')
            for key in ['cross_case', 'cross_case_inference_time', 'cross_case_avg_iterations', 'cross_case_summary_table']:
                fig = all_figs.get(key)
                if fig is None:
                    continue
                out_png = os.path.join(cross_dir, f'chart_{_sanitize_filename(key)}.png')
                _save_fig(fig, out_png)
                saved_files.append(out_png)
    else:
        # Keep table available even for a single case run.
        all_figs['cross_case_summary_table'] = beautiful_summary_table(all_summaries)
        if save_to_output and all_figs['cross_case_summary_table'] is not None:
            cross_dir = os.path.join(output_dir, 'cross_case_latest')
            out_png = os.path.join(cross_dir, 'chart_cross_case_summary_table.png')
            _save_fig(all_figs['cross_case_summary_table'], out_png)
            saved_files.append(out_png)

    # Remove None and remove closed figures when saving to disk only.
    if save_to_output:
        all_figs = {}
    else:
        all_figs = {k: v for k, v in all_figs.items() if v is not None}

    if save_to_output:
        print(f"\n  Saved {len(saved_files)} chart files under {output_dir}/")
    print(f"  Generated {len(all_figs)} in-memory figures.")
    if not save_to_output:
        print("  Call plt.show() to display them.\n")
    return all_figs


# ═══════════════════════════════════════════════════════════════════
# 4.  AUTO-RUN if all_summaries exists in scope
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # When run standalone, try to find all_summaries in caller's globals
    print("\n" + "═" * 70)
    print("  Beautiful Plots for PDL-GAT v6 Divergence Rescue Results")
    print("═" * 70)
    print("\n  NOTE: Run this AFTER c6.py in the same Python session.")
    print("  Usage:")
    print("    exec(open('c6_beautiful_plots.py').read())")
    print("  or import and call:")
    print("    from c6_beautiful_plots import generate_all_beautiful_plots")
    print("    figs = generate_all_beautiful_plots(all_summaries)")
    print("    plt.show()")
    print()

    # If all_summaries is already available (e.g. via exec), run automatically
    try:
        _summaries = all_summaries  # noqa: F821
        beautiful_figs = generate_all_beautiful_plots(_summaries, output_dir='outputs', save_to_output=True)
        if beautiful_figs:
            plt.show()
    except NameError:
        print("  'all_summaries' not found in scope. Loading from outputs/ ...")
        beautiful_figs = generate_all_beautiful_plots(output_dir='outputs', save_to_output=True)
        if beautiful_figs:
            plt.show()