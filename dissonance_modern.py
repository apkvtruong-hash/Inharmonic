"""
Dissonance Explorer — a modern, GPU-accelerated interface for the
Plomp-Levelt / Sethares sensory-dissonance model, built with PySide6 + pyqtgraph.

Features
--------
* Three independent spectra (A, B, C) with presets, linking and live previews
* Interactive 2D dissonance curve with crosshair readout, zoom/pan, click-to-play
* Heatmap of the triad space with topographic iso-contours
* True OpenGL 3D dissonance surface with accurate ray-cast click-to-play
* Ranked lists of consonant intervals and consonant triads
* Additive-synthesis playback, WAV export and publication-figure export

Requirements: numpy, sounddevice, PySide6, pyqtgraph, PyOpenGL
Run:          python dissonance_modern.py
"""

from __future__ import annotations

import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

from PySide6.QtCore import (Qt, QThread, Signal, QRectF, QSize, QSizeF, QTimer,
                            QPointF, QMarginsF)
from PySide6.QtGui import (QColor, QFont, QFontMetricsF, QImage, QLinearGradient,
                           QPageSize, QPainter, QPalette, QPdfWriter, QPen,
                           QPixmap, QShortcut, QKeySequence, QTextDocument,
                           QVector3D, QVector4D)
from PySide6.QtSvg import QSvgGenerator
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox,
                               QDialog, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QFrame,
                               QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMainWindow,
                               QProgressBar, QPushButton, QScrollArea,
                               QSplitter, QStackedWidget,
                               QVBoxLayout, QWidget)

import pyqtgraph as pg

try:
    import pyqtgraph.opengl as gl
    HAS_GL = True
except Exception:  # pragma: no cover - depends on local OpenGL support
    gl = None
    HAS_GL = False


# ═══════════════════════════════════════════════════════════════════════════
#  Dissonance model
# ═══════════════════════════════════════════════════════════════════════════

SAMPLE_RATE = 44100
DURATION = 1.5

# The curve is computed this far so it can be panned past MAX RATIO and minima near it aren't clipped.
CURVE_MAX = 3.3


def amp_to_loudness(amp):
    dB = 20 * np.log10(np.maximum(amp, 1e-10))
    return 2 ** (dB / 10) / 16


def dissonance_pair_vectorized(f1, f2, l1, l2):
    x = 0.24
    s1 = 0.0207
    s2 = 18.96
    b1 = 3.51
    b2 = 5.75
    fmin = np.minimum(f1, f2)
    fmax = np.maximum(f1, f2)
    s = x / (s1 * fmin + s2)
    p = s * (fmax - fmin)
    l12 = np.minimum(l1, l2)
    return l12 * (np.exp(-b1 * p) - np.exp(-b2 * p))


def compute_dissonance_2d(spec_a, spec_b, ref_freq, max_interval, step_size):
    """Compute dissonance curve: voice A fixed, voice B varies in pitch."""
    freq_a = np.array(spec_a["freq"])
    amp_a = np.array(spec_a["amp"])
    loud_a = amp_to_loudness(amp_a)

    freq_b = np.array(spec_b["freq"])
    amp_b = np.array(spec_b["amp"])
    loud_b = amp_to_loudness(amp_b)

    ratios = np.arange(0.9, max_interval, step_size)
    num_ratios = len(ratios)

    base_a = ref_freq * freq_a
    base_b = ref_freq * freq_b

    # Cross pairs A x B
    fa_grid, fb_grid = np.meshgrid(base_a, base_b, indexing='ij')
    la_grid, lb_grid = np.meshgrid(loud_a, loud_b, indexing='ij')
    fa = fa_grid.ravel()
    fb = fb_grid.ravel()
    la = la_grid.ravel()
    lb = lb_grid.ravel()

    # Self-dissonance of A (constant)
    fa_s1, fa_s2 = np.meshgrid(base_a, base_a, indexing='ij')
    la_s1, la_s2 = np.meshgrid(loud_a, loud_a, indexing='ij')
    d_self_a = dissonance_pair_vectorized(
        fa_s1.ravel(), fa_s2.ravel(), la_s1.ravel(), la_s2.ravel()).sum() * 0.5

    # Self-dissonance of B, cross A-B (vary with ratio)
    fb_s1, fb_s2 = np.meshgrid(base_b, base_b, indexing='ij')
    lb_s1, lb_s2 = np.meshgrid(loud_b, loud_b, indexing='ij')
    fb_self = fb_s1.ravel()
    fb_self2 = fb_s2.ravel()
    lb_self = lb_s1.ravel()
    lb_self2 = lb_s2.ravel()

    dissonance_values = np.zeros(num_ratios)

    chunk_size = 200
    for start in range(0, num_ratios, chunk_size):
        end = min(start + chunk_size, num_ratios)
        c = ratios[start:end, np.newaxis]

        d_self_b = dissonance_pair_vectorized(
            c * fb_self, c * fb_self2, lb_self, lb_self2).sum(axis=1) * 0.5
        d_cross = dissonance_pair_vectorized(
            fa[np.newaxis, :], c * fb[np.newaxis, :], la, lb).sum(axis=1)

        dissonance_values[start:end] = d_self_a + d_self_b + d_cross

    max_d = dissonance_values.max()
    if max_d > 0:
        dissonance_values /= max_d
    return ratios, dissonance_values


def compute_dissonance_3d(spec_a, spec_b, spec_c, ref_freq, max_interval, step_size,
                          progress_callback=None):
    """Compute dissonance surface: A fixed, B at ratio r, C at ratio s."""
    freq_a = np.array(spec_a["freq"])
    amp_a = np.array(spec_a["amp"])
    loud_a = amp_to_loudness(amp_a)

    freq_b = np.array(spec_b["freq"])
    amp_b = np.array(spec_b["amp"])
    loud_b = amp_to_loudness(amp_b)

    freq_c = np.array(spec_c["freq"])
    amp_c = np.array(spec_c["amp"])
    loud_c = amp_to_loudness(amp_c)

    ratios = np.arange(1.0, max_interval, step_size)
    num_steps = len(ratios)

    base_a = ref_freq * freq_a
    base_b = ref_freq * freq_b
    base_c = ref_freq * freq_c

    def make_pairs(freqs1, loud1, freqs2, loud2):
        f1g, f2g = np.meshgrid(freqs1, freqs2, indexing='ij')
        l1g, l2g = np.meshgrid(loud1, loud2, indexing='ij')
        return f1g.ravel(), f2g.ravel(), l1g.ravel(), l2g.ravel()

    # A-A self (constant)
    aa_f1, aa_f2, aa_l1, aa_l2 = make_pairs(base_a, loud_a, base_a, loud_a)
    d_aa = dissonance_pair_vectorized(aa_f1, aa_f2, aa_l1, aa_l2).sum()

    # B-B self, A-B cross (depend on r)
    bb_f1, bb_f2, bb_l1, bb_l2 = make_pairs(base_b, loud_b, base_b, loud_b)
    ab_f1, ab_f2, ab_l1, ab_l2 = make_pairs(base_a, loud_a, base_b, loud_b)

    # C-C self, A-C cross (depend on s)
    cc_f1, cc_f2, cc_l1, cc_l2 = make_pairs(base_c, loud_c, base_c, loud_c)
    ac_f1, ac_f2, ac_l1, ac_l2 = make_pairs(base_a, loud_a, base_c, loud_c)

    # B-C cross (depends on r and s)
    bc_f1, bc_f2, bc_l1, bc_l2 = make_pairs(base_b, loud_b, base_c, loud_c)

    # Precompute r-dependent terms
    d_bb_all = np.zeros(num_steps)
    d_ab_all = np.zeros(num_steps)
    for ri, r in enumerate(ratios):
        d_bb_all[ri] = dissonance_pair_vectorized(r * bb_f1, r * bb_f2, bb_l1, bb_l2).sum()
        d_ab_all[ri] = dissonance_pair_vectorized(ab_f1, r * ab_f2, ab_l1, ab_l2).sum()

    # Precompute s-dependent terms
    d_cc_all = np.zeros(num_steps)
    d_ac_all = np.zeros(num_steps)
    for si, s in enumerate(ratios):
        d_cc_all[si] = dissonance_pair_vectorized(s * cc_f1, s * cc_f2, cc_l1, cc_l2).sum()
        d_ac_all[si] = dissonance_pair_vectorized(ac_f1, s * ac_f2, ac_l1, ac_l2).sum()

    # Full grid: B-C cross depends on both r and s
    z_data = np.zeros((num_steps, num_steps))
    for ri in range(num_steps):
        if progress_callback and ri % 20 == 0:
            progress_callback(ri / num_steps)

        r_bc_f1 = ratios[ri] * bc_f1
        s_bc_f2 = ratios[:, np.newaxis] * bc_f2[np.newaxis, :]

        d_bc = dissonance_pair_vectorized(
            r_bc_f1[np.newaxis, :], s_bc_f2,
            bc_l1[np.newaxis, :], bc_l2[np.newaxis, :]
        ).sum(axis=1)

        # Self terms count each pair twice; cross terms already count each pair once.
        z_data[ri, :] = ((d_aa + d_bb_all[ri] + d_cc_all) / 2.0 +
                         d_ab_all[ri] + d_ac_all + d_bc)

    max_z = z_data.max()
    if max_z > 0:
        z_data /= max_z
    if progress_callback:
        progress_callback(1.0)
    return ratios, ratios, z_data


def find_minima(ratios, dissonance_values, window=5, min_prominence=0.002):
    """Find local minima with significant prominence (depth relative to surroundings).

    A minimum is kept only if it sits at least min_prominence (on the 0-1
    normalised scale) below the lower of the two peaks that separate it from
    lower ground (or from the end of the curve).
    """
    # Step 1: find all local minima
    candidates = []
    for i in range(window, len(dissonance_values) - window):
        if dissonance_values[i] == min(dissonance_values[i - window:i + window + 1]):
            candidates.append(i)

    # Step 2: compute prominence for each candidate
    minima_x = []
    minima_y = []
    n = len(dissonance_values)

    for idx in candidates:
        val = dissonance_values[idx]

        # Highest point on each side before the curve drops below the valley
        left_max = val
        for j in range(idx - 1, -1, -1):
            if dissonance_values[j] < val:
                break
            left_max = max(left_max, dissonance_values[j])

        right_max = val
        for j in range(idx + 1, n):
            if dissonance_values[j] < val:
                break
            right_max = max(right_max, dissonance_values[j])

        # Prominence = height of the lower of the two surrounding peaks minus the valley
        prominence = min(left_max, right_max) - val

        if prominence >= min_prominence:
            minima_x.append(ratios[idx])
            minima_y.append(dissonance_values[idx])

    return np.array(minima_x), np.array(minima_y)


def _triad_cost(spec_a, spec_b, spec_c, ref_freq):
    """Exact, unnormalised dissonance of A : rB : sC as a function of (ln r, ln s)."""
    voices = (spec_a, spec_b, spec_c)
    base = np.concatenate([ref_freq * np.asarray(v["freq"], dtype=float) for v in voices])
    loud = np.concatenate([amp_to_loudness(np.asarray(v["amp"], dtype=float))
                           for v in voices])
    owner = np.repeat([0, 1, 2], [len(v["freq"]) for v in voices])
    i, j = np.triu_indices(len(base), k=1)
    li, lj = loud[i], loud[j]

    def cost(u, v):
        f = base * np.array([1.0, np.exp(u), np.exp(v)])[owner]
        return float(dissonance_pair_vectorized(f[i], f[j], li, lj).sum())
    return cost


# In (ln r, ln s) the valleys from A-B, A-C and B-C partial coincidences run along
# these three directions, so the search can slide along any of them.
_TRIAD_DIRS = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1))


def _descend(cost, u, v, h, lo, hi, tol=1e-6):
    """Pattern search down to a local minimum; None if it leaves the [lo, hi] box."""
    best = cost(u, v)
    while h > tol:
        moved = False
        for du, dv in _TRIAD_DIRS:
            c = cost(u + du * h, v + dv * h)
            if c < best:
                u, v, best = u + du * h, v + dv * h, c
                moved = True
                break
        if not moved:
            h *= 0.5
        elif not (lo[0] <= u <= hi[0] and lo[1] <= v <= hi[1]):
            return None
    return u, v, best


def find_minima_3d(ratios_x, ratios_y, z_data, spectra, ref_freq, top_n=10):
    """Find the most consonant triads (local minima of the dissonance surface).

    *z_data* is indexed ``[voice B index, voice C index]``.  Every grid local
    minimum seeds a search on the exact model, so results aren't snapped to the
    grid and the aliasing ripples a grid leaves along narrow valleys merge into
    the true minimum they belong to.  Minima that run off the grid are dropped.
    Results are sorted by dissonance (lowest first) and capped at *top_n*.
    """
    nx, ny = z_data.shape
    if nx < 3 or ny < 3:
        return np.array([]), np.array([]), np.array([])
    # The surface is symmetric only when B and C share a spectrum; mirrored pairs
    # are duplicate voicings then, and genuinely different chords otherwise.
    mirrored = nx == ny and np.allclose(z_data, z_data.T)

    padded = np.pad(z_data, 1, constant_values=np.inf)
    neighbours = np.min([padded[1 + di:nx + 1 + di, 1 + dj:ny + 1 + dj]
                         for di in (-1, 0, 1) for dj in (-1, 0, 1)
                         if di or dj], axis=0)
    seeds = zip(*np.nonzero(z_data <= neighbours))

    cost = _triad_cost(*spectra, ref_freq)
    peak = np.unravel_index(np.argmax(z_data), z_data.shape)
    norm = cost(np.log(ratios_x[peak[0]]), np.log(ratios_y[peak[1]])) / z_data[peak]
    lo = (np.log(ratios_x[0]), np.log(ratios_y[0]))
    hi = (np.log(ratios_x[-1]), np.log(ratios_y[-1]))
    step = float(ratios_x[1] - ratios_x[0])
    same = 2.0 / 1200 * np.log(2)  # minima within 2 cents on both axes are one chord

    found = []
    tol = 0.02  # tolerance for "same note" check
    for i, j in seeds:
        hit = _descend(cost, np.log(ratios_x[i]), np.log(ratios_y[j]), step, lo, hi)
        if hit is None:
            continue
        u, v, val = hit
        if mirrored and u < v:
            u, v = v, u
        bx, cy = np.exp(u), np.exp(v)
        # Skip triads where two voices are effectively the same pitch
        if abs(bx - 1.0) < tol or abs(cy - 1.0) < tol or abs(bx - cy) < tol:
            continue
        if any(abs(u - fu) < same and abs(v - fv) < same for fu, fv, _ in found):
            continue
        found.append((u, v, val / norm))

    results = [(np.exp(u), np.exp(v), z) for u, v, z in found]
    if not results:
        return np.array([]), np.array([]), np.array([])

    # Sort by dissonance (most consonant first), keep top_n
    results.sort(key=lambda t: t[2])
    if top_n is not None:
        results = results[:top_n]

    rx, ry, rz = zip(*results)
    return np.array(rx), np.array(ry), np.array(rz)


def synthesize_multi(spectra_and_ratios, ref_freq, duration=DURATION):
    """Synthesize multiple voices, each with its own spectrum and pitch ratio."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    envelope = np.ones_like(t)
    fade = min(int(0.05 * SAMPLE_RATE), len(t) // 2)
    if fade > 0:
        envelope[:fade] = np.linspace(0, 1, fade)
        envelope[-fade:] = np.linspace(1, 0, fade)

    signal = np.zeros_like(t)
    for spectrum, ratio in spectra_and_ratios:
        freq_ratios = np.array(spectrum["freq"])
        amps = np.array(spectrum["amp"])
        base = ref_freq * ratio
        gain = max(1.0 / len(freq_ratios), 0.5)
        for fr, amp in zip(freq_ratios, amps):
            signal += gain * amp * np.sin(2 * np.pi * base * fr * t)

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.7
    signal *= envelope
    return signal.astype(np.float32)


def play_multi(spectra_and_ratios, ref_freq):
    signal = synthesize_multi(spectra_and_ratios, ref_freq)
    sd.stop()
    sd.play(signal, SAMPLE_RATE)


def write_wav(path, signal, sample_rate=SAMPLE_RATE):
    """Write a mono 24-bit PCM WAV (lossless) from a float signal in [-1, 1]."""
    clipped = np.clip(np.asarray(signal, dtype=np.float64), -1.0, 1.0)
    # 2**23 - 1 keeps +1.0 from wrapping to the most negative code.
    ints = np.rint(clipped * (2 ** 23 - 1)).astype('<i4')
    packed = ints.view(np.uint8).reshape(-1, 4)[:, :3]  # little-endian low 3 bytes
    with wave.open(str(path), 'wb') as fh:
        fh.setnchannels(1)
        fh.setsampwidth(3)
        fh.setframerate(sample_rate)
        fh.writeframes(np.ascontiguousarray(packed).tobytes())


# ═══════════════════════════════════════════════════════════════════════════
#  Interval labelling and presets
# ═══════════════════════════════════════════════════════════════════════════

INTERVAL_LABELS = [
    'unison', 'minor 2nd', 'major 2nd', 'minor 3rd', 'major 3rd',
    'perfect 4th', 'tritone', 'perfect 5th', 'minor 6th', 'major 6th',
    'minor 7th', 'major 7th', 'octave'
]


def ratio_to_interval_name(ratio):
    """Label with nearest 12-TET interval and show cents deviation."""
    cents = 1200 * np.log2(ratio)
    nearest_semitone = round(cents / 100)
    deviation_cents = cents - nearest_semitone * 100

    if 0 <= nearest_semitone < len(INTERVAL_LABELS):
        name = INTERVAL_LABELS[nearest_semitone]
    else:
        name = f"{nearest_semitone} semitones"

    if abs(deviation_cents) < 2:
        return name
    elif deviation_cents > 0:
        return f"{name} +{deviation_cents:.0f}\u00a2"
    else:
        return f"{name} {deviation_cents:.0f}\u00a2"


def ratio_to_cents(ratio):
    return 1200 * np.log2(ratio)


PRESETS = [
    ("Harmonic", [1, 2, 3, 4, 5, 6], [1, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 6]),
    ("Pan Flute (Odd Harmonics)", [1, 3, 5, 7, 9, 11],
     [1, 1 / 3, 1 / 5, 1 / 7, 1 / 9, 1 / 11]),
    ("Glockenspiel (Ideal Bar)", [1, 2.758, 5.406, 8.936, 13.35, 18.645],
     [0.88 ** i for i in range(6)]),
    ("Bonang", [1, 1.52, 3.46, 3.92], [0.88 ** i for i in range(4)]),
    ("Saron (Swastigitha)", [1, 2.76, 4.72, 5.92], [0.88 ** i for i in range(4)]),
    ("Saron (Kyai Kaduk Manis)", [1, 2.39, 2.78, 4.75, 5.08, 5.96],
     [0.88 ** i for i in range(6)]),
]

PRESET_NOTES = {
    "Harmonic":
        "Harmonic partials 1\u20136 at 1/n: an ideal sawtooth, not a measured instrument.",
    "Pan Flute (Odd Harmonics)":
        "Odd partials 1\u201311, as in Sethares' measured pan flute (TTSS Fig. 6.8); "
        "clarinets are similar. Amplitudes are 1/n (an ideal square wave), not measured.",
    "Glockenspiel (Ideal Bar)":
        "Free\u2013free Euler\u2013Bernoulli bar (mode ratios exact to <0.06%), "
        "like a glockenspiel bar. Real xylophone and marimba bars are undercut to retune "
        "these. Amplitudes are Sethares' generic 0.88\u207f roll-off.",
    "Bonang":
        "Sethares' generic Javanese bonang kettle (TTSS \u00a710.3.3). Gamelans are "
        "individually tuned, so no spectrum is canonical. Amplitudes are a generic "
        "0.88\u207f roll-off, not measured; Sethares uses equal amplitudes (\u00a710.5.1).",
    "Saron (Swastigitha)":
        "Sethares' generic saron key for gamelan Swastigitha, the median of its consistent "
        "partials (TTSS \u00a710.3.1). Near an ideal bar but with lower upper partials; he "
        "pairs it with a harmonic tone to explain that gamelan's pelog (Fig. 10.11). "
        "Amplitudes are a generic 0.88\u207f roll-off, not measured.",
    "Saron (Kyai Kaduk Manis)":
        "Sethares' generic saron for gamelan Kyai Kaduk Manis (1997), where the partials "
        "near 2.7 and 4.8 split into close pairs, apparently by design (TTSS \u00a710.3.1). "
        "Explains that gamelan's pelog with a harmonic tone (Fig. 10.12). "
        "Amplitudes are a generic 0.88\u207f roll-off, not measured.",
}


def format_partials(values):
    """The comma-separated text the spectrum fields use for a list of numbers."""
    return ", ".join(f"{v:.4g}" for v in values)


# ═══════════════════════════════════════════════════════════════════════════
#  Theme
# ═══════════════════════════════════════════════════════════════════════════

DARK_PALETTE = dict(
    BASE="#11111b", MANTLE="#181825", SURFACE="#1e1e2e",
    PANEL="#313244", LINE="#26263a", BORDER="#45475a", TEXT="#cdd6f4",
    SUBTEXT="#a6adc8", MUTED="#6c7086", BLUE="#89b4fa",
    YELLOW="#f9e2af", GREEN="#a6e3a1", RED="#f38ba8",
    CMAP="magma", PLOT_LW=2.0, GRID_ALPHA=0.12,
    MARKER_EDGE="#ffffff", TICK_LEVEL=2,
    VOICE_A="#5aa9ff", VOICE_B="#ff8a3d", VOICE_C="#3ddc97",
)

# Okabe-Ito colourblind-safe accents on soft greys; figure exports still render on white.
PAPER_PALETTE = dict(
    BASE="#e1e3e7", MANTLE="#ebecef", SURFACE="#f2f3f5",
    PANEL="#cdd0d6", LINE="#d2d5da", BORDER="#868b94", TEXT="#24262b",
    SUBTEXT="#3a3d44", MUTED="#686d75", BLUE="#0072b2",
    YELLOW="#d55e00", GREEN="#009e73", RED="#b22222",
    CMAP="viridis", PLOT_LW=1.6, GRID_ALPHA=0.25,
    MARKER_EDGE="#24262b", TICK_LEVEL=0,
    VOICE_A="#0072b2", VOICE_B="#d55e00", VOICE_C="#009e73",
)

# Declared explicitly so static analysis sees them; apply_palette() rebinds them.
BASE = DARK_PALETTE["BASE"]
MANTLE = DARK_PALETTE["MANTLE"]
SURFACE = DARK_PALETTE["SURFACE"]
PANEL = DARK_PALETTE["PANEL"]
LINE = DARK_PALETTE["LINE"]
BORDER = DARK_PALETTE["BORDER"]
TEXT = DARK_PALETTE["TEXT"]
SUBTEXT = DARK_PALETTE["SUBTEXT"]
MUTED = DARK_PALETTE["MUTED"]
BLUE = DARK_PALETTE["BLUE"]
YELLOW = DARK_PALETTE["YELLOW"]
GREEN = DARK_PALETTE["GREEN"]
RED = DARK_PALETTE["RED"]
CMAP = DARK_PALETTE["CMAP"]
PLOT_LW = DARK_PALETTE["PLOT_LW"]
GRID_ALPHA = DARK_PALETTE["GRID_ALPHA"]
MARKER_EDGE = DARK_PALETTE["MARKER_EDGE"]
TICK_LEVEL = DARK_PALETTE["TICK_LEVEL"]

# Blue / orange / green: the most separable Okabe-Ito triple, colour-blind safe.
VOICE_COLORS = {k: DARK_PALETTE[f"VOICE_{k}"] for k in "ABC"}
CURRENT_PALETTE = DARK_PALETTE


def apply_palette(palette):
    """Rebind the module-level colour names; widgets then re-read them."""
    global CURRENT_PALETTE
    CURRENT_PALETTE = palette
    globals().update(palette)
    VOICE_COLORS.update({k: palette[f"VOICE_{k}"] for k in "ABC"})


QSS = """
* {{ outline: none; }}

QWidget {{
    background: {BASE};
    color: {TEXT};
    font-family: "Segoe UI Variable Display", "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}}

QLabel {{ background: transparent; }}
#plain {{ background: transparent; }}

#cardTitle  {{ font-size: 10px; font-weight: 700; color: {MUTED}; letter-spacing: 1.6px; }}
#fieldLabel {{ font-size: 11px; color: {SUBTEXT}; font-weight: 600; }}
#hint       {{ font-size: 11px; color: {MUTED}; }}
#statusText {{ font-size: 12px; color: {SUBTEXT}; }}
#emptyText  {{ font-size: 12px; color: {MUTED}; padding: 18px; }}

#card {{
    background: {MANTLE};
    border: 1px solid {LINE};
    border-radius: 12px;
}}

#plotCard {{
    background: {MANTLE};
    border: 1px solid {LINE};
    border-radius: 14px;
}}

QLineEdit, QComboBox, QDoubleSpinBox {{
    background: {SURFACE};
    border: 1px solid {PANEL};
    border-radius: 8px;
    padding: 4px 8px;
    color: {TEXT};
    selection-background-color: {BLUE};
    selection-color: {BASE};
}}
QLineEdit:hover, QComboBox:hover, QDoubleSpinBox:hover {{ border-color: {BORDER}; }}
QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus {{ border-color: {BLUE}; }}
QLineEdit[invalid="true"] {{ border-color: {RED}; }}
QCheckBox {{ background: transparent; spacing: 8px; }}

QComboBox::drop-down {{ border: none; width: 24px; background: transparent; }}
QComboBox::down-arrow {{
    image: url("__CHEVRON__");
    width: 14px; height: 14px; margin-right: 8px;
}}
QComboBox::down-arrow:hover {{ image: url("__CHEVRON_HOVER__"); }}
QComboBox QAbstractItemView {{
    background: {SURFACE};
    border: 1px solid {PANEL};
    border-radius: 8px;
    padding: 4px;
    selection-background-color: {PANEL};
    selection-color: {BLUE};
    outline: none;
}}

QPushButton {{
    background: {SURFACE};
    border: 1px solid {PANEL};
    border-radius: 8px;
    padding: 5px 12px;
    color: {TEXT};
}}
QPushButton:hover  {{ background: {PANEL}; border-color: {BORDER}; }}
QPushButton:pressed {{ background: {BORDER}; }}

QPushButton#primary {{
    background: {BLUE}; color: {BASE}; border: none; font-weight: 700;
    padding: 6px 16px;
}}
QPushButton#primary:hover  {{ background: #a3c6ff; }}
QPushButton#primary:pressed {{ background: #6f9de6; }}
QPushButton#primary:disabled {{ background: {PANEL}; color: {MUTED}; }}

QPushButton#danger {{ background: {SURFACE}; color: {RED}; border: 1px solid #4a2f3c; }}
QPushButton#danger:hover {{ background: {RED}; color: {BASE}; border-color: {RED}; }}

QPushButton#icon {{
    background: {SURFACE}; border: 1px solid {PANEL}; border-radius: 8px;
    padding: 4px 0px; min-width: 32px; color: {GREEN}; font-size: 12px;
}}
QPushButton#icon:hover {{ background: {GREEN}; color: {BASE}; border-color: {GREEN}; }}

#segment {{ background: {MANTLE}; border: 1px solid {LINE}; border-radius: 10px; }}
QPushButton#segBtn {{
    background: transparent; border: none; border-radius: 7px;
    padding: 5px 14px; color: {MUTED}; font-weight: 600;
}}
QPushButton#segBtn:hover {{ color: {TEXT}; }}
QPushButton#segBtn:checked {{ background: {PANEL}; color: {BLUE}; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px 2px 2px 0; }}
QScrollBar::handle:vertical {{ background: {PANEL}; border-radius: 5px; min-height: 36px; }}
QScrollBar::handle:vertical:hover {{ background: {BORDER}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0 2px 2px 2px; }}
QScrollBar::handle:horizontal {{ background: {PANEL}; border-radius: 5px; min-width: 36px; }}
QScrollBar::handle:horizontal:hover {{ background: {BORDER}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QListWidget {{ background: transparent; border: none; padding: 2px; }}
QListWidget::item {{ border-radius: 9px; margin: 2px 3px; }}
QListWidget::item:hover {{ background: {SURFACE}; }}
QListWidget::item:selected {{ background: {PANEL}; }}

QProgressBar {{ background: {SURFACE}; border: none; border-radius: 3px; }}
QProgressBar::chunk {{ background: {BLUE}; border-radius: 3px; }}

QSplitter::handle {{ background: transparent; }}
QSplitter::handle:hover {{ background: {PANEL}; }}

QToolTip {{
    background: {SURFACE}; color: {TEXT}; border: 1px solid {PANEL};
    padding: 6px 8px; border-radius: 6px;
}}
"""


def _chevron(color, hover=False):
    """Render a small chevron to a temp PNG and return a QSS-friendly path."""
    # Colour is in the filename so a theme switch can't reuse a cached icon.
    name = f"dissonance_chevron_{color.lstrip('#')}_{'hover' if hover else 'idle'}.png"
    path = Path(tempfile.gettempdir()) / name
    img = QImage(28, 28, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    pen = QPen(QColor(color))
    pen.setWidthF(3.0)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.drawPolyline([QPointF(8, 11), QPointF(14, 18), QPointF(20, 11)])
    p.end()
    img.save(str(path))
    return path.as_posix()


def build_qss(palette=None):
    pal = palette or CURRENT_PALETTE
    return (QSS.format(**pal)
               .replace("__CHEVRON__", _chevron(pal["SUBTEXT"]))
               .replace("__CHEVRON_HOVER__", _chevron(pal["BLUE"], hover=True)))


def mono_font(size=12, bold=False):
    f = QFont("Cascadia Mono")
    if not f.exactMatch():
        f = QFont("Consolas")
    f.setPointSize(size)
    f.setBold(bold)
    return f


def diss_color(value):
    """Green (consonant) -> yellow -> red (dissonant)."""
    v = float(np.clip(value, 0.0, 1.0))
    stops = [(0.0, QColor(GREEN)), (0.45, QColor(YELLOW)), (1.0, QColor(RED))]
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        if v <= p1:
            t = 0.0 if p1 == p0 else (v - p0) / (p1 - p0)
            return QColor(int(c0.red() + (c1.red() - c0.red()) * t),
                          int(c0.green() + (c1.green() - c0.green()) * t),
                          int(c0.blue() + (c1.blue() - c0.blue()) * t))
    return QColor(RED)


# ═══════════════════════════════════════════════════════════════════════════
#  Small reusable widgets
# ═══════════════════════════════════════════════════════════════════════════

class Card(QFrame):
    """Rounded surface with a small-caps title."""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 9, 12, 10)
        outer.setSpacing(7)

        lbl = QLabel(title.upper())
        lbl.setObjectName("cardTitle")
        outer.addWidget(lbl)

        self.body = QVBoxLayout()
        self.body.setSpacing(6)
        outer.addLayout(self.body)


class Segmented(QWidget):
    """Pill-style segmented control."""

    changed = Signal(int)

    def __init__(self, labels, parent=None):
        super().__init__(parent)
        self.setObjectName("segment")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(3)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        for i, text in enumerate(labels):
            btn = QPushButton(text)
            btn.setObjectName("segBtn")
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            if i == 0:
                btn.setChecked(True)
            self.group.addButton(btn, i)
            lay.addWidget(btn)
        self.group.idClicked.connect(self.changed.emit)


class Field(QWidget):
    """Labelled line edit; *inline* puts the label to the left."""

    def __init__(self, label, value="", inline=False, label_width=74, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self) if inline else QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8 if inline else 3)
        cap = QLabel(label)
        cap.setObjectName("fieldLabel")
        if inline:
            cap.setFixedWidth(label_width)
        lay.addWidget(cap)
        self.edit = QLineEdit(value)
        lay.addWidget(self.edit)

    def text(self):
        return self.edit.text()

    def setInvalid(self, invalid):
        self.edit.setProperty("invalid", "true" if invalid else "false")
        self.edit.style().unpolish(self.edit)
        self.edit.style().polish(self.edit)


class ResultRow(QWidget):
    """One line in the intervals / triads list."""

    def __init__(self, primary, secondary, value, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(10)

        col = QVBoxLayout()
        col.setSpacing(2)
        p = QLabel(primary)
        p.setFont(mono_font(10, bold=True))
        p.setStyleSheet(f"color: {TEXT};")
        s = QLabel(secondary)
        s.setObjectName("hint")
        col.addWidget(p)
        col.addWidget(s)
        lay.addLayout(col, 1)

        right = QVBoxLayout()
        right.setSpacing(4)
        right.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        c = diss_color(value)
        v = QLabel(f"{value:.3f}")
        v.setFont(mono_font(10, bold=True))
        v.setStyleSheet(f"color: {c.name()};")
        v.setAlignment(Qt.AlignRight)
        bar = QProgressBar()
        bar.setRange(0, 1000)
        bar.setValue(int(np.clip(value, 0, 1) * 1000))
        bar.setTextVisible(False)
        bar.setFixedSize(62, 4)
        bar.setStyleSheet(
            f"QProgressBar {{ background: {SURFACE}; border-radius: 2px; }}"
            f"QProgressBar::chunk {{ background: {c.name()}; border-radius: 2px; }}")
        right.addWidget(v)
        right.addWidget(bar, 0, Qt.AlignRight)
        lay.addLayout(right, 0)


class ResultList(QListWidget):
    """List of ResultRow widgets; emits the payload of the activated row."""

    activatedRow = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QListWidget.SingleSelection)
        self.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self.setUniformItemSizes(False)
        self.itemClicked.connect(self._on_item)

    def _on_item(self, item):
        payload = item.data(Qt.UserRole)
        if payload is not None:
            self.activatedRow.emit(payload)

    def setRows(self, rows, empty_text="Nothing to show yet"):
        self.clear()
        if not rows:
            item = QListWidgetItem()
            item.setFlags(Qt.NoItemFlags)
            lbl = QLabel(empty_text)
            lbl.setObjectName("emptyText")
            lbl.setAlignment(Qt.AlignCenter)
            item.setSizeHint(QSize(10, 60))
            self.addItem(item)
            self.setItemWidget(item, lbl)
            return
        for primary, secondary, value, payload in rows:
            widget = ResultRow(primary, secondary, value)
            item = QListWidgetItem()
            item.setSizeHint(widget.sizeHint())
            item.setData(Qt.UserRole, payload)
            self.addItem(item)
            self.setItemWidget(item, widget)


# ═══════════════════════════════════════════════════════════════════════════
#  Spectrum editor
# ═══════════════════════════════════════════════════════════════════════════

class SpectrumCard(QWidget):
    """Editor for one voice's spectrum, with a live partial preview."""

    playRequested = Signal(object)
    sourceChanged = Signal()

    def __init__(self, letter, copy_sources=None, parent=None):
        super().__init__(parent)
        self.letter = letter
        self.copy_sources = copy_sources or []
        self.color = VOICE_COLORS[letter]
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        top = QHBoxLayout()
        top.setSpacing(6)
        self.combo = QComboBox()
        for i, (name, _) in enumerate(self.copy_sources):
            self.combo.addItem(f"Same as {name}", f"link:{i}")
            self.combo.setItemData(self.combo.count() - 1,
                                   f"Mirror voice {name}; editing switches to Custom",
                                   Qt.ToolTipRole)
        for i, (name, _, _) in enumerate(PRESETS):
            self.combo.addItem(name, f"preset:{i}")
            self.combo.setItemData(self.combo.count() - 1, PRESET_NOTES[name],
                                   Qt.ToolTipRole)
        self.combo.addItem("Custom", "custom")
        self.combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo.setMinimumContentsLength(10)
        top.addWidget(self.combo, 1)
        play_btn = QPushButton("\u25b6")
        play_btn.setObjectName("icon")
        play_btn.setToolTip(f"Play voice {letter} alone")
        play_btn.setCursor(Qt.PointingHandCursor)
        play_btn.clicked.connect(lambda: self.playRequested.emit(self))
        top.addWidget(play_btn, 0)
        lay.addLayout(top)

        _, freqs, amps = PRESETS[0]
        self.freq_field = Field("Partials", format_partials(freqs), inline=True)
        self.amp_field = Field("Amplitudes", format_partials(amps), inline=True)
        lay.addWidget(self.freq_field)
        lay.addWidget(self.amp_field)

        self.preview = pg.PlotWidget()
        self.preview.setFixedHeight(34)
        self.preview.setMouseEnabled(False, False)
        self.preview.setMenuEnabled(False)
        self.preview.hideButtons()
        self.preview.setBackground(SURFACE)
        for axis in ("left", "bottom", "top", "right"):
            self.preview.getPlotItem().hideAxis(axis)
        self.preview.getPlotItem().setContentsMargins(6, 4, 6, 2)
        self._bars = pg.BarGraphItem(x=[0], height=[0], width=0.06,
                                     brush=pg.mkBrush(self.color))
        self.preview.addItem(self._bars)
        lay.addWidget(self.preview)
        lay.addStretch(1)

        self.freq_field.edit.textChanged.connect(self._on_text_changed)
        self.amp_field.edit.textChanged.connect(self._on_text_changed)
        # textEdited fires only for typing, not for text mirrored from the source.
        self.freq_field.edit.textEdited.connect(self._detach)
        self.amp_field.edit.textEdited.connect(self._detach)
        for _, card in self.copy_sources:
            card.freq_field.edit.textChanged.connect(self._sync_from_source)
            card.amp_field.edit.textChanged.connect(self._sync_from_source)

        if not self.copy_sources:
            self._select_by_text()
        self._on_combo()
        self.combo.currentIndexChanged.connect(self._on_combo)

    # ── behaviour ──────────────────────────────────────────────────────────

    def apply_palette(self):
        self.color = VOICE_COLORS[self.letter]
        self.preview.setBackground(SURFACE)
        self._bars.setOpts(brush=pg.mkBrush(self.color))

    def _kind(self):
        kind, _, i = (self.combo.currentData() or "custom").partition(":")
        return kind, int(i) if i else None

    def is_linked(self):
        return self._kind()[0] == "link"

    def _source_card(self):
        kind, i = self._kind()
        return self.copy_sources[i][1] if kind == "link" else None

    def source_name(self):
        kind, i = self._kind()
        return self.copy_sources[i][0] if kind == "link" else None

    def _matched_preset(self):
        current = ("".join(self.freq_field.text().split()),
                   "".join(self.amp_field.text().split()))
        for i, (_, freqs, amps) in enumerate(PRESETS):
            if current == (format_partials(freqs).replace(" ", ""),
                           format_partials(amps).replace(" ", "")):
                return i
        return None

    def _select_by_text(self):
        """Point the menu at the preset the fields match, or Custom."""
        i = self._matched_preset()
        target = f"preset:{i}" if i is not None else "custom"
        self.combo.blockSignals(True)
        self.combo.setCurrentIndex(self.combo.findData(target))
        self.combo.blockSignals(False)
        self._update_tooltip()

    def _update_tooltip(self):
        tip = self.combo.itemData(self.combo.currentIndex(), Qt.ToolTipRole)
        self.combo.setToolTip(tip or "Spectrum source")

    def _sync_from_source(self):
        src = self._source_card()
        if src is not None:
            self.freq_field.edit.setText(src.freq_field.text())
            self.amp_field.edit.setText(src.amp_field.text())

    def _detach(self):
        if self.is_linked():
            self._select_by_text()
            self.sourceChanged.emit()

    def _on_text_changed(self):
        self._refresh_preview()
        if not self.is_linked():
            self._select_by_text()

    def _on_combo(self, _index=None):
        kind, i = self._kind()
        if kind == "link":
            self._sync_from_source()
        elif kind == "preset":
            _, freqs, amps = PRESETS[i]
            self.freq_field.edit.setText(format_partials(freqs))
            self.amp_field.edit.setText(format_partials(amps))
        self._update_tooltip()
        self._refresh_preview()
        self.sourceChanged.emit()

    def _refresh_preview(self):
        spec = self.get_spectrum()
        ok = spec is not None
        self.freq_field.setInvalid(not ok)
        self.amp_field.setInvalid(not ok)
        if not ok:
            self._bars.setOpts(x=[0], height=[0], width=0.06)
            return
        f = np.asarray(spec["freq"], dtype=float)
        a = np.abs(np.asarray(spec["amp"], dtype=float))
        peak = a.max() if a.size and a.max() > 0 else 1.0
        span = f.max() if f.size else 1.0
        self._bars.setOpts(x=f, height=a / peak, width=max(span * 0.012, 0.02),
                           brush=pg.mkBrush(self.color))
        self.preview.setXRange(0, span * 1.05, padding=0)
        self.preview.setYRange(0, 1.08, padding=0)

    def get_spectrum(self):
        """Spectrum dict, copied source, or None when the text is invalid."""
        src = self._source_card()
        if src is not None:
            return src.get_spectrum()
        try:
            freqs = [float(x.strip()) for x in self.freq_field.text().split(',')]
            amps = [float(x.strip()) for x in self.amp_field.text().split(',')]
        except ValueError:
            return None
        if not freqs or len(freqs) != len(amps):
            return None
        return {"freq": freqs, "amp": amps}


# ═══════════════════════════════════════════════════════════════════════════
#  Plot views
# ═══════════════════════════════════════════════════════════════════════════

class SmoothImageItem(pg.ImageItem):
    """ImageItem drawn with bilinear filtering when scaled up."""

    def paint(self, p, *args):
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        super().paint(p, *args)


class CornerZoomViewBox(pg.ViewBox):
    """Zooms (wheel and right-drag) about the bottom-left corner, so only the top and right move."""

    def scaleBy(self, s=None, center=None, x=None, y=None):
        r = self.targetRect().normalized()
        super().scaleBy(s, pg.Point(r.left(), r.top()), x, y)


class CurveView(pg.PlotWidget):
    """2D dissonance curve with crosshair readout."""

    picked = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent, viewBox=CornerZoomViewBox())
        pi = self.getPlotItem()
        pi.setMenuEnabled(False)
        pi.setMouseEnabled(x=True, y=False)

        self._curve = pi.plot([], [], fillLevel=0.0)
        self._dots = pg.ScatterPlotItem(size=9)
        pi.addItem(self._dots)

        self._vline = pg.InfiniteLine(angle=90, movable=False)
        self._vline.setVisible(False)
        pi.addItem(self._vline, ignoreBounds=True)

        self._readout = pg.TextItem(anchor=(0, 1))
        self._readout.setVisible(False)
        pi.addItem(self._readout, ignoreBounds=True)

        self._labels = []
        self._x = None
        self._y = None
        self._mx = None
        self._my = None

        self.apply_palette()
        self.scene().sigMouseMoved.connect(self._on_move)
        self.scene().sigMouseClicked.connect(self._on_click)

    def apply_palette(self):
        self.setBackground(MANTLE)
        self.showGrid(x=True, y=True, alpha=GRID_ALPHA)
        pi = self.getPlotItem()
        pi.setLabel('bottom', 'Frequency ratio  B / A', color=SUBTEXT, size='10pt')
        pi.setLabel('left', 'Sensory dissonance (normalised)', color=SUBTEXT,
                    size='10pt')
        for axis in ('bottom', 'left'):
            ax = pi.getAxis(axis)
            ax.setPen(pg.mkPen(BORDER))
            ax.setTextPen(pg.mkPen(SUBTEXT))
            # Paper mode drops the sub-minor grid, which prints as noise.
            ax.setStyle(maxTickLevel=TICK_LEVEL)

        fill = QColor(BLUE)
        fill.setAlpha(32)
        self._curve.setPen(pg.mkPen(BLUE, width=PLOT_LW))
        self._curve.setBrush(pg.mkBrush(fill))
        self._dots.setPen(pg.mkPen(MARKER_EDGE, width=1.0))
        self._dots.setBrush(pg.mkBrush(YELLOW))
        self._vline.setPen(pg.mkPen(MUTED, width=1, style=Qt.DashLine))

        box = QColor(SURFACE)
        box.setAlpha(235)
        self._readout.fill = pg.mkBrush(box)
        self._readout.border = pg.mkPen(PANEL)
        self._readout.setColor(TEXT)
        for t in self._labels:
            t.setColor(YELLOW)

    def setData(self, ratios, values, minima_x, minima_y, xmax):
        self._x, self._y = ratios, values
        self._mx, self._my = minima_x, minima_y
        self._curve.setData(ratios, values)
        self.getPlotItem().vb.setLimits(xMin=float(ratios[0]), yMin=0.0)
        self._rebuild_labels()
        self.setViewRatioRange(xmax)

    def _rebuild_labels(self):
        pi = self.getPlotItem()
        for item in self._labels:
            pi.removeItem(item)
        self._labels.clear()
        if self._mx is None or not len(self._mx):
            self._dots.setData([], [])
            return
        self._dots.setData(self._mx, self._my)
        for x, y in zip(self._mx, self._my):
            t = pg.TextItem(f"{ratio_to_interval_name(x)}\n{x:.3f}",
                            anchor=(0.5, 0.0), color=YELLOW)
            t.setFont(QFont("Segoe UI", 7))
            t.setPos(x, y - 0.012)
            pi.addItem(t, ignoreBounds=True)
            self._labels.append(t)

    def setViewRatioRange(self, xmax):
        """Zoom to ratios 0.9..xmax and rescale y to what's actually visible."""
        if self._x is None or not len(self._x):
            return
        self.setXRange(float(self._x[0]), xmax, padding=0)
        visible = self._y[self._x <= xmax]
        top = float(np.max(visible)) if len(visible) else float(np.max(self._y))
        self.setYRange(0, top * 1.08, padding=0)
        if self._mx is not None and len(self._mx) == len(self._labels):
            for x, t in zip(self._mx, self._labels):
                t.setVisible(bool(x <= xmax))

    def _nearest(self, x):
        if self._x is None or not len(self._x):
            return None
        i = int(np.argmin(np.abs(self._x - x)))
        return self._x[i], self._y[i]

    def _on_move(self, pos):
        if self._x is None or not self.getPlotItem().sceneBoundingRect().contains(pos):
            self._vline.setVisible(False)
            self._readout.setVisible(False)
            return
        p = self.getPlotItem().vb.mapSceneToView(pos)
        near = self._nearest(p.x())
        if near is None:
            return
        x, y = near
        self._vline.setPos(x)
        self._vline.setVisible(True)
        self._readout.setHtml(
            f"<div style='font-family:Consolas;font-size:11px;line-height:150%'>"
            f"<b style='color:{BLUE}'>{x:.4f}</b> &nbsp;"
            f"<span style='color:{MUTED}'>{ratio_to_cents(x):.0f}\u00a2</span><br>"
            f"<span style='color:{SUBTEXT}'>{ratio_to_interval_name(x)}</span><br>"
            f"<span style='color:{diss_color(y).name()}'>diss {y:.3f}</span></div>")
        self._readout.setPos(x, y)
        self._readout.setVisible(True)

    def _on_click(self, ev):
        if ev.button() != Qt.LeftButton:
            return
        vb = self.getPlotItem().vb
        if not self.getPlotItem().sceneBoundingRect().contains(ev.scenePos()):
            return
        p = vb.mapSceneToView(ev.scenePos())
        self.picked.emit(float(p.x()))

    def hideCrosshair(self):
        self._vline.setVisible(False)
        self._readout.setVisible(False)


class OvertoneView(pg.PlotWidget):
    """Partials of the last sound on one log-frequency axis; partials of
    different voices that coincide get a ring."""

    MATCH_CENTS = 5.0
    WIDTHS = (6.0, 3.5, 1.8)          # nested so coinciding stems stay visible

    def __init__(self, parent=None):
        super().__init__(parent)
        pi = self.getPlotItem()
        pi.setMenuEnabled(False)
        pi.hideButtons()
        pi.setMouseEnabled(False, False)
        pi.hideAxis('left')
        pi.layout.setContentsMargins(4, 2, 10, 0)
        self._voices = None
        self._ref = None
        self.apply_palette()

    def apply_palette(self):
        self.setBackground(MANTLE)
        ax = self.getPlotItem().getAxis('bottom')
        ax.setPen(pg.mkPen(BORDER))
        ax.setTextPen(pg.mkPen(SUBTEXT))
        if self._voices is not None:
            self.setVoices(self._voices, self._ref)

    def setVoices(self, voices, ref_freq):
        """*voices*: list of (letter, spectrum, ratio). Returns the number of shared partials."""
        self._voices, self._ref = voices, ref_freq
        pi = self.getPlotItem()
        pi.clear()
        parts = []
        for letter, spec, ratio in voices:
            f = ref_freq * ratio * np.asarray(spec["freq"], dtype=float)
            a = np.abs(np.asarray(spec["amp"], dtype=float))
            a = a / a.max() if a.size and a.max() > 0 else a
            parts.append((letter, f, a))
        if not parts:
            return 0

        def lx(f):
            return np.log2(np.asarray(f) / ref_freq)

        matches = 0
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                fi, ai = parts[i][1], parts[i][2]
                fj, aj = parts[j][1], parts[j][2]
                F1, F2 = (g.ravel() for g in np.meshgrid(fi, fj, indexing='ij'))
                A1, A2 = (g.ravel() for g in np.meshgrid(ai, aj, indexing='ij'))
                cents = 1200 * np.abs(np.log2(F2 / F1))
                low = np.minimum(A1, A2)
                for k in np.nonzero((cents < self.MATCH_CENTS) & (low > 0.02))[0]:
                    pi.addItem(pg.ScatterPlotItem(
                        x=[float(lx(F1[k]))], y=[max(A1[k], A2[k]) + 0.12], size=9,
                        symbol='o', pen=pg.mkPen(TEXT, width=1.4), brush=pg.mkBrush(0, 0, 0, 0)))
                    matches += 1

        for k, (letter, f, a) in enumerate(parts):
            color = QColor(VOICE_COLORS[letter])
            color.setAlpha(200)
            x = np.repeat(lx(f), 2)
            y = np.column_stack([np.zeros_like(a), a]).ravel()
            pi.plot(x, y, connect='pairs',
                    pen=pg.mkPen(color, width=self.WIDTHS[min(k, 2)]))

        all_x = np.concatenate([lx(f) for _, f, _ in parts])
        lo, hi = float(all_x.min()), float(all_x.max())
        pad = max(0.08, 0.04 * (hi - lo))
        pi.setXRange(lo - pad, hi + pad, padding=0)
        pi.setYRange(0, 1.25, padding=0)
        octaves = range(int(np.floor(lo - pad)), int(np.ceil(hi + pad)) + 1)
        pi.getAxis('bottom').setTicks([
            [(n, f"{ref_freq * 2 ** n:.0f} Hz") for n in octaves],
            [(n + np.log2(h), "") for n in octaves for h in (1.25, 1.5, 1.75)],
        ])
        return matches


class HeatmapView(pg.PlotWidget):
    """Triad space as an image with iso-contours."""

    picked = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        pi = self.getPlotItem()
        pi.setMenuEnabled(False)
        pi.setAspectLocked(True)
        # Headroom so the top colour-bar tick label isn't clipped by the widget edge.
        pi.layout.setContentsMargins(1, 10, 1, 1)

        self.cmap = _theme_cmap()
        self.img = SmoothImageItem()
        self.img.setColorMap(self.cmap)
        pi.addItem(self.img)

        self.bar = _dissonance_colorbar(colorMap=self.cmap)
        self.bar.setImageItem(self.img, insert_in=pi)

        self._iso = []
        # Overlay stays white in both themes: minima sit in the dark end of the
        # colour map, so white reads better than the theme's text colour.
        self._dots = pg.ScatterPlotItem(size=12, pen=pg.mkPen('#ffffff', width=1.4),
                                        brush=pg.mkBrush(YELLOW))
        pi.addItem(self._dots)
        self._labels = []
        self._x = self._y = self._z = None

        self.apply_palette()
        self.scene().sigMouseClicked.connect(self._on_click)

    def apply_palette(self):
        self.setBackground(MANTLE)
        pi = self.getPlotItem()
        pi.setLabel('bottom', 'Voice B ratio  (B / A)', color=SUBTEXT, size='10pt')
        pi.setLabel('left', 'Voice C ratio  (C / A)', color=SUBTEXT, size='10pt')
        for axis in ('bottom', 'left'):
            ax = pi.getAxis(axis)
            ax.setPen(pg.mkPen(BORDER))
            ax.setTextPen(pg.mkPen(SUBTEXT))
        self.cmap = _theme_cmap()
        self.img.setColorMap(self.cmap)
        _style_colorbar(self.bar, self.cmap)
        self._dots.setBrush(pg.mkBrush(YELLOW))

    def setData(self, x, y, z, tri_x, tri_y, tri_z):
        self._x, self._y, self._z = x, y, z
        self.img.setImage(z, autoLevels=False)
        self.img.setRect(QRectF(x[0], y[0], x[-1] - x[0], y[-1] - y[0]))
        lo, hi = float(z.min()), float(z.max())
        self.img.setLevels((lo, hi))
        self.bar.setLevels((lo, hi))

        pi = self.getPlotItem()
        for it in self._iso:
            pi.removeItem(it)
        self._iso.clear()

        smooth = pg.gaussianFilter(z, (1.6, 1.6))
        for frac in np.linspace(0.12, 0.9, 7):
            level = lo + (hi - lo) * frac
            iso = pg.IsocurveItem(data=smooth, level=level,
                                  pen=pg.mkPen(255, 255, 255, 34, width=1))
            iso.setParentItem(self.img)
            iso.setZValue(5)
            self._iso.append(iso)

        for t in self._labels:
            pi.removeItem(t)
        self._labels.clear()

        if len(tri_x):
            self._dots.setData(tri_x, tri_y)
            for bx, cy in zip(tri_x, tri_y):
                t = pg.TextItem(f"{bx:.2f} \u00b7 {cy:.2f}", anchor=(0, 1),
                                color='#ffffff', fill=pg.mkBrush(0, 0, 0, 130))
                t.setFont(QFont("Segoe UI", 7, QFont.Bold))
                t.setPos(bx, cy)
                pi.addItem(t, ignoreBounds=True)
                self._labels.append(t)
        else:
            self._dots.setData([], [])

        pi.vb.setRange(xRange=(x[0], x[-1]), yRange=(y[0], y[-1]), padding=0)

    def _on_click(self, ev):
        if ev.button() != Qt.LeftButton or self._z is None:
            return
        pi = self.getPlotItem()
        if not pi.sceneBoundingRect().contains(ev.scenePos()):
            return
        p = pi.vb.mapSceneToView(ev.scenePos())
        if not (self._x[0] <= p.x() <= self._x[-1] and
                self._y[0] <= p.y() <= self._y[-1]):
            return
        self.picked.emit(float(p.x()), float(p.y()))


class ColorBarStrip(pg.GraphicsLayoutWidget):
    """Stand-alone dissonance colour bar styled like the heatmap's."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ci.layout.setContentsMargins(4, 10, 4, 10)
        self.bar = _dissonance_colorbar(values=(0.0, 1.0), colorMap=_theme_cmap())
        self.addItem(self.bar)
        self.setFixedWidth(84)
        self.apply_palette()

    def setLevels(self, lo, hi):
        self.bar.setLevels((lo, hi))

    def apply_palette(self):
        self.setBackground(MANTLE)
        _style_colorbar(self.bar, _theme_cmap())


def _dissonance_colorbar(**kwargs):
    bar = pg.ColorBarItem(interactive=False, label='Sensory dissonance', width=12, **kwargs)
    # Let the end tick labels draw into the margin instead of being hidden.
    bar.axis.setStyle(hideOverlappingLabels=False)
    return bar


def _style_colorbar(bar, cmap):
    bar.axis.setPen(pg.mkPen(BORDER))
    bar.axis.setTextPen(pg.mkPen(SUBTEXT))
    # ColorBarItem puts its tick values on one axis and the caption on the
    # opposite one, so the caption has to be recoloured separately.
    bar.getAxis('left').setLabel('Sensory dissonance', color=SUBTEXT)
    bar.setColorMap(cmap)


def _theme_cmap():
    # Name comes from the active palette; pyqtgraph bundles magma and viridis.
    for name in (CMAP, 'magma', 'CET-L8', 'CET-L16'):
        try:
            cmap = pg.colormap.get(name)
        except Exception:
            continue
        if cmap is not None:
            return cmap
    raise RuntimeError('pyqtgraph has no usable colormap installed')


def _to_unit(values, axis):
    """Map *values* on the ratio *axis* to GL scene coordinates in [-1, 1]."""
    lo, hi = axis[0], axis[-1]
    return (np.asarray(values, dtype=float) - lo) / max(hi - lo, 1e-12) * 2.0 - 1.0


if HAS_GL:

    from OpenGL import GL as _GL
    from PySide6 import QtOpenGL

    SURFACE_CAMERA = dict(distance=4.2, elevation=40.0, azimuth=68.0)
    # On-screen only: the look-at offset a middle-button drag would give, as
    # (right, down) in world units. Exports set their own centre and crop.
    SURFACE_PAN = (-0.08, 0.3)

    # Markers are drawn without depth testing so a consonant valley stays
    # visible even when a taller ridge sits between it and the camera.
    ALWAYS_ON_TOP = {
        _GL.GL_DEPTH_TEST: False,
        _GL.GL_BLEND: True,
        _GL.GL_ALPHA_TEST: False,
        _GL.GL_CULL_FACE: False,
        'glBlendFunc': (_GL.GL_SRC_ALPHA, _GL.GL_ONE_MINUS_SRC_ALPHA),
    }

    class SurfaceView(gl.GLViewWidget):
        """OpenGL dissonance surface with ray-cast click-to-play."""

        picked = Signal(float, float)

        Z_SCALE = 1.1

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setCameraPosition(**SURFACE_CAMERA)
            self.opts['fov'] = 50
            self._default_pan()

            self._grid = gl.GLGridItem()
            self._grid.setSize(2.0, 2.0)
            self._grid.setSpacing(0.2, 0.2)
            self.addItem(self._grid)

            self._surface = None
            self._dots = None
            self._text = []
            self._press = None
            self._x = self._y = self._z = None
            self._zs = None
            self._cmap = _theme_cmap()
            self.apply_palette()

        def apply_palette(self):
            """Background/grid/colormap only; call setData again to recolour."""
            self.setBackgroundColor(QColor(MANTLE))
            g = QColor(BORDER)
            g.setAlpha(110)
            self._grid.setColor(g)
            self._cmap = _theme_cmap()
            for t in self._text:
                t.setData(color=QColor(SUBTEXT))

        # ── data ───────────────────────────────────────────────────────────

        def setData(self, x, y, z, tri_x, tri_y, tri_z):
            self._x, self._y, self._z = x, y, z

            step = max(1, len(x) // 140)
            xs, ys = x[::step], y[::step]
            zs = np.ascontiguousarray(z[::step, ::step], dtype=np.float32)

            self._sx, self._sy = xs, ys
            self._zs = zs * self.Z_SCALE

            nx = self._norm(xs, x)
            ny = self._norm(ys, y)

            lo, hi = float(z.min()), float(z.max())
            rng = max(hi - lo, 1e-9)
            colors = self._cmap.map((zs - lo) / rng, mode='float')
            colors = np.ascontiguousarray(
                colors.reshape(-1, 4).astype(np.float32))

            if self._surface is not None:
                self.removeItem(self._surface)
            self._surface = gl.GLSurfacePlotItem(
                x=nx.astype(np.float32), y=ny.astype(np.float32),
                z=zs * self.Z_SCALE, colors=colors,
                shader='shaded', smooth=True, glOptions='opaque')
            self.addItem(self._surface)

            if self._dots is not None:
                self.removeItem(self._dots)
                self._dots = None
            if len(tri_x):
                pts = np.column_stack([
                    self._norm(np.asarray(tri_x), x),
                    self._norm(np.asarray(tri_y), y),
                    np.asarray(tri_z) * self.Z_SCALE + 0.012,
                ]).astype(np.float32)
                c = np.array(QColor(YELLOW).getRgbF(), dtype=np.float32)
                self._dots = gl.GLScatterPlotItem(
                    pos=pts, size=14.0,
                    color=np.tile(c, (len(pts), 1)), pxMode=True)
                self._dots.setGLOptions(ALWAYS_ON_TOP)
                self._dots.setDepthValue(10)
                self.addItem(self._dots)

            self._rebuild_labels(x, y)

        def _rebuild_labels(self, x, y):
            for t in self._text:
                self.removeItem(t)
            self._text.clear()
            if not hasattr(gl, 'GLTextItem'):
                return
            font = QFont("Segoe UI", 8)
            for frac in (0.0, 0.5, 1.0):
                vx = x[0] + (x[-1] - x[0]) * frac
                vy = y[0] + (y[-1] - y[0]) * frac
                for pos, text in (
                    ((frac * 2 - 1, -1.12, 0.0), f"B {vx:.2f}"),
                    ((-1.16, frac * 2 - 1, 0.0), f"C {vy:.2f}"),
                ):
                    t = gl.GLTextItem(pos=np.array(pos, dtype=np.float32),
                                      text=text, color=QColor(SUBTEXT), font=font)
                    self.addItem(t)
                    self._text.append(t)

        _norm = staticmethod(_to_unit)

        # ── picking ────────────────────────────────────────────────────────

        def mousePressEvent(self, ev):
            self._press = ev.position()
            super().mousePressEvent(ev)

        def mouseReleaseEvent(self, ev):
            super().mouseReleaseEvent(ev)
            if self._press is None or ev.button() != Qt.LeftButton:
                self._press = None
                return
            delta = ev.position() - self._press
            self._press = None
            if abs(delta.x()) > 4 or abs(delta.y()) > 4:
                return
            hit = self._raycast(ev.position().x(), ev.position().y())
            if hit is not None:
                self.picked.emit(hit[0], hit[1])

        def _raycast(self, px, py):
            if self._zs is None:
                return None
            w, h = self.width(), self.height()
            if w <= 0 or h <= 0:
                return None
            ndc_x = 2.0 * px / w - 1.0
            ndc_y = 1.0 - 2.0 * py / h
            try:
                viewport = self.getViewport()
                proj = self.projectionMatrix(viewport, viewport)
                mvp = proj * self.viewMatrix()
                inv, ok = mvp.inverted()
            except Exception:
                return None
            if not ok:
                return None

            def unproject(z):
                v = inv.map(QVector4D(ndc_x, ndc_y, z, 1.0))
                if v.w() == 0:
                    return None
                return np.array([v.x() / v.w(), v.y() / v.w(), v.z() / v.w()])

            a = unproject(-1.0)
            b = unproject(1.0)
            if a is None or b is None:
                return None

            direction = b - a
            length = float(np.linalg.norm(direction))
            if length <= 0:
                return None
            direction /= length

            # Clip the ray to the surface's bounding box so the march stays fine
            zmax = float(self._zs.max())
            lo = np.array([-1.0, -1.0, -0.05])
            hi = np.array([1.0, 1.0, zmax + 0.05])
            with np.errstate(divide='ignore', invalid='ignore'):
                inv_d = 1.0 / direction
                t_lo = (lo - a) * inv_d
                t_hi = (hi - a) * inv_d
            t_lo = np.nan_to_num(t_lo, nan=-np.inf)
            t_hi = np.nan_to_num(t_hi, nan=np.inf)
            t_near = float(np.max(np.minimum(t_lo, t_hi)))
            t_far = float(np.min(np.maximum(t_lo, t_hi)))
            t_near = max(t_near, 0.0)
            t_far = min(t_far, length)
            if t_far <= t_near:
                return None

            ts = np.linspace(t_near, t_far, 2048)[:, None]
            pts = a[None, :] + direction[None, :] * ts

            sx, sy = self._sx, self._sy
            gx = (pts[:, 0] + 1.0) * 0.5 * (self._x[-1] - self._x[0]) + self._x[0]
            gy = (pts[:, 1] + 1.0) * 0.5 * (self._y[-1] - self._y[0]) + self._y[0]

            inside = ((pts[:, 0] >= -1.0) & (pts[:, 0] <= 1.0) &
                      (pts[:, 1] >= -1.0) & (pts[:, 1] <= 1.0))
            if not inside.any():
                return None

            ix = np.clip(np.searchsorted(sx, gx), 0, len(sx) - 1)
            iy = np.clip(np.searchsorted(sy, gy), 0, len(sy) - 1)
            surf = self._zs[ix, iy]
            below = inside & (pts[:, 2] <= surf)
            if not below.any():
                return None
            i = int(np.argmax(below))
            return float(np.clip(gx[i], self._x[0], self._x[-1])), \
                float(np.clip(gy[i], self._y[0], self._y[-1]))

        def resetView(self):
            self.setCameraPosition(**SURFACE_CAMERA)
            self._default_pan()

        def _default_pan(self):
            # Same basis GLViewWidget.pan(..., relative='view-upright') uses for middle-drag.
            self.opts['center'] = pg.Vector(0, 0, 0)
            look = self.opts['center'] - self.cameraPosition()
            up = pg.Vector(0, 0, 1)
            right = pg.Vector(QVector3D.crossProduct(up, look).normalized())
            down = pg.Vector(QVector3D.crossProduct(right, up).normalized())
            dx, dy = SURFACE_PAN
            self.opts['center'] = self.opts['center'] + right * dx + down * dy
            self.update()

else:  # pragma: no cover - fallback when OpenGL is unavailable

    class SurfaceView(QLabel):
        picked = Signal(float, float)

        def __init__(self, parent=None):
            super().__init__("3D surface unavailable \u2014 PyOpenGL could not be "
                             "initialised on this machine.", parent)
            self.setAlignment(Qt.AlignCenter)
            self.setObjectName("emptyText")

        def setData(self, *args, **kwargs):
            pass

        def apply_palette(self):
            pass

        def resetView(self):
            pass


# ═══════════════════════════════════════════════════════════════════════════
#  Background computation
# ═══════════════════════════════════════════════════════════════════════════

class ComputeWorker(QThread):
    progress = Signal(int)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, spec_a, spec_b, spec_c, ref_freq, max_interval,
                 step_2d, step_3d, parent=None):
        super().__init__(parent)
        self.args = (spec_a, spec_b, spec_c, ref_freq, max_interval,
                     step_2d, step_3d)

    def run(self):
        (spec_a, spec_b, spec_c, ref_freq, max_interval,
         step_2d, step_3d) = self.args
        try:
            ratios_2d, diss_2d = compute_dissonance_2d(
                spec_a, spec_b, ref_freq, max(CURVE_MAX, max_interval), step_2d)
            minima_x, minima_y = find_minima(ratios_2d, diss_2d)

            x3, y3, z3 = compute_dissonance_3d(
                spec_a, spec_b, spec_c, ref_freq, max_interval, step_3d,
                progress_callback=lambda p: self.progress.emit(int(p * 100)))

            tx, ty, tz = find_minima_3d(x3, y3, z3, (spec_a, spec_b, spec_c),
                                        ref_freq, top_n=None)
        except Exception as exc:  # surfaced in the status bar
            self.failed.emit(str(exc))
            return

        self.finished_ok.emit({
            "ratios_2d": ratios_2d, "diss_2d": diss_2d,
            "minima_x": minima_x, "minima_y": minima_y,
            "x_3d": x3, "y_3d": y3, "z_3d": z3,
            "tri_x": tx, "tri_y": ty, "tri_z": tz,
            "max_interval": max_interval,
        })


# ═══════════════════════════════════════════════════════════════════════════
#  Figure export  (publication figures)
# ═══════════════════════════════════════════════════════════════════════════

SCREEN_DPI = 96.0
FIG_FONTS = [("Serif \u2014 Times New Roman", "Times New Roman"),
             ("Sans \u2014 Arial", "Arial"),
             ("Sans \u2014 Calibri", "Calibri")]
FIG_CMAPS = [("Viridis", "viridis"), ("Cividis", "cividis"), ("Magma", "magma"),
             ("Inferno", "inferno"), ("Greyscale", "CET-L1")]
FIG_FORMATS = [("PDF \u2014 vector", "pdf", "PDF document (*.pdf)"),
               ("SVG \u2014 vector", "svg", "SVG vector (*.svg)"),
               ("PNG \u2014 raster", "png", "PNG image (*.png)")]
FIG_LABEL_MODES = ["None", "Ratio", "Interval name", "Both"]

FIGURE_COMMON = dict(
    font="Times New Roman", size=9.0, line=1.0, axis=0.8, ticks_in=True,
    frame=True, grid=False, fill=False, markers=True, cmap="viridis",
    contours=True, square=False, dpi=600, fmt="pdf", xmin=0.9, xmax=2.3, ymax=0.0,
)
FIGURE_DEFAULTS = {
    "curve": dict(title="", xlabel="Frequency ratio",
                  ylabel="Sensory dissonance (normalized)", cbar="",
                  labels="Ratio", width=6.0, height=3.5),
    "heatmap": dict(title="", xlabel="Voice B frequency ratio",
                    ylabel="Voice C frequency ratio",
                    cbar="Sensory dissonance (normalized)",
                    labels="None", width=4.8, height=4.0),
}


def _figure_label(x, mode):
    name = ratio_to_interval_name(x)
    return {"Ratio": f"{x:.3f}", "Interval name": name}.get(mode, f"{name}\n{x:.3f}")


def _triad_figure_label(bx, cy, mode, newline="\n"):
    ratios = f"{bx:.3f} : {cy:.3f}"
    if mode == "Ratio":
        return ratios
    names = f"{ratio_to_interval_name(bx)} \u00b7 {ratio_to_interval_name(cy)}"
    return names + newline + ratios if mode == "Both" else names


def _pad_axis_label(ax, gap):
    """Clear the axis title from the tick numbers; pyqtgraph reserves only 80 % of it."""
    def width():
        pg.AxisItem._updateWidth(ax)
        if ax.fixedWidth is None and ax.label.isVisible() and ax.style["showValues"]:
            w = ax.maximumWidth() + 0.2 * ax.label.boundingRect().height() + gap
            ax.setMaximumWidth(w)
            ax.setMinimumWidth(w)

    def height():
        pg.AxisItem._updateHeight(ax)
        if ax.fixedHeight is None and ax.label.isVisible() and ax.style["showValues"]:
            h = ax.maximumHeight() + 0.2 * ax.label.boundingRect().height() + gap
            ax.setMaximumHeight(h)
            ax.setMinimumHeight(h)

    ax._updateWidth = width
    ax._updateHeight = height


def make_figure(kind, data, st, ppi):
    """Build a white print-styled plot, st['width'] x st['height'] inches at *ppi*.

    All sizes are given in points and converted here, because pyqtgraph pens are
    cosmetic: scaling a finished render up would thicken the text but not the lines.
    """
    pt = ppi / 72.0
    fam = st["font"]
    black = "#000000"

    def font(scale=1.0):
        f = QFont(fam)
        f.setPixelSize(max(1, round(st["size"] * scale * pt)))
        return f

    css = {"color": black, "font-size": f"{max(1, round(st['size'] * pt))}px",
           "font-family": fam}
    axis_pen = pg.mkPen(black, width=st["axis"] * pt)
    tick_len = round(3.5 * pt) * (-1 if st["ticks_in"] else 1)

    def style_axis(ax, values=True):
        ax.setPen(axis_pen)
        ax.setTextPen(pg.mkPen(black))
        ax.setTickFont(font())
        ax.setStyle(tickLength=tick_len, tickTextOffset=round(2.5 * pt),
                    showValues=values, maxTickLevel=1)
        _pad_axis_label(ax, round(3 * pt))

    w = pg.PlotWidget(background="w")
    w.setFixedSize(round(st["width"] * ppi), round(st["height"] * ppi))
    w.setAntialiasing(True)
    w._keep = []
    pi = w.getPlotItem()
    pi.hideButtons()
    pi.setMenuEnabled(False)
    pi.setMouseEnabled(False, False)
    m = round(5 * pt)
    pi.layout.setContentsMargins(m, m, m, m)

    for name in ("left", "bottom", "top", "right"):
        if name in ("top", "right"):
            if not st["frame"]:
                pi.hideAxis(name)
                continue
            pi.showAxis(name)
        ax = pi.getAxis(name)
        style_axis(ax, values=name in ("left", "bottom"))
        ax.setGrid(int(0.2 * 255) if st["grid"] and name in ("left", "bottom") else False)

    for name, key in (("bottom", "xlabel"), ("left", "ylabel")):
        if st[key].strip():
            pi.setLabel(name, st[key], **css)
        else:
            pi.getAxis(name).showLabel(False)
    if st["title"].strip():
        tpx = max(1, round(st["size"] * 1.15 * pt))
        pi.setTitle(f"<span style='font-family:{fam}'>{st['title']}</span>",
                    color=black, size=f"{tpx}px")
        # pyqtgraph pins the title row at 30 px regardless of font size.
        pi.titleLabel.setMaximumHeight(tpx * 1.8)
        pi.layout.setRowFixedHeight(0, tpx * 1.8)

    if kind == "curve":
        r, d = data["ratios_2d"], data["diss_2d"]
        x0, x1 = st["xmin"], st["xmax"]
        vis = (r >= x0) & (r <= x1)
        top = st["ymax"] if st["ymax"] > 0 else (
            float(d[vis].max()) * 1.05 if vis.any() else 1.0)
        extra = dict(fillLevel=0.0, brush=pg.mkBrush(0, 0, 0, 30)) if st["fill"] else {}
        pi.plot(r, d, pen=pg.mkPen(black, width=st["line"] * pt), **extra)

        mx, my = data["minima_x"], data["minima_y"]
        keep = (mx >= x0) & (mx <= x1)
        mx, my = mx[keep], my[keep]
        if st["markers"] and len(mx):
            pi.addItem(pg.ScatterPlotItem(x=mx, y=my, size=4.5 * pt,
                                          pen=pg.mkPen(black, width=0.5 * pt),
                                          brush=pg.mkBrush("#d55e00")))
        if st["labels"] != "None":
            for x, y in zip(mx, my):
                # Labels under a minimum near zero would fall off the axis.
                low = y < 0.15 * top
                t = pg.TextItem(_figure_label(x, st["labels"]), color=black,
                                anchor=(0.5, 1.2) if low else (0.5, -0.15),
                                fill=pg.mkBrush(255, 255, 255, 220) if low else None)
                t.setFont(font(0.8))
                t.setPos(x, y)
                pi.addItem(t, ignoreBounds=True)
        pi.setXRange(x0, x1, padding=0)
        pi.setYRange(0, top, padding=0)
        return w

    x, y, z = data["x_3d"], data["y_3d"], data["z_3d"]
    lo, hi = float(z.min()), float(z.max())
    cmap = pg.colormap.get(st["cmap"])
    img = SmoothImageItem()
    img.setImage(z, autoLevels=False)
    img.setLevels((lo, hi))
    img.setColorMap(cmap)
    img.setRect(QRectF(x[0], y[0], x[-1] - x[0], y[-1] - y[0]))
    pi.addItem(img)
    if st["contours"]:
        smooth = pg.gaussianFilter(z, (1.6, 1.6))
        for frac in np.linspace(0.12, 0.9, 7):
            iso = pg.IsocurveItem(data=smooth, level=lo + (hi - lo) * frac,
                                  pen=pg.mkPen(255, 255, 255, 110, width=0.4 * pt))
            iso.setParentItem(img)
            iso.setZValue(5)
            w._keep.append(iso)

    tx, ty = data["tri_vis"][0], data["tri_vis"][1]
    if st["markers"] and len(tx):
        pi.addItem(pg.ScatterPlotItem(x=tx, y=ty, size=5 * pt,
                                      pen=pg.mkPen(black, width=0.5 * pt),
                                      brush=pg.mkBrush("#ffffff")))
    if st["labels"] != "None":
        for bx, cy in zip(tx, ty):
            t = pg.TextItem(_triad_figure_label(bx, cy, st["labels"]), color=black,
                            anchor=(-0.08, 1.08), fill=pg.mkBrush(255, 255, 255, 210))
            t.setFont(font(0.75))
            t.setPos(bx, cy)
            pi.addItem(t, ignoreBounds=True)

    bar = pg.ColorBarItem(colorMap=cmap, values=(lo, hi), interactive=False,
                          width=max(4, round(9 * pt)))
    bar.setImageItem(img, insert_in=pi)
    for name in ("left", "top", "bottom"):
        bar.getAxis(name).setPen(axis_pen)
    style_axis(bar.axis)
    bar.axis.setWidth(None)
    if st["cbar"].strip():
        bar.axis.setLabel(st["cbar"], **css)
    w._keep.append(bar)

    pi.setAspectLocked(st["square"])
    pi.setXRange(x[0], x[-1], padding=0)
    pi.setYRange(y[0], y[-1], padding=0)
    return w


def render_figure(kind, data, st, path):
    """Rebuild the figure natively at output resolution and write it to *path*."""
    fmt = st["fmt"]
    # Vector output is resolution-free; 96 keeps 1 px = 1/96 in so points stay exact.
    ppi = float(st["dpi"]) if fmt == "png" else SCREEN_DPI
    fig = make_figure(kind, data, st, ppi)
    fig.setAttribute(Qt.WA_DontShowOnScreen, True)
    fig.show()
    # Axes only grow to fit their tick text after a first paint, so paint twice.
    for _ in range(2):
        QApplication.processEvents()
        fig.grab()
    QApplication.processEvents()
    try:
        src = fig.viewport().rect()
        rect = QRectF(0, 0, src.width(), src.height())

        def paint(p):
            p.fillRect(rect, QColor("#ffffff"))
            fig.render(p, rect, src)

        _write_device(path, fmt, src.width(), src.height(), ppi, st, paint)
    finally:
        fig.close()
        fig.deleteLater()


def _write_device(path, fmt, w, h, ppi, st, paint):
    """Run *paint(painter)* on a w x h px canvas at *ppi* and save it as *fmt*."""
    def run(device):
        p = QPainter(device)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            paint(p)
        finally:
            p.end()

    if fmt == "png":
        image = QImage(w, h, QImage.Format_ARGB32)
        image.fill(QColor("#ffffff"))
        run(image)
        # DPI metadata only after painting: pyqtgraph axes lay out against the device DPI.
        dpm = round(ppi / 0.0254)
        image.setDotsPerMeterX(dpm)
        image.setDotsPerMeterY(dpm)
        if not image.save(str(path)):
            raise RuntimeError("Qt could not write the PNG")
    elif fmt == "svg":
        gen = QSvgGenerator()
        gen.setFileName(str(path))
        gen.setSize(QSize(w, h))
        gen.setViewBox(QRectF(0, 0, w, h))
        gen.setResolution(int(ppi))
        gen.setTitle("Dissonance Explorer figure")
        run(gen)
    elif fmt == "pdf":
        writer = QPdfWriter(str(path))
        writer.setResolution(int(ppi))
        writer.setPageSize(QPageSize(QSizeF(st["width"], st["height"]),
                                     QPageSize.Unit.Inch))
        writer.setPageMargins(QMarginsF(0, 0, 0, 0))
        run(writer)
    else:
        raise RuntimeError(f"unknown figure format {fmt!r}")


class FigureEditor(QDialog):
    """Live, true-size preview of a paper figure with editable text and style."""

    def __init__(self, kind, data, state, parent=None):
        super().__init__(parent)
        self.kind, self.data = kind, data
        self.state = dict(state)
        self.saved_path = None
        self.setWindowTitle(f"Figure editor \u2014 {kind}")
        avail = self.screen().availableGeometry()
        self.resize(min(1200, avail.width() - 60), min(760, avail.height() - 60))

        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        left = QVBoxLayout()
        left.setSpacing(6)
        self.canvas = QScrollArea()
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setStyleSheet("QScrollArea { background: #8a8a8a; border-radius: 8px; }")
        left.addWidget(self.canvas, 1)
        self.msg = QLabel()
        self.msg.setObjectName("hint")
        left.addWidget(self.msg)
        root.addLayout(left, 1)

        side = QWidget()
        form = QFormLayout(side)
        form.setContentsMargins(0, 0, 6, 0)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(5)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(side)
        scroll.setFixedWidth(330)
        right = QVBoxLayout()
        right.setSpacing(8)
        right.addWidget(scroll, 1)
        root.addLayout(right, 0)

        def section(text):
            lbl = QLabel(text)
            lbl.setObjectName("cardTitle")
            lbl.setContentsMargins(0, 8, 0, 0)
            form.addRow(lbl)

        self._widgets = {}
        self._build_form(form, section)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        save = QPushButton("Save\u2026")
        save.setObjectName("primary")
        save.setCursor(Qt.PointingHandCursor)
        save.clicked.connect(self._save)
        close = QPushButton("Close")
        close.setCursor(Qt.PointingHandCursor)
        close.clicked.connect(self.accept)
        buttons.addWidget(save, 1)
        buttons.addWidget(close, 0)
        right.addLayout(buttons)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._rebuild)
        self._rebuild()

    def _build_form(self, form, section):
        curve = self.kind == "curve"
        section("TEXT")
        form.addRow("Title", self._line("title"))
        form.addRow("X axis", self._line("xlabel"))
        form.addRow("Y axis", self._line("ylabel"))
        if not curve:
            form.addRow("Colour bar", self._line("cbar"))
        self._rich_text_tip(form)

        section("STYLE")
        form.addRow("Font", self._combo("font", FIG_FONTS))
        form.addRow("Font size (pt)", self._num("size", 5, 24, 0.5, 1))
        if curve:
            form.addRow("Line (pt)", self._num("line", 0.1, 6, 0.1, 2))
        form.addRow("Axes (pt)", self._num("axis", 0.1, 4, 0.1, 2))
        form.addRow("Ticks", self._combo("ticks_in", [("Inside", True), ("Outside", False)]))
        form.addRow(self._check("frame", "Box frame (top and right axes)"))
        form.addRow(self._check("grid", "Grid lines"))
        if curve:
            form.addRow(self._check("fill", "Shade under curve"))
            form.addRow(self._check("markers", "Mark minima"))
            form.addRow("Minima labels", self._combo(
                "labels", [(m, m) for m in FIG_LABEL_MODES]))
        else:
            form.addRow("Colour map", self._combo("cmap", FIG_CMAPS))
            form.addRow(self._check("contours", "Contour lines"))
            form.addRow(self._check("square", "Equal axis scaling"))
            form.addRow(self._check("markers", "Mark best triads"))
            form.addRow("Triad labels", self._combo(
                "labels", [(m, m) for m in FIG_LABEL_MODES]))

        if curve:
            section("RANGE")
            top = float(self.data["ratios_2d"][-1])
            form.addRow("X from", self._num("xmin", 0.5, top, 0.05, 3))
            form.addRow("X to", self._num("xmax", 0.5, top, 0.05, 3))
            form.addRow("Y max (0 = auto)", self._num("ymax", 0, 2, 0.05, 3))

        self._output_rows(form, section, "PNG resolution (dpi)")

    def _rich_text_tip(self, form):
        tip = QLabel("Rich text works: <i>…</i>, <sub>…</sub>, <sup>…</sup>")
        tip.setObjectName("hint")
        tip.setTextFormat(Qt.PlainText)
        form.addRow(tip)

    def _output_rows(self, form, section, dpi_label):
        section("OUTPUT")
        form.addRow("Width (in)", self._num("width", 1, 14, 0.1, 2))
        form.addRow("Height (in)", self._num("height", 1, 14, 0.1, 2))
        form.addRow("Format", self._combo("fmt", [(f[0], f[1]) for f in FIG_FORMATS]))
        form.addRow(dpi_label, self._num("dpi", 72, 1200, 50, 0))
        sizes = QLabel("Single column \u2248 3.5 in, double column \u2248 7 in")
        sizes.setObjectName("hint")
        form.addRow(sizes)

    # ── control factories ─────────────────────────────────────────────────

    def _set(self, key, value):
        self.state[key] = value
        self._timer.start()

    def _line(self, key):
        e = QLineEdit(self.state[key])
        e.textChanged.connect(lambda v: self._set(key, v))
        return e

    def _num(self, key, lo, hi, step, decimals):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(decimals)
        s.setSingleStep(step)
        s.setButtonSymbols(QDoubleSpinBox.NoButtons)
        s.setValue(float(self.state[key]))
        s.valueChanged.connect(lambda v: self._set(key, v))
        self._widgets[key] = s
        return s

    def _check(self, key, text):
        c = QCheckBox(text)
        c.setChecked(bool(self.state[key]))
        c.toggled.connect(lambda v: self._set(key, v))
        return c

    def _combo(self, key, items):
        c = QComboBox()
        for label, value in items:
            c.addItem(label, value)
        c.setCurrentIndex(max(0, c.findData(self.state[key])))
        c.currentIndexChanged.connect(lambda _: self._set(key, c.currentData()))
        return c

    # ── preview / save ────────────────────────────────────────────────────

    def _valid(self):
        if self.kind == "curve" and self.state["xmin"] >= self.state["xmax"]:
            self.msg.setText("X from must be less than X to")
            return False
        return True

    def _rebuild(self):
        if not self._valid():
            return
        fig = make_figure(self.kind, self.data, self.state, SCREEN_DPI)
        old = self.canvas.takeWidget()
        if old is not None:
            old.deleteLater()
        self.canvas.setWidget(fig)
        self.msg.setText(f"Actual size preview \u00b7 {self.state['width']:g} \u00d7 "
                         f"{self.state['height']:g} in")

    def _save(self):
        if not self._valid():
            return
        fmt = self.state["fmt"]
        filt = next(f[2] for f in FIG_FORMATS if f[1] == fmt)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save figure", f"dissonance_{self.kind}.{fmt}", filt)
        if not path:
            return
        if not path.lower().endswith("." + fmt):
            path += "." + fmt
        try:
            self._write(path)
        except Exception as exc:
            self.msg.setText(f"Save failed: {exc}")
            return
        self.saved_path = path
        self.msg.setText(f"Saved {Path(path).name}")

    def _write(self, path):
        render_figure(self.kind, self.data, self.state, path)


SURFACE_DEFAULTS = dict(
    title="", xlabel="Voice B ratio", ylabel="Voice C ratio",
    zlabel="Sensory dissonance", cbar="Sensory dissonance (normalized)",
    font="Times New Roman", size=9.0, axis=0.8, cmap="viridis", shading=True,
    floor=True, zaxis=True, colorbar=True, markers=True, labels="None",
    zscale=1.1, azimuth=68.0, elevation=40.0, distance=4.2,
    width=6.0, height=4.5, dpi=600, fmt="pdf",
)
SURFACE_CAMERA_KEYS = ("azimuth", "elevation", "distance")


def _nice_ticks(lo, hi, target=5):
    span = max(hi - lo, 1e-12)
    mag = 10 ** np.floor(np.log10(span / target))
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if span / step <= target + 1:
            break
    start = np.ceil(lo / step - 1e-9) * step
    return [round(v, 6) for v in np.arange(start, hi + step * 1e-6, step)]


def _html_doc(html, font):
    doc = QTextDocument()
    doc.setDocumentMargin(0)
    doc.setDefaultFont(font)
    doc.setHtml(f"<span style='color:#000000'>{html}</span>")
    return doc


def _text_at(p, html, font, cx, cy, angle=0.0):
    """Draw rich text centred on (cx, cy), optionally rotated; returns its size."""
    doc = _html_doc(html, font)
    size = doc.size()
    p.save()
    p.translate(cx, cy)
    p.rotate(angle)
    p.translate(-size.width() / 2, -size.height() / 2)
    doc.drawContents(p)
    p.restore()
    return size


class SurfaceRenderer:
    """Offscreen GL scene for print: raster surface, vector annotations on top.

    GL text is painted straight onto the widget and never reaches an offscreen
    render, so every label is drawn afterwards with QPainter.
    """

    STEPS = 220

    def __init__(self, data):
        self.data = data
        self.view = gl.GLViewWidget()
        self.view.setAttribute(Qt.WA_DontShowOnScreen, True)
        self.view.resize(320, 240)
        self.view.show()
        QApplication.processEvents()
        self.view.setBackgroundColor(QColor("#ffffff"))
        # Wider than the on-screen 50 deg so the whole surface is always in frame; a
        # field-of-view change only scales the image, and it's cropped to fit anyway.
        self.view.opts['fov'] = 90
        self._key = None
        self._items = []
        self._snap = (None, None)
        self._ink = {}

    def close(self):
        self.view.close()
        self.view.deleteLater()

    unit = staticmethod(_to_unit)

    def _sync(self, st):
        key = (st["cmap"], st["shading"], st["floor"], round(st["zscale"], 4))
        if key != self._key:
            self._key = key
            for item in self._items:
                self.view.removeItem(item)
            self._items = []
            x, y, z = self.data["x_3d"], self.data["y_3d"], self.data["z_3d"]
            step = max(1, len(x) // self.STEPS)
            xs, ys = x[::step], y[::step]
            zs = np.ascontiguousarray(z[::step, ::step], dtype=np.float32)
            lo, hi = float(z.min()), float(z.max())
            colors = pg.colormap.get(st["cmap"]).map(
                (zs - lo) / max(hi - lo, 1e-9), mode='float')
            surface = gl.GLSurfacePlotItem(
                x=self.unit(xs, x).astype(np.float32),
                y=self.unit(ys, y).astype(np.float32),
                z=zs * np.float32(st["zscale"]),
                colors=np.ascontiguousarray(colors.reshape(-1, 4).astype(np.float32)),
                shader='shaded' if st["shading"] else None,
                smooth=True, glOptions='opaque')
            self._items.append(surface)
            if st["floor"]:
                grid = gl.GLGridItem()
                grid.setSize(2.0, 2.0)
                grid.setSpacing(0.2, 0.2)
                grid.setColor(QColor(0, 0, 0, 70))
                self._items.append(grid)
            for item in self._items:
                self.view.addItem(item)
        self.view.setCameraPosition(distance=st["distance"], elevation=st["elevation"],
                                    azimuth=st["azimuth"])
        # Aim at mid-height so the surface, not the floor, is centred in frame.
        self.view.opts['center'] = pg.Vector(0, 0, 0.4 * st["zscale"])

    def _style_key(self, st):
        return (st["cmap"], st["shading"], st["floor"], st["zscale"],
                st["azimuth"], st["elevation"], st["distance"])

    def ink_box(self, st, aspect):
        """Fractions (x0, y0, x1, y1) of a full render that the surface occupies."""
        key = ("ink", self._style_key(st), round(aspect, 4))
        if self._ink.get(key) is None:
            self._sync(st)
            w = 960
            h = max(1, round(w / aspect))
            a = np.ascontiguousarray(self.view.renderToArray((w, h)))[:, :, :3]
            mask = (a < 245).any(axis=2)
            rows, cols = np.nonzero(mask.any(axis=1))[0], np.nonzero(mask.any(axis=0))[0]
            if not len(rows):
                box = (0.0, 0.0, 1.0, 1.0)
            else:
                # One probe pixel of slack each way so antialiased edges aren't clipped.
                box = (max(cols[0] - 1, 0) / w, max(rows[0] - 1, 0) / h,
                       min(cols[-1] + 2, w) / w, min(rows[-1] + 2, h) / h)
            self._ink = {key: box}
        return self._ink[key]

    def crop(self, st, W, H, x0, y0, cw, ch, supersample=False):
        """Render only the (x0, y0, cw, ch) window of a virtual W x H frame."""
        ss = 2 if supersample and cw * ch <= 12e6 else 1
        key = (self._style_key(st), W, H, x0, y0, cw, ch, ss)
        if self._snap[0] == key:
            return self._snap[1]
        self._sync(st)
        arr = self._render_region(W * ss, H * ss, x0 * ss, y0 * ss, cw * ss, ch * ss)
        img = QImage(arr.data, cw * ss, ch * ss, QImage.Format_ARGB32).copy()
        if ss > 1:
            img = img.scaled(cw, ch, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self._snap = (key, img)
        return img

    def _render_region(self, W, H, x0, y0, cw, ch, tex=1024, pad=256):
        # Same tiling as GLViewWidget.renderToArray, restricted to a sub-window.
        view = self.view
        view.makeCurrent()
        fbo = QtOpenGL.QOpenGLFramebufferObject(
            tex, tex, QtOpenGL.QOpenGLFramebufferObject.Attachment.CombinedDepthStencil,
            _GL.GL_TEXTURE_2D)
        out = np.full((ch, cw, 4), 255, dtype=np.ubyte)
        data = np.empty((tex, tex, 4), dtype=np.ubyte)
        try:
            for x in range(x0 - pad, x0 + cw - pad, tex - 2 * pad):
                for y in range(y0 - pad, y0 + ch - pad, tex - 2 * pad):
                    x2 = min(x + tex, x0 + cw + pad)
                    y2 = min(y + tex, y0 + ch + pad)
                    w2, h2 = x2 - x, y2 - y
                    fbo.bind()
                    _GL.glViewport(0, 0, w2, h2)
                    view.paint(region=(x, H - y - h2, w2, h2), viewport=(0, 0, W, H))
                    fbo.bind()
                    _GL.glReadPixels(0, 0, tex, tex, _GL.GL_BGRA, _GL.GL_UNSIGNED_BYTE, data)
                    flip = data[::-1]
                    out[y + pad - y0:y2 - pad - y0, x + pad - x0:x2 - pad - x0] = \
                        flip[-(h2 - pad):-pad, pad:w2 - pad]
        finally:
            fbo.release()
        return out

    def projector(self, st, w, h):
        """Map GL scene points to pixel positions in a w x h render."""
        self._sync(st)
        mvp = self.view.projectionMatrix((0, 0, w, h), (0, 0, w, h)) * self.view.viewMatrix()

        def project(px, py, pz):
            v = mvp.map(QVector4D(px, py, pz, 1.0))
            if v.w() <= 0:
                return None
            return QPointF((v.x() / v.w() + 1) * 0.5 * w, (1 - v.y() / v.w()) * 0.5 * h)

        return project


def paint_surface_figure(p, renderer, st, W, H, ppi, img_scale=1.0, final=False):
    """Lay out title, surface and colour bar on a W x H px canvas at *ppi*."""
    data = renderer.data
    pt = ppi / 72.0
    fam = st["font"]

    def font(scale=1.0):
        f = QFont(fam)
        f.setPixelSize(max(1, round(st["size"] * scale * pt)))
        return f

    black = QColor("#000000")
    axis_pen = QPen(black, st["axis"] * pt)
    tick_len, gap, m = 3.5 * pt, 2.5 * pt, 5 * pt
    tick_font = font()
    fm = QFontMetricsF(tick_font)
    p.fillRect(QRectF(0, 0, W, H), QColor("#ffffff"))

    top = m
    if st["title"].strip():
        title_h = _html_doc(st["title"], font(1.15)).size().height()
        _text_at(p, st["title"], font(1.15), W / 2, m + title_h / 2)
        top += title_h + m

    z = data["z_3d"]
    lo, hi = float(z.min()), float(z.max())
    bar_w = 0.0
    if st["colorbar"]:
        bar_ticks = _nice_ticks(lo, hi, 5)
        label_w = max(fm.horizontalAdvance(f"{v:g}") for v in bar_ticks)
        bar_w = 9 * pt + tick_len + gap + label_w + m
        if st["cbar"].strip():
            bar_w += _html_doc(st["cbar"], font()).size().height() + gap

    bar_block = bar_w - m if st["colorbar"] else 0.0
    sep = 4 * gap if st["colorbar"] else 0.0
    area = QRectF(m, top, W - 2 * m - bar_block - sep, H - top - m)
    aspect = area.width() / max(area.height(), 1.0)
    fx0, fy0, fx1, fy1 = renderer.ink_box(st, aspect)
    fw, fh = fx1 - fx0, fy1 - fy0

    def layout(T):
        """Fit the surface's bounding box into T: (box on page, virtual W, H, projector)."""
        Wv = min(T.width() / fw, T.height() * aspect / fh)
        Hv = Wv / aspect
        D = QRectF(0, 0, Wv * fw, Hv * fh)
        D.moveCenter(T.center())
        project = renderer.projector(st, Wv, Hv)
        off = D.topLeft() - QPointF(fx0 * Wv, fy0 * Hv)

        def P(gx, gy, gz):
            q = project(gx, gy, gz)
            return None if q is None else q + off

        return D, Wv, Hv, P

    def unit_dir(a, b):
        d = b - a
        n = (d.x() ** 2 + d.y() ** 2) ** 0.5 or 1.0
        return QPointF(d.x() / n, d.y() / n)

    th = fm.height()

    def scene(p, L, image, mark):
        """Surface, axes and markers laid out by *L*; *mark(rect)* records extent."""
        D, _, _, P = L
        if image is not None:
            p.drawImage(D, image)
        mark(D)
        p.setPen(axis_pen)
        p.setFont(tick_font)
        drawn = []

        def label(rect, flags, text):
            # Where two axes meet at a corner their end labels can collide; keep the first.
            if any(rect.intersects(r) for r in drawn):
                return
            drawn.append(rect)
            p.drawText(rect, flags, text)
            mark(rect)

        def title(html, cx, cy, angle):
            size = _text_at(p, html, font(), cx, cy, angle)
            r = np.radians(angle)
            hw = abs(np.cos(r)) * size.width() / 2 + abs(np.sin(r)) * size.height() / 2
            hh = abs(np.sin(r)) * size.width() / 2 + abs(np.cos(r)) * size.height() / 2
            mark(QRectF(cx - hw, cy - hh, 2 * hw, 2 * hh))

        # Floor axes: draw each along whichever edge sits nearer the viewer.
        centre = P(0, 0, 0)
        used = {}
        for along, values, axis_title in (("x", data["x_3d"], st["xlabel"]),
                                          ("y", data["y_3d"], st["ylabel"])):
            edges = []
            for e in (-1.0, 1.0):
                a = P(-1, e, 0) if along == "x" else P(e, -1, 0)
                b = P(1, e, 0) if along == "x" else P(e, 1, 0)
                if a is not None and b is not None:
                    edges.append(((a.y() + b.y()) / 2, e, a, b))
            if not edges or centre is None:
                continue
            _, e, a, b = max(edges, key=lambda t: t[0])
            used[along] = e
            p.setPen(axis_pen)
            p.drawLine(a, b)
            mid = (a + b) / 2
            out = unit_dir(centre, mid)
            extent = 0.0
            for v in _nice_ticks(float(values[0]), float(values[-1]), 5):
                u = float(renderer.unit([v], values)[0])
                q = P(u, e, 0) if along == "x" else P(e, u, 0)
                if q is None:
                    continue
                p.drawLine(q, q + out * tick_len)
                text = f"{v:g}"
                tw = fm.horizontalAdvance(text)
                half = abs(out.x()) * tw / 2 + abs(out.y()) * th / 2
                c = q + out * (tick_len + gap + half)
                label(QRectF(c.x() - tw / 2, c.y() - th / 2, tw, th), Qt.AlignCenter, text)
                extent = max(extent, 2 * half)
            if axis_title.strip():
                angle = np.degrees(np.arctan2(b.y() - a.y(), b.x() - a.x()))
                if angle > 90:
                    angle -= 180
                elif angle <= -90:
                    angle += 180
                title_h = _html_doc(axis_title, font()).size().height()
                c = mid + out * (tick_len + gap + extent + gap + title_h / 2)
                title(axis_title, c.x(), c.y(), angle)

        # Vertical dissonance axis on the leftmost side corner: never the front corner
        # both floor axes meet at, never the back one where it would cross the surface.
        if st["zaxis"] and centre is not None:
            ex, ey = used.get("x", -1.0), used.get("y", -1.0)
            sides = [(P(c[0], c[1], 0), c) for c in ((ey, -ex), (-ey, ex))]
            sides = [s for s in sides if s[0] is not None]
            cx, cy = min(sides, key=lambda s: s[0].x())[1] if sides else (ey, -ex)
            base, tip = P(cx, cy, 0), P(cx, cy, st["zscale"])
            if base is not None and tip is not None:
                side = -1.0 if base.x() <= centre.x() else 1.0
                p.setPen(axis_pen)
                p.drawLine(base, tip)
                widest = 0.0
                for v in _nice_ticks(0.0, 1.0, 5):
                    q = P(cx, cy, v * st["zscale"])
                    if q is None:
                        continue
                    p.drawLine(q, q + QPointF(side * tick_len, 0))
                    if v == 0:
                        continue  # the floor corner already carries a floor-axis label
                    text = f"{v:g}"
                    tw = fm.horizontalAdvance(text)
                    widest = max(widest, tw)
                    x0 = q.x() - tick_len - gap - tw if side < 0 else q.x() + tick_len + gap
                    label(QRectF(x0, q.y() - th / 2, tw, th),
                          (Qt.AlignRight if side < 0 else Qt.AlignLeft) | Qt.AlignVCenter, text)
                if st["zlabel"].strip():
                    title_h = _html_doc(st["zlabel"], font()).size().height()
                    reach = tick_len + gap + widest + gap + title_h / 2
                    x = (min(base.x(), tip.x()) - reach) if side < 0 else (max(base.x(), tip.x()) + reach)
                    title(st["zlabel"], x, (base.y() + tip.y()) / 2, -90)

        # Best triads, drawn on top like the on-screen view.
        tx, ty = data["tri_vis"][0], data["tri_vis"][1]
        if (st["markers"] or st["labels"] != "None") and len(tx):
            bi = [int(np.argmin(np.abs(data["x_3d"] - b))) for b in tx]
            ci = [int(np.argmin(np.abs(data["y_3d"] - c))) for c in ty]
            ub = renderer.unit(tx, data["x_3d"])
            uc = renderer.unit(ty, data["y_3d"])
            label_font = font(0.75)
            for k in range(len(tx)):
                q = P(float(ub[k]), float(uc[k]), float(z[bi[k], ci[k]]) * st["zscale"])
                if q is None:
                    continue
                if st["markers"]:
                    p.setPen(QPen(black, 0.5 * pt))
                    p.setBrush(QColor("#ffffff"))
                    r = 2.5 * pt
                    p.drawEllipse(q, r, r)
                    p.setBrush(Qt.NoBrush)
                if st["labels"] != "None":
                    text = _triad_figure_label(tx[k], ty[k], st["labels"], newline="<br>")
                    doc = _html_doc(text, label_font)
                    s = doc.size()
                    box = QRectF(q.x() + 3 * pt, q.y() - s.height() - 2 * pt, s.width(), s.height())
                    p.fillRect(box.adjusted(-pt, 0, pt, 0), QColor(255, 255, 255, 215))
                    mark(box.adjusted(-pt, 0, pt, 0))
                    p.save()
                    p.translate(box.topLeft())
                    doc.drawContents(p)
                    p.restore()

    probe = QImage(1, 1, QImage.Format_ARGB32)

    def measure(L):
        rects = []
        pp = QPainter(probe)
        try:
            scene(pp, L, None, rects.append)
        finally:
            pp.end()
        out = QRectF(rects[0])
        for r in rects[1:]:
            out = out.united(r)
        return out

    # Labels overhang the surface by roughly fixed amounts; shrink the target by
    # that overhang so surface + labels fill the area, then re-check once.
    T = QRectF(area)
    for _ in range(2):
        D = layout(T)[0]
        C = measure(layout(T))
        T = area.adjusted(D.left() - C.left(), D.top() - C.top(),
                          D.right() - C.right(), D.bottom() - C.bottom())
        if T.width() < 0.3 * area.width() or T.height() < 0.3 * area.height():
            T = QRectF(area)
            break
    L = layout(T)
    D, Wv, Hv, _ = L
    C = measure(L)

    group = C.width() + sep + bar_block
    shift_x = m + max(0.0, (W - 2 * m - group) / 2) - C.left()
    shift_y = top + max(0.0, (H - m - top - C.height()) / 2) - C.top()

    s = img_scale
    Wr, Hr = max(1, round(Wv * s)), max(1, round(Hv * s))
    image = renderer.crop(st, Wr, Hr, int(fx0 * Wr), int(fy0 * Hr),
                          max(1, round(fw * Wr)), max(1, round(fh * Hr)), supersample=final)
    p.save()
    p.translate(shift_x, shift_y)
    scene(p, L, image, lambda r: None)
    p.restore()

    if st["colorbar"]:
        bar = QRectF(C.right() + shift_x + sep, D.top() + shift_y + D.height() * 0.1,
                     9 * pt, D.height() * 0.8)
        cmap = pg.colormap.get(st["cmap"])
        grad = QLinearGradient(bar.bottomLeft(), bar.topLeft())
        for t, c in zip(np.linspace(0, 1, 33), cmap.map(np.linspace(0, 1, 33), mode='qcolor')):
            grad.setColorAt(float(t), c)
        p.fillRect(bar, grad)
        p.setPen(axis_pen)
        p.drawRect(bar)
        widest = 0.0
        for v in _nice_ticks(lo, hi, 5):
            yv = bar.bottom() - (v - lo) / max(hi - lo, 1e-12) * bar.height()
            p.drawLine(QPointF(bar.right(), yv), QPointF(bar.right() + tick_len, yv))
            text = f"{v:g}"
            tw = fm.horizontalAdvance(text)
            widest = max(widest, tw)
            p.drawText(QRectF(bar.right() + tick_len + gap, yv - th / 2, tw, th),
                       Qt.AlignLeft | Qt.AlignVCenter, text)
        if st["cbar"].strip():
            title_h = _html_doc(st["cbar"], font()).size().height()
            x = bar.right() + tick_len + gap + widest + gap + title_h / 2
            _text_at(p, st["cbar"], font(), x, bar.center().y(), -90)


def render_surface_figure(renderer, st, path):
    fmt = st["fmt"]
    dpi = float(st["dpi"])
    ppi = dpi if fmt == "png" else SCREEN_DPI
    W, H = round(st["width"] * ppi), round(st["height"] * ppi)
    _write_device(path, fmt, W, H, ppi, st,
                  lambda p: paint_surface_figure(p, renderer, st, W, H, ppi, dpi / ppi,
                                                 final=True))


class SurfaceEditor(FigureEditor):
    """Figure editor for the 3D surface: a snapshot at the on-screen angle,
    with every label, the colour bar and the styling editable on top."""

    def __init__(self, renderer, data, state, parent=None):
        self._renderer = renderer
        self._preview = None
        super().__init__("surface", data, state, parent)

    def _build_form(self, form, section):
        section("TEXT")
        form.addRow("Title", self._line("title"))
        form.addRow("B axis", self._line("xlabel"))
        form.addRow("C axis", self._line("ylabel"))
        form.addRow("Height axis", self._line("zlabel"))
        form.addRow("Colour bar", self._line("cbar"))
        self._rich_text_tip(form)

        section("STYLE")
        form.addRow("Font", self._combo("font", FIG_FONTS))
        form.addRow("Font size (pt)", self._num("size", 5, 24, 0.5, 1))
        form.addRow("Axes (pt)", self._num("axis", 0.1, 4, 0.1, 2))
        form.addRow("Colour map", self._combo("cmap", FIG_CMAPS))
        form.addRow("Surface", self._combo("shading", [("Lit", True), ("Flat colour", False)]))
        form.addRow(self._check("floor", "Floor grid"))
        form.addRow(self._check("zaxis", "Dissonance axis"))
        form.addRow(self._check("colorbar", "Colour bar"))
        form.addRow(self._check("markers", "Mark best triads"))
        form.addRow("Triad labels", self._combo("labels", [(m, m) for m in FIG_LABEL_MODES]))
        angle = QLabel("The view angle is taken from the Surface tab — rotate "
                       "it there before opening this editor.")
        angle.setObjectName("hint")
        angle.setWordWrap(True)
        form.addRow(angle)

        self._output_rows(form, section, "Resolution (dpi)")
        note = QLabel("PDF/SVG keep text and axes as vectors; the surface is "
                      "embedded at this resolution.")
        note.setObjectName("hint")
        note.setWordWrap(True)
        form.addRow(note)

    def _rebuild(self):
        if self._preview is None:
            self._preview = QLabel()
            self.canvas.setWidget(self._preview)
        st = self.state
        W, H = round(st["width"] * SCREEN_DPI), round(st["height"] * SCREEN_DPI)
        # Render at the screen's physical pixel density so scaled displays stay sharp.
        dpr = self.devicePixelRatioF()
        image = QImage(round(W * dpr), round(H * dpr), QImage.Format_ARGB32)
        image.fill(QColor("#ffffff"))
        p = QPainter(image)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        try:
            paint_surface_figure(p, self._renderer, st, image.width(), image.height(),
                                 SCREEN_DPI * dpr, final=True)
        finally:
            p.end()
        image.setDevicePixelRatio(dpr)
        self._preview.setPixmap(QPixmap.fromImage(image))
        self._preview.setFixedSize(W, H)
        self.msg.setText(f"Actual size preview \u00b7 {st['width']:g} \u00d7 {st['height']:g} in")

    def _write(self, path):
        render_surface_figure(self._renderer, self.state, path)


def _safe_filename(text):
    keep = [ch if (ch.isalnum() or ch in "._-") else "_" for ch in text]
    return "".join(keep).strip("_") or "export"


# ═══════════════════════════════════════════════════════════════════════════
#  Main window
# ═══════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dissonance Explorer")
        self.resize(1680, 980)

        self.ref_freq = 261.63
        self.max_interval = 2.3
        self.step_2d = 0.002
        self.step_3d = 0.005

        self.data = None
        self.worker = None
        self._last_sound = None
        self._shown_overtones = []
        self._dot_color = BLUE
        self._fig_state = {}
        self._surface_renderer = None

        self._build_ui()
        QTimer.singleShot(80, self.analyse)

    # ── construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        lay = QHBoxLayout(root)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        lay.addWidget(self._build_sidebar(), 0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(10)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_plot_area())
        splitter.addWidget(self._build_results())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([1050, 340])
        lay.addWidget(splitter, 1)

        QShortcut(QKeySequence("Ctrl+Return"), self, self.analyse)
        QShortcut(QKeySequence("Ctrl+Enter"), self, self.analyse)
        QShortcut(QKeySequence(Qt.Key_Escape), self, sd.stop)

    def _build_sidebar(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(340)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        col = QVBoxLayout(inner)
        col.setContentsMargins(0, 0, 8, 0)
        col.setSpacing(8)

        voices = Card("voices")
        self.voice_seg = Segmented(["A", "B", "C"])
        voices.body.addWidget(self.voice_seg)
        self.voice_stack = QStackedWidget()
        self.voice_stack.setObjectName("plain")
        self.card_a = SpectrumCard("A")
        self.card_b = SpectrumCard("B", copy_sources=[("A", self.card_a)])
        self.card_c = SpectrumCard("C", copy_sources=[("A", self.card_a),
                                                      ("B", self.card_b)])
        for card in (self.card_a, self.card_b, self.card_c):
            card.playRequested.connect(self._play_card)
            card.sourceChanged.connect(self._label_voice_tabs)
            self.voice_stack.addWidget(card)
        self.voice_seg.changed.connect(self.voice_stack.setCurrentIndex)
        voices.body.addWidget(self.voice_stack)
        col.addWidget(voices)
        self._label_voice_tabs()

        settings = Card("analysis")
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        self.f_ref = Field("FUNDAMENTAL Hz", "261.63")
        self.f_max = Field("MAX RATIO", "2.3")
        self.f_step = Field("3D STEP", "0.005")
        grid.addWidget(self.f_ref, 0, 0)
        grid.addWidget(self.f_max, 0, 1)
        grid.addWidget(self.f_step, 0, 2)
        settings.body.addLayout(grid)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.btn_run = QPushButton("Analyse")
        self.btn_run.setObjectName("primary")
        self.btn_run.setCursor(Qt.PointingHandCursor)
        self.btn_run.setToolTip("Recompute everything  (Ctrl+Enter)")
        self.btn_run.clicked.connect(self.analyse)
        btn_stop = QPushButton("Stop audio")
        btn_stop.setObjectName("danger")
        btn_stop.setCursor(Qt.PointingHandCursor)
        btn_stop.setToolTip("Silence playback  (Esc)")
        btn_stop.clicked.connect(sd.stop)
        actions.addWidget(self.btn_run, 1)
        actions.addWidget(btn_stop, 0)
        settings.body.addLayout(actions)
        col.addWidget(settings)

        manual = Card("manual playback")
        self.f_note = Field("Note A", "1.0", inline=True)
        self.f_interval = Field("A + B", "1.5", inline=True)
        self.f_triad = Field("Triad B, C", "1.25, 1.5", inline=True)
        rows = ((self.f_note, self._play_manual_note, "Play voice A on its own at this ratio"),
                (self.f_interval, self._play_manual_interval, "Play A with B at this ratio"),
                (self.f_triad, self._play_manual_triad, "Play A, B and C at these ratios"))
        for field, slot, tip in rows:
            row = QHBoxLayout()
            row.setSpacing(6)
            row.addWidget(field, 1)
            btn = QPushButton("\u25b6")
            btn.setObjectName("icon")
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            row.addWidget(btn, 0)
            manual.body.addLayout(row)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.f_duration = Field("Length (s)", "1.5", inline=True)
        row.addWidget(self.f_duration, 1)
        btn_wav = QPushButton("Save WAV\u2026")
        btn_wav.setCursor(Qt.PointingHandCursor)
        btn_wav.setToolTip("Write the last sound you played to a 24-bit WAV "
                           "file of this length")
        btn_wav.clicked.connect(self._export_wav)
        row.addWidget(btn_wav, 0)
        manual.body.addLayout(row)
        col.addWidget(manual)

        col.addStretch(1)
        scroll.setWidget(inner)
        return scroll

    def _label_voice_tabs(self):
        if not hasattr(self, "voice_seg"):
            return
        for i, card in enumerate((self.card_a, self.card_b, self.card_c)):
            text = card.letter
            if card.is_linked():
                text += f" = {card.source_name()}"
            btn = self.voice_seg.group.button(i)
            btn.setText(text)
            btn.setStyleSheet(
                f"QPushButton#segBtn, QPushButton#segBtn:hover, "
                f"QPushButton#segBtn:checked {{ color: {card.color}; }}")

    def _build_plot_area(self):
        wrap = QWidget()
        col = QVBoxLayout(wrap)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(10)

        top = QHBoxLayout()
        top.setSpacing(8)
        self.tabs = Segmented(["Curve", "Surface", "Heatmap"])
        self.tabs.changed.connect(self._switch_tab)
        top.addWidget(self.tabs, 0)
        self.view_hint = QLabel()
        self.view_hint.setObjectName("hint")
        top.addStretch(1)

        self.btn_export = QPushButton("Save figure\u2026")
        self.btn_export.setCursor(Qt.PointingHandCursor)
        self.btn_export.setToolTip("Open the figure editor for a publication-ready export")
        self.btn_export.clicked.connect(self._export_figure)
        top.addWidget(self.btn_export, 0)

        self.btn_reset_view = QPushButton("Reset")
        self.btn_reset_view.setToolTip("Reset the view")
        self.btn_reset_view.setCursor(Qt.PointingHandCursor)
        self.btn_reset_view.clicked.connect(self._reset_view)
        top.addWidget(self.btn_reset_view, 0)
        col.addLayout(top)

        card = QFrame()
        card.setObjectName("plotCard")
        inner = QVBoxLayout(card)
        inner.setContentsMargins(10, 10, 10, 10)

        self.stack = QStackedWidget()
        self.curve_view = CurveView()
        self.surface_view = SurfaceView()
        self.heatmap_view = HeatmapView()
        self.stack.addWidget(self.curve_view)
        surface_page = QWidget()
        surface_page.setObjectName("plain")
        sp = QHBoxLayout(surface_page)
        sp.setContentsMargins(0, 0, 0, 0)
        sp.setSpacing(0)
        sp.addWidget(self.surface_view, 1)
        self.surface_bar = ColorBarStrip()
        self.surface_bar.setVisible(HAS_GL)
        sp.addWidget(self.surface_bar, 0)
        self.stack.addWidget(surface_page)
        self.stack.addWidget(self.heatmap_view)
        inner.addWidget(self.stack)
        col.addWidget(card, 1)

        ot_card = QFrame()
        ot_card.setObjectName("plotCard")
        ot = QVBoxLayout(ot_card)
        ot.setContentsMargins(10, 6, 10, 4)
        ot.setSpacing(2)
        ot_head = QHBoxLayout()
        ot_head.setSpacing(10)
        ot_title = QLabel("OVERTONES")
        ot_title.setObjectName("cardTitle")
        ot_head.addWidget(ot_title, 0)
        self.overtone_legend = QLabel()
        self.overtone_legend.setObjectName("hint")
        ot_head.addWidget(self.overtone_legend, 1)
        self.overtone_info = QLabel("Play something to see how its partials line up")
        self.overtone_info.setObjectName("hint")
        ot_head.addWidget(self.overtone_info, 0)
        ot.addLayout(ot_head)
        self.overtone_view = OvertoneView()
        self.overtone_view.setFixedHeight(92)
        ot.addWidget(self.overtone_view)
        col.addWidget(ot_card, 0)

        self.curve_view.picked.connect(self._play_interval_ratio)
        self.heatmap_view.picked.connect(self._play_triad_ratio)
        self.surface_view.picked.connect(self._play_triad_ratio)
        self._switch_tab(0)

        status = QHBoxLayout()
        status.setSpacing(10)
        self.status_dot = QLabel()
        self.status_dot.setFixedSize(8, 8)
        self._set_dot(BLUE)
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusText")
        self.progress = QProgressBar()
        self.progress.setFixedHeight(6)
        self.progress.setFixedWidth(190)
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        status.addWidget(self.status_dot, 0)
        status.addWidget(self.status_label, 1)
        status.addWidget(self.view_hint, 0)
        status.addWidget(self.progress, 0)
        col.addLayout(status)
        return wrap

    def _build_results(self):
        wrap = QWidget()
        col = QVBoxLayout(wrap)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(10)

        self.res_tabs = Segmented(["Intervals", "Triads"])
        col.addWidget(self.res_tabs, 0)

        card = QFrame()
        card.setObjectName("plotCard")
        inner = QVBoxLayout(card)
        inner.setContentsMargins(8, 10, 8, 10)
        inner.setSpacing(8)

        self.sort_seg = Segmented(["Most consonant", "By ratio"])
        self.sort_seg.changed.connect(lambda _: self._fill_intervals())
        sort_row = QHBoxLayout()
        sort_row.setContentsMargins(4, 0, 4, 0)
        sort_row.addWidget(self.sort_seg, 1)

        self.res_stack = QStackedWidget()
        self.res_stack.setObjectName("plain")

        intervals_page = QWidget()
        intervals_page.setObjectName("plain")
        ip = QVBoxLayout(intervals_page)
        ip.setContentsMargins(0, 0, 0, 0)
        ip.setSpacing(8)
        ip.addLayout(sort_row)
        self.list_intervals = ResultList()
        self.list_intervals.activatedRow.connect(self._play_interval_ratio)
        ip.addWidget(self.list_intervals, 1)

        triads_page = QWidget()
        triads_page.setObjectName("plain")
        tp = QVBoxLayout(triads_page)
        tp.setContentsMargins(0, 0, 0, 0)
        tp.setSpacing(8)
        self.triad_seg = Segmented(["Within octave", "All voicings"])
        self.triad_seg.changed.connect(lambda _: self._refresh_triads())
        triad_row = QHBoxLayout()
        triad_row.setContentsMargins(4, 0, 4, 0)
        triad_row.addWidget(self.triad_seg, 1)
        tp.addLayout(triad_row)
        self.list_triads = ResultList()
        self.list_triads.activatedRow.connect(
            lambda p: self._play_triad_ratio(p[0], p[1]))
        tp.addWidget(self.list_triads, 1)

        self.res_stack.addWidget(intervals_page)
        self.res_stack.addWidget(triads_page)
        self.res_tabs.changed.connect(self.res_stack.setCurrentIndex)
        inner.addWidget(self.res_stack)
        col.addWidget(card, 1)

        theme_row = QHBoxLayout()
        theme_row.setSpacing(8)
        theme_row.addStretch(1)
        theme_cap = QLabel("Theme")
        theme_cap.setObjectName("fieldLabel")
        theme_row.addWidget(theme_cap, 0)
        self.theme_seg = Segmented(["Dark", "Light"])
        self.theme_seg.changed.connect(self._set_theme)
        theme_row.addWidget(self.theme_seg, 0)
        col.addLayout(theme_row)

        wrap.setMinimumWidth(240)
        return wrap

    # ── helpers ────────────────────────────────────────────────────────────

    def _set_dot(self, color):
        self._dot_color = color
        self.status_dot.setStyleSheet(f"background: {color}; border-radius: 4px;")

    def _status(self, text, color=BLUE):
        self.status_label.setText(text)
        self._set_dot(color)

    VIEW_HINTS = (
        "scroll to zoom  \u00b7  drag to pan  \u00b7  click to hear the interval",
        "drag to rotate  \u00b7  scroll to zoom  \u00b7  click the surface to hear the triad",
        "dark = consonant  \u00b7  click anywhere to hear that triad",
    )

    def _switch_tab(self, index):
        self.stack.setCurrentIndex(index)
        self.view_hint.setText(self.VIEW_HINTS[index])

    def _view_xmax(self):
        """Upper ratio currently shown on the curve."""
        return self.data["max_interval"] if self.data is not None else self.max_interval

    def _reset_view(self):
        i = self.stack.currentIndex()
        if i == 0 and self.data is not None:
            self.curve_view.setViewRatioRange(self._view_xmax())
        elif i == 1:
            self.surface_view.resetView()
        elif i == 2 and self.data is not None:
            x, y = self.data["x_3d"], self.data["y_3d"]
            self.heatmap_view.getPlotItem().vb.setRange(
                xRange=(x[0], x[-1]), yRange=(y[0], y[-1]), padding=0)

    def _spectra(self):
        a = self.card_a.get_spectrum()
        b = self.card_b.get_spectrum()
        c = self.card_c.get_spectrum()
        return a, b, c

    def _read_settings(self):
        try:
            ref = float(self.f_ref.text())
            mx = float(self.f_max.text())
            st = float(self.f_step.text())
        except ValueError:
            self._status("Settings must be numbers", RED)
            return False
        if ref <= 0 or mx <= 1.0 or st <= 0:
            self._status("Fundamental > 0, max ratio > 1, step > 0", RED)
            return False
        self.ref_freq, self.max_interval, self.step_3d = ref, mx, st
        return True

    # ── computation ────────────────────────────────────────────────────────

    def analyse(self):
        if self.worker is not None and self.worker.isRunning():
            return
        if not self._read_settings():
            return
        spec_a, spec_b, spec_c = self._spectra()
        for name, spec in (("A", spec_a), ("B", spec_b), ("C", spec_c)):
            if spec is None:
                self._status(f"Voice {name}: partials and amplitudes must be "
                             f"equal-length number lists", RED)
                return

        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self._status("Computing dissonance space\u2026", YELLOW)

        self.worker = ComputeWorker(spec_a, spec_b, spec_c, self.ref_freq,
                                    self.max_interval, self.step_2d,
                                    self.step_3d)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished_ok.connect(self._on_results)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_failed(self, message):
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)
        self._status(f"Computation failed: {message}", RED)

    def _on_results(self, data):
        self.data = data
        if self._surface_renderer is not None:
            self._surface_renderer.close()
            self._surface_renderer = None
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)

        self.curve_view.setData(data["ratios_2d"], data["diss_2d"],
                                data["minima_x"], data["minima_y"],
                                self._view_xmax())
        self._fill_intervals()
        self._refresh_triads()
        self._status("Ready \u2014 click any plot or list entry to hear it", GREEN)

    def _fill_intervals(self):
        if self.data is None:
            return
        xmax = self._view_xmax()
        pairs = [(x, y) for x, y in zip(self.data["minima_x"],
                                        self.data["minima_y"]) if x <= xmax]
        if self.sort_seg.group.checkedId() == 0:
            pairs.sort(key=lambda p: p[1])
        else:
            pairs.sort(key=lambda p: p[0])
        rows = [(f"{x:.4f}",
                 f"{ratio_to_cents(x):.0f}\u00a2  \u00b7  {ratio_to_interval_name(x)}",
                 float(y), float(x))
                for x, y in pairs]
        self.list_intervals.setRows(rows, "No consonant minima found")

    def _visible_triads(self, limit=10):
        """Best triads for display, optionally restricted to within one octave."""
        bx = self.data["tri_x"]
        cy = self.data["tri_y"]
        dz = self.data["tri_z"]
        if self.triad_seg.group.checkedId() == 0:
            # 20-cent guard: anything nearer than this just reads as an octave.
            octave = 2.0 * 2 ** (-20 / 1200)
            keep = (bx < octave) & (cy < octave)
            bx, cy, dz = bx[keep], cy[keep], dz[keep]
        order = np.argsort(dz)[:limit]
        return bx[order], cy[order], dz[order]

    def _refresh_triads(self):
        if self.data is None:
            return
        tx, ty, tz = self._visible_triads()
        d = self.data
        self.heatmap_view.setData(d["x_3d"], d["y_3d"], d["z_3d"], tx, ty, tz)
        self.surface_view.setData(d["x_3d"], d["y_3d"], d["z_3d"], tx, ty, tz)
        self.surface_bar.setLevels(float(d["z_3d"].min()), float(d["z_3d"].max()))
        rows = [(f"1.000 : {bx:.3f} : {cy:.3f}",
                 f"{ratio_to_interval_name(bx)}  \u00b7  {ratio_to_interval_name(cy)}",
                 float(dz), (float(bx), float(cy)))
                for bx, cy, dz in zip(tx, ty, tz)]
        self.list_triads.setRows(rows, "No consonant triads found")

    # ── playback ───────────────────────────────────────────────────────────

    def _sound(self, voices, label, letters="ABC"):
        """Play *voices*, show their overtones, and remember them for WAV export."""
        self._last_sound = (voices, label)
        self._shown_overtones = [(letters[i], spec, ratio)
                                 for i, (spec, ratio) in enumerate(voices)]
        self._show_overtones()
        play_multi(voices, self.ref_freq)

    def _show_overtones(self):
        shown = self._shown_overtones
        if not shown:
            return
        matches = self.overtone_view.setVoices(shown, self.ref_freq)
        dots = "  ".join(f"<span style='color:{VOICE_COLORS[l]}'>\u25cf</span> "
                          f"{l} {ratio:.3f}" for l, _, ratio in shown)
        key = (f"  \u00b7  <span style='color:{TEXT}'>\u25ef</span> shared"
               if len(shown) > 1 else "")
        self.overtone_legend.setText(dots + key)
        self.overtone_info.setText(
            f"{matches} shared partials" if len(shown) > 1 else "single voice")

    def _play_card(self, card):
        if not self._read_settings():
            return
        spec = card.get_spectrum()
        if spec is None:
            self._status(f"Voice {card.letter} has invalid input", RED)
            return
        self._status(f"Playing voice {card.letter} alone", GREEN)
        self._sound([(spec, 1.0)], f"voice_{card.letter}", letters=card.letter)

    def _diss_at_ratio(self, ratio):
        if self.data is None:
            return None
        r = self.data["ratios_2d"]
        i = int(np.argmin(np.abs(r - ratio)))
        return float(self.data["diss_2d"][i])

    def _diss_at_triad(self, rb, rc):
        if self.data is None:
            return None
        bi = int(np.argmin(np.abs(self.data["x_3d"] - rb)))
        ci = int(np.argmin(np.abs(self.data["y_3d"] - rc)))
        return float(self.data["z_3d"][bi, ci])

    def _play_interval_ratio(self, ratio):
        if not self._read_settings():
            return
        spec_a, spec_b, _ = self._spectra()
        if spec_a is None or spec_b is None:
            self._status("Voice A or B has invalid input", RED)
            return
        d = self._diss_at_ratio(ratio)
        tail = f" \u00b7 diss {d:.3f}" if d is not None else ""
        self._status(f"A 1.000 + B {ratio:.4f}  ({ratio_to_interval_name(ratio)})"
                     f"{tail}", GREEN)
        self._sound([(spec_a, 1.0), (spec_b, ratio)],
                    f"interval_{ratio:.4f}")

    def _play_triad_ratio(self, rb, rc):
        if not self._read_settings():
            return
        spec_a, spec_b, spec_c = self._spectra()
        if spec_a is None or spec_b is None or spec_c is None:
            self._status("One of the voices has invalid input", RED)
            return
        d = self._diss_at_triad(rb, rc)
        tail = f" \u00b7 diss {d:.3f}" if d is not None else ""
        self._status(f"A 1.000 + B {rb:.3f} + C {rc:.3f}{tail}", GREEN)
        self._sound([(spec_a, 1.0), (spec_b, rb), (spec_c, rc)],
                    f"triad_{rb:.3f}_{rc:.3f}")

    def _play_manual_interval(self):
        try:
            ratio = float(self.f_interval.text())
        except ValueError:
            self._status("Interval must be a single number", RED)
            return
        self._play_interval_ratio(ratio)

    def _play_manual_note(self):
        try:
            ratio = float(self.f_note.text())
        except ValueError:
            self._status("Note must be a single number", RED)
            return
        if ratio <= 0:
            self._status("Note ratio must be positive", RED)
            return
        if not self._read_settings():
            return
        spec_a = self.card_a.get_spectrum()
        if spec_a is None:
            self._status("Voice A has invalid input", RED)
            return
        self._status(f"A {ratio:.4f} alone  ({ratio_to_interval_name(ratio)})", GREEN)
        self._sound([(spec_a, ratio)], f"note_{ratio:.4f}")

    def _play_manual_triad(self):
        try:
            parts = [float(p.strip()) for p in self.f_triad.text().split(',')]
        except ValueError:
            self._status("Triad must be two numbers, e.g. 1.25, 1.5", RED)
            return
        if len(parts) != 2:
            self._status("Enter exactly two ratios (B, C)", RED)
            return
        self._play_triad_ratio(parts[0], parts[1])

    # ── theme ──────────────────────────────────────────────────────────────

    def _set_theme(self, index):
        apply_app_theme(QApplication.instance(),
                        PAPER_PALETTE if index == 1 else DARK_PALETTE)
        self._restyle()

    def _restyle(self):
        """Re-read the active palette everywhere it was baked in at build time."""
        for card in (self.card_a, self.card_b, self.card_c):
            card.apply_palette()
        self.curve_view.apply_palette()
        self.heatmap_view.apply_palette()
        self.surface_view.apply_palette()
        self.surface_bar.apply_palette()
        self.overtone_view.apply_palette()
        self._show_overtones()
        self._set_dot(self._dot_color)
        self._label_voice_tabs()
        if self.data is not None:
            self._fill_intervals()
            self._refresh_triads()

    # ── export ─────────────────────────────────────────────────────────────

    VIEW_NAMES = ("curve", "surface", "heatmap")

    def _export_figure(self):
        if self.data is None:
            self._status("Nothing to export yet \u2014 run an analysis first", RED)
            return
        index = self.stack.currentIndex()
        kind = self.VIEW_NAMES[index]
        if kind == "surface" and not HAS_GL:
            self._status("The 3D surface is unavailable on this machine", RED)
            return
        st = dict(SURFACE_DEFAULTS) if kind == "surface" else {
            **FIGURE_COMMON, **FIGURE_DEFAULTS[kind]}
        st.update(self._fig_state.get(kind, {}))
        data = self.data
        if kind == "curve":
            r = data["ratios_2d"]
            (x0, x1), _ = self.curve_view.viewRange()
            st["xmin"] = round(max(float(x0), float(r[0])), 3)
            st["xmax"] = round(min(float(x1), float(r[-1])), 3)
            st["ymax"] = 0.0
        else:
            data = dict(data, tri_vis=self._visible_triads())
        if kind == "surface":
            cam = self.surface_view.cameraParams()
            st.update(azimuth=float(cam["azimuth"]), elevation=float(cam["elevation"]),
                      distance=float(cam["distance"]))

        self.curve_view.hideCrosshair()
        if kind == "surface":
            if self._surface_renderer is None:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                try:
                    self._surface_renderer = SurfaceRenderer(data)
                finally:
                    QApplication.restoreOverrideCursor()
            self._surface_renderer.data = data
            editor = SurfaceEditor(self._surface_renderer, data, st, parent=self)
        else:
            editor = FigureEditor(kind, data, st, parent=self)
        editor.exec()
        # Text and style carry over to the next figure; ranges and camera follow the view.
        skip = ("xmin", "xmax", "ymax") + SURFACE_CAMERA_KEYS
        self._fig_state[kind] = {k: v for k, v in editor.state.items() if k not in skip}
        if editor.saved_path:
            self._status(f"Figure saved to {Path(editor.saved_path).name}", GREEN)

    def _export_wav(self):
        if self._last_sound is None:
            self._status("Play something first, then save it", RED)
            return
        if not self._read_settings():
            return
        try:
            seconds = float(self.f_duration.text())
        except ValueError:
            self._status("Length must be a number of seconds", RED)
            return
        if not 0.05 <= seconds <= 600:
            self._status("Length must be between 0.05 and 600 seconds", RED)
            return

        voices, label = self._last_sound
        path, _ = QFileDialog.getSaveFileName(
            self, "Save sound", _safe_filename(label) + ".wav",
            "WAV audio (*.wav)")
        if not path:
            return
        if not path.lower().endswith(".wav"):
            path += ".wav"
        try:
            write_wav(path, synthesize_multi(voices, self.ref_freq,
                                             duration=seconds))
        except Exception as exc:
            self._status(f"Could not write the file: {exc}", RED)
            return
        self._status(f"Saved {seconds:g}\u202fs to {Path(path).name}", GREEN)

    # ── lifecycle ──────────────────────────────────────────────────────────

    def closeEvent(self, event):
        sd.stop()
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(3000)
        super().closeEvent(event)


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

def apply_app_theme(app, palette_dict):
    """Switch the whole application over to *palette_dict*."""
    apply_palette(palette_dict)
    pg.setConfigOptions(antialias=True, background=MANTLE, foreground=TEXT)

    qp = app.palette()
    qp.setColor(QPalette.Window, QColor(BASE))
    qp.setColor(QPalette.Base, QColor(SURFACE))
    qp.setColor(QPalette.Text, QColor(TEXT))
    qp.setColor(QPalette.WindowText, QColor(TEXT))
    qp.setColor(QPalette.Highlight, QColor(BLUE))
    qp.setColor(QPalette.HighlightedText, QColor(BASE))
    app.setPalette(qp)
    app.setStyleSheet(build_qss(palette_dict))


def main():
    # pyqtgraph caches compiled shaders globally, so every GL widget must share one context group.
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    app.setApplicationName("Dissonance Explorer")
    app.setStyle("Fusion")
    apply_app_theme(app, DARK_PALETTE)

    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
