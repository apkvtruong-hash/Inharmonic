"""
Dissonance Studio — a fluid, live front-end for the Plomp-Levelt / Sethares
sensory-dissonance model, built with PySide6 + pyqtgraph + OpenGL and Numba.

* dissonance curve, 3D dissonance surface and triad map, with the most consonant
  intervals and triads ranked and playable
* live analysis while you type or drag partials (compiled, multi-threaded kernels)
* a click-free audio engine with crossfades, press-and-drag glides and volume
* a draggable spectrum editor, hover readouts, linked highlights, a detail panel
* publication figure export (PDF, SVG, PNG) and WAV export
* animated controls, view and theme transitions, toasts and a themed title bar

Requirements: numpy, sounddevice, PySide6, pyqtgraph, PyOpenGL, numba, qtawesome
Run:          python dissonance_studio.py
"""

from __future__ import annotations

import ctypes
import math
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

from PySide6.QtCore import (QAbstractAnimation, QEasingCurve, QEvent, QMarginsF,
                            QObject, QPointF, QRectF, QSize, QSizeF, Qt, QThread,
                            QTimer, QUrl, QVariantAnimation, Signal)
from PySide6.QtGui import (QColor, QDesktopServices, QFont, QFontMetricsF, QImage,
                           QKeySequence, QLinearGradient, QPageSize, QPainter,
                           QPainterPath, QPalette, QPdfWriter, QPen, QPixmap,
                           QShortcut, QStandardItem, QStandardItemModel,
                           QSurfaceFormat, QTextDocument, QTransform, QVector3D,
                           QVector4D)
from PySide6.QtSvg import QSvgGenerator
from PySide6.QtWidgets import (QAbstractButton, QAbstractItemView, QApplication,
                               QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QFrame, QGraphicsItem,
                               QGraphicsOpacityEffect, QGraphicsPathItem, QGridLayout,
                               QHBoxLayout, QLabel, QLineEdit, QListView,
                               QMainWindow, QPushButton, QScrollArea, QSizePolicy,
                               QSlider, QSplitter, QStackedWidget, QStyle,
                               QStyledItemDelegate, QVBoxLayout, QWidget)

import pyqtgraph as pg

try:
    import qtawesome as qta
except Exception:  # pragma: no cover - icons fall back to text glyphs
    qta = None

try:
    import numba as nb
    HAS_NUMBA = True
except Exception:  # pragma: no cover - plain numpy fallback
    nb = None
    HAS_NUMBA = False

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

# The curve runs CURVE_MARGIN past MAX RATIO so minima near it aren't clipped; the view stops
# at MAX RATIO.
CURVE_MARGIN = 0.05
STEP_2D = 0.002
# Points per axis of the triad grid; finer grids would need gigabytes and minutes.
MAX_GRID = 1600

# Plomp-Levelt curve as fitted by Sethares.
PL_X, PL_S1, PL_S2, PL_B1, PL_B2 = 0.24, 0.0207, 18.96, 3.51, 5.75


class Cancelled(Exception):
    """Raised inside a computation that a newer request has superseded."""


def amp_to_loudness(amp):
    dB = 20 * np.log10(np.maximum(amp, 1e-10))
    return 2 ** (dB / 10) / 16


def dissonance_pair_vectorized(f1, f2, l1, l2):
    fmin = np.minimum(f1, f2)
    fmax = np.maximum(f1, f2)
    s = PL_X / (PL_S1 * fmin + PL_S2)
    p = s * (fmax - fmin)
    l12 = np.minimum(l1, l2)
    return l12 * (np.exp(-PL_B1 * p) - np.exp(-PL_B2 * p))


def _voice_arrays(spec, ref_freq):
    """Absolute partial frequencies and loudnesses of one voice at ratio 1."""
    return (ref_freq * np.asarray(spec["freq"], dtype=float),
            amp_to_loudness(np.asarray(spec["amp"], dtype=float)))


# In (ln r, ln s) the valleys from A-B, A-C and B-C partial coincidences run along
# these three directions, so the search can slide along any of them.
_TRIAD_DIRS = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1))
_DIRS = np.array(_TRIAD_DIRS, dtype=np.float64)


if HAS_NUMBA:
    # nogil lets the GUI and the audio callback run while a kernel is busy.
    _JIT = dict(nogil=True, cache=True, error_model="numpy")

    @nb.njit(**_JIT)
    def _nb_pair(f1, f2, l1, l2):
        if f1 < f2:
            fmin, fmax = f1, f2
        else:
            fmin, fmax = f2, f1
        p = PL_X / (PL_S1 * fmin + PL_S2) * (fmax - fmin)
        l12 = l1 if l1 < l2 else l2
        return l12 * (np.exp(-PL_B1 * p) - np.exp(-PL_B2 * p))

    @nb.njit(**_JIT)
    def _nb_self(base, loud, r):
        """Self-dissonance of one voice at ratio r, every ordered pair (so twice)."""
        tot = 0.0
        for i in range(base.size):
            fi = r * base[i]
            for j in range(base.size):
                tot += _nb_pair(fi, r * base[j], loud[i], loud[j])
        return tot

    @nb.njit(**_JIT)
    def _nb_cross(base1, loud1, r1, base2, loud2, r2):
        tot = 0.0
        for i in range(base1.size):
            fi = r1 * base1[i]
            for j in range(base2.size):
                tot += _nb_pair(fi, r2 * base2[j], loud1[i], loud2[j])
        return tot

    @nb.njit(**_JIT)
    def _nb_curve(base_a, loud_a, base_b, loud_b, ratios):
        d_self_a = _nb_self(base_a, loud_a, 1.0) * 0.5
        out = np.empty(ratios.size)
        for k in range(ratios.size):
            c = ratios[k]
            out[k] = (d_self_a + _nb_self(base_b, loud_b, c) * 0.5
                      + _nb_cross(base_a, loud_a, 1.0, base_b, loud_b, c))
        return out

    @nb.njit(**_JIT)
    def _nb_terms(base_a, loud_a, base_v, loud_v, ratios):
        """Per-ratio self-dissonance of a voice and its cross-dissonance with A."""
        self_t = np.empty(ratios.size)
        cross_t = np.empty(ratios.size)
        for k in range(ratios.size):
            r = ratios[k]
            self_t[k] = _nb_self(base_v, loud_v, r)
            cross_t[k] = _nb_cross(base_a, loud_a, 1.0, base_v, loud_v, r)
        return self_t, cross_t

    @nb.njit(parallel=True, **_JIT)
    def _nb_surface_rows(base_b, loud_b, base_c, loud_c, ratios,
                         d_aa, d_bb, d_ab, d_cc, d_ac, r0, r1, out):
        n = ratios.size
        for ri in nb.prange(r0, r1):
            r = ratios[ri]
            for si in range(n):
                d_bc = _nb_cross(base_b, loud_b, r, base_c, loud_c, ratios[si])
                # Self terms count each pair twice; cross terms already count each pair once.
                out[ri, si] = ((d_aa + d_bb[ri] + d_cc[si]) / 2.0
                               + d_ab[ri] + d_ac[si] + d_bc)

    @nb.njit(**_JIT)
    def _nb_triad_cost(base, loud, owner, u, v):
        """Exact, unnormalised dissonance of A : e^u B : e^v C."""
        fac = np.array([1.0, np.exp(u), np.exp(v)])
        n = base.size
        f = np.empty(n)
        for i in range(n):
            f[i] = base[i] * fac[owner[i]]
        tot = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                tot += _nb_pair(f[i], f[j], loud[i], loud[j])
        return tot

    @nb.njit(**_JIT)
    def _nb_descend(base, loud, owner, u, v, h, lo0, lo1, hi0, hi1, tol, dirs):
        """Pattern search down to a local minimum; ok=False if it leaves the box."""
        best = _nb_triad_cost(base, loud, owner, u, v)
        while h > tol:
            moved = False
            for k in range(dirs.shape[0]):
                c = _nb_triad_cost(base, loud, owner, u + dirs[k, 0] * h, v + dirs[k, 1] * h)
                if c < best:
                    u = u + dirs[k, 0] * h
                    v = v + dirs[k, 1] * h
                    best = c
                    moved = True
                    break
            if not moved:
                h *= 0.5
            elif not (lo0 <= u <= hi0 and lo1 <= v <= hi1):
                return u, v, best, False
        return u, v, best, True

    @nb.njit(parallel=True, **_JIT)
    def _nb_descend_many(base, loud, owner, us, vs, h, lo0, lo1, hi0, hi1, tol, dirs):
        n = us.size
        ou = np.empty(n)
        ov = np.empty(n)
        ob = np.empty(n)
        ok = np.empty(n, dtype=np.bool_)
        for k in nb.prange(n):
            a, b, c, d = _nb_descend(base, loud, owner, us[k], vs[k], h,
                                     lo0, lo1, hi0, hi1, tol, dirs)
            ou[k] = a
            ov[k] = b
            ob[k] = c
            ok[k] = d
        return ou, ov, ob, ok


def warm_up_kernels():
    """Compile (or load from cache) every kernel with the argument types used later."""
    if not HAS_NUMBA:
        return
    b = np.array([261.63, 523.26])
    lo = np.array([0.0625, 0.03])
    r = np.array([1.0, 1.5])
    _nb_curve(b, lo, b, lo, r)
    t1, t2 = _nb_terms(b, lo, b, lo, r)
    _nb_surface_rows(b, lo, b, lo, r, 0.0, t1, t2, t1, t2, 0, 2, np.empty((2, 2)))
    base = np.concatenate([b, b, b])
    loud = np.concatenate([lo, lo, lo])
    owner = np.repeat(np.arange(3), 2).astype(np.int64)
    _nb_triad_cost(base, loud, owner, 0.1, 0.2)
    _nb_descend_many(base, loud, owner, np.array([0.1]), np.array([0.2]), 0.01,
                     0.0, 0.0, 1.0, 1.0, 1e-3, _DIRS)


def _np_self(base, loud, r):
    f1, f2 = np.meshgrid(r * base, r * base, indexing='ij')
    l1, l2 = np.meshgrid(loud, loud, indexing='ij')
    return dissonance_pair_vectorized(f1.ravel(), f2.ravel(), l1.ravel(), l2.ravel()).sum()


def _np_cross(base1, loud1, base2, loud2, r2):
    f1, f2 = np.meshgrid(base1, r2 * base2, indexing='ij')
    l1, l2 = np.meshgrid(loud1, loud2, indexing='ij')
    return dissonance_pair_vectorized(f1.ravel(), f2.ravel(), l1.ravel(), l2.ravel()).sum()


def _curve_raw(spec_a, spec_b, ref_freq, ratios):
    """Unnormalised dissonance of fixed voice A with voice B at each ratio."""
    base_a, loud_a = _voice_arrays(spec_a, ref_freq)
    base_b, loud_b = _voice_arrays(spec_b, ref_freq)
    if HAS_NUMBA:
        return _nb_curve(base_a, loud_a, base_b, loud_b, ratios)
    d_self_a = _np_self(base_a, loud_a, 1.0) * 0.5
    return np.array([d_self_a + _np_self(base_b, loud_b, c) * 0.5
                     + _np_cross(base_a, loud_a, base_b, loud_b, c) for c in ratios])


def compute_curve(spec_a, spec_b, ref_freq, stop):
    """Dissonance curve from ratio 0.9 to *stop*, normalised to its peak, with its minima."""
    ratios = np.arange(0.9, stop, STEP_2D)
    raw = _curve_raw(spec_a, spec_b, ref_freq, ratios)
    norm = float(raw.max()) if len(raw) and raw.max() > 0 else 1.0
    diss = raw / norm
    minima_x, minima_y = find_minima(ratios, diss)
    return dict(ratios_2d=ratios, diss_2d=diss, minima_x=minima_x, minima_y=minima_y)


def compute_dissonance_3d(spec_a, spec_b, spec_c, ref_freq, max_interval, step_size,
                          progress=None, check=None):
    """Dissonance surface indexed [voice B ratio, voice C ratio]; A is fixed."""
    base_a, loud_a = _voice_arrays(spec_a, ref_freq)
    base_b, loud_b = _voice_arrays(spec_b, ref_freq)
    base_c, loud_c = _voice_arrays(spec_c, ref_freq)
    ratios = np.arange(1.0, max_interval, step_size)
    n = len(ratios)

    if HAS_NUMBA:
        d_aa = _nb_self(base_a, loud_a, 1.0)
        d_bb, d_ab = _nb_terms(base_a, loud_a, base_b, loud_b, ratios)
        d_cc, d_ac = _nb_terms(base_a, loud_a, base_c, loud_c, ratios)
    else:
        d_aa = _np_self(base_a, loud_a, 1.0)
        d_bb = np.array([_np_self(base_b, loud_b, r) for r in ratios])
        d_ab = np.array([_np_cross(base_a, loud_a, base_b, loud_b, r) for r in ratios])
        d_cc = np.array([_np_self(base_c, loud_c, s) for s in ratios])
        d_ac = np.array([_np_cross(base_a, loud_a, base_c, loud_c, s) for s in ratios])
        bc_f1, bc_f2 = (g.ravel() for g in np.meshgrid(base_b, base_c, indexing='ij'))
        bc_l1, bc_l2 = (g.ravel() for g in np.meshgrid(loud_b, loud_c, indexing='ij'))

    z = np.empty((n, n))
    chunk = max(4, -(-n // 6))
    for r0 in range(0, n, chunk):
        if check is not None:
            check()
        r1 = min(n, r0 + chunk)
        if HAS_NUMBA:
            _nb_surface_rows(base_b, loud_b, base_c, loud_c, ratios,
                             d_aa, d_bb, d_ab, d_cc, d_ac, r0, r1, z)
        else:
            for ri in range(r0, r1):
                d_bc = dissonance_pair_vectorized(
                    ratios[ri] * bc_f1[np.newaxis, :], ratios[:, np.newaxis] * bc_f2[np.newaxis, :],
                    bc_l1[np.newaxis, :], bc_l2[np.newaxis, :]).sum(axis=1)
                z[ri, :] = (d_aa + d_bb[ri] + d_cc) / 2.0 + d_ab[ri] + d_ac + d_bc
        if progress is not None:
            progress(r1 / n)

    max_z = z.max()
    if max_z > 0:
        z /= max_z
    return ratios, ratios, z


def find_minima(ratios, dissonance_values, window=5, min_prominence=0.002):
    """Find local minima with significant prominence (depth relative to surroundings).

    A minimum is kept only if it sits at least min_prominence (on the 0-1
    normalised scale) below the lower of the two peaks that separate it from
    lower ground (or from the end of the curve).
    """
    candidates = []
    n = len(dissonance_values)
    if n > 2 * window:
        # Same test as comparing each point with min() of its window, without a Python loop.
        win = np.lib.stride_tricks.sliding_window_view(dissonance_values, 2 * window + 1)
        inner = np.asarray(dissonance_values)[window:n - window]
        candidates = (np.nonzero(inner == win.min(axis=1))[0] + window).tolist()

    minima_x = []
    minima_y = []
    vals = np.asarray(dissonance_values).tolist()  # plain floats walk ~10x faster
    for idx in candidates:
        val = vals[idx]
        left_max = val
        for j in range(idx - 1, -1, -1):
            if vals[j] < val:
                break
            left_max = max(left_max, vals[j])
        right_max = val
        for j in range(idx + 1, n):
            if vals[j] < val:
                break
            right_max = max(right_max, vals[j])
        if min(left_max, right_max) - val >= min_prominence:
            minima_x.append(ratios[idx])
            minima_y.append(dissonance_values[idx])
    return np.array(minima_x), np.array(minima_y)


def _triad_arrays(spectra, ref_freq):
    base = np.concatenate([ref_freq * np.asarray(v["freq"], dtype=float) for v in spectra])
    loud = np.concatenate([amp_to_loudness(np.asarray(v["amp"], dtype=float))
                           for v in spectra])
    owner = np.repeat(np.arange(3), [len(v["freq"]) for v in spectra]).astype(np.int64)
    return base, loud, owner


def _triad_cost(spec_a, spec_b, spec_c, ref_freq):
    """Exact, unnormalised dissonance of A : rB : sC as a function of (ln r, ln s)."""
    base, loud, owner = _triad_arrays((spec_a, spec_b, spec_c), ref_freq)
    i, j = np.triu_indices(len(base), k=1)
    li, lj = loud[i], loud[j]

    def cost(u, v):
        f = base * np.array([1.0, np.exp(u), np.exp(v)])[owner]
        return float(dissonance_pair_vectorized(f[i], f[j], li, lj).sum())
    return cost


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


def find_minima_3d(ratios_x, ratios_y, z_data, spectra, ref_freq, top_n=10,
                   progress=None, check=None):
    """Find the most consonant triads (local minima of the dissonance surface).

    *z_data* is indexed ``[voice B index, voice C index]``.  Every grid local
    minimum seeds a search on the exact model, so results aren't snapped to the
    grid and the aliasing ripples a grid leaves along narrow valleys merge into
    the true minimum they belong to.  Minima that run off the grid are dropped.
    Results are sorted by dissonance (lowest first) and capped at *top_n*.
    """
    empty = (np.array([]), np.array([]), np.array([]))
    nx, ny = z_data.shape
    if nx < 3 or ny < 3:
        return empty
    # The surface is symmetric only when B and C share a spectrum; mirrored pairs
    # are duplicate voicings then, and genuinely different chords otherwise.
    mirrored = nx == ny and np.allclose(z_data, z_data.T)

    padded = np.pad(z_data, 1, constant_values=np.inf)
    neighbours = np.min([padded[1 + di:nx + 1 + di, 1 + dj:ny + 1 + dj]
                         for di in (-1, 0, 1) for dj in (-1, 0, 1)
                         if di or dj], axis=0)
    si, sj = np.nonzero(z_data <= neighbours)
    us, vs = np.log(ratios_x[si]), np.log(ratios_y[sj])

    peak = np.unravel_index(np.argmax(z_data), z_data.shape)
    pu, pv = np.log(ratios_x[peak[0]]), np.log(ratios_y[peak[1]])
    lo = (np.log(ratios_x[0]), np.log(ratios_y[0]))
    hi = (np.log(ratios_x[-1]), np.log(ratios_y[-1]))
    step = float(ratios_x[1] - ratios_x[0])
    n = len(us)

    hits = []
    if HAS_NUMBA:
        base, loud, owner = _triad_arrays(spectra, ref_freq)
        norm = _nb_triad_cost(base, loud, owner, pu, pv) / z_data[peak]
        chunk = max(16, -(-n // 4))
        for k0 in range(0, n, chunk):
            if check is not None:
                check()
            k1 = min(n, k0 + chunk)
            ou, ov, ob, ok = _nb_descend_many(base, loud, owner, us[k0:k1], vs[k0:k1],
                                              step, lo[0], lo[1], hi[0], hi[1], 1e-6, _DIRS)
            hits.extend((u, v, b) if good else None
                        for u, v, b, good in zip(ou.tolist(), ov.tolist(), ob.tolist(),
                                                 ok.tolist()))
            if progress is not None:
                progress(k1 / n)
    else:
        cost = _triad_cost(*spectra, ref_freq)
        norm = cost(pu, pv) / z_data[peak]
        for k in range(n):
            if check is not None and k % 8 == 0:
                check()
            hits.append(_descend(cost, us[k], vs[k], step, lo, hi))
            if progress is not None and k % 8 == 7:
                progress((k + 1) / n)

    same = 2.0 / 1200 * np.log(2)  # minima within 2 cents on both axes are one chord
    tol = 0.02  # tolerance for "same note" check
    found = []
    fu, fv = np.empty(len(hits)), np.empty(len(hits))
    for hit in hits:
        if hit is None:
            continue
        u, v, val = hit
        if mirrored and u < v:
            u, v = v, u
        bx, cy = np.exp(u), np.exp(v)
        # Skip triads where two voices are effectively the same pitch
        if abs(bx - 1.0) < tol or abs(cy - 1.0) < tol or abs(bx - cy) < tol:
            continue
        k = len(found)
        if k and np.any((np.abs(fu[:k] - u) < same) & (np.abs(fv[:k] - v) < same)):
            continue
        fu[k], fv[k] = u, v
        found.append((u, v, val / norm))

    results = [(np.exp(u), np.exp(v), z) for u, v, z in found]
    if not results:
        return empty
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


def just_ratio(ratio, max_den=16, max_num=48, tol_cents=12.0):
    """Simplest p/q near *ratio*: (p, q, cents ratio is sharp of p/q), or None."""
    if not ratio > 0:
        return None
    cents = 1200 * math.log2(ratio)
    cands = []
    for q in range(1, max_den + 1):
        p = round(ratio * q)
        if 1 <= p <= max_num and math.gcd(p, q) == 1:
            cands.append((p, q, cents - 1200 * math.log2(p / q)))
    if not cands:
        return None
    close = [c for c in cands if abs(c[2]) <= tol_cents]
    if close:
        return min(close, key=lambda c: (c[0] * c[1], abs(c[2])))
    return min(cands, key=lambda c: abs(c[2]))


def triad_proportion(rb, rc, max_base=24, tol_cents=12.0):
    """Smallest whole-number chord a:b:c matching 1 : rb : rc, with cents offsets."""
    for a in range(1, max_base + 1):
        b, c = round(a * rb), round(a * rc)
        if b < 1 or c < 1:
            continue
        eb = 1200 * math.log2(rb * a / b)
        ec = 1200 * math.log2(rc * a / c)
        if abs(eb) <= tol_cents and abs(ec) <= tol_cents:
            g = math.gcd(math.gcd(a, b), c)
            return a // g, b // g, c // g, eb, ec
    return None


def _cents_text(off):
    return "exact" if abs(off) < 0.05 else f"{off:+.1f}\u00a2"


MATCH_CENTS = 5.0


def shared_partials(voices, ref_freq):
    """Partials of different voices within MATCH_CENTS of each other.

    *voices*: list of (letter, spectrum, ratio). Returns [(freq Hz, height)], where
    height is the louder of the two normalised amplitudes.
    """
    parts = []
    for _, spec, ratio in voices:
        f = ref_freq * ratio * np.asarray(spec["freq"], dtype=float)
        a = np.abs(np.asarray(spec["amp"], dtype=float))
        a = a / a.max() if a.size and a.max() > 0 else a
        parts.append((f, a))
    out = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            (fi, ai), (fj, aj) = parts[i], parts[j]
            F1, F2 = (g.ravel() for g in np.meshgrid(fi, fj, indexing='ij'))
            A1, A2 = (g.ravel() for g in np.meshgrid(ai, aj, indexing='ij'))
            cents = 1200 * np.abs(np.log2(F2 / F1))
            for k in np.nonzero((cents < MATCH_CENTS) & (np.minimum(A1, A2) > 0.02))[0]:
                out.append((float(F1[k]), float(max(A1[k], A2[k]))))
    return out


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


def parse_spectrum(freq_text, amp_text):
    """Spectrum dict from the two text fields, or (None, reason)."""
    try:
        freqs = [float(x.strip()) for x in freq_text.split(',')]
        amps = [float(x.strip()) for x in amp_text.split(',')]
    except ValueError:
        return None, "Partials and amplitudes must be comma-separated numbers"
    if not freqs or len(freqs) != len(amps):
        return None, "Partials and amplitudes need the same number of values"
    if not all(math.isfinite(v) for v in freqs + amps):
        return None, "Values must be finite numbers"
    if min(freqs) <= 0:
        return None, "Partial ratios must be greater than 0"
    return {"freq": freqs, "amp": amps}, None


# ═══════════════════════════════════════════════════════════════════════════
#  Audio engine
# ═══════════════════════════════════════════════════════════════════════════

class _Clip:
    """A pre-rendered sound in playback; a negative step fades it out."""

    __slots__ = ("buf", "pos", "gain", "step")

    def __init__(self, buf):
        self.buf = buf
        self.pos = 0
        self.gain = 1.0
        self.step = 0.0

    def fade(self, seconds):
        if self.step == 0.0:
            self.step = -1.0 / max(1.0, seconds * SAMPLE_RATE)


class _Glide:
    """Phase-continuous additive voices whose pitch ratios glide toward a target."""

    TAU = 0.03       # glide time constant: smooth, but still follows the mouse closely
    ATTACK = 0.05    # the same fade-in as synthesize_multi
    RELEASE = 0.06

    def __init__(self, voices, ref_freq):
        fr, amp, owner = [], [], []
        for v, (spec, _) in enumerate(voices):
            f = np.asarray(spec["freq"], dtype=float)
            gain = max(1.0 / len(f), 0.5)
            fr.append(f)
            amp.append(gain * np.asarray(spec["amp"], dtype=float))
            owner.append(np.full(len(f), v))
        self.fr = np.concatenate(fr)
        self.amp = np.concatenate(amp)
        self.owner = np.concatenate(owner)
        self.ref = float(ref_freq)
        self.cur = np.log(np.array([float(r) for _, r in voices]))
        self.target = self.cur.copy()
        self.phase = np.zeros(self.fr.size)
        self.env = 0.0
        self.releasing = False
        self.done = False
        # Match a clicked sound's loudness: synthesize_multi peak-normalises to 0.7.
        t = np.arange(int(0.5 * SAMPLE_RATE)) / SAMPLE_RATE
        f0 = self.ref * np.exp(self.cur)[self.owner] * self.fr
        peak = float(np.max(np.abs(np.sin(2 * np.pi * np.outer(t, f0)) @ self.amp)))
        self.scale = 0.7 / peak if peak > 0 else 0.0

    def set_ratios(self, ratios):
        self.target = np.log(np.asarray(ratios, dtype=float))

    def render(self, n):
        a = 1.0 - math.exp(-n / (SAMPLE_RATE * self.TAU))
        new = self.cur + (self.target - self.cur) * a
        ramp = (np.arange(n) / max(n, 1))[:, None]
        logr = self.cur[None, :] + (new - self.cur)[None, :] * ramp
        inc = (2 * np.pi / SAMPLE_RATE) * self.ref * np.exp(logr)[:, self.owner] * self.fr[None, :]
        ph = self.phase[None, :] + np.cumsum(inc, axis=0) - inc
        self.phase = np.mod(ph[-1] + inc[-1], 2 * np.pi)
        self.cur = new
        step = (-1.0 / (self.RELEASE * SAMPLE_RATE) if self.releasing
                else 1.0 / (self.ATTACK * SAMPLE_RATE))
        env = np.clip(self.env + step * np.arange(1, n + 1), 0.0, 1.0)
        self.env = float(env[-1])
        if self.releasing and self.env <= 0.0:
            self.done = True
        return (np.sin(ph) @ self.amp) * env * self.scale


class AudioEngine(QObject):
    """One output stream: clicked sounds crossfade, glides follow the mouse."""

    activityChanged = Signal(bool)
    level = Signal(float)
    failed = Signal(str)

    XFADE = 0.025
    STOP_FADE = 0.03
    # Close the device after this much silence; an open stream keeps Windows awake.
    IDLE_CLOSE = 8.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lock = threading.Lock()
        self._clips = []
        self._glides = []
        self._volume = 1.0
        self._gain = 1.0
        self._stream = None
        self._rms = 0.0
        self._active = False
        self._quiet_since = None
        self._poll = QTimer(self)
        self._poll.setInterval(40)
        self._poll.timeout.connect(self._on_poll)

    # ── public API ─────────────────────────────────────────────────────────

    def play(self, signal):
        if not self._open():
            return False
        clip = _Clip(np.asarray(signal, dtype=np.float64))
        with self._lock:
            for c in self._clips:
                c.fade(self.XFADE)
            for g in self._glides:
                g.releasing = True
            self._clips.append(clip)
        self._quiet_since = None
        return True

    def glide_begin(self, voices, ref_freq):
        if not self._open():
            return False
        glide = _Glide(voices, ref_freq)
        with self._lock:
            for c in self._clips:
                c.fade(self.XFADE)
            for g in self._glides:
                g.releasing = True
            self._glides.append(glide)
        self._quiet_since = None
        return True

    def glide_to(self, ratios):
        with self._lock:
            if self._glides and not self._glides[-1].releasing:
                self._glides[-1].set_ratios(ratios)

    def glide_end(self):
        with self._lock:
            for g in self._glides:
                g.releasing = True

    def stop(self):
        with self._lock:
            for c in self._clips:
                c.step = 0.0
                c.fade(self.STOP_FADE)
            for g in self._glides:
                g.releasing = True

    def set_volume(self, value):
        self._volume = float(np.clip(value, 0.0, 1.0))

    def is_active(self):
        return bool(self._clips or self._glides)

    def close(self):
        self._poll.stop()
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    # ── stream ─────────────────────────────────────────────────────────────

    @staticmethod
    def _stream_options():
        opts = []
        if sys.platform == "win32":
            # WASAPI shared mode has a few ms of latency instead of MME's ~90 ms.
            try:
                for api in sd.query_hostapis():
                    dev = api.get("default_output_device", -1)
                    if "WASAPI" in api["name"] and dev is not None and dev >= 0:
                        opts.append(dict(device=dev, latency="low",
                                         extra_settings=sd.WasapiSettings(auto_convert=True)))
            except Exception:
                pass
        return opts + [dict(latency="low"), {}]

    def _open(self):
        if self._stream is not None:
            return True
        last = None
        for kwargs in self._stream_options():
            for channels in (2, 1):
                try:
                    stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=channels,
                                             dtype="float32", callback=self._callback,
                                             **kwargs)
                    stream.start()
                except Exception as exc:
                    last = exc
                    continue
                self._stream = stream
                self._poll.start()
                return True
        self.failed.emit(f"No audio output available ({last})")
        return False

    def _callback(self, outdata, frames, time_info, status):
        try:
            mix = np.zeros(frames)
            with self._lock:
                keep = []
                for c in self._clips:
                    n = min(frames, c.buf.size - c.pos)
                    if n > 0:
                        seg = c.buf[c.pos:c.pos + n]
                        if c.step:
                            g = np.clip(c.gain + c.step * np.arange(1, n + 1), 0.0, 1.0)
                            mix[:n] += seg * g
                            c.gain = float(g[-1])
                        else:
                            mix[:n] += seg
                        c.pos += n
                    if c.pos < c.buf.size and c.gain > 0.0:
                        keep.append(c)
                self._clips = keep
                for g in self._glides:
                    mix += g.render(frames)
                self._glides = [g for g in self._glides if not g.done]
            v0, v1 = self._gain, self._volume
            mix *= np.linspace(v0, v1, frames) if v0 != v1 else v1
            self._gain = v1
            # Soft knee: only overlapping crossfades ever get near full scale.
            over = np.abs(mix) > 0.8
            if over.any():
                x = mix[over]
                mix[over] = np.sign(x) * (0.8 + 0.2 * np.tanh((np.abs(x) - 0.8) / 0.2))
            outdata[:] = mix[:, None].astype(np.float32)
            self._rms = float(np.sqrt(np.mean(mix * mix))) if frames else 0.0
        except Exception:
            outdata.fill(0)

    def _on_poll(self):
        active = self.is_active()
        if active != self._active:
            self._active = active
            self.activityChanged.emit(active)
        self.level.emit(self._rms if active else 0.0)
        if active:
            self._quiet_since = None
        elif self._quiet_since is None:
            self._quiet_since = time.monotonic()
        elif time.monotonic() - self._quiet_since > self.IDLE_CLOSE:
            self.close()


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


def is_dark():
    return CURRENT_PALETTE is DARK_PALETTE


QSS = """
* {{ outline: none; }}

QWidget {{
    color: {TEXT};
    font-family: "Segoe UI Variable Display", "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}}
QMainWindow, QDialog {{ background: {BASE}; }}
QLabel, QCheckBox, QStackedWidget, #plain {{ background: transparent; }}

#cardTitle  {{ font-size: 10px; font-weight: 700; color: {MUTED}; letter-spacing: 1.6px; }}
#fieldLabel {{ font-size: 11px; color: {SUBTEXT}; font-weight: 600; }}
#hint       {{ font-size: 11px; color: {MUTED}; }}
#statusText {{ font-size: 12px; color: {SUBTEXT}; }}
#emptyText  {{ font-size: 12px; color: {MUTED}; padding: 18px; }}
#errorText  {{ font-size: 11px; color: {RED}; }}
#appTitle   {{ font-size: 15px; font-weight: 700; color: {TEXT}; }}
#appSub     {{ font-size: 11px; color: {MUTED}; }}
#viewTitle  {{ font-size: 13px; font-weight: 700; color: {TEXT}; }}
#detailKey  {{ font-size: 11px; color: {MUTED}; }}
#detailKind {{ font-size: 10px; font-weight: 700; color: {BLUE}; letter-spacing: 1.6px; }}
#vsep       {{ background: {PANEL}; }}

#card {{
    background: {MANTLE};
    border: 1px solid {LINE};
    border-radius: 14px;
}}

QLineEdit, QComboBox, QDoubleSpinBox {{
    background: {SURFACE};
    border: 1px solid {PANEL};
    border-radius: 8px;
    padding: 5px 9px;
    color: {TEXT};
    selection-background-color: {BLUE};
    selection-color: {BASE};
}}
QLineEdit:hover, QComboBox:hover, QDoubleSpinBox:hover {{ border-color: {BORDER}; }}
QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus {{ border-color: {BLUE}; }}
QLineEdit[invalid="true"] {{ border-color: {RED}; }}
QCheckBox {{ spacing: 8px; }}

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
QPushButton#primary:hover  {{ background: {BLUE_HOVER}; }}
QPushButton#primary:pressed {{ background: {BLUE_PRESS}; }}
QPushButton#primary:disabled {{ background: {PANEL}; color: {MUTED}; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px 2px 2px 0; }}
QScrollBar::handle:vertical {{ background: {PANEL}; border-radius: 4px; min-height: 36px; }}
QScrollBar::handle:vertical:hover {{ background: {BORDER}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0 2px 2px 2px; }}
QScrollBar::handle:horizontal {{ background: {PANEL}; border-radius: 4px; min-width: 36px; }}
QScrollBar::handle:horizontal:hover {{ background: {BORDER}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QListView {{ background: transparent; border: none; }}

QSlider {{ background: transparent; }}
QSlider::groove:horizontal {{ height: 4px; background: {PANEL}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {BLUE}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {TEXT}; width: 12px; height: 12px; margin: -4px 0; border-radius: 6px;
}}
QSlider::handle:horizontal:hover {{ background: {BLUE}; }}

QSplitter::handle {{ background: transparent; }}
QSplitter::handle:hover {{ background: {PANEL}; }}

QToolTip {{
    background: {SURFACE}; color: {TEXT}; border: 1px solid {PANEL};
    padding: 6px 8px; border-radius: 6px;
}}
"""


def _chevron(color, hover=False):
    """Render a small chevron to a temp PNG and return a QSS-friendly path."""
    import tempfile
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
    pal = dict(palette or CURRENT_PALETTE)
    accent = QColor(pal["BLUE"])
    pal["BLUE_HOVER"] = accent.lighter(112).name()
    pal["BLUE_PRESS"] = accent.darker(115).name()
    return (QSS.format(**pal)
               .replace("__CHEVRON__", _chevron(pal["SUBTEXT"]))
               .replace("__CHEVRON_HOVER__", _chevron(pal["BLUE"], hover=True)))


_FONTS = {}


def mono_font(size=12, bold=False):
    # Font matching (exactMatch) is slow, and list rows ask for these on every paint.
    f = _FONTS.get((size, bold))
    if f is None:
        f = QFont("Cascadia Mono")
        if not f.exactMatch():
            f = QFont("Consolas")
        f.setPointSize(size)
        f.setBold(bold)
        _FONTS[(size, bold)] = f
    return QFont(f)


def mix(a, b, t):
    """Linear blend of two colours (t = 0 gives a, 1 gives b)."""
    a, b = QColor(a), QColor(b)
    t = float(np.clip(t, 0.0, 1.0))
    return QColor.fromRgbF(a.redF() + (b.redF() - a.redF()) * t,
                           a.greenF() + (b.greenF() - a.greenF()) * t,
                           a.blueF() + (b.blueF() - a.blueF()) * t,
                           a.alphaF() + (b.alphaF() - a.alphaF()) * t)


def with_alpha(color, alpha):
    c = QColor(color)
    c.setAlphaF(float(np.clip(alpha, 0.0, 1.0)))
    return c


def diss_color(value):
    """Green (consonant) -> yellow -> red (dissonant)."""
    v = float(np.clip(value, 0.0, 1.0))
    if v <= 0.45:
        return mix(GREEN, YELLOW, v / 0.45)
    return mix(YELLOW, RED, (v - 0.45) / 0.55)


_GLYPHS = {"ph.play-fill": "\u25b6", "ph.stop-fill": "\u25a0", "ph.x": "\u2715",
           "ph.sun": "\u263c", "ph.moon": "\u263e", "ph.check-circle": "\u2713",
           "ph.warning-circle": "!", "ph.info": "i"}
_ICON_CACHE = {}


def icon_pixmap(name, color, size, dpr=1.0):
    """A qtawesome icon rendered at *size* logical px, cached per colour."""
    if qta is None or not name:
        return None
    key = (name, QColor(color).name(QColor.HexArgb), size, round(dpr, 2))
    pm = _ICON_CACHE.get(key)
    if pm is None:
        try:
            pm = qta.icon(name, color=QColor(color)).pixmap(QSize(size, size), dpr)
        except Exception:
            pm = None
        _ICON_CACHE[key] = pm
    return pm


def draw_icon(p, name, color, rect, size):
    """Draw icon *name* centred in *rect*; falls back to a text glyph."""
    pm = icon_pixmap(name, color, size, p.device().devicePixelRatioF())
    if pm is not None and not pm.isNull():
        x = rect.center().x() - size / 2
        y = rect.center().y() - size / 2
        p.drawPixmap(QPointF(round(x), round(y)), pm)
        return
    glyph = _GLYPHS.get(name)
    if glyph:
        p.setPen(QColor(color))
        p.drawText(rect, Qt.AlignCenter, glyph)


def style_title_bar(window):
    """Colour the native Windows title bar to match the theme (Windows 11; dark mode on 10)."""
    if sys.platform != "win32":
        return
    try:
        hwnd = ctypes.c_void_p(int(window.winId()))
        dwm = ctypes.windll.dwmapi

        def put(attr, value):
            v = ctypes.c_int(value)
            dwm.DwmSetWindowAttribute(hwnd, ctypes.c_uint(attr), ctypes.byref(v),
                                      ctypes.sizeof(v))

        def colorref(hex_color):
            c = QColor(hex_color)
            return c.red() | (c.green() << 8) | (c.blue() << 16)

        put(20, 1 if is_dark() else 0)          # DWMWA_USE_IMMERSIVE_DARK_MODE
        put(35, colorref(BASE))                 # DWMWA_CAPTION_COLOR
        put(36, colorref(TEXT))                 # DWMWA_TEXT_COLOR
        put(34, colorref(LINE))                 # DWMWA_BORDER_COLOR
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  Motion helpers and custom-painted controls
# ═══════════════════════════════════════════════════════════════════════════

class Tween:
    """One eased float that can be retargeted mid-flight without a jump."""

    def __init__(self, owner, value=0.0, ms=160, on_change=None,
                 curve=QEasingCurve.OutCubic):
        self.value = float(value)
        self._cb = on_change
        self._anim = QVariantAnimation(owner)
        self._anim.setDuration(ms)
        self._anim.setEasingCurve(curve)
        self._anim.valueChanged.connect(self._step)

    def _step(self, v):
        self.value = float(v)
        if self._cb is not None:
            self._cb()

    def to(self, target, ms=None):
        self._anim.stop()
        if ms is not None:
            self._anim.setDuration(ms)
        if abs(float(target) - self.value) < 1e-9:
            self._step(target)
            return
        self._anim.setStartValue(self.value)
        self._anim.setEndValue(float(target))
        self._anim.start()

    def set(self, value):
        self._anim.stop()
        self._step(value)

    def running(self):
        return self._anim.state() == QAbstractAnimation.Running


def rounded(rect, radius):
    path = QPainterPath()
    path.addRoundedRect(rect, radius, radius)
    return path


class Button(QAbstractButton):
    """Rounded button whose colours ease between idle, hover and pressed states.

    kind: 'default', 'primary', 'ghost', 'danger' or 'play'.
    """

    def __init__(self, text="", icon=None, kind="default", tooltip=None, height=32,
                 parent=None):
        super().__init__(parent)
        self.setText(text)
        self.kind = kind
        self._icon = icon
        self._h = height
        self._hover = Tween(self, 0.0, 140, self.update)
        self._down = Tween(self, 0.0, 90, self.update)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.TabFocus)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        if tooltip:
            self.setToolTip(tooltip)

    def setIconName(self, name):
        self._icon = name
        self.updateGeometry()
        self.update()

    def sizeHint(self):
        fm = QFontMetricsF(self._font())
        if not self.text():
            return QSize(self._h, self._h)
        w = fm.horizontalAdvance(self.text()) + 28 + (22 if self._icon else 0)
        return QSize(int(math.ceil(w)), self._h)

    def minimumSizeHint(self):
        return self.sizeHint()

    def _font(self):
        f = QFont(self.font())
        if self.kind == "primary":
            f.setBold(True)
        return f

    def enterEvent(self, ev):
        self._hover.to(1.0)
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        self._hover.to(0.0, 220)
        super().leaveEvent(ev)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._down.to(1.0, 60)
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        self._down.to(0.0, 200)
        super().mouseReleaseEvent(ev)

    def changeEvent(self, ev):
        if ev.type() == QEvent.EnabledChange:
            self.update()
        super().changeEvent(ev)

    def _colors(self):
        h, d = self._hover.value, self._down.value
        if not self.isEnabled():
            return QColor(PANEL), None, QColor(MUTED)
        k = self.kind
        if k == "primary":
            accent = QColor(BLUE)
            bg = mix(mix(accent, accent.lighter(115), h), accent.darker(118), d)
            return bg, None, QColor(BASE)
        if k == "ghost":
            bg = mix(with_alpha(SURFACE, 0.0), QColor(SURFACE), h)
            return mix(bg, QColor(PANEL), d), None, mix(SUBTEXT, TEXT, h)
        if k == "danger":
            bg = mix(mix(SURFACE, RED, h), QColor(RED).darker(120), d)
            return bg, mix(mix(SURFACE, RED, 0.35), RED, h), mix(RED, BASE, h)
        if k == "play":
            bg = mix(mix(SURFACE, GREEN, h), QColor(GREEN).darker(120), d)
            return bg, mix(PANEL, GREEN, h), mix(GREEN, BASE, h)
        bg = mix(mix(SURFACE, PANEL, h), QColor(BORDER), d * 0.7)
        return bg, mix(PANEL, BORDER, h), QColor(TEXT)

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        bg, border, fg = self._colors()
        path = rounded(r, 8)
        p.fillPath(path, bg)
        if border is not None:
            p.setPen(QPen(border, 1.0))
            p.drawPath(path)
        if self.hasFocus() and self.focusPolicy() != Qt.NoFocus:
            p.setPen(QPen(with_alpha(BLUE, 0.8), 1.4))
            p.drawPath(rounded(r.adjusted(1.5, 1.5, -1.5, -1.5), 7))
        p.setFont(self._font())
        fm = QFontMetricsF(self._font())
        text = self.text()
        isz = 15 if text else 16
        if text:
            tw = fm.horizontalAdvance(text)
            total = tw + ((isz + 7) if self._icon else 0)
            x = r.center().x() - total / 2
            if self._icon:
                draw_icon(p, self._icon, fg, QRectF(x, r.top(), isz, r.height()), isz)
                x += isz + 7
            p.setPen(fg)
            p.drawText(QRectF(x, r.top(), tw + 2, r.height()), Qt.AlignVCenter | Qt.AlignLeft,
                       text)
        elif self._icon:
            draw_icon(p, self._icon, fg, r, isz)


class Segmented(QWidget):
    """Pill-style segmented control whose highlight slides between segments."""

    changed = Signal(int)

    def __init__(self, labels, icons=None, height=34, stretch=True, parent=None):
        super().__init__(parent)
        self._labels = list(labels)
        self._icons = list(icons) if icons else [None] * len(self._labels)
        self._colors = [None] * len(self._labels)
        self._index = 0
        self._hover = -1
        self._stretch = stretch
        self._pos = Tween(self, 0.0, 280, self.update)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(height)
        self.setFocusPolicy(Qt.TabFocus)
        self.setSizePolicy(QSizePolicy.Expanding if stretch else QSizePolicy.Fixed,
                           QSizePolicy.Fixed)

    def _font(self, selected=False):
        f = QFont(self.font())
        f.setWeight(QFont.DemiBold)
        return f

    def _natural(self, i):
        fm = QFontMetricsF(self._font())
        return fm.horizontalAdvance(self._labels[i]) + 30 + (22 if self._icons[i] else 0)

    def sizeHint(self):
        n = len(self._labels)
        return QSize(int(sum(self._natural(i) for i in range(n)) + 3 * (n - 1) + 8),
                     self.height())

    def minimumSizeHint(self):
        return self.sizeHint() if not self._stretch else QSize(60, self.height())

    def _rects(self):
        inner = QRectF(self.rect()).adjusted(4, 4, -4, -4)
        n = len(self._labels)
        gap = 3.0
        if self._stretch:
            widths = [(inner.width() - gap * (n - 1)) / n] * n
        else:
            widths = [self._natural(i) for i in range(n)]
        out, x = [], inner.left()
        for w in widths:
            out.append(QRectF(x, inner.top(), w, inner.height()))
            x += w + gap
        return out

    def currentIndex(self):
        return self._index

    def setCurrentIndex(self, index, animate=True):
        index = int(np.clip(index, 0, len(self._labels) - 1))
        self._index = index
        if animate and self.isVisible():
            self._pos.to(float(index))
        else:
            self._pos.set(float(index))

    def setLabel(self, i, text):
        self._labels[i] = text
        self.updateGeometry()
        self.update()

    def setColor(self, i, color):
        self._colors[i] = color
        self.update()

    def _hit(self, pos):
        for i, r in enumerate(self._rects()):
            if r.contains(QPointF(pos)):
                return i
        return -1

    def mouseMoveEvent(self, ev):
        h = self._hit(ev.position())
        if h != self._hover:
            self._hover = h
            self.update()

    def leaveEvent(self, ev):
        self._hover = -1
        self.update()
        super().leaveEvent(ev)

    def mousePressEvent(self, ev):
        i = self._hit(ev.position())
        if ev.button() == Qt.LeftButton and i >= 0 and i != self._index:
            self.setCurrentIndex(i)
            self.changed.emit(i)

    def keyPressEvent(self, ev):
        step = {Qt.Key_Left: -1, Qt.Key_Right: 1}.get(ev.key())
        if step:
            i = int(np.clip(self._index + step, 0, len(self._labels) - 1))
            if i != self._index:
                self.setCurrentIndex(i)
                self.changed.emit(i)
            return
        super().keyPressEvent(ev)

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        outer = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.fillPath(rounded(outer, 10), QColor(MANTLE))
        p.setPen(QPen(QColor(LINE), 1.0))
        p.drawPath(rounded(outer, 10))

        rects = self._rects()
        pos = float(np.clip(self._pos.value, 0, len(rects) - 1))
        i0 = int(math.floor(pos))
        i1 = min(i0 + 1, len(rects) - 1)
        t = pos - i0
        a, b = rects[i0], rects[i1]
        pill = QRectF(a.left() + (b.left() - a.left()) * t, a.top(),
                      a.width() + (b.width() - a.width()) * t, a.height())
        p.fillPath(rounded(pill, 7), QColor(PANEL))
        if self.hasFocus():
            p.setPen(QPen(with_alpha(BLUE, 0.7), 1.2))
            p.drawPath(rounded(pill, 7))

        p.setFont(self._font())
        fm = QFontMetricsF(self._font())
        for i, r in enumerate(rects):
            weight = max(0.0, 1.0 - abs(pos - i))
            custom = self._colors[i]
            if custom:
                idle, sel = mix(custom, MUTED, 0.3), QColor(custom)
            else:
                idle, sel = QColor(MUTED), QColor(BLUE)
            if i == self._hover and i != self._index:
                idle = mix(idle, TEXT, 0.6)
            col = mix(idle, sel, weight)
            text = self._labels[i]
            tw = fm.horizontalAdvance(text)
            icon = self._icons[i]
            total = tw + (22 if icon else 0)
            x = r.center().x() - total / 2
            if icon:
                draw_icon(p, icon, col, QRectF(x, r.top(), 16, r.height()), 16)
                x += 22
            p.setPen(col)
            p.drawText(QRectF(x, r.top(), tw + 2, r.height()), Qt.AlignVCenter | Qt.AlignLeft, text)


class Toggle(QAbstractButton):
    """Animated on/off switch with an optional caption."""

    def __init__(self, text="", checked=False, tooltip=None, parent=None):
        super().__init__(parent)
        self.setText(text)
        self.setCheckable(True)
        self.setChecked(checked)
        self._t = Tween(self, 1.0 if checked else 0.0, 200, self.update)
        self._hover = Tween(self, 0.0, 140, self.update)
        self.toggled.connect(lambda on: self._t.to(1.0 if on else 0.0))
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(28)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        if tooltip:
            self.setToolTip(tooltip)

    def sizeHint(self):
        fm = QFontMetricsF(self.font())
        w = 36 + (10 + fm.horizontalAdvance(self.text()) if self.text() else 0)
        return QSize(int(math.ceil(w)) + 2, 28)

    def enterEvent(self, ev):
        self._hover.to(1.0)
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        self._hover.to(0.0)
        super().leaveEvent(ev)

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        t, h = self._t.value, self._hover.value
        track = QRectF(1, (self.height() - 20) / 2, 36, 20)
        p.fillPath(rounded(track, 10), mix(mix(PANEL, BORDER, h * 0.5), BLUE, t))
        knob_x = track.left() + 10 + 16 * t
        knob = mix(SUBTEXT, "#ffffff" if is_dark() else BASE, t)
        p.setPen(Qt.NoPen)
        p.setBrush(knob)
        p.drawEllipse(QPointF(knob_x, track.center().y()), 7, 7)
        if self.text():
            p.setPen(mix(SUBTEXT, TEXT, max(h, t * 0.6)))
            p.drawText(QRectF(track.right() + 10, 0, self.width() - track.right() - 10,
                              self.height()), Qt.AlignVCenter | Qt.AlignLeft, self.text())


class _Snapshot(QWidget):
    """A frozen picture of a widget that fades away over the live one."""

    def __init__(self, parent, pixmap):
        super().__init__(parent)
        self.pix = pixmap
        self.opacity = 1.0
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setGeometry(parent.rect())

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setOpacity(self.opacity)
        p.drawPixmap(0, 0, self.pix)


class _Cover(QWidget):
    """Plain panel that hides its parent while something slow happens underneath."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setGeometry(parent.rect())
        parent.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Resize:
            self.setGeometry(obj.rect())
        return False

    def paintEvent(self, ev):
        QPainter(self).fillRect(self.rect(), QColor(MANTLE))


def crossfade(container, change, ms=220):
    """Snapshot *container*, apply *change()*, then dissolve the snapshot."""
    if not container.isVisible() or container.width() < 2 or container.height() < 2:
        change()
        return
    snap = _Snapshot(container, container.grab())
    snap.show()
    snap.raise_()
    change()
    snap.raise_()
    anim = QVariantAnimation(snap)
    anim.setDuration(ms)
    anim.setStartValue(1.0)
    anim.setEndValue(0.0)
    anim.setEasingCurve(QEasingCurve.InOutQuad)

    def step(v):
        snap.opacity = float(v)
        snap.update()

    anim.valueChanged.connect(step)
    anim.finished.connect(snap.deleteLater)
    anim.start()


class Toast(QWidget):
    """Transient notification that slides in at the bottom-right of its host."""

    SHADOW = 10

    def __init__(self, host):
        super().__init__(host)
        self.host = host
        self.kind = "ok"
        self._action = None
        self._closing = False
        self._fx = QGraphicsOpacityEffect(self)
        self._fx.setOpacity(0.0)
        self.setGraphicsEffect(self._fx)
        self._t = Tween(self, 0.0, 240, self._apply)
        lay = QHBoxLayout(self)
        s = self.SHADOW
        lay.setContentsMargins(14 + s, 10 + s, 10 + s, 10 + s)
        lay.setSpacing(10)
        self.icon = QLabel()
        self.icon.setFixedSize(18, 18)
        lay.addWidget(self.icon, 0, Qt.AlignVCenter)
        self.text = QLabel()
        self.text.setWordWrap(True)
        self.text.setMaximumWidth(340)
        lay.addWidget(self.text, 1)
        self.action_btn = Button("", kind="ghost", height=28)
        self.action_btn.clicked.connect(self._run_action)
        lay.addWidget(self.action_btn, 0)
        close = Button(icon="ph.x", kind="ghost", height=26, tooltip="Dismiss")
        close.clicked.connect(self.dismiss)
        lay.addWidget(close, 0)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.dismiss)
        host.installEventFilter(self)
        self.hide()

    def eventFilter(self, obj, ev):
        if obj is self.host and ev.type() == QEvent.Resize and self.isVisible():
            self._apply()
        return False

    def show_message(self, text, kind="ok", action=None, timeout=4500):
        self.kind = kind
        self._action = action
        color = {"ok": GREEN, "error": RED}.get(kind, BLUE)
        name = {"ok": "ph.check-circle", "error": "ph.warning-circle"}.get(kind, "ph.info")
        pm = icon_pixmap(name, color, 18, self.devicePixelRatioF())
        if pm is not None:
            self.icon.setPixmap(pm)
        else:
            self.icon.setText(_GLYPHS.get(name, ""))
        self.text.setText(text)
        self.action_btn.setVisible(action is not None)
        if action is not None:
            self.action_btn.setText(action[0])
            self.action_btn.updateGeometry()
        self.adjustSize()
        self._closing = False
        if not self.isVisible():
            self._t.set(0.0)
            self.show()
        self.raise_()
        self._t.to(1.0)
        self._timer.start(timeout)

    def dismiss(self):
        self._timer.stop()
        self._closing = True
        self._t.to(0.0, 200)

    def _run_action(self):
        if self._action is not None:
            self._action[1]()
        self.dismiss()

    def _apply(self):
        t = self._t.value
        if self._closing and t <= 0.001:
            self.hide()
            return
        x = self.host.width() - self.width() - 8
        y = self.host.height() - self.height() - 30 + round((1.0 - t) * 14)
        self.move(x, y)
        self._fx.setOpacity(t)

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        s = self.SHADOW
        r = QRectF(self.rect()).adjusted(s, s, -s, -s)
        for k in range(s, 0, -2):
            p.fillPath(rounded(r.adjusted(-k, -k + 3, k, k + 3), 12 + k),
                       QColor(0, 0, 0, int(26 * (1 - k / s) ** 2)))
        p.fillPath(rounded(r, 12), QColor(SURFACE))
        p.setPen(QPen(QColor(PANEL), 1.0))
        p.drawPath(rounded(r, 12))
        accent = {"ok": GREEN, "error": RED}.get(self.kind, BLUE)
        p.fillPath(rounded(QRectF(r.left() + 1, r.top() + 10, 3, r.height() - 20), 1.5),
                   QColor(accent))


class ActivityBar(QWidget):
    """Slim progress line; shown only when a computation is slow enough to notice."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(3)
        self._value = Tween(self, 0.0, 260, self.update)
        self._alpha = Tween(self, 0.0, 260, self.update)
        self._phase = 0.0
        self._tick = QTimer(self)
        self._tick.setInterval(16)
        self._tick.timeout.connect(self._advance)

    def start(self):
        self._value.set(0.0)
        self._alpha.to(1.0)
        self._tick.start()

    def setProgress(self, fraction):
        if self._alpha.value > 0 or self._alpha.running():
            self._value.to(max(self._value.value, float(fraction)), 180)

    def finish(self):
        if self._alpha.value <= 0.001 and not self._alpha.running():
            return
        self._value.to(1.0, 160)
        self._alpha.to(0.0, 420)
        QTimer.singleShot(460, self._stop_if_hidden)

    def _stop_if_hidden(self):
        if self._alpha.value <= 0.001:
            self._tick.stop()

    def _advance(self):
        self._phase = (self._phase + 0.012) % 1.0
        self.update()

    def paintEvent(self, ev):
        a = self._alpha.value
        if a <= 0.001:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        p.setOpacity(a)
        p.fillRect(QRectF(0, 0, w, h), with_alpha(BLUE, 0.12))
        fill = max(0.04, self._value.value) * w
        p.fillPath(rounded(QRectF(0, 0, fill, h), 1.5), QColor(BLUE))
        # A soft highlight sweeping along the filled part reads as "still working".
        cx = self._phase * fill
        grad = QLinearGradient(cx - 60, 0, cx + 60, 0)
        grad.setColorAt(0.0, with_alpha("#ffffff", 0.0))
        grad.setColorAt(0.5, with_alpha("#ffffff", 0.45))
        grad.setColorAt(1.0, with_alpha("#ffffff", 0.0))
        p.fillRect(QRectF(max(0, cx - 60), 0, min(120, fill), h), grad)


class LevelMeter(QWidget):
    """Tiny output level bar with fast attack and slow release."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(4, 18)
        self._level = 0.0

    def setLevel(self, rms):
        target = float(np.clip(rms * 2.6, 0.0, 1.0))
        self._level = target if target > self._level else self._level * 0.82 + target * 0.18
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = QRectF(self.rect())
        p.fillPath(rounded(r, 2), QColor(PANEL))
        h = r.height() * self._level
        if h > 0.5:
            p.fillPath(rounded(QRectF(r.left(), r.bottom() - h, r.width(), h), 2),
                       mix(GREEN, YELLOW, self._level))


class SmoothScroller(QObject):
    """Eases mouse-wheel scrolling of a scroll area instead of jumping by lines."""

    STEP = 72

    def __init__(self, area):
        super().__init__(area)
        self.bar = area.verticalScrollBar()
        self.target = None
        self.anim = QVariantAnimation(self)
        self.anim.setDuration(200)
        self.anim.setEasingCurve(QEasingCurve.OutCubic)
        self.anim.valueChanged.connect(lambda v: self.bar.setValue(int(round(v))))
        area.viewport().installEventFilter(self)

    def eventFilter(self, obj, ev):
        if ev.type() != QEvent.Wheel or ev.modifiers() != Qt.NoModifier:
            return False
        if not ev.pixelDelta().isNull():
            return False  # touchpads already scroll smoothly
        dy = ev.angleDelta().y()
        if dy == 0 or self.bar.maximum() <= self.bar.minimum():
            return False
        running = self.anim.state() == QAbstractAnimation.Running
        base = self.target if running and self.target is not None else self.bar.value()
        self.target = float(np.clip(base - dy / 120.0 * self.STEP,
                                    self.bar.minimum(), self.bar.maximum()))
        self.anim.stop()
        self.anim.setStartValue(float(self.bar.value()))
        self.anim.setEndValue(self.target)
        self.anim.start()
        return True


class NoWheelCombo(QComboBox):
    """Combo box that lets the wheel scroll its container unless it has focus."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, ev):
        if self.hasFocus():
            super().wheelEvent(ev)
        else:
            ev.ignore()


class Card(QFrame):
    """Rounded surface with a small-caps title and an optional header row."""

    def __init__(self, title=None, parent=None, margins=(14, 11, 14, 12)):
        super().__init__(parent)
        self.setObjectName("card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(*margins)
        outer.setSpacing(8)
        self.header = QHBoxLayout()
        self.header.setSpacing(8)
        if title:
            self.title = QLabel(title.upper())
            self.title.setObjectName("cardTitle")
            self.header.addWidget(self.title, 0, Qt.AlignVCenter)
        self.header.addStretch(1)
        outer.addLayout(self.header)
        self.body = QVBoxLayout()
        self.body.setSpacing(7)
        outer.addLayout(self.body, 1)


class Field(QWidget):
    """Labelled line edit; *inline* puts the label to the left."""

    def __init__(self, label, value="", inline=False, label_width=74, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self) if inline else QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8 if inline else 4)
        cap = QLabel(label)
        cap.setObjectName("fieldLabel")
        if inline:
            cap.setFixedWidth(label_width)
        lay.addWidget(cap)
        self.edit = QLineEdit(value)
        lay.addWidget(self.edit)

    def text(self):
        return self.edit.text()

    def setInvalid(self, invalid, message=None):
        if (self.edit.property("invalid") == "true") != bool(invalid):
            self.edit.setProperty("invalid", "true" if invalid else "false")
            self.edit.style().unpolish(self.edit)
            self.edit.style().polish(self.edit)
        self.edit.setToolTip(message if invalid and message else "")


class DissBar(QWidget):
    """Small rounded bar showing a 0-1 dissonance in its consonance colour."""

    def __init__(self, width=62, height=4, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.value = 0.0

    def setValue(self, value):
        self.value = float(np.clip(value, 0.0, 1.0))
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = QRectF(self.rect())
        rad = r.height() / 2
        p.fillPath(rounded(r, rad), QColor(SURFACE))
        if self.value > 0:
            p.fillPath(rounded(QRectF(r.left(), r.top(), max(r.height(), r.width() * self.value),
                                      r.height()), rad), diss_color(self.value))


# ═══════════════════════════════════════════════════════════════════════════
#  Spectrum editor
# ═══════════════════════════════════════════════════════════════════════════

class SpectrumEditor(QWidget):
    """Partials as stems you can drag: sideways retunes, up and down re-weights."""

    edited = Signal(list, list)
    editStarted = Signal()
    editFinished = Signal()

    PAD = (12, 16, 14, 20)  # left, top, right, bottom
    HIT = 12

    def __init__(self, color, parent=None):
        super().__init__(parent)
        self.color = QColor(color)
        self.freqs, self.amps = [], []
        self.valid = True
        self.linked_to = None
        self._hover = -1
        self._drag = None
        self._grow = Tween(self, 1.0, 340, self.update)
        self._hot = Tween(self, 0.0, 130, self.update)
        self.setMouseTracking(True)
        self.setFixedHeight(118)
        self.setToolTip("Drag a partial to retune or re-weight it\n"
                        "Shift: amplitude only  \u00b7  Ctrl: snap to a whole harmonic\n"
                        "Double-click to add a partial  \u00b7  right-click one to remove it")

    def setSpectrum(self, freqs, amps, valid=True, animate=False):
        if self._drag is not None:
            return
        self.freqs = [float(f) for f in freqs]
        self.amps = [float(a) for a in amps]
        self.valid = valid
        if self._hover >= len(self.freqs):
            self._hover = -1
        if animate and self.isVisible():
            self._grow.set(0.0)
            self._grow.to(1.0)
        self.update()

    def _ranges(self):
        if self._drag is not None:
            return self._drag["xmax"], self._drag["ymax"]
        fmax = max(self.freqs, default=1.0)
        amax = max((abs(a) for a in self.amps), default=1.0)
        return max(fmax * 1.12, 2.4), max(amax, 1.0) * 1.15

    def _plot(self):
        left, top, right, bottom = self.PAD
        return QRectF(self.rect()).adjusted(left, top, -right, -bottom)

    def _px(self, f, a):
        pr = self._plot()
        xmax, ymax = self._ranges()
        return QPointF(pr.left() + f / xmax * pr.width(),
                       pr.bottom() - max(a, 0.0) / ymax * pr.height())

    def _val(self, pos):
        pr = self._plot()
        xmax, ymax = self._ranges()
        return ((pos.x() - pr.left()) / max(pr.width(), 1.0) * xmax,
                (pr.bottom() - pos.y()) / max(pr.height(), 1.0) * ymax)

    def _hit(self, pos):
        best, best_d = -1, float(self.HIT)
        for i, (f, a) in enumerate(zip(self.freqs, self.amps)):
            q = self._px(f, abs(a))
            d = math.hypot(q.x() - pos.x(), q.y() - pos.y())
            if d < best_d:
                best, best_d = i, d
        return best

    @staticmethod
    def _round(v):
        return float(f"{v:.4g}")

    # ── interaction ────────────────────────────────────────────────────────

    def mousePressEvent(self, ev):
        pos = ev.position()
        i = self._hit(pos)
        if ev.button() == Qt.LeftButton and i >= 0 and self.valid:
            xmax, ymax = self._ranges()
            self._drag = dict(i=i, xmax=xmax, ymax=ymax, f0=self.freqs[i], moved=False)
            self._hot.to(1.0)
            self.setCursor(Qt.ClosedHandCursor)
            return
        if ev.button() == Qt.RightButton and i >= 0 and self.valid and len(self.freqs) > 1:
            self.editStarted.emit()
            del self.freqs[i]
            del self.amps[i]
            self._hover = -1
            self.edited.emit(list(self.freqs), list(self.amps))
            self.editFinished.emit()
            self.update()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        pos = ev.position()
        d = self._drag
        if d is not None:
            if not d["moved"]:
                d["moved"] = True
                self.editStarted.emit()
            f, a = self._val(pos)
            f = float(np.clip(f, 0.01, d["xmax"]))
            a = float(np.clip(a, 0.0, d["ymax"]))
            if ev.modifiers() & Qt.ShiftModifier:
                f = d["f0"]
            elif ev.modifiers() & Qt.ControlModifier:
                f = float(max(1, round(f)))
            self.freqs[d["i"]] = self._round(f)
            self.amps[d["i"]] = self._round(a)
            self.edited.emit(list(self.freqs), list(self.amps))
            self.update()
            return
        h = self._hit(pos)
        if h != self._hover:
            self._hover = h
            self._hot.set(0.0)
            if h >= 0:
                self._hot.to(1.0)
            self.setCursor(Qt.OpenHandCursor if h >= 0 else Qt.ArrowCursor)
            self.update()

    def mouseReleaseEvent(self, ev):
        if self._drag is not None and ev.button() == Qt.LeftButton:
            moved = self._drag["moved"]
            self._drag = None
            self.setCursor(Qt.OpenHandCursor if self._hit(ev.position()) >= 0
                           else Qt.ArrowCursor)
            if moved:
                self.editFinished.emit()
            self.update()
            return
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        pos = ev.position()
        if (ev.button() == Qt.LeftButton and self.valid and self._hit(pos) < 0
                and self._plot().contains(pos)):
            f, a = self._val(pos)
            self.editStarted.emit()
            self.freqs.append(self._round(max(f, 0.01)))
            self.amps.append(self._round(max(a, 0.0)))
            self.edited.emit(list(self.freqs), list(self.amps))
            self.editFinished.emit()
            self.update()
            return
        super().mouseDoubleClickEvent(ev)

    def leaveEvent(self, ev):
        if self._drag is None and self._hover != -1:
            self._hover = -1
            self._hot.set(0.0)
            self.update()
        super().leaveEvent(ev)

    # ── painting ───────────────────────────────────────────────────────────

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.fillPath(rounded(r, 10), QColor(SURFACE))
        p.setPen(QPen(QColor(LINE), 1.0))
        p.drawPath(rounded(r, 10))

        pr = self._plot()
        xmax, _ = self._ranges()
        small = QFont(self.font())
        small.setPixelSize(9)
        p.setFont(small)
        step = 1
        while pr.width() / xmax * step < 18:
            step *= 2
        for k in range(step, int(xmax) + 1, step):
            x = pr.left() + k / xmax * pr.width()
            p.setPen(QPen(with_alpha(BORDER, 0.4), 1.0, Qt.DotLine))
            p.drawLine(QPointF(x, pr.top()), QPointF(x, pr.bottom()))
            p.setPen(QColor(MUTED))
            p.drawText(QRectF(x - 14, pr.bottom() + 4, 28, 12), Qt.AlignHCenter | Qt.AlignTop,
                       str(k))
        p.setPen(QPen(QColor(BORDER), 1.0))
        p.drawLine(QPointF(pr.left(), pr.bottom()), QPointF(pr.right(), pr.bottom()))

        if not self.valid:
            p.setPen(QColor(RED))
            p.drawText(pr, Qt.AlignCenter, "Invalid spectrum")
            return

        g = self._grow.value
        active = self._drag["i"] if self._drag is not None else self._hover
        for i, (f, a) in enumerate(zip(self.freqs, self.amps)):
            top = self._px(f, abs(a) * g)
            hot = i == active
            pen = QPen(with_alpha(self.color, 0.95 if hot else 0.78), 2.6 if hot else 2.0)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.drawLine(QPointF(top.x(), pr.bottom()), top)
            rad = 4.0 + (2.2 * self._hot.value if hot else 0.0)
            if hot:
                p.setPen(Qt.NoPen)
                p.setBrush(with_alpha(self.color, 0.22 * self._hot.value))
                p.drawEllipse(top, rad + 5, rad + 5)
            p.setBrush(self.color)
            p.setPen(QPen(QColor(MANTLE), 1.4))
            p.drawEllipse(top, rad, rad)

        label = None
        if 0 <= active < len(self.freqs):
            label = (f"#{active + 1}   {self.freqs[active]:.4g}\u00d7   "
                     f"amp {self.amps[active]:.3g}")
        elif self.linked_to:
            label = f"same as {self.linked_to}  \u00b7  drag to detach"
        if label:
            font = mono_font(8)
            p.setFont(font)
            fm = QFontMetricsF(font)
            w = fm.horizontalAdvance(label) + 14
            box = QRectF(r.right() - w - 6, r.top() + 5, w, fm.height() + 6)
            p.fillPath(rounded(box, 6), with_alpha(MANTLE, 0.92))
            p.setPen(QColor(TEXT) if active >= 0 else QColor(MUTED))
            p.drawText(box, Qt.AlignCenter, label)


class SpectrumCard(QWidget):
    """Editor for one voice's spectrum: source menu, draggable partials, text fields."""

    playRequested = Signal(object)
    sourceChanged = Signal()
    spectrumChanged = Signal()
    dragging = Signal(bool)

    def __init__(self, letter, copy_sources=None, parent=None):
        super().__init__(parent)
        self.letter = letter
        self.copy_sources = copy_sources or []
        self.color = VOICE_COLORS[letter]
        self._animate_next = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(7)

        top = QHBoxLayout()
        top.setSpacing(6)
        self.combo = NoWheelCombo()
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
        self.combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo.setMinimumContentsLength(10)
        top.addWidget(self.combo, 1)
        play_btn = Button(icon="ph.play-fill", kind="play", tooltip=f"Play voice {letter} alone")
        play_btn.clicked.connect(lambda: self.playRequested.emit(self))
        top.addWidget(play_btn, 0)
        lay.addLayout(top)

        self.editor = SpectrumEditor(self.color)
        lay.addWidget(self.editor)

        _, freqs, amps = PRESETS[0]
        self.freq_field = Field("Partials", format_partials(freqs), inline=True)
        self.amp_field = Field("Amplitudes", format_partials(amps), inline=True)
        lay.addWidget(self.freq_field)
        lay.addWidget(self.amp_field)
        self.error = QLabel()
        self.error.setObjectName("errorText")
        self.error.setWordWrap(True)
        self.error.hide()
        lay.addWidget(self.error)
        lay.addStretch(1)

        self.freq_field.edit.textChanged.connect(self._on_text_changed)
        self.amp_field.edit.textChanged.connect(self._on_text_changed)
        # textEdited fires only for typing, not for text mirrored from the source.
        self.freq_field.edit.textEdited.connect(self._detach)
        self.amp_field.edit.textEdited.connect(self._detach)
        for _, card in self.copy_sources:
            card.freq_field.edit.textChanged.connect(self._sync_from_source)
            card.amp_field.edit.textChanged.connect(self._sync_from_source)
        self.editor.editStarted.connect(self._on_edit_started)
        self.editor.edited.connect(self._on_edited)
        self.editor.editFinished.connect(lambda: self.dragging.emit(False))

        if not self.copy_sources:
            self._select_by_text()
        self._on_combo()
        self.combo.currentIndexChanged.connect(self._on_combo)

    # ── behaviour ──────────────────────────────────────────────────────────

    def apply_palette(self):
        self.color = VOICE_COLORS[self.letter]
        self.editor.color = QColor(self.color)
        self.editor.update()

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

    def _set_texts(self, freq_text, amp_text):
        """Replace both fields, then announce one consistent change."""
        fe, ae = self.freq_field.edit, self.amp_field.edit
        if fe.text() == freq_text and ae.text() == amp_text:
            return
        for e in (fe, ae):
            e.blockSignals(True)
        fe.setText(freq_text)
        ae.setText(amp_text)
        for e in (fe, ae):
            e.blockSignals(False)
        ae.textChanged.emit(ae.text())

    def _sync_from_source(self):
        src = self._source_card()
        if src is not None:
            self._set_texts(src.freq_field.text(), src.amp_field.text())

    def _detach(self):
        if self.is_linked():
            self._select_by_text()
            self.sourceChanged.emit()

    def _on_edit_started(self):
        self._detach()
        self.dragging.emit(True)

    def _on_edited(self, freqs, amps):
        self._detach()
        self._set_texts(format_partials(freqs), format_partials(amps))

    def _on_text_changed(self):
        self._refresh_preview()
        if not self.is_linked():
            self._select_by_text()
        self.spectrumChanged.emit()

    def _on_combo(self, _index=None):
        kind, i = self._kind()
        self._animate_next = True
        if kind == "link":
            self._sync_from_source()
        elif kind == "preset":
            _, freqs, amps = PRESETS[i]
            self._set_texts(format_partials(freqs), format_partials(amps))
        self._update_tooltip()
        self._refresh_preview()
        self._animate_next = False
        self.sourceChanged.emit()
        self.spectrumChanged.emit()

    def _refresh_preview(self):
        spec, err = self.parsed()
        ok = spec is not None
        self.freq_field.setInvalid(not ok, err)
        self.amp_field.setInvalid(not ok, err)
        self.error.setText(err or "")
        self.error.setVisible(not ok)
        self.editor.linked_to = self.source_name()
        if ok:
            self.editor.setSpectrum(spec["freq"], spec["amp"], True, animate=self._animate_next)
        else:
            self.editor.setSpectrum(self.editor.freqs, self.editor.amps, False)

    def parsed(self):
        """(spectrum, None) or (None, reason); a linked voice reports its source."""
        src = self._source_card()
        if src is not None:
            return src.parsed()
        return parse_spectrum(self.freq_field.text(), self.amp_field.text())

    def get_spectrum(self):
        """Spectrum dict, copied source, or None when the text is invalid."""
        return self.parsed()[0]


# ═══════════════════════════════════════════════════════════════════════════
#  Plot views
# ═══════════════════════════════════════════════════════════════════════════

class SmoothImageItem(pg.ImageItem):
    """ImageItem drawn with bilinear filtering when scaled up."""

    def paint(self, p, *args):
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        super().paint(p, *args)


class ScrubViewBox(pg.ViewBox):
    """ViewBox where a left-drag glides the sound; Shift-drag or middle-drag pans."""

    scrubStarted = Signal(object)
    scrubMoved = Signal(object)
    scrubEnded = Signal()

    def __init__(self, corner_zoom=False, **kwargs):
        super().__init__(**kwargs)
        self.corner_zoom = corner_zoom
        self._scrubbing = False

    def scaleBy(self, s=None, center=None, x=None, y=None):
        # Corner zoom: wheel and right-drag zoom about the bottom-left, so only top and right move.
        if self.corner_zoom:
            r = self.targetRect().normalized()
            center = pg.Point(r.left(), r.top())
        super().scaleBy(s, center, x, y)

    def mouseDragEvent(self, ev, axis=None):
        if (axis is None and ev.button() == Qt.LeftButton
                and not ev.modifiers() & Qt.ShiftModifier):
            ev.accept()
            pos = self.mapSceneToView(ev.scenePos())
            if ev.isStart():
                self._scrubbing = True
                self.scrubStarted.emit(pos)
            elif self._scrubbing:
                self.scrubMoved.emit(pos)
                if ev.isFinish():
                    self._scrubbing = False
                    self.scrubEnded.emit()
            return
        super().mouseDragEvent(ev, axis)


def _flip_anchor(vb, scene_pt, width=170, height=80):
    """Readout anchor that keeps a tooltip-sized label inside the view box."""
    view = vb.sceneBoundingRect()
    ax = 1.04 if scene_pt.x() > view.right() - width else -0.04
    ay = -0.08 if scene_pt.y() < view.top() + height else 1.06
    return ax, ay


class CurveView(pg.PlotWidget):
    """2D dissonance curve: crosshair readout, snapping to minima, glide, morphs."""

    picked = Signal(float)
    scrubStarted = Signal(float)
    scrubMoved = Signal(float)
    scrubEnded = Signal()
    hoveredMinimum = Signal(object)

    SNAP_PX = 10

    def __init__(self, parent=None):
        self.vb = ScrubViewBox(corner_zoom=True)
        super().__init__(parent, viewBox=self.vb)
        pi = self.getPlotItem()
        pi.setMenuEnabled(False)
        pi.setMouseEnabled(x=True, y=False)
        pi.hideButtons()
        # The view is always set explicitly; auto-range would show the margin past MAX RATIO.
        pi.disableAutoRange()

        self._curve = pi.plot([], [], fillLevel=0.0)
        self._dots = pg.ScatterPlotItem(size=9)
        pi.addItem(self._dots)
        self._ring = pg.ScatterPlotItem(size=22, brush=pg.mkBrush(0, 0, 0, 0))
        self._ring.setVisible(False)
        pi.addItem(self._ring, ignoreBounds=True)
        self._play_line = pg.InfiniteLine(angle=90, movable=False)
        self._play_dot = pg.ScatterPlotItem(size=13)
        for item in (self._play_line, self._play_dot):
            item.setVisible(False)
            pi.addItem(item, ignoreBounds=True)

        self._vline = pg.InfiniteLine(angle=90, movable=False)
        self._vline.setVisible(False)
        pi.addItem(self._vline, ignoreBounds=True)
        self._readout = pg.TextItem(anchor=(0, 1))
        self._readout.setVisible(False)
        self._readout.setZValue(50)
        pi.addItem(self._readout, ignoreBounds=True)

        self._labels = []
        self._x = self._y = self._mx = self._my = None
        self._xmax = None
        self._shown_y = None
        self._from_y = self._to_y = None
        self._hl = None
        self._snap = None
        self._playing = None

        self._morph = QVariantAnimation(self)
        self._morph.setDuration(170)
        self._morph.setEasingCurve(QEasingCurve.OutCubic)
        self._morph.setStartValue(0.0)
        self._morph.setEndValue(1.0)
        self._morph.valueChanged.connect(self._morph_step)
        self._range_anim = QVariantAnimation(self)
        self._range_anim.setDuration(420)
        self._range_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._range_anim.setStartValue(0.0)
        self._range_anim.setEndValue(1.0)
        self._range_anim.valueChanged.connect(self._range_step)
        self._range_from = self._range_to = None
        self._pulse = QVariantAnimation(self)
        self._pulse.setDuration(1100)
        self._pulse.setStartValue(0.0)
        self._pulse.setEndValue(1.0)
        self._pulse.setLoopCount(-1)
        self._pulse.valueChanged.connect(self._pulse_step)
        self._play_alpha = Tween(self, 0.0, 300, self._style_playing)

        self._place_timer = QTimer(self)
        self._place_timer.setSingleShot(True)
        self._place_timer.setInterval(20)
        self._place_timer.timeout.connect(self._place_labels)
        self.vb.sigRangeChanged.connect(lambda *_: self._place_timer.start())
        self.vb.sigResized.connect(lambda *_: self._place_timer.start())
        self.vb.scrubStarted.connect(lambda p: self.scrubStarted.emit(self._clamp(p.x())))
        self.vb.scrubMoved.connect(lambda p: self.scrubMoved.emit(self._clamp(p.x())))
        self.vb.scrubEnded.connect(self.scrubEnded.emit)

        self.apply_palette()
        # Axes paint the grid too; caching them makes live curve updates cheap to redraw.
        for name in ('bottom', 'left'):
            pi.getAxis(name).setCacheMode(QGraphicsItem.DeviceCoordinateCache)
        self.scene().sigMouseMoved.connect(self._on_move)
        self.scene().sigMouseClicked.connect(self._on_click)

    @staticmethod
    def _clamp(x):
        return float(np.clip(x, 0.25, 8.0))

    def apply_palette(self):
        self.setBackground(MANTLE)
        self.showGrid(x=True, y=True, alpha=GRID_ALPHA)
        pi = self.getPlotItem()
        pi.setLabel('bottom', 'Frequency ratio  B / A', color=SUBTEXT, size='10pt')
        pi.setLabel('left', 'Sensory dissonance (normalised)', color=SUBTEXT, size='10pt')
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
        self._play_dot.setPen(pg.mkPen(MARKER_EDGE, width=1.2))
        self._play_dot.setBrush(pg.mkBrush(GREEN))
        self._style_playing()

        box = QColor(SURFACE)
        box.setAlpha(235)
        self._readout.fill = pg.mkBrush(box)
        self._readout.border = pg.mkPen(PANEL)
        self._readout.setColor(TEXT)
        for t in self._labels:
            t.setColor(YELLOW)

    # ── data ───────────────────────────────────────────────────────────────

    def setData(self, ratios, values, minima_x, minima_y, xmax, reset_view=True, animate=False):
        same_grid = (self._x is not None and len(self._x) == len(ratios)
                     and np.isclose(self._x[0], ratios[0]) and np.isclose(self._x[-1], ratios[-1]))
        self._x, self._y = ratios, values
        self._mx, self._my = minima_x, minima_y
        self._xmax = xmax
        if animate and same_grid and self._shown_y is not None:
            self._from_y = self._shown_y.copy()
            self._to_y = values
            self._morph.stop()
            self._morph.start()
        else:
            self._morph.stop()
            self._shown_y = values
            self._curve.setData(ratios, values)
        self.vb.setLimits(xMin=float(ratios[0]), xMax=float(xmax), yMin=0.0)
        self._rebuild_labels()
        if reset_view:
            self.setViewRatioRange(xmax, animate=animate)
        else:
            self._fit_y()
        self._refresh_markers()

    def _morph_step(self, t):
        if self._from_y is None or self._to_y is None:
            return
        self._shown_y = self._from_y + (self._to_y - self._from_y) * float(t)
        self._curve.setData(self._x, self._shown_y)
        self._refresh_markers()

    def _y_at(self, x):
        y = self._shown_y if self._shown_y is not None else self._y
        return float(np.interp(x, self._x, y))

    def _rebuild_labels(self):
        pi = self.getPlotItem()
        n = 0 if self._mx is None else len(self._mx)
        self._dots.setData(self._mx if n else [], self._my if n else [])
        while len(self._labels) > n:
            pi.removeItem(self._labels.pop())
        while len(self._labels) < n:
            t = pg.TextItem(anchor=(0.5, 0.0), color=YELLOW)
            t.setFont(QFont("Segoe UI", 7))
            pi.addItem(t, ignoreBounds=True)
            self._labels.append(t)
        for t, x, y in zip(self._labels, self._mx if n else [], self._my if n else []):
            t.setText(f"{ratio_to_interval_name(x)}\n{x:.3f}")
            t.setPos(x, y)
            t.setVisible(bool(x <= self._xmax))
        self._place_timer.start()

    def _place_labels(self):
        """Nudge minima labels (below first, then above) so none overlap."""
        if not self._labels or self._mx is None:
            return
        view = self.vb.sceneBoundingRect()
        placed = []
        for k in np.argsort(self._my):
            t = self._labels[k]
            if not t.isVisible():
                continue
            pt = self.vb.mapViewToScene(QPointF(float(self._mx[k]), float(self._my[k])))
            br = t.textItem.boundingRect()
            w, h = br.width(), br.height()
            # Slide the label sideways rather than let it run off either edge.
            ax = float(np.clip(0.5, 1 - (view.right() - 2 - pt.x()) / w,
                               (pt.x() - view.left() - 2) / w))
            chosen = None
            for ay in (-0.12, -1.1, -2.1, 1.15, 2.15):
                rect = QRectF(pt.x() - ax * w, pt.y() - ay * h, w, h)
                if rect.bottom() > view.bottom() + 2 or rect.top() < view.top() - 2:
                    continue
                if any(rect.intersects(r) for r in placed):
                    continue
                chosen = (ay, rect)
                break
            if chosen is None:
                chosen = (-0.12, QRectF(pt.x() - ax * w, pt.y() + 0.12 * h, w, h))
            t.setAnchor((ax, chosen[0]))
            placed.append(chosen[1])

    def setViewRatioRange(self, xmax, animate=False):
        """Zoom to ratios 0.9..xmax and rescale y to what's actually visible."""
        if self._x is None or not len(self._x):
            return
        visible = self._y[self._x <= xmax]
        top = float(np.max(visible)) if len(visible) else float(np.max(self._y))
        xr, yr = (float(self._x[0]), float(xmax)), (0.0, top * 1.08)
        if animate:
            (ax0, ax1), (ay0, ay1) = self.vb.viewRange()
            self._range_from = ((ax0, ax1), (ay0, ay1))
            self._range_to = (xr, yr)
            self._range_anim.stop()
            self._range_anim.start()
        else:
            self.setXRange(*xr, padding=0)
            self.setYRange(*yr, padding=0)
        if self._mx is not None and len(self._mx) == len(self._labels):
            for x, t in zip(self._mx, self._labels):
                t.setVisible(bool(x <= xmax))
        self._place_timer.start()

    def _range_step(self, t):
        if self._range_from is None:
            return
        t = float(t)
        (fx, fy), (tx, ty) = self._range_from, self._range_to
        self.vb.setRange(xRange=(fx[0] + (tx[0] - fx[0]) * t, fx[1] + (tx[1] - fx[1]) * t),
                         yRange=(fy[0] + (ty[0] - fy[0]) * t, fy[1] + (ty[1] - fy[1]) * t),
                         padding=0)

    def _fit_y(self):
        (x0, x1), _ = self.vb.viewRange()
        vis = self._y[(self._x >= x0) & (self._x <= x1)]
        top = float(np.max(vis)) if len(vis) else float(np.max(self._y))
        self.setYRange(0, top * 1.08, padding=0)

    # ── highlight, playing marker ──────────────────────────────────────────

    def setHighlight(self, x):
        self._hl = x
        self._refresh_markers()

    def setPlaying(self, x):
        if x is None:
            self._play_alpha.to(0.0, 420)
            return
        self._playing = float(x)
        self._refresh_markers()
        self._play_alpha.to(1.0, 120)

    def _refresh_markers(self):
        if self._x is None:
            return
        target = self._snap if self._snap is not None else self._hl
        if target is not None:
            self._ring.setData([target], [self._y_at(target)])
            self._ring.setVisible(True)
            if self._pulse.state() != QAbstractAnimation.Running:
                self._pulse.start()
        else:
            self._ring.setVisible(False)
            self._pulse.stop()
        if self._playing is not None:
            self._play_line.setPos(self._playing)
            self._play_dot.setData([self._playing], [self._y_at(self._playing)])

    def _pulse_step(self, t):
        s = math.sin(math.pi * float(t))
        self._ring.setSize(18 + 8 * s)
        self._ring.setPen(pg.mkPen(with_alpha(BLUE, 0.95 - 0.55 * s), width=2.0))

    def _style_playing(self):
        a = self._play_alpha.value
        on = a > 0.01 and self._playing is not None
        self._play_line.setVisible(on)
        self._play_dot.setVisible(on)
        if on:
            self._play_line.setPen(pg.mkPen(with_alpha(GREEN, 0.75 * a), width=1.6))
            self._play_dot.setOpacity(a)

    # ── mouse ──────────────────────────────────────────────────────────────

    def _on_move(self, pos):
        if self._x is None or not self.vb.sceneBoundingRect().contains(pos):
            self.hideCrosshair()
            return
        x = self.vb.mapSceneToView(pos).x()
        snap = None
        if self._mx is not None and len(self._mx):
            k = int(np.argmin(np.abs(self._mx - x)))
            px_w = self.vb.viewPixelSize()[0]
            if abs(self._mx[k] - x) <= self.SNAP_PX * px_w and self._mx[k] <= self._xmax:
                snap = float(self._mx[k])
        if snap is not None:
            x, y = snap, float(self._my[k])
        else:
            i = int(np.argmin(np.abs(self._x - x)))
            x, y = float(self._x[i]), float(self._y[i])
        if snap != self._snap:
            self._snap = snap
            self.hoveredMinimum.emit(snap)
            self._refresh_markers()
        self._vline.setPos(x)
        self._vline.setVisible(True)
        jr = just_ratio(x)
        just = (f"<br><span style='color:{MUTED}'>\u2248 {jr[0]}/{jr[1]}  {_cents_text(jr[2])}</span>"
                if jr and abs(jr[2]) <= 12 else "")
        self._readout.setHtml(
            f"<div style='font-family:Consolas;font-size:11px;line-height:150%'>"
            f"<b style='color:{BLUE}'>{x:.4f}</b> &nbsp;"
            f"<span style='color:{MUTED}'>{ratio_to_cents(x):.0f}\u00a2</span><br>"
            f"<span style='color:{SUBTEXT}'>{ratio_to_interval_name(x)}</span>{just}<br>"
            f"<span style='color:{diss_color(y).name()}'>diss {y:.3f}</span></div>")
        self._readout.setAnchor(_flip_anchor(self.vb, self.vb.mapViewToScene(QPointF(x, y))))
        self._readout.setPos(x, y)
        self._readout.setVisible(True)

    def _on_click(self, ev):
        if ev.button() != Qt.LeftButton or ev.double():
            return
        if not self.vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        x = self._snap if self._snap is not None else self.vb.mapSceneToView(ev.scenePos()).x()
        self.picked.emit(float(x))

    def leaveEvent(self, ev):
        self.hideCrosshair()
        super().leaveEvent(ev)

    def hideCrosshair(self):
        self._vline.setVisible(False)
        self._readout.setVisible(False)
        if self._snap is not None:
            self._snap = None
            self.hoveredMinimum.emit(None)
            self._refresh_markers()


class OvertoneView(pg.PlotWidget):
    """Partials of the last sound on one log-frequency axis; partials of
    different voices that coincide get a ring."""

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
        self._parts = []
        self._rings = []
        self._grow = Tween(self, 1.0, 300, self._draw)
        self.apply_palette()

    def apply_palette(self):
        self.setBackground(MANTLE)
        ax = self.getPlotItem().getAxis('bottom')
        ax.setPen(pg.mkPen(BORDER))
        ax.setTextPen(pg.mkPen(SUBTEXT))
        self._draw()

    def setVoices(self, voices, ref_freq, animate=False):
        """*voices*: list of (letter, spectrum, ratio). Returns the number of shared partials."""
        self._voices, self._ref = voices, ref_freq
        self._parts = []
        for letter, spec, ratio in voices:
            f = ref_freq * ratio * np.asarray(spec["freq"], dtype=float)
            a = np.abs(np.asarray(spec["amp"], dtype=float))
            a = a / a.max() if a.size and a.max() > 0 else a
            self._parts.append((letter, f, a))
        self._rings = shared_partials(voices, ref_freq)
        if animate:
            self._grow.set(0.0)
            self._grow.to(1.0)
        else:
            self._draw()
        return len(self._rings)

    def _draw(self):
        pi = self.getPlotItem()
        pi.clear()
        if not self._parts:
            pi.hideAxis('bottom')
            msg = pg.TextItem("Play something to see how its partials line up",
                              anchor=(0.5, 0.5), color=MUTED)
            pi.addItem(msg)
            msg.setPos(0.5, 0.5)
            pi.setXRange(0, 1, padding=0)
            pi.setYRange(0, 1, padding=0)
            return
        pi.showAxis('bottom')
        ref = self._ref
        g = self._grow.value

        def lx(f):
            return np.log2(np.asarray(f) / ref)

        for f, h in self._rings:
            pi.addItem(pg.ScatterPlotItem(
                x=[float(lx(f))], y=[(h + 0.12) * g], size=9, symbol='o',
                pen=pg.mkPen(with_alpha(TEXT, g), width=1.4), brush=pg.mkBrush(0, 0, 0, 0)))

        for k, (letter, f, a) in enumerate(self._parts):
            color = QColor(VOICE_COLORS[letter])
            color.setAlpha(200)
            x = np.repeat(lx(f), 2)
            y = np.column_stack([np.zeros_like(a), a * g]).ravel()
            pi.plot(x, y, connect='pairs', pen=pg.mkPen(color, width=self.WIDTHS[min(k, 2)]))

        all_x = np.concatenate([lx(f) for _, f, _ in self._parts])
        lo, hi = float(all_x.min()), float(all_x.max())
        pad = max(0.08, 0.04 * (hi - lo))
        pi.setXRange(lo - pad, hi + pad, padding=0)
        pi.setYRange(0, 1.25, padding=0)
        octaves = range(int(np.floor(lo - pad)), int(np.ceil(hi + pad)) + 1)
        pi.getAxis('bottom').setTicks([
            [(n, f"{ref * 2 ** n:.0f} Hz") for n in octaves],
            [(n + np.log2(h), "") for n in octaves for h in (1.25, 1.5, 1.75)],
        ])


class HeatmapView(pg.PlotWidget):
    """Triad space as an image with iso-contours, hover readout and glide."""

    picked = Signal(float, float)
    scrubStarted = Signal(float, float)
    scrubMoved = Signal(float, float)
    scrubEnded = Signal()
    hoveredTriad = Signal(object)

    SNAP_PX = 10

    def __init__(self, parent=None):
        self.vb = ScrubViewBox()
        super().__init__(parent, viewBox=self.vb)
        pi = self.getPlotItem()
        pi.setMenuEnabled(False)
        pi.setAspectLocked(True)
        pi.hideButtons()
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
        self._ring = pg.ScatterPlotItem(size=24, brush=pg.mkBrush(0, 0, 0, 0))
        self._play = pg.ScatterPlotItem(size=18, brush=pg.mkBrush(0, 0, 0, 0))
        self._hline = pg.InfiniteLine(angle=0, movable=False)
        self._vline = pg.InfiniteLine(angle=90, movable=False)
        for item in (self._ring, self._play, self._hline, self._vline):
            item.setVisible(False)
            pi.addItem(item, ignoreBounds=True)
        self._readout = pg.TextItem(anchor=(0, 1))
        self._readout.setVisible(False)
        self._readout.setZValue(50)
        pi.addItem(self._readout, ignoreBounds=True)

        self._labels = []
        self._x = self._y = self._z = None
        self._tri = (np.array([]), np.array([]))
        self._hl = None
        self._snap = None
        self._playing = None
        self._play_alpha = Tween(self, 0.0, 300, self._style_playing)
        self._pulse = QVariantAnimation(self)
        self._pulse.setDuration(1100)
        self._pulse.setStartValue(0.0)
        self._pulse.setEndValue(1.0)
        self._pulse.setLoopCount(-1)
        self._pulse.valueChanged.connect(self._pulse_step)
        self._place_timer = QTimer(self)
        self._place_timer.setSingleShot(True)
        self._place_timer.setInterval(20)
        self._place_timer.timeout.connect(self._place_labels)
        self.vb.sigRangeChanged.connect(lambda *_: self._place_timer.start())
        self.vb.sigResized.connect(lambda *_: self._place_timer.start())
        self.vb.scrubStarted.connect(lambda p: self._emit_scrub(p, self.scrubStarted))
        self.vb.scrubMoved.connect(lambda p: self._emit_scrub(p, self.scrubMoved))
        self.vb.scrubEnded.connect(self.scrubEnded.emit)

        self.apply_palette()
        self.scene().sigMouseClicked.connect(self._on_click)
        self.scene().sigMouseMoved.connect(self._on_move)

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
        cross = pg.mkPen(255, 255, 255, 90, width=1, style=Qt.DashLine)
        self._hline.setPen(cross)
        self._vline.setPen(cross)
        box = QColor(SURFACE)
        box.setAlpha(235)
        self._readout.fill = pg.mkBrush(box)
        self._readout.border = pg.mkPen(PANEL)
        self._readout.setColor(TEXT)
        self._style_playing()

    def setData(self, x, y, z, tri_x, tri_y, tri_z):
        self._x, self._y, self._z = x, y, z
        self._tri = (np.asarray(tri_x), np.asarray(tri_y))
        self.img.setImage(z, autoLevels=False)
        self.img.setRect(QRectF(x[0], y[0], x[-1] - x[0], y[-1] - y[0]))
        lo, hi = float(z.min()), float(z.max())
        self.img.setLevels((lo, hi))
        self.bar.setLevels((lo, hi))

        pi = self.getPlotItem()
        n = len(tri_x)
        self._dots.setData(tri_x if n else [], tri_y if n else [])
        while len(self._labels) > n:
            pi.removeItem(self._labels.pop())
        while len(self._labels) < n:
            t = pg.TextItem(anchor=(0, 1), color='#ffffff', fill=pg.mkBrush(0, 0, 0, 130))
            t.setFont(QFont("Segoe UI", 7, QFont.Bold))
            pi.addItem(t, ignoreBounds=True)
            self._labels.append(t)
        for t, bx, cy in zip(self._labels, tri_x, tri_y):
            t.setText(f"{bx:.2f} \u00b7 {cy:.2f}")
            t.setPos(bx, cy)
        self._refresh_markers()
        self._place_timer.start()

    def setContours(self, contours):
        """*contours*: (scale, [QPainterPath per level]) from contour_paths(), or None to hide."""
        if contours is None:
            for item in self._iso:
                item.setVisible(False)
            return
        scale, paths = contours
        while len(self._iso) < len(paths):
            item = QGraphicsPathItem(self.img)
            item.setPen(pg.mkPen(255, 255, 255, 34, width=1))
            item.setZValue(5)
            self._iso.append(item)
        for item, path in zip(self._iso, paths):
            item.setPath(path)
            item.setTransform(QTransform.fromScale(scale, scale))
            item.setVisible(True)
        for item in self._iso[len(paths):]:
            item.setVisible(False)

    def resetRange(self, animate=True):
        if self._x is None:
            return
        target = ((float(self._x[0]), float(self._x[-1])), (float(self._y[0]), float(self._y[-1])))
        if not animate:
            self.vb.setRange(xRange=target[0], yRange=target[1], padding=0)
            return
        start = tuple(tuple(r) for r in self.vb.viewRange())
        anim = QVariantAnimation(self)
        anim.setDuration(420)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)

        def step(t):
            t = float(t)
            (fx, fy), (tx, ty) = start, target
            self.vb.setRange(xRange=(fx[0] + (tx[0] - fx[0]) * t, fx[1] + (tx[1] - fx[1]) * t),
                             yRange=(fy[0] + (ty[0] - fy[0]) * t, fy[1] + (ty[1] - fy[1]) * t),
                             padding=0)

        anim.valueChanged.connect(step)
        anim.start(QAbstractAnimation.DeleteWhenStopped)

    def _place_labels(self):
        """Try the four corners around each triad so labels don't cover each other."""
        if not self._labels:
            return
        placed = []
        tx, ty = self._tri
        for k, t in enumerate(self._labels):
            pt = self.vb.mapViewToScene(QPointF(float(tx[k]), float(ty[k])))
            br = t.textItem.boundingRect()
            w, h = br.width(), br.height()
            chosen = None
            for ax, ay in ((-0.06, 1.06), (1.06, 1.06), (-0.06, -0.06), (1.06, -0.06),
                           (-0.06, 2.1), (-0.06, -1.1)):
                rect = QRectF(pt.x() - ax * w, pt.y() - ay * h, w, h)
                if not any(rect.intersects(r) for r in placed):
                    chosen = ((ax, ay), rect)
                    break
            if chosen is None:
                chosen = ((-0.06, 1.06), QRectF(pt.x() + 0.06 * w, pt.y() - 1.06 * h, w, h))
            t.setAnchor(chosen[0])
            placed.append(chosen[1])

    def _inside(self, p):
        return (self._x is not None and self._x[0] <= p.x() <= self._x[-1]
                and self._y[0] <= p.y() <= self._y[-1])

    def _emit_scrub(self, p, signal):
        if self._x is None:
            return
        signal.emit(float(np.clip(p.x(), self._x[0], self._x[-1])),
                    float(np.clip(p.y(), self._y[0], self._y[-1])))

    def _z_at(self, bx, cy):
        bi = int(np.argmin(np.abs(self._x - bx)))
        ci = int(np.argmin(np.abs(self._y - cy)))
        return float(self._z[bi, ci])

    def setHighlight(self, bc):
        self._hl = bc
        self._refresh_markers()

    def setPlaying(self, bc):
        if bc is None:
            self._play_alpha.to(0.0, 420)
            return
        self._playing = bc
        self._refresh_markers()
        self._play_alpha.to(1.0, 120)

    def _refresh_markers(self):
        target = self._snap if self._snap is not None else self._hl
        if target is not None:
            self._ring.setData([target[0]], [target[1]])
            self._ring.setVisible(True)
            if self._pulse.state() != QAbstractAnimation.Running:
                self._pulse.start()
        else:
            self._ring.setVisible(False)
            self._pulse.stop()
        if self._playing is not None:
            self._play.setData([self._playing[0]], [self._playing[1]])

    def _pulse_step(self, t):
        s = math.sin(math.pi * float(t))
        self._ring.setSize(20 + 9 * s)
        self._ring.setPen(pg.mkPen(255, 255, 255, int(240 - 140 * s), width=2.0))

    def _style_playing(self):
        a = self._play_alpha.value
        on = a > 0.01 and self._playing is not None
        self._play.setVisible(on)
        if on:
            self._play.setPen(pg.mkPen(with_alpha(GREEN, a), width=2.4))

    def _nearest_triad(self, pos):
        """The best triad within SNAP_PX of scene point *pos*, or None."""
        tx, ty = self._tri
        best, best_d = None, self.SNAP_PX
        for bx, cy in zip(tx, ty):
            q = self.vb.mapViewToScene(QPointF(float(bx), float(cy)))
            d = math.hypot(q.x() - pos.x(), q.y() - pos.y())
            if d <= best_d:
                best, best_d = (float(bx), float(cy)), d
        return best

    def _set_snap(self, snap):
        if snap != self._snap:
            self._snap = snap
            self.hoveredTriad.emit(snap)
            self._refresh_markers()

    def _on_move(self, pos):
        if self._z is None or not self.vb.sceneBoundingRect().contains(pos):
            self.hideCrosshair()
            return
        p = self.vb.mapSceneToView(pos)
        snap = self._nearest_triad(pos)
        if snap is None and not self._inside(p):
            self.hideCrosshair()
            return
        self._set_snap(snap)
        bx, cy = snap if snap is not None else (float(p.x()), float(p.y()))
        d = self._z_at(bx, cy)
        self._hline.setPos(cy)
        self._vline.setPos(bx)
        self._hline.setVisible(True)
        self._vline.setVisible(True)
        prop = triad_proportion(bx, cy, max_base=16, tol_cents=8)
        chord = (f" &nbsp;<span style='color:{MUTED}'>\u2248 {prop[0]}:{prop[1]}:{prop[2]}</span>"
                 if prop else "")
        self._readout.setHtml(
            f"<div style='font-family:Consolas;font-size:11px;line-height:150%'>"
            f"<b style='color:{VOICE_COLORS['B']}'>B {bx:.3f}</b> &nbsp;"
            f"<b style='color:{VOICE_COLORS['C']}'>C {cy:.3f}</b><br>"
            f"<span style='color:{SUBTEXT}'>{ratio_to_interval_name(bx)} \u00b7 "
            f"{ratio_to_interval_name(cy)}</span><br>"
            f"<span style='color:{diss_color(d).name()}'>diss {d:.3f}</span>{chord}</div>")
        self._readout.setAnchor(_flip_anchor(self.vb, self.vb.mapViewToScene(QPointF(bx, cy)), 220, 80))
        self._readout.setPos(bx, cy)
        self._readout.setVisible(True)

    def hideCrosshair(self):
        for item in (self._hline, self._vline, self._readout):
            item.setVisible(False)
        self._set_snap(None)

    def leaveEvent(self, ev):
        self.hideCrosshair()
        super().leaveEvent(ev)

    def _on_click(self, ev):
        if ev.button() != Qt.LeftButton or self._z is None or ev.double():
            return
        if not self.vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        if self._snap is not None:
            self.picked.emit(*self._snap)
            return
        p = self.vb.mapSceneToView(ev.scenePos())
        if self._inside(p):
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


def contour_paths(z, levels=7, max_side=140):
    """Iso-contours of the smoothed surface as (scale, [QPainterPath per level]).

    pyqtgraph traces contours in Python, so this runs on the worker thread, on a grid
    capped at *max_side*; the smoothing removes finer detail than the cap anyway.
    """
    s = max(1, -(-max(z.shape) // max_side))
    smooth = pg.gaussianFilter(z, (1.6, 1.6))[::s, ::s]
    lo, hi = float(z.min()), float(z.max())
    return s, [pg.functions.isocurve(smooth, lo + (hi - lo) * frac, connected=True,
                                     extendToEdge=True, path=True)
               for frac in np.linspace(0.12, 0.9, levels)]


if HAS_GL:

    from OpenGL import GL as _GL
    from PySide6 import QtOpenGL
    from pyqtgraph.opengl import MeshData as _MeshData

    def _vertex_normals(self, indexed=None):
        """MeshData.vertexNormals without its Python loop per vertex (seconds on a 260x260 grid).

        Same result: each vertex normal is the normalised sum of its faces' normals.
        """
        if self._vertexNormals is None:
            face_norms = self.faceNormals()
            idx = self.faces().ravel().astype(np.intp)
            n = self._vertexes.shape[0]
            acc = np.column_stack([np.bincount(idx, weights=np.repeat(face_norms[:, c], 3),
                                               minlength=n) for c in range(3)])
            length = np.sqrt((acc ** 2).sum(axis=1, keepdims=True))
            with np.errstate(invalid="ignore", divide="ignore"):
                acc = np.where(length > 0, acc / length, 0.0)
            self._vertexNormals = acc.astype(np.float32)
        if indexed is None:
            return self._vertexNormals
        if indexed == "faces":
            return self._vertexNormals[self.faces()]
        raise Exception("Invalid indexing mode. Accepts: None, 'faces'")

    _MeshData.vertexNormals = _vertex_normals

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

    def _pan_center(distance, elevation, azimuth):
        """Look-at point GLViewWidget.pan(..., 'view-upright') would give for SURFACE_PAN."""
        el, az = math.radians(elevation), math.radians(azimuth)
        cam = QVector3D(distance * math.cos(el) * math.cos(az),
                        distance * math.cos(el) * math.sin(az), distance * math.sin(el))
        look = -cam
        up = QVector3D(0, 0, 1)
        right = QVector3D.crossProduct(up, look).normalized()
        down = QVector3D.crossProduct(right, up).normalized()
        dx, dy = SURFACE_PAN
        return right * dx + down * dy

    class SurfaceView(gl.GLViewWidget):
        """OpenGL dissonance surface with ray-cast click-to-play and a hover probe."""

        picked = Signal(float, float)
        hoveredTriad = Signal(object)

        Z_SCALE = 1.1
        SNAP_PX = 12

        def __init__(self, parent=None):
            super().__init__(parent)
            fmt = QSurfaceFormat(self.format())
            fmt.setSamples(4)
            self.setFormat(fmt)
            self.setCameraPosition(**SURFACE_CAMERA)
            self.opts['fov'] = 50
            self.opts['center'] = pg.Vector(_pan_center(**SURFACE_CAMERA))

            self._grid = gl.GLGridItem()
            self._grid.setSize(2.0, 2.0)
            self._grid.setSpacing(0.2, 0.2)
            self.addItem(self._grid)

            self._surface = None
            self._dots = None
            self._text = []
            self._press = None
            self._x = self._y = self._z = None
            self._sx = self._sy = None
            self._zs = self._zs_raw = None
            self._last_set = 0.0
            self._lo, self._hi = 0.0, 1.0
            self._cmap = _theme_cmap()
            self.lit = False
            self._hl = None
            self._playing = None
            self._hover_pos = None
            self._tri = (np.array([]), np.array([]))
            self._tri_pts = np.zeros((0, 3), dtype=np.float32)
            self._snap = self._press_snap = None

            self._marks = {}
            for key, size in (("probe", 10.0), ("hl", 18.0), ("play", 16.0)):
                item = gl.GLScatterPlotItem(pos=np.zeros((1, 3), dtype=np.float32), size=size,
                                            color=(1, 1, 1, 1), pxMode=True)
                item.setGLOptions(ALWAYS_ON_TOP)
                item.setDepthValue(20)
                item.setVisible(False)
                self.addItem(item)
                self._marks[key] = item

            self._readout = QLabel(self)
            self._readout.setAttribute(Qt.WA_TransparentForMouseEvents)
            self._readout.setTextFormat(Qt.RichText)
            self._readout.hide()

            self._hover_timer = QTimer(self)
            self._hover_timer.setSingleShot(True)
            self._hover_timer.setInterval(16)
            self._hover_timer.timeout.connect(self._probe)

            self._morph = QVariantAnimation(self)
            self._morph.setDuration(180)
            self._morph.setEasingCurve(QEasingCurve.OutCubic)
            self._morph.setStartValue(0.0)
            self._morph.setEndValue(1.0)
            self._morph.valueChanged.connect(self._morph_step)
            self._morph_from = self._morph_to = None

            self._cam = QVariantAnimation(self)
            self._cam.setDuration(650)
            self._cam.setEasingCurve(QEasingCurve.InOutCubic)
            self._cam.setStartValue(0.0)
            self._cam.setEndValue(1.0)
            self._cam.valueChanged.connect(self._cam_step)
            self._cam_from = self._cam_to = None

            self._pulse = QVariantAnimation(self)
            self._pulse.setDuration(1100)
            self._pulse.setStartValue(0.0)
            self._pulse.setEndValue(1.0)
            self._pulse.setLoopCount(-1)
            self._pulse.valueChanged.connect(self._pulse_step)
            self._play_alpha = Tween(self, 0.0, 300, self._style_playing)

            self.setMouseTracking(True)
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
            self._readout.setStyleSheet(
                f"QLabel {{ background: {with_alpha(SURFACE, 0.94).name(QColor.HexArgb)};"
                f" border: 1px solid {PANEL}; border-radius: 6px; padding: 5px 8px; }}")
            self._marks["probe"].setData(color=QColor(TEXT).getRgbF())
            self._marks["hl"].setData(color=with_alpha(BLUE, 0.9).getRgbF())
            self._style_playing()
            if self._zs_raw is not None:
                self._show_z(self._zs_raw)

        # ── data ───────────────────────────────────────────────────────────

        def setData(self, x, y, z, tri_x, tri_y, tri_z, animate=False, coarse=False):
            # Dragging a partial refreshes the surface many times a second; a lighter mesh keeps up.
            step = max(1, -(-len(x) // 90)) if coarse else max(1, len(x) // 140)
            xs, ys = x[::step], y[::step]
            zs = np.ascontiguousarray(z[::step, ::step], dtype=np.float32)
            same = (self._zs_raw is not None and self._zs_raw.shape == zs.shape
                    and np.allclose(self._sx, xs) and np.allclose(self._sy, ys))
            # Updates arriving faster than a morph (dragging a partial) just replace the mesh.
            now = time.monotonic()
            animate = animate and now - self._last_set > 0.25
            self._last_set = now
            self._x, self._y, self._z = x, y, z
            self._sx, self._sy = xs, ys
            self._lo, self._hi = float(z.min()), float(z.max())
            if animate and same:
                self._morph_from, self._morph_to = self._zs_raw.copy(), zs
                self._morph.stop()
                self._morph.start()
            else:
                self._morph.stop()
                self._show_z(zs)

            if len(tri_x):
                pts = np.column_stack([
                    _to_unit(np.asarray(tri_x), x),
                    _to_unit(np.asarray(tri_y), y),
                    np.asarray(tri_z) * self.Z_SCALE + 0.012,
                ]).astype(np.float32)
                self._tri = (np.asarray(tri_x), np.asarray(tri_y))
                self._tri_pts = pts
                c = np.array(QColor(YELLOW).getRgbF(), dtype=np.float32)
                if self._dots is None:
                    self._dots = gl.GLScatterPlotItem(pos=pts, size=14.0, pxMode=True,
                                                      color=np.tile(c, (len(pts), 1)))
                    self._dots.setGLOptions(ALWAYS_ON_TOP)
                    self._dots.setDepthValue(10)
                    self.addItem(self._dots)
                else:
                    self._dots.setData(pos=pts, color=np.tile(c, (len(pts), 1)))
                self._dots.setVisible(True)
            else:
                self._tri = (np.array([]), np.array([]))
                self._tri_pts = np.zeros((0, 3), dtype=np.float32)
                if self._dots is not None:
                    self._dots.setVisible(False)
            self._rebuild_labels(x, y)
            self._refresh_markers()

        def _show_z(self, zs):
            self._zs_raw = zs
            self._zs = zs * self.Z_SCALE
            colors = self._cmap.map((zs - self._lo) / max(self._hi - self._lo, 1e-9), mode='float')
            colors = np.ascontiguousarray(colors.reshape(-1, 4).astype(np.float32))
            nx = _to_unit(self._sx, self._x).astype(np.float32)
            ny = _to_unit(self._sy, self._y).astype(np.float32)
            if self._surface is None:
                self._surface = gl.GLSurfacePlotItem(
                    x=nx, y=ny, z=self._zs, colors=colors,
                    shader='shaded' if self.lit else None, smooth=True, glOptions='opaque')
                self.addItem(self._surface)
            else:
                self._surface.setData(x=nx, y=ny, z=self._zs, colors=colors)

        def setLit(self, on):
            self.lit = bool(on)
            if self._surface is not None:
                self._surface.setShader('shaded' if self.lit else None)

        def _morph_step(self, t):
            if self._morph_from is None:
                return
            t = float(t)
            self._show_z(self._morph_from + (self._morph_to - self._morph_from) * t)

        def _rebuild_labels(self, x, y):
            if not hasattr(gl, 'GLTextItem'):
                return
            font = QFont("Segoe UI", 8)
            k = 0
            for frac in (0.0, 0.5, 1.0):
                vx = x[0] + (x[-1] - x[0]) * frac
                vy = y[0] + (y[-1] - y[0]) * frac
                for pos, text in (
                    ((frac * 2 - 1, -1.12, 0.0), f"B {vx:.2f}"),
                    ((-1.16, frac * 2 - 1, 0.0), f"C {vy:.2f}"),
                ):
                    pos = np.array(pos, dtype=np.float32)
                    if k < len(self._text):
                        self._text[k].setData(pos=pos, text=text)
                    else:
                        t = gl.GLTextItem(pos=pos, text=text, color=QColor(SUBTEXT), font=font)
                        self.addItem(t)
                        self._text.append(t)
                    k += 1

        _norm = staticmethod(_to_unit)

        # ── markers ────────────────────────────────────────────────────────

        def _point(self, bx, cy):
            if self._z is None:
                return None
            bi = int(np.argmin(np.abs(self._x - bx)))
            ci = int(np.argmin(np.abs(self._y - cy)))
            return np.array([[float(_to_unit([bx], self._x)[0]), float(_to_unit([cy], self._y)[0]),
                              float(self._z[bi, ci]) * self.Z_SCALE + 0.014]], dtype=np.float32)

        def setHighlight(self, bc):
            self._hl = bc
            self._refresh_markers()

        def setPlaying(self, bc):
            if bc is None:
                self._play_alpha.to(0.0, 420)
                return
            self._playing = bc
            self._refresh_markers()
            self._play_alpha.to(1.0, 120)

        def _refresh_markers(self):
            hl = self._marks["hl"]
            target = self._snap if self._snap is not None else self._hl
            pt = self._point(*target) if target is not None else None
            if pt is not None:
                hl.setData(pos=pt)
                hl.setVisible(True)
                if self._pulse.state() != QAbstractAnimation.Running:
                    self._pulse.start()
            else:
                hl.setVisible(False)
                self._pulse.stop()
            if self._playing is not None:
                pt = self._point(*self._playing)
                if pt is not None:
                    self._marks["play"].setData(pos=pt)

        def _pulse_step(self, t):
            s = math.sin(math.pi * float(t))
            self._marks["hl"].setData(size=16 + 10 * s,
                                      color=with_alpha(BLUE, 0.95 - 0.5 * s).getRgbF())

        def _style_playing(self):
            a = self._play_alpha.value
            item = self._marks["play"]
            on = a > 0.01 and self._playing is not None
            item.setVisible(on)
            if on:
                item.setData(color=with_alpha(GREEN, a).getRgbF())

        # ── picking and hover ──────────────────────────────────────────────

        def mousePressEvent(self, ev):
            self._press = ev.position()
            self._press_snap = self._snap
            self._hide_probe()
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
            if self._press_snap is not None:
                self.picked.emit(*self._press_snap)
                return
            hit = self._raycast(ev.position().x(), ev.position().y())
            if hit is not None:
                self.picked.emit(hit[0], hit[1])

        def mouseMoveEvent(self, ev):
            super().mouseMoveEvent(ev)
            if ev.buttons() == Qt.NoButton:
                self._hover_pos = ev.position()
                if not self._hover_timer.isActive():
                    self._hover_timer.start()

        def leaveEvent(self, ev):
            self._hover_pos = None
            self._hide_probe()
            super().leaveEvent(ev)

        def _hide_probe(self):
            self._marks["probe"].setVisible(False)
            self._readout.hide()
            self._set_snap(None)

        def _set_snap(self, snap):
            if snap != self._snap:
                self._snap = snap
                self.hoveredTriad.emit(snap)
                self._refresh_markers()

        def _nearest_triad(self, pos):
            """The best triad whose marker is within SNAP_PX of widget point *pos*."""
            if not len(self._tri_pts):
                return None
            w, h = self.width(), self.height()
            viewport = self.getViewport()
            mvp = self.projectionMatrix(viewport, viewport) * self.viewMatrix()
            best, best_d = None, float(self.SNAP_PX)
            for k, (gx, gy, gz) in enumerate(self._tri_pts):
                v = mvp.map(QVector4D(float(gx), float(gy), float(gz), 1.0))
                if v.w() <= 0:
                    continue
                sx = (v.x() / v.w() + 1.0) * 0.5 * w
                sy = (1.0 - v.y() / v.w()) * 0.5 * h
                d = math.hypot(sx - pos.x(), sy - pos.y())
                if d <= best_d:
                    best, best_d = (float(self._tri[0][k]), float(self._tri[1][k])), d
            return best

        def _probe(self):
            pos = self._hover_pos
            snap = self._nearest_triad(pos) if pos is not None else None
            self._set_snap(snap)
            hit = snap if snap is not None else (
                self._raycast(pos.x(), pos.y()) if pos is not None else None)
            if hit is None:
                self._hide_probe()
                return
            bx, cy = hit
            self._marks["probe"].setData(pos=self._point(bx, cy))
            self._marks["probe"].setVisible(True)
            d = float(self._z[int(np.argmin(np.abs(self._x - bx))),
                              int(np.argmin(np.abs(self._y - cy)))])
            self._readout.setText(
                f"<div style='font-family:Consolas;font-size:11px;line-height:140%'>"
                f"<b style='color:{VOICE_COLORS['B']}'>B {bx:.3f}</b> &nbsp;"
                f"<b style='color:{VOICE_COLORS['C']}'>C {cy:.3f}</b><br>"
                f"<span style='color:{SUBTEXT}'>{ratio_to_interval_name(bx)} \u00b7 "
                f"{ratio_to_interval_name(cy)}</span><br>"
                f"<span style='color:{diss_color(d).name()}'>diss {d:.3f}</span></div>")
            self._readout.adjustSize()
            x = pos.x() + 16
            y = pos.y() - self._readout.height() - 10
            if x + self._readout.width() > self.width() - 6:
                x = pos.x() - self._readout.width() - 16
            if y < 6:
                y = pos.y() + 18
            self._readout.move(int(x), int(y))
            self._readout.show()
            self._readout.raise_()

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

        # ── camera ─────────────────────────────────────────────────────────

        def resetView(self, animate=True):
            target = dict(SURFACE_CAMERA)
            center = _pan_center(**target)
            if not animate or not self.isVisible():
                self._cam.stop()
                self.setCameraPosition(**target)
                self.opts['center'] = pg.Vector(center)
                self.update()
                return
            start_az = float(self.opts['azimuth'])
            # Take the short way round.
            d_az = ((target['azimuth'] - start_az + 180.0) % 360.0) - 180.0
            self._cam_from = (float(self.opts['distance']), float(self.opts['elevation']),
                              start_az, QVector3D(self.opts['center']))
            self._cam_to = (target['distance'], target['elevation'], start_az + d_az, center)
            self._cam.stop()
            self._cam.start()

        def _cam_step(self, t):
            if self._cam_from is None:
                return
            t = float(t)
            (d0, e0, a0, c0), (d1, e1, a1, c1) = self._cam_from, self._cam_to
            self.opts['distance'] = d0 + (d1 - d0) * t
            self.opts['elevation'] = e0 + (e1 - e0) * t
            self.opts['azimuth'] = a0 + (a1 - a0) * t
            self.opts['center'] = pg.Vector(c0 + (c1 - c0) * t)
            self.update()

else:  # pragma: no cover - fallback when OpenGL is unavailable

    class SurfaceView(QLabel):
        picked = Signal(float, float)
        hoveredTriad = Signal(object)

        def __init__(self, parent=None):
            super().__init__("3D surface unavailable \u2014 PyOpenGL could not be "
                             "initialised on this machine.", parent)
            self.setAlignment(Qt.AlignCenter)
            self.setObjectName("emptyText")

        def setData(self, *args, **kwargs):
            pass

        def apply_palette(self):
            pass

        def resetView(self, animate=True):
            pass

        def setHighlight(self, bc):
            pass

        def setPlaying(self, bc):
            pass

        lit = False

        def setLit(self, on):
            pass


# ═══════════════════════════════════════════════════════════════════════════
#  Results list and inspector
# ═══════════════════════════════════════════════════════════════════════════

ROLE_PRIMARY = Qt.UserRole + 1
ROLE_SECONDARY = Qt.UserRole + 2
ROLE_VALUE = Qt.UserRole + 3
ROLE_ROW = Qt.UserRole + 4


class ResultDelegate(QStyledItemDelegate):
    """Paints one ranked result: ratio, interval names, dissonance value and bar."""

    ROW_H = 56

    def __init__(self, view):
        super().__init__(view)
        self.view = view

    def sizeHint(self, option, index):
        return QSize(1, self.ROW_H)

    def paint(self, p, option, index):
        p.save()
        p.setRenderHint(QPainter.Antialiasing, True)
        r = QRectF(option.rect).adjusted(3, 2, -3, -2)
        hovered = bool(option.state & QStyle.State_MouseOver)
        selected = bool(option.state & QStyle.State_Selected)
        payload = self.view.payload(index)
        linked = self.view.linked_key is not None and payload == self.view.linked_key
        if selected:
            p.fillPath(rounded(r, 10), QColor(PANEL))
        elif hovered or linked:
            p.fillPath(rounded(r, 10), QColor(SURFACE))
        if selected or linked:
            p.fillPath(rounded(QRectF(r.left() + 1, r.top() + 13, 3, r.height() - 26), 1.5),
                       QColor(BLUE))

        value = float(index.data(ROLE_VALUE))
        color = diss_color(value)
        vfont = mono_font(10, bold=True)
        fm = QFontMetricsF(vfont)
        right = r.right() - 12
        vtext = f"{value:.3f}"
        p.setFont(vfont)
        p.setPen(color)
        p.drawText(QRectF(right - 80, r.top() + 9, 80, fm.height()),
                   Qt.AlignRight | Qt.AlignTop, vtext)
        bar = QRectF(right - 62, r.top() + 14 + fm.height(), 62, 4)
        p.fillPath(rounded(bar, 2), with_alpha(BORDER, 0.35))
        p.fillPath(rounded(QRectF(bar.left(), bar.top(),
                                  max(4.0, bar.width() * float(np.clip(value, 0, 1))), 4), 2),
                   color)

        text_right = right - 74
        if hovered:
            draw_icon(p, "ph.play-fill", GREEN, QRectF(text_right - 16, r.top(), 16, r.height()), 13)
            text_right -= 24
        left = r.left() + 14
        width = max(10.0, text_right - left)
        p.setFont(mono_font(10, bold=True))
        p.setPen(QColor(TEXT))
        p.drawText(QRectF(left, r.top() + 8, width, 20), Qt.AlignLeft | Qt.AlignTop,
                   index.data(ROLE_PRIMARY))
        sfont = QFont(option.font)
        sfont.setPixelSize(11)
        p.setFont(sfont)
        p.setPen(QColor(MUTED))
        secondary = QFontMetricsF(sfont).elidedText(index.data(ROLE_SECONDARY), Qt.ElideRight,
                                                     width)
        p.drawText(QRectF(left, r.top() + 30, width, 18), Qt.AlignLeft | Qt.AlignTop, secondary)
        p.restore()


class ResultList(QListView):
    """Ranked results; emits the payload of clicked and hovered rows."""

    activatedRow = Signal(object)
    hoveredRow = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = QStandardItemModel(self)
        self.setModel(self._model)
        self.setItemDelegate(ResultDelegate(self))
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.viewport().setAttribute(Qt.WA_Hover, True)
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setUniformItemSizes(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setCursor(Qt.PointingHandCursor)
        self.linked_key = None
        self._payloads = []
        self._hover_key = None
        self._empty = "Nothing to show yet"
        self.clicked.connect(self._on_click)
        self._scroller = SmoothScroller(self)

    def payload(self, index):
        row = index.data(ROLE_ROW)
        return self._payloads[row] if row is not None and 0 <= row < len(self._payloads) else None

    def _on_click(self, index):
        payload = self.payload(index)
        if payload is not None:
            self.activatedRow.emit(payload)

    def _set_hover(self, key):
        if key != self._hover_key:
            self._hover_key = key
            self.hoveredRow.emit(key)

    def mouseMoveEvent(self, ev):
        super().mouseMoveEvent(ev)
        index = self.indexAt(ev.position().toPoint())
        self._set_hover(self.payload(index) if index.isValid() else None)

    def leaveEvent(self, ev):
        self._set_hover(None)
        super().leaveEvent(ev)

    def setRows(self, rows, empty_text="Nothing to show yet"):
        self._empty = empty_text
        if len(rows) == self._model.rowCount() and rows:
            # Same shape (typical for live updates): rewrite in place, no relayout.
            for row, (primary, secondary, value, payload) in enumerate(rows):
                item = self._model.item(row)
                item.setData(primary, ROLE_PRIMARY)
                item.setData(secondary, ROLE_SECONDARY)
                item.setData(float(value), ROLE_VALUE)
                self._payloads[row] = payload
            self.viewport().update()
            return
        selected = self.selectedPayload()
        self._model.clear()
        self._payloads = []
        for primary, secondary, value, payload in rows:
            item = QStandardItem()
            item.setEditable(False)
            item.setData(primary, ROLE_PRIMARY)
            item.setData(secondary, ROLE_SECONDARY)
            item.setData(float(value), ROLE_VALUE)
            item.setData(len(self._payloads), ROLE_ROW)
            self._payloads.append(payload)
            self._model.appendRow(item)
        if selected is not None:
            self.selectPayload(selected)
        self.viewport().update()

    def selectedPayload(self):
        sel = self.selectionModel().selectedIndexes() if self.selectionModel() else []
        return self.payload(sel[0]) if sel else None

    def selectPayload(self, key):
        for row, payload in enumerate(self._payloads):
            if payload == key:
                self.setCurrentIndex(self._model.index(row, 0))
                return True
        self.clearSelection()
        return False

    def setLinked(self, key):
        if key != self.linked_key:
            self.linked_key = key
            if key is not None:
                for row, payload in enumerate(self._payloads):
                    if payload == key:
                        self.scrollTo(self._model.index(row, 0))
                        break
            self.viewport().update()

    def paintEvent(self, ev):
        super().paintEvent(ev)
        if self._model.rowCount() == 0:
            p = QPainter(self.viewport())
            f = QFont(self.font())
            f.setPixelSize(12)
            p.setFont(f)
            p.setPen(QColor(MUTED))
            p.drawText(QRectF(self.viewport().rect()).adjusted(16, 28, -16, -16),
                       Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap, self._empty)


class DetailPanel(Card):
    """Inspector for the last sound: ratios, cents, just-intonation match, dissonance."""

    replay = Signal()

    KEYS = ("Cents", "12-TET", "Just ratio", "Dissonance", "Shared partials")

    def __init__(self, parent=None):
        super().__init__("inspector", parent)
        self.kind = QLabel()
        self.kind.setObjectName("detailKind")
        self.header.addWidget(self.kind, 0, Qt.AlignVCenter)
        self._last = None

        self.stack = QStackedWidget()
        empty = QLabel("Click a plot, a result or a \u25b6 button to inspect the sound here.")
        empty.setObjectName("emptyText")
        empty.setWordWrap(True)
        empty.setAlignment(Qt.AlignCenter)
        self.stack.addWidget(empty)

        page = QWidget()
        col = QVBoxLayout(page)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(8)
        self.big = QLabel()
        self.big.setFont(mono_font(15, bold=True))
        self.big.setTextInteractionFlags(Qt.TextSelectableByMouse)
        col.addWidget(self.big)
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(5)
        self.values = {}
        for row, key in enumerate(self.KEYS):
            k = QLabel(key)
            k.setObjectName("detailKey")
            grid.addWidget(k, row, 0, Qt.AlignLeft | Qt.AlignVCenter)
            v = QLabel()
            v.setFont(mono_font(9))
            v.setTextFormat(Qt.RichText)
            v.setTextInteractionFlags(Qt.TextSelectableByMouse)
            if key == "Dissonance":
                cell = QHBoxLayout()
                cell.setSpacing(8)
                cell.addWidget(v, 0)
                self.bar = DissBar(70, 5)
                cell.addWidget(self.bar, 0, Qt.AlignVCenter)
                cell.addStretch(1)
                grid.addLayout(cell, row, 1)
            else:
                grid.addWidget(v, row, 1)
            self.values[key] = v
        grid.setColumnStretch(1, 1)
        col.addLayout(grid)
        self.play = Button("Play again", "ph.play-fill", kind="play", height=30,
                           tooltip="Replay this sound  (Space)")
        self.play.clicked.connect(self.replay.emit)
        col.addWidget(self.play, 0, Qt.AlignLeft)
        self.stack.addWidget(page)
        self.body.addWidget(self.stack)
        self.setMinimumHeight(240)

    def _fill(self, kind, big, cents, tet, just, diss, shared):
        self.kind.setText(kind)
        self.big.setText(big)
        self.values["Cents"].setText(cents)
        self.values["12-TET"].setText(tet)
        self.values["Just ratio"].setText(just)
        if diss is None:
            self.values["Dissonance"].setText(f"<span style='color:{MUTED}'>\u2014</span>")
            self.bar.setValue(0.0)
        else:
            self.values["Dissonance"].setText(
                f"<span style='color:{diss_color(diss).name()}'>{diss:.3f}</span>")
            self.bar.setValue(diss)
        self.values["Shared partials"].setText(
            f"<span style='color:{MUTED}'>\u2014</span>" if shared is None else str(shared))
        self.stack.setCurrentIndex(1)

    @staticmethod
    def _just(ratio):
        jr = just_ratio(ratio)
        if jr is None or abs(jr[2]) > 12:
            return f"<span style='color:{MUTED}'>none within 12\u00a2</span>"
        return f"{jr[0]}/{jr[1]} &nbsp;<span style='color:{MUTED}'>{_cents_text(jr[2])}</span>"

    def show_interval(self, ratio, diss, shared):
        self._last = ("show_interval", ratio, diss, shared)
        self._fill("INTERVAL", f"1 : {ratio:.4f}", f"{ratio_to_cents(ratio):.1f}\u00a2",
                   ratio_to_interval_name(ratio), self._just(ratio), diss, shared)

    def show_triad(self, rb, rc, diss, shared):
        self._last = ("show_triad", rb, rc, diss, shared)
        prop = triad_proportion(rb, rc)
        if prop:
            just = (f"{prop[0]} : {prop[1]} : {prop[2]} &nbsp;<span style='color:{MUTED}'>"
                    f"{_cents_text(prop[3])}, {_cents_text(prop[4])}</span>")
        else:
            just = f"<span style='color:{MUTED}'>no small whole-number chord</span>"
        self._fill("TRIAD", f"1 : {rb:.3f} : {rc:.3f}",
                   f"{ratio_to_cents(rb):.0f}\u00a2 \u00b7 {ratio_to_cents(rc):.0f}\u00a2",
                   f"{ratio_to_interval_name(rb)} \u00b7 {ratio_to_interval_name(rc)}",
                   just, diss, shared)

    def show_note(self, ratio):
        self._last = ("show_note", ratio)
        self._fill("NOTE", f"A \u00d7 {ratio:.4f}", f"{ratio_to_cents(ratio):.1f}\u00a2",
                   ratio_to_interval_name(ratio), self._just(ratio), None, None)

    def show_voice(self, letter, source):
        self._last = ("show_voice", letter, source)
        self._fill(f"VOICE {letter}", f"{letter} alone", "0\u00a2", "unison",
                   f"<span style='color:{MUTED}'>{source}</span>", None, None)

    def refresh(self):
        """Re-render with the current palette."""
        if self._last is not None:
            getattr(self, self._last[0])(*self._last[1:])


# ═══════════════════════════════════════════════════════════════════════════
#  Background computation
# ═══════════════════════════════════════════════════════════════════════════

class ComputeWorker(QThread):
    """Long-lived worker: always computes the newest request, in three stages.

    A preempting request makes the running job stop at its next chunk boundary,
    so typed edits never queue up behind stale work. Requests made while dragging
    don't preempt: the running job finishes and the newest request follows, so
    results keep arriving at the worker's full rate.
    """

    ready = Signal()
    stage = Signal(int, str, object)
    progress = Signal(int, float)
    failed = Signal(int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cv = threading.Condition()
        self._job = None
        self._gen = 0
        self._preempt = True
        self._quit = False

    def submit(self, params, preempt=True):
        with self._cv:
            self._gen += 1
            self._preempt = preempt
            self._job = (self._gen, params)
            self._cv.notify()
            return self._gen

    def shutdown(self):
        with self._cv:
            self._quit = True
            self._cv.notify()
        self.wait(5000)

    def _check(self, gen):
        if self._quit or (self._gen != gen and self._preempt):
            raise Cancelled()

    def run(self):
        global HAS_NUMBA
        try:
            warm_up_kernels()
        except Exception:  # pragma: no cover - fall back to numpy if compilation fails
            HAS_NUMBA = False
        self.ready.emit()
        while True:
            with self._cv:
                while self._job is None and not self._quit:
                    self._cv.wait()
                if self._quit:
                    return
                gen, params = self._job
                self._job = None
            try:
                self._run_job(gen, params)
            except Cancelled:
                continue
            except Exception as exc:  # surfaced in the status bar
                self.failed.emit(gen, str(exc))

    def _run_job(self, gen, p):
        def check():
            self._check(gen)

        spec_a, spec_b, spec_c = p["spectra"]
        ref, mx = p["ref"], p["max_interval"]
        curve = compute_curve(spec_a, spec_b, ref, mx + CURVE_MARGIN)
        check()
        self.stage.emit(gen, "curve", dict(curve, max_interval=mx))
        self.progress.emit(gen, 0.05)

        x3, y3, z3 = compute_dissonance_3d(
            spec_a, spec_b, spec_c, ref, mx, p["step_3d"],
            progress=lambda f: self.progress.emit(gen, 0.05 + 0.3 * f), check=check)
        check()
        self.stage.emit(gen, "surface", dict(x_3d=x3, y_3d=y3, z_3d=z3))

        tx, ty, tz = find_minima_3d(
            x3, y3, z3, (spec_a, spec_b, spec_c), ref, top_n=None,
            progress=lambda f: self.progress.emit(gen, 0.35 + 0.65 * f), check=check)
        check()
        self.stage.emit(gen, "triads", dict(tri_x=tx, tri_y=ty, tri_z=tz))
        if p.get("contours", True):
            contours = contour_paths(z3)
            check()
            self.stage.emit(gen, "contours", dict(contours=contours))


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
        gen.setTitle("Dissonance Studio figure")
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
    font="Times New Roman", size=9.0, axis=0.8, cmap="viridis", shading=False,
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
        form.addRow("Surface", self._combo("shading", [("Flat", False), ("Shaded", True)]))
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


def reveal_in_folder(path):
    """Open the folder holding *path* (selecting the file on Windows)."""
    path = Path(path).resolve()
    if sys.platform == "win32":
        subprocess.Popen(["explorer", "/select,", str(path)])
    else:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))


# ═══════════════════════════════════════════════════════════════════════════
#  Main window
# ═══════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):

    VIEW_NAMES = ("curve", "surface", "heatmap")
    VIEW_TITLES = (
        ("Dissonance curve", "voice A fixed \u00b7 voice B swept"),
        ("Dissonance surface", "voice A fixed \u00b7 voices B and C swept"),
        ("Triad map", "voice A fixed \u00b7 B across \u00b7 C up"),
    )
    VIEW_HINTS = (
        "click to hear  \u00b7  drag to glide  \u00b7  shift-drag to pan  \u00b7  scroll to zoom",
        "drag to rotate  \u00b7  scroll to zoom  \u00b7  click the surface to hear the triad",
        "dark = consonant  \u00b7  click to hear  \u00b7  drag to glide  \u00b7  shift-drag to pan",
    )

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dissonance Studio")
        self.resize(1680, 980)
        self.setMinimumSize(1180, 720)

        self.ref_freq = 261.63
        self.max_interval = 2.3
        self.step_3d = 0.005

        self.data = None            # last complete analysis
        self.curve = None           # newest curve stage; can be ahead of self.data
        self._partial = {}
        self._gen = 0               # newest request
        self._stage_gen = 0         # job whose stages are arriving
        self._shown_gen = 0         # newest job fully on screen
        self._job_reset = False
        self._busy = False
        self._ready = False
        self._view3d = None
        self._contours = None       # heatmap contours belonging to the data on screen
        self._dirty = set()         # 3D views that are hidden and behind on data
        self._dragging = False
        self._last_sound = None
        self._glide = None
        self._shown_overtones = []
        self._fig_state = {}
        self._surface_renderer = None
        self._dot_role = "BLUE"
        self._status_kind = "info"
        self._muted_volume = None

        self.engine = AudioEngine(self)
        self.worker = ComputeWorker(self)
        self._debounce = self._timer(200, self._submit)
        self._slow = self._timer(160, self._show_busy)
        self._surface_wait = self._timer(120, self._show_surface_early)
        self._glide_tick = self._timer(33, self._glide_ui)
        self._warm_note = self._timer(1500, self._note_compiling)

        self._build_ui()

        self.engine.activityChanged.connect(self._on_audio_activity)
        self.engine.level.connect(self.meter.setLevel)
        self.engine.failed.connect(lambda m: self._notify(m, "error"))
        self.worker.ready.connect(self._on_ready)
        self.worker.stage.connect(self._on_stage)
        self.worker.progress.connect(self._on_progress)
        self.worker.failed.connect(self._on_failed)

        self._status("Preparing the dissonance engine\u2026", "YELLOW", "busy")
        self._warm_note.start()
        self.worker.start()
        self.analyse()

    def _timer(self, ms, slot):
        t = QTimer(self)
        t.setSingleShot(True)
        t.setInterval(ms)
        t.timeout.connect(slot)
        return t

    # ── construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(12, 6, 12, 8)
        outer.setSpacing(10)
        outer.addWidget(self._build_topbar())

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self._build_sidebar(), 0)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(12)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_plot_area())
        splitter.addWidget(self._build_results())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([1060, 360])
        body.addWidget(splitter, 1)
        outer.addLayout(body, 1)
        outer.addLayout(self._build_statusbar())

        self.toast = Toast(root)

        QShortcut(QKeySequence("Ctrl+Return"), self, self.analyse)
        QShortcut(QKeySequence("Ctrl+Enter"), self, self.analyse)
        QShortcut(QKeySequence(Qt.Key_Escape), self, self.engine.stop)
        QShortcut(QKeySequence(Qt.Key_Space), self, self._replay)
        self._switch_tab(0, animate=False)

    def _build_topbar(self):
        bar = QWidget()
        bar.setFixedHeight(48)
        grid = QGridLayout(bar)
        grid.setContentsMargins(2, 0, 0, 0)
        grid.setHorizontalSpacing(12)

        left = QHBoxLayout()
        left.setSpacing(10)
        self.logo = QLabel()
        self.logo.setFixedSize(30, 30)
        self.logo.setAlignment(Qt.AlignCenter)
        left.addWidget(self.logo)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        title = QLabel("Dissonance Studio")
        title.setObjectName("appTitle")
        sub = QLabel("Plomp\u2013Levelt \u00b7 Sethares sensory dissonance")
        sub.setObjectName("appSub")
        titles.addWidget(title)
        titles.addWidget(sub)
        left.addLayout(titles)
        left.addStretch(1)

        self.tabs = Segmented(["Curve", "Surface", "Heatmap"],
                              icons=["ph.chart-line", "ph.cube", "ph.grid-four"],
                              height=36, stretch=False)
        self.tabs.changed.connect(self._switch_tab)

        right = QHBoxLayout()
        right.setSpacing(8)
        right.addStretch(1)
        self.live = Toggle("Live", checked=True,
                           tooltip="Re-analyse automatically while you edit")
        self.live.toggled.connect(self._on_live_toggled)
        right.addWidget(self.live)
        self.btn_run = Button("Analyse", "ph.lightning-fill", kind="primary",
                              tooltip="Recompute everything and reset the views  (Ctrl+Enter)")
        self.btn_run.clicked.connect(self.analyse)
        right.addWidget(self.btn_run)
        right.addSpacing(6)
        right.addWidget(self._vsep())
        right.addSpacing(2)
        self.btn_mute = Button(icon="ph.speaker-high-fill", kind="ghost", tooltip="Mute")
        self.btn_mute.clicked.connect(self._toggle_mute)
        right.addWidget(self.btn_mute)
        self.volume = QSlider(Qt.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(100)
        self.volume.setFixedWidth(96)
        self.volume.setToolTip("Volume")
        self.volume.valueChanged.connect(self._on_volume)
        right.addWidget(self.volume)
        self.meter = LevelMeter()
        right.addWidget(self.meter)
        self.btn_stop = Button(icon="ph.stop-fill", kind="danger", tooltip="Stop audio  (Esc)")
        self.btn_stop.clicked.connect(self.engine.stop)
        right.addWidget(self.btn_stop)
        right.addSpacing(2)
        right.addWidget(self._vsep())
        right.addSpacing(2)
        self.btn_theme = Button(icon="ph.sun", kind="ghost", tooltip="Switch to the light theme")
        self.btn_theme.clicked.connect(self._toggle_theme)
        right.addWidget(self.btn_theme)

        grid.addLayout(left, 0, 0, Qt.AlignLeft | Qt.AlignVCenter)
        grid.addWidget(self.tabs, 0, 1, Qt.AlignCenter)
        grid.addLayout(right, 0, 2, Qt.AlignRight | Qt.AlignVCenter)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(2, 1)
        self._on_volume(self.volume.value())
        self._refresh_chrome_icons()
        return bar

    @staticmethod
    def _vsep():
        line = QFrame()
        line.setObjectName("vsep")
        line.setFixedSize(1, 22)
        return line

    def _build_sidebar(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(356)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.viewport().setAutoFillBackground(False)

        inner = QWidget()
        inner.setObjectName("plain")
        col = QVBoxLayout(inner)
        col.setContentsMargins(0, 0, 10, 0)
        col.setSpacing(10)

        voices = Card("voices")
        self.voice_seg = Segmented(["A", "B", "C"])
        voices.body.addWidget(self.voice_seg)
        self.voice_stack = QStackedWidget()
        self.card_a = SpectrumCard("A")
        self.card_b = SpectrumCard("B", copy_sources=[("A", self.card_a)])
        self.card_c = SpectrumCard("C", copy_sources=[("A", self.card_a),
                                                      ("B", self.card_b)])
        for card in self._cards():
            card.playRequested.connect(self._play_card)
            card.sourceChanged.connect(self._label_voice_tabs)
            card.spectrumChanged.connect(self._on_inputs_changed)
            card.dragging.connect(self._on_dragging)
            self.voice_stack.addWidget(card)
        self.voice_seg.changed.connect(lambda i: crossfade(
            self.voice_stack, lambda: self.voice_stack.setCurrentIndex(i), 170))
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
        for i, field in enumerate((self.f_ref, self.f_max, self.f_step)):
            grid.addWidget(field, 0, i)
            field.edit.textChanged.connect(self._on_settings_changed)
        settings.body.addLayout(grid)
        self.settings_error = QLabel()
        self.settings_error.setObjectName("errorText")
        self.settings_error.setWordWrap(True)
        self.settings_error.hide()
        settings.body.addWidget(self.settings_error)
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
            btn = Button(icon="ph.play-fill", kind="play", tooltip=tip)
            btn.clicked.connect(slot)
            row.addWidget(btn, 0)
            manual.body.addLayout(row)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.f_duration = Field("Length (s)", "1.5", inline=True)
        row.addWidget(self.f_duration, 1)
        btn_wav = Button("Save WAV\u2026", "ph.download-simple",
                         tooltip="Write the last sound you played to a 24-bit WAV "
                                 "file of this length")
        btn_wav.clicked.connect(self._export_wav)
        row.addWidget(btn_wav, 0)
        manual.body.addLayout(row)
        col.addWidget(manual)

        col.addStretch(1)
        scroll.setWidget(inner)
        self._sidebar_scroller = SmoothScroller(scroll)
        return scroll

    def _build_plot_area(self):
        wrap = QWidget()
        col = QVBoxLayout(wrap)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(10)

        card = QFrame()
        card.setObjectName("card")
        inner = QVBoxLayout(card)
        inner.setContentsMargins(14, 10, 12, 12)
        inner.setSpacing(6)
        head = QHBoxLayout()
        head.setSpacing(8)
        self.view_title = QLabel()
        self.view_title.setObjectName("viewTitle")
        self.view_sub = QLabel()
        self.view_sub.setObjectName("hint")
        head.addWidget(self.view_title)
        head.addWidget(self.view_sub)
        head.addStretch(1)
        self.shade_seg = Segmented(["Flat", "Shaded"], height=32, stretch=False)
        self.shade_seg.setToolTip("Surface shading: flat colour or shaded")
        self.shade_seg.changed.connect(lambda i: self.surface_view.setLit(i == 1))
        head.addWidget(self.shade_seg)
        self.btn_export = Button("Save figure\u2026", "ph.image",
                                 tooltip="Open the figure editor for a publication-ready export")
        self.btn_export.clicked.connect(self._export_figure)
        head.addWidget(self.btn_export)
        self.btn_reset_view = Button("Reset view", "ph.arrow-counter-clockwise",
                                     tooltip="Reset the view")
        self.btn_reset_view.clicked.connect(self._reset_view)
        head.addWidget(self.btn_reset_view)
        inner.addLayout(head)
        self.activity = ActivityBar()
        inner.addWidget(self.activity)

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
        inner.addWidget(self.stack, 1)
        col.addWidget(card, 1)

        ot = Card("overtones", margins=(14, 9, 12, 6))
        self.overtone_legend = QLabel()
        self.overtone_legend.setObjectName("hint")
        ot.header.insertWidget(1, self.overtone_legend)
        self.overtone_info = QLabel()
        self.overtone_info.setObjectName("hint")
        ot.header.addWidget(self.overtone_info)
        self.overtone_view = OvertoneView()
        self.overtone_view.setFixedHeight(92)
        ot.body.addWidget(self.overtone_view)
        col.addWidget(ot, 0)

        cv, hv = self.curve_view, self.heatmap_view
        cv.picked.connect(self._play_interval_ratio)
        cv.scrubStarted.connect(lambda r: self._glide_begin("interval", r))
        cv.scrubMoved.connect(lambda r: self._glide_move(r))
        cv.scrubEnded.connect(self._glide_end)
        hv.picked.connect(self._play_triad_ratio)
        hv.scrubStarted.connect(lambda b, c: self._glide_begin("triad", b, c))
        hv.scrubMoved.connect(lambda b, c: self._glide_move(b, c))
        hv.scrubEnded.connect(self._glide_end)
        self.surface_view.picked.connect(self._play_triad_ratio)
        return wrap

    def _build_results(self):
        wrap = QWidget()
        col = QVBoxLayout(wrap)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(10)

        self.res_tabs = Segmented(["Intervals", "Triads"], height=36)
        col.addWidget(self.res_tabs, 0)

        card = QFrame()
        card.setObjectName("card")
        inner = QVBoxLayout(card)
        inner.setContentsMargins(8, 10, 8, 8)
        inner.setSpacing(8)
        self.res_stack = QStackedWidget()

        intervals_page = QWidget()
        intervals_page.setObjectName("plain")
        ip = QVBoxLayout(intervals_page)
        ip.setContentsMargins(0, 0, 0, 0)
        ip.setSpacing(8)
        self.sort_seg = Segmented(["Most consonant", "By ratio"], height=30)
        self.sort_seg.changed.connect(lambda _: self._fill_intervals())
        sort_row = QHBoxLayout()
        sort_row.setContentsMargins(4, 0, 4, 0)
        sort_row.addWidget(self.sort_seg, 1)
        ip.addLayout(sort_row)
        self.list_intervals = ResultList()
        self.list_intervals.activatedRow.connect(self._play_interval_ratio)
        self.list_intervals.hoveredRow.connect(self.curve_view.setHighlight)
        ip.addWidget(self.list_intervals, 1)

        triads_page = QWidget()
        triads_page.setObjectName("plain")
        tp = QVBoxLayout(triads_page)
        tp.setContentsMargins(0, 0, 0, 0)
        tp.setSpacing(8)
        self.triad_seg = Segmented(["Within octave", "All voicings"], height=30)
        self.triad_seg.changed.connect(lambda _: self._refresh_triads())
        triad_row = QHBoxLayout()
        triad_row.setContentsMargins(4, 0, 4, 0)
        triad_row.addWidget(self.triad_seg, 1)
        tp.addLayout(triad_row)
        self.list_triads = ResultList()
        self.list_triads.activatedRow.connect(lambda p: self._play_triad_ratio(p[0], p[1]))
        self.list_triads.hoveredRow.connect(self._highlight_triad)
        tp.addWidget(self.list_triads, 1)

        self.res_stack.addWidget(intervals_page)
        self.res_stack.addWidget(triads_page)
        self.res_tabs.changed.connect(lambda i: crossfade(
            self.res_stack, lambda: self.res_stack.setCurrentIndex(i), 180))
        inner.addWidget(self.res_stack)
        col.addWidget(card, 1)

        self.detail = DetailPanel()
        self.detail.replay.connect(self._replay)
        col.addWidget(self.detail, 0)

        self.curve_view.hoveredMinimum.connect(self.list_intervals.setLinked)
        self.heatmap_view.hoveredTriad.connect(self.list_triads.setLinked)
        self.surface_view.hoveredTriad.connect(self.list_triads.setLinked)
        wrap.setMinimumWidth(280)
        return wrap

    def _build_statusbar(self):
        row = QHBoxLayout()
        row.setContentsMargins(4, 0, 4, 0)
        row.setSpacing(10)
        self.status_dot = QLabel()
        self.status_dot.setFixedSize(8, 8)
        self.status_label = QLabel()
        self.status_label.setObjectName("statusText")
        self.view_hint = QLabel()
        self.view_hint.setObjectName("hint")
        row.addWidget(self.status_dot, 0, Qt.AlignVCenter)
        row.addWidget(self.status_label, 1)
        row.addWidget(self.view_hint, 0)
        return row

    # ── helpers ────────────────────────────────────────────────────────────

    def _cards(self):
        return (self.card_a, self.card_b, self.card_c)

    def _set_dot(self, role):
        self._dot_role = role
        self.status_dot.setStyleSheet(f"background: {globals()[role]}; border-radius: 4px;")

    def _status(self, text, role="BLUE", kind="info"):
        self.status_label.setText(text)
        self._set_dot(role)
        self._status_kind = kind

    def _notify(self, text, kind="ok", action=None):
        self.toast.show_message(text, kind, action)

    def _refresh_chrome_icons(self):
        dpr = max(self.devicePixelRatioF(), QApplication.primaryScreen().devicePixelRatio())
        pm = icon_pixmap("ph.waves", BLUE, 26, dpr)
        if pm is not None:
            self.logo.setPixmap(pm)
        dark = is_dark()
        self.btn_theme.setIconName("ph.sun" if dark else "ph.moon")
        self.btn_theme.setToolTip("Switch to the light theme" if dark else
                                  "Switch to the dark theme")

    def _label_voice_tabs(self):
        if not hasattr(self, "voice_seg"):
            return
        for i, card in enumerate(self._cards()):
            text = card.letter
            if card.is_linked():
                text += f" = {card.source_name()}"
            self.voice_seg.setLabel(i, text)
            self.voice_seg.setColor(i, card.color)

    def _switch_tab(self, index, animate=True):
        self._ensure_view(index)
        if self.stack.currentIndex() != index:
            if animate:
                crossfade(self.stack, lambda: self.stack.setCurrentIndex(index), 200)
            else:
                self.stack.setCurrentIndex(index)
        if self.tabs.currentIndex() != index:
            self.tabs.setCurrentIndex(index, animate=animate)
        self.shade_seg.setVisible(index == 1 and HAS_GL)
        title, sub = self.VIEW_TITLES[index]
        self.view_title.setText(title)
        self.view_sub.setText(sub)
        self.view_hint.setText(self.VIEW_HINTS[index])
        self.curve_view.hideCrosshair()
        self.heatmap_view.hideCrosshair()

    def _view_xmax(self):
        """Upper ratio currently shown on the curve."""
        return self.curve["max_interval"] if self.curve is not None else self.max_interval

    def _reset_view(self):
        i = self.stack.currentIndex()
        if i == 0 and self.curve is not None:
            self.curve_view.setViewRatioRange(self._view_xmax(), animate=True)
        elif i == 1:
            self.surface_view.resetView(animate=True)
        elif i == 2:
            self.heatmap_view.resetRange(animate=True)

    def _settings_values(self):
        """Parse the analysis fields, flagging each bad one; returns (values, problems)."""
        values, problems = {}, []
        for key, field, test, message in (
                ("ref", self.f_ref, lambda v: v > 0, "Fundamental must be above 0 Hz"),
                ("max", self.f_max, lambda v: v > 1.0, "Max ratio must be above 1"),
                ("step", self.f_step, lambda v: v > 0, "3D step must be above 0")):
            try:
                v = float(field.text())
                ok = math.isfinite(v) and test(v)
            except ValueError:
                ok, message = False, "Settings must be numbers"
            field.setInvalid(not ok, message)
            if ok:
                values[key] = v
            else:
                problems.append(message)
        if not problems:
            n = math.ceil((values["max"] - 1.0) / values["step"] - 1e-9)
            if n > MAX_GRID:
                problems.append(f"3D grid would be {n:,} \u00d7 {n:,} points (limit {MAX_GRID:,}) "
                                f"\u2014 raise 3D STEP or lower MAX RATIO")
            elif n < 3:
                problems.append("MAX RATIO is too close to 1 for this 3D STEP")
            for field in (self.f_max, self.f_step):
                field.setInvalid(bool(problems), problems[0] if problems else None)
        self.settings_error.setText(problems[0] if problems else "")
        self.settings_error.setVisible(bool(problems))
        return values, problems

    def _read_settings(self):
        values, problems = self._settings_values()
        if problems:
            self._status(problems[0], "RED", "input-error")
            return False
        self.ref_freq, self.max_interval, self.step_3d = values["ref"], values["max"], values["step"]
        return True

    # ── computation ────────────────────────────────────────────────────────

    def analyse(self):
        """Recompute everything now and reset the views (button / Ctrl+Enter)."""
        self._debounce.stop()
        self._submit(reset=True)

    def _on_inputs_changed(self):
        if self.live.isChecked():
            self._debounce.start(0 if self._dragging else 200)

    def _on_settings_changed(self):
        _, problems = self._settings_values()
        if not problems and self.live.isChecked():
            self._debounce.start(350)

    def _on_live_toggled(self, on):
        if on:
            self._debounce.start(0)

    def _on_dragging(self, active):
        self._dragging = active
        # Drag-time jobs skip the heatmap contours; one more pass draws them.
        if not active and self.live.isChecked():
            self._debounce.start(0)

    def _submit(self, reset=False):
        if not self._read_settings():
            return
        spectra = []
        for card in self._cards():
            spec, err = card.parsed()
            if spec is None:
                self._status(f"Voice {card.letter}: {err}", "RED", "input-error")
                return
            spectra.append(spec)
        self._job_reset = (self._job_reset or reset or self.curve is None
                           or self.curve["max_interval"] != self.max_interval)
        self._gen = self.worker.submit(dict(spectra=tuple(spectra), ref=self.ref_freq,
                                            max_interval=self.max_interval,
                                            step_3d=self.step_3d,
                                            contours=not self._dragging),
                                       preempt=not self._dragging)
        self._busy = True
        if not self._slow.isActive():
            self._slow.start()

    def _show_busy(self):
        if self._busy and self._ready and not self._dragging:
            self.activity.start()
            self._status("Computing dissonance space\u2026", "YELLOW", "busy")

    def _note_compiling(self):
        if not self._ready:
            self.activity.start()
            self._status("Compiling the dissonance engine \u2014 first launch only, "
                         "cached afterwards\u2026", "YELLOW", "busy")

    def _on_ready(self):
        self._ready = True
        self._warm_note.stop()

    def _on_progress(self, gen, fraction):
        if gen == self._gen:
            self.activity.setProgress(fraction)

    def _on_stage(self, gen, name, payload):
        if name == "contours":
            if gen == self._shown_gen:
                self._contours = payload["contours"]
                if self.stack.currentIndex() == 2 and 2 not in self._dirty:
                    self.heatmap_view.setContours(self._contours)
            return
        if gen <= self._shown_gen:
            return
        if gen != self._stage_gen:
            self._stage_gen = gen
            self._partial = {}
        self._partial.update(payload)
        if name == "curve":
            first = self.curve is None
            self.curve = payload
            self.curve_view.setData(payload["ratios_2d"], payload["diss_2d"],
                                    payload["minima_x"], payload["minima_y"],
                                    payload["max_interval"], reset_view=self._job_reset,
                                    animate=not first and not self._dragging)
            self._fill_intervals()
        elif name == "surface":
            self._surface_wait.start()
        else:
            self._surface_wait.stop()
            self._shown_gen = gen
            self.data = dict(self._partial)
            self._contours = None
            if self._surface_renderer is not None:
                self._surface_renderer.close()
                self._surface_renderer = None
            if gen == self._gen:
                self._finish_job()
            self._refresh_triads(animate=not self._dragging)

    def _finish_job(self):
        self._busy = False
        self._job_reset = False
        self._slow.stop()
        self.activity.finish()
        if self._status_kind in ("busy", "input-error", "analysis-error"):
            self._status("Ready \u2014 click any plot or list entry to hear it", "GREEN")

    def _show_surface_early(self):
        """Slow triad search (no Numba): show the surface before its markers arrive."""
        p = self._partial
        if "x_3d" in p and "tri_x" not in p:
            empty = np.array([])
            self._view3d = (p["x_3d"], p["y_3d"], p["z_3d"], empty, empty, empty)
            self._contours = None
            self._dirty = {1, 2}
            self._ensure_view(self.stack.currentIndex())
            self.surface_bar.setLevels(float(p["z_3d"].min()), float(p["z_3d"].max()))

    def _on_failed(self, gen, message):
        if gen != self._gen:
            return
        self._busy = False
        self._slow.stop()
        self.activity.finish()
        self._status(f"Computation failed: {message}", "RED", "analysis-error")
        self._notify(f"Computation failed: {message}", "error")

    def _fill_intervals(self):
        if self.curve is None:
            return
        xmax = self._view_xmax()
        pairs = [(x, y) for x, y in zip(self.curve["minima_x"], self.curve["minima_y"])
                 if x <= xmax]
        if self.sort_seg.currentIndex() == 0:
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
        if self.triad_seg.currentIndex() == 0:
            # 20-cent guard: anything nearer than this just reads as an octave.
            octave = 2.0 * 2 ** (-20 / 1200)
            keep = (bx < octave) & (cy < octave)
            bx, cy, dz = bx[keep], cy[keep], dz[keep]
        order = np.argsort(dz)[:limit]
        return bx[order], cy[order], dz[order]

    def _refresh_triads(self, animate=False):
        if self.data is None:
            return
        tx, ty, tz = self._visible_triads()
        d = self.data
        self._view3d = (d["x_3d"], d["y_3d"], d["z_3d"], tx, ty, tz)
        self._dirty = {1, 2}
        self._ensure_view(self.stack.currentIndex(), animate=animate)
        self.surface_bar.setLevels(float(d["z_3d"].min()), float(d["z_3d"].max()))
        rows = [(f"1.000 : {bx:.3f} : {cy:.3f}",
                 f"{ratio_to_interval_name(bx)}  \u00b7  {ratio_to_interval_name(cy)}",
                 float(dz), (float(bx), float(cy)))
                for bx, cy, dz in zip(tx, ty, tz)]
        self.list_triads.setRows(rows, "No consonant triads found")

    def _ensure_view(self, index, animate=False):
        """Bring a 3D view up to date; hidden ones wait until they're shown."""
        if index not in self._dirty or self._view3d is None:
            return
        self._dirty.discard(index)
        x, y, z, tx, ty, tz = self._view3d
        if index == 1:
            self.surface_view.setData(x, y, z, tx, ty, tz, animate=animate,
                                      coarse=self._dragging)
        elif index == 2:
            self.heatmap_view.setData(x, y, z, tx, ty, tz)
            self.heatmap_view.setContours(self._contours)

    def _highlight_triad(self, bc):
        self.heatmap_view.setHighlight(bc)
        self.surface_view.setHighlight(bc)

    # ── playback ───────────────────────────────────────────────────────────

    def _sound(self, voices, label, letters="ABC", marker=None):
        """Play *voices*, show their overtones, and remember them for WAV export."""
        self._last_sound = (voices, label, letters, marker)
        self._shown_overtones = [(letters[i], spec, ratio)
                                 for i, (spec, ratio) in enumerate(voices)]
        matches = self._show_overtones(animate=True)
        if self.engine.play(synthesize_multi(voices, self.ref_freq)):
            self._set_sounding(marker)
        return matches

    def _show_overtones(self, animate=False):
        shown = self._shown_overtones
        if not shown:
            return 0
        matches = self.overtone_view.setVoices(shown, self.ref_freq, animate=animate)
        dots = "  ".join(f"<span style='color:{VOICE_COLORS[l]}'>\u25cf</span> "
                         f"{l} {ratio:.3f}" for l, _, ratio in shown)
        key = (f"  \u00b7  <span style='color:{TEXT}'>\u25ef</span> shared"
               if len(shown) > 1 else "")
        self.overtone_legend.setText(dots + key)
        self.overtone_info.setText(
            f"{matches} shared partials" if len(shown) > 1 else "single voice")
        return matches

    def _set_sounding(self, marker):
        kind = marker[0] if marker else None
        self.curve_view.setPlaying(marker[1] if kind == "interval" else None)
        bc = (marker[1], marker[2]) if kind == "triad" else None
        self.heatmap_view.setPlaying(bc)
        self.surface_view.setPlaying(bc)

    def _on_audio_activity(self, active):
        if not active and self._glide is None:
            self._set_sounding(None)

    def _play_card(self, card):
        if not self._read_settings():
            return
        spec, err = card.parsed()
        if spec is None:
            self._status(f"Voice {card.letter}: {err}", "RED", "input-error")
            return
        self._status(f"Playing voice {card.letter} alone", "GREEN")
        self._sound([(spec, 1.0)], f"voice_{card.letter}", letters=card.letter)
        self.detail.show_voice(card.letter, card.combo.currentText())

    def _diss_at_ratio(self, ratio):
        if self.curve is None:
            return None
        r = self.curve["ratios_2d"]
        i = int(np.argmin(np.abs(r - ratio)))
        return float(self.curve["diss_2d"][i])

    def _diss_at_triad(self, rb, rc):
        if self.data is None:
            return None
        bi = int(np.argmin(np.abs(self.data["x_3d"] - rb)))
        ci = int(np.argmin(np.abs(self.data["y_3d"] - rc)))
        return float(self.data["z_3d"][bi, ci])

    def _play_interval_ratio(self, ratio):
        if not self._read_settings():
            return
        spec_a, spec_b = self.card_a.get_spectrum(), self.card_b.get_spectrum()
        if spec_a is None or spec_b is None:
            self._status("Voice A or B has invalid input", "RED", "input-error")
            return
        d = self._diss_at_ratio(ratio)
        tail = f" \u00b7 diss {d:.3f}" if d is not None else ""
        self._status(f"A 1.000 + B {ratio:.4f}  ({ratio_to_interval_name(ratio)}){tail}",
                     "GREEN")
        matches = self._sound([(spec_a, 1.0), (spec_b, ratio)], f"interval_{ratio:.4f}",
                              marker=("interval", ratio))
        self.detail.show_interval(ratio, d, matches)
        self.list_intervals.selectPayload(float(ratio))

    def _play_triad_ratio(self, rb, rc):
        if not self._read_settings():
            return
        spec_a, spec_b, spec_c = (card.get_spectrum() for card in self._cards())
        if spec_a is None or spec_b is None or spec_c is None:
            self._status("One of the voices has invalid input", "RED", "input-error")
            return
        d = self._diss_at_triad(rb, rc)
        tail = f" \u00b7 diss {d:.3f}" if d is not None else ""
        self._status(f"A 1.000 + B {rb:.3f} + C {rc:.3f}{tail}", "GREEN")
        matches = self._sound([(spec_a, 1.0), (spec_b, rb), (spec_c, rc)],
                              f"triad_{rb:.3f}_{rc:.3f}", marker=("triad", rb, rc))
        self.detail.show_triad(rb, rc, d, matches)
        self.list_triads.selectPayload((float(rb), float(rc)))

    def _play_manual_interval(self):
        try:
            ratio = float(self.f_interval.text())
        except ValueError:
            self._status("Interval must be a single number", "RED", "input-error")
            return
        if not (math.isfinite(ratio) and ratio > 0):
            self._status("Interval ratio must be positive", "RED", "input-error")
            return
        self._play_interval_ratio(ratio)

    def _play_manual_note(self):
        try:
            ratio = float(self.f_note.text())
        except ValueError:
            self._status("Note must be a single number", "RED", "input-error")
            return
        if not (math.isfinite(ratio) and ratio > 0):
            self._status("Note ratio must be positive", "RED", "input-error")
            return
        if not self._read_settings():
            return
        spec_a, err = self.card_a.parsed()
        if spec_a is None:
            self._status("Voice A has invalid input", "RED", "input-error")
            return
        self._status(f"A {ratio:.4f} alone  ({ratio_to_interval_name(ratio)})", "GREEN")
        self._sound([(spec_a, ratio)], f"note_{ratio:.4f}")
        self.detail.show_note(ratio)

    def _play_manual_triad(self):
        try:
            parts = [float(p.strip()) for p in self.f_triad.text().split(',')]
        except ValueError:
            self._status("Triad must be two numbers, e.g. 1.25, 1.5", "RED", "input-error")
            return
        if len(parts) != 2:
            self._status("Enter exactly two ratios (B, C)", "RED", "input-error")
            return
        if not all(math.isfinite(p) and p > 0 for p in parts):
            self._status("Triad ratios must be positive", "RED", "input-error")
            return
        self._play_triad_ratio(parts[0], parts[1])

    def _replay(self):
        if self._last_sound is None:
            self._status("Nothing to replay yet \u2014 play something first", "RED",
                         "input-error")
            return
        voices, label, letters, marker = self._last_sound
        self._sound(voices, label, letters, marker)
        self._status(f"Replaying {label.replace('_', ' ')}", "GREEN")

    # ── glide (press and drag on the curve or heatmap) ─────────────────────

    def _glide_begin(self, kind, *ratios):
        if not self._read_settings():
            return
        spec_a, spec_b, spec_c = (card.get_spectrum() for card in self._cards())
        if kind == "interval":
            if spec_a is None or spec_b is None:
                self._status("Voice A or B has invalid input", "RED", "input-error")
                return
            specs, letters = [spec_a, spec_b], "AB"
        else:
            if spec_a is None or spec_b is None or spec_c is None:
                self._status("One of the voices has invalid input", "RED", "input-error")
                return
            specs, letters = [spec_a, spec_b, spec_c], "ABC"
        values = [1.0, *ratios]
        if not self.engine.glide_begin(list(zip(specs, values)), self.ref_freq):
            return
        self._glide = dict(kind=kind, specs=specs, letters=letters, values=values)
        self._glide_ui()

    def _glide_move(self, *ratios):
        g = self._glide
        if g is None:
            return
        g["values"] = [1.0, *ratios]
        self.engine.glide_to(g["values"])
        if not self._glide_tick.isActive():
            self._glide_tick.start()

    def _glide_end(self):
        g = self._glide
        if g is None:
            return
        self.engine.glide_end()
        self._glide_tick.stop()
        self._glide_ui(final=True)
        voices = list(zip(g["specs"], g["values"]))
        if g["kind"] == "interval":
            r = g["values"][1]
            label, marker = f"interval_{r:.4f}", ("interval", r)
        else:
            rb, rc = g["values"][1:]
            label, marker = f"triad_{rb:.3f}_{rc:.3f}", ("triad", rb, rc)
        self._last_sound = (voices, label, g["letters"], marker)
        self._glide = None

    def _glide_ui(self, final=False):
        g = self._glide
        if g is None:
            return
        vals = g["values"]
        self._shown_overtones = [(g["letters"][i], spec, r)
                                 for i, (spec, r) in enumerate(zip(g["specs"], vals))]
        matches = self._show_overtones()
        verb = "Glided to" if final else "Gliding \u00b7"
        if g["kind"] == "interval":
            r = vals[1]
            d = self._diss_at_ratio(r)
            tail = f" \u00b7 diss {d:.3f}" if d is not None else ""
            self._status(f"{verb}  A 1.000 + B {r:.4f}  ({ratio_to_interval_name(r)}){tail}",
                         "GREEN")
            self._set_sounding(("interval", r))
            self.detail.show_interval(r, d, matches)
        else:
            rb, rc = vals[1], vals[2]
            d = self._diss_at_triad(rb, rc)
            tail = f" \u00b7 diss {d:.3f}" if d is not None else ""
            self._status(f"{verb}  A 1.000 + B {rb:.3f} + C {rc:.3f}{tail}", "GREEN")
            self._set_sounding(("triad", rb, rc))
            self.detail.show_triad(rb, rc, d, matches)

    # ── volume and theme ───────────────────────────────────────────────────

    def _on_volume(self, value):
        self.engine.set_volume((value / 100.0) ** 2)
        name = ("ph.speaker-x-fill" if value == 0 else
                "ph.speaker-low-fill" if value < 50 else "ph.speaker-high-fill")
        self.btn_mute.setIconName(name)
        self.btn_mute.setToolTip("Unmute" if value == 0 else "Mute")

    def _toggle_mute(self):
        if self.volume.value() > 0:
            self._muted_volume = self.volume.value()
            self.volume.setValue(0)
        else:
            self.volume.setValue(self._muted_volume or 80)

    def _toggle_theme(self):
        target = PAPER_PALETTE if is_dark() else DARK_PALETTE

        def change():
            apply_app_theme(QApplication.instance(), target)
            self._restyle()

        crossfade(self.centralWidget(), change, 360)
        style_title_bar(self)

    def _restyle(self):
        """Re-read the active palette everywhere it was baked in at build time."""
        for card in self._cards():
            card.apply_palette()
        self.curve_view.apply_palette()
        self.heatmap_view.apply_palette()
        self.surface_view.apply_palette()
        self.surface_bar.apply_palette()
        self.overtone_view.apply_palette()
        self._show_overtones()
        self._set_dot(self._dot_role)
        self._label_voice_tabs()
        self._refresh_chrome_icons()
        self.detail.refresh()
        if self.curve is not None:
            self._fill_intervals()
        if self.data is not None:
            self._refresh_triads()
        for w in QApplication.allWidgets():
            w.update()

    # ── export ─────────────────────────────────────────────────────────────

    def _export_figure(self):
        if self.data is None:
            self._status("Nothing to export yet \u2014 run an analysis first", "RED",
                         "input-error")
            return
        index = self.stack.currentIndex()
        kind = self.VIEW_NAMES[index]
        if kind == "surface" and not HAS_GL:
            self._status("The 3D surface is unavailable on this machine", "RED", "input-error")
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
                      distance=float(cam["distance"]), shading=self.surface_view.lit)

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
        style_title_bar(editor)
        editor.exec()
        # Text and style carry over to the next figure; ranges, camera and shading follow the view.
        skip = ("xmin", "xmax", "ymax", "shading") + SURFACE_CAMERA_KEYS
        self._fig_state[kind] = {k: v for k, v in editor.state.items() if k not in skip}
        if editor.saved_path:
            path = editor.saved_path
            self._status(f"Figure saved to {Path(path).name}", "GREEN")
            self._notify(f"Figure saved to {Path(path).name}", "ok",
                         ("Show in folder", lambda: reveal_in_folder(path)))

    def _export_wav(self):
        if self._last_sound is None:
            self._status("Play something first, then save it", "RED", "input-error")
            return
        if not self._read_settings():
            return
        try:
            seconds = float(self.f_duration.text())
        except ValueError:
            self._status("Length must be a number of seconds", "RED", "input-error")
            return
        if not 0.05 <= seconds <= 600:
            self._status("Length must be between 0.05 and 600 seconds", "RED", "input-error")
            return

        voices, label = self._last_sound[0], self._last_sound[1]
        path, _ = QFileDialog.getSaveFileName(
            self, "Save sound", _safe_filename(label) + ".wav", "WAV audio (*.wav)")
        if not path:
            return
        if not path.lower().endswith(".wav"):
            path += ".wav"
        try:
            write_wav(path, synthesize_multi(voices, self.ref_freq, duration=seconds))
        except Exception as exc:
            self._status(f"Could not write the file: {exc}", "RED", "input-error")
            self._notify(f"Could not write the file: {exc}", "error")
            return
        message = f"Saved {seconds:g}\u202fs to {Path(path).name}"
        self._status(message, "GREEN")
        self._notify(message, "ok", ("Show in folder", lambda: reveal_in_folder(path)))

    # ── lifecycle ──────────────────────────────────────────────────────────

    def showEvent(self, event):
        super().showEvent(event)
        style_title_bar(self)
        self._refresh_chrome_icons()
        if HAS_GL and not getattr(self, "_gl_warm", False):
            self._gl_warm = True
            QTimer.singleShot(0, self._prewarm_gl)

    def _prewarm_gl(self):
        """Show the GL page once behind a cover: its first appearance costs ~1 s."""
        cover = _Cover(self.stack)
        cover.show()
        cover.raise_()
        self.stack.setCurrentIndex(1)
        state = {"done": False}

        def done():
            if state["done"]:
                return
            state["done"] = True
            self.stack.setCurrentIndex(self.tabs.currentIndex())
            cover.deleteLater()

        self.surface_view.frameSwapped.connect(done, Qt.SingleShotConnection)
        QTimer.singleShot(3000, done)

    def closeEvent(self, event):
        self.engine.close()
        self.worker.shutdown()
        if self.worker.isRunning():  # still compiling on a first launch
            self.worker.terminate()
            self.worker.wait(1000)
        if self._surface_renderer is not None:
            self._surface_renderer.close()
        super().closeEvent(event)


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

def apply_app_theme(app, palette_dict):
    """Switch the whole application over to *palette_dict*."""
    apply_palette(palette_dict)
    pg.setConfigOptions(antialias=True, background=MANTLE, foreground=TEXT)

    qp = app.palette()
    for role, color in ((QPalette.Window, BASE), (QPalette.Base, SURFACE),
                        (QPalette.AlternateBase, MANTLE), (QPalette.Text, TEXT),
                        (QPalette.WindowText, TEXT), (QPalette.Button, SURFACE),
                        (QPalette.ButtonText, TEXT), (QPalette.Highlight, BLUE),
                        (QPalette.HighlightedText, BASE), (QPalette.ToolTipBase, SURFACE),
                        (QPalette.ToolTipText, TEXT), (QPalette.PlaceholderText, MUTED)):
        qp.setColor(role, QColor(color))
    app.setPalette(qp)
    app.setStyleSheet(build_qss(palette_dict))


def main():
    # Windows rounds the default 5 ms GIL hand-off up to its ~15 ms timer tick, which starves
    # the compute worker and the audio callback whenever the GUI is busy in Python.
    sys.setswitchinterval(0.0005)
    if sys.platform == "win32":
        # Own taskbar identity, so Windows shows this app's icon rather than Python's.
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("DissonanceStudio")
        except Exception:
            pass
    # pyqtgraph caches compiled shaders globally, so every GL widget must share one context group.
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    app.setApplicationName("Dissonance Studio")
    app.setStyle("Fusion")
    apply_app_theme(app, DARK_PALETTE)
    if qta is not None:
        app.setWindowIcon(qta.icon("ph.waves", color=QColor(DARK_PALETTE["BLUE"])))

    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
