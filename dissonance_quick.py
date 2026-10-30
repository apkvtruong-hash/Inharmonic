"""
Dissonance Quick — a Qt Quick (QML) front-end for the Plomp-Levelt / Sethares
sensory-dissonance model: native FluentWinUI3 controls and Windows icons,
GPU-composited animations, QtGraphs for the dissonance curve and the 3D surface,
compiled (Numba) analysis kernels, a click-free audio engine with glides, and
publication figure export (PDF, SVG, PNG).

Requirements: numpy, sounddevice, PySide6 (with QtGraphs), pyqtgraph, PyOpenGL, numba
Run:          python dissonance_quick.py
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
import pyqtgraph as pg
import sounddevice as sd

from PySide6.QtCore import (QAbstractListModel, QByteArray, QEasingCurve, QMarginsF,
                            QModelIndex, QObject, QPointF, QRectF, QSize, QSizeF, Qt,
                            QThread, QTimer, QUrl, QVariantAnimation, Property, Signal, Slot)
from PySide6.QtGui import (QColor, QDesktopServices, QFont, QFontDatabase, QFontMetricsF,
                           QGuiApplication, QIcon, QImage, QLinearGradient, QPageSize,
                           QPainter, QPalette, QPdfWriter, QPen, QPixmap, QTextDocument,
                           QTransform, QVector3D, QVector4D)
from PySide6.QtQml import QQmlApplicationEngine, qmlRegisterType
from PySide6.QtQuick import QQuickImageProvider, QQuickItem
from PySide6.QtQuickControls2 import QQuickStyle
from PySide6.QtSvg import QSvgGenerator
from PySide6.QtWidgets import QApplication, QFileDialog
from PySide6 import QtGraphs

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
#  Background computation
# ═══════════════════════════════════════════════════════════════════════════

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
SURFACE_DEFAULTS = dict(
    title="", xlabel="Voice B ratio", ylabel="Voice C ratio",
    zlabel="Sensory dissonance", cbar="Sensory dissonance (normalized)",
    font="Times New Roman", size=9.0, axis=0.8, cmap="viridis", shading=False,
    floor=True, zaxis=True, colorbar=True, markers=True, labels="None",
    zscale=1.1, azimuth=68.0, elevation=40.0, distance=4.2,
    width=6.0, height=4.5, dpi=600, fmt="pdf",
)
SURFACE_CAMERA_KEYS = ("azimuth", "elevation", "distance")


class SmoothImageItem(pg.ImageItem):
    """ImageItem drawn with bilinear filtering when scaled up."""

    def paint(self, p, *args):
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        super().paint(p, *args)


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


def _laid_out_figure(kind, data, st, ppi):
    """make_figure, shown offscreen and painted until its axes have sized themselves."""
    fig = make_figure(kind, data, st, ppi)
    fig.setAttribute(Qt.WA_DontShowOnScreen, True)
    fig.show()
    # Axes only grow to fit their tick text after a first paint, so paint twice.
    for _ in range(2):
        QApplication.processEvents()
        fig.grab()
    QApplication.processEvents()
    return fig


def render_figure(kind, data, st, path):
    """Rebuild the figure natively at output resolution and write it to *path*."""
    fmt = st["fmt"]
    # Vector output is resolution-free; 96 keeps 1 px = 1/96 in so points stay exact.
    ppi = float(st["dpi"]) if fmt == "png" else SCREEN_DPI
    fig = _laid_out_figure(kind, data, st, ppi)
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
        gen.setTitle("Dissonance Quick figure")
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


def _to_unit(values, axis):
    """Map *values* on the ratio *axis* to GL scene coordinates in [-1, 1]."""
    lo, hi = axis[0], axis[-1]
    return (np.asarray(values, dtype=float) - lo) / max(hi - lo, 1e-12) * 2.0 - 1.0


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
#  Theme: neutral surfaces that sit with the Fluent controls
# ═══════════════════════════════════════════════════════════════════════════

QUICK_THEMES = {
    True: dict(window="#202020", card="#2b2b2b", raised="#3a3a3a", inset="#242424",
               border="#383838", text="#ffffff", subtext="#d0d0d0", muted="#9d9d9d",
               grid="#3c3c3c", gridSub="#303030", popup="#2f2f2f", cursor="#9d9d9d",
               yellow="#f9e2af", green="#6ccb5f", red="#ff99a4", markerEdge="#ffffff",
               voiceA="#5aa9ff", voiceB="#ff8a3d", voiceC="#3ddc97", contour="#22ffffff",
               cmap="magma"),
    False: dict(window="#f3f3f3", card="#ffffff", raised="#ffffff", inset="#f5f5f5",
                border="#e3e3e3", text="#1b1b1b", subtext="#3d3d3d", muted="#6e6e6e",
                grid="#e6e6e6", gridSub="#f0f0f0", popup="#ffffff", cursor="#8a8a8a",
                yellow="#d55e00", green="#0f7b0f", red="#c42b1c", markerEdge="#1b1b1b",
                voiceA="#0072b2", voiceB="#d55e00", voiceC="#009e73", contour="#55ffffff",
                cmap="viridis"),
}

VIEW_HINTS = (
    "click to hear  \u00b7  drag to glide  \u00b7  shift-drag to pan  \u00b7  scroll to zoom",
    "drag to rotate  \u00b7  scroll to zoom  \u00b7  click the surface to hear the triad",
    "click to hear  \u00b7  drag to glide  \u00b7  shift-drag to pan  \u00b7  scroll to zoom",
)

# QtGraphs' camera at rotation (x, y) shows the surface as the figure export's renderer does at
# azimuth 270 - x, elevation y.
CAMERA_DEFAULT = dict(x=-158.0, y=40.0, zoom=100.0)


def _font_family(*names):
    families = set(QFontDatabase.families())
    return next((n for n in names if n in families), names[-1])


def mix(a, b, t):
    """Linear blend of two colours (t = 0 gives a, 1 gives b)."""
    a, b = QColor(a), QColor(b)
    t = float(np.clip(t, 0.0, 1.0))
    return QColor.fromRgbF(a.redF() + (b.redF() - a.redF()) * t,
                           a.greenF() + (b.greenF() - a.greenF()) * t,
                           a.blueF() + (b.blueF() - a.blueF()) * t,
                           a.alphaF() + (b.alphaF() - a.alphaF()) * t)


def _diss_hex(value, th):
    v = float(np.clip(value, 0.0, 1.0))
    if v <= 0.45:
        return mix(th["green"], th["yellow"], v / 0.45).name()
    return mix(th["yellow"], th["red"], (v - 0.45) / 0.55).name()


def _gradient_stops(cmap_name, n=9):
    cm = pg.colormap.get(cmap_name)
    return [c.name() for c in cm.map(np.linspace(0.0, 1.0, n), mode="qcolor")]


def _title_bar(window, dark, caption, text, border):
    """Match the native Windows title bar to the theme (Windows 11; dark mode on 10)."""
    if sys.platform != "win32":
        return
    try:
        hwnd = ctypes.c_void_p(int(window.winId()))
        dwm = ctypes.windll.dwmapi

        def put(attr, value):
            v = ctypes.c_int(value)
            dwm.DwmSetWindowAttribute(hwnd, ctypes.c_uint(attr), ctypes.byref(v),
                                      ctypes.sizeof(v))

        def colorref(h):
            c = QColor(h)
            return c.red() | (c.green() << 8) | (c.blue() << 16)

        put(20, 1 if dark else 0)
        put(35, colorref(caption))
        put(36, colorref(text))
        put(34, colorref(border))
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  Voices
# ═══════════════════════════════════════════════════════════════════════════

class Voice(QObject):
    """One voice: where its spectrum comes from (preset, another voice or custom) and its partials."""

    changed = Signal()          # source, label, colour
    textChanged = Signal()      # the partial and amplitude strings
    spectrumChanged = Signal()  # the parsed partials
    presetApplied = Signal()
    dragging = Signal(bool)
    playRequested = Signal(object)

    def __init__(self, letter, sources=(), parent=None):
        super().__init__(parent)
        self._letter = letter
        self.sources = list(sources)
        self._color = "#ffffff"
        self._entries = []
        for i, (name, _) in enumerate(self.sources):
            self._entries.append((f"Same as {name}", "link", i,
                                  f"Mirror voice {name}; editing switches to Custom"))
        for i, (name, _, _) in enumerate(PRESETS):
            self._entries.append((name, "preset", i, PRESET_NOTES[name]))
        self._entries.append(("Custom", "custom", None, "Your own partials"))
        _, freqs, amps = PRESETS[0]
        self._freq_text = format_partials(freqs)
        self._amp_text = format_partials(amps)
        self._freqs, self._amps = [float(f) for f in freqs], [float(a) for a in amps]
        self._valid, self._error = True, ""
        self._index = 0
        self._shown_index = None
        for _, src in self.sources:
            src.textChanged.connect(self._sync_from_source)
        if not self.sources:
            self._select_by_text()
        self._apply_index()

    # ── QML properties ─────────────────────────────────────────────────────

    letter = Property(str, lambda self: self._letter, constant=True)
    entries = Property("QVariantList", lambda self: [e[0] for e in self._entries], constant=True)
    tips = Property("QVariantList", lambda self: [e[3] for e in self._entries], constant=True)
    index = Property(int, lambda self: self._index, notify=changed)
    freqText = Property(str, lambda self: self._freq_text, notify=textChanged)
    ampText = Property(str, lambda self: self._amp_text, notify=textChanged)
    freqs = Property("QVariantList", lambda self: self._freqs, notify=spectrumChanged)
    amps = Property("QVariantList", lambda self: self._amps, notify=spectrumChanged)
    count = Property(int, lambda self: len(self._freqs), notify=spectrumChanged)
    valid = Property(bool, lambda self: self._valid, notify=spectrumChanged)
    error = Property(str, lambda self: self._error, notify=spectrumChanged)
    color = Property(str, lambda self: self._color, notify=changed)
    linkedTo = Property(str, lambda self: self.source_name() or "", notify=changed)
    label = Property(str, lambda self: self._letter + (f" = {self.source_name()}"
                                                       if self.is_linked() else ""), notify=changed)

    def setColor(self, color):
        self._color = color
        self.changed.emit()

    # ── source logic ──────────────────────────────────────────────────────

    def _kind(self):
        _, kind, i, _ = self._entries[self._index]
        return kind, i

    def is_linked(self):
        return self._kind()[0] == "link"

    def source(self):
        kind, i = self._kind()
        return self.sources[i][1] if kind == "link" else None

    def source_name(self):
        kind, i = self._kind()
        return self.sources[i][0] if kind == "link" else None

    def source_label(self):
        return self._entries[self._index][0]

    def parsed(self):
        src = self.source()
        if src is not None:
            return src.parsed()
        return parse_spectrum(self._freq_text, self._amp_text)

    def get_spectrum(self):
        return self.parsed()[0]

    def _select_by_text(self):
        current = ("".join(self._freq_text.split()), "".join(self._amp_text.split()))
        target = ("custom", None)
        for i, (_, freqs, amps) in enumerate(PRESETS):
            if current == (format_partials(freqs).replace(" ", ""),
                           format_partials(amps).replace(" ", "")):
                target = ("preset", i)
                break
        self._index = next(k for k, e in enumerate(self._entries) if (e[1], e[2]) == target)

    def _set_texts(self, freq_text, amp_text):
        if freq_text == self._freq_text and amp_text == self._amp_text:
            return False
        self._freq_text, self._amp_text = freq_text, amp_text
        self.textChanged.emit()
        return True

    def _refresh(self):
        spec, err = self.parsed()
        self._valid, self._error = spec is not None, err or ""
        if spec is not None:
            self._freqs = [float(f) for f in spec["freq"]]
            self._amps = [float(a) for a in spec["amp"]]
        if self._index != self._shown_index:
            self._shown_index = self._index
            self.changed.emit()
        self.spectrumChanged.emit()

    def _sync_from_source(self):
        src = self.source()
        if src is not None:
            self._set_texts(src._freq_text, src._amp_text)
            self._refresh()

    def _detach(self):
        if self.is_linked():
            self._select_by_text()

    def _apply_index(self):
        kind, i = self._kind()
        if kind == "link":
            src = self.sources[i][1]
            self._set_texts(src._freq_text, src._amp_text)
        elif kind == "preset":
            _, freqs, amps = PRESETS[i]
            self._set_texts(format_partials(freqs), format_partials(amps))
        self._refresh()

    def _set_partials(self, freqs, amps):
        self._set_texts(format_partials(freqs), format_partials(amps))
        self._select_by_text()
        self._refresh()

    # ── slots ──────────────────────────────────────────────────────────────

    @Slot(int)
    def setIndex(self, index):
        if 0 <= index < len(self._entries):
            self._index = index
            self._apply_index()
            self.presetApplied.emit()

    @Slot(str)
    def setFreqText(self, text):
        self._detach()
        self._freq_text = text
        self.textChanged.emit()
        self._select_by_text()
        self._refresh()

    @Slot(str)
    def setAmpText(self, text):
        self._detach()
        self._amp_text = text
        self.textChanged.emit()
        self._select_by_text()
        self._refresh()

    @Slot()
    def beginDrag(self):
        self._detach()
        self.dragging.emit(True)

    @Slot(int, float, float)
    def dragPartial(self, i, f, a):
        if not 0 <= i < len(self._freqs):
            return
        freqs, amps = list(self._freqs), list(self._amps)
        freqs[i], amps[i] = float(f"{f:.4g}"), float(f"{a:.4g}")
        self._set_partials(freqs, amps)

    @Slot()
    def endDrag(self):
        self.dragging.emit(False)

    @Slot(float, float)
    def addPartial(self, f, a):
        self._detach()
        self._set_partials(self._freqs + [float(f"{max(f, 0.01):.4g}")],
                           self._amps + [float(f"{max(a, 0.0):.4g}")])

    @Slot(int)
    def removePartial(self, i):
        if len(self._freqs) > 1 and 0 <= i < len(self._freqs):
            self._detach()
            self._set_partials(self._freqs[:i] + self._freqs[i + 1:],
                               self._amps[:i] + self._amps[i + 1:])

    @Slot()
    def play(self):
        self.playRequested.emit(self)


# ═══════════════════════════════════════════════════════════════════════════
#  Result lists
# ═══════════════════════════════════════════════════════════════════════════

class ResultsModel(QAbstractListModel):
    """Rows of dicts exposed by role name; same-length updates change rows in place."""

    ROLES = ("primary", "secondary", "value", "tint", "key", "b", "c")

    def __init__(self, parent=None, roles=ROLES):
        super().__init__(parent)
        self._rows = []
        self._role_ids = {Qt.UserRole + 1 + k: name.encode() for k, name in enumerate(roles)}
        self._by_name = {v.decode(): k for k, v in self._role_ids.items()}

    def roleNames(self):
        return self._role_ids

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._rows)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role not in self._role_ids:
            return None
        return self._rows[index.row()][self._role_ids[role].decode()]

    def setRows(self, rows):
        # Change rows in place and only insert or remove the tail, so views keep their delegates.
        rows = list(rows)
        old, new = len(self._rows), len(rows)
        if new < old:
            self.beginRemoveRows(QModelIndex(), new, old - 1)
            del self._rows[new:]
            self.endRemoveRows()
        elif new > old:
            self.beginInsertRows(QModelIndex(), old, new - 1)
            self._rows.extend(rows[old:])
            self.endInsertRows()
        common = min(old, new)
        if common:
            self._rows[:common] = rows[:common]
            self.dataChanged.emit(self.index(0), self.index(common - 1))

    def row(self, i):
        return self._rows[i] if 0 <= i < len(self._rows) else None


# ═══════════════════════════════════════════════════════════════════════════
#  Heatmap (QtGraphs has no image series, so this one is painted)
# ═══════════════════════════════════════════════════════════════════════════

_FRAMES = {}


class FrameProvider(QQuickImageProvider):
    """Serves images painted in Python to QML (image://frames/<key>?<serial>).

    Used for the heatmap's frames and the theme-switch snapshot. The heatmap is painted on
    the GUI thread: a QQuickPaintedItem would run Python on the scene graph's render thread,
    which deadlocks whenever the GUI thread holds the GIL.
    """

    def __init__(self):
        super().__init__(QQuickImageProvider.ImageType.Image)

    def requestImage(self, image_id, size, requested_size):
        img = _FRAMES.get(image_id.split("?")[0])
        if img is None:
            img = QImage(1, 1, QImage.Format_ARGB32_Premultiplied)
            img.fill(Qt.transparent)
        return img


class HeatmapItem(QQuickItem):
    """Triad map: colour-mapped surface, contours, best triads, axes and colour bar.

    Hover, glide and the pulsing rings are QML overlays; the frame is only repainted when
    the data, the view or the size changes.
    """

    viewChanged = Signal()
    frameChanged = Signal()

    MARGINS = (62, 14, 104, 48)  # left, top, right, bottom

    def __init__(self, parent=None):
        super().__init__(parent)
        self._key = f"h{id(self):x}"
        self._frame_no = 0
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(0)
        self._render_timer.timeout.connect(self._render)
        self._x = self._y = self._z = None
        self._img = None
        self._lut = None
        self._lo, self._hi = 0.0, 1.0
        self._tri = (np.array([]), np.array([]))
        self._contours = None
        self._view = None
        self._version = 0
        self._theme = dict(QUICK_THEMES[True])
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(420)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.valueChanged.connect(self._anim_step)
        self._anim_from = self._anim_to = None
        self.widthChanged.connect(self._bump)
        self.heightChanged.connect(self._bump)

    # ── QML API ────────────────────────────────────────────────────────────

    def _get_plot_rect(self):
        return self._geometry()[0]

    plotRect = Property(QRectF, _get_plot_rect, notify=viewChanged)
    viewVersion = Property(int, lambda self: self._version, notify=viewChanged)
    hasData = Property(bool, lambda self: self._z is not None, notify=viewChanged)
    frameSource = Property(str, lambda self: f"image://frames/{self._key}?{self._frame_no}",
                           notify=frameChanged)

    @Slot(float, float, result=QPointF)
    def toPixel(self, b, c):
        rect, (x0, x1, y0, y1) = self._geometry()
        return QPointF(rect.left() + (b - x0) / (x1 - x0) * rect.width(),
                       rect.bottom() - (c - y0) / (y1 - y0) * rect.height())

    @Slot(float, float, result=QPointF)
    def toData(self, px, py):
        rect, (x0, x1, y0, y1) = self._geometry()
        return QPointF(x0 + (px - rect.left()) / max(rect.width(), 1.0) * (x1 - x0),
                       y0 + (rect.bottom() - py) / max(rect.height(), 1.0) * (y1 - y0))

    @Slot(float, float, result=bool)
    def inData(self, b, c):
        return bool(self._x is not None and self._x[0] <= b <= self._x[-1]
                    and self._y[0] <= c <= self._y[-1])

    @Slot(float, float, float, result="QVariantMap")
    def snapAt(self, px, py, radius):
        """The best triad whose marker is within *radius* px of (px, py), or {}."""
        tx, ty = self._tri
        best, best_d = {}, radius
        for b, c in zip(tx, ty):
            q = self.toPixel(float(b), float(c))
            d = math.hypot(q.x() - px, q.y() - py)
            if d <= best_d:
                best, best_d = dict(b=float(b), c=float(c), key=_triad_key(b, c)), d
        return best

    @Slot(float, float, float)
    def zoomAt(self, px, py, factor):
        if self._view is None:
            return
        anchor = self.toData(px, py)
        x0, x1, y0, y1 = self._view
        span = max(x1 - x0, y1 - y0) * factor
        if span < 0.02 or span > 20:
            return
        b, c = anchor.x(), anchor.y()
        self._anim.stop()
        self._view = (b - (b - x0) * factor, b + (x1 - b) * factor,
                      c - (c - y0) * factor, c + (y1 - c) * factor)
        self._bump()

    @Slot(float, float)
    def panBy(self, dx, dy):
        if self._view is None:
            return
        rect, (x0, x1, y0, y1) = self._geometry()
        sx = (x1 - x0) / max(rect.width(), 1.0)
        sy = (y1 - y0) / max(rect.height(), 1.0)
        a, b, c, d = self._view
        self._anim.stop()
        self._view = (a - dx * sx, b - dx * sx, c + dy * sy, d + dy * sy)
        self._bump()

    @Slot()
    def resetView(self):
        if self._x is None or self._view is None:
            return
        self._anim_from = self._view
        self._anim_to = self._full_view()
        self._anim.stop()
        self._anim.start()

    # ── data ───────────────────────────────────────────────────────────────

    def setTheme(self, theme):
        self._theme = dict(theme)
        if self._z is not None:
            self._build_image()
        self._render_timer.start()

    def setData(self, x, y, z, tri_x, tri_y):
        extent_changed = (self._x is None or len(self._x) != len(x) or self._x[0] != x[0]
                          or self._x[-1] != x[-1] or self._y[-1] != y[-1])
        self._x, self._y, self._z = x, y, z
        self._tri = (np.asarray(tri_x, dtype=float), np.asarray(tri_y, dtype=float))
        self._build_image()
        if self._view is None or extent_changed:
            self._view = self._full_view()
        self._bump()

    def setContours(self, contours):
        self._contours = contours
        self._render_timer.start()

    def _full_view(self):
        return (float(self._x[0]), float(self._x[-1]), float(self._y[0]), float(self._y[-1]))

    def _build_image(self):
        z = self._z
        self._lo, self._hi = float(z.min()), float(z.max())
        self._lut = pg.colormap.get(self._theme["cmap"]).getLookupTable(0.0, 1.0, 256, alpha=True)
        idx = np.clip(((z - self._lo) / max(self._hi - self._lo, 1e-12) * 255).astype(np.intp),
                      0, 255)
        # z is [B, C]; image rows run from high C at the top down to low C.
        rgba = np.ascontiguousarray(self._lut[idx].transpose(1, 0, 2)[::-1])
        h, w = rgba.shape[:2]
        self._img = QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888).copy()

    def _anim_step(self, t):
        if self._anim_from is None:
            return
        t = float(t)
        self._view = tuple(a + (b - a) * t for a, b in zip(self._anim_from, self._anim_to))
        self._bump()

    def _bump(self):
        self._version += 1
        self.viewChanged.emit()
        self._render_timer.start()

    def _render(self):
        win = self.window()
        dpr = win.devicePixelRatio() if win is not None else 1.0
        w, h = round(self.width() * dpr), round(self.height() * dpr)
        if w < 2 or h < 2:
            return
        img = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
        img.setDevicePixelRatio(dpr)
        img.fill(Qt.transparent)
        p = QPainter(img)
        self._paint(p)
        p.end()
        _FRAMES[self._key] = img
        self._frame_no += 1
        self.frameChanged.emit()

    def _geometry(self):
        """Plot rectangle and the data range it shows (square data units)."""
        left, top, right, bottom = self.MARGINS
        rect = QRectF(left, top, max(self.width() - left - right, 10.0),
                      max(self.height() - top - bottom, 10.0))
        if self._view is None:
            return rect, (1.0, 2.0, 1.0, 2.0)
        x0, x1, y0, y1 = self._view
        aspect = rect.width() / rect.height()
        if (x1 - x0) / (y1 - y0) < aspect:
            cx, half = (x0 + x1) / 2, (y1 - y0) * aspect / 2
            x0, x1 = cx - half, cx + half
        else:
            cy, half = (y0 + y1) / 2, (x1 - x0) / aspect / 2
            y0, y1 = cy - half, cy + half
        return rect, (x0, x1, y0, y1)

    # ── painting ───────────────────────────────────────────────────────────

    def _paint(self, p):
        th = self._theme
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        p.setRenderHint(QPainter.TextAntialiasing, True)
        rect, (x0, x1, y0, y1) = self._geometry()

        def px(b):
            return rect.left() + (b - x0) / (x1 - x0) * rect.width()

        def py(c):
            return rect.bottom() - (c - y0) / (y1 - y0) * rect.height()

        tick_font = QFont(self._ui_font())
        tick_font.setPixelSize(11)
        fm = QFontMetricsF(tick_font)
        if self._img is None:
            p.setFont(tick_font)
            p.setPen(QColor(th["muted"]))
            p.drawText(QRectF(0, 0, self.width(), self.height()), Qt.AlignCenter,
                       "Waiting for the first analysis\u2026")
            return

        dx0, dx1, dy0, dy1 = (float(self._x[0]), float(self._x[-1]),
                              float(self._y[0]), float(self._y[-1]))
        p.save()
        p.setClipRect(rect)
        p.drawImage(QRectF(px(dx0), py(dy1), px(dx1) - px(dx0), py(dy0) - py(dy1)), self._img)
        nx, ny = self._z.shape
        if self._contours is not None:
            scale, paths = self._contours
            t = QTransform(scale * (px(dx1) - px(dx0)) / nx, 0, 0,
                           scale * (py(dy1) - py(dy0)) / ny, px(dx0), py(dy0))
            pen = QPen(QColor(th["contour"]), 1.0)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.setTransform(t, True)
            for path in paths:
                p.drawPath(path)
            p.resetTransform()
        self._paint_triads(p, rect, px, py)
        p.restore()

        axis = QColor(th["border"])
        p.setPen(QPen(axis, 1.0))
        p.drawRect(rect)
        p.setFont(tick_font)
        text = QColor(th["subtext"])
        for v in _nice_ticks(x0, x1, max(3, int(rect.width() / 90))):
            x = px(v)
            if rect.left() - 0.5 <= x <= rect.right() + 0.5:
                p.setPen(QPen(axis, 1.0))
                p.drawLine(QPointF(x, rect.bottom()), QPointF(x, rect.bottom() + 5))
                p.setPen(text)
                p.drawText(QRectF(x - 30, rect.bottom() + 7, 60, fm.height()), Qt.AlignHCenter,
                           f"{v:g}")
        for v in _nice_ticks(y0, y1, max(3, int(rect.height() / 70))):
            y = py(v)
            if rect.top() - 0.5 <= y <= rect.bottom() + 0.5:
                p.setPen(QPen(axis, 1.0))
                p.drawLine(QPointF(rect.left() - 5, y), QPointF(rect.left(), y))
                p.setPen(text)
                p.drawText(QRectF(rect.left() - 52, y - fm.height() / 2, 44, fm.height()),
                           Qt.AlignRight | Qt.AlignVCenter, f"{v:g}")
        title_font = QFont(self._ui_font())
        title_font.setPixelSize(12)
        p.setFont(title_font)
        p.setPen(text)
        p.drawText(QRectF(rect.left(), rect.bottom() + 24, rect.width(), 20), Qt.AlignHCenter,
                   "Voice B ratio  (B / A)")
        p.save()
        p.translate(14, rect.center().y())
        p.rotate(-90)
        p.drawText(QRectF(-rect.height() / 2, -9, rect.height(), 20), Qt.AlignHCenter,
                   "Voice C ratio  (C / A)")
        p.restore()
        self._paint_colorbar(p, rect, tick_font, fm)

    def _paint_triads(self, p, rect, px, py):
        tx, ty = self._tri
        if not len(tx):
            return
        label_font = QFont(self._ui_font())
        label_font.setPixelSize(10)
        label_font.setBold(True)
        fm = QFontMetricsF(label_font)
        p.setFont(label_font)
        placed = []
        for bx, cy in zip(tx, ty):
            q = QPointF(px(bx), py(cy))
            p.setPen(QPen(QColor("#ffffff"), 1.4))
            p.setBrush(QColor(self._theme["yellow"]))
            p.drawEllipse(q, 5.5, 5.5)
            text = f"{bx:.2f} \u00b7 {cy:.2f}"
            w, h = fm.horizontalAdvance(text) + 8, fm.height() + 2
            box = None
            for ax, ay in ((0.0, 1.0), (1.0, 1.0), (0.0, 0.0), (1.0, 0.0), (0.0, 2.0), (0.0, -1.0)):
                r = QRectF(q.x() + (8 if ax == 0 else -8 - w), q.y() - ay * h - (4 if ay > 0 else -6),
                           w, h)
                if not any(r.intersects(o) for o in placed):
                    box = r
                    break
            box = box or QRectF(q.x() + 8, q.y() - h - 4, w, h)
            placed.append(box)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 140))
            p.drawRoundedRect(box, 3, 3)
            p.setPen(QColor("#ffffff"))
            p.drawText(box, Qt.AlignCenter, text)

    def _paint_colorbar(self, p, rect, font, fm):
        bar = QRectF(rect.right() + 22, rect.top() + rect.height() * 0.08, 12, rect.height() * 0.84)
        grad = QLinearGradient(bar.bottomLeft(), bar.topLeft())
        for k in range(0, 256, 16):
            r, g, b, a = (int(v) for v in self._lut[k])
            grad.setColorAt(k / 255.0, QColor(r, g, b, a))
        r, g, b, a = (int(v) for v in self._lut[255])
        grad.setColorAt(1.0, QColor(r, g, b, a))
        p.setPen(QPen(QColor(self._theme["border"]), 1.0))
        p.setBrush(grad)
        p.drawRoundedRect(bar, 3, 3)
        p.setFont(font)
        lo, hi = self._lo, self._hi
        for v in _nice_ticks(lo, hi, 5):
            y = bar.bottom() - (v - lo) / max(hi - lo, 1e-12) * bar.height()
            p.setPen(QColor(self._theme["subtext"]))
            p.drawText(QRectF(bar.right() + 6, y - fm.height() / 2, 40, fm.height()),
                       Qt.AlignLeft | Qt.AlignVCenter, f"{v:g}")
        p.save()
        p.translate(bar.right() + 58, bar.center().y())
        p.rotate(-90)
        p.setPen(QColor(self._theme["muted"]))
        p.drawText(QRectF(-bar.height() / 2, -8, bar.height(), 16), Qt.AlignHCenter,
                   "Sensory dissonance")
        p.restore()

    @staticmethod
    def _ui_font():
        return QFont("Segoe UI Variable Text")


# ═══════════════════════════════════════════════════════════════════════════
#  Figure editor: form, live true-size preview and saving (the QML dialog binds to it)
# ═══════════════════════════════════════════════════════════════════════════

def _section(label):
    return dict(type="section", key="", label=label)


def _hint(label):
    return dict(type="hint", key="", label=label)


class FigureSession(QObject):
    """One open figure editor: its settings, preview image and save."""

    changed = Signal()
    previewChanged = Signal()
    messageChanged = Signal()
    finished = Signal(str, "QVariantMap", str)  # kind, settings, saved path or ""

    RICH_TEXT = "Rich text works: <i>\u2026</i>, <sub>\u2026</sub>, <sup>\u2026</sup>"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._open = False
        self._kind = ""
        self._data = None
        self._state = {}
        self._renderer = None
        self._fields = []
        self._limits = {}
        self._choices = {}
        self._dpr = 1.0
        self._serial = 0
        self._size = (0, 0)
        self._message = ""
        self._saved = ""
        self._folder = None
        self._timer = _timer(self, 120, self._rebuild)

    # ── QML API ────────────────────────────────────────────────────────────

    open = Property(bool, lambda s: s._open, notify=changed)
    kind = Property(str, lambda s: s._kind, notify=changed)
    title = Property(str, lambda s: f"Figure editor \u2014 {s._kind}", notify=changed)
    fields = Property("QVariantList", lambda s: s._fields, notify=changed)
    fmt = Property(str, lambda s: s._state.get("fmt", "pdf"), notify=previewChanged)
    fileFilter = Property(str, lambda s: next((f[2] for f in FIG_FORMATS
                                               if f[1] == s._state.get("fmt")), ""),
                          notify=previewChanged)
    folder = Property(QUrl, lambda s: QUrl.fromLocalFile(str(s._save_folder())), notify=changed)
    defaultFile = Property(QUrl, lambda s: QUrl.fromLocalFile(
        str(s._save_folder() / f"dissonance_{s._kind}.{s._state.get('fmt', 'pdf')}")),
        notify=previewChanged)
    preview = Property(str, lambda s: f"image://frames/figure?{s._serial}" if s._serial else "",
                       notify=previewChanged)
    previewWidth = Property(float, lambda s: s._size[0], notify=previewChanged)
    previewHeight = Property(float, lambda s: s._size[1], notify=previewChanged)
    message = Property(str, lambda s: s._message, notify=messageChanged)

    def begin(self, kind, data, state, renderer=None, dpr=1.0):
        self._kind, self._data, self._state = kind, data, dict(state)
        self._renderer, self._dpr, self._saved = renderer, dpr, ""
        self._fields = self._build_fields()
        self._open = True
        self.changed.emit()
        self._rebuild()

    @Slot()
    def close(self):
        if not self._open:
            return
        self._open = False
        self._timer.stop()
        self.changed.emit()
        _FRAMES.pop("figure", None)
        self._data = self._renderer = None
        self.finished.emit(self._kind, self._state, self._saved)

    @Slot(str, "QVariant")
    def setValue(self, key, value):
        self._state[key] = value
        self._timer.start()

    @Slot(str, str)
    def setNumber(self, key, text):
        try:
            value = float(text)
        except ValueError:
            return
        lo, hi = self._limits[key]
        if math.isfinite(value) and lo <= value <= hi:
            self._state[key] = value
            self._timer.start()

    @Slot(str, int)
    def setChoice(self, key, index):
        values = self._choices[key]
        if 0 <= index < len(values):
            self._state[key] = values[index]
            self._timer.start()

    @Slot(QUrl)
    def save(self, url):
        if not self._valid():
            return
        path = url.toLocalFile()
        fmt = self._state["fmt"]
        if not path.lower().endswith("." + fmt):
            path += "." + fmt
        try:
            if self._kind == "surface":
                render_surface_figure(self._renderer, self._state, path)
            else:
                render_figure(self._kind, self._data, self._state, path)
        except Exception as exc:
            self._set_message(f"Save failed: {exc}")
            return
        self._saved = path
        self._folder = Path(path).parent
        self._set_message(f"Saved {Path(path).name}")

    # ── form ───────────────────────────────────────────────────────────────

    def _text(self, key, label):
        return dict(type="text", key=key, label=label, value=str(self._state[key]))

    def _num(self, key, label, lo, hi, decimals):
        self._limits[key] = (lo, hi)
        return dict(type="num", key=key, label=label, lo=lo, hi=hi, decimals=decimals,
                    value=f"{float(self._state[key]):.{decimals}f}")

    def _check(self, key, label):
        return dict(type="check", key=key, label=label, value=bool(self._state[key]))

    def _combo(self, key, label, items):
        values = [v for _, v in items]
        self._choices[key] = values
        index = values.index(self._state[key]) if self._state[key] in values else 0
        return dict(type="combo", key=key, label=label, options=[t for t, _ in items],
                    index=index)

    def _build_fields(self):
        modes = [(m, m) for m in FIG_LABEL_MODES]
        if self._kind == "surface":
            return [
                _section("Text"), self._text("title", "Title"),
                self._text("xlabel", "B axis"), self._text("ylabel", "C axis"),
                self._text("zlabel", "Height axis"), self._text("cbar", "Colour bar"),
                _hint(self.RICH_TEXT),
                _section("Style"), self._combo("font", "Font", FIG_FONTS),
                self._num("size", "Font size (pt)", 5, 24, 1),
                self._num("axis", "Axes (pt)", 0.1, 4, 2),
                self._combo("cmap", "Colour map", FIG_CMAPS),
                self._combo("shading", "Surface", [("Flat", False), ("Shaded", True)]),
                self._check("floor", "Floor grid"), self._check("zaxis", "Dissonance axis"),
                self._check("colorbar", "Colour bar"), self._check("markers", "Mark best triads"),
                self._combo("labels", "Triad labels", modes),
                _hint("The view angle is taken from the Surface tab \u2014 rotate it there "
                      "before opening this editor."),
                *self._output_fields("Resolution (dpi)"),
                _hint("PDF/SVG keep text and axes as vectors; the surface is embedded at "
                      "this resolution."),
            ]
        curve = self._kind == "curve"
        fields = [_section("Text"), self._text("title", "Title"),
                  self._text("xlabel", "X axis"), self._text("ylabel", "Y axis")]
        if not curve:
            fields.append(self._text("cbar", "Colour bar"))
        fields += [_hint(self.RICH_TEXT),
                   _section("Style"), self._combo("font", "Font", FIG_FONTS),
                   self._num("size", "Font size (pt)", 5, 24, 1)]
        if curve:
            fields.append(self._num("line", "Line (pt)", 0.1, 6, 2))
        fields += [self._num("axis", "Axes (pt)", 0.1, 4, 2),
                   self._combo("ticks_in", "Ticks", [("Inside", True), ("Outside", False)]),
                   self._check("frame", "Box frame (top and right axes)"),
                   self._check("grid", "Grid lines")]
        if curve:
            top = float(self._data["ratios_2d"][-1])
            fields += [self._check("fill", "Shade under curve"),
                       self._check("markers", "Mark minima"),
                       self._combo("labels", "Minima labels", modes),
                       _section("Range"), self._num("xmin", "X from", 0.5, top, 3),
                       self._num("xmax", "X to", 0.5, top, 3),
                       self._num("ymax", "Y max (0 = auto)", 0, 2, 3)]
        else:
            fields += [self._combo("cmap", "Colour map", FIG_CMAPS),
                       self._check("contours", "Contour lines"),
                       self._check("square", "Equal axis scaling"),
                       self._check("markers", "Mark best triads"),
                       self._combo("labels", "Triad labels", modes)]
        return fields + self._output_fields("PNG resolution (dpi)")

    def _output_fields(self, dpi_label):
        return [_section("Output"), self._num("width", "Width (in)", 1, 14, 2),
                self._num("height", "Height (in)", 1, 14, 2),
                self._combo("fmt", "Format", [(f[0], f[1]) for f in FIG_FORMATS]),
                self._num("dpi", dpi_label, 72, 1200, 0),
                _hint("Single column \u2248 3.5 in, double column \u2248 7 in")]

    # ── preview ────────────────────────────────────────────────────────────

    def _save_folder(self):
        if self._folder is not None and self._folder.is_dir():
            return self._folder
        docs = Path.home() / "Documents"
        return docs if docs.is_dir() else Path.home()

    def _set_message(self, text):
        self._message = text
        self.messageChanged.emit()

    def _valid(self):
        if self._kind == "curve" and self._state["xmin"] >= self._state["xmax"]:
            self._set_message("X from must be less than X to")
            return False
        return True

    def _rebuild(self):
        if not self._open or not self._valid():
            return
        st, dpr = self._state, self._dpr
        if self._kind == "surface":
            w, h = round(st["width"] * SCREEN_DPI), round(st["height"] * SCREEN_DPI)
            image = QImage(round(w * dpr), round(h * dpr), QImage.Format_ARGB32)
            image.fill(QColor("#ffffff"))
            p = QPainter(image)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            try:
                paint_surface_figure(p, self._renderer, st, image.width(), image.height(),
                                     SCREEN_DPI * dpr, final=True)
            finally:
                p.end()
        else:
            fig = _laid_out_figure(self._kind, self._data, st, SCREEN_DPI * dpr)
            try:
                src = fig.viewport().rect()
                image = QImage(src.width(), src.height(), QImage.Format_ARGB32)
                image.fill(QColor("#ffffff"))
                p = QPainter(image)
                p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
                try:
                    fig.render(p, QRectF(0, 0, src.width(), src.height()), src)
                finally:
                    p.end()
            finally:
                fig.close()
                fig.deleteLater()
            w, h = image.width() / dpr, image.height() / dpr
        _FRAMES["figure"] = image
        self._serial += 1
        self._size = (float(w), float(h))
        self.previewChanged.emit()
        self._set_message(f"Actual size preview \u00b7 {st['width']:g} \u00d7 {st['height']:g} in")


# ═══════════════════════════════════════════════════════════════════════════
#  Backend: the app logic QML binds to
# ═══════════════════════════════════════════════════════════════════════════

def _timer(owner, ms, slot):
    t = QTimer(owner)
    t.setSingleShot(True)
    t.setInterval(ms)
    t.timeout.connect(slot)
    return t


def _interval_key(x):
    return f"{float(x):.6f}"


def _triad_key(b, c):
    return f"{float(b):.6f}|{float(c):.6f}"


class Backend(QObject):
    themeChanged = Signal()
    statusChanged = Signal()
    busyChanged = Signal()
    settingsChanged = Signal()
    curveChanged = Signal()
    minimaChanged = Signal()
    curveReset = Signal()
    curveRefit = Signal()
    surfaceChanged = Signal()
    markersChanged = Signal()
    inspectorChanged = Signal()
    overtonesChanged = Signal()
    overtonesAnimate = Signal()
    linkedChanged = Signal()
    audioChanged = Signal()
    levelChanged = Signal()
    toast = Signal(str, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._dark = True
        self._theme = dict(QUICK_THEMES[True])
        self._accent = "#4cc2ff"
        self._gradient = _gradient_stops("magma")
        self._mono = _font_family("Cascadia Mono", "Consolas")
        self._icons = _font_family("Segoe Fluent Icons", "Segoe MDL2 Assets")

        self.voice_a = Voice("A", parent=self)
        self.voice_b = Voice("B", [("A", self.voice_a)], parent=self)
        self.voice_c = Voice("C", [("A", self.voice_a), ("B", self.voice_b)], parent=self)
        for v in self._voices():
            v.spectrumChanged.connect(self._on_inputs_changed)
            v.playRequested.connect(self._play_voice)
            v.dragging.connect(self._on_dragging)

        self._ref_text, self._max_text, self._step_text = "261.63", "2.3", "0.005"
        self._settings_error = ""
        self.ref_freq, self.max_interval, self.step_3d = 261.63, 2.3, 0.005

        self._live = True
        self._volume = 1.0
        self._muted_volume = None
        self._level = 0.0
        self._audio_active = False

        self._status_text = "Preparing the dissonance engine\u2026"
        self._status_role = "yellow"
        self._status_kind = "busy"
        self._busy = False
        self._busy_visible = False
        self._progress = 0.0
        self._ready = False
        self._view = 0

        self.data = None            # last complete analysis
        self.curve = None           # newest curve stage
        self._partial = {}
        self._gen = self._stage_gen = self._shown_gen = 0
        self._job_reset = False
        self._view3d = None
        self._contours = None
        self._dirty = set()
        self._dragging = False
        self._last_sound = None
        self._glide = None
        self._shown_overtones = []
        self._fig_state = {}
        self._surface_renderer = None
        self._toast_action = None
        self.figure = FigureSession(self)
        self.figure.finished.connect(self._figure_closed)

        self._curve_x = self._curve_y = self._curve_shown = None
        self._curve_minima = []
        self._curve_version = 0
        self._grid = (1.0, 2.3)
        self._sx = self._sz = self._surf_shown = None

        self._playing = dict(kind="", x=0.0, b=0.0, c=0.0)
        self._highlight = dict(kind="", x=0.0, b=0.0, c=0.0)
        self._linked_interval = ""
        self._linked_triad = ""
        self._snap = None
        self._inspector = dict(empty=True)
        self._inspector_args = None
        self._overtones = dict(empty=True)

        self.intervals = ResultsModel(self)
        self.triads = ResultsModel(self)
        self.minima = ResultsModel(self, roles=("rx", "ry", "label", "key"))
        self._sort_mode = 0
        self._triad_mode = 0

        self.engine = AudioEngine(self)
        self.engine.activityChanged.connect(self._on_audio_activity)
        self.engine.level.connect(self._on_level)
        self.engine.failed.connect(lambda m: self.toast.emit(m, "error", ""))

        self.worker = ComputeWorker(self)
        self.worker.ready.connect(self._on_ready)
        self.worker.stage.connect(self._on_stage)
        self.worker.progress.connect(self._on_progress)
        self.worker.failed.connect(self._on_failed)

        self._debounce = _timer(self, 200, self._submit)
        self._slow = _timer(self, 160, self._show_busy)
        self._surface_wait = _timer(self, 120, self._show_surface_early)
        self._glide_tick = _timer(self, 33, self._glide_ui)
        self._warm_note = _timer(self, 1500, self._note_compiling)

        self._morph = self._animation(170, self._morph_step)
        self._morph_from = self._morph_to = None
        self._surf_morph = self._animation(180, self._surf_morph_step)
        self._surf_from = self._surf_to = None

        self._line = self._fill = self._proxy = self._surface_series = None
        self._tri_proxy = self._play_proxy = self._hl_proxy = None
        self._heat = self._window = None
        self._snap_no = 0
        self._apply_theme()

    def _animation(self, ms, slot):
        a = QVariantAnimation(self)
        a.setDuration(ms)
        a.setEasingCurve(QEasingCurve.OutCubic)
        a.setStartValue(0.0)
        a.setEndValue(1.0)
        a.valueChanged.connect(slot)
        return a

    def attach(self, root):
        """Hook up the QML scene's graph objects, then start computing."""
        def find(name):
            obj = root.findChild(QObject, name)
            if obj is None:
                raise RuntimeError(f"QML object {name!r} not found")
            return obj

        self._window = root
        self._line, self._fill = find("curveLine"), find("curveFill")
        series = find("surfaceSeries")
        self._proxy = QtGraphs.QSurfaceDataProxy()
        series.setDataProxy(self._proxy)
        self._surface_series = series
        self._apply_surface_gradient()
        proxies = []
        for name in ("triadDots", "playDot", "hlDot"):
            s = find(name)
            proxy = s.dataProxy()
            if proxy is None:
                proxy = QtGraphs.QScatterDataProxy()
                s.setDataProxy(proxy)
            proxies.append(proxy)
        self._tri_proxy, self._play_proxy, self._hl_proxy = proxies
        self._heat = find("heatmapItem")
        self._heat.setTheme(self._theme)
        self._style_window()
        self._warm_note.start()
        self.worker.start()
        self.analyse()

    def shutdown(self):
        self.engine.close()
        self.worker.shutdown()
        if self.worker.isRunning():  # still compiling on a first launch
            self.worker.terminate()
            self.worker.wait(1000)
        if self._surface_renderer is not None:
            self._surface_renderer.close()

    def _voices(self):
        return (self.voice_a, self.voice_b, self.voice_c)

    # ── QML properties ─────────────────────────────────────────────────────

    theme = Property("QVariantMap", lambda s: s._theme, notify=themeChanged)
    dark = Property(bool, lambda s: s._dark, notify=themeChanged)
    accent = Property(str, lambda s: s._accent, notify=themeChanged)
    gradient = Property("QVariantList", lambda s: s._gradient, notify=themeChanged)
    monoFamily = Property(str, lambda s: s._mono, constant=True)
    iconFamily = Property(str, lambda s: s._icons, constant=True)
    voices = Property("QVariantList", lambda s: [s.voice_a, s.voice_b, s.voice_c], constant=True)
    intervalModel = Property(QObject, lambda s: s.intervals, constant=True)
    triadModel = Property(QObject, lambda s: s.triads, constant=True)
    figureSession = Property(QObject, lambda s: s.figure, constant=True)
    minimaModel = Property(QObject, lambda s: s.minima, constant=True)
    cameraDefault = Property("QVariantMap", lambda s: CAMERA_DEFAULT, constant=True)

    refText = Property(str, lambda s: s._ref_text, notify=settingsChanged)
    maxText = Property(str, lambda s: s._max_text, notify=settingsChanged)
    stepText = Property(str, lambda s: s._step_text, notify=settingsChanged)
    settingsError = Property(str, lambda s: s._settings_error, notify=settingsChanged)
    live = Property(bool, lambda s: s._live, notify=settingsChanged)

    volume = Property(float, lambda s: s._volume, notify=audioChanged)
    audioActive = Property(bool, lambda s: s._audio_active, notify=audioChanged)
    level = Property(float, lambda s: s._level, notify=levelChanged)

    statusText = Property(str, lambda s: s._status_text, notify=statusChanged)
    statusColor = Property(str, lambda s: s._role_color(s._status_role), notify=statusChanged)
    viewHint = Property(str, lambda s: VIEW_HINTS[s._view], notify=statusChanged)
    busyVisible = Property(bool, lambda s: s._busy_visible, notify=busyChanged)
    progress = Property(float, lambda s: s._progress, notify=busyChanged)

    curveMinima = Property("QVariantList", lambda s: s._curve_minima, notify=minimaChanged)
    curveVersion = Property(int, lambda s: s._curve_version, notify=curveChanged)
    curveXmin = Property(float, lambda s: float(s._curve_x[0]) if s._curve_x is not None else 0.9,
                         notify=curveChanged)
    curveXmax = Property(float, lambda s: float(s.curve["max_interval"]) if s.curve else 2.3,
                         notify=curveChanged)
    gridMin = Property(float, lambda s: s._grid[0], notify=surfaceChanged)
    gridMax = Property(float, lambda s: s._grid[1], notify=surfaceChanged)
    colorLo = Property(float, lambda s: float(s.data["z_3d"].min()) if s.data else 0.0,
                       notify=surfaceChanged)
    colorHi = Property(float, lambda s: float(s.data["z_3d"].max()) if s.data else 1.0,
                       notify=surfaceChanged)

    playing = Property("QVariantMap", lambda s: s._playing, notify=markersChanged)
    highlight = Property("QVariantMap", lambda s: s._highlight, notify=markersChanged)
    linkedInterval = Property(str, lambda s: s._linked_interval, notify=linkedChanged)
    linkedTriad = Property(str, lambda s: s._linked_triad, notify=linkedChanged)
    snap = Property("QVariantMap", lambda s: dict(kind="triad", b=s._snap[0], c=s._snap[1])
                    if s._snap is not None else dict(kind="", b=0.0, c=0.0), notify=markersChanged)
    inspector = Property("QVariantMap", lambda s: s._inspector, notify=inspectorChanged)
    overtones = Property("QVariantMap", lambda s: s._overtones, notify=overtonesChanged)

    def _role_color(self, role):
        return self._accent if role == "accent" else self._theme[role]

    def _status(self, text, role="accent", kind="info"):
        self._status_text, self._status_role, self._status_kind = text, role, kind
        self.statusChanged.emit()

    def _set_busy_visible(self, on):
        if on != self._busy_visible:
            self._busy_visible = on
            if on:
                self._progress = 0.0
            self.busyChanged.emit()

    # ── theme ──────────────────────────────────────────────────────────────

    def _apply_theme(self):
        self._theme = dict(QUICK_THEMES[self._dark])
        self._accent = QGuiApplication.palette().color(QPalette.ColorRole.Accent).name()
        self._gradient = _gradient_stops(self._theme["cmap"])
        for v in self._voices():
            v.setColor(self._theme[f"voice{v._letter}"])
        self.themeChanged.emit()
        self.statusChanged.emit()

    def _style_window(self):
        if self._window is not None:
            th = self._theme
            _title_bar(self._window, self._dark, th["window"], th["text"], th["border"])

    def _apply_surface_gradient(self):
        grad = QLinearGradient(0.0, 0.0, 1.0, 0.0)
        for i, color in enumerate(self._gradient):
            grad.setColorAt(i / (len(self._gradient) - 1), QColor(color))
        self._surface_series.setBaseGradient(grad)
        self._surface_series.setColorStyle(QtGraphs.QGraphsTheme.ColorStyle.RangeGradient)

    @Slot()
    def toggleTheme(self):
        self._dark = not self._dark
        QGuiApplication.styleHints().setColorScheme(
            Qt.ColorScheme.Dark if self._dark else Qt.ColorScheme.Light)
        # The palette (and its accent) follows the colour scheme a moment later.
        QTimer.singleShot(0, self._restyle)

    @Slot(result=str)
    def snapshot(self):
        """Grab the window for the theme cross-fade; returns its image URL."""
        self._snap_no += 1
        _FRAMES["snapshot"] = self._window.grabWindow() if self._window is not None else QImage()
        return f"image://frames/snapshot?{self._snap_no}"

    def _restyle(self):
        _FRAMES.pop("snapshot", None)  # QML has loaded it by now
        self._apply_theme()
        self._style_window()
        if self._surface_series is not None:
            self._apply_surface_gradient()
        if self._heat is not None:
            self._heat.setTheme(self._theme)
        self._show_overtones()
        if self._inspector_args is not None:
            getattr(self, self._inspector_args[0])(*self._inspector_args[1:])
        if self.curve is not None:
            self._fill_intervals()
            self._push_curve_labels()
        if self.data is not None:
            self._refresh_triads()

    # ── settings ───────────────────────────────────────────────────────────

    @Slot(str)
    def setRef(self, text):
        self._ref_text = text
        self._on_settings_changed()

    @Slot(str)
    def setMax(self, text):
        self._max_text = text
        self._on_settings_changed()

    @Slot(str)
    def setStep(self, text):
        self._step_text = text
        self._on_settings_changed()

    def _settings_values(self):
        values, problems = {}, []
        for key, text, test, message in (
                ("ref", self._ref_text, lambda v: v > 0, "Fundamental must be above 0 Hz"),
                ("max", self._max_text, lambda v: v > 1.0, "Max ratio must be above 1"),
                ("step", self._step_text, lambda v: v > 0, "3D step must be above 0")):
            try:
                v = float(text)
                ok = math.isfinite(v) and test(v)
            except ValueError:
                ok, message = False, "Settings must be numbers"
            if ok:
                values[key] = v
            else:
                problems.append(message)
        if not problems:
            n = math.ceil((values["max"] - 1.0) / values["step"] - 1e-9)
            if n > MAX_GRID:
                problems.append(f"3D grid would be {n:,} \u00d7 {n:,} points (limit "
                                f"{MAX_GRID:,}) \u2014 raise 3D STEP or lower MAX RATIO")
            elif n < 3:
                problems.append("MAX RATIO is too close to 1 for this 3D STEP")
        self._settings_error = problems[0] if problems else ""
        self.settingsChanged.emit()
        return values, problems

    def _read_settings(self):
        values, problems = self._settings_values()
        if problems:
            self._status(problems[0], "red", "input-error")
            return False
        self.ref_freq, self.max_interval, self.step_3d = values["ref"], values["max"], values["step"]
        return True

    def _on_settings_changed(self):
        _, problems = self._settings_values()
        if not problems and self._live:
            self._debounce.start(350)

    # ── analysis ───────────────────────────────────────────────────────────

    @Slot()
    def analyse(self):
        self._debounce.stop()
        self._submit(reset=True)

    @Slot(bool)
    def setLive(self, on):
        self._live = bool(on)
        self.settingsChanged.emit()
        if on:
            self._debounce.start(0)

    def _on_inputs_changed(self):
        if self._live:
            self._debounce.start(0 if self._dragging else 200)

    def _on_dragging(self, active):
        self._dragging = active
        # Drag-time jobs skip the heatmap contours; one more pass draws them.
        if not active and self._live:
            self._debounce.start(0)

    def _submit(self, reset=False):
        if not self._read_settings():
            return
        spectra = []
        for v in self._voices():
            spec, err = v.parsed()
            if spec is None:
                self._status(f"Voice {v._letter}: {err}", "red", "input-error")
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
            self._set_busy_visible(True)
            self._status("Computing dissonance space\u2026", "yellow", "busy")

    def _note_compiling(self):
        if not self._ready:
            self._set_busy_visible(True)
            self._status("Compiling the dissonance engine \u2014 first launch only, cached "
                         "afterwards\u2026", "yellow", "busy")

    def _on_ready(self):
        self._ready = True
        self._warm_note.stop()

    def _on_progress(self, gen, fraction):
        if gen == self._gen and self._busy_visible:
            self._progress = float(fraction)
            self.busyChanged.emit()

    def _on_stage(self, gen, name, payload):
        if name == "contours":
            if gen == self._shown_gen:
                self._contours = payload["contours"]
                if self._view == 2 and 2 not in self._dirty:
                    self._heat.setContours(self._contours)
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
            self._push_curve(payload, animate=not first and not self._dragging)
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
        self._set_busy_visible(False)
        if self._status_kind in ("busy", "input-error", "analysis-error"):
            self._status("Ready \u2014 click any plot or list entry to hear it", "green")

    def _show_surface_early(self):
        """Slow triad search (no Numba): show the surface before its markers arrive."""
        p = self._partial
        if "x_3d" in p and "tri_x" not in p:
            empty = np.array([])
            self._view3d = (p["x_3d"], p["y_3d"], p["z_3d"], empty, empty, empty)
            self._contours = None
            self._dirty = {1, 2}
            self._ensure_view(self._view)

    def _on_failed(self, gen, message):
        if gen != self._gen:
            return
        self._busy = False
        self._slow.stop()
        self._set_busy_visible(False)
        self._status(f"Computation failed: {message}", "red", "analysis-error")
        self.toast.emit(f"Computation failed: {message}", "error", "")

    # ── curve ──────────────────────────────────────────────────────────────

    def _push_curve(self, payload, animate):
        r, d = payload["ratios_2d"], payload["diss_2d"]
        same = (self._curve_shown is not None and len(self._curve_x) == len(r)
                and np.isclose(self._curve_x[0], r[0]) and np.isclose(self._curve_x[-1], r[-1]))
        self._curve_x, self._curve_y = r, d
        if animate and same:
            self._morph_from, self._morph_to = self._curve_shown.copy(), d
            self._morph.stop()
            self._morph.start()
        else:
            self._morph.stop()
            self._set_curve_points(d)
        self._push_curve_labels()
        (self.curveReset if self._job_reset else self.curveRefit).emit()

    def _push_curve_labels(self):
        c = self.curve
        self._curve_minima = [dict(x=float(x), y=float(y), key=_interval_key(x),
                                   label=f"{ratio_to_interval_name(x)}\n{x:.3f}")
                              for x, y in zip(c["minima_x"], c["minima_y"])]
        self.minima.setRows([dict(rx=m["x"], ry=m["y"], label=m["label"], key=m["key"])
                             for m in self._curve_minima])
        self.minimaChanged.emit()

    def _set_curve_points(self, y):
        self._curve_shown = y
        if self._line is not None:
            self._line.replaceNp(self._curve_x, y)
            self._fill.replaceNp(self._curve_x, y)
        self._curve_version += 1
        self.curveChanged.emit()

    def _morph_step(self, t):
        if self._morph_from is not None:
            self._set_curve_points(self._morph_from + (self._morph_to - self._morph_from) * float(t))

    @Slot(float, float, result=float)
    def visibleTop(self, x0, x1):
        if self._curve_y is None:
            return 1.0
        vis = self._curve_y[(self._curve_x >= x0) & (self._curve_x <= x1)]
        return float(vis.max()) if len(vis) else float(self._curve_y.max())

    @Slot(float, result=float)
    def curveY(self, x):
        if self._curve_shown is None:
            return 0.0
        return float(np.interp(x, self._curve_x, self._curve_shown))

    @Slot(float, bool, result="QVariantMap")
    def curveReadout(self, x, snapped):
        if self._curve_y is None:
            return {}
        th = self._theme
        if snapped:
            m = min(self._curve_minima, key=lambda m: abs(m["x"] - x))
            x, y = m["x"], m["y"]
        else:
            i = int(np.argmin(np.abs(self._curve_x - x)))
            x, y = float(self._curve_x[i]), float(self._curve_y[i])
        jr = just_ratio(x)
        just = (f"<br><span style='color:{th['muted']}'>\u2248 {jr[0]}/{jr[1]}  "
                f"{_cents_text(jr[2])}</span>" if jr and abs(jr[2]) <= 12 else "")
        html = (f"<b style='color:{self._accent}'>{x:.4f}</b>&nbsp;&nbsp;"
                f"<span style='color:{th['muted']}'>{ratio_to_cents(x):.0f}\u00a2</span><br>"
                f"<span style='color:{th['subtext']}'>{ratio_to_interval_name(x)}</span>"
                f"{just}<br><span style='color:{_diss_hex(y, th)}'>diss {y:.3f}</span>")
        return dict(x=x, y=y, html=html)

    @Slot(float, float, result="QVariantMap")
    def heatmapReadout(self, b, c):
        d = self._diss_at_triad(b, c)
        if d is None:
            return {}
        th = self._theme
        prop = triad_proportion(b, c, max_base=16, tol_cents=8)
        chord = (f"&nbsp;&nbsp;<span style='color:{th['muted']}'>\u2248 {prop[0]}:{prop[1]}:"
                 f"{prop[2]}</span>" if prop else "")
        html = (f"<b style='color:{th['voiceB']}'>B {b:.3f}</b>&nbsp;&nbsp;"
                f"<b style='color:{th['voiceC']}'>C {c:.3f}</b><br>"
                f"<span style='color:{th['subtext']}'>{ratio_to_interval_name(b)} \u00b7 "
                f"{ratio_to_interval_name(c)}</span><br>"
                f"<span style='color:{_diss_hex(d, th)}'>diss {d:.3f}</span>{chord}")
        return dict(html=html)

    # ── 3D views ───────────────────────────────────────────────────────────

    @Slot(int)
    def setViewIndex(self, index):
        self._view = index
        self._ensure_view(index)
        self.statusChanged.emit()

    def _refresh_triads(self, animate=False):
        if self.data is None:
            return
        tx, ty, tz = self._visible_triads()
        d = self.data
        self._view3d = (d["x_3d"], d["y_3d"], d["z_3d"], tx, ty, tz)
        self._dirty = {1, 2}
        self._ensure_view(self._view, animate=animate)
        th = self._theme
        rows = [dict(primary=f"1.000 : {bx:.3f} : {cy:.3f}",
                     secondary=f"{ratio_to_interval_name(bx)}  \u00b7  "
                               f"{ratio_to_interval_name(cy)}",
                     value=float(dz), tint=_diss_hex(dz, th), key=_triad_key(bx, cy),
                     b=float(bx), c=float(cy))
                for bx, cy, dz in zip(tx, ty, tz)]
        self.triads.setRows(rows)
        self.surfaceChanged.emit()

    def _ensure_view(self, index, animate=False):
        if index not in self._dirty or self._view3d is None:
            return
        self._dirty.discard(index)
        x, y, z, tx, ty, tz = self._view3d
        if index == 1:
            self._push_surface(x, y, z, tx, ty, tz, animate)
        elif index == 2:
            self._heat.setData(x, y, z, tx, ty)
            self._heat.setContours(self._contours)

    def _push_surface(self, x, y, z, tx, ty, tz, animate):
        # Dragging a partial refreshes many times a second; a lighter mesh keeps up.
        step = max(1, -(-len(x) // 130)) if self._dragging else max(1, -(-len(x) // 300))
        xs, zs = x[::step], y[::step]
        grid = np.ascontiguousarray(z[::step, ::step].T)
        same = (self._surf_shown is not None and self._surf_shown.shape == grid.shape
                and np.allclose(self._sx, xs) and np.allclose(self._sz, zs))
        self._sx, self._sz = xs, zs
        self._grid = (float(x[0]), float(x[-1]))
        if animate and same:
            self._surf_from, self._surf_to = self._surf_shown.copy(), grid
            self._surf_morph.stop()
            self._surf_morph.start()
        else:
            self._surf_morph.stop()
            self._set_surface_grid(grid)
        self._tri_proxy.resetArray([QtGraphs.QScatterDataItem(QVector3D(float(b), float(v) + 0.012,
                                                                        float(c)))
                                    for b, c, v in zip(tx, ty, tz)])
        self._update_3d_markers()
        self.surfaceChanged.emit()

    def _set_surface_grid(self, grid):
        self._surf_shown = grid
        xs, zs = self._sx, self._sz
        self._proxy.resetArrayNp(float(xs[0]), float(xs[1] - xs[0]), float(zs[0]),
                                 float(zs[1] - zs[0]), grid)

    def _surf_morph_step(self, t):
        if self._surf_from is not None:
            self._set_surface_grid(np.ascontiguousarray(
                self._surf_from + (self._surf_to - self._surf_from) * float(t)))

    def _update_3d_markers(self):
        if self._play_proxy is None:
            return
        hl = (dict(kind="triad", b=self._snap[0], c=self._snap[1]) if self._snap is not None
              else self._highlight)
        for proxy, m in ((self._play_proxy, self._playing), (self._hl_proxy, hl)):
            d = self._diss_at_triad(m["b"], m["c"]) if m["kind"] == "triad" else None
            proxy.resetArray([] if d is None else
                             [QtGraphs.QScatterDataItem(QVector3D(m["b"], d + 0.02, m["c"]))])

    def _shown_triad(self, k):
        """Best triad *k* in the order of the 3D markers, or None."""
        if self._view3d is None or not 0 <= k < len(self._view3d[3]):
            return None
        return float(self._view3d[3][k]), float(self._view3d[4][k])

    def _on_grid(self, row, col):
        return self._sx is not None and 0 <= col < len(self._sx) and 0 <= row < len(self._sz)

    @Slot(int, int)
    def surfacePicked(self, row, col):
        if self._on_grid(row, col):
            self.playTriad(float(self._sx[col]), float(self._sz[row]))

    @Slot(int, int, result="QVariantMap")
    def surfaceReadout(self, row, col):
        if not self._on_grid(row, col):
            return {}
        out = self.heatmapReadout(float(self._sx[col]), float(self._sz[row]))
        out["key"] = ""
        return out

    @Slot(int, result="QVariantMap")
    def triadReadout(self, k):
        """Readout for the 3D marker *k* the pointer snapped to."""
        bc = self._shown_triad(k)
        if bc is None:
            return {}
        out = self.heatmapReadout(*bc)
        out["key"] = _triad_key(*bc)
        return out

    @Slot(int)
    def playShownTriad(self, k):
        bc = self._shown_triad(k)
        if bc is not None:
            self.playTriad(*bc)

    @Slot(str)
    def setSnapTriad(self, key):
        """The triad the pointer has snapped to on the heatmap or surface ("" for none)."""
        snap = tuple(float(v) for v in key.split("|")) if key else None
        if snap == self._snap:
            return
        self._snap = snap
        self._linked_triad = key
        self.linkedChanged.emit()
        self._update_3d_markers()
        self.markersChanged.emit()

    # ── results ────────────────────────────────────────────────────────────

    @Slot(int)
    def setSortMode(self, mode):
        self._sort_mode = mode
        self._fill_intervals()

    @Slot(int)
    def setTriadMode(self, mode):
        self._triad_mode = mode
        self._refresh_triads()

    def _view_xmax(self):
        return self.curve["max_interval"] if self.curve is not None else self.max_interval

    def _fill_intervals(self):
        if self.curve is None:
            return
        xmax = self._view_xmax()
        pairs = [(x, y) for x, y in zip(self.curve["minima_x"], self.curve["minima_y"]) if x <= xmax]
        pairs.sort(key=lambda p: p[1] if self._sort_mode == 0 else p[0])
        th = self._theme
        self.intervals.setRows([dict(primary=f"{x:.4f}",
                                     secondary=f"{ratio_to_cents(x):.0f}\u00a2  \u00b7  "
                                               f"{ratio_to_interval_name(x)}",
                                     value=float(y), tint=_diss_hex(y, th), key=_interval_key(x),
                                     b=float(x), c=0.0)
                                for x, y in pairs])

    def _visible_triads(self, limit=10):
        bx, cy, dz = self.data["tri_x"], self.data["tri_y"], self.data["tri_z"]
        if self._triad_mode == 0:
            # 20-cent guard: anything nearer than this just reads as an octave.
            octave = 2.0 * 2 ** (-20 / 1200)
            keep = (bx < octave) & (cy < octave)
            bx, cy, dz = bx[keep], cy[keep], dz[keep]
        order = np.argsort(dz)[:limit]
        return bx[order], cy[order], dz[order]

    @Slot(str)
    def hoverInterval(self, key):
        self._highlight = (dict(kind="interval", x=float(key), b=0.0, c=0.0) if key
                           else dict(kind="", x=0.0, b=0.0, c=0.0))
        self.markersChanged.emit()

    @Slot(str)
    def hoverTriad(self, key):
        if key:
            b, c = (float(v) for v in key.split("|"))
            self._highlight = dict(kind="triad", x=0.0, b=b, c=c)
        else:
            self._highlight = dict(kind="", x=0.0, b=0.0, c=0.0)
        self._update_3d_markers()
        self.markersChanged.emit()

    @Slot(str)
    def setLinkedInterval(self, key):
        if key != self._linked_interval:
            self._linked_interval = key
            self.linkedChanged.emit()

    # ── playback ───────────────────────────────────────────────────────────

    def _sound(self, voices, label, letters="ABC", marker=None):
        """Play *voices*, show their overtones, and remember them for WAV export."""
        self._last_sound = (voices, label, letters, marker)
        self._shown_overtones = [(letters[i], spec, ratio) for i, (spec, ratio) in enumerate(voices)]
        matches = self._show_overtones(animate=True)
        if self.engine.play(synthesize_multi(voices, self.ref_freq)):
            self._set_sounding(marker)
        return matches

    def _show_overtones(self, animate=False):
        shown = self._shown_overtones
        if not shown:
            return 0
        th, ref = self._theme, self.ref_freq
        stems, all_x = [], []
        for k, (letter, spec, ratio) in enumerate(shown):
            f = ref * ratio * np.asarray(spec["freq"], dtype=float)
            a = np.abs(np.asarray(spec["amp"], dtype=float))
            a = a / a.max() if a.size and a.max() > 0 else a
            for fi, ai in zip(f, a):
                x = math.log2(fi / ref)
                all_x.append(x)
                stems.append(dict(x=x, h=float(ai), color=th[f"voice{letter}"],
                                  w=(6.0, 3.5, 1.8)[min(k, 2)]))
        rings = shared_partials(shown, ref)
        lo, hi = min(all_x), max(all_x)
        pad = max(0.08, 0.04 * (hi - lo))
        lo, hi = lo - pad, hi + pad
        octaves = range(int(math.floor(lo)), int(math.ceil(hi)) + 1)
        legend = "&nbsp;&nbsp;".join(f"<span style='color:{th['voice' + l]}'>\u25cf</span> "
                                     f"{l} {ratio:.3f}" for l, _, ratio in shown)
        if len(shown) > 1:
            legend += (f"&nbsp;&nbsp;\u00b7&nbsp;&nbsp;<span style='color:{th['text']}'>\u25ef"
                       f"</span> shared")
        self._overtones = dict(
            empty=False, xmin=lo, xmax=hi, stems=stems,
            rings=[dict(x=math.log2(f / ref), h=h) for f, h in rings],
            ticks=[dict(x=float(n), label=f"{ref * 2 ** n:.0f} Hz") for n in octaves if lo <= n <= hi],
            minor=[n + math.log2(h) for n in octaves for h in (1.25, 1.5, 1.75)
                   if lo <= n + math.log2(h) <= hi],
            legend=legend,
            info=f"{len(rings)} shared partials" if len(shown) > 1 else "single voice")
        self.overtonesChanged.emit()
        if animate:
            self.overtonesAnimate.emit()
        return len(rings)

    def _set_sounding(self, marker):
        kind = marker[0] if marker else ""
        self._playing = dict(kind=kind, x=float(marker[1]) if kind == "interval" else 0.0,
                             b=float(marker[1]) if kind == "triad" else 0.0,
                             c=float(marker[2]) if kind == "triad" else 0.0)
        self._update_3d_markers()
        self.markersChanged.emit()

    def _on_audio_activity(self, active):
        self._audio_active = active
        self.audioChanged.emit()

    def _on_level(self, rms):
        target = float(np.clip(rms * 2.6, 0.0, 1.0))
        self._level = target if target > self._level else self._level * 0.82 + target * 0.18
        self.levelChanged.emit()

    @Slot(float)
    def setVolume(self, value):
        self._volume = float(np.clip(value, 0.0, 1.0))
        self.engine.set_volume(self._volume ** 2)
        self.audioChanged.emit()

    @Slot()
    def toggleMute(self):
        if self._volume > 0:
            self._muted_volume = self._volume
            self.setVolume(0.0)
        else:
            self.setVolume(self._muted_volume or 0.8)

    @Slot()
    def stopAudio(self):
        self.engine.stop()

    def _play_voice(self, voice):
        if not self._read_settings():
            return
        spec, err = voice.parsed()
        if spec is None:
            self._status(f"Voice {voice._letter}: {err}", "red", "input-error")
            return
        self._status(f"Playing voice {voice._letter} alone", "green")
        self._sound([(spec, 1.0)], f"voice_{voice._letter}", letters=voice._letter)
        self._show_voice(voice._letter, voice.source_label())

    def _diss_at_ratio(self, ratio):
        if self.curve is None:
            return None
        i = int(np.argmin(np.abs(self.curve["ratios_2d"] - ratio)))
        return float(self.curve["diss_2d"][i])

    def _diss_at_triad(self, rb, rc):
        if self.data is None:
            return None
        bi = int(np.argmin(np.abs(self.data["x_3d"] - rb)))
        ci = int(np.argmin(np.abs(self.data["y_3d"] - rc)))
        return float(self.data["z_3d"][bi, ci])

    @Slot(float)
    def playInterval(self, ratio):
        if not self._read_settings():
            return
        spec_a, spec_b = self.voice_a.get_spectrum(), self.voice_b.get_spectrum()
        if spec_a is None or spec_b is None:
            self._status("Voice A or B has invalid input", "red", "input-error")
            return
        d = self._diss_at_ratio(ratio)
        tail = f" \u00b7 diss {d:.3f}" if d is not None else ""
        self._status(f"A 1.000 + B {ratio:.4f}  ({ratio_to_interval_name(ratio)}){tail}",
                     "green")
        matches = self._sound([(spec_a, 1.0), (spec_b, ratio)], f"interval_{ratio:.4f}",
                              marker=("interval", ratio))
        self._show_interval(ratio, d, matches)

    @Slot(float, float)
    def playTriad(self, rb, rc):
        if not self._read_settings():
            return
        specs = [v.get_spectrum() for v in self._voices()]
        if any(s is None for s in specs):
            self._status("One of the voices has invalid input", "red", "input-error")
            return
        d = self._diss_at_triad(rb, rc)
        tail = f" \u00b7 diss {d:.3f}" if d is not None else ""
        self._status(f"A 1.000 + B {rb:.3f} + C {rc:.3f}{tail}", "green")
        matches = self._sound(list(zip(specs, (1.0, rb, rc))), f"triad_{rb:.3f}_{rc:.3f}",
                              marker=("triad", rb, rc))
        self._show_triad(rb, rc, d, matches)

    @Slot(str)
    def playNote(self, text):
        try:
            ratio = float(text)
        except ValueError:
            self._status("Note must be a single number", "red", "input-error")
            return
        if not (math.isfinite(ratio) and ratio > 0):
            self._status("Note ratio must be positive", "red", "input-error")
            return
        if not self._read_settings():
            return
        spec = self.voice_a.get_spectrum()
        if spec is None:
            self._status("Voice A has invalid input", "red", "input-error")
            return
        self._status(f"A {ratio:.4f} alone  ({ratio_to_interval_name(ratio)})", "green")
        self._sound([(spec, ratio)], f"note_{ratio:.4f}")
        self._show_note(ratio)

    @Slot(str)
    def playIntervalText(self, text):
        try:
            ratio = float(text)
        except ValueError:
            self._status("Interval must be a single number", "red", "input-error")
            return
        if not (math.isfinite(ratio) and ratio > 0):
            self._status("Interval ratio must be positive", "red", "input-error")
            return
        self.playInterval(ratio)

    @Slot(str)
    def playTriadText(self, text):
        try:
            parts = [float(p.strip()) for p in text.split(',')]
        except ValueError:
            self._status("Triad must be two numbers, e.g. 1.25, 1.5", "red", "input-error")
            return
        if len(parts) != 2:
            self._status("Enter exactly two ratios (B, C)", "red", "input-error")
            return
        if not all(math.isfinite(p) and p > 0 for p in parts):
            self._status("Triad ratios must be positive", "red", "input-error")
            return
        self.playTriad(parts[0], parts[1])

    @Slot()
    def replay(self):
        if self._last_sound is None:
            self._status("Nothing to replay yet \u2014 play something first", "red", "input-error")
            return
        voices, label, letters, marker = self._last_sound
        self._sound(voices, label, letters, marker)
        self._status(f"Replaying {label.replace('_', ' ')}", "green")

    # ── glide ──────────────────────────────────────────────────────────────

    @Slot(str, float, float)
    def glideBegin(self, kind, a, b):
        if not self._read_settings():
            return
        specs = [v.get_spectrum() for v in self._voices()]
        if kind == "interval":
            specs, letters, values = specs[:2], "AB", [1.0, a]
        else:
            letters, values = "ABC", [1.0, a, b]
        if any(s is None for s in specs):
            self._status("A voice has invalid input", "red", "input-error")
            return
        if not self.engine.glide_begin(list(zip(specs, values)), self.ref_freq):
            return
        self._glide = dict(kind=kind, specs=specs, letters=letters, values=values)
        self._glide_ui()

    @Slot(float, float)
    def glideMove(self, a, b):
        g = self._glide
        if g is None:
            return
        g["values"] = [1.0, a] if g["kind"] == "interval" else [1.0, a, b]
        self.engine.glide_to(g["values"])
        if not self._glide_tick.isActive():
            self._glide_tick.start()

    @Slot()
    def glideEnd(self):
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
                         "green")
            self._set_sounding(("interval", r))
            self._show_interval(r, d, matches)
        else:
            rb, rc = vals[1], vals[2]
            d = self._diss_at_triad(rb, rc)
            tail = f" \u00b7 diss {d:.3f}" if d is not None else ""
            self._status(f"{verb}  A 1.000 + B {rb:.3f} + C {rc:.3f}{tail}", "green")
            self._set_sounding(("triad", rb, rc))
            self._show_triad(rb, rc, d, matches)

    # ── inspector ──────────────────────────────────────────────────────────

    def _set_inspector(self, kind, big, rows, diss, args):
        th = self._theme
        self._inspector_args = args
        self._inspector = dict(empty=False, kind=kind, big=big,
                               rows=[dict(k=k, v=v) for k, v in rows],
                               diss=-1.0 if diss is None else float(diss),
                               dissText="\u2014" if diss is None else f"{diss:.3f}",
                               dissColor=th["muted"] if diss is None else _diss_hex(diss, th))
        self.inspectorChanged.emit()

    def _just_html(self, ratio):
        jr = just_ratio(ratio)
        muted = self._theme["muted"]
        if jr is None or abs(jr[2]) > 12:
            return f"<span style='color:{muted}'>none within 12\u00a2</span>"
        return f"{jr[0]}/{jr[1]}&nbsp;&nbsp;<span style='color:{muted}'>{_cents_text(jr[2])}</span>"

    def _show_interval(self, ratio, diss, shared):
        self._set_inspector("INTERVAL", f"1 : {ratio:.4f}", [
            ("Cents", f"{ratio_to_cents(ratio):.1f}\u00a2"),
            ("12-TET", ratio_to_interval_name(ratio)),
            ("Just ratio", self._just_html(ratio)),
            ("Shared partials", str(shared))], diss, ("_show_interval", ratio, diss, shared))

    def _show_triad(self, rb, rc, diss, shared):
        prop = triad_proportion(rb, rc)
        muted = self._theme["muted"]
        just = (f"{prop[0]} : {prop[1]} : {prop[2]}&nbsp;&nbsp;<span style='color:{muted}'>"
                f"{_cents_text(prop[3])}, {_cents_text(prop[4])}</span>" if prop
                else f"<span style='color:{muted}'>no small whole-number chord</span>")
        self._set_inspector("TRIAD", f"1 : {rb:.3f} : {rc:.3f}", [
            ("Cents", f"{ratio_to_cents(rb):.0f}\u00a2 \u00b7 {ratio_to_cents(rc):.0f}\u00a2"),
            ("12-TET", f"{ratio_to_interval_name(rb)} \u00b7 {ratio_to_interval_name(rc)}"),
            ("Just ratio", just),
            ("Shared partials", str(shared))], diss, ("_show_triad", rb, rc, diss, shared))

    def _show_note(self, ratio):
        self._set_inspector("NOTE", f"A \u00d7 {ratio:.4f}", [
            ("Cents", f"{ratio_to_cents(ratio):.1f}\u00a2"),
            ("12-TET", ratio_to_interval_name(ratio)),
            ("Just ratio", self._just_html(ratio)),
            ("Shared partials", "\u2014")], None, ("_show_note", ratio))

    def _show_voice(self, letter, source):
        self._set_inspector(f"VOICE {letter}", f"{letter} alone", [
            ("Cents", "0\u00a2"), ("12-TET", "unison"),
            ("Spectrum", source), ("Shared partials", "\u2014")], None,
            ("_show_voice", letter, source))

    # ── export ─────────────────────────────────────────────────────────────

    @Slot(int, float, float, float, float, float)
    def exportFigure(self, view, x0, x1, cam_x, cam_y, cam_zoom):
        if self.data is None:
            self._status("Nothing to export yet \u2014 run an analysis first", "red", "input-error")
            return
        kind = ("curve", "surface", "heatmap")[view]
        if kind == "surface" and not HAS_GL:
            self._status("The 3D export needs PyOpenGL", "red", "input-error")
            return
        st = dict(SURFACE_DEFAULTS) if kind == "surface" else {
            **FIGURE_COMMON, **FIGURE_DEFAULTS[kind]}
        st.update(self._fig_state.get(kind, {}))
        data = self.data
        renderer = None
        if kind == "curve":
            r = data["ratios_2d"]
            st["xmin"] = round(max(float(x0), float(r[0])), 3)
            st["xmax"] = round(min(float(x1), float(r[-1])), 3)
            st["ymax"] = 0.0
        else:
            data = dict(data, tri_vis=self._visible_triads())
        if kind == "surface":
            st.update(azimuth=(270.0 - float(cam_x)) % 360.0, elevation=float(cam_y),
                      distance=SURFACE_DEFAULTS["distance"] * 100.0 / max(cam_zoom, 1.0))
            if self._surface_renderer is None:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                try:
                    self._surface_renderer = SurfaceRenderer(data)
                finally:
                    QApplication.restoreOverrideCursor()
            self._surface_renderer.data = data
            renderer = self._surface_renderer
        dpr = self._window.devicePixelRatio() if self._window is not None else 1.0
        self.figure.begin(kind, data, st, renderer, dpr)

    def _figure_closed(self, kind, state, path):
        # Text and style carry over to the next figure; ranges and camera follow the view.
        skip = ("xmin", "xmax", "ymax") + SURFACE_CAMERA_KEYS
        self._fig_state[kind] = {k: v for k, v in state.items() if k not in skip}
        if path:
            self._status(f"Figure saved to {Path(path).name}", "green")
            self._toast_with_action(f"Figure saved to {Path(path).name}", "ok", "Show in folder",
                                    lambda: reveal_in_folder(path))

    @Slot(str)
    def saveWav(self, length_text):
        if self._last_sound is None:
            self._status("Play something first, then save it", "red", "input-error")
            return
        if not self._read_settings():
            return
        try:
            seconds = float(length_text)
        except ValueError:
            self._status("Length must be a number of seconds", "red", "input-error")
            return
        if not 0.05 <= seconds <= 600:
            self._status("Length must be between 0.05 and 600 seconds", "red", "input-error")
            return
        voices, label = self._last_sound[0], self._last_sound[1]
        path, _ = QFileDialog.getSaveFileName(None, "Save sound",
                                              _safe_filename(label) + ".wav",
                                              "WAV audio (*.wav)")
        if not path:
            return
        if not path.lower().endswith(".wav"):
            path += ".wav"
        try:
            write_wav(path, synthesize_multi(voices, self.ref_freq, duration=seconds))
        except Exception as exc:
            self._status(f"Could not write the file: {exc}", "red", "input-error")
            self.toast.emit(f"Could not write the file: {exc}", "error", "")
            return
        message = f"Saved {seconds:g}\u202fs to {Path(path).name}"
        self._status(message, "green")
        self._toast_with_action(message, "ok", "Show in folder",
                                lambda: reveal_in_folder(path))

    def _toast_with_action(self, text, kind, label, action):
        self._toast_action = action
        self.toast.emit(text, kind, label)

    @Slot()
    def toastAction(self):
        if self._toast_action is not None:
            self._toast_action()


# ═══════════════════════════════════════════════════════════════════════════
#  QML interface
# ═══════════════════════════════════════════════════════════════════════════

QML = r"""
import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs
import QtQuick.Layouts
import QtQuick.Effects
import QtGraphs
import Dissonance

ApplicationWindow {
    id: win
    width: 1680
    height: 980
    minimumWidth: 1180
    minimumHeight: 720
    title: "Dissonance Quick"
    color: backend.theme.window

    readonly property var viewTitles: ["Dissonance curve", "Dissonance surface", "Triad map"]
    readonly property var viewSubs: ["voice A fixed \u00b7 voice B swept",
                                     "voice A fixed \u00b7 voices B and C swept",
                                     "voice A fixed \u00b7 B across \u00b7 C up"]

    function exportFigure() {
        backend.exportFigure(viewTabs.currentIndex, curvePanel.xMin, curvePanel.xMax,
                             surfacePanel.camX, surfacePanel.camY, surfacePanel.camZoom)
    }
    function resetView() {
        if (viewTabs.currentIndex === 0) curvePanel.resetView()
        else if (viewTabs.currentIndex === 1) surfacePanel.resetView()
        else heatmapPanel.resetView()
    }
    function switchTheme() {
        // Cross-fade from a snapshot of the old theme.
        themeSnap.source = backend.snapshot()
        themeSnap.opacity = 1
        themeSnap.visible = true
        backend.toggleTheme()
        themeFade.restart()
    }

    Shortcut { sequences: ["Ctrl+Return", "Ctrl+Enter"]; onActivated: backend.analyse() }
    Shortcut { sequence: "Esc"; onActivated: backend.stopAudio() }
    Shortcut { sequence: "Space"; onActivated: backend.replay() }

    // ── building blocks ─────────────────────────────────────────────────

    component Card: Rectangle {
        radius: 8
        color: backend.theme.card
        border.color: backend.theme.border
        border.width: 1
    }

    component Caption: Label {
        font.pixelSize: 11
        font.weight: Font.DemiBold
        font.letterSpacing: 1.1
        font.capitalization: Font.AllUppercase
        color: backend.theme.muted
    }

    component Segmented: Rectangle {
        id: seg
        property var labels: []
        property var colors: []
        property int currentIndex: 0
        signal activated(int index)
        readonly property real segW: (width - 8) / Math.max(1, labels.length)
        implicitHeight: 34
        radius: 8
        color: backend.theme.inset
        border.color: backend.theme.border
        Rectangle {
            x: 4 + seg.currentIndex * seg.segW
            y: 4
            width: seg.segW
            height: seg.height - 8
            radius: 6
            color: backend.theme.raised
            border.color: backend.theme.border
            Behavior on x { NumberAnimation { duration: 260; easing.type: Easing.OutCubic } }
        }
        Repeater {
            model: seg.labels
            delegate: MouseArea {
                id: segItem
                required property int index
                required property string modelData
                x: 4 + index * seg.segW
                y: 4
                width: seg.segW
                height: seg.height - 8
                hoverEnabled: true
                cursorShape: Qt.PointingHandCursor
                onClicked: if (seg.currentIndex !== index) { seg.currentIndex = index; seg.activated(index) }
                Label {
                    anchors.centerIn: parent
                    text: segItem.modelData
                    font.weight: Font.DemiBold
                    readonly property bool custom: seg.colors.length > segItem.index
                    readonly property color own: custom ? seg.colors[segItem.index] : backend.accent
                    color: seg.currentIndex === segItem.index ? own
                         : segItem.containsMouse ? backend.theme.text
                         : custom ? Qt.rgba(own.r, own.g, own.b, 0.65) : backend.theme.muted
                    Behavior on color { ColorAnimation { duration: 180 } }
                }
            }
        }
    }

    component SpectrumEditor: Rectangle {
        id: ed
        property var voice
        property bool dragging: false
        property real frozenX: 2.4
        property real frozenY: 1.15
        property real f0: 1
        property int hot: -1
        property real grow: 1
        // One conversion per change, shared by every handle's bindings.
        readonly property var freqs: voice ? voice.freqs : []
        readonly property var amps: voice ? voice.amps : []
        readonly property real xmax: dragging ? frozenX : Math.max(2.4, maxOf(freqs) * 1.12)
        readonly property real ymax: dragging ? frozenY : Math.max(1.0, maxAbs(amps)) * 1.15
        readonly property real pl: 12
        readonly property real pt: 16
        readonly property real pw: width - 26
        readonly property real ph: height - 36
        readonly property int gridStep: {
            var s = 1
            while (pw > 0 && pw / xmax * s < 18 && s < 1024) s *= 2
            return s
        }
        function maxOf(a) { var m = 0; for (var i = 0; i < a.length; ++i) m = Math.max(m, a[i]); return m }
        function maxAbs(a) { var m = 0; for (var i = 0; i < a.length; ++i) m = Math.max(m, Math.abs(a[i])); return m }
        function px(f) { return pl + f / xmax * pw }
        function py(a) { return pt + ph - Math.max(a, 0) / ymax * ph }
        function fx(x) { return (x - pl) / pw * xmax }
        function ay(y) { return (pt + ph - y) / ph * ymax }
        implicitHeight: 118
        radius: 8
        color: backend.theme.inset
        border.color: backend.theme.border

        Connections { target: ed.voice; function onPresetApplied() { growAnim.restart() } }
        NumberAnimation { id: growAnim; target: ed; property: "grow"; from: 0; to: 1; duration: 360; easing.type: Easing.OutCubic }
        HoverHandler { id: edHover }
        ToolTip.visible: edHover.hovered && ed.hot < 0 && !ed.dragging
        ToolTip.delay: 1000
        ToolTip.text: "Drag a partial to retune or re-weight it\nShift: amplitude only  \u00b7  Ctrl: snap to a whole harmonic\nDouble-click to add a partial  \u00b7  right-click one to remove it"

        Repeater {
            model: Math.floor(ed.xmax / ed.gridStep)
            delegate: Item {
                required property int index
                readonly property int k: (index + 1) * ed.gridStep
                Rectangle { x: ed.px(parent.k); y: ed.pt; width: 1; height: ed.ph; color: backend.theme.border }
                Label {
                    x: ed.px(parent.k) - 14; y: ed.pt + ed.ph + 3; width: 28
                    horizontalAlignment: Text.AlignHCenter
                    text: parent.k; font.pixelSize: 9; color: backend.theme.muted
                }
            }
        }
        Rectangle { x: ed.pl; y: ed.pt + ed.ph; width: ed.pw; height: 1; color: backend.theme.muted; opacity: 0.6 }

        MouseArea {
            anchors.fill: parent
            onDoubleClicked: (m) => {
                if (m.x >= ed.pl && m.x <= ed.pl + ed.pw && m.y >= ed.pt && m.y <= ed.pt + ed.ph)
                    ed.voice.addPartial(ed.fx(m.x), ed.ay(m.y))
            }
        }

        Repeater {
            model: ed.voice ? ed.voice.count : 0
            delegate: Item {
                id: part
                required property int index
                readonly property real f: ed.freqs[index] !== undefined ? ed.freqs[index] : 0
                readonly property real a: ed.amps[index] !== undefined ? Math.abs(ed.amps[index]) : 0
                readonly property bool isHot: ed.hot === index
                readonly property real hx: ed.px(f)
                readonly property real hy: ed.py(a * ed.grow)
                Rectangle {
                    x: part.hx - width / 2
                    y: part.hy
                    width: part.isHot ? 2.8 : 2
                    height: Math.max(0, ed.pt + ed.ph - part.hy)
                    radius: 1
                    color: ed.voice.color
                    opacity: part.isHot ? 0.95 : 0.78
                }
                Rectangle {
                    width: 20; height: 20; radius: 10
                    x: part.hx - 10; y: part.hy - 10
                    color: ed.voice.color
                    opacity: part.isHot ? 0.22 : 0
                    Behavior on opacity { NumberAnimation { duration: 120 } }
                }
                Rectangle {
                    width: 10; height: 10; radius: 5
                    x: part.hx - 5; y: part.hy - 5
                    color: ed.voice.color
                    border.color: backend.theme.card
                    border.width: 1.5
                    scale: part.isHot ? 1.4 : 1
                    Behavior on scale { NumberAnimation { duration: 120 } }
                    MouseArea {
                        anchors.fill: parent
                        anchors.margins: -7
                        hoverEnabled: true
                        preventStealing: true
                        acceptedButtons: Qt.LeftButton | Qt.RightButton
                        cursorShape: pressed ? Qt.ClosedHandCursor : Qt.OpenHandCursor
                        onEntered: ed.hot = part.index
                        onExited: if (!pressed) ed.hot = -1
                        onPressed: (m) => {
                            if (m.button === Qt.RightButton) {
                                var i = part.index
                                ed.hot = -1
                                Qt.callLater(function() { ed.voice.removePartial(i) })
                                return
                            }
                            ed.frozenX = ed.xmax
                            ed.frozenY = ed.ymax
                            ed.f0 = part.f
                            ed.dragging = true
                            ed.voice.beginDrag()
                        }
                        onPositionChanged: (m) => {
                            if (!pressed || !ed.dragging) return
                            var p = mapToItem(ed, m.x, m.y)
                            var f = Math.min(Math.max(ed.fx(p.x), 0.01), ed.xmax)
                            var a = Math.min(Math.max(ed.ay(p.y), 0), ed.ymax)
                            if (m.modifiers & Qt.ShiftModifier) f = ed.f0
                            else if (m.modifiers & Qt.ControlModifier) f = Math.max(1, Math.round(f))
                            ed.voice.dragPartial(part.index, f, a)
                        }
                        onReleased: {
                            if (ed.dragging) { ed.dragging = false; ed.voice.endDrag() }
                            if (!containsMouse) ed.hot = -1
                        }
                    }
                }
            }
        }

        Rectangle {
            readonly property string info: ed.hot >= 0 && ed.hot < ed.voice.count
                ? "#" + (ed.hot + 1) + "   " + Number(ed.freqs[ed.hot]).toPrecision(4)
                  + "\u00d7   amp " + Number(ed.amps[ed.hot]).toPrecision(3)
                : (ed.voice.linkedTo ? "same as " + ed.voice.linkedTo + "  \u00b7  drag to detach" : "")
            visible: info.length > 0
            anchors.top: parent.top
            anchors.right: parent.right
            anchors.margins: 5
            width: edInfo.implicitWidth + 14
            height: edInfo.implicitHeight + 6
            radius: 6
            color: backend.theme.card
            opacity: 0.94
            Label {
                id: edInfo
                anchors.centerIn: parent
                text: parent.info
                font.family: backend.monoFamily
                font.pixelSize: 11
                color: ed.hot >= 0 ? backend.theme.text : backend.theme.muted
            }
        }
        Label { anchors.centerIn: parent; visible: !ed.voice.valid; text: "Invalid spectrum"; color: backend.theme.red }
    }

    component VoicePane: ColumnLayout {
        id: pane
        property var voice
        spacing: 8
        RowLayout {
            Layout.fillWidth: true
            spacing: 6
            ComboBox {
                id: combo
                Layout.fillWidth: true
                model: pane.voice.entries
                currentIndex: pane.voice.index
                onActivated: (i) => { pane.voice.setIndex(i); currentIndex = Qt.binding(() => pane.voice.index) }
                ToolTip.visible: hovered && !popup.visible
                ToolTip.delay: 700
                ToolTip.text: pane.voice.tips[currentIndex] || ""
            }
            Button {
                icon.name: "media-playback-start"
                onClicked: pane.voice.play()
                ToolTip.visible: hovered
                ToolTip.delay: 500
                ToolTip.text: "Play voice " + pane.voice.letter + " alone"
            }
        }
        SpectrumEditor { objectName: "editor" + pane.voice.letter; Layout.fillWidth: true; voice: pane.voice }
        RowLayout {
            Layout.fillWidth: true
            Label { text: "Partials"; Layout.preferredWidth: 78; color: backend.theme.subtext }
            TextField {
                Layout.fillWidth: true
                text: pane.voice.freqText
                font.family: backend.monoFamily
                onTextEdited: pane.voice.setFreqText(text)
                onTextChanged: if (!activeFocus) cursorPosition = 0
                onActiveFocusChanged: if (!activeFocus) cursorPosition = 0
            }
        }
        RowLayout {
            Layout.fillWidth: true
            Label { text: "Amplitudes"; Layout.preferredWidth: 78; color: backend.theme.subtext }
            TextField {
                Layout.fillWidth: true
                text: pane.voice.ampText
                font.family: backend.monoFamily
                onTextEdited: pane.voice.setAmpText(text)
                onTextChanged: if (!activeFocus) cursorPosition = 0
                onActiveFocusChanged: if (!activeFocus) cursorPosition = 0
            }
        }
        Label {
            Layout.fillWidth: true
            visible: !pane.voice.valid
            text: pane.voice.error
            color: backend.theme.red
            font.pixelSize: 12
            wrapMode: Text.WordWrap
        }
    }

    // A GraphsTheme resets custom colours when its colorScheme changes, so each graph keeps one
    // theme per scheme (colorScheme is a literal) and swaps between them.
    component CurveTheme: GraphsTheme {
        backgroundVisible: false
        plotAreaBackgroundVisible: false
        labelTextColor: backend.theme.subtext
        grid.mainColor: backend.theme.grid
        grid.subColor: backend.theme.gridSub
        axisX.mainColor: backend.theme.border
        axisY.mainColor: backend.theme.border
        axisXLabelFont.pixelSize: 11
        axisYLabelFont.pixelSize: 11
    }
    component SurfaceTheme: GraphsTheme {
        backgroundVisible: false
        plotAreaBackgroundVisible: false
        labelBackgroundVisible: false
        labelTextColor: backend.theme.subtext
        labelFont.pointSize: 28
        grid.mainColor: backend.theme.grid
        grid.subColor: backend.theme.gridSub
        singleHighlightColor: backend.accent
    }

    component CurvePanel: Item {
        id: cp
        property real xMin: 0.9
        property real xMax: 2.3
        property real yMax: 1.1
        property bool smooth: true
        property bool hovering: false
        property var readout: ({})
        property real snapX: NaN
        property bool gliding: false
        property bool panning: false
        property bool zooming: false
        property bool moved: false
        property point pressPos: Qt.point(0, 0)
        property real panX0: 0
        property real panX1: 0
        readonly property rect plot: gv.plotArea
        readonly property color accent: backend.accent

        function niceStep(span, target) {
            var raw = span / Math.max(1, target)
            var mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10))
            var steps = [1, 2, 2.5, 5, 10]
            for (var i = 0; i < steps.length; ++i)
                if (mag * steps[i] >= raw) return mag * steps[i]
            return mag * 10
        }
        function px(x) { return (x - axX.min) / (axX.max - axX.min) * plot.width }
        function py(y) { return plot.height - (y - axY.min) / (axY.max - axY.min) * plot.height }
        function xAt(lx) { return axX.min + lx / plot.width * (axX.max - axX.min) }
        function fitY() { yMax = Math.max(0.05, backend.visibleTop(xMin, xMax) * 1.08) }
        function resetView() { smooth = true; xMin = backend.curveXmin; xMax = backend.curveXmax; fitY() }
        function updateHover(lx) {
            var x = xAt(lx)
            var snap = NaN
            var best = 10
            var mins = backend.curveMinima
            for (var i = 0; i < mins.length; ++i) {
                if (mins[i].x > backend.curveXmax) continue
                var d = Math.abs(px(mins[i].x) - lx)
                if (d <= best) { best = d; snap = mins[i].x }
            }
            snapX = snap
            readout = backend.curveReadout(isNaN(snap) ? x : snap, !isNaN(snap))
            hovering = readout.x !== undefined
            backend.setLinkedInterval(isNaN(snap) ? "" : snap.toFixed(6))
        }
        function clearHover() { hovering = false; snapX = NaN; backend.setLinkedInterval("") }
        function placeLabels() {
            var items = []
            for (var i = 0; i < labelRep.count; ++i) {
                var it = labelRep.itemAt(i)
                if (it && it.visible) items.push(it)
            }
            items.sort(function(a, b) { return a.ry - b.ry })
            var placed = []
            var rows = [0, 1, 2, -1, -2]
            for (var k = 0; k < items.length; ++k) {
                var it2 = items[k]
                var w = it2.labW, h = it2.labH
                var ax = Math.min(Math.max(0.5, 1 - (plot.width - 2 - it2.mx) / w), (it2.mx - 2) / w)
                var chosen = 0, rect = null
                for (var r = 0; r < rows.length; ++r) {
                    var row = rows[r]
                    var top = row >= 0 ? it2.my + 6 + row * (h + 2) : it2.my - 8 - h - (-row - 1) * (h + 2)
                    var rc = { x: it2.mx - ax * w, y: top, w: w, h: h }
                    if (rc.y + h > plot.height + 2 || rc.y < -2) continue
                    var clash = false
                    for (var q = 0; q < placed.length; ++q) {
                        var o = placed[q]
                        if (rc.x < o.x + o.w && o.x < rc.x + rc.w && rc.y < o.y + o.h && o.y < rc.y + rc.h) { clash = true; break }
                    }
                    if (!clash) { chosen = row; rect = rc; break }
                }
                if (rect === null) rect = { x: it2.mx - ax * w, y: it2.my + 6, w: w, h: h }
                it2.row = chosen
                it2.ax = ax
                placed.push(rect)
            }
        }

        Connections {
            target: backend
            function onCurveReset() { cp.resetView() }
            function onCurveRefit() { cp.fitY() }
            function onMinimaChanged() { placeTimer.restart() }
        }
        Timer { id: placeTimer; interval: 30; onTriggered: cp.placeLabels() }

        GraphsView {
            id: gv
            anchors.fill: parent
            marginLeft: 4
            marginRight: 16
            marginTop: 12
            marginBottom: 2
            onPlotAreaChanged: placeTimer.restart()
            property GraphsTheme darkTheme: CurveTheme { colorScheme: GraphsTheme.ColorScheme.Dark }
            property GraphsTheme lightTheme: CurveTheme { colorScheme: GraphsTheme.ColorScheme.Light }
            theme: backend.dark ? darkTheme : lightTheme
            axisX: ValueAxis {
                id: axX
                min: cp.xMin
                max: cp.xMax
                tickInterval: cp.niceStep(cp.xMax - cp.xMin, 8)
                subTickCount: 1
                labelDecimals: tickInterval < 0.1 ? 2 : 1
                titleText: "Frequency ratio  B / A"
                titleColor: backend.theme.subtext
                titleFont.pixelSize: 12
                Behavior on min { enabled: cp.smooth; NumberAnimation { duration: 380; easing.type: Easing.OutCubic } }
                Behavior on max { enabled: cp.smooth; NumberAnimation { duration: 380; easing.type: Easing.OutCubic } }
                onMinChanged: placeTimer.restart()
                onMaxChanged: placeTimer.restart()
            }
            axisY: ValueAxis {
                id: axY
                min: 0
                max: cp.yMax
                tickInterval: cp.niceStep(cp.yMax, 6)
                subTickCount: 1
                labelDecimals: tickInterval < 0.1 ? 2 : 1
                titleText: "Sensory dissonance (normalised)"
                titleColor: backend.theme.subtext
                titleFont.pixelSize: 12
                Behavior on max { NumberAnimation { duration: 380; easing.type: Easing.OutCubic } }
                onMaxChanged: placeTimer.restart()
            }
            AreaSeries {
                color: Qt.rgba(cp.accent.r, cp.accent.g, cp.accent.b, 0.14)
                borderColor: "transparent"
                borderWidth: 0
                upperSeries: LineSeries { objectName: "curveFill" }
            }
            LineSeries { objectName: "curveLine"; color: cp.accent; width: 2 }
        }

        Item {
            id: plotClip
            x: cp.plot.x
            y: cp.plot.y
            width: cp.plot.width
            height: cp.plot.height
            clip: true

            Repeater {
                id: labelRep
                model: backend.minimaModel
                delegate: Item {
                    id: mk
                    required property real rx
                    required property real ry
                    required property string label
                    required property int index
                    property int row: 0
                    property real ax: 0.5
                    visible: rx <= backend.curveXmax
                    readonly property real mx: cp.px(rx)
                    readonly property real my: cp.py(ry)
                    readonly property real labW: lab.width
                    readonly property real labH: lab.height
                    Rectangle {
                        width: 9; height: 9; radius: 4.5
                        x: mk.mx - 4.5
                        y: mk.my - 4.5
                        color: backend.theme.yellow
                        border.color: backend.theme.markerEdge
                        border.width: 1
                    }
                    Label {
                        id: lab
                        text: mk.label
                        font.pixelSize: 10
                        lineHeight: 0.9
                        color: backend.theme.yellow
                        horizontalAlignment: Text.AlignHCenter
                        x: mk.mx - width * mk.ax
                        y: mk.row >= 0 ? mk.my + 6 + mk.row * (height + 2)
                                       : mk.my - 8 - height - (-mk.row - 1) * (height + 2)
                    }
                }
            }

            Rectangle {
                id: ring
                readonly property real tx: !isNaN(cp.snapX) ? cp.snapX
                                         : (backend.highlight.kind === "interval" ? backend.highlight.x : NaN)
                property real pulse: 1
                visible: !isNaN(tx)
                width: 22 * pulse
                height: width
                radius: width / 2
                color: "transparent"
                border.color: cp.accent
                border.width: 2
                opacity: Math.max(0, 1.7 - pulse)
                x: cp.px(tx) - width / 2
                y: { backend.curveVersion; return cp.py(backend.curveY(tx)) - height / 2 }
                SequentialAnimation on pulse {
                    running: ring.visible
                    loops: Animation.Infinite
                    NumberAnimation { from: 0.8; to: 1.3; duration: 550; easing.type: Easing.InOutSine }
                    NumberAnimation { from: 1.3; to: 0.8; duration: 550; easing.type: Easing.InOutSine }
                }
            }

            Item {
                id: playMark
                readonly property bool on: backend.playing.kind === "interval" && backend.audioActive
                readonly property real lx: cp.px(backend.playing.x)
                anchors.fill: parent
                opacity: on ? 1 : 0
                visible: opacity > 0.01
                Behavior on opacity { NumberAnimation { duration: playMark.on ? 120 : 420 } }
                Rectangle { x: playMark.lx - 0.8; width: 1.6; height: parent.height; color: backend.theme.green; opacity: 0.7 }
                Rectangle {
                    width: 13; height: 13; radius: 6.5
                    x: playMark.lx - 6.5
                    y: { backend.curveVersion; return cp.py(backend.curveY(backend.playing.x)) - 6.5 }
                    color: backend.theme.green
                    border.color: backend.theme.markerEdge
                    border.width: 1.2
                }
            }

            Rectangle {
                visible: cp.hovering && !cp.panning
                x: cp.px(cp.readout.x || 0)
                width: 1
                height: parent.height
                color: backend.theme.cursor
                opacity: 0.6
            }
            Rectangle {
                visible: cp.hovering && !cp.panning
                readonly property real ax: cp.px(cp.readout.x || 0)
                readonly property real ay: cp.py(cp.readout.y || 0)
                x: ax + 14 + width > parent.width ? ax - width - 14 : ax + 14
                y: ay - height - 12 < 0 ? ay + 12 : ay - height - 12
                width: roText.implicitWidth + 18
                height: roText.implicitHeight + 12
                radius: 6
                color: backend.theme.popup
                border.color: backend.theme.border
                Label {
                    id: roText
                    anchors.centerIn: parent
                    textFormat: Text.RichText
                    text: cp.readout.html || ""
                    font.family: backend.monoFamily
                    font.pixelSize: 11
                }
            }
        }

        MouseArea {
            anchors.fill: plotClip
            hoverEnabled: true
            acceptedButtons: Qt.LeftButton | Qt.MiddleButton | Qt.RightButton
            cursorShape: cp.panning ? Qt.ClosedHandCursor : (cp.zooming ? Qt.SizeHorCursor : Qt.CrossCursor)
            onPositionChanged: (m) => {
                if (pressed) {
                    if (cp.zooming) {
                        // Right-drag zooms about the left edge.
                        var span = (cp.panX1 - cp.panX0) * Math.pow(1.02, -(m.x - cp.pressPos.x))
                        cp.smooth = false
                        cp.xMax = cp.xMin + Math.max(0.02, Math.min(span, backend.curveXmax - cp.xMin))
                        cp.fitY()
                        return
                    }
                    if (!cp.moved && Math.abs(m.x - cp.pressPos.x) + Math.abs(m.y - cp.pressPos.y) < 4) return
                    if (!cp.moved) {
                        cp.moved = true
                        if (!cp.panning) { cp.gliding = true; backend.glideBegin("interval", Math.max(0.25, cp.xAt(m.x)), 0) }
                    }
                    if (cp.panning) {
                        var dx = (m.x - cp.pressPos.x) / cp.plot.width * (cp.panX1 - cp.panX0)
                        var width = cp.panX1 - cp.panX0
                        var x0 = Math.max(backend.curveXmin, Math.min(cp.panX0 - dx, backend.curveXmax - width))
                        cp.smooth = false
                        cp.xMin = x0
                        cp.xMax = x0 + width
                        return
                    }
                    backend.glideMove(Math.max(0.25, cp.xAt(m.x)), 0)
                }
                cp.updateHover(m.x)
            }
            onPressed: (m) => {
                cp.moved = false
                cp.pressPos = Qt.point(m.x, m.y)
                cp.zooming = m.button === Qt.RightButton
                cp.panning = !cp.zooming && (m.button === Qt.MiddleButton || !!(m.modifiers & Qt.ShiftModifier))
                cp.panX0 = cp.xMin
                cp.panX1 = cp.xMax
            }
            onReleased: (m) => {
                if (cp.zooming) {
                    cp.zooming = false
                    cp.smooth = true
                    return
                }
                if (!cp.moved && !cp.panning)
                    backend.playInterval(isNaN(cp.snapX) ? cp.xAt(m.x) : cp.snapX)
                if (cp.gliding) backend.glideEnd()
                var wasPanning = cp.panning && cp.moved
                cp.gliding = false
                cp.panning = false
                cp.smooth = true
                if (wasPanning) cp.fitY()
            }
            onExited: cp.clearHover()
            onEntered: cp.updateHover(mouseX)
            onWheel: (w) => {
                var f = Math.pow(0.85, w.angleDelta.y / 120)
                cp.smooth = true
                var span = Math.max(0.02, Math.min((cp.xMax - cp.xMin) * f, backend.curveXmax - cp.xMin))
                cp.xMax = cp.xMin + span
                cp.fitY()
            }
        }
    }

    component SurfacePanel: Item {
        id: sp
        readonly property real camX: surf.cameraXRotation
        readonly property real camY: surf.cameraYRotation
        readonly property real camZoom: surf.cameraZoomLevel
        property bool tapPending: false
        property bool probing: false
        property var probe: ({})
        property point probePos: Qt.point(0, 0)
        property var resetFrom: ({})
        property real resetT: 0
        function wrap(a) { return ((a + 540) % 360) - 180 }
        // QtGraphs does not repaint after camera properties are set from code; ask it to.
        function redraw() { surf.needRender(); dots.needRender(); marks.needRender() }
        // Index of the best-triad marker within about 10 px of *pos* (screen space), or -1:
        // the overlay is picked on a 3 px grid of points around the pointer, nearest first.
        readonly property var snapOffsets: {
            var o = []
            for (var i = -3; i <= 3; ++i)
                for (var j = -3; j <= 3; ++j)
                    if (i * i + j * j <= 12) o.push([i * 3, j * 3])
            return o.sort((a, b) => (a[0] * a[0] + a[1] * a[1]) - (b[0] * b[0] + b[1] * b[1]))
        }
        function triadAt(pos) {
            var hit = -1
            for (var n = 0; n < snapOffsets.length && hit < 0; ++n) {
                dots.doPicking(Qt.point(pos.x + snapOffsets[n][0], pos.y + snapOffsets[n][1]))
                hit = triSeries.selectedItem
            }
            triSeries.selectedItem = triSeries.invalidSelectionIndex
            return hit
        }
        Connections {
            target: backend
            function onThemeChanged() { Qt.callLater(sp.redraw) }
            function onMarkersChanged() { Qt.callLater(sp.redraw) }
            function onSurfaceChanged() { Qt.callLater(sp.redraw) }
        }
        function resetView() {
            var d = backend.cameraDefault
            var t = surf.cameraTargetPosition
            resetFrom = { x: surf.cameraXRotation, dx: wrap(d.x - surf.cameraXRotation),
                          y: surf.cameraYRotation, zoom: surf.cameraZoomLevel, tx: t.x, ty: t.y, tz: t.z }
            resetAnim.restart()
        }
        function clearProbe() {
            probe = ({})
            probing = false
            backend.setSnapTriad("")
            if (!tapPending) clearSurfacePick()
        }
        function clearSurfacePick() {
            if (surfSeries.selectedPoint.x < 0) return
            surf.clearSelection()
            Qt.callLater(sp.redraw)
        }
        // Camera X rotation must stay within [-180, 180], so the reset animates a phase and
        // wraps the angle itself (the short way round).
        NumberAnimation {
            id: resetAnim
            target: sp; property: "resetT"; from: 0; to: 1
            duration: 650; easing.type: Easing.InOutCubic
        }
        onResetTChanged: {
            if (!resetAnim.running && resetT !== 1) return
            var f = resetFrom, d = backend.cameraDefault, t = resetT, k = 1 - resetT
            surf.cameraXRotation = wrap(f.x + f.dx * t)
            surf.cameraYRotation = f.y + (d.y - f.y) * t
            surf.cameraZoomLevel = f.zoom + (d.zoom - f.zoom) * t
            surf.cameraTargetPosition = Qt.vector3d(f.tx * k, f.ty * k, f.tz * k)
        }
        Timer {
            id: tapTimer
            interval: 160
            onTriggered: {
                if (!sp.tapPending) return
                sp.tapPending = false
                var p = surfSeries.selectedPoint
                if (p.x >= 0 && p.y >= 0) backend.surfacePicked(p.x, p.y)
            }
        }
        Timer {
            id: probeTimer
            interval: 40
            onTriggered: {
                if (!surfHover.hovered || rotateDrag.active || panDrag.active) return
                var k = sp.triadAt(sp.probePos)
                if (k >= 0) {
                    sp.probing = false
                    sp.probe = backend.triadReadout(k)
                    backend.setSnapTriad(sp.probe.key || "")
                    sp.clearSurfacePick()
                    return
                }
                backend.setSnapTriad("")
                sp.probing = true
                surf.doPicking(sp.probePos)
            }
        }

        Surface3D {
            id: surf
            objectName: "surface3d"
            anchors.fill: parent
            anchors.rightMargin: 84
            msaaSamples: 4
            shadowQuality: Graphs3D.ShadowQuality.SoftLow
            cameraXRotation: backend.cameraDefault.x
            cameraYRotation: backend.cameraDefault.y
            cameraZoomLevel: backend.cameraDefault.zoom
            aspectRatio: 1.6
            margin: 0.1
            selectionMode: Graphs3D.SelectionFlag.Item
            // Our handlers below rotate and zoom: QtGraphs' own drag uses the right button and its
            // wheel steps are tiny.
            Component.onCompleted: { unsetDefaultDragHandler(); unsetDefaultWheelHandler() }
            onCameraXRotationChanged: Qt.callLater(sp.redraw)
            onCameraYRotationChanged: Qt.callLater(sp.redraw)
            onCameraZoomLevelChanged: Qt.callLater(sp.redraw)
            onCameraTargetPositionChanged: Qt.callLater(sp.redraw)
            onTapped: (eventPoint, button) => {
                var k = sp.triadAt(eventPoint.position)
                if (k >= 0) {
                    backend.playShownTriad(k)
                    return
                }
                sp.tapPending = true
                tapTimer.restart()
            }
            property GraphsTheme darkTheme: SurfaceTheme { colorScheme: GraphsTheme.ColorScheme.Dark }
            property GraphsTheme lightTheme: SurfaceTheme { colorScheme: GraphsTheme.ColorScheme.Light }
            theme: backend.dark ? darkTheme : lightTheme
            axisX: Value3DAxis {
                min: backend.gridMin; max: backend.gridMax; autoAdjustRange: false
                title: "Voice B ratio"; titleVisible: true; labelFormat: "%.2f"; segmentCount: 4
                labelAutoAngle: 90; titleFixed: false
            }
            axisZ: Value3DAxis {
                min: backend.gridMin; max: backend.gridMax; autoAdjustRange: false
                title: "Voice C ratio"; titleVisible: true; labelFormat: "%.2f"; segmentCount: 4
                labelAutoAngle: 90; titleFixed: false
            }
            axisY: Value3DAxis {
                // Untitled: the colour bar beside the graph already names it.
                min: 0; max: 1; autoAdjustRange: false
                labelFormat: "%.1f"; segmentCount: 5
                labelAutoAngle: 90
            }
            Surface3DSeries {
                id: surfSeries
                objectName: "surfaceSeries"
                drawMode: Surface3DSeries.DrawFlag.DrawSurface
                shading: Surface3DSeries.Shading.Smooth
                colorStyle: GraphsTheme.ColorStyle.RangeGradient
                // The built-in item label swaps the X and Z values; the QML probe replaces it.
                itemLabelVisible: false
                // baseGradient is set from Python (Backend._apply_surface_gradient).
                onSelectedPointChanged: (p) => {
                    var ok = p.x >= 0 && p.y >= 0
                    if (sp.tapPending) {
                        if (!ok) return
                        sp.tapPending = false
                        tapTimer.stop()
                        backend.surfacePicked(p.x, p.y)
                    } else if (sp.probing) {
                        sp.probing = false
                        sp.probe = ok ? backend.surfaceReadout(p.x, p.y) : ({})
                    }
                }
            }
        }

        // Triad markers: a transparent scatter graph sharing the surface's camera and axes,
        // drawn on top so the markers are never hidden by the surface.
        Scatter3D {
            id: dots
            objectName: "triadOverlay"
            anchors.fill: surf
            enabled: false
            msaaSamples: 4
            cameraXRotation: surf.cameraXRotation
            cameraYRotation: surf.cameraYRotation
            cameraZoomLevel: surf.cameraZoomLevel
            cameraTargetPosition: surf.cameraTargetPosition
            aspectRatio: surf.aspectRatio
            horizontalAspectRatio: surf.horizontalAspectRatio
            margin: surf.margin
            theme: GraphsTheme {
                colorScheme: GraphsTheme.ColorScheme.Dark
                backgroundVisible: false
                plotAreaBackgroundVisible: false
                gridVisible: false
                labelsVisible: false
            }
            axisX: Value3DAxis { min: backend.gridMin; max: backend.gridMax; autoAdjustRange: false; labelsVisible: false }
            axisZ: Value3DAxis { min: backend.gridMin; max: backend.gridMax; autoAdjustRange: false; labelsVisible: false }
            axisY: Value3DAxis { min: 0; max: 1; autoAdjustRange: false; labelsVisible: false }
            // Picking selects markers for an instant; keep that invisible.
            Scatter3DSeries {
                id: triSeries
                objectName: "triadDots"
                mesh: Abstract3DSeries.Mesh.Sphere
                meshSmooth: true
                baseColor: backend.theme.yellow
                singleHighlightColor: backend.theme.yellow
                itemLabelVisible: false
                itemSize: 0.12
            }
        }

        // The playing and highlight markers get their own overlay so they never catch the
        // picks that snap the probe to the best-triad markers.
        Scatter3D {
            id: marks
            anchors.fill: surf
            enabled: false
            msaaSamples: 4
            cameraXRotation: surf.cameraXRotation
            cameraYRotation: surf.cameraYRotation
            cameraZoomLevel: surf.cameraZoomLevel
            cameraTargetPosition: surf.cameraTargetPosition
            aspectRatio: surf.aspectRatio
            horizontalAspectRatio: surf.horizontalAspectRatio
            margin: surf.margin
            theme: GraphsTheme {
                colorScheme: GraphsTheme.ColorScheme.Dark
                backgroundVisible: false
                plotAreaBackgroundVisible: false
                gridVisible: false
                labelsVisible: false
            }
            axisX: Value3DAxis { min: backend.gridMin; max: backend.gridMax; autoAdjustRange: false; labelsVisible: false }
            axisZ: Value3DAxis { min: backend.gridMin; max: backend.gridMax; autoAdjustRange: false; labelsVisible: false }
            axisY: Value3DAxis { min: 0; max: 1; autoAdjustRange: false; labelsVisible: false }
            Scatter3DSeries {
                objectName: "playDot"
                mesh: Abstract3DSeries.Mesh.Sphere
                meshSmooth: true
                baseColor: backend.theme.green
                itemSize: 0.17
                visible: backend.playing.kind === "triad" && backend.audioActive
            }
            Scatter3DSeries {
                id: hlSeries
                objectName: "hlDot"
                mesh: Abstract3DSeries.Mesh.Sphere
                meshSmooth: true
                baseColor: backend.accent
                itemSize: 0.15
                visible: backend.highlight.kind === "triad" || backend.snap.kind === "triad"
                onItemSizeChanged: marks.needRender()
            }
        }
        SequentialAnimation {
            running: hlSeries.visible
            loops: Animation.Infinite
            NumberAnimation { target: hlSeries; property: "itemSize"; from: 0.13; to: 0.2; duration: 550; easing.type: Easing.InOutSine }
            NumberAnimation { target: hlSeries; property: "itemSize"; from: 0.2; to: 0.13; duration: 550; easing.type: Easing.InOutSine }
        }

        // Pointer input on top of both graphs. Taps fall through to the surface (picking);
        // drags rotate or pan, the wheel zooms and hovering probes the surface.
        Item {
            anchors.fill: surf
            WheelHandler {
                target: null
                onWheel: (event) => {
                    var z = surf.cameraZoomLevel * Math.pow(1.127, event.angleDelta.y / 120)
                    surf.cameraZoomLevel = Math.max(surf.minCameraZoomLevel, Math.min(surf.maxCameraZoomLevel, z))
                }
            }
            HoverHandler {
                id: surfHover
                onPointChanged: {
                    sp.probePos = point.position
                    if (!probeTimer.running) probeTimer.start()
                }
                onHoveredChanged: if (!hovered) sp.clearProbe()
            }
            DragHandler {
                id: rotateDrag
                target: null
                acceptedButtons: Qt.LeftButton
                acceptedModifiers: Qt.NoModifier
                property vector2d last: Qt.vector2d(0, 0)
                cursorShape: Qt.ClosedHandCursor
                onActiveChanged: { last = Qt.vector2d(0, 0); if (active) sp.clearProbe() }
                onActiveTranslationChanged: {
                    var dx = activeTranslation.x - last.x, dy = activeTranslation.y - last.y
                    last = activeTranslation
                    surf.cameraXRotation = sp.wrap(surf.cameraXRotation + dx * 0.8)
                    surf.cameraYRotation = Math.max(0, Math.min(90, surf.cameraYRotation + dy * 0.8))
                }
            }
            DragHandler {
                id: panDrag
                target: null
                acceptedButtons: Qt.MiddleButton
                property vector2d last: Qt.vector2d(0, 0)
                cursorShape: Qt.ClosedHandCursor
                onActiveChanged: { last = Qt.vector2d(0, 0); if (active) sp.clearProbe() }
                onActiveTranslationChanged: {
                    var dx = activeTranslation.x - last.x, dy = activeTranslation.y - last.y
                    last = activeTranslation
                    // The target is in normalised axis units (QtGraphs keeps it inside the box)
                    // and its z runs against voice C.
                    var k = 9.0 / Math.max(1, surf.width) * 100 / surf.cameraZoomLevel
                    var th = surf.cameraXRotation * Math.PI / 180
                    var t = surf.cameraTargetPosition
                    var c = (v) => Math.max(-1, Math.min(1, v))
                    surf.cameraTargetPosition = Qt.vector3d(c(t.x - Math.cos(th) * dx * k), c(t.y + dy * k),
                                                            c(t.z - Math.sin(th) * dx * k))
                }
            }
        }

        Rectangle {
            visible: sp.probe.html !== undefined && surfHover.hovered && !rotateDrag.active && !panDrag.active
            x: sp.probePos.x + 16 + width > surf.width ? sp.probePos.x - width - 16 : sp.probePos.x + 16
            y: sp.probePos.y - height - 12 < 0 ? sp.probePos.y + 18 : sp.probePos.y - height - 12
            width: probeText.implicitWidth + 18
            height: probeText.implicitHeight + 12
            radius: 6
            color: backend.theme.popup
            border.color: backend.theme.border
            Label {
                id: probeText
                anchors.centerIn: parent
                textFormat: Text.RichText
                text: sp.probe.html || ""
                font.family: backend.monoFamily
                font.pixelSize: 11
            }
        }

        Item {
            anchors.right: parent.right
            anchors.top: parent.top
            anchors.bottom: parent.bottom
            width: 84
            Rectangle {
                id: cbar
                x: 8
                y: parent.height * 0.12
                width: 12
                height: parent.height * 0.76
                radius: 3
                border.color: backend.theme.border
                gradient: Gradient {
                    GradientStop { position: 0.0; color: backend.gradient[8] }
                    GradientStop { position: 0.125; color: backend.gradient[7] }
                    GradientStop { position: 0.25; color: backend.gradient[6] }
                    GradientStop { position: 0.375; color: backend.gradient[5] }
                    GradientStop { position: 0.5; color: backend.gradient[4] }
                    GradientStop { position: 0.625; color: backend.gradient[3] }
                    GradientStop { position: 0.75; color: backend.gradient[2] }
                    GradientStop { position: 0.875; color: backend.gradient[1] }
                    GradientStop { position: 1.0; color: backend.gradient[0] }
                }
            }
            Repeater {
                model: 6
                delegate: Label {
                    required property int index
                    x: cbar.x + cbar.width + 6
                    y: cbar.y + cbar.height * (1 - index / 5) - height / 2
                    text: (index / 5).toFixed(1)
                    font.pixelSize: 11
                    color: backend.theme.subtext
                }
            }
            Label {
                x: cbar.x + cbar.width + 34
                y: cbar.y + cbar.height / 2
                rotation: -90
                transformOrigin: Item.TopLeft
                text: "Sensory dissonance"
                font.pixelSize: 11
                color: backend.theme.muted
            }
        }
    }

    component HeatmapPanel: Item {
        id: hp
        property bool hovering: false
        property bool snapped: false
        property real hb: 1
        property real hc: 1
        property var readout: ({})
        property bool gliding: false
        property bool panning: false
        property bool zooming: false
        property bool moved: false
        property point pressPos: Qt.point(0, 0)
        property point lastPos: Qt.point(0, 0)
        readonly property rect pr: heat.plotRect
        function resetView() { heat.resetView() }
        function at(b, c) { heat.viewVersion; return heat.toPixel(b, c) }
        function clampB(v) { return Math.min(Math.max(v, backend.gridMin), backend.gridMax) }
        function setSnapped(on, key) {
            if (on !== snapped || on) backend.setSnapTriad(on ? key : "")
            snapped = on
        }
        // (px, py): pointer in item pixels; d: the data point under it.
        function updateHover(px, py, d) {
            var s = heat.snapAt(px, py, 10)
            if (s.key !== undefined) {
                hb = s.b
                hc = s.c
                setSnapped(true, s.key)
            } else if (heat.hasData && heat.inData(d.x, d.y)) {
                hb = d.x
                hc = d.y
                setSnapped(false, "")
            } else {
                setSnapped(false, "")
                hovering = false
                return
            }
            readout = backend.heatmapReadout(hb, hc)
            hovering = readout.html !== undefined
        }
        function clearHover() {
            hovering = false
            setSnapped(false, "")
        }

        HeatmapItem {
            id: heat
            objectName: "heatmapItem"
            anchors.fill: parent
            Image {
                anchors.fill: parent
                source: heat.frameSource
                cache: false
                asynchronous: false
                retainWhileLoading: true
            }
        }

        Item {
            anchors.fill: parent
            clip: true
            Rectangle {
                visible: hp.hovering && !hp.panning
                x: hp.at(hp.hb, hp.hc).x
                y: hp.pr.y
                width: 1
                height: hp.pr.height
                color: "#80ffffff"
            }
            Rectangle {
                visible: hp.hovering && !hp.panning
                x: hp.pr.x
                y: hp.at(hp.hb, hp.hc).y
                width: hp.pr.width
                height: 1
                color: "#80ffffff"
            }
            Rectangle {
                id: hring
                readonly property bool on: hp.snapped || backend.highlight.kind === "triad"
                readonly property point p: hp.snapped ? hp.at(hp.hb, hp.hc)
                                                      : hp.at(backend.highlight.b, backend.highlight.c)
                property real pulse: 1
                visible: on
                width: 24 * pulse
                height: width
                radius: width / 2
                x: p.x - width / 2
                y: p.y - height / 2
                color: "transparent"
                border.color: "white"
                border.width: 2
                opacity: Math.max(0, 1.7 - pulse)
                SequentialAnimation on pulse {
                    running: hring.on
                    loops: Animation.Infinite
                    NumberAnimation { from: 0.8; to: 1.3; duration: 550; easing.type: Easing.InOutSine }
                    NumberAnimation { from: 1.3; to: 0.8; duration: 550; easing.type: Easing.InOutSine }
                }
            }
            Rectangle {
                id: pring
                readonly property bool on: backend.playing.kind === "triad" && backend.audioActive
                readonly property point p: hp.at(backend.playing.b, backend.playing.c)
                opacity: on ? 1 : 0
                visible: opacity > 0.01
                Behavior on opacity { NumberAnimation { duration: pring.on ? 120 : 420 } }
                width: 18; height: 18; radius: 9
                x: p.x - 9
                y: p.y - 9
                color: "transparent"
                border.color: backend.theme.green
                border.width: 2.4
            }
            Rectangle {
                visible: hp.hovering && !hp.panning
                readonly property point p: hp.at(hp.hb, hp.hc)
                x: p.x + 14 + width > hp.pr.right ? p.x - width - 14 : p.x + 14
                y: p.y - height - 12 < hp.pr.y ? p.y + 12 : p.y - height - 12
                width: hText.implicitWidth + 18
                height: hText.implicitHeight + 12
                radius: 6
                color: backend.theme.popup
                border.color: backend.theme.border
                Label {
                    id: hText
                    anchors.centerIn: parent
                    textFormat: Text.RichText
                    text: hp.readout.html || ""
                    font.family: backend.monoFamily
                    font.pixelSize: 11
                }
            }
        }

        MouseArea {
            id: hma
            x: hp.pr.x
            y: hp.pr.y
            width: hp.pr.width
            height: hp.pr.height
            hoverEnabled: true
            acceptedButtons: Qt.LeftButton | Qt.MiddleButton | Qt.RightButton
            cursorShape: hp.panning ? Qt.ClosedHandCursor : (hp.zooming ? Qt.SizeAllCursor : Qt.CrossCursor)
            function dataAt(m) { return heat.toData(hma.x + m.x, hma.y + m.y) }
            onPositionChanged: (m) => {
                var d = dataAt(m)
                if (pressed) {
                    if (hp.zooming) {
                        // Right-drag zooms about the press point: right or up zooms in.
                        var f = Math.pow(1.02, -(m.x - hp.lastPos.x) + (m.y - hp.lastPos.y))
                        heat.zoomAt(hma.x + hp.pressPos.x, hma.y + hp.pressPos.y, f)
                        hp.lastPos = Qt.point(m.x, m.y)
                        return
                    }
                    if (!hp.moved && Math.abs(m.x - hp.pressPos.x) + Math.abs(m.y - hp.pressPos.y) < 4) return
                    if (!hp.moved) {
                        hp.moved = true
                        if (!hp.panning && heat.inData(d.x, d.y)) { hp.gliding = true; backend.glideBegin("triad", d.x, d.y) }
                    }
                    if (hp.panning) {
                        heat.panBy(m.x - hp.lastPos.x, m.y - hp.lastPos.y)
                        hp.lastPos = Qt.point(m.x, m.y)
                        return
                    }
                    if (hp.gliding) backend.glideMove(hp.clampB(d.x), hp.clampB(d.y))
                }
                hp.updateHover(hma.x + m.x, hma.y + m.y, d)
            }
            onPressed: (m) => {
                hp.moved = false
                hp.pressPos = Qt.point(m.x, m.y)
                hp.lastPos = Qt.point(m.x, m.y)
                hp.zooming = m.button === Qt.RightButton
                hp.panning = !hp.zooming && (m.button === Qt.MiddleButton || !!(m.modifiers & Qt.ShiftModifier))
            }
            onReleased: (m) => {
                if (hp.zooming) {
                    hp.zooming = false
                    return
                }
                var d = dataAt(m)
                if (!hp.moved && !hp.panning) {
                    if (hp.snapped) backend.playTriad(hp.hb, hp.hc)
                    else if (heat.inData(d.x, d.y)) backend.playTriad(d.x, d.y)
                }
                if (hp.gliding) backend.glideEnd()
                hp.gliding = false
                hp.panning = false
            }
            onExited: hp.clearHover()
            onEntered: hp.updateHover(hma.x + mouseX, hma.y + mouseY, dataAt(Qt.point(mouseX, mouseY)))
            onWheel: (w) => heat.zoomAt(hma.x + w.x, hma.y + w.y, Math.pow(0.85, w.angleDelta.y / 120))
        }
    }

    component OvertonePanel: Item {
        id: op
        readonly property var ov: backend.overtones
        property real grow: 1
        readonly property real base: height - 20
        readonly property real span: Math.max(1, base - 6)
        function lx(x) { return 8 + (x - ov.xmin) / (ov.xmax - ov.xmin) * (width - 18) }
        Connections { target: backend; function onOvertonesAnimate() { growAnim.restart() } }
        NumberAnimation { id: growAnim; target: op; property: "grow"; from: 0; to: 1; duration: 340; easing.type: Easing.OutCubic }

        Label {
            anchors.centerIn: parent
            visible: op.ov.empty
            text: "Play something to see how its partials line up"
            color: backend.theme.muted
        }
        Item {
            anchors.fill: parent
            visible: !op.ov.empty
            Rectangle { x: 8; y: op.base; width: op.width - 18; height: 1; color: backend.theme.border }
            Repeater {
                model: op.ov.empty ? [] : op.ov.minor
                delegate: Rectangle {
                    required property var modelData
                    x: op.lx(modelData); y: op.base; width: 1; height: 3; color: backend.theme.border
                }
            }
            Repeater {
                model: op.ov.empty ? [] : op.ov.ticks
                delegate: Item {
                    required property var modelData
                    Rectangle { x: op.lx(parent.modelData.x); y: op.base; width: 1; height: 5; color: backend.theme.muted }
                    Label {
                        x: op.lx(parent.modelData.x) - 40; y: op.base + 5; width: 80
                        horizontalAlignment: Text.AlignHCenter
                        text: parent.modelData.label
                        font.pixelSize: 10
                        color: backend.theme.subtext
                    }
                }
            }
            Repeater {
                model: op.ov.empty ? [] : op.ov.stems
                delegate: Rectangle {
                    required property var modelData
                    width: modelData.w
                    height: modelData.h / 1.25 * op.span * op.grow
                    x: op.lx(modelData.x) - width / 2
                    y: op.base - height
                    radius: width / 2
                    color: modelData.color
                    opacity: 0.8
                }
            }
            Repeater {
                model: op.ov.empty ? [] : op.ov.rings
                delegate: Rectangle {
                    required property var modelData
                    width: 9; height: 9; radius: 4.5
                    x: op.lx(modelData.x) - 4.5
                    y: op.base - (modelData.h + 0.12) / 1.25 * op.span * op.grow - 4.5
                    color: "transparent"
                    border.color: backend.theme.text
                    border.width: 1.4
                    opacity: op.grow
                }
            }
        }
    }

    component ResultList: ListView {
        id: lv
        property string linkedKey: ""
        property string selectedKey: ""
        property string emptyText: "Nothing to show yet"
        signal picked(real b, real c)
        signal hover(string key)
        clip: true
        spacing: 2
        boundsBehavior: Flickable.StopAtBounds
        ScrollBar.vertical: ScrollBar { }
        populate: Transition {
            NumberAnimation { property: "opacity"; from: 0; to: 1; duration: 260 }
            NumberAnimation { property: "x"; from: 16; duration: 260; easing.type: Easing.OutCubic }
        }
        delegate: ItemDelegate {
            id: row
            required property int index
            required property string primary
            required property string secondary
            required property real value
            required property string tint
            required property string key
            required property real b
            required property real c
            width: ListView.view.width
            height: 56
            highlighted: key === lv.linkedKey || key === lv.selectedKey
            onClicked: { lv.selectedKey = key; lv.picked(b, c) }
            onHoveredChanged: lv.hover(hovered ? key : "")
            contentItem: RowLayout {
                spacing: 10
                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 2
                    Label { text: row.primary; font.family: backend.monoFamily; font.pixelSize: 13; font.bold: true; color: backend.theme.text }
                    Label { Layout.fillWidth: true; text: row.secondary; font.pixelSize: 11; color: backend.theme.muted; elide: Text.ElideRight }
                }
                Label {
                    text: "\uE768"
                    font.family: backend.iconFamily
                    font.pixelSize: 12
                    color: backend.theme.green
                    opacity: row.hovered ? 1 : 0
                    Behavior on opacity { NumberAnimation { duration: 150 } }
                }
                ColumnLayout {
                    spacing: 5
                    Label {
                        Layout.alignment: Qt.AlignRight
                        text: row.value.toFixed(3)
                        font.family: backend.monoFamily
                        font.pixelSize: 13
                        font.bold: true
                        color: row.tint
                    }
                    Rectangle {
                        Layout.alignment: Qt.AlignRight
                        width: 62; height: 4; radius: 2
                        color: Qt.rgba(0.5, 0.5, 0.5, 0.25)
                        Rectangle {
                            width: Math.max(4, parent.width * Math.min(1, Math.max(0, row.value)))
                            height: 4; radius: 2
                            color: row.tint
                            Behavior on width { NumberAnimation { duration: 200 } }
                        }
                    }
                }
            }
        }
        Label {
            anchors.centerIn: parent
            visible: lv.count === 0
            text: lv.emptyText
            color: backend.theme.muted
        }
    }

    // ── layout ──────────────────────────────────────────────────────────

    ColumnLayout {
        anchors.fill: parent
        anchors.leftMargin: 12
        anchors.rightMargin: 12
        anchors.topMargin: 6
        anchors.bottomMargin: 8
        spacing: 10

        RowLayout {
            Layout.fillWidth: true
            Layout.preferredHeight: 48
            spacing: 10
            Rectangle {
                width: 32; height: 32; radius: 8
                gradient: Gradient {
                    GradientStop { position: 0.0; color: Qt.lighter(backend.accent, 1.15) }
                    GradientStop { position: 1.0; color: Qt.darker(backend.accent, 1.35) }
                }
                Label { anchors.centerIn: parent; text: "\u223F"; color: "white"; font.pixelSize: 22; font.bold: true }
            }
            ColumnLayout {
                spacing: 0
                Label { text: "Dissonance Quick"; font.pixelSize: 15; font.weight: Font.DemiBold; color: backend.theme.text }
                Label { text: "Qt Quick \u00b7 Fluent \u00b7 QtGraphs"; font.pixelSize: 11; color: backend.theme.muted }
            }
            Item { Layout.fillWidth: true }
            TabBar {
                id: viewTabs
                objectName: "viewTabs"
                Layout.preferredWidth: 330
                onCurrentIndexChanged: backend.setViewIndex(currentIndex)
                TabButton { text: "Curve" }
                TabButton { text: "Surface" }
                TabButton { text: "Heatmap" }
            }
            Item { Layout.fillWidth: true }
            Switch {
                text: "Live"
                checked: backend.live
                onToggled: backend.setLive(checked)
                ToolTip.visible: hovered; ToolTip.delay: 600
                ToolTip.text: "Re-analyse automatically while you edit"
            }
            Button {
                text: "Analyse"
                icon.name: "view-refresh"
                highlighted: true
                onClicked: backend.analyse()
                ToolTip.visible: hovered; ToolTip.delay: 600
                ToolTip.text: "Recompute everything and reset the views  (Ctrl+Enter)"
            }
            ToolSeparator { }
            ToolButton {
                icon.name: backend.volume === 0 ? "audio-volume-muted"
                         : (backend.volume < 0.5 ? "audio-volume-low" : "audio-volume-high")
                onClicked: backend.toggleMute()
                ToolTip.visible: hovered; ToolTip.delay: 600
                ToolTip.text: backend.volume === 0 ? "Unmute" : "Mute"
            }
            Slider {
                Layout.preferredWidth: 110
                from: 0
                to: 1
                value: backend.volume
                onMoved: backend.setVolume(value)
            }
            Rectangle {
                width: 4; height: 18; radius: 2
                color: backend.theme.inset
                border.color: backend.theme.border
                Rectangle {
                    anchors.bottom: parent.bottom
                    width: parent.width
                    height: parent.height * backend.level
                    radius: 2
                    color: backend.level > 0.75 ? backend.theme.yellow : backend.theme.green
                }
            }
            ToolButton {
                icon.name: "media-playback-stop"
                onClicked: backend.stopAudio()
                ToolTip.visible: hovered; ToolTip.delay: 600
                ToolTip.text: "Stop audio  (Esc)"
            }
            ToolSeparator { }
            ToolButton {
                text: backend.dark ? "\uE706" : "\uE708"
                font.family: backend.iconFamily
                font.pixelSize: 16
                onClicked: win.switchTheme()
                ToolTip.visible: hovered; ToolTip.delay: 600
                ToolTip.text: backend.dark ? "Switch to the light theme" : "Switch to the dark theme"
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 12

            ScrollView {
                id: side
                Layout.preferredWidth: 364
                Layout.fillHeight: true
                contentWidth: availableWidth
                clip: true

                ColumnLayout {
                    width: side.availableWidth - 12
                    spacing: 10

                    Card {
                        Layout.fillWidth: true
                        implicitHeight: voicesCol.implicitHeight + 26
                        ColumnLayout {
                            id: voicesCol
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.top: parent.top
                            anchors.margins: 13
                            spacing: 9
                            Caption { text: "Voices" }
                            Segmented {
                                id: voiceSeg
                                Layout.fillWidth: true
                                labels: [backend.voices[0].label, backend.voices[1].label, backend.voices[2].label]
                                colors: [backend.voices[0].color, backend.voices[1].color, backend.voices[2].color]
                            }
                            Item {
                                Layout.fillWidth: true
                                implicitHeight: [paneA, paneB, paneC][voiceSeg.currentIndex].implicitHeight
                                VoicePane {
                                    id: paneA; width: parent.width; voice: backend.voices[0]
                                    opacity: voiceSeg.currentIndex === 0 ? 1 : 0; visible: opacity > 0.01
                                    Behavior on opacity { NumberAnimation { duration: 180 } }
                                }
                                VoicePane {
                                    id: paneB; width: parent.width; voice: backend.voices[1]
                                    opacity: voiceSeg.currentIndex === 1 ? 1 : 0; visible: opacity > 0.01
                                    Behavior on opacity { NumberAnimation { duration: 180 } }
                                }
                                VoicePane {
                                    id: paneC; width: parent.width; voice: backend.voices[2]
                                    opacity: voiceSeg.currentIndex === 2 ? 1 : 0; visible: opacity > 0.01
                                    Behavior on opacity { NumberAnimation { duration: 180 } }
                                }
                            }
                        }
                    }

                    Card {
                        Layout.fillWidth: true
                        implicitHeight: anaCol.implicitHeight + 26
                        ColumnLayout {
                            id: anaCol
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.top: parent.top
                            anchors.margins: 13
                            spacing: 8
                            Caption { text: "Analysis" }
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 8
                                ColumnLayout {
                                    spacing: 4
                                    Label { text: "Fundamental Hz"; font.pixelSize: 11; color: backend.theme.subtext }
                                    TextField { Layout.fillWidth: true; Layout.preferredWidth: 90; text: backend.refText; onTextEdited: backend.setRef(text) }
                                }
                                ColumnLayout {
                                    spacing: 4
                                    Label { text: "Max ratio"; font.pixelSize: 11; color: backend.theme.subtext }
                                    TextField { Layout.fillWidth: true; Layout.preferredWidth: 90; text: backend.maxText; onTextEdited: backend.setMax(text) }
                                }
                                ColumnLayout {
                                    spacing: 4
                                    Label { text: "3D step"; font.pixelSize: 11; color: backend.theme.subtext }
                                    TextField { Layout.fillWidth: true; Layout.preferredWidth: 90; text: backend.stepText; onTextEdited: backend.setStep(text) }
                                }
                            }
                            Label {
                                Layout.fillWidth: true
                                visible: backend.settingsError.length > 0
                                text: backend.settingsError
                                color: backend.theme.red
                                font.pixelSize: 12
                                wrapMode: Text.WordWrap
                            }
                        }
                    }

                    Card {
                        Layout.fillWidth: true
                        implicitHeight: manCol.implicitHeight + 26
                        ColumnLayout {
                            id: manCol
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.top: parent.top
                            anchors.margins: 13
                            spacing: 8
                            Caption { text: "Manual playback" }
                            RowLayout {
                                Label { text: "Note A"; Layout.preferredWidth: 78; color: backend.theme.subtext }
                                TextField { id: noteField; Layout.fillWidth: true; text: "1.0" }
                                Button {
                                    icon.name: "media-playback-start"
                                    onClicked: backend.playNote(noteField.text)
                                    ToolTip.visible: hovered; ToolTip.delay: 500
                                    ToolTip.text: "Play voice A on its own at this ratio"
                                }
                            }
                            RowLayout {
                                Label { text: "A + B"; Layout.preferredWidth: 78; color: backend.theme.subtext }
                                TextField { id: intervalField; Layout.fillWidth: true; text: "1.5" }
                                Button {
                                    icon.name: "media-playback-start"
                                    onClicked: backend.playIntervalText(intervalField.text)
                                    ToolTip.visible: hovered; ToolTip.delay: 500
                                    ToolTip.text: "Play A with B at this ratio"
                                }
                            }
                            RowLayout {
                                Label { text: "Triad B, C"; Layout.preferredWidth: 78; color: backend.theme.subtext }
                                TextField { id: triadField; Layout.fillWidth: true; text: "1.25, 1.5" }
                                Button {
                                    icon.name: "media-playback-start"
                                    onClicked: backend.playTriadText(triadField.text)
                                    ToolTip.visible: hovered; ToolTip.delay: 500
                                    ToolTip.text: "Play A, B and C at these ratios"
                                }
                            }
                            RowLayout {
                                Label { text: "Length (s)"; Layout.preferredWidth: 78; color: backend.theme.subtext }
                                TextField { id: lengthField; Layout.fillWidth: true; text: "1.5" }
                                Button {
                                    text: "Save WAV\u2026"
                                    icon.name: "document-save"
                                    onClicked: backend.saveWav(lengthField.text)
                                    ToolTip.visible: hovered; ToolTip.delay: 500
                                    ToolTip.text: "Write the last sound you played to a 24-bit WAV file of this length"
                                }
                            }
                        }
                    }
                }
            }

            ColumnLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.minimumWidth: 320
                Layout.preferredWidth: 900
                spacing: 10

                Card {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 12
                        spacing: 6
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8
                            Label { text: win.viewTitles[viewTabs.currentIndex]; font.pixelSize: 14; font.weight: Font.DemiBold; color: backend.theme.text }
                            Label { text: win.viewSubs[viewTabs.currentIndex]; font.pixelSize: 12; color: backend.theme.muted }
                            Item { Layout.fillWidth: true }
                            Button {
                                text: "Save figure\u2026"
                                icon.name: "document-save-as"
                                onClicked: win.exportFigure()
                                ToolTip.visible: hovered; ToolTip.delay: 600
                                ToolTip.text: "Open the figure editor for a publication-ready export"
                            }
                            Button {
                                text: "Reset view"
                                icon.name: "zoom-fit-best"
                                onClicked: win.resetView()
                            }
                        }
                        ProgressBar {
                            Layout.fillWidth: true
                            value: backend.progress
                            opacity: backend.busyVisible ? 1 : 0
                            Behavior on opacity { NumberAnimation { duration: 250 } }
                        }
                        Item {
                            id: views
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            property bool prewarm: true
                            Timer { interval: 1500; running: true; onTriggered: views.prewarm = false }
                            CurvePanel {
                                id: curvePanel
                                objectName: "curvePanel"
                                anchors.fill: parent
                                opacity: viewTabs.currentIndex === 0 ? 1 : 0
                                scale: viewTabs.currentIndex === 0 ? 1 : 0.985
                                visible: opacity > 0.01
                                Behavior on opacity { NumberAnimation { duration: 220 } }
                                Behavior on scale { NumberAnimation { duration: 260; easing.type: Easing.OutCubic } }
                            }
                            SurfacePanel {
                                id: surfacePanel
                                objectName: "surfacePanel"
                                anchors.fill: parent
                                opacity: viewTabs.currentIndex === 1 ? 1 : 0
                                scale: viewTabs.currentIndex === 1 ? 1 : 0.985
                                visible: opacity > 0.01 || views.prewarm
                                Behavior on opacity { NumberAnimation { duration: 220 } }
                                Behavior on scale { NumberAnimation { duration: 260; easing.type: Easing.OutCubic } }
                            }
                            HeatmapPanel {
                                id: heatmapPanel
                                objectName: "heatmapPanel"
                                anchors.fill: parent
                                opacity: viewTabs.currentIndex === 2 ? 1 : 0
                                scale: viewTabs.currentIndex === 2 ? 1 : 0.985
                                visible: opacity > 0.01
                                Behavior on opacity { NumberAnimation { duration: 220 } }
                                Behavior on scale { NumberAnimation { duration: 260; easing.type: Easing.OutCubic } }
                            }
                        }
                    }
                }

                Card {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 136
                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 12
                        anchors.bottomMargin: 6
                        spacing: 2
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10
                            Caption { text: "Overtones" }
                            Label { text: backend.overtones.legend || ""; textFormat: Text.RichText; font.pixelSize: 12; color: backend.theme.muted }
                            Item { Layout.fillWidth: true }
                            Label { text: backend.overtones.info || ""; font.pixelSize: 12; color: backend.theme.muted }
                        }
                        OvertonePanel { Layout.fillWidth: true; Layout.fillHeight: true }
                    }
                }
            }

            ColumnLayout {
                Layout.fillWidth: false
                Layout.preferredWidth: 360
                Layout.maximumWidth: 360
                Layout.fillHeight: true
                spacing: 10

                Segmented { id: resTabs; Layout.fillWidth: true; implicitHeight: 36; labels: ["Intervals", "Triads"] }

                Card {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Item {
                        anchors.fill: parent
                        anchors.margins: 8
                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 8
                            opacity: resTabs.currentIndex === 0 ? 1 : 0
                            visible: opacity > 0.01
                            Behavior on opacity { NumberAnimation { duration: 180 } }
                            Segmented {
                                Layout.fillWidth: true
                                implicitHeight: 30
                                labels: ["Most consonant", "By ratio"]
                                onActivated: (i) => backend.setSortMode(i)
                            }
                            ResultList {
                                objectName: "intervalList"
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                model: backend.intervalModel
                                linkedKey: backend.linkedInterval
                                emptyText: "No consonant minima found"
                                onPicked: (b, c) => backend.playInterval(b)
                                onHover: (key) => backend.hoverInterval(key)
                            }
                        }
                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 8
                            opacity: resTabs.currentIndex === 1 ? 1 : 0
                            visible: opacity > 0.01
                            Behavior on opacity { NumberAnimation { duration: 180 } }
                            Segmented {
                                Layout.fillWidth: true
                                implicitHeight: 30
                                labels: ["Within octave", "All voicings"]
                                onActivated: (i) => backend.setTriadMode(i)
                            }
                            ResultList {
                                objectName: "triadList"
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                model: backend.triadModel
                                linkedKey: backend.linkedTriad
                                emptyText: "No consonant triads found"
                                onPicked: (b, c) => backend.playTriad(b, c)
                                onHover: (key) => backend.hoverTriad(key)
                            }
                        }
                    }
                }

                Card {
                    id: insp
                    readonly property var ins: backend.inspector
                    Layout.fillWidth: true
                    Layout.preferredHeight: 252
                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 14
                        spacing: 7
                        RowLayout {
                            Layout.fillWidth: true
                            Caption { text: "Inspector" }
                            Item { Layout.fillWidth: true }
                            Label { text: insp.ins.kind || ""; color: backend.accent; font.pixelSize: 11; font.weight: Font.DemiBold; font.letterSpacing: 1.2 }
                        }
                        Label {
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            visible: insp.ins.empty === true
                            text: "Click a plot, a result or a play button to inspect the sound here."
                            wrapMode: Text.WordWrap
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                            color: backend.theme.muted
                        }
                        ColumnLayout {
                            visible: insp.ins.empty !== true
                            Layout.fillWidth: true
                            spacing: 5
                            Label { text: insp.ins.big || ""; font.family: backend.monoFamily; font.pixelSize: 19; font.bold: true; color: backend.theme.text }
                            Repeater {
                                model: insp.ins.rows || []
                                delegate: RowLayout {
                                    required property var modelData
                                    Label { text: modelData.k; Layout.preferredWidth: 112; font.pixelSize: 12; color: backend.theme.muted }
                                    Label { text: modelData.v; textFormat: Text.RichText; font.family: backend.monoFamily; font.pixelSize: 12; color: backend.theme.text }
                                }
                            }
                            RowLayout {
                                Label { text: "Dissonance"; Layout.preferredWidth: 112; font.pixelSize: 12; color: backend.theme.muted }
                                Label { text: insp.ins.dissText || ""; font.family: backend.monoFamily; font.pixelSize: 12; color: insp.ins.dissColor || backend.theme.muted }
                                Rectangle {
                                    visible: (insp.ins.diss || -1) >= 0
                                    width: 70; height: 5; radius: 2.5
                                    color: Qt.rgba(0.5, 0.5, 0.5, 0.25)
                                    Rectangle {
                                        width: Math.max(5, parent.width * Math.min(1, Math.max(0, insp.ins.diss || 0)))
                                        height: 5; radius: 2.5
                                        color: insp.ins.dissColor || "gray"
                                        Behavior on width { NumberAnimation { duration: 200 } }
                                    }
                                }
                            }
                            Button {
                                text: "Play again"
                                icon.name: "media-playback-start"
                                onClicked: backend.replay()
                                ToolTip.visible: hovered; ToolTip.delay: 600
                                ToolTip.text: "Replay this sound  (Space)"
                            }
                        }
                    }
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 10
            Rectangle {
                width: 8; height: 8; radius: 4
                color: backend.statusColor
                Behavior on color { ColorAnimation { duration: 250 } }
            }
            Label { Layout.fillWidth: true; text: backend.statusText; elide: Text.ElideRight; font.pixelSize: 12; color: backend.theme.subtext }
            Label { text: backend.viewHint; font.pixelSize: 12; color: backend.theme.muted }
        }
    }

    Rectangle {
        id: toast
        property string kind: "ok"
        property string message: ""
        property string action: ""
        property bool shown: false
        z: 100
        anchors.right: parent.right
        anchors.rightMargin: 20
        y: shown ? parent.height - height - 44 : parent.height - height - 24
        opacity: shown ? 1 : 0
        visible: opacity > 0.01
        Behavior on opacity { NumberAnimation { duration: 220 } }
        Behavior on y { NumberAnimation { duration: 280; easing.type: Easing.OutCubic } }
        width: toastRow.implicitWidth + 30
        height: Math.max(52, toastRow.implicitHeight + 22)
        radius: 8
        color: backend.theme.popup
        border.color: backend.theme.border
        layer.enabled: true
        layer.effect: MultiEffect {
            shadowEnabled: true
            shadowColor: "#90000000"
            shadowBlur: 0.7
            shadowVerticalOffset: 4
        }
        Rectangle {
            x: 1; y: 10; width: 3; height: parent.height - 20; radius: 1.5
            color: toast.kind === "error" ? backend.theme.red : backend.theme.green
        }
        RowLayout {
            id: toastRow
            x: 16
            anchors.verticalCenter: parent.verticalCenter
            spacing: 10
            Label {
                text: toast.kind === "error" ? "\uE7BA" : "\uE930"
                font.family: backend.iconFamily
                font.pixelSize: 16
                color: toast.kind === "error" ? backend.theme.red : backend.theme.green
            }
            Label { text: toast.message; color: backend.theme.text; Layout.maximumWidth: 360; wrapMode: Text.WordWrap }
            Button {
                visible: toast.action.length > 0
                text: toast.action
                flat: true
                onClicked: { backend.toastAction(); toast.shown = false }
            }
            ToolButton {
                text: "\uE711"
                font.family: backend.iconFamily
                font.pixelSize: 10
                onClicked: toast.shown = false
            }
        }
        Timer { id: toastTimer; interval: 4500; onTriggered: toast.shown = false }
        Connections {
            target: backend
            function onToast(message, kind, action) {
                toast.message = message
                toast.kind = kind
                toast.action = action
                toast.shown = true
                toastTimer.restart()
            }
        }
    }

    Image {
        id: themeSnap
        anchors.fill: parent
        z: 1000
        visible: false
        cache: false
        NumberAnimation {
            id: themeFade
            target: themeSnap
            property: "opacity"
            from: 1
            to: 0
            duration: 380
            easing.type: Easing.InOutQuad
            onFinished: { themeSnap.visible = false; themeSnap.source = "" }
        }
    }

    // ── figure editor ────────────────────────────────────────────────────────

    Dialog {
        id: figDialog
        readonly property var fig: backend.figureSession
        parent: Overlay.overlay
        anchors.centerIn: parent
        width: Math.min(1240, win.width - 60)
        height: Math.min(820, win.height - 60)
        modal: true
        visible: fig.open
        closePolicy: Popup.CloseOnEscape
        title: fig.title
        onClosed: fig.close()

        contentItem: RowLayout {
            spacing: 16

            ColumnLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 8
                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    radius: 8
                    color: backend.dark ? "#3a3a3a" : "#c9c9c9"
                    clip: true
                    Flickable {
                        id: page
                        anchors.fill: parent
                        contentWidth: Math.max(width, sheet.width + 48)
                        contentHeight: Math.max(height, sheet.height + 48)
                        boundsBehavior: Flickable.StopAtBounds
                        ScrollBar.vertical: ScrollBar { }
                        ScrollBar.horizontal: ScrollBar { }
                        Image {
                            id: sheet
                            x: (page.contentWidth - width) / 2
                            y: (page.contentHeight - height) / 2
                            width: figDialog.fig.previewWidth
                            height: figDialog.fig.previewHeight
                            source: figDialog.fig.preview
                            cache: false
                            smooth: true
                            layer.enabled: true
                            layer.effect: MultiEffect {
                                shadowEnabled: true
                                shadowColor: "#70000000"
                                shadowBlur: 0.6
                                shadowVerticalOffset: 3
                            }
                        }
                    }
                }
                Label {
                    Layout.fillWidth: true
                    text: figDialog.fig.message
                    elide: Text.ElideRight
                    font.pixelSize: 12
                    color: figDialog.fig.message.indexOf("failed") >= 0
                           || figDialog.fig.message.indexOf("must") >= 0 ? backend.theme.red
                                                                         : backend.theme.muted
                }
            }

            ColumnLayout {
                Layout.fillWidth: false
                Layout.preferredWidth: 360
                Layout.maximumWidth: 360
                Layout.fillHeight: true
                spacing: 10
                ScrollView {
                    id: figForm
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    contentWidth: availableWidth
                    clip: true
                    ColumnLayout {
                        width: figForm.availableWidth - 14
                        spacing: 6
                        Repeater {
                            model: figDialog.fig.fields
                            delegate: ColumnLayout {
                                id: field
                                required property var modelData
                                readonly property string type: modelData.type
                                Layout.fillWidth: true
                                spacing: 0
                                Caption {
                                    visible: field.type === "section"
                                    Layout.topMargin: 10
                                    text: field.modelData.label
                                }
                                Label {
                                    visible: field.type === "hint"
                                    Layout.fillWidth: true
                                    text: field.modelData.label
                                    textFormat: Text.PlainText
                                    wrapMode: Text.WordWrap
                                    font.pixelSize: 11
                                    color: backend.theme.muted
                                }
                                CheckBox {
                                    visible: field.type === "check"
                                    text: field.modelData.label
                                    checked: field.modelData.value === true
                                    onToggled: figDialog.fig.setValue(field.modelData.key, checked)
                                }
                                RowLayout {
                                    visible: field.type === "text" || field.type === "num" || field.type === "combo"
                                    Layout.fillWidth: true
                                    spacing: 8
                                    Label {
                                        text: field.modelData.label
                                        Layout.preferredWidth: 122
                                        elide: Text.ElideRight
                                        color: backend.theme.subtext
                                    }
                                    TextField {
                                        visible: field.type === "text" || field.type === "num"
                                        Layout.fillWidth: true
                                        text: field.type === "text" || field.type === "num" ? field.modelData.value : ""
                                        Component.onCompleted: cursorPosition = 0
                                        validator: field.type === "num" ? numCheck : null
                                        onTextEdited: field.type === "num"
                                                      ? figDialog.fig.setNumber(field.modelData.key, text)
                                                      : figDialog.fig.setValue(field.modelData.key, text)
                                        DoubleValidator {
                                            id: numCheck
                                            bottom: field.modelData.lo !== undefined ? field.modelData.lo : 0
                                            top: field.modelData.hi !== undefined ? field.modelData.hi : 0
                                            decimals: field.modelData.decimals !== undefined ? field.modelData.decimals : 0
                                            notation: DoubleValidator.StandardNotation
                                            locale: "C"
                                        }
                                    }
                                    ComboBox {
                                        visible: field.type === "combo"
                                        Layout.fillWidth: true
                                        model: field.modelData.options || []
                                        currentIndex: field.modelData.index !== undefined ? field.modelData.index : 0
                                        onActivated: (i) => figDialog.fig.setChoice(field.modelData.key, i)
                                    }
                                }
                            }
                        }
                    }
                }
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    Button {
                        Layout.fillWidth: true
                        text: "Save\u2026"
                        icon.name: "document-save"
                        highlighted: true
                        onClicked: figSaveDialog.open()
                    }
                    Button {
                        text: "Close"
                        onClicked: figDialog.close()
                    }
                }
            }
        }
    }

    FileDialog {
        id: figSaveDialog
        title: "Save figure"
        fileMode: FileDialog.SaveFile
        nameFilters: [figDialog.fig.fileFilter]
        defaultSuffix: figDialog.fig.fmt
        currentFolder: figDialog.fig.folder
        selectedFile: figDialog.fig.defaultFile
        onAccepted: figDialog.fig.save(selectedFile)
    }
}
"""


def _app_icon(accent):
    pm = QPixmap(64, 64)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    grad = QLinearGradient(0, 0, 0, 64)
    grad.setColorAt(0.0, QColor(accent).lighter(115))
    grad.setColorAt(1.0, QColor(accent).darker(135))
    p.setPen(Qt.NoPen)
    p.setBrush(grad)
    p.drawRoundedRect(QRectF(2, 2, 60, 60), 14, 14)
    font = QFont("Segoe UI")
    font.setPixelSize(44)
    font.setBold(True)
    p.setFont(font)
    p.setPen(QColor("white"))
    p.drawText(QRectF(0, 0, 64, 60), Qt.AlignCenter, "\u223f")
    p.end()
    return QIcon(pm)


def main():
    # Windows rounds the default 5 ms GIL hand-off up to its timer tick, starving the worker.
    sys.setswitchinterval(0.0005)
    if sys.platform == "win32":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("DissonanceQuick")
        except Exception:
            pass
    QQuickStyle.setStyle("FluentWinUI3")
    # The publication-figure export renders the 3D surface with pyqtgraph's OpenGL widget.
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    app.setApplicationName("Dissonance Quick")
    QGuiApplication.styleHints().setColorScheme(Qt.ColorScheme.Dark)
    qmlRegisterType(HeatmapItem, "Dissonance", 1, 0, "HeatmapItem")

    engine = QQmlApplicationEngine()
    engine.warnings.connect(lambda ws: [print("QML:", w.toString(), file=sys.stderr) for w in ws])
    engine.addImageProvider("frames", FrameProvider())
    backend = Backend()
    app.setWindowIcon(_app_icon(backend.accent))
    engine.rootContext().setContextProperty("backend", backend)
    engine.loadData(QByteArray(QML.encode("utf-8")), QUrl("dissonance_quick.qml"))
    if not engine.rootObjects():
        sys.exit("The QML interface failed to load")
    root = engine.rootObjects()[0]
    backend.attach(root)
    root.showMaximized()
    app.aboutToQuit.connect(backend.shutdown)
    code = app.exec()
    # Tear the QML scene down before the backend so no binding sees a null context property.
    del root
    del engine
    sys.exit(code)


if __name__ == "__main__":
    main()
