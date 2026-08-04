"""
Dissonance Explorer — a modern, GPU-accelerated interface for the
Plomp-Levelt / Sethares sensory-dissonance model.

This is a standalone rewrite of `dissonance_gui.py` using PySide6 + pyqtgraph.
The mathematics (dissonance model, minima detection, additive synthesis) is
carried over unchanged; only the presentation layer is new.

Features
--------
* Three independent spectra (A, B, C) with live partial previews
* Interactive 2D dissonance curve with crosshair readout, zoom/pan, click-to-play
* Heatmap of the triad space with topographic iso-contours
* True OpenGL 3D dissonance surface with accurate ray-cast click-to-play
* Ranked lists of consonant intervals and consonant triads
* Additive-synthesis playback of everything

Requirements: numpy, sounddevice, PySide6, pyqtgraph, PyOpenGL
Run:          python dissonance_modern.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import sounddevice as sd

from PySide6.QtCore import Qt, QThread, Signal, QRectF, QSize, QTimer, QPointF
from PySide6.QtGui import (QColor, QFont, QImage, QPainter, QPalette, QPen,
                           QShortcut, QKeySequence, QVector4D)
from PySide6.QtWidgets import (QApplication, QButtonGroup, QComboBox, QFrame,
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
#  Dissonance model  (unchanged from dissonance_gui.py)
# ═══════════════════════════════════════════════════════════════════════════

SAMPLE_RATE = 44100
DURATION = 1.5


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

        z_data[ri, :] = (d_aa + d_bb_all[ri] + d_ab_all[ri] +
                         d_cc_all + d_ac_all + d_bc) / 2.0

    max_z = z_data.max()
    if max_z > 0:
        z_data /= max_z
    if progress_callback:
        progress_callback(1.0)
    return ratios, ratios, z_data


def find_minima(ratios, dissonance_values, window=5, min_prominence=0.0000135):
    """Find local minima with significant prominence (depth relative to surroundings).

    A minimum is only kept if the difference between the highest nearby peak
    and the minimum value is at least min_prominence (on the 0-1 normalized scale).
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

        # Left: scan left until we find a value higher than val, track the max on the way
        left_max = val
        for j in range(idx - 1, -1, -1):
            left_max = max(left_max, dissonance_values[j])
            if dissonance_values[j] > val:
                break

        # Right: scan right
        right_max = val
        for j in range(idx + 1, n):
            right_max = max(right_max, dissonance_values[j])
            if dissonance_values[j] > val:
                break

        # Prominence = height of the lower of the two surrounding peaks minus the valley
        prominence = min(left_max, right_max) - val

        if prominence >= min_prominence:
            minima_x.append(ratios[idx])
            minima_y.append(dissonance_values[idx])

    return np.array(minima_x), np.array(minima_y)


def find_minima_3d(ratios_x, ratios_y, z_data, window=5, top_n=10):
    """Find the most consonant triads (lowest dissonance local minima).

    A grid point is a local minimum if it's the smallest value in its
    (2*window+1) x (2*window+1) neighbourhood.  Results are sorted by
    absolute dissonance (lowest first) and capped at *top_n*.
    """
    ny, nx = z_data.shape
    candidates = []

    for i in range(window, ny - window):
        for j in range(window, nx - window):
            patch = z_data[i - window:i + window + 1, j - window:j + window + 1]
            if z_data[i, j] == patch.min():
                candidates.append((i, j))

    results = []
    tol = 0.02  # tolerance for "same note" check
    for i, j in candidates:
        bx = ratios_x[j]
        cy = ratios_y[i]
        # Only keep notes strictly within one octave (ratio < 2)
        if bx >= 2.0 - tol or cy >= 2.0 - tol:
            continue
        # Skip triads where two voices are effectively the same pitch
        if abs(bx - 1.0) < tol or abs(cy - 1.0) < tol or abs(bx - cy) < tol:
            continue
        # Keep only B >= C (below the x=y diagonal) to avoid mirror duplicates
        if bx < cy:
            continue
        results.append((bx, cy, z_data[i, j]))

    if not results:
        return np.array([]), np.array([]), np.array([])

    # Sort by dissonance (most consonant first), keep top_n
    results.sort(key=lambda t: t[2])
    results = results[:top_n]

    rx, ry, rz = zip(*results)
    return np.array(rx), np.array(ry), np.array(rz)


def synthesize_multi(spectra_and_ratios, ref_freq, duration=DURATION):
    """Synthesize multiple voices, each with its own spectrum and pitch ratio."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    envelope = np.ones_like(t)
    fade = int(0.05 * SAMPLE_RATE)
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


# ═══════════════════════════════════════════════════════════════════════════
#  Interval labelling  (unchanged)
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
    ("Odd harmonics (square)", [1, 3, 5, 7, 9, 11], [0.88 ** i for i in range(6)]),
    ("Free beam (xylophone)", [1, 2.758, 5.406, 8.936, 13.35, 18.645],
     [0.88 ** i for i in range(6)]),
    ("Bonang", [1, 1.52, 3.46, 3.92], [0.88 ** i for i in range(4)]),
]

DEFAULT_FREQS = [1, 2, 3, 4, 5, 6]
DEFAULT_AMPS = [1, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 6]


# ═══════════════════════════════════════════════════════════════════════════
#  Theme
# ═══════════════════════════════════════════════════════════════════════════

CRUST = "#0e0e16"
BASE = "#11111b"
MANTLE = "#181825"
SURFACE = "#1e1e2e"
PANEL = "#313244"
LINE = "#26263a"
BORDER = "#45475a"
TEXT = "#cdd6f4"
SUBTEXT = "#a6adc8"
MUTED = "#6c7086"
BLUE = "#89b4fa"
SKY = "#89dceb"
YELLOW = "#f9e2af"
GREEN = "#a6e3a1"
RED = "#f38ba8"
MAUVE = "#cba6f7"
PEACH = "#fab387"

VOICE_COLORS = {"A": BLUE, "B": MAUVE, "C": PEACH}

QSS = f"""
* {{ outline: none; }}

QWidget {{
    background: {BASE};
    color: {TEXT};
    font-family: "Segoe UI Variable Display", "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}}

QLabel {{ background: transparent; }}

#appTitle   {{ font-size: 19px; font-weight: 700; color: {TEXT}; letter-spacing: 0.5px; }}
#appSub     {{ font-size: 11px; color: {MUTED}; letter-spacing: 2.2px; font-weight: 600; }}
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

#dot {{ border-radius: 4px; }}

QLineEdit, QComboBox {{
    background: {SURFACE};
    border: 1px solid {PANEL};
    border-radius: 8px;
    padding: 6px 10px;
    color: {TEXT};
    selection-background-color: {BLUE};
    selection-color: {BASE};
}}
QLineEdit:hover, QComboBox:hover {{ border-color: {BORDER}; }}
QLineEdit:focus, QComboBox:focus {{ border-color: {BLUE}; }}
QLineEdit[invalid="true"] {{ border-color: {RED}; }}

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
    padding: 7px 14px;
    color: {TEXT};
}}
QPushButton:hover  {{ background: {PANEL}; border-color: {BORDER}; }}
QPushButton:pressed {{ background: {BORDER}; }}

QPushButton#primary {{
    background: {BLUE}; color: {BASE}; border: none; font-weight: 700;
    padding: 9px 16px;
}}
QPushButton#primary:hover  {{ background: #a3c6ff; }}
QPushButton#primary:pressed {{ background: #6f9de6; }}
QPushButton#primary:disabled {{ background: {PANEL}; color: {MUTED}; }}

QPushButton#danger {{ background: {SURFACE}; color: {RED}; border: 1px solid #4a2f3c; }}
QPushButton#danger:hover {{ background: {RED}; color: {BASE}; border-color: {RED}; }}

QPushButton#icon {{
    background: {SURFACE}; border: 1px solid {PANEL}; border-radius: 8px;
    padding: 6px 0px; min-width: 34px; color: {GREEN}; font-size: 12px;
}}
QPushButton#icon:hover {{ background: {GREEN}; color: {BASE}; border-color: {GREEN}; }}

#segment {{ background: {MANTLE}; border: 1px solid {LINE}; border-radius: 10px; }}
QPushButton#segBtn {{
    background: transparent; border: none; border-radius: 7px;
    padding: 7px 18px; color: {MUTED}; font-weight: 600;
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
    name = f"dissonance_chevron_{'hover' if hover else 'idle'}.png"
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


def build_qss():
    return (QSS.replace("__CHEVRON__", _chevron(SUBTEXT))
               .replace("__CHEVRON_HOVER__", _chevron(BLUE, hover=True)))


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
    """Rounded surface with an optional small-caps title."""

    def __init__(self, title=None, accent=None, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 14)
        outer.setSpacing(10)

        if title is not None:
            head = QHBoxLayout()
            head.setSpacing(8)
            if accent:
                dot = QLabel()
                dot.setObjectName("dot")
                dot.setFixedSize(8, 8)
                dot.setStyleSheet(f"background: {accent}; border-radius: 4px;")
                head.addWidget(dot)
            lbl = QLabel(title.upper())
            lbl.setObjectName("cardTitle")
            head.addWidget(lbl)
            head.addStretch(1)
            self.head = head
            outer.addLayout(head)

        self.body = QVBoxLayout()
        self.body.setSpacing(8)
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

    def setIndex(self, index):
        btn = self.group.button(index)
        if btn:
            btn.setChecked(True)


class Field(QWidget):
    """Labelled line edit."""

    def __init__(self, label, value="", width=None, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        cap = QLabel(label)
        cap.setObjectName("fieldLabel")
        lay.addWidget(cap)
        self.edit = QLineEdit(value)
        if width:
            self.edit.setFixedWidth(width)
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

class SpectrumCard(Card):
    """Editor for one voice's spectrum, with a live partial preview."""

    playRequested = Signal(object)

    def __init__(self, letter, subtitle, copy_sources=None, parent=None):
        super().__init__(f"{letter}  \u00b7  {subtitle}", accent=VOICE_COLORS[letter],
                         parent=parent)
        self.letter = letter
        self.copy_sources = copy_sources or []
        self.color = VOICE_COLORS[letter]

        if self.copy_sources:
            self.copy_combo = QComboBox()
            self.copy_combo.addItems(
                [f"Same as {name}" for name, _ in self.copy_sources] + ["Custom"])
            self.copy_combo.setCurrentIndex(0)
            self.copy_combo.currentIndexChanged.connect(self._toggle_fields)
            self.body.addWidget(self.copy_combo)
        else:
            self.copy_combo = None

        self.fields = QWidget()
        fl = QVBoxLayout(self.fields)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(8)

        self.freq_field = Field("PARTIAL RATIOS",
                                ", ".join(f"{f:.4g}" for f in DEFAULT_FREQS))
        self.amp_field = Field("AMPLITUDES",
                               ", ".join(f"{a:.4g}" for a in DEFAULT_AMPS))
        fl.addWidget(self.freq_field)
        fl.addWidget(self.amp_field)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Presets\u2026")
        self.preset_combo.addItems([p[0] for p in PRESETS])
        self.preset_combo.currentIndexChanged.connect(self._apply_preset)
        row.addWidget(self.preset_combo, 1)
        play_btn = QPushButton("\u25b6")
        play_btn.setObjectName("icon")
        play_btn.setToolTip(f"Play voice {letter} alone")
        play_btn.setCursor(Qt.PointingHandCursor)
        play_btn.clicked.connect(lambda: self.playRequested.emit(self))
        row.addWidget(play_btn, 0)
        fl.addLayout(row)

        self.preview = pg.PlotWidget()
        self.preview.setFixedHeight(58)
        self.preview.setMouseEnabled(False, False)
        self.preview.setMenuEnabled(False)
        self.preview.hideButtons()
        self.preview.setBackground(SURFACE)
        for axis in ("left", "bottom", "top", "right"):
            self.preview.getPlotItem().hideAxis(axis)
        self.preview.getPlotItem().setContentsMargins(6, 6, 6, 2)
        self._bars = pg.BarGraphItem(x=[0], height=[0], width=0.06,
                                     brush=pg.mkBrush(self.color))
        self.preview.addItem(self._bars)
        fl.addWidget(self.preview)

        self.body.addWidget(self.fields)

        self.freq_field.edit.textChanged.connect(self._refresh_preview)
        self.amp_field.edit.textChanged.connect(self._refresh_preview)

        self._toggle_fields()
        self._refresh_preview()

    # ── behaviour ──────────────────────────────────────────────────────────

    def _toggle_fields(self):
        custom = self.copy_combo is None or \
            self.copy_combo.currentText() == "Custom"
        self.fields.setVisible(custom)
        if not custom:
            self._refresh_preview()

    def _apply_preset(self, index):
        if index <= 0:
            return
        _, freqs, amps = PRESETS[index - 1]
        self.freq_field.edit.setText(", ".join(f"{f:.4g}" for f in freqs))
        self.amp_field.edit.setText(", ".join(f"{a:.4g}" for a in amps))
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(0)
        self.preset_combo.blockSignals(False)

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
        if self.copy_combo is not None and self.copy_combo.currentText() != "Custom":
            for name, card in self.copy_sources:
                if self.copy_combo.currentText() == f"Same as {name}":
                    return card.get_spectrum()
            return None
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


class CurveView(pg.PlotWidget):
    """2D dissonance curve with crosshair readout."""

    picked = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground(MANTLE)
        self.showGrid(x=True, y=True, alpha=0.12)
        pi = self.getPlotItem()
        pi.setLabel('bottom', 'frequency ratio  B / A', color=SUBTEXT, size='10pt')
        pi.setLabel('left', 'dissonance', color=SUBTEXT, size='10pt')
        for axis in ('bottom', 'left'):
            ax = pi.getAxis(axis)
            ax.setPen(pg.mkPen(BORDER))
            ax.setTextPen(pg.mkPen(SUBTEXT))
        pi.setMenuEnabled(False)

        self._curve = pi.plot([], [], pen=pg.mkPen(BLUE, width=2),
                              fillLevel=0.0, brush=pg.mkBrush(137, 180, 250, 32))
        self._dots = pg.ScatterPlotItem(size=9, pen=pg.mkPen('#ffffff', width=1.0),
                                        brush=pg.mkBrush(YELLOW))
        pi.addItem(self._dots)

        self._vline = pg.InfiniteLine(angle=90, movable=False,
                                      pen=pg.mkPen(MUTED, width=1,
                                                   style=Qt.DashLine))
        self._vline.setVisible(False)
        pi.addItem(self._vline, ignoreBounds=True)

        self._readout = pg.TextItem(anchor=(0, 1), color=TEXT,
                                    fill=pg.mkBrush(30, 30, 46, 225),
                                    border=pg.mkPen(PANEL))
        self._readout.setVisible(False)
        pi.addItem(self._readout, ignoreBounds=True)

        self._labels = []
        self._x = None
        self._y = None

        self.scene().sigMouseMoved.connect(self._on_move)
        self.scene().sigMouseClicked.connect(self._on_click)

    def setData(self, ratios, values, minima_x, minima_y, xmax):
        self._x, self._y = ratios, values
        self._curve.setData(ratios, values)
        for item in self._labels:
            self.getPlotItem().removeItem(item)
        self._labels.clear()

        if len(minima_x):
            self._dots.setData(minima_x, minima_y)
            for x, y in zip(minima_x, minima_y):
                t = pg.TextItem(f"{ratio_to_interval_name(x)}\n{x:.3f}",
                                anchor=(0.5, 0.0), color=YELLOW)
                t.setFont(QFont("Segoe UI", 7))
                t.setPos(x, y - 0.012)
                self.getPlotItem().addItem(t, ignoreBounds=True)
                self._labels.append(t)
        else:
            self._dots.setData([], [])

        self.setXRange(0.9, xmax, padding=0.01)
        self.setYRange(0, float(np.max(values)) * 1.08, padding=0)

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


class HeatmapView(pg.PlotWidget):
    """Triad space as an image with iso-contours."""

    picked = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground(MANTLE)
        pi = self.getPlotItem()
        pi.setLabel('bottom', 'voice B ratio', color=SUBTEXT, size='10pt')
        pi.setLabel('left', 'voice C ratio', color=SUBTEXT, size='10pt')
        for axis in ('bottom', 'left'):
            ax = pi.getAxis(axis)
            ax.setPen(pg.mkPen(BORDER))
            ax.setTextPen(pg.mkPen(SUBTEXT))
        pi.setMenuEnabled(False)
        pi.setAspectLocked(True)

        self.cmap = _magma()
        self.img = SmoothImageItem()
        self.img.setColorMap(self.cmap)
        pi.addItem(self.img)

        self.bar = pg.ColorBarItem(colorMap=self.cmap, interactive=False,
                                   label='dissonance', width=12)
        self.bar.setImageItem(self.img, insert_in=pi)
        self.bar.axis.setPen(pg.mkPen(BORDER))
        self.bar.axis.setTextPen(pg.mkPen(SUBTEXT))

        self._iso = []
        self._dots = pg.ScatterPlotItem(size=12, pen=pg.mkPen('#ffffff', width=1.4),
                                        brush=pg.mkBrush(YELLOW))
        pi.addItem(self._dots)
        self._labels = []
        self._x = self._y = self._z = None

        self.scene().sigMouseClicked.connect(self._on_click)

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
                t = pg.TextItem(f"{bx:.2f} \u00b7 {cy:.2f}", anchor=(0, 1), color='w')
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


def _magma():
    try:
        return pg.colormap.getFromMatplotlib('magma')
    except Exception:
        return pg.colormap.get('CET-L8')


if HAS_GL:

    from OpenGL import GL as _GL

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
            self.setBackgroundColor(QColor(MANTLE))
            self.setCameraPosition(distance=4.6, elevation=25, azimuth=69)
            self.opts['fov'] = 50

            grid = gl.GLGridItem()
            grid.setSize(2.0, 2.0)
            grid.setSpacing(0.2, 0.2)
            grid.setColor(QColor(69, 71, 90, 110))
            self.addItem(grid)

            self._surface = None
            self._dots = None
            self._text = []
            self._press = None
            self._x = self._y = self._z = None
            self._zs = None
            self._cmap = _magma()

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

        @staticmethod
        def _norm(values, axis):
            lo, hi = axis[0], axis[-1]
            return (np.asarray(values, dtype=float) - lo) / max(hi - lo, 1e-12) * 2.0 - 1.0

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
            self.setCameraPosition(distance=4.6, elevation=25, azimuth=69)

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
        spec_a, spec_b, spec_c, ref_freq, max_interval, step_2d, step_3d = self.args
        try:
            ratios_2d, diss_2d = compute_dissonance_2d(
                spec_a, spec_b, ref_freq, max_interval, step_2d)
            minima_x, minima_y = find_minima(ratios_2d, diss_2d)

            x3, y3, z3 = compute_dissonance_3d(
                spec_a, spec_b, spec_c, ref_freq, max_interval, step_3d,
                progress_callback=lambda p: self.progress.emit(int(p * 100)))

            tx, ty, tz = find_minima_3d(x3, y3, z3)
        except Exception as exc:  # surfaced in the status bar
            self.failed.emit(str(exc))
            return

        self.finished_ok.emit({
            "ratios_2d": ratios_2d, "diss_2d": diss_2d,
            "minima_x": minima_x, "minima_y": minima_y,
            "x_3d": x3, "y_3d": y3, "z_3d": z3,
            "tri_x": tx, "tri_y": ty, "tri_z": tz,
        })


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

        self._build_ui()
        QTimer.singleShot(80, self.analyse)

    # ── construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        lay = QHBoxLayout(root)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(12)

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
        scroll.setFixedWidth(392)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        col = QVBoxLayout(inner)
        col.setContentsMargins(0, 0, 8, 0)
        col.setSpacing(12)

        header = QVBoxLayout()
        header.setSpacing(2)
        sub = QLabel("SENSORY DISSONANCE")
        sub.setObjectName("appSub")
        title = QLabel("Dissonance Explorer")
        title.setObjectName("appTitle")
        header.addWidget(sub)
        header.addWidget(title)
        col.addLayout(header)

        self.card_a = SpectrumCard("A", "root voice")
        self.card_b = SpectrumCard("B", "moving voice",
                                   copy_sources=[("A", self.card_a)])
        self.card_c = SpectrumCard("C", "third voice",
                                   copy_sources=[("A", self.card_a),
                                                 ("B", self.card_b)])
        for card in (self.card_a, self.card_b, self.card_c):
            card.playRequested.connect(self._play_card)
            col.addWidget(card)

        settings = Card("analysis")
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        self.f_ref = Field("FUNDAMENTAL (Hz)", "261.63")
        self.f_max = Field("MAX RATIO", "2.3")
        self.f_step = Field("3D STEP", "0.005")
        grid.addWidget(self.f_ref, 0, 0)
        grid.addWidget(self.f_max, 0, 1)
        grid.addWidget(self.f_step, 1, 0)
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
        row1 = QHBoxLayout()
        row1.setSpacing(8)
        self.f_interval = Field("INTERVAL  A + B", "1.5")
        row1.addWidget(self.f_interval, 1)
        b1 = QPushButton("\u25b6")
        b1.setObjectName("icon")
        b1.setCursor(Qt.PointingHandCursor)
        b1.clicked.connect(self._play_manual_interval)
        row1.addWidget(b1, 0, Qt.AlignBottom)
        manual.body.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(8)
        self.f_triad = Field("TRIAD  B, C", "1.25, 1.5")
        row2.addWidget(self.f_triad, 1)
        b2 = QPushButton("\u25b6")
        b2.setObjectName("icon")
        b2.setCursor(Qt.PointingHandCursor)
        b2.clicked.connect(self._play_manual_triad)
        row2.addWidget(b2, 0, Qt.AlignBottom)
        manual.body.addLayout(row2)
        col.addWidget(manual)

        col.addStretch(1)
        scroll.setWidget(inner)
        return scroll

    def _build_plot_area(self):
        wrap = QWidget()
        col = QVBoxLayout(wrap)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(10)

        top = QHBoxLayout()
        top.setSpacing(12)
        self.tabs = Segmented(["Curve", "Surface", "Heatmap"])
        self.tabs.changed.connect(self._switch_tab)
        top.addWidget(self.tabs, 0)
        self.view_hint = QLabel()
        self.view_hint.setObjectName("hint")
        top.addWidget(self.view_hint, 0)
        top.addStretch(1)

        self.btn_reset_view = QPushButton("Reset view")
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
        self.stack.addWidget(self.surface_view)
        self.stack.addWidget(self.heatmap_view)
        inner.addWidget(self.stack)
        col.addWidget(card, 1)

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
        status.addWidget(self.progress, 0)
        col.addLayout(status)
        return wrap

    def _build_results(self):
        wrap = QWidget()
        col = QVBoxLayout(wrap)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(10)

        self.res_tabs = Segmented(["Intervals", "Triads"])
        self.res_tabs.changed.connect(lambda i: self.res_stack.setCurrentIndex(i))
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

        intervals_page = QWidget()
        ip = QVBoxLayout(intervals_page)
        ip.setContentsMargins(0, 0, 0, 0)
        ip.setSpacing(8)
        ip.addLayout(sort_row)
        self.list_intervals = ResultList()
        self.list_intervals.activatedRow.connect(
            lambda p: self._play_interval_ratio(p))
        ip.addWidget(self.list_intervals, 1)

        triads_page = QWidget()
        tp = QVBoxLayout(triads_page)
        tp.setContentsMargins(0, 0, 0, 0)
        tp.setSpacing(8)
        hint = QLabel("Ranked by lowest dissonance")
        hint.setObjectName("hint")
        hint.setContentsMargins(8, 2, 0, 0)
        tp.addWidget(hint)
        self.list_triads = ResultList()
        self.list_triads.activatedRow.connect(
            lambda p: self._play_triad_ratio(p[0], p[1]))
        tp.addWidget(self.list_triads, 1)

        self.res_stack.addWidget(intervals_page)
        self.res_stack.addWidget(triads_page)
        inner.addWidget(self.res_stack)
        col.addWidget(card, 1)

        wrap.setMinimumWidth(300)
        return wrap

    # ── helpers ────────────────────────────────────────────────────────────

    def _set_dot(self, color):
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

    def _reset_view(self):
        i = self.stack.currentIndex()
        if i == 0 and self.data is not None:
            self.curve_view.setXRange(0.9, self.max_interval, padding=0.01)
            self.curve_view.setYRange(
                0, float(self.data["diss_2d"].max()) * 1.08, padding=0)
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
                                    self.max_interval, self.step_2d, self.step_3d)
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
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)

        self.curve_view.setData(data["ratios_2d"], data["diss_2d"],
                                data["minima_x"], data["minima_y"],
                                self.max_interval)
        self.heatmap_view.setData(data["x_3d"], data["y_3d"], data["z_3d"],
                                  data["tri_x"], data["tri_y"], data["tri_z"])
        self.surface_view.setData(data["x_3d"], data["y_3d"], data["z_3d"],
                                  data["tri_x"], data["tri_y"], data["tri_z"])
        self._fill_intervals()
        self._fill_triads()
        self._status("Ready \u2014 click any plot or list entry to hear it", GREEN)

    def _fill_intervals(self):
        if self.data is None:
            return
        pairs = list(zip(self.data["minima_x"], self.data["minima_y"]))
        if self.sort_seg.group.checkedId() == 0:
            pairs.sort(key=lambda p: p[1])
        else:
            pairs.sort(key=lambda p: p[0])
        rows = [(f"{x:.4f}",
                 f"{ratio_to_cents(x):.0f}\u00a2  \u00b7  {ratio_to_interval_name(x)}",
                 float(y), float(x))
                for x, y in pairs]
        self.list_intervals.setRows(rows, "No consonant minima found")

    def _fill_triads(self):
        if self.data is None:
            return
        triads = list(zip(self.data["tri_x"], self.data["tri_y"], self.data["tri_z"]))
        triads.sort(key=lambda t: t[2])
        rows = [(f"1.000 : {bx:.3f} : {cy:.3f}",
                 f"{ratio_to_interval_name(bx)}  \u00b7  {ratio_to_interval_name(cy)}",
                 float(dz), (float(bx), float(cy)))
                for bx, cy, dz in triads]
        self.list_triads.setRows(rows, "No consonant triads found")

    # ── playback ───────────────────────────────────────────────────────────

    def _play_card(self, card):
        if not self._read_settings():
            return
        spec = card.get_spectrum()
        if spec is None:
            self._status(f"Voice {card.letter} has invalid input", RED)
            return
        self._status(f"Playing voice {card.letter} alone", GREEN)
        play_multi([(spec, 1.0)], self.ref_freq)

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
        play_multi([(spec_a, 1.0), (spec_b, ratio)], self.ref_freq)

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
        play_multi([(spec_a, 1.0), (spec_b, rb), (spec_c, rc)], self.ref_freq)

    def _play_manual_interval(self):
        try:
            ratio = float(self.f_interval.text())
        except ValueError:
            self._status("Interval must be a single number", RED)
            return
        self._play_interval_ratio(ratio)

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

    # ── lifecycle ──────────────────────────────────────────────────────────

    def closeEvent(self, event):
        sd.stop()
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(3000)
        super().closeEvent(event)


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    pg.setConfigOptions(antialias=True, background=MANTLE, foreground=TEXT)

    app = QApplication(sys.argv)
    app.setApplicationName("Dissonance Explorer")
    app.setStyle("Fusion")

    palette = app.palette()
    palette.setColor(QPalette.Window, QColor(BASE))
    palette.setColor(QPalette.Base, QColor(SURFACE))
    palette.setColor(QPalette.Text, QColor(TEXT))
    palette.setColor(QPalette.WindowText, QColor(TEXT))
    palette.setColor(QPalette.Highlight, QColor(BLUE))
    palette.setColor(QPalette.HighlightedText, QColor(BASE))
    app.setPalette(palette)
    app.setStyleSheet(build_qss())

    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
