"""
Dissonance GUI: All-in-one interactive application.

Features:
- Three independent spectra (A, B, C) for full triad customization
- 2D dissonance curve (A vs B intervals) with click-to-play
- 3D dissonance heatmap + 3D surface (A vs B vs C triads) with click-to-play
- Interval labels showing cents deviation from nearest 12-TET
- Audio synthesis
- Preset spectra

Requirements: numpy, matplotlib, sounddevice
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams.update({
    'figure.facecolor': '#1e1e2e',
    'axes.facecolor': '#1e1e2e',
    'axes.edgecolor': '#6c7086',
    'axes.labelcolor': '#cdd6f4',
    'text.color': '#cdd6f4',
    'xtick.color': '#a6adc8',
    'ytick.color': '#a6adc8',
    'grid.color': '#45475a',
    'grid.alpha': 0.5,
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
})
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import sounddevice as sd
import threading

# ─── Theme Colors ─────────────────────────────────────────────────────────────

BG = '#1e1e2e'
BG_SURFACE = '#181825'
BG_PANEL = '#313244'
FG = '#cdd6f4'
FG_DIM = '#a6adc8'
ACCENT = '#89b4fa'
ACCENT2 = '#f9e2af'
BORDER = '#45475a'

# ─── Dissonance Model ─────────────────────────────────────────────────────────

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

        # Find the highest point between this minimum and the next higher point
        # on each side (or the edge of the data)
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


# ─── Interval Labeling ────────────────────────────────────────────────────────

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

    # Show deviation
    if abs(deviation_cents) < 2:
        return name
    elif deviation_cents > 0:
        return f"{name} +{deviation_cents:.0f}\u00a2"
    else:
        return f"{name} {deviation_cents:.0f}\u00a2"


def ratio_to_cents(ratio):
    return 1200 * np.log2(ratio)


# ─── Presets ──────────────────────────────────────────────────────────────────

PRESETS = [
    # Sethares' default: 6 harmonic partials, 1/n amplitude (matches original JS)
    ("Harmonic", [1, 2, 3, 4, 5, 6], [1, 1/2, 1/3, 1/4, 1/5, 1/6]),
    # Odd harmonics only (square wave approximation)
    ("Odd harmonics (square)", [1, 3, 5, 7, 9, 11], [0.88**i for i in range(6)]),
    # Free beam / xylophone (Fletcher & Rossing, cited by Sethares)
    ("Free beam (xylophone)", [1, 2.758, 5.406, 8.936, 13.35, 18.645],
     [0.88**i for i in range(6)]),
    # Bonang (Sethares' TTSS measurements)
    ("Bonang", [1, 1.52, 3.46, 3.92], [0.88**i for i in range(4)]),
]


# ─── GUI Application ──────────────────────────────────────────────────────────

class SpectrumPanel:
    """Reusable widget for configuring a single spectrum."""

    def __init__(self, parent, label, default_freqs, default_amps, copy_sources=None):
        self.frame = ttk.LabelFrame(parent, text=label, padding=8)
        self.copy_sources = copy_sources or []  # List of (name, SpectrumPanel) to copy from
        pad = {'padx': 6, 'pady': 3}

        # "Copy from" dropdown (only if copy_sources given)
        self.copy_mode = tk.StringVar(value="Same as A" if self.copy_sources else "Custom")
        if self.copy_sources:
            copy_frame = ttk.Frame(self.frame)
            copy_frame.pack(**pad, anchor='w', fill='x')
            options = [f"Same as {name}" for name, _ in self.copy_sources] + ["Custom"]
            self.copy_combo = ttk.Combobox(copy_frame, textvariable=self.copy_mode,
                                           values=options, state='readonly', width=16)
            self.copy_combo.pack(side=tk.LEFT)
            self.copy_combo.bind("<<ComboboxSelected>>", lambda e: self._toggle())

        # Fields frame (hidden when "same as" is checked)
        self.fields = ttk.Frame(self.frame)
        self.fields.pack(**pad, anchor='w', fill='x')

        ttk.Label(self.fields, text="Freq ratios:").pack(**pad, anchor='w')
        self.freq_var = tk.StringVar(value=", ".join(f"{f:.4g}" for f in default_freqs))
        ttk.Entry(self.fields, textvariable=self.freq_var, width=34).pack(**pad, anchor='w')

        ttk.Label(self.fields, text="Amplitudes:").pack(**pad, anchor='w')
        self.amp_var = tk.StringVar(value=", ".join(f"{a:.4g}" for a in default_amps))
        ttk.Entry(self.fields, textvariable=self.amp_var, width=34).pack(**pad, anchor='w')

        # Preset combo
        preset_frame = ttk.Frame(self.fields)
        preset_frame.pack(**pad, anchor='w', fill='x')
        ttk.Label(preset_frame, text="Preset:").pack(side=tk.LEFT)
        self.preset_combo = ttk.Combobox(preset_frame,
            values=[p[0] for p in PRESETS], state='readonly', width=20)
        self.preset_combo.set("Harmonic (default)")
        self.preset_combo.pack(side=tk.LEFT, padx=4)
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset)

        # Play button
        ttk.Button(self.fields, text="\u25b6 Play", command=self._play).pack(**pad, anchor='w')

        # Initial toggle
        if self.copy_sources:
            self._toggle()

        self._play_callback = None
        self._ref_freq_getter = None

    def set_play_callback(self, callback, ref_freq_getter):
        self._play_callback = callback
        self._ref_freq_getter = ref_freq_getter

    def _play(self):
        if self._play_callback and self._ref_freq_getter:
            spec = self.get_spectrum()
            if spec:
                self._play_callback([(spec, 1.0)], self._ref_freq_getter())

    def _toggle(self):
        if self.copy_sources and self.copy_mode.get() != "Custom":
            self.fields.pack_forget()
        else:
            self.fields.pack(padx=4, pady=2, anchor='w', fill='x')

    def _on_preset(self, event=None):
        idx = self.preset_combo.current()
        if idx >= 0:
            _, freqs, amps = PRESETS[idx]
            self.freq_var.set(", ".join(f"{f:.4g}" for f in freqs))
            self.amp_var.set(", ".join(f"{a:.4g}" for a in amps))

    def get_spectrum(self):
        """Returns spectrum dict or copies from source. Returns None on error."""
        if self.copy_sources and self.copy_mode.get() != "Custom":
            # Find which source to copy from
            for name, panel in self.copy_sources:
                if self.copy_mode.get() == f"Same as {name}":
                    return panel.get_spectrum()
            return None
        try:
            freqs = [float(x.strip()) for x in self.freq_var.get().split(',')]
            amps = [float(x.strip()) for x in self.amp_var.get().split(',')]
            if len(freqs) != len(amps):
                return None
            return {"freq": freqs, "amp": amps}
        except ValueError:
            return None


class DissonanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dissonance — Musical Possibility Space")
        self.root.state('zoomed')
        self.root.configure(bg=BG)

        # ─── Dark ttk theme ───
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', background=BG, foreground=FG, fieldbackground=BG_PANEL,
                        bordercolor=BORDER, darkcolor=BG_SURFACE, lightcolor=BG_PANEL,
                        troughcolor=BG_SURFACE, selectbackground=ACCENT,
                        selectforeground=BG, font=('Segoe UI', 9))
        style.configure('TFrame', background=BG)
        style.configure('TLabelframe', background=BG, foreground=ACCENT,
                        bordercolor=BORDER, relief='flat')
        style.configure('TLabelframe.Label', background=BG, foreground=ACCENT,
                        font=('Segoe UI', 9, 'bold'))
        style.configure('TLabel', background=BG, foreground=FG)
        style.configure('TButton', background=BG_PANEL, foreground=FG,
                        bordercolor=BORDER, padding=(8, 4),
                        font=('Segoe UI', 9))
        style.map('TButton',
                  background=[('active', ACCENT), ('pressed', ACCENT)],
                  foreground=[('active', BG), ('pressed', BG)])
        style.configure('TEntry', fieldbackground=BG_PANEL, foreground=FG,
                        insertcolor=FG, bordercolor=BORDER, padding=3)
        style.configure('TCombobox', fieldbackground=BG_PANEL, foreground=FG,
                        selectbackground=ACCENT, selectforeground=BG, padding=3)
        style.map('TCombobox', fieldbackground=[('readonly', BG_PANEL)])
        style.configure('TRadiobutton', background=BG, foreground=FG,
                        indicatorcolor=BG_PANEL, font=('Segoe UI', 9))
        style.map('TRadiobutton',
                  indicatorcolor=[('selected', ACCENT)],
                  background=[('active', BG)])
        style.configure('TNotebook', background=BG_SURFACE, bordercolor=BORDER)
        style.configure('TNotebook.Tab', background=BG_PANEL, foreground=FG_DIM,
                        padding=(12, 6), font=('Segoe UI', 9))
        style.map('TNotebook.Tab',
                  background=[('selected', BG)],
                  foreground=[('selected', ACCENT)])
        style.configure('TPanedwindow', background=BG)
        style.configure('Vertical.TScrollbar', background=BG_PANEL,
                        troughcolor=BG_SURFACE, bordercolor=BORDER, arrowcolor=FG_DIM)
        style.configure('Status.TLabel', background=BG_SURFACE, foreground=ACCENT2,
                        font=('Segoe UI', 9, 'italic'), padding=(6, 4))

        self.ref_freq = 261.63
        self.max_interval = 2.3
        self.step_2d = 0.002
        self.step_3d = 0.005

        # Cached data
        self.ratios_2d = None
        self.diss_2d = None
        self.minima_x = None
        self.minima_y = None
        self.x_3d = None
        self.y_3d = None
        self.z_3d = None
        self.minima_3d_x = None  # B ratios of consonant triads
        self.minima_3d_y = None  # C ratios of consonant triads
        self.minima_3d_z = None  # dissonance values

        self._build_ui()
        self._compute_all()

    def _build_ui(self):
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # ─── Left panel ───
        left_frame = ttk.Frame(main_pane, width=380)
        main_pane.add(left_frame, weight=0)

        canvas_scroll = tk.Canvas(left_frame, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas_scroll.yview)
        scroll_frame = ttk.Frame(canvas_scroll)
        scroll_frame.bind("<Configure>",
            lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all")))
        canvas_scroll.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            canvas_scroll.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas_scroll.bind_all("<MouseWheel>", _on_mousewheel)

        ctrl = scroll_frame
        pad = {'padx': 6, 'pady': 3}

        # ─── Spectrum A (fixed voice) ───
        default_f = [1, 2, 3, 4, 5, 6]
        default_a = [1, 1/2, 1/3, 1/4, 1/5, 1/6]

        self.panel_a = SpectrumPanel(ctrl, "A — Voice 1 (fixed)", default_f, default_a)
        self.panel_a.frame.pack(**pad, fill='x')

        # ─── Spectrum B (2D varying, 3D x-axis) ───
        self.panel_b = SpectrumPanel(ctrl, "B — Voice 2 (2D interval / 3D x-axis)",
                                     default_f, default_a,
                                     copy_sources=[("A", self.panel_a)])
        self.panel_b.frame.pack(**pad, fill='x')

        # ─── Spectrum C (3D y-axis) ───
        self.panel_c = SpectrumPanel(ctrl, "C — Voice 3 (3D y-axis only)",
                                     default_f, default_a,
                                     copy_sources=[("A", self.panel_a), ("B", self.panel_b)])
        self.panel_c.frame.pack(**pad, fill='x')

        # Set play callbacks
        self.panel_a.set_play_callback(play_multi, lambda: self.ref_freq)
        self.panel_b.set_play_callback(play_multi, lambda: self.ref_freq)
        self.panel_c.set_play_callback(play_multi, lambda: self.ref_freq)

        # ─── General Settings ───
        settings_frame = ttk.LabelFrame(ctrl, text="Settings", padding=4)
        settings_frame.pack(**pad, fill='x')

        sg = settings_frame
        ttk.Label(sg, text="Fundamental (Hz):").grid(row=0, column=0, sticky='w', padx=2)
        self.ref_freq_var = tk.StringVar(value="261.63")
        ttk.Entry(sg, textvariable=self.ref_freq_var, width=8).grid(row=0, column=1, padx=2)

        ttk.Label(sg, text="Max ratio:").grid(row=0, column=2, sticky='w', padx=2)
        self.max_int_var = tk.StringVar(value="2.3")
        ttk.Entry(sg, textvariable=self.max_int_var, width=6).grid(row=0, column=3, padx=2)

        ttk.Label(sg, text="3D step:").grid(row=1, column=0, sticky='w', padx=2)
        self.step_var = tk.StringVar(value="0.005")
        ttk.Entry(sg, textvariable=self.step_var, width=8).grid(row=1, column=1, padx=2)

        # ─── Action Buttons ───
        btn_frame = ttk.Frame(ctrl)
        btn_frame.pack(padx=6, pady=(10, 4), fill='x')
        ttk.Button(btn_frame, text="\u27f3 Update Plots", command=self._on_update).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="\u25a0 Stop Audio", command=sd.stop).pack(side=tk.LEFT, padx=2)

        # ─── Manual Playback ───
        play_frame = ttk.LabelFrame(ctrl, text="Manual Playback", padding=4)
        play_frame.pack(**pad, fill='x')

        int_row = ttk.Frame(play_frame)
        int_row.pack(fill='x', pady=2)
        ttk.Label(int_row, text="Interval (A+B):").pack(side=tk.LEFT)
        self.interval_var = tk.StringVar(value="1.5")
        ttk.Entry(int_row, textvariable=self.interval_var, width=7).pack(side=tk.LEFT, padx=4)
        ttk.Button(int_row, text="\u25b6", command=self._play_interval, width=3).pack(side=tk.LEFT)

        tri_row = ttk.Frame(play_frame)
        tri_row.pack(fill='x', pady=2)
        ttk.Label(tri_row, text="Triad (A+B+C):").pack(side=tk.LEFT)
        self.triad_var = tk.StringVar(value="1.25, 1.5")
        ttk.Entry(tri_row, textvariable=self.triad_var, width=10).pack(side=tk.LEFT, padx=4)
        ttk.Button(tri_row, text="\u25b6", command=self._play_triad, width=3).pack(side=tk.LEFT)

        # ─── Status ───
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(ctrl, textvariable=self.status_var, style='Status.TLabel').pack(**pad, anchor='w', fill='x')

        # ─── Consonant Intervals (2D only) ───
        int_list_frame = ttk.LabelFrame(ctrl, text="Consonant Intervals (2D: A vs B)", padding=4)
        int_list_frame.pack(**pad, fill='x')

        sort_frame = ttk.Frame(int_list_frame)
        sort_frame.pack(fill='x', pady=2)
        ttk.Label(sort_frame, text="Sort:").pack(side=tk.LEFT)
        self.sort_var = tk.StringVar(value="consonance")
        ttk.Radiobutton(sort_frame, text="Most consonant", variable=self.sort_var,
                        value="consonance", command=self._draw_intervals_list).pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(sort_frame, text="By ratio", variable=self.sort_var,
                        value="ratio", command=self._draw_intervals_list).pack(side=tk.LEFT, padx=4)

        self.intervals_frame = ttk.Frame(int_list_frame)
        self.intervals_frame.pack(fill='x')

        # ─── Consonant Triads (3D) ───
        triad_list_frame = ttk.LabelFrame(ctrl, text="Consonant Triads (3D: A + B + C)", padding=4)
        triad_list_frame.pack(**pad, fill='x')

        self.triads_frame = ttk.Frame(triad_list_frame)
        self.triads_frame.pack(fill='x')

        # ─── Right panel: Plots ───
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=1)

        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: 2D curve
        tab_2d = ttk.Frame(self.notebook)
        self.notebook.add(tab_2d, text="  Dissonance Curve  ")
        self.fig_2d = Figure(figsize=(10, 5), dpi=100)
        self.ax_2d = self.fig_2d.add_subplot(111)
        self.canvas_2d = FigureCanvasTkAgg(self.fig_2d, master=tab_2d)
        self.canvas_2d.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar_2d = NavigationToolbar2Tk(self.canvas_2d, tab_2d)
        toolbar_2d.update()
        self.canvas_2d.mpl_connect('button_press_event', self._on_click_2d)

        # Tab 2: 3D surface
        tab_surf = ttk.Frame(self.notebook)
        self.notebook.add(tab_surf, text="  Dissonance Surface  ")
        self.fig_surf = Figure(figsize=(10, 8), dpi=100)
        self.ax_surf = self.fig_surf.add_subplot(111, projection='3d')
        self.ax_surf.view_init(elev=30, azim=69, roll=0)
        self.canvas_surf = FigureCanvasTkAgg(self.fig_surf, master=tab_surf)
        self.canvas_surf.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar_surf = NavigationToolbar2Tk(self.canvas_surf, tab_surf)
        toolbar_surf.update()
        self.canvas_surf.mpl_connect('button_press_event', self._on_click_surf)

        # Tab 3: Heatmap
        tab_hm = ttk.Frame(self.notebook)
        self.notebook.add(tab_hm, text="  Surface Heatmap  ")
        self.fig_hm = Figure(figsize=(10, 8), dpi=100)
        self.ax_hm = self.fig_hm.add_subplot(111)
        self.canvas_hm = FigureCanvasTkAgg(self.fig_hm, master=tab_hm)
        self.canvas_hm.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar_hm = NavigationToolbar2Tk(self.canvas_hm, tab_hm)
        toolbar_hm.update()
        self.canvas_hm.mpl_connect('button_press_event', self._on_click_3d)

    # ─── Input Parsing ───

    def _parse_inputs(self):
        try:
            self.ref_freq = float(self.ref_freq_var.get())
            self.max_interval = float(self.max_int_var.get())
            self.step_3d = float(self.step_var.get())
        except ValueError as e:
            self.status_var.set(f"Settings error: {e}")
            return False

        spec_a = self.panel_a.get_spectrum()
        spec_b = self.panel_b.get_spectrum()
        spec_c = self.panel_c.get_spectrum()

        if spec_a is None:
            self.status_var.set("Error in Spectrum A (check freq/amp counts)")
            return False
        if spec_b is None:
            self.status_var.set("Error in Spectrum B (check freq/amp counts)")
            return False
        if spec_c is None:
            self.status_var.set("Error in Spectrum C (check freq/amp counts)")
            return False

        self.spec_a = spec_a
        self.spec_b = spec_b
        self.spec_c = spec_c
        return True

    # ─── Computation ───

    def _on_update(self):
        if not self._parse_inputs():
            return
        self.status_var.set("Computing...")
        self.root.update()
        threading.Thread(target=self._compute_all, daemon=True).start()

    def _compute_all(self):
        if not hasattr(self, 'spec_a'):
            # First run, use defaults
            self.spec_a = self.panel_a.get_spectrum()
            self.spec_b = self.panel_b.get_spectrum()
            self.spec_c = self.panel_c.get_spectrum()

        # 2D: A vs B
        self.ratios_2d, self.diss_2d = compute_dissonance_2d(
            self.spec_a, self.spec_b, self.ref_freq, self.max_interval, self.step_2d)
        self.minima_x, self.minima_y = find_minima(self.ratios_2d, self.diss_2d)

        # 3D: A + B(r) + C(s)
        self.x_3d, self.y_3d, self.z_3d = compute_dissonance_3d(
            self.spec_a, self.spec_b, self.spec_c,
            self.ref_freq, self.max_interval, self.step_3d,
            progress_callback=self._progress)

        # Find consonant triads (3D local minima)
        self.minima_3d_x, self.minima_3d_y, self.minima_3d_z = find_minima_3d(
            self.x_3d, self.y_3d, self.z_3d)

        self.root.after(0, self._draw_all)

    def _progress(self, p):
        self.root.after(0, lambda: self.status_var.set(f"Computing 3D... {int(p*100)}%"))

    # ─── Drawing ───

    def _draw_all(self):
        self._draw_2d()
        self._draw_heatmap()
        self._draw_surface()
        self._draw_intervals_list()
        self._draw_triads_list()
        self.status_var.set("Ready — Click on plots to hear sounds")

    def _draw_2d(self):
        ax = self.ax_2d
        ax.clear()
        ax.plot(self.ratios_2d, self.diss_2d, color=ACCENT, linewidth=1.5, alpha=0.9)
        ax.fill_between(self.ratios_2d, self.diss_2d, alpha=0.08, color=ACCENT)
        if len(self.minima_x) > 0:
            ax.scatter(self.minima_x, self.minima_y, color=ACCENT2, zorder=5, s=50,
                       edgecolors='white', linewidths=0.6)
            for x, y in zip(self.minima_x, self.minima_y):
                name = ratio_to_interval_name(x)
                ax.annotate(f'{name}\n{x:.3f}', (x, y),
                            textcoords="offset points", xytext=(0, -18),
                            ha='center', fontsize=7, color=ACCENT2)
        ax.set_xlabel('Frequency Ratio (B / A)')
        ax.set_ylabel('Dissonance (normalized)')
        ax.set_title('Intervals: Voice A (fixed) + Voice B (varying) — Click to hear')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.9, self.max_interval)
        ax.set_ylim(bottom=0)
        self.fig_2d.tight_layout()
        self.canvas_2d.draw()

    def _draw_heatmap(self):
        ax = self.ax_hm
        ax.clear()
        extent = [self.x_3d[0], self.x_3d[-1], self.y_3d[0], self.y_3d[-1]]
        ax.imshow(self.z_3d.T, origin='lower', aspect='auto',
                  extent=extent, cmap='magma', interpolation='bilinear')
        # Overlay consonant triads
        if len(self.minima_3d_x) > 0:
            ax.scatter(self.minima_3d_x, self.minima_3d_y,
                       color=ACCENT2, edgecolors='white', s=55, zorder=5, linewidths=1.0)
            for bx, cy, dz in zip(self.minima_3d_x, self.minima_3d_y, self.minima_3d_z):
                ax.annotate(f'{bx:.2f},{cy:.2f}', (bx, cy),
                            textcoords="offset points", xytext=(5, 5),
                            fontsize=7, color='white', alpha=0.95,
                            fontweight='bold')
        ax.set_xlabel('Voice B ratio')
        ax.set_ylabel('Voice C ratio')
        ax.set_title('Triads: A(fixed) + B(x) + C(y) — Click to hear (dark = consonant)')
        self.fig_hm.tight_layout()
        self.canvas_hm.draw()

    def _draw_surface(self):
        ax = self.ax_surf
        elev, azim, roll = ax.elev, ax.azim, ax.roll
        ax.clear()
        ax.view_init(elev=elev, azim=azim, roll=roll)
        # Subsample to keep the 3D surface responsive
        step = max(1, len(self.x_3d) // 100)
        x_sub = self.x_3d[::step]
        y_sub = self.y_3d[::step]
        z_sub = self.z_3d[::step, ::step]
        X, Y = np.meshgrid(x_sub, y_sub)
        ax.plot_surface(X, Y, z_sub.T, cmap='magma', alpha=0.88,
                        rstride=1, cstride=1, linewidth=0, antialiased=False)
        # Overlay consonant triads
        if len(self.minima_3d_x) > 0:
            ax.scatter(self.minima_3d_x, self.minima_3d_y, self.minima_3d_z,
                       color=ACCENT2, s=120, zorder=10, depthshade=False,
                       edgecolors='white', linewidths=1.5, alpha=1.0)
        ax.set_xlabel('Voice B ratio')
        ax.set_ylabel('Voice C ratio')
        ax.set_zlabel('Dissonance')
        ax.set_title('Drag to rotate')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(BORDER)
        ax.yaxis.pane.set_edgecolor(BORDER)
        ax.zaxis.pane.set_edgecolor(BORDER)
        self.fig_surf.tight_layout()
        self.canvas_surf.draw()

    def _draw_intervals_list(self):
        for widget in self.intervals_frame.winfo_children():
            widget.destroy()

        if len(self.minima_x) == 0:
            ttk.Label(self.intervals_frame, text="(none found)").pack()
            return

        # Sort by consonance (lowest dissonance first) or by ratio
        pairs = list(zip(self.minima_x, self.minima_y))
        if self.sort_var.get() == "consonance":
            pairs.sort(key=lambda p: p[1])  # lowest dissonance = most consonant
        else:
            pairs.sort(key=lambda p: p[0])  # ascending ratio

        for x, y in pairs:
            name = ratio_to_interval_name(x)
            cents = ratio_to_cents(x)
            text = f"{x:.4f}  |  {cents:.0f}\u00a2  |  {name}  |  diss: {y:.3f}"
            btn = ttk.Button(self.intervals_frame, text=text,
                             command=lambda r=x: self._play_ratio(r))
            btn.pack(anchor='w', fill='x', pady=1)

    # ─── Audio Playback ───

    def _draw_triads_list(self):
        for widget in self.triads_frame.winfo_children():
            widget.destroy()

        if self.minima_3d_x is None or len(self.minima_3d_x) == 0:
            ttk.Label(self.triads_frame, text="(none found)").pack()
            return

        triads = list(zip(self.minima_3d_x, self.minima_3d_y, self.minima_3d_z))
        triads.sort(key=lambda t: t[2])  # most consonant first

        shown = triads

        for bx, cy, dz in shown:
            b_name = ratio_to_interval_name(bx)
            c_name = ratio_to_interval_name(cy)
            text = f"B={bx:.3f} ({b_name})  C={cy:.3f} ({c_name})  diss: {dz:.3f}"
            btn = ttk.Button(self.triads_frame, text=text,
                             command=lambda rb=bx, rc=cy: self._play_triad_ratio(rb, rc))
            btn.pack(anchor='w', fill='x', pady=1)

    def _play_triad_ratio(self, ratio_b, ratio_c):
        spec_a = self.panel_a.get_spectrum()
        spec_b = self.panel_b.get_spectrum()
        spec_c = self.panel_c.get_spectrum()
        b_name = ratio_to_interval_name(ratio_b)
        c_name = ratio_to_interval_name(ratio_c)
        self.status_var.set(f"Playing triad: A@1, B@{ratio_b:.3f} ({b_name}), C@{ratio_c:.3f} ({c_name})")
        play_multi([(spec_a, 1.0), (spec_b, ratio_b), (spec_c, ratio_c)], self.ref_freq)

    def _on_click_2d(self, event):
        if event.inaxes == self.ax_2d and event.xdata is not None:
            ratio = event.xdata
            name = ratio_to_interval_name(ratio)
            # Look up dissonance from cached curve
            idx = np.argmin(np.abs(self.ratios_2d - ratio))
            dval = self.diss_2d[idx]
            self.status_var.set(f"Playing: A@1 + B@{ratio:.4f} ({name}) — diss: {dval:.3f}")
            spec_a = self.panel_a.get_spectrum()
            spec_b = self.panel_b.get_spectrum()
            play_multi([(spec_a, 1.0), (spec_b, ratio)], self.ref_freq)

    def _on_click_3d(self, event):
        if event.inaxes == self.ax_hm and event.xdata is not None:
            r_b = event.xdata
            r_c = event.ydata
            bi = np.argmin(np.abs(self.x_3d - r_b))
            ci = np.argmin(np.abs(self.y_3d - r_c))
            dval = self.z_3d[bi, ci]
            self.status_var.set(f"Playing triad: A@1, B@{r_b:.3f}, C@{r_c:.3f} — diss: {dval:.3f}")
            spec_a = self.panel_a.get_spectrum()
            spec_b = self.panel_b.get_spectrum()
            spec_c = self.panel_c.get_spectrum()
            play_multi([(spec_a, 1.0), (spec_b, r_b), (spec_c, r_c)], self.ref_freq)

    def _on_click_surf(self, event):
        if event.inaxes == self.ax_surf and event.xdata is not None:
            r_b = event.xdata
            r_c = event.ydata
            bi = np.argmin(np.abs(self.x_3d - r_b))
            ci = np.argmin(np.abs(self.y_3d - r_c))
            dval = self.z_3d[bi, ci]
            self.status_var.set(f"Playing triad: A@1, B@{r_b:.3f}, C@{r_c:.3f} — diss: {dval:.3f}")
            spec_a = self.panel_a.get_spectrum()
            spec_b = self.panel_b.get_spectrum()
            spec_c = self.panel_c.get_spectrum()
            play_multi([(spec_a, 1.0), (spec_b, r_b), (spec_c, r_c)], self.ref_freq)

    def _play_interval(self):
        if not self._parse_inputs():
            return
        try:
            ratio = float(self.interval_var.get())
            name = ratio_to_interval_name(ratio)
            # Look up dissonance from cached curve
            dval = 0
            if self.ratios_2d is not None:
                idx = np.argmin(np.abs(self.ratios_2d - ratio))
                dval = self.diss_2d[idx]
            self.status_var.set(f"Playing: A@1 + B@{ratio:.4f} ({name}) — diss: {dval:.3f}")
            play_multi([(self.spec_a, 1.0), (self.spec_b, ratio)], self.ref_freq)
        except ValueError:
            self.status_var.set("Invalid ratio")

    def _play_triad(self):
        if not self._parse_inputs():
            return
        try:
            parts = [float(x.strip()) for x in self.triad_var.get().split(',')]
            if len(parts) == 2:
                # Compute dissonance at this point
                r_b, r_c = parts
                if self.z_3d is not None and self.x_3d is not None:
                    bi = np.argmin(np.abs(self.x_3d - r_b))
                    ci = np.argmin(np.abs(self.y_3d - r_c))
                    dval = self.z_3d[bi, ci]
                    self.status_var.set(f"Playing: A@1, B@{r_b:.3f}, C@{r_c:.3f} — diss: {dval:.3f}")
                else:
                    self.status_var.set(f"Playing: A@1, B@{r_b:.3f}, C@{r_c:.3f}")
                play_multi([
                    (self.spec_a, 1.0),
                    (self.spec_b, parts[0]),
                    (self.spec_c, parts[1])
                ], self.ref_freq)
            else:
                self.status_var.set("Enter exactly two ratios (B, C)")
        except ValueError:
            self.status_var.set("Invalid ratios")

    def _play_ratio(self, ratio):
        spec_a = self.panel_a.get_spectrum()
        spec_b = self.panel_b.get_spectrum()
        name = ratio_to_interval_name(ratio)
        self.status_var.set(f"Playing: A@1 + B@{ratio:.4f} ({name})")
        play_multi([(spec_a, 1.0), (spec_b, ratio)], self.ref_freq)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    root = tk.Tk()
    app = DissonanceApp(root)
    root.mainloop()
