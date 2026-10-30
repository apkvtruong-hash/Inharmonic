# Dissonance Explorer

An interactive tool for deriving musical scales and chords from sound spectra, using the psychoacoustic model of sensory dissonance developed by Plomp, Levelt, and Sethares.

Given any sound spectrum (set of partials), the tool computes which intervals and triads will sound most consonant—and the results depend entirely on the timbre. Harmonic timbres produce the familiar Western intervals; inharmonic timbres (bells, metallophones) produce entirely different scales.

![Python](https://img.shields.io/badge/Python-PySide6%20%7C%20pyqtgraph-blue)
![Web](https://img.shields.io/badge/Web-Vue.js%20%7C%20Plotly-green)

---

## The Research

### Why Do Some Note Combinations Sound Good?

In 1965, Dutch researchers **Reinier Plomp** and **Willem Levelt** conducted experiments measuring how pleasant or unpleasant pairs of pure tones (sine waves) sounded when played together. They found that dissonance (roughness) between two pure tones depends on their frequency separation relative to the **critical bandwidth** of the auditory system—a frequency-dependent range determined by the physical structure of the inner ear (the basilar membrane).

> R. Plomp and W.J.M. Levelt, "Tonal Consonance and Critical Bandwidth," *Journal of the Acoustical Society of America*, 38, 548–560 (1965).

In 1993, **William Sethares** extended this work by showing that the consonance of complex tones (real musical sounds with multiple partials) can be predicted by summing the dissonance contributions of every pair of partials. Crucially, he demonstrated that **musical scales are not universal—they emerge from timbre**. The Western 12-tone equal temperament scale is well-suited to harmonic timbres (strings, winds), but instruments with inharmonic spectra (bells, gamelans) are naturally consonant with entirely different scales.

> W.A. Sethares, "Local Consonance and the Relationship Between Timbre and Scale," *Journal of the Acoustical Society of America*, 94(3), 1218–1228 (1993).
>
> W.A. Sethares, *Tuning, Timbre, Spectrum, Scale*, Springer (1998/2005).

---

## The Math

### Step 1: Dissonance Between Two Pure Tones

The Plomp-Levelt model gives the sensory dissonance $d$ between two sine tones at frequencies $f_1$ and $f_2$ with loudnesses $\ell_1$ and $\ell_2$:

$$
d(f_1, f_2, \ell_1, \ell_2) = \ell_{12} \left( e^{-b_1 p} - e^{-b_2 p} \right)
$$

where:

- $\ell_{12} = \min(\ell_1, \ell_2)$ — dissonance is limited by the quieter tone
- $p = s \cdot (f_{\max} - f_{\min})$ — frequency difference scaled to the critical bandwidth
- $s = \dfrac{x}{s_1 \cdot f_{\min} + s_2}$ — critical bandwidth scaling factor

The constants, fit to Plomp and Levelt's experimental data by Sethares:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| $b_1$ | 3.51 | Controls the rise of dissonance |
| $b_2$ | 5.75 | Controls the fall of dissonance |
| $x$ | 0.24 | Peak dissonance at ~24% of critical bandwidth |
| $s_1$ | 0.0207 | Critical bandwidth slope |
| $s_2$ | 18.96 | Critical bandwidth offset |

The function rises quickly from zero (unison) to a peak near 25% of the critical bandwidth, then falls off gradually. Two tones very close together "beat" and sound rough; tones far apart don't interact.

### Step 2: Loudness from Amplitude

Amplitudes are converted to a perceptual loudness scale:

$$
\ell = \frac{2^{(\text{dB}/10)}}{16}, \qquad \text{dB} = 20 \log_{10}(a)
$$

This approximates Stevens' power law for loudness perception.

### Step 3: Dissonance of Complex Tones (Intervals)

A real musical sound is a spectrum of partials: a fundamental frequency $f_0$ multiplied by frequency ratios $r_1, r_2, \ldots, r_n$ with amplitudes $a_1, a_2, \ldots, a_n$.

To find which **intervals** sound consonant, we fix one voice (voice A at pitch ratio 1) and sweep a second voice (voice B at pitch ratio $c$) from unison to the octave. The total dissonance at ratio $c$ is the sum over all pairs of partials between and within the two voices:

$$
D(c) = \frac{1}{2} \sum_{i,j} \Big[ d(f_i^A,\, f_j^A) + d(c \cdot f_i^B,\, c \cdot f_j^B) + 2\, d(f_i^A,\, c \cdot f_j^B) \Big]
$$

where $f_i^A = f_0 \cdot r_i$ are the partial frequencies of voice A, and similarly for B.

The first term is the self-dissonance of voice A (constant), the second is the self-dissonance of voice B (varies with $c$ because the Plomp-Levelt model is frequency-dependent), and the third is the cross-dissonance between voices.

**Local minima** in the curve $D(c)$ correspond to consonant intervals—pitch ratios where partials tend to align rather than clash.

### Step 4: Dissonance of Triads (3D Surface)

For three voices at pitch ratios $1$, $r$, and $s$, the total dissonance sums all pairwise interactions among three sets of partials:

$$
D(r, s) = \sum_{i,j} \Big[ d(f_i, f_j) + d(r f_i, r f_j) + d(s f_i, s f_j) + d(f_i, r f_j) + d(f_i, s f_j) + d(r f_i, s f_j) \Big]
$$

This produces a 2D surface over the $(r, s)$ plane. **Valleys** in this surface are consonant triads—three-note chords with minimal roughness.

### Step 5: Finding Consonant Intervals and Triads

- **2D (intervals)**: Local minima are detected using peak-finding on the inverted dissonance curve, filtered by a prominence threshold to ignore shallow dips.
- **3D (triads)**: Local minima on the surface are found by checking each grid point against its neighbors, then filtered to keep only triads with distinct pitches within one octave.

---

## What the Results Show

For a **harmonic spectrum** (partials at 1×, 2×, 3×, 4×, 5×, 6× the fundamental), the dissonance curve minima land on the familiar just intonation intervals:

| Ratio | Interval | Cents |
|-------|----------|-------|
| 1:1 | Unison | 0 |
| 6:5 | Minor third | 316 |
| 5:4 | Major third | 386 |
| 4:3 | Perfect fourth | 498 |
| 3:2 | Perfect fifth | 702 |
| 5:3 | Major sixth | 884 |
| 2:1 | Octave | 1200 |

For **inharmonic spectra**, entirely different intervals emerge. A gamelan bonang spectrum produces intervals that align with the slendro and pelog tuning systems. A xylophone (free vibrating beam with partials at 1, 2.76, 5.40, 8.93, ...) produces its own unique set of consonances.

This is the core insight of Sethares' work: **scales and timbre are fundamentally linked**.

---

## Spectrum Presets

| Preset | Partial ratios | Character |
|--------|---------------|-----------|
| Harmonic | 1, 2, 3, 4, 5, 6 | Strings, winds — produces Western intervals |
| Odd harmonics | 1, 3, 5, 7, 9, 11 | Square wave — hollow, clarinet-like |
| Free beam | 1, 2.76, 5.40, 8.93 | Xylophone, marimba bars |
| Bonang | 1, 1.52, 3.46, 3.92 | Gamelan bronze pot — produces non-Western scales |

---

## Running the Project

### Python (Desktop)

```bash
pip install numpy sounddevice PySide6 pyqtgraph PyOpenGL

# Live, animated studio GUI (recommended; first launch compiles its Numba kernels once)
pip install numba qtawesome
python dissonance_studio.py

# Qt Quick (QML) version: Fluent controls, QtGraphs plots (needs numba; no qtawesome)
python dissonance_quick.py

# Modern GPU-accelerated GUI
python dissonance_modern.py

# Matplotlib version (lighter dependencies)
pip install matplotlib scipy
python dissonance.py

# Tkinter GUI
python dissonance_gui.py
```

### Web (Browser)

Open `index.html` in a modern browser. No build step required—uses Vue.js, Plotly.js, and Web Audio API directly.

---

## Architecture

The core dissonance algorithms are implemented identically across Python and JavaScript:

```
                    ┌─────────────────┐
                    │  Dissonance     │
                    │  Model          │
                    │  (Plomp-Levelt) │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
     ┌────────▼────────┐          ┌─────────▼────────┐
     │  Python          │          │  JavaScript       │
     │  numpy vectorized│          │  helper.js loops   │
     │  dissonance.py   │          │  dissonance-worker │
     └────────┬────────┘          └─────────┬────────┘
              │                             │
     ┌────────▼────────┐          ┌─────────▼────────┐
     │  GUI Frontends   │          │  Web Frontend     │
     │  PySide6/Qt      │          │  Vue.js + Plotly  │
     │  matplotlib      │          │  Web Audio API    │
     │  Tkinter         │          │  p5.js animation  │
     └─────────────────┘          └──────────────────┘
```

---

## References

1. R. Plomp and W.J.M. Levelt, "Tonal Consonance and Critical Bandwidth," *J. Acoust. Soc. Am.*, 38, 548–560 (1965).
2. W.A. Sethares, "Local Consonance and the Relationship Between Timbre and Scale," *J. Acoust. Soc. Am.*, 94(3), 1218–1228 (1993).
3. W.A. Sethares, *Tuning, Timbre, Spectrum, Scale*, 2nd ed., Springer (2005).
4. H. Helmholtz, *On the Sensations of Tone*, Dover (1877/1954).
