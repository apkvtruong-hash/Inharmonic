"""
Dissonance: Deriving musical scales and chords from sound spectra.

Uses the Plomp-Levelt model of sensory dissonance to find consonant
intervals and triads for any given sound spectrum.

Features:
- Customizable spectrum (frequency ratios and amplitudes)
- 2D dissonance curve (intervals)
- 3D dissonance surface (triads)
- Audio synthesis (click on graphs to hear intervals/triads)

Requirements:
    pip install numpy matplotlib sounddevice
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sounddevice as sd

# ─── Spectrum Configuration ───────────────────────────────────────────────────

SPECTRUM = {
    "freq": [1, 2, 3, 4, 5, 6],        # partial frequency ratios
    "amp": [1, 1/2, 1/3, 1/4, 1/5, 1/6] # partial amplitudes
}

REF_FREQ = 261.63  # fundamental frequency in Hz (middle C)
MAX_INTERVAL = 2.05  # max pitch ratio to explore (slightly above octave)

STEP_SIZE_2D = 0.002
STEP_SIZE_3D = 0.0025  # matches JS original resolution

SAMPLE_RATE = 44100
DURATION = 1.5  # seconds for audio playback


# ─── Dissonance Model (Plomp-Levelt) ─────────────────────────────────────────

def amp_to_loudness(amp):
    """Convert amplitude to perceptual loudness."""
    dB = 20 * np.log10(np.maximum(amp, 1e-10))
    loudness = 2 ** (dB / 10) / 16
    return loudness


def dissonance_pair(f1, f2, l1, l2):
    """Compute dissonance between two pure tones (Sethares model)."""
    x = 0.24
    s1 = 0.0207
    s2 = 18.96
    b1 = 3.51
    b2 = 5.75

    fmin = min(f1, f2)
    fmax = max(f1, f2)
    s = x / (s1 * fmin + s2)
    p = s * (fmax - fmin)

    l12 = min(l1, l2)
    return l12 * (np.exp(-b1 * p) - np.exp(-b2 * p))


def dissonance_pair_vectorized(f1, f2, l1, l2):
    """Vectorized dissonance between two pure tones."""
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


# ─── 2D Dissonance Curve ─────────────────────────────────────────────────────

def compute_dissonance_2d(spectrum, ref_freq, max_interval, step_size):
    """Compute dissonance as a function of interval ratio (vectorized)."""
    freq_ratios = np.array(spectrum["freq"])
    amps = np.array(spectrum["amp"])
    n = len(freq_ratios)
    loudness = amp_to_loudness(amps)

    ratios = np.arange(1.0, max_interval, step_size)
    num_ratios = len(ratios)

    base_freqs = ref_freq * freq_ratios  # shape (n,)
    base_loud = loudness  # shape (n,)

    # Build all partial pairs (n x n)
    f1_grid, f2_grid = np.meshgrid(base_freqs, base_freqs, indexing='ij')  # (n, n)
    l1_grid, l2_grid = np.meshgrid(base_loud, base_loud, indexing='ij')  # (n, n)

    # Flatten for vectorized dissonance computation
    f1_flat = f1_grid.ravel()  # (n*n,)
    f2_flat = f2_grid.ravel()
    l1_flat = l1_grid.ravel()
    l2_flat = l2_grid.ravel()

    # Term 1: dissonance(f1, f2, l1, l2) — constant across ratios
    term1 = dissonance_pair_vectorized(f1_flat, f2_flat, l1_flat, l2_flat).sum() * 0.5

    # Terms that vary with ratio c:
    # For each c: 0.5 * sum(d(c*f1, c*f2, l1, l2)) + sum(d(f1, c*f2, l1, l2))
    # d(c*f1, c*f2) has same result as d(f1, f2) scaled — but the Plomp-Levelt
    # model is frequency-dependent, so we compute per ratio.
    dissonance_values = np.zeros(num_ratios)

    # Broadcast: ratios (R,) x pairs (P,) -> (R, P)
    # Process in chunks to manage memory
    chunk_size = 200
    for start in range(0, num_ratios, chunk_size):
        end = min(start + chunk_size, num_ratios)
        c_chunk = ratios[start:end, np.newaxis]  # (chunk, 1)

        # Term 2: d(c*f1, c*f2)
        cf1 = c_chunk * f1_flat[np.newaxis, :]  # (chunk, n*n)
        cf2 = c_chunk * f2_flat[np.newaxis, :]
        t2 = dissonance_pair_vectorized(cf1, cf2, l1_flat, l2_flat).sum(axis=1) * 0.5

        # Term 3: d(f1, c*f2)
        t3 = dissonance_pair_vectorized(f1_flat[np.newaxis, :], cf2, l1_flat, l2_flat).sum(axis=1)

        dissonance_values[start:end] = (term1 + t2 + t3) / 2.0

    # Normalize
    max_d = dissonance_values.max()
    if max_d > 0:
        dissonance_values /= max_d

    return ratios, dissonance_values


def find_minima_2d(ratios, dissonance_values, prominence=0.02):
    """Find local minima (consonant intervals) in the dissonance curve."""
    from scipy.signal import find_peaks
    # Invert to find minima as peaks
    inverted = -dissonance_values
    peaks, properties = find_peaks(inverted, prominence=prominence)
    return ratios[peaks], dissonance_values[peaks]


def find_minima_2d_simple(ratios, dissonance_values, window=5):
    """Find local minima without scipy dependency."""
    minima_x = []
    minima_y = []
    for i in range(window, len(dissonance_values) - window):
        if dissonance_values[i] == min(dissonance_values[i - window:i + window + 1]):
            if dissonance_values[i] < 0.8:  # filter out shallow minima
                minima_x.append(ratios[i])
                minima_y.append(dissonance_values[i])
    return np.array(minima_x), np.array(minima_y)


# ─── 3D Dissonance Surface ───────────────────────────────────────────────────

def compute_dissonance_3d(spectrum, ref_freq, max_interval, step_size):
    """Compute dissonance surface for triads (vectorized with NumPy)."""
    freq_ratios = np.array(spectrum["freq"])
    amps = np.array(spectrum["amp"])
    n = len(freq_ratios)
    loudness = amp_to_loudness(amps)

    ratios = np.arange(1.0, max_interval, step_size)
    num_steps = len(ratios)

    base_freqs = ref_freq * freq_ratios  # (n,)
    base_loud = loudness  # (n,)

    # All partial pairs
    f1_grid, f2_grid = np.meshgrid(base_freqs, base_freqs, indexing='ij')
    l1_grid, l2_grid = np.meshgrid(base_loud, base_loud, indexing='ij')
    f1 = f1_grid.ravel()  # (P,) where P = n*n
    f2 = f2_grid.ravel()
    l1 = l1_grid.ravel()
    l2 = l2_grid.ravel()
    P = len(f1)

    print(f"Computing 3D dissonance surface ({num_steps}x{num_steps} = {num_steps**2} points)...")

    # Term 1: d(f1, f2) — constant, independent of r and s
    term1 = dissonance_pair_vectorized(f1, f2, l1, l2).sum()

    # Precompute row terms that depend only on r (not s):
    # term2(r) = sum d(r*f1, r*f2)
    # term3(r) = sum d(f1, r*f2)
    # These are reused across all columns.
    term2_all = np.zeros(num_steps)
    term3_all = np.zeros(num_steps)

    for ri, r in enumerate(ratios):
        rf1 = r * f1
        rf2 = r * f2
        term2_all[ri] = dissonance_pair_vectorized(rf1, rf2, l1, l2).sum()
        term3_all[ri] = dissonance_pair_vectorized(f1, rf2, l1, l2).sum()

    # term4(s) = sum d(s*f1, s*f2) — same structure as term2 but for s
    # term5(s) = sum d(f1, s*f2) — same structure as term3 but for s
    # Since ratios are the same for r and s, term4 = term2_all, term5 = term3_all
    term4_all = term2_all
    term5_all = term3_all

    # Term 6: d(r*f1, s*f2) — depends on both r and s, compute per row
    z_data = np.zeros((num_steps, num_steps))

    for ri in range(num_steps):
        if ri % 20 == 0:
            print(f"  Progress: {ri}/{num_steps} ({100*ri//num_steps}%)", end='\r')

        rf1 = ratios[ri] * f1  # (P,)

        # Vectorize across all s values for this r
        # sf2 for all s: (num_steps, P)
        sf2 = ratios[:, np.newaxis] * f2[np.newaxis, :]  # (num_steps, P)

        # d(r*f1, s*f2) for all s at once
        # rf1 broadcast: (1, P) vs sf2: (num_steps, P)
        t6 = dissonance_pair_vectorized(
            rf1[np.newaxis, :], sf2, l1[np.newaxis, :], l2[np.newaxis, :]
        ).sum(axis=1)  # (num_steps,)

        # Combine all 6 terms
        z_data[ri, :] = (term1 + term2_all[ri] + term3_all[ri] +
                         term4_all + term5_all + t6) / 2.0

    print(f"  Progress: {num_steps}/{num_steps} (100%) - Done!")

    # Normalize
    max_z = z_data.max()
    if max_z > 0:
        z_data /= max_z

    return ratios, ratios, z_data


# ─── Audio Synthesis ──────────────────────────────────────────────────────────

def synthesize(spectrum, ref_freq, tuning, duration=DURATION, sample_rate=SAMPLE_RATE):
    """Synthesize a sound using additive synthesis for given tuning ratios."""
    freq_ratios = np.array(spectrum["freq"])
    amps = np.array(spectrum["amp"])

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Envelope: fade in/out to avoid clicks
    envelope = np.ones_like(t)
    fade_samples = int(0.05 * sample_rate)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

    signal = np.zeros_like(t)

    for multiplier in tuning:
        base = ref_freq * multiplier
        gain = max(1.0 / len(freq_ratios), 0.5)
        for freq_ratio, amp in zip(freq_ratios, amps):
            freq = base * freq_ratio
            signal += gain * amp * np.sin(2 * np.pi * freq * t)

    # Normalize to prevent clipping
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.7

    signal *= envelope
    return signal.astype(np.float32)


def play_sound(spectrum, ref_freq, tuning):
    """Play a sound with given tuning ratios."""
    signal = synthesize(spectrum, ref_freq, tuning)
    sd.stop()
    sd.play(signal, SAMPLE_RATE)


# ─── Interval Labels ─────────────────────────────────────────────────────────

INTERVAL_LABELS = [
    'unison', 'minor 2nd', 'major 2nd', 'minor 3rd', 'major 3rd',
    'perfect 4th', 'tritone', 'perfect 5th', 'minor 6th', 'major 6th',
    'minor 7th', 'major 7th', 'octave'
]


def ratio_to_midi(ratio):
    """Convert frequency ratio to MIDI note offset from C4 (60)."""
    return 60 + 12 * np.log2(ratio)


def ratio_to_interval_name(ratio):
    """Get the closest interval name for a frequency ratio."""
    semitones = 12 * np.log2(ratio)
    idx = int(round(semitones))
    if 0 <= idx < len(INTERVAL_LABELS):
        return INTERVAL_LABELS[idx]
    return f"{semitones:.1f} semitones"


# ─── Interactive Plotting ─────────────────────────────────────────────────────

def plot_2d_interactive(spectrum, ref_freq):
    """Plot the 2D dissonance curve with click-to-play."""
    print("Computing 2D dissonance curve...")
    ratios, dissonance_vals = compute_dissonance_2d(
        spectrum, ref_freq, MAX_INTERVAL, STEP_SIZE_2D
    )

    # Find minima
    minima_x, minima_y = find_minima_2d_simple(ratios, dissonance_vals)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ratios, dissonance_vals, 'b-', linewidth=1.5, label='Dissonance')
    ax.scatter(minima_x, minima_y, color='orange', zorder=5, s=60,
               label='Consonant intervals')

    # Annotate minima
    for x, y in zip(minima_x, minima_y):
        name = ratio_to_interval_name(x)
        ax.annotate(f'{name}\n({x:.3f})', (x, y),
                    textcoords="offset points", xytext=(0, -20),
                    ha='center', fontsize=7, color='darkorange')

    ax.set_xlabel('Frequency Ratio')
    ax.set_ylabel('Dissonance (normalized)')
    ax.set_title('Dissonance Curve — Click to hear an interval')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Click handler
    def on_click(event):
        if event.inaxes == ax and event.xdata is not None:
            ratio = event.xdata
            print(f"  Playing interval: ratio = {ratio:.4f} "
                  f"({ratio_to_interval_name(ratio)})")
            play_sound(spectrum, ref_freq, [1.0, ratio])

    fig.canvas.mpl_connect('button_press_event', on_click)

    print("Click on the graph to hear intervals. Close the window to continue.")
    plt.tight_layout()
    plt.show()

    # Print consonant intervals table
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║              Consonant Intervals Found                      ║")
    print("╠════════════╦═══════════════════╦═══════════════════════════╣")
    print("║   Ratio    ║   MIDI Note       ║   Interval Name           ║")
    print("╠════════════╬═══════════════════╬═══════════════════════════╣")
    for x in minima_x:
        midi = ratio_to_midi(x)
        name = ratio_to_interval_name(x)
        print(f"║  {x:8.4f}  ║  {midi:13.2f}    ║   {name:<24s}║")
    print("╚════════════╩═══════════════════╩═══════════════════════════╝")

    return ratios, dissonance_vals, minima_x


def plot_3d_interactive(spectrum, ref_freq):
    """Plot the 3D dissonance surface with click-to-play."""
    x_data, y_data, z_data = compute_dissonance_3d(
        spectrum, ref_freq, MAX_INTERVAL, STEP_SIZE_3D
    )

    X, Y = np.meshgrid(x_data, y_data)

    # --- 3D Surface Plot ---
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    surf = ax1.plot_surface(X, Y, z_data.T, cmap='coolwarm',
                            alpha=0.85, rstride=2, cstride=2)
    ax1.set_xlabel('Ratio 1')
    ax1.set_ylabel('Ratio 2')
    ax1.set_zlabel('Dissonance')
    ax1.set_title('Dissonance Surface (Triads) — Drag to rotate')
    fig1.colorbar(surf, shrink=0.6)

    plt.tight_layout()
    plt.show()

    # --- 2D Heatmap ---
    fig2, ax2 = plt.subplots(figsize=(9, 8))
    im = ax2.imshow(z_data.T, origin='lower', aspect='auto',
                    extent=[x_data[0], x_data[-1], y_data[0], y_data[-1]],
                    cmap='viridis')
    ax2.set_xlabel('Ratio 1')
    ax2.set_ylabel('Ratio 2')
    ax2.set_title('Dissonance Heatmap (Triads) — Click to hear a triad')
    fig2.colorbar(im, label='Dissonance')

    # Click handler for heatmap
    def on_click(event):
        if event.inaxes == ax2 and event.xdata is not None:
            r1 = event.xdata
            r2 = event.ydata
            print(f"  Playing triad: ratios = 1.0, {r1:.4f}, {r2:.4f}")
            play_sound(spectrum, ref_freq, [1.0, r1, r2])

    fig2.canvas.mpl_connect('button_press_event', on_click)

    print("Click on the heatmap to hear triads. Close the window to continue.")
    plt.tight_layout()
    plt.show()


# ─── Main Menu ────────────────────────────────────────────────────────────────

def print_spectrum(spectrum, ref_freq):
    """Display current spectrum settings."""
    print(f"\n  Fundamental frequency: {ref_freq} Hz")
    print(f"  Partial ratios: {spectrum['freq']}")
    print(f"  Partial amplitudes: {[round(a, 4) for a in spectrum['amp']]}")
    print(f"  Number of partials: {len(spectrum['freq'])}")


def customize_spectrum(spectrum, ref_freq):
    """Interactive spectrum customization."""
    print("\n── Customize Spectrum ──")
    print_spectrum(spectrum, ref_freq)
    print("\nOptions:")
    print("  1. Set partial frequency ratios (e.g. 1,2,3,4,5,6)")
    print("  2. Set partial amplitudes (e.g. 1,0.5,0.33,0.25,0.2,0.17)")
    print("  3. Set fundamental frequency (Hz)")
    print("  4. Preset: Harmonic (default)")
    print("  5. Preset: Odd harmonics only")
    print("  6. Preset: Stretched (inharmonic)")
    print("  7. Preset: Compressed")
    print("  8. Preset: Gamelan-like (metallic)")
    print("  0. Back")

    choice = input("\n  Choice: ").strip()

    if choice == '1':
        raw = input("  Enter frequency ratios (comma-separated): ").strip()
        try:
            freqs = [float(x) for x in raw.split(',')]
            if len(freqs) != len(spectrum['amp']):
                print(f"  Adjusting amplitudes to match {len(freqs)} partials (1/n falloff)")
                spectrum['amp'] = [1.0 / (i + 1) for i in range(len(freqs))]
            spectrum['freq'] = freqs
        except ValueError:
            print("  Invalid input.")

    elif choice == '2':
        raw = input("  Enter amplitudes (comma-separated): ").strip()
        try:
            amps = [float(x) for x in raw.split(',')]
            if len(amps) != len(spectrum['freq']):
                print(f"  Error: need {len(spectrum['freq'])} values to match partials.")
            else:
                spectrum['amp'] = amps
        except ValueError:
            print("  Invalid input.")

    elif choice == '3':
        raw = input("  Enter fundamental frequency in Hz: ").strip()
        try:
            ref_freq = float(raw)
        except ValueError:
            print("  Invalid input.")

    elif choice == '4':
        spectrum['freq'] = [1, 2, 3, 4, 5, 6]
        spectrum['amp'] = [1, 1/2, 1/3, 1/4, 1/5, 1/6]
        print("  Set to harmonic spectrum.")

    elif choice == '5':
        spectrum['freq'] = [1, 3, 5, 7, 9, 11]
        spectrum['amp'] = [1, 1/3, 1/5, 1/7, 1/9, 1/11]
        print("  Set to odd harmonics (clarinet-like).")

    elif choice == '6':
        # Stretched partials (like a stiff piano string)
        spectrum['freq'] = [1 * (1 + 0.0013 * (i**2)) for i in range(1, 9)]
        spectrum['freq'] = [f / spectrum['freq'][0] for f in spectrum['freq']]
        spectrum['amp'] = [1.0 / i for i in range(1, 9)]
        print("  Set to stretched (inharmonic, piano-like).")

    elif choice == '7':
        # Compressed partials
        spectrum['freq'] = [i**0.9 for i in range(1, 9)]
        spectrum['amp'] = [1.0 / i for i in range(1, 9)]
        print("  Set to compressed partials.")

    elif choice == '8':
        # Gamelan-like metallic spectrum
        spectrum['freq'] = [1, 2.76, 5.40, 8.93, 13.34, 18.64]
        spectrum['amp'] = [1, 0.7, 0.5, 0.3, 0.2, 0.1]
        print("  Set to gamelan-like (metallic) spectrum.")

    return spectrum, ref_freq


def main():
    """Main interactive loop."""
    spectrum = dict(SPECTRUM)  # copy defaults
    ref_freq = REF_FREQ

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                     DISSONANCE                              ║")
    print("║        A Journey Through Musical Possibility Space          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print("\nDeriving musical scales and chords from sound spectra.")
    print("Based on the Plomp-Levelt / Sethares model of sensory dissonance.\n")

    while True:
        print("\n── Main Menu ──")
        print_spectrum(spectrum, ref_freq)
        print("\n  1. Play current sound")
        print("  2. Customize spectrum")
        print("  3. Plot 2D dissonance curve (intervals)")
        print("  4. Plot 3D dissonance surface (triads)")
        print("  5. Play a specific interval (enter ratio)")
        print("  6. Play a specific triad (enter two ratios)")
        print("  0. Quit")

        choice = input("\n  Choice: ").strip()

        if choice == '0':
            print("Goodbye!")
            sd.stop()
            break

        elif choice == '1':
            print("  Playing sound...")
            play_sound(spectrum, ref_freq, [1.0])

        elif choice == '2':
            spectrum, ref_freq = customize_spectrum(spectrum, ref_freq)

        elif choice == '3':
            plot_2d_interactive(spectrum, ref_freq)

        elif choice == '4':
            plot_3d_interactive(spectrum, ref_freq)

        elif choice == '5':
            raw = input("  Enter interval ratio (e.g. 1.5): ").strip()
            try:
                ratio = float(raw)
                print(f"  Playing interval: 1.0, {ratio} "
                      f"({ratio_to_interval_name(ratio)})")
                play_sound(spectrum, ref_freq, [1.0, ratio])
            except ValueError:
                print("  Invalid input.")

        elif choice == '6':
            raw = input("  Enter two ratios (comma-separated, e.g. 1.25,1.5): ").strip()
            try:
                parts = [float(x) for x in raw.split(',')]
                if len(parts) == 2:
                    print(f"  Playing triad: 1.0, {parts[0]}, {parts[1]}")
                    play_sound(spectrum, ref_freq, [1.0, parts[0], parts[1]])
                else:
                    print("  Please enter exactly two ratios.")
            except ValueError:
                print("  Invalid input.")

        else:
            print("  Invalid choice.")


if __name__ == '__main__':
    main()
