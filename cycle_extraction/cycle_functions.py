import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
from scipy.signal import butter, sosfiltfilt
from scipy.stats import norm

from utils.utils import contains_nan


def butter_bandpass_sos(a, b, fs=1.0, order=10, mode='f'):
    """
    Create a Butterworth bandpass filter in second-order-section (SOS) form.
    Supports either frequency or period input, automatically handling inversion.

    Parameters
    ----------
    a, b : float
        Band edges.
        - If mode='f': frequencies
        - If mode='T': periods
        Order (a < b) is handled automatically.
    fs : float, optional
        Sampling frequency.
        Example: hourly data -> Default = 1.0
    order : int, optional
        The TOTAL filter order. This must be an even number. Default = 10.
    mode : {'f', 'T'}, optional
        Interpretation of 'a' and 'b':
        - 'f' → input values are frequencies
        - 'T' → input values are periods

    Returns
    -------
    sos : ndarray
        Second-order-section coefficients of the Butterworth bandpass filter.
    wn : tuple
        Normalized cutoff frequencies (low_norm, high_norm) relative to Nyquist.
    """

    # --- 1. Check if total order is even ---
    if order % 2 != 0:
        raise ValueError("Total filter order 'order' must be an even number.")

    # For a bandpass filter, the scipy.butter 'N' parameter is the
    # prototype order, which is half the total desired order.
    prototype_order_N = order // 2

    # --- 2. Convert to frequency domain if periods are given ---
    if mode == 'T':
        f1 = 1.0 / a  # higher frequency (shorter period)
        f2 = 1.0 / b  # lower frequency (longer period)
    elif mode == 'f':
        f1, f2 = a, b
    else:
        raise ValueError("mode must be 'f' (frequency) or 'T' (period).")

    # --- 3. Ensure proper order ---
    f_low, f_high = sorted([f1, f2])

    # --- 4. Normalize by Nyquist frequency ---
    nyq = 0.5 * fs
    low_norm = f_low / nyq
    high_norm = f_high / nyq

    # print(f"--- Butterworth Bandpass Filter ---")
    # print(f"Mode: {'Period' if mode == 'T' else 'Frequency'} input")
    # print(f"Sampling freq (fs): {fs} samples/day → Nyquist = {nyq:.4f} cycles/day")
    # print(f"Cutoff frequencies (cycles/day): {f_low:.6f} – {f_high:.6f}")
    # print(f"Normalized cutoff (0–1 Nyquist): {low_norm:.6f} – {high_norm:.6f}")
    # print(f"Total Filter Order: {order} (Prototype Order N={prototype_order_N})")
    # print("-----------------------------------")

    # --- 5. Sanity checks ---
    if low_norm <= 0 or high_norm >= 1 or low_norm >= high_norm:
        raise ValueError("Invalid normalized frequencies — check fs and input values.")

    # --- 6. Design Butterworth filter ---
    # Pass the correct prototype_order_N
    sos = butter(prototype_order_N, [low_norm, high_norm], btype='bandpass', output='sos')

    return sos, (low_norm, high_norm)


def apply_nc_filter(data, sos):
    """Apply zero-phase filtering using sosfiltfilt."""
    return sosfiltfilt(sos, data)


def nc_filter_circadian(x, range=0.33, fs=1.0, order=10):
    cf = 1 / 24
    # like code from Proix 2021, although Baud 2018 (in whom Proix refers to) describes +-33% in period
    low_cutoff = cf * (1 - range)
    # Karoly 2021 refers to +-33% in freq, but does +-33% in period, then converts to freq
    high_cutoff = cf * (1 + range)
    sos, _ = butter_bandpass_sos(low_cutoff, high_cutoff, fs=fs, order=order, mode='f')
    filtered_signal = apply_nc_filter(x, sos)
    return filtered_signal


def nc_filter_multidien(x, min_period=5 * 24, max_period=50 * 24, fs=1.0, order=10):
    sos, _ = butter_bandpass_sos(min_period, max_period, fs=fs, order=order, mode='T')
    filtered_signal = apply_nc_filter(x, sos)
    return filtered_signal


def nc_filter(x, fs=1.0, range_circadian=0.33, multid_min=5 * 24, multid_max=50 * 24, order=10,
              type_=('circadian', 'multidien'), figure=True, label=None):
    """
    Applies separate NON-CAUSAL bandpass filters for the circadian and multidien ranges
    and optionally generates a visualization of the results.

    Assumes the base unit of time is hours (i.e., fs=1.0 is 1 sample/hour).

    Parameters
    ----------
    x : array_like
        The input time-series signal (e.g., activity, EEG).
    fs : float, optional
        Sampling frequency in samples per hour. Default is 1.0.
    range_circadian : float, optional
        The fractional range (±) around the 24-hour circadian frequency. Default is 0.33 (33%).
    multid_min, multid_max : float, optional
        Min and max periods for the multidien filter defined in hours.
        Defaults are 120 hours (5 days) and 1200 hours (50 days).
    order : int, optional
        The TOTAL filter order. Must be even. Default is 10.
    type_: tuple, optional
        The type of periodicity.
    figure : bool, optional
        If True, generates and displays a plot of the original and filtered signals.
        Default is True.
    label : array_like, optional
        An array of binary or categorical labels (e.g., seizure markers) corresponding
        to the time points in `x`. Non-zero values are treated as event markers.
        Must have the same length as `x`. Default is None (no markers).

    Returns
    -------
    filtered_signals : DataFrame
        A df containing the filtered signals:
        {'circadian': array, 'multidien': array}.
    """
    if contains_nan(x): raise ValueError("x must not contain NaNs.")
    filtered_signals = pd.DataFrame()

    if 'circadian' in type_:
        filtered_signals['circadian'] = nc_filter_circadian(x, range=range_circadian, fs=fs, order=order)
    if 'multidien' in type_:
        filtered_signals['multidien'] = nc_filter_multidien(x, min_period=multid_min, max_period=multid_max, fs=fs,
                                                            order=order)

    if label is not None:
        label = np.asarray(label)
        if len(label) != len(x):
            raise ValueError("label must have the same length as x.")
        seizure_indices = np.where(label != 0)[0]
    else:
        seizure_indices = np.array([])

    if figure:
        plot_filtered(x, fs, filtered_signals, range_circadian, multid_min, multid_max, seizure_indices)

    return filtered_signals


def plot_filtered(x, fs, filtered_signals, range_circ, multid_min, multid_max, seizure_indices):
    t, n_features = filtered_signals.shape
    # Create subplots: n+1 for original + n filtered signals
    fig, axes = plt.subplots(n_features + 1, 1, figsize=(14, 2.2 * (n_features + 1)), sharex=True)

    # X-axis: time in days
    samples_per_day = fs * 24
    t = np.arange(t) / samples_per_day

    # Plot original signal
    axes[0].plot(t, x, color='black', lw=0.25)
    axes[0].set_title("Original Signal")

    # Plot seizure markers on original signal
    if len(seizure_indices) > 0:
        axes[0].scatter(t[seizure_indices], x[seizure_indices],
                        color='red', s=10, label=f'Seizures (n={len(seizure_indices)})', zorder=3)
        axes[0].legend(loc='upper right', fontsize=8)

    # Plot each filtered signal
    for i, name in enumerate(filtered_signals.columns, 1):
        if name == 'circadian':
            title = f"Circadian Band (Central T ≈ {24} hours ± {range_circ * 100}%)"
        if name == 'multidien':
            title = f"Multidien Band (T: {multid_min / 24}–{multid_max / 24} days)"

        y = filtered_signals[name]

        axes[i].plot(t, y, lw=1.0, label=name)

        # Title conversion: 1/cf is the period in hours. Divide by 24 to get period in days.
        axes[i].set_title(f"{title}")
        axes[i].grid(True, alpha=0.3)

        # Seizure markers
        if len(seizure_indices) > 0:
            # Use the filtered signal's value for the scatter marker
            axes[i].scatter(t[seizure_indices], y[seizure_indices],
                            color='red', s=10, zorder=3)

    axes[-1].set_xlabel("Time (days)")
    plt.tight_layout()
    plt.show()
    return


def compute_plv(signal, seizure_indices, n_events):
    """Computes the phase locking value of events (seizures) to a signal"""
    if contains_nan(signal): raise ValueError("signal contains NaN(s).")

    analytic_signal = scipy.signal.hilbert(signal)
    instantaneous_phase = np.angle(analytic_signal)
    event_phases = instantaneous_phase[seizure_indices]
    mean_complex_vector = np.sum(np.exp(1j * event_phases)) / n_events
    plv = np.abs(mean_complex_vector)
    mean_angle = np.angle(mean_complex_vector)
    mean_angle_deg = np.degrees(mean_angle) % 360

    return plv, mean_angle, mean_angle_deg, event_phases


def compute_plv_for_split_signal(signals: list[np.ndarray], szr_indices_list: list[np.ndarray]):
    """Computes the phase locking value of events (seizures) to split signals"""
    for i, sig in enumerate(signals):
        if contains_nan(sig): raise ValueError(f"signal {i} contains NaN(s).")

    # Process the signals
    analytic_signals = [scipy.signal.hilbert(sig) for sig in signals]
    instantaneous_phases = [np.angle(analytic_sig) for analytic_sig in analytic_signals]
    event_phases_list = [inst_phase[szr_idxs] for inst_phase, szr_idxs in zip(instantaneous_phases, szr_indices_list)]

    # Combine all extracted phases
    event_phases = np.concatenate(event_phases_list)
    n_events = len(event_phases)

    # Compute overall PLV and mean angles
    mean_complex_vector = np.sum(np.exp(1j * event_phases)) / n_events
    plv = np.abs(mean_complex_vector)
    mean_angle = np.angle(mean_complex_vector)
    mean_angle_deg = np.degrees(mean_angle) % 360

    return plv, mean_angle, mean_angle_deg, event_phases


def rayleigh_test(n_events, plv):
    """
    Circular statistics test for non-uniform, unimodal distribution around a circle.
    DO NOT use for testing PLV between two signals. Independency of N is violated due to autocorrelation
    """
    z_stat = n_events * (plv ** 2)
    if n_events < 50:
        p_value = np.exp(-z_stat) * (1 + (2 * z_stat - z_stat ** 2) / (4 * n_events))
        # correcting for negative p_value at high PLV cause (2*z_stat - z_stat**2) will become negative when z_stat > 2
        p_value = max(0, p_value)
    else:
        p_value = np.exp(-z_stat)

    return p_value, z_stat


def plot_single_phase_histogram(event_phases, plv, mean_angle, p_value,
                                title_text="Phase Distribution of Seizure Occurrences"):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    n_bins = 24
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    mean_angle_deg = np.degrees(mean_angle) % 360

    counts, _ = np.histogram(event_phases, bins=bin_edges)

    bars = ax.bar(bin_centers, counts, width=bin_width, bottom=0.0,
                  color='#1f77b4', alpha=0.6, edgecolor='k', linewidth=1)

    max_count = np.max(counts) if len(counts) > 0 else 1
    scaled_plv_length = plv * max_count

    ax.annotate('',
                xy=(mean_angle, scaled_plv_length),
                xytext=(0, 0),
                arrowprops=dict(edgecolor='r', facecolor='r', arrowstyle='-|>', lw=2, mutation_scale=20))

    ax.plot([], [], color='r', linestyle='-', markersize=5,
            label=(f'Mean Phase Angle = {mean_angle_deg:.1f}°'
                   f'\nPLV = {plv:.2f}'
                   f'\np = {p_value:.4f}'))

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.set_ylim(0, max_count * 1.2)

    tick_values = np.arange(0, int(max_count) + 1, 2)
    ax.set_yticks(tick_values)

    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax.set_xticklabels(['0°', '90°', '180°', '270°'], fontsize=12)

    # --- Add 'Falling' and 'Rising' text on the outside curvature ---
    text_radius = max_count * 1.05

    # 90 degrees (pi/2) is the center of the 0 to 180 falling side
    ax.text(np.pi / 2, text_radius, 'Falling', ha='center', va='center',
            rotation=-90, fontsize=14, color='darkred', weight='bold')

    # 270 degrees (3*pi/2) is the center of the 180 to 360 rising side
    ax.text(3 * np.pi / 2, text_radius, 'Rising', ha='center', va='center',
            rotation=90, fontsize=14, color='darkgreen', weight='bold')

    # Changed to ax.set_title to apply to the specific subplot
    ax.set_title(title_text, fontsize=14, y=1.1)
    ax.legend(loc='lower right', bbox_to_anchor=(1.3, 0.1))

    plt.tight_layout(pad=1.5, h_pad=1.5, w_pad=0.5)
    plt.show()

    return fig, ax


def hanley_mcneil_test(N1, N2, auc):
    """
    Hanley-McNeil method for ROC-AUC significance/better than chance classification performance.
    """
    # for chance predictor
    a = 0.5
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)

    if not isinstance(N1, (list, tuple, np.ndarray)):
        n1 = N1
        n2 = N2
        std_auc = np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n2 - 1) * (q2 - a * a)) / (n1 * n2))
        p = norm.sf(auc, loc=a, scale=std_auc)

        if auc < 0.5 and p < 0.5:
            p = 1 - p
    else:
        # auc=auc.flatten()
        p = []
        for n1, n2, auc_ in zip(N1, N2, auc):
            std_auc = np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n2 - 1) * (q2 - a * a)) / (n1 * n2))
            p_ = norm.sf(auc_, loc=a, scale=std_auc)

            if auc_ < 0.5 and p_ < 0.5:
                p_ = 1 - p_
                p.append(p_)
        p = np.array(p)
    return p
