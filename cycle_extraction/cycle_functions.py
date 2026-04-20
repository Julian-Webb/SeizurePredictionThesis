from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import pandas as pd
import scipy.signal
from scipy.signal import butter, sosfiltfilt

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
    prototype_order_n = order // 2

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
    # print(f"Total Filter Order: {order} (Prototype Order N={prototype_order_n})")
    # print("-----------------------------------")

    # --- 5. Sanity checks ---
    if low_norm <= 0 or high_norm >= 1 or low_norm >= high_norm:
        raise ValueError("Invalid normalized frequencies — check fs and input values.")

    # --- 6. Design Butterworth filter ---
    # Pass the correct prototype_order_n
    sos = butter(prototype_order_n, [low_norm, high_norm], btype='bandpass', output='sos')

    return sos, (low_norm, high_norm)


def apply_nc_filter(data, sos):
    """Apply zero-phase filtering using sosfiltfilt."""
    return sosfiltfilt(sos, data)


def nc_filter_circadian(x, range_=0.33, fs=1.0, order=10):
    cf = 1 / 24
    # like code from Proix 2021, although Baud 2018 (in whom Proix refers to) describes +-33% in period
    low_cutoff = cf * (1 - range_)
    # Karoly 2021 refers to +-33% in freq, but does +-33% in period, then converts to freq
    high_cutoff = cf * (1 + range_)
    sos, _ = butter_bandpass_sos(low_cutoff, high_cutoff, fs=fs, order=order, mode='f')
    filtered_signal = apply_nc_filter(x, sos)
    return filtered_signal


def nc_filter_multidien(x, min_period=5 * 24.0, max_period=50 * 24.0, fs=1.0, order=10):
    if contains_nan(x): raise ValueError("x must not contain NaNs.")

    sos, _ = butter_bandpass_sos(min_period, max_period, fs=fs, order=order, mode='T')
    filtered_signal = apply_nc_filter(x, sos)
    return filtered_signal


def nc_filter(
        x,
        fs=1.0,
        range_circadian=0.33,
        multid_min=5 * 24,
        multid_max=50 * 24,
        order=10,
        types=('circadian', 'multidien'),
        figure=False,
        event_indices: np.ndarray = np.array([])
):
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
    types: tuple, optional
        The type of periodicity.
    figure : bool, optional
        If True, generates and displays a plot of the original and filtered signals.
        Default is True.
    event_indices : array_like, optional
        An array of indices corresponding to ``x`` which represent events (e.g. seizures).

    Returns
    -------
    filtered_signals : dict
        A df containing the filtered signals and plot:
        {'circadian': array, 'multidien': array, 'figure': matplotlib.pyplot.figure}.
    """
    if contains_nan(x): raise ValueError("x must not contain NaNs.")
    res = {}
    filtered_signals = pd.DataFrame()

    if 'circadian' in types:
        circ = nc_filter_circadian(x, range_=range_circadian, fs=fs, order=order)
        filtered_signals['circadian'], res['circadian'] = circ, circ
    if 'multidien' in types:
        md = nc_filter_multidien(x, min_period=multid_min, max_period=multid_max, fs=fs, order=order)
        filtered_signals['multidien'], res['multidien'] = md, md

    if figure:
        res['figure'] = plot_filtered(x, fs, filtered_signals, range_circadian, multid_min, multid_max, event_indices)

    return res


def plot_filtered_feature(
        original_sig: np.ndarray,
        filtered_sig: np.ndarray,
        samples_per_hour: float,
        time: np.ndarray = None,
        events: dict[str, np.ndarray] = None,
):
    """

    Parameters
    ----------
    original_sig
    filtered_sig
    samples_per_hour
    time
    events
        dict with event name as keys and event indices as values

    Returns
    -------

    """
    if time is None:
        samples_per_day = samples_per_hour * 24
        time = np.arange(len(original_sig)) / samples_per_day
        x_label = "Time (days)"
    else:
        x_label = "Time"

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 8))
    axes[-1].set_xlabel(x_label)

    axes[0].set_title("Original Signal")
    axes[0].plot(time, original_sig, color='k', lw=0.25)

    axes[1].set_title("Filtered Signal")
    axes[1].plot(time, filtered_sig, color='k')

    # Plot event markers
    if events is not None:
        for e_name, e_idxs in events.items():
            for ax, sig in zip(axes, [original_sig, filtered_sig]):
                xs, ys = time[e_idxs], sig[e_idxs]
                label = f'{e_name} (n={len(e_idxs)})'

                if e_name == 'seizures':
                    for i, x in enumerate(xs):
                        label = label if i == 0 else '_nolegend_'  # Make legend entry only appear once
                        ax.axvline(x, label=label, color="r", alpha=0.7)
                else:
                    ax.scatter(xs, ys, label=label, s=10, zorder=3, alpha=0.7, marker='x')

                ax.grid(True, alpha=0.3)

        axes[0].legend(loc='upper left')

    return fig


def plot_filtered(x, fs, filtered_signals, range_circ, multid_min, multid_max, event_indices):
    t, n_features = filtered_signals.shape
    # Create subplots: n+1 for original + n filtered signals
    fig, axes = plt.subplots(n_features + 1, 1, figsize=(14, 2.2 * (n_features + 1)), sharex=True, squeeze=False)

    # X-axis: time in days
    samples_per_day = fs * 24
    t = np.arange(t) / samples_per_day

    # Plot original signal
    axes[0].plot(t, x, color='black', lw=0.25)
    axes[0].set_title("Original Signal")

    # Plot seizure markers on original signal
    if len(event_indices) > 0:
        axes[0].scatter(t[event_indices], x[event_indices],
                        s=10, label=f'Seizures (n={len(event_indices)})', zorder=3)
        axes[0].legend(loc='upper right', fontsize=8)

    # Plot each filtered signal
    for i, name in enumerate(filtered_signals.columns, 1):
        if name == 'circadian':
            title = f"Circadian Band (Central T ≈ {24} hours ± {range_circ * 100}%)"
        elif name == 'multidien':
            title = f"Multidien Band (T: {multid_min / 24}–{multid_max / 24} days)"
        else:
            raise ValueError(f"Invalid name: {name}")

        y = filtered_signals[name]

        axes[i].plot(t, y, lw=1.0, label=name)

        # Title conversion: 1/cf is the period in hours. Divide by 24 to get period in days.
        axes[i].set_title(title)
        axes[i].grid(True, alpha=0.3)

        # Seizure markers
        if len(event_indices) > 0:
            # Use the filtered signal's value for the scatter marker
            axes[i].scatter(t[event_indices], y[event_indices], color='red', s=10, zorder=3)

    axes[-1].set_xlabel("Time (days)")
    return fig


def compute_plv(signal, event_indices, n_events):
    """Computes the phase locking value of events (e.g., seizures) to a signal"""
    if contains_nan(signal): raise ValueError("signal contains NaN(s).")

    analytic_signal = scipy.signal.hilbert(signal)
    instantaneous_phase = np.angle(analytic_signal)
    event_phases = instantaneous_phase[event_indices]
    mean_complex_vector = np.sum(np.exp(1j * event_phases)) / n_events
    plv = np.abs(mean_complex_vector)
    mean_angle = np.angle(mean_complex_vector)
    mean_angle_deg = np.degrees(mean_angle) % 360

    return plv, mean_angle, mean_angle_deg, event_phases


def compute_plv_for_split_signal(signals_per_chunk: list[np.ndarray], event_indices_per_chunk: list[np.ndarray]):
    """Computes the phase locking value of events (e.g., seizures) to split signals"""
    for i, sig in enumerate(signals_per_chunk):
        if contains_nan(sig): raise ValueError(f"signal {i} contains NaN(s).")

    # Process the signals
    analytic_signals = [scipy.signal.hilbert(sig) for sig in signals_per_chunk]
    instantaneous_phases = [np.angle(analytic_sig) for analytic_sig in analytic_signals]
    event_phases_list = [inst_phase[event_idxs] for inst_phase, event_idxs in
                         zip(instantaneous_phases, event_indices_per_chunk)]

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


def plot_phase_histogram_for_single_feature(
        ax,
        event_phases,
        plv,
        mean_angle,
        n_bins=24,
        show_x_ticks: frozenset[str] = frozenset({'top', 'right', 'bottom', 'left'}),
        max_y_ticks: int = 6,
):
    # Get bins and counts per bin
    bin_edges = np.linspace(-pi, pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    counts, _ = np.histogram(event_phases, bins=bin_edges)

    # Plot the histogram values in a polar fashion
    ax.bar(bin_centers, counts, width=bin_width, bottom=0.0, color='#1f77b4', alpha=0.6, edgecolor='k', linewidth=1)

    max_count = np.max(counts) if len(counts) > 0 else 1
    ax.set_ylim(0, max_count)

    # Show an arrow for the mean angle.
    # Disable shrink so PLV=1 reaches the exact outer radius.
    scaled_plv_length = plv * max_count
    if scaled_plv_length > 0:
        ax.annotate('',
                    xy=(mean_angle, scaled_plv_length),
                    xytext=(mean_angle, 0.0),
                    arrowprops=dict(edgecolor='r', facecolor='r', arrowstyle='-|>', lw=2,
                                    mutation_scale=20, shrinkA=0, shrinkB=0),
                    annotation_clip=False)

    ax.set_theta_zero_location("N")  # Set 0° to the top
    ax.set_theta_direction(-1)  # Make clockwise

    # Make circular y-ticks.
    max_y_ticks = max(2, int(max_y_ticks))
    step = max(1, int(np.ceil(max_count / (max_y_ticks - 1))))
    tick_values = np.arange(0, max_count + 1, step)
    ax.set_yticks(tick_values)

    # Make x-ticks: 0, 45, 90, ... 315 degrees
    x_tick_angles = [k * pi / 4 for k in range(8)]
    x_tick_labels = [
        '0°' if 'top' in show_x_ticks else '',
        '45°' if 'top_right' in show_x_ticks else '',
        'Falling' if 'right' in show_x_ticks else '',
        '135°' if 'bottom_right' in show_x_ticks else '',
        '180°' if 'bottom' in show_x_ticks else '',
        '225°' if 'bottom_left' in show_x_ticks else '',
        'Rising' if 'left' in show_x_ticks else '',
        '315°' if 'top_left' in show_x_ticks else '',
    ]
    ax.set_xticks(x_tick_angles, x_tick_labels, fontsize=12, color='grey')

    # Keep Falling/Rising as real tick labels and enforce rotation by tick index on each draw.
    # Tick order is [0, 45, 90, 135, 180, 225, 270, 315].
    def _apply_falling_rising_tick_rotation(_event=None):
        for i, tick in enumerate(ax.xaxis.get_major_ticks()):
            lbl = tick.label1
            if i in {2, 6}:  # 90° and 270°
                lbl.set_rotation(90)
                lbl.set_rotation_mode('default')
                lbl.set_transform_rotates_text(True)
            else:
                lbl.set_rotation(0)
            lbl.set_ha('center')
            lbl.set_va('center')

    _apply_falling_rising_tick_rotation()
    ax.figure.canvas.mpl_connect('draw_event', _apply_falling_rising_tick_rotation)

    # # --- Add 'Falling' and 'Rising' text on the outside curvature ---
    # text_radius = max_count * 1.05
    #
    # # 90 degrees (pi/2) is the center of the 0 to 180 falling side
    # ax.text(pi / 2, text_radius, 'Falling', ha='center', va='center',
    #         rotation=-90, fontsize=14, color='darkred', weight='bold')
    #
    # # 270 degrees (3*pi/2) is the center of the 180 to 360 rising side
    # ax.text(3 * pi / 2, text_radius, 'Rising', ha='center', va='center',
    #         rotation=90, fontsize=14, color='darkgreen', weight='bold')

    return ax


# noinspection PyDefaultArgument
def plot_phase_histogram_for_all_features(
        event_phases_per_feat: dict[str, np.ndarray],
        plv_per_feat: dict[str, float],
        mean_angle_per_feat: dict[str, float],
        p_bh_per_feat: dict[str, float],
        n_bins=24,
        ncols=5,
        max_n_y_ticks: int = 6,
        subplots_kwargs: dict = {'figsize': (15, 10)},
):
    n_feats = len(event_phases_per_feat)
    nrows = ceil(n_feats / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, subplot_kw={'projection': 'polar'}, **subplots_kwargs)

    # Plot on one axis per feature.
    feats = event_phases_per_feat.keys()
    for feat_i, feat in enumerate(feats):
        row, col = divmod(feat_i, ncols)
        ax = axes[row, col]

        # Configure x ticks
        top_row = row == 0
        bottom_row = row == nrows - 1
        left_col = col == 0
        right_col = col == ncols - 1

        show_x_ticks = frozenset(
            side for cond, side in (
                (top_row, "top"),
                (right_col, "right"),
                (bottom_row, "bottom"),
                (left_col, "left"),
                (top_row and right_col, "top_right"),
                (bottom_row and right_col, "bottom_right"),
                (bottom_row and left_col, "bottom_left"),
                (top_row and left_col, "top_left"),
            )
            if cond
        )

        plv = plv_per_feat[feat]
        mean_angle = mean_angle_per_feat[feat]
        mean_angle_deg = np.degrees(mean_angle) % 360
        p_bh = p_bh_per_feat[feat]

        plot_phase_histogram_for_single_feature(ax, event_phases_per_feat[feat], plv, mean_angle, n_bins, show_x_ticks,
                                                max_n_y_ticks)

        ax.set_title(feat, fontsize=14, y=1.1)
        # Add statistics info
        info = f'Mean Angle: {mean_angle_deg:.1f}°\np BH: {p_bh:.4f}\nPLV: {plv:.2f}'
        ax.annotate(info, xy=(0.02, 0.98), xycoords='axes fraction',
                    fontsize=8, ha='left', )

    return fig
