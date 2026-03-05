import os
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import circmean, circstd
import numpy as np


SAMPLING_RATE = 10000  # Hz

# Step 1: Extract durations from .h5 files
def get_durations(folder_path):
    durations = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.h5'):
            full_path = os.path.join(folder_path, filename)
            with h5py.File(full_path, 'r') as f:
                try:
                    stream = f['Data']['Recording_0']['AnalogStream']['Stream_0']
                    samples = stream['ChannelData'].shape[1]
                    duration = samples / SAMPLING_RATE
                    durations[filename] = duration
                except Exception as e:
                    print(f"Error in {filename}: {e}")
    return durations

# Step 2: Define offsets based on durations
def define_offsets(durations):
    keys = sorted(durations.keys())  # sort for consistency
    offsets = {}
    current_offset = 0
    for key in keys:
        offsets[key] = current_offset
        current_offset += durations[key]
    return offsets

# Step 3: Load and align stimulus intervals
def load_and_align_stimuli(file_path, offset, section_name):
    df = pd.read_csv(file_path, sep='\t')
    df['Start_global'] = df['Start'] + offset
    df['End_global'] = df['End'] + offset
    df['Section'] = section_name
    df['Group'] = df['Identity'] + '-' + df['Waveform'] + '-' + df['Intensity Group']
    df['Type'] = 'Stimulus'
    return df

# Step 4: Define background blocks using known durations
def define_background_blocks(offsets, durations):
    background_blocks = []
    for label in ['ContBack', 'CChBack']:
        # Match filenames based on known patterns
        for fname in offsets:
            if label in fname:
                background_blocks.append({
                    'Start_global': offsets[fname],
                    'End_global': offsets[fname] + durations[fname],
                    'Group': 'Background',
                    'Section': label,
                    'Type': 'Background'
                })
    return pd.DataFrame(background_blocks)

# Step 5: Plot the timeline
def plot_stimulus_timeline(df, title='Experimental Stimulus Timeline'):
    fig, ax = plt.subplots(figsize=(14, 2))  # Wide and short layout

    df = df.sort_values('Start_global').reset_index(drop=True)
    df['Group'] = df['Group'].fillna('Unknown')
    df['Color'] = df['Type'].map({'Background': 'gray', 'Stimulus': 'C0'})

    # Plot each block as a horizontal bar on the same line
    for i, row in df.iterrows():
        start = row['Start_global']
        end = row['End_global']
        midpoint = (start + end) / 2

        ax.barh(y=0, width=end - start, left=start, height=0.4,
                color=row['Color'], edgecolor='black')

        # Add label directly above the bar
        ax.text(midpoint, 0.5, row['Group'],
                rotation=90, ha='center', va='bottom',
                fontsize=8, color='black', clip_on=True)

    ax.set_yticks([])
    ax.set_xlabel('Time (s)')
    ax.set_title(title)
    ax.set_xlim(0, df['End_global'].max() * 1.01)
    ax.set_ylim(-0.5, 1.5)  # Ensure label space is visible
    plt.tight_layout()
    #plt.show()

# Step 6: Main pipeline
def build_full_timeline(mouse_folder):
    base_path = rf'T:\James_q\Experiments\ACh\{mouse_folder}'
    raw_path = os.path.join(base_path, 'rawMCS')
    timeline_path = os.path.join(base_path, 'exp_timeline')

    durations = get_durations(raw_path)
    offsets = define_offsets(durations)

    cont_file = os.path.join(timeline_path, 'odor_intervals_summaryCONT.txt')
    cch_file = os.path.join(timeline_path, 'odor_intervals_summaryCCH.txt')

    # Match filenames for experimental sections
    cont_exp_key = next(k for k in offsets if 'ContExp' in k)
    cch_exp_key = next(k for k in offsets if 'CChExp' in k)

    cont_df = load_and_align_stimuli(cont_file, offsets[cont_exp_key], 'ContExp')
    cch_df = load_and_align_stimuli(cch_file, offsets[cch_exp_key], 'CChExp')
    stimulus_df = pd.concat([cont_df, cch_df], ignore_index=True)

    background_df = define_background_blocks(offsets, durations)
    timeline_df = pd.concat([stimulus_df, background_df], ignore_index=True)

    plot_stimulus_timeline(timeline_df)
    return timeline_df

def split_spike_times_by_condition(spike_times_sec, spike_clusters, split_time):
    # Control: spikes before split_time
    control_mask = spike_times_sec < split_time
    cch_mask = spike_times_sec >= split_time

    control_spike_times = spike_times_sec[control_mask]
    control_spike_clusters = spike_clusters[control_mask]

    cch_spike_times = spike_times_sec[cch_mask]
    cch_spike_clusters = spike_clusters[cch_mask]

    return (control_spike_times, control_spike_clusters), (cch_spike_times, cch_spike_clusters)


# BUILD TIMELINE OF EXPERIMENT
timeline_df = build_full_timeline('20251020m2slice3')

# === 1. Load spike times ===
spike_times_path = r"C:\Users\James\Desktop\Interim\2025-10-20Merge_npconv\2025-10-20Merge_npconvsc_.GUI\spike_times.npy"
cluster_path = r"C:\Users\James\Desktop\Interim\2025-10-20Merge_npconv\2025-10-20Merge_npconvsc_.GUI\spike_clusters.npy"

spike_times = np.load(spike_times_path)   # in seconds
spike_clusters = np.load(cluster_path)    # cluster IDs for each spike

units = np.unique(spike_clusters)
print(f"Found {len(units)} units")

# Convert spike times from samples → seconds
spike_times_samples = np.load(spike_times_path)   # raw output
spike_times_sec = spike_times_samples / 10000        # convert to seconds

# After building timeline_df
background_blocks = timeline_df[timeline_df['Group'] == 'Background']
split_time = background_blocks['Start_global'].max()
print(f"Using split time: {split_time:.2f} seconds")

(control_spike_times, control_spike_clusters), (cch_spike_times, cch_spike_clusters) = \
    split_spike_times_by_condition(spike_times_sec, spike_clusters, split_time)


def plot_trial_aligned_raster(unit_id, spike_times_sec, spike_clusters, timeline_df, condition_label, window=None):
    """
    Plot a raster for one unit, aligned to the start of each trial of a given condition.
    All trials are overlaid on the same time axis.
    """
    # Select spikes for this unit
    unit_spikes = spike_times_sec[spike_clusters == unit_id]

    # Select blocks of the chosen condition
    condition_blocks = timeline_df[timeline_df['Group'] == condition_label]

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (_, row) in enumerate(condition_blocks.iterrows(), 1):
        start, end = row['Start_global'], row['End_global']
        block_end = start + window if window else end

        # Spikes in this block, aligned to block start
        mask = (unit_spikes >= start) & (unit_spikes < block_end)
        trial_spikes = unit_spikes[mask] - start

        # Plot them as dots, all on the same y-level (overlayed)
        ax.scatter(trial_spikes, np.ones_like(trial_spikes)*i,
                   marker='|', s=60, color='black')

    ax.set_xlabel("Time from trial start (s)")
    ax.set_ylabel("Trial #")
    ax.set_title(f"Unit {unit_id} — {condition_label} trials")
    plt.tight_layout()
    plt.show()

print("Loading LFP")
# === 1. Load LFP ===
lfp_path = r"C:\Users\James\Desktop\Interim\2025-10-20Merge_npconv.npy"
lfp_data = np.load(lfp_path)
lfp_data = lfp_data[:,10]  # pick channel (electrode)

fs = 10000  # Hz

lfp_duration_sec = len(lfp_data) / fs
print(f"LFP duration: {lfp_duration_sec:.2f} seconds")

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)

print("Filtering LFP for gamma band")
gamma_lfp = bandpass_filter(lfp_data, 40, 110, fs)
print("hilbert is up in here")
analytic_signal = hilbert(gamma_lfp)
print("getting analytic signal")
gamma_phase = np.angle(analytic_signal)

def get_spike_phases(spike_times, gamma_phase, fs, lfp_start_sec=0):
    indices = ((spike_times - lfp_start_sec) * fs).astype(int)
    indices = indices[(indices >= 0) & (indices < len(gamma_phase))]
    return gamma_phase[indices]

def compute_plv(phases):
    """
    Compute Phase Locking Value (PLV) from spike phases.
    """
    return np.abs(np.mean(np.exp(1j * phases)))

def compute_mean_vector(phases):
    """
    Compute mean angle and vector length.
    """
    mean_angle = np.angle(np.mean(np.exp(1j * phases)))
    vector_length = np.abs(np.mean(np.exp(1j * phases)))
    return mean_angle, vector_length

def plot_circular_histograms_batch(units, spike_times_sec, spike_clusters, gamma_phase, fs, timeline_df,
                                   condition_label, batch_size=8, plv_thresh=0.15, group_mode='base'):
    """
    group_mode: 'base' → use base odor label (e.g., 'S1'), combining all subconditions
                'full' → use full condition label (e.g., 'S1-Co-H')
    """
    # Determine which blocks to use
    if group_mode == 'base':
        timeline_df = timeline_df.copy()
        timeline_df['BaseOdor'] = timeline_df['Group'].str.extract(r'^(C′|C|S1|S2|D)')
        condition_blocks = timeline_df[timeline_df['BaseOdor'] == condition_label]
    else:
        condition_blocks = timeline_df[timeline_df['Group'] == condition_label]

    def get_condition_spikes(spike_times, blocks):
        spikes = []
        for _, row in blocks.iterrows():
            start, end = row['Start_global'], row['End_global']
            mask = (spike_times >= start) & (spike_times < end)
            spikes.extend(spike_times[mask])
        return np.array(spikes)

    for i in range(0, len(units), batch_size):
        fig, axs = plt.subplots(2, 4, subplot_kw={'projection': 'polar'}, figsize=(16, 8))
        axs = axs.flatten()

        for j, unit_id in enumerate(units[i:i + batch_size]):
            unit_spikes = spike_times_sec[spike_clusters == unit_id]
            condition_spikes = get_condition_spikes(unit_spikes, condition_blocks)
            spike_phases = get_spike_phases(condition_spikes, gamma_phase, fs)

            if len(spike_phases) < 15:
                axs[j].set_title(f"Unit {unit_id}\n<15 spikes", va='bottom')
                axs[j].set_xticks([])
                axs[j].set_yticks([])
                continue

            plv = compute_plv(spike_phases)
            mean_angle, vector_length = compute_mean_vector(spike_phases)
            color = 'crimson' if plv >= plv_thresh else 'teal'

            bins = 20
            bin_edges = np.linspace(-np.pi, np.pi, bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            counts, _ = np.histogram(spike_phases, bins=bin_edges)
            proportions = counts / len(spike_phases)

            axs[j].bar(bin_centers, proportions, width=(2 * np.pi) / bins, bottom=0.0, color=color, edgecolor='black')
            axs[j].set_ylim(0, 0.2)
            axs[j].set_yticks([0.1, 0.2])
            axs[j].arrow(mean_angle, 0, 0, vector_length,
                         width=0.02, color='black', alpha=0.8,
                         length_includes_head=True, head_width=0.1, head_length=0.1)
            axs[j].set_title(f"Unit {unit_id}\nPLV={plv:.2f}, n={len(spike_phases)}", va='bottom')

        for k in range(j + 1, batch_size):
            axs[k].axis('off')

        plt.tight_layout()
        plt.show()


print("starting plotting stuff")

def plot_significant_units_by_base_odor_dual(units, spike_times_sec, spike_clusters, gamma_phase, fs, timeline_df,
                                             plv_thresh=0.10, batch_size=8, bins=20):
    # Step 1: Split spike times by split_time
    background_blocks = timeline_df[timeline_df['Group'] == 'Background']
    split_time = background_blocks['Start_global'].max()
    print(f"Using split time: {split_time:.2f} seconds")

    control_mask = spike_times_sec < split_time
    cch_mask = spike_times_sec >= split_time

    control_spike_times = spike_times_sec[control_mask]
    control_spike_clusters = spike_clusters[control_mask]
    cch_spike_times = spike_times_sec[cch_mask]
    cch_spike_clusters = spike_clusters[cch_mask]

    # Step 2: Extract base odors
    timeline_df = timeline_df.copy()
    timeline_df['BaseOdor'] = timeline_df['Group'].str.extract(r'^(C′|C|S1|S2|D)')
    base_odors = ["C′", "C", "S1", "S2", "D"]

    def get_spikes_for_blocks(spike_times, spike_clusters, unit_id, blocks):
        spikes = []
        unit_spikes = spike_times[spike_clusters == unit_id]
        for _, row in blocks.iterrows():
            start, end = row['Start_global'], row['End_global']
            mask = (unit_spikes >= start) & (unit_spikes < end)
            spikes.extend(unit_spikes[mask])
        return np.array(spikes)

    for odor in base_odors:
        odor_blocks = timeline_df[timeline_df['BaseOdor'] == odor]
        significant_units = []

        for unit_id in units:
            control_spikes = get_spikes_for_blocks(control_spike_times, control_spike_clusters, unit_id, odor_blocks)
            cch_spikes = get_spikes_for_blocks(cch_spike_times, cch_spike_clusters, unit_id, odor_blocks)

            control_phases = get_spike_phases(control_spikes, gamma_phase, fs)
            cch_phases = get_spike_phases(cch_spikes, gamma_phase, fs)

            control_plv = compute_plv(control_phases) if len(control_phases) >= 15 else 0
            cch_plv = compute_plv(cch_phases) if len(cch_phases) >= 15 else 0

            if control_plv > plv_thresh or cch_plv > plv_thresh:
                significant_units.append((unit_id, control_phases, cch_phases))

        if not significant_units:
            print(f"No significant units for odor: {odor}")
            continue

        print(f"Plotting {len(significant_units)} units for odor: {odor}")
        for i in range(0, len(significant_units), batch_size):
            fig, axs = plt.subplots(2, batch_size, subplot_kw={'projection': 'polar'}, figsize=(4 * batch_size, 8))
            axs = axs.flatten()

            for j, (unit_id, control_phases, cch_phases) in enumerate(significant_units[i:i + batch_size]):
                for k, (phases, label) in enumerate([(control_phases, 'Control'), (cch_phases, 'CCh')]):
                    ax = axs[j + k * batch_size]
                    if len(phases) < 1:
                        ax.set_title(f"Unit {unit_id} — {label}\n<15 spikes", va='bottom')
                        ax.set_xticks([]); ax.set_yticks([])
                        continue

                    plv = compute_plv(phases)
                    mean_angle, vector_length = compute_mean_vector(phases)
                    color = 'crimson' if plv >= plv_thresh else 'teal'

                    bin_edges = np.linspace(-np.pi, np.pi, bins + 1)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    counts, _ = np.histogram(phases, bins=bin_edges)
                    proportions = counts / len(phases)

                    ax.bar(bin_centers, proportions, width=(2 * np.pi) / bins, bottom=0.0,
                           color=color, edgecolor='black')
                    ax.set_ylim(0, 0.2)
                    ax.set_yticks([0.1, 0.2])
                    ax.arrow(mean_angle, 0, 0, vector_length,
                             width=0.02, color='black', alpha=0.8,
                             length_includes_head=True, head_width=0.1, head_length=0.1)
                    ax.set_title(f"Unit {unit_id} — {label}\nPLV={plv:.2f}, n={len(phases)}", va='bottom')

            plt.suptitle(f"Base Odor: {odor}", fontsize=16)
            plt.tight_layout()
            plt.show()


#plot_significant_units_by_base_odor_dual(units, spike_times_sec, spike_clusters,
#                                         gamma_phase, fs, timeline_df,
#                                         plv_thresh=0.15, batch_size=4, bins=30)

def plot_overlay_all_units(units, spike_times, spike_clusters, gamma_phase, fs, timeline_df,
                           odor, bins=30, plv_thresh=0.15):
    odor_blocks = timeline_df[timeline_df['Group'].str.contains(fr'^{odor}[-]', regex=True)]

    all_phases = []
    for unit_id in units:
        unit_spikes = spike_times[spike_clusters == unit_id]
        unit_phases = []

        for _, row in odor_blocks.iterrows():
            start, end = row['Start_global'], row['End_global']
            spikes = unit_spikes[(unit_spikes >= start) & (unit_spikes < end)]
            phases = get_spike_phases(spikes, gamma_phase, fs)
            unit_phases.extend(phases)

        if len(unit_phases) >= 15:
            unit_phases = np.array(unit_phases)
            plv = compute_plv(unit_phases)
            if plv >= plv_thresh:
                all_phases.extend(unit_phases)

    if len(all_phases) < 15:
        print(f"Not enough significant spikes to plot overlay for odor {odor}")
        return

    all_phases = np.array(all_phases)
    plv = compute_plv(all_phases)
    mean_angle, vector_length = compute_mean_vector(all_phases)
    color = 'crimson' if plv >= plv_thresh else 'teal'

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
    bin_edges = np.linspace(-np.pi, np.pi, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    counts, _ = np.histogram(all_phases, bins=bin_edges)
    proportions = counts / len(all_phases)

    ax.set_ylim(0, 0.1)
    ax.set_yticks([0.05, 0.1])
    ax.bar(bin_centers, proportions, width=(2 * np.pi) / bins, bottom=0.0,
           color=color, edgecolor='black', alpha=0.7)
    #ax.arrow(mean_angle, 0, 0, vector_length,
    #         width=0.02, color='black', alpha=0.9,
    #         length_includes_head=True, head_width=0.1, head_length=0.1)
    ax.set_title(f"Overlay: {odor} — PLV={plv:.2f}, n={len(all_phases)}", va='bottom')
    plt.show()


#plot_overlay_all_units(units, control_spike_times, control_spike_clusters, gamma_phase, fs, timeline_df, odor='C')

#plot_overlay_all_units(units, cch_spike_times, cch_spike_clusters, gamma_phase, fs, timeline_df, odor='C')

def plot_dual_overlay_phase_aligned_by_mode(units, control_spike_times, control_spike_clusters,
                                            cch_spike_times, cch_spike_clusters,
                                            gamma_phase, fs, timeline_df, odor,
                                            bins=20, plv_thresh=0.10, figure_title=None):
    odor_blocks = timeline_df[timeline_df['Group'].str.contains(fr'^{odor}[-]', regex=True)]

    control_aligned = []
    cch_aligned = []

    for unit_id in units:
        def get_aligned_phases(spike_times, spike_clusters):
            unit_spikes = spike_times[spike_clusters == unit_id]
            unit_phases = []
            for _, row in odor_blocks.iterrows():
                start, end = row['Start_global'], row['End_global']
                spikes = unit_spikes[(unit_spikes >= start) & (unit_spikes < end)]
                phases = get_spike_phases(spikes, gamma_phase, fs)
                unit_phases.extend(phases)
            return np.array(unit_phases)

        control_phases = get_aligned_phases(control_spike_times, control_spike_clusters)
        cch_phases = get_aligned_phases(cch_spike_times, cch_spike_clusters)

        control_plv = compute_plv(control_phases) if len(control_phases) >= 15 else 0
        cch_plv = compute_plv(cch_phases) if len(cch_phases) >= 15 else 0

        if control_plv >= plv_thresh or cch_plv >= plv_thresh:
            def align_by_mode(phases):
                if len(phases) < 15:
                    return []
                bin_edges = np.linspace(-np.pi, np.pi, bins + 1)
                counts, _ = np.histogram(phases, bins=bin_edges)
                mode_bin_index = np.argmax(counts)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                mode_phase = bin_centers[mode_bin_index]
                return (phases - mode_phase + np.pi) % (2 * np.pi) - np.pi

            control_aligned.extend(align_by_mode(control_phases))
            cch_aligned.extend(align_by_mode(cch_phases))

    if len(control_aligned) < 15 and len(cch_aligned) < 15:
        print(f"Not enough significant aligned spikes to plot dual overlay for odor {odor}")
        return

    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(12, 6))
    titles = ['Control', 'CCh']
    data_sets = [control_aligned, cch_aligned]
    colors = ['royalblue', 'darkorange']

    for ax, title, phases, color in zip(axs, titles, data_sets, colors):
        if len(phases) < 15:
            ax.set_title(f"{title}: <15 spikes", va='bottom')
            ax.set_xticks([]); ax.set_yticks([])
            continue

        phases = np.array(phases)
        plv = compute_plv(phases)
        z_stat = len(phases) * plv ** 2

        bin_edges = np.linspace(-np.pi, np.pi, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        counts, _ = np.histogram(phases, bins=bin_edges)
        proportions = counts / len(phases)

        ax.bar(bin_centers, proportions, width=(2 * np.pi) / bins, bottom=0.0,
               color=color, edgecolor='black', alpha=0.8)
        ax.set_title(f"{title}: rPLV={z_stat:.2f}, n={len(phases)}", va='bottom')
        ax.set_ylim(0, 0.1)
        ax.set_yticks([0.05, 0.1])

    if figure_title:
        plt.suptitle(figure_title, fontsize=16)
    else:
        plt.suptitle(f"Mode-Aligned Dual Overlay: {odor}", fontsize=16)

    plt.tight_layout()
    plt.show()

#plot_dual_overlay_phase_aligned_by_mode(units,
#    control_spike_times, control_spike_clusters,
#    cch_spike_times, cch_spike_clusters,
#    gamma_phase, fs, timeline_df,
#    odor="C′", bins=30, plv_thresh=0.15,
#    figure_title="C' Units — Mode-Aligned Dual Overlay")




import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

def compute_rplv(phases):
    phases = np.array(phases)
    plv = np.abs(np.mean(np.exp(1j * phases)))
    return len(phases) * plv**2  # Rayleigh's Z

def get_unit_phases(spike_times, spike_clusters, unit_id, blocks, gamma_phase, fs):
    unit_spikes = spike_times[spike_clusters == unit_id]
    phases = []
    for _, row in blocks.iterrows():
        start, end = row['Start_global'], row['End_global']
        spikes = unit_spikes[(unit_spikes >= start) & (unit_spikes < end)]
        phases.extend(get_spike_phases(spikes, gamma_phase, fs))
    return np.array(phases)

def compare_rplv_significant_units(units, control_spike_times, control_spike_clusters,
                                    cch_spike_times, cch_spike_clusters,
                                    gamma_phase, fs, timeline_df, plv_thresh=0.10):
    base_odors = ["C′", "C", "S1", "S2", "D"]
    control_rplvs = []
    cch_rplvs = []

    for unit_id in units:
        control_all = []
        cch_all = []

        for odor in base_odors:
            odor_blocks = timeline_df[timeline_df['Group'].str.contains(fr'^{odor}[-]', regex=True)]
            control_phases = get_unit_phases(control_spike_times, control_spike_clusters, unit_id, odor_blocks, gamma_phase, fs)
            cch_phases = get_unit_phases(cch_spike_times, cch_spike_clusters, unit_id, odor_blocks, gamma_phase, fs)

            if len(control_phases) >= 15:
                control_all.extend(control_phases)
            if len(cch_phases) >= 15:
                cch_all.extend(cch_phases)

        control_all = np.array(control_all)
        cch_all = np.array(cch_all)

        control_plv = np.abs(np.mean(np.exp(1j * control_all))) if len(control_all) >= 15 else 0
        cch_plv = np.abs(np.mean(np.exp(1j * cch_all))) if len(cch_all) >= 15 else 0

        if control_plv >= plv_thresh or cch_plv >= plv_thresh:
            if len(control_all) >= 15:
                control_rplvs.append(len(control_all) * control_plv**2)
            else:
                control_rplvs.append(0)
            if len(cch_all) >= 15:
                cch_rplvs.append(len(cch_all) * cch_plv**2)
            else:
                cch_rplvs.append(0)

    # Significance test
    stat, p = wilcoxon(control_rplvs, cch_rplvs)

    # Plotting
    means = [np.mean(control_rplvs), np.mean(cch_rplvs)]
    labels = ['Control', 'CCh']
    colors = ['royalblue', 'darkorange']

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(labels, means, color=colors)
    ax.set_ylabel("Mean Rayleigh Z (rPLV)")
    ax.set_title(f"rPLV Comparison (Significant Units)\nWilcoxon p = {p:.4f}")
    plt.tight_layout()
    plt.show()

#compare_rplv_significant_units(units,
#    control_spike_times, control_spike_clusters,
#    cch_spike_times, cch_spike_clusters,
#    gamma_phase, fs, timeline_df,
#    plv_thresh=0.12)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def compute_rayleigh_matrix_by_base_odor(units, spike_times_sec, spike_clusters, gamma_phase, fs, timeline_df, plv_thresh=0.10):
    timeline_df = timeline_df.copy()
    timeline_df['BaseOdor'] = timeline_df['Group'].str.extract(r'^(C′|C|S1|S2|D)')
    base_odors = timeline_df['BaseOdor'].dropna().unique()

    rayleigh_matrix = pd.DataFrame(index=units, columns=base_odors)

    def get_spikes_for_blocks(spike_times, blocks):
        spikes = []
        for _, row in blocks.iterrows():
            start, end = row['Start_global'], row['End_global']
            mask = (spike_times >= start) & (spike_times < end)
            spikes.extend(spike_times[mask])
        return np.array(spikes)

    for odor in base_odors:
        blocks = timeline_df[timeline_df['BaseOdor'] == odor]

        for unit_id in units:
            unit_spikes = spike_times_sec[spike_clusters == unit_id]
            odor_spikes = get_spikes_for_blocks(unit_spikes, blocks)
            spike_phases = get_spike_phases(odor_spikes, gamma_phase, fs)

            if len(spike_phases) >= 15:
                plv = compute_plv(spike_phases)
                z_score = len(spike_phases) * plv**2
                rayleigh_matrix.loc[unit_id, odor] = z_score
            else:
                rayleigh_matrix.loc[unit_id, odor] = np.nan

    rayleigh_matrix = rayleigh_matrix.astype(float)

    # Filter units that are significantly phase locked in at least one odor
    plv_mask = rayleigh_matrix.apply(lambda row: any(row.dropna() >= (15 * plv_thresh**2)), axis=1)
    rayleigh_matrix = rayleigh_matrix[plv_mask]

    return rayleigh_matrix

# Compute Rayleigh Z matrix
rayleigh_matrix = compute_rayleigh_matrix_by_base_odor(units, spike_times_sec, spike_clusters,
                                                       gamma_phase, fs, timeline_df, plv_thresh=0.10)

# Reorder columns
ordered_odors = ["C′", "C", "S1", "S2", "D"]
rayleigh_matrix_ordered = rayleigh_matrix.reindex(columns=[col for col in ordered_odors if col in rayleigh_matrix.columns])

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(rayleigh_matrix_ordered, cmap='magma', vmin=0, vmax=10, cbar_kws={'label': 'Rayleigh Z'})
plt.xlabel("Base Odor")
plt.ylabel("Unit ID")
plt.title("Rayleigh Z (PLV Strength) by Unit and Odor")

# Show only first and last unit label
yticks = [0, len(rayleigh_matrix_ordered.index) - 1]
plt.yticks(yticks, [rayleigh_matrix_ordered.index[yticks[0]], rayleigh_matrix_ordered.index[yticks[1]]])
plt.tight_layout()
plt.show()


# Step 1: Drop "C′" column
rayleigh_matrix_filtered = rayleigh_matrix_ordered.drop(columns=["C′"], errors='ignore')

# Step 2: Sort units by Rayleigh Z for odor "C"
if "C" in rayleigh_matrix_filtered.columns:
    rayleigh_matrix_sorted = rayleigh_matrix_filtered.sort_values(by="C", ascending=False)
else:
    rayleigh_matrix_sorted = rayleigh_matrix_filtered.copy()

# Step 3: Plot the sorted heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(rayleigh_matrix_sorted, cmap='magma', vmin=0, vmax=10, cbar_kws={'label': 'Rayleigh Z'})
plt.xlabel("Base Odor (excluding C′)")
plt.ylabel("Unit ID (sorted by C)")
plt.title("Rayleigh Z by Unit and Odor (Sorted by C)")

# Show only first and last unit label
yticks = [0, len(rayleigh_matrix_sorted.index) - 1]
plt.yticks(yticks, [rayleigh_matrix_sorted.index[yticks[0]], rayleigh_matrix_sorted.index[yticks[1]]])
plt.tight_layout()
plt.show()


# Correlation matrix
correlation_matrix = rayleigh_matrix_ordered.corr(method='pearson')

# Plot correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Rayleigh Z Similarity Between Odors (Correlation)")
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Step 1: Define odor order and distances
odor_order = ["C", "S1", "S2", "D"]
odor_distances = {}

for i, odor_a in enumerate(odor_order):
    for j, odor_b in enumerate(odor_order):
        if i < j:
            distance = abs(i - j)
            odor_distances[(odor_a, odor_b)] = distance

# Step 2: Extract correlations for those pairs
correlation_pairs = []
distance_values = []

for (odor_a, odor_b), dist in odor_distances.items():
    if odor_a in correlation_matrix.columns and odor_b in correlation_matrix.columns:
        corr = correlation_matrix.loc[odor_a, odor_b]
        correlation_pairs.append(corr)
        distance_values.append(dist)

# Step 3: Plot correlation vs. odor distance
# Convert distances to strings for categorical x-axis
distance_labels = [str(d) for d in distance_values]

plt.figure(figsize=(6, 4))
sns.stripplot(x=distance_labels, y=correlation_pairs, jitter=True, size=8, color='mediumblue')
plt.xlabel("Odor Distance")
plt.ylabel("Correlation (Rayleigh Z)")
plt.title("Correlation vs. Odor Distance (Categorical)")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

def compute_ppc(phases):
    phases = np.array(phases)
    n = len(phases)
    if n < 2:
        return np.nan
    diffs = np.subtract.outer(phases, phases)
    cos_diffs = np.cos(diffs)
    ppc = np.sum(cos_diffs) / (n * (n - 1))
    return ppc

def compute_ppc_matrix_by_base_odor(units, spike_times_sec, spike_clusters, gamma_phase, fs, timeline_df):
    timeline_df = timeline_df.copy()
    timeline_df['BaseOdor'] = timeline_df['Group'].str.extract(r'^(C′|C|S1|S2|D)')
    base_odors = timeline_df['BaseOdor'].dropna().unique()

    ppc_matrix = pd.DataFrame(index=units, columns=base_odors)

    def get_spikes_for_blocks(spike_times, blocks):
        spikes = []
        for _, row in blocks.iterrows():
            start, end = row['Start_global'], row['End_global']
            mask = (spike_times >= start) & (spike_times < end)
            spikes.extend(spike_times[mask])
        return np.array(spikes)

    for odor in base_odors:
        blocks = timeline_df[timeline_df['BaseOdor'] == odor]
        for unit_id in units:
            unit_spikes = spike_times_sec[spike_clusters == unit_id]
            odor_spikes = get_spikes_for_blocks(unit_spikes, blocks)
            spike_phases = get_spike_phases(odor_spikes, gamma_phase, fs)

            if len(spike_phases) >= 15:
                ppc = compute_ppc(spike_phases)
                ppc_matrix.loc[unit_id, odor] = ppc
            else:
                ppc_matrix.loc[unit_id, odor] = np.nan

    return ppc_matrix.astype(float)

# Compute PPC matrix
ppc_matrix = compute_ppc_matrix_by_base_odor(units, spike_times_sec, spike_clusters,
                                             gamma_phase, fs, timeline_df)

reference_odor = "C"
sorted_units = ppc_matrix[reference_odor].sort_values(ascending=True).dropna().index

import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Create a rainbow colormap with as many colors as units
cmap = cm.get_cmap('rainbow', len(sorted_units))
unit_colors = {unit: cmap(i) for i, unit in enumerate(sorted_units)}

ordered_odors = ["C", "S1", "S2", "D"]
n_odors = len(ordered_odors)

# Compute global max PPC for shared y-axis
global_max_ppc = ppc_matrix.loc[sorted_units].max().max()

fig, axes = plt.subplots(n_odors, 1, figsize=(12, 2.5 * n_odors), sharex=True)

for i, odor in enumerate(ordered_odors):
    if odor not in ppc_matrix.columns:
        continue

    ppc_values = ppc_matrix.loc[sorted_units, odor]

    # Assign colors based on unit identity
    bar_colors = [unit_colors[unit] for unit in sorted_units]

    axes[i].bar(range(len(sorted_units)), ppc_values, color=bar_colors)
    axes[i].set_ylabel("PPC")
    axes[i].set_title(f"Odor: {odor}")
    #axes[i].set_ylim(0, global_max_ppc * 1.05)
    axes[i].set_ylim(0, 0.05)


# Label x-axis only on the bottom plot
axes[-1].set_xlabel("Unit (sorted by PPC to C)")
axes[-1].set_xticks([0, len(sorted_units)-1])
axes[-1].set_xticklabels([sorted_units[0], sorted_units[-1]])

plt.tight_layout()
plt.show()

# Step 1: Define odor distances
odor_order = ["C", "S1", "S2", "D"]
odor_distances = {
    ("C", "S1"): 1,
    ("C", "S2"): 2,
    ("C", "D"): 3,
    ("S1", "S2"): 1,
    ("S1", "D"): 2,
    ("S2", "D"): 1
}

# Step 2: Compute similarity matrix
similarity_matrix = ppc_matrix[odor_order].corr(method='pearson')

# Step 3: Extract similarity values and distances
distance_vals = []
similarity_vals = []

for (odor_a, odor_b), dist in odor_distances.items():
    if odor_a in similarity_matrix.columns and odor_b in similarity_matrix.columns:
        sim = similarity_matrix.loc[odor_a, odor_b]
        distance_vals.append(dist)
        similarity_vals.append(sim)

# Step 4: Plot
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
sns.stripplot(x=[str(d) for d in distance_vals], y=similarity_vals, jitter=True, size=8, color='darkgreen')
plt.xlabel("Odor Distance")
plt.ylabel("PPC Similarity (Correlation)")
plt.title("Odor Distance vs PPC Similarity")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()


