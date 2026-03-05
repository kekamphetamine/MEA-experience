import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

# Constants
SAMPLING_RATE = 10000  # Hz
WINDOW = 10000         # 1 second
BASELINE_OFFSET = 100000  # 10 seconds
NPERSEG = 8192
NOVERLAP = NPERSEG // 2
EXCLUDE_ELECTRODES = []

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
    keys = sorted(durations.keys())
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

# Step 4: Define background blocks
def define_background_blocks(offsets, durations):
    background_blocks = []
    for label in ['ContBack', 'CChBack']:
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

# Step 5: Build full timeline
def build_full_timeline(mouse_folder):
    base_path = rf'T:\James_q\Experiments\ACh\{mouse_folder}'
    raw_path = os.path.join(base_path, 'rawMCS')
    timeline_path = os.path.join(base_path, 'exp_timeline')

    durations = get_durations(raw_path)
    offsets = define_offsets(durations)

    cont_file = os.path.join(timeline_path, 'odor_intervals_summaryCONT.txt')
    cch_file = os.path.join(timeline_path, 'odor_intervals_summaryCCH.txt')

    cont_exp_key = next(k for k in offsets if 'ContExp' in k)
    cch_exp_key = next(k for k in offsets if 'CChExp' in k)

    cont_df = load_and_align_stimuli(cont_file, offsets[cont_exp_key], 'ContExp')
    cch_df = load_and_align_stimuli(cch_file, offsets[cch_exp_key], 'CChExp')
    stimulus_df = pd.concat([cont_df, cch_df], ignore_index=True)

    background_df = define_background_blocks(offsets, durations)
    timeline_df = pd.concat([stimulus_df, background_df], ignore_index=True)

    return timeline_df

# Step 6: Compute normalized PSD
def compute_avg_norm_psd(channel_data, stim_times, baseline_starts=None):
    normalized_psds = []
    for i, evt in enumerate(stim_times):
        start_evoked = int(evt * SAMPLING_RATE)
        segment_evoked = channel_data[start_evoked:start_evoked + WINDOW]

        if baseline_starts is not None:
            start_baseline = baseline_starts[i]
        else:
            start_baseline = start_evoked - BASELINE_OFFSET

        segment_baseline = channel_data[start_baseline:start_baseline + WINDOW]

        f, Pxx_evoked = welch(segment_evoked, fs=SAMPLING_RATE, nperseg=NPERSEG, noverlap=NOVERLAP)
        _, Pxx_baseline = welch(segment_baseline, fs=SAMPLING_RATE, nperseg=NPERSEG, noverlap=NOVERLAP)

        Pxx_baseline[Pxx_baseline == 0] = np.finfo(float).eps
        Pxx_norm = Pxx_evoked / Pxx_baseline
        normalized_psds.append(Pxx_norm)

    return f, np.mean(normalized_psds, axis=0)

# Step 7: Run PSD analysis
def run_psd_analysis(mouse_folder, lfp_path):
    timeline_df = build_full_timeline(mouse_folder)
    lfp_data = np.load(lfp_path)

    stim_times_set1 = timeline_df[timeline_df['Section'] == 'ContExp']['Start_global'].values
    stim_times_set2 = timeline_df[timeline_df['Section'] == 'CChExp']['Start_global'].values

    shared_baselines = [int(evt * SAMPLING_RATE) - BASELINE_OFFSET for evt in stim_times_set1]

    psds_set1 = []
    psds_set2 = []
    n_channels = lfp_data.shape[1]

    for ch in range(n_channels):
        if ch in EXCLUDE_ELECTRODES:
            continue
        lfp_channel = lfp_data[:, ch]

        f, psd1 = compute_avg_norm_psd(lfp_channel, stim_times_set1, baseline_starts=shared_baselines)
        _, psd2 = compute_avg_norm_psd(lfp_channel, stim_times_set2, baseline_starts=shared_baselines)

        psds_set1.append(psd1)
        psds_set2.append(psd2)

    psds_set1 = np.array(psds_set1)
    psds_set2 = np.array(psds_set2)

    # Convert to arrays
    psds_set1 = np.array(psds_set1)
    psds_set2 = np.array(psds_set2)

    # Restrict to frequency ≤150 Hz
    f_limit = f <= 150
    f_plot = f[f_limit]
    psds_set1 = psds_set1[:, f_limit]
    psds_set2 = psds_set2[:, f_limit]

    # Plot 8 electrodes at a time
    n_electrodes = psds_set1.shape[0]
    group_size = 8

    for i in range(0, n_electrodes, group_size):
        fig, axs = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
        axs = axs.flatten()

        for j in range(group_size):
            idx = i + j
            if idx >= n_electrodes:
                axs[j].axis('off')
                continue

            axs[j].plot(f_plot, psds_set1[idx], label='Control', color='royalblue')
            axs[j].plot(f_plot, psds_set2[idx], label='CCh', color='darkorange')
            axs[j].set_title(f'Electrode {idx}')
            axs[j].legend(fontsize=8)

        fig.suptitle(f'Normalized PSDs — Electrodes {i} to {min(i + group_size - 1, n_electrodes - 1)}', fontsize=16)
        fig.supxlabel('Frequency (Hz)')
        fig.supylabel('Normalized PSD')
        plt.tight_layout()
        plt.show()


run_psd_analysis('20251020m2slice3', r"C:\Users\James\Desktop\Interim\2025-10-20Merge_npconv.npy")
