import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


# Load the Excel file
file_path = r"C:\Users\James\Desktop\sliceAnalysis-master\spykingcircus_output04_09\04_09spike_timing_with_stimulus.csv"
df = pd.read_csv(file_path)

# Load cluster group information
clustergroup_path = r"C:\Users\James\Desktop\sliceAnalysis-master\spykingcircus_output04_09\sorter_output\recording\recording_sc.GUI\cluster_group.tsv"
cluster_df = pd.read_csv(clustergroup_path, sep='\t')

# Filter for 'good' units
units_to_plot = cluster_df[cluster_df['group'] == 'good']['cluster_id'].tolist()
# --- PARAMETERS YOU CAN EDIT ---

# Define units of interest (optional: override auto-filtered good units)
units_to_plot = [3, 41, 61, 86, 95, 113]

# --- PARAMETERS ---
stim_start = 1495
stim_end = 1895
trial_duration = 20        # full trial
stim_on_duration = 10      # stimulus duration (first half of each trial)
stim_windows = np.arange(stim_start, stim_end, trial_duration)

# --- SDF SETUP ---
dt = 0.001
time_vector = np.arange(0, stim_on_duration, dt)
kernel_sd = 0.3
kernel = norm.pdf(np.arange(-3*kernel_sd, 3*kernel_sd, dt), 0, kernel_sd)

# --- ACCUMULATE ALL SPIKES FROM SELECTED UNITS ---
filtered_df = df[df['Unit'].isin(units_to_plot)]

# Initialize sdf
sdf = np.zeros_like(time_vector)

# Go through each trial and add spikes from ALL units
for window_start in stim_windows:
    window_end = window_start + stim_on_duration

    # Get all spikes from all selected units during stimulus ON
    trial_spikes = filtered_df[
        (filtered_df['Spike_Time'] >= window_start) &
        (filtered_df['Spike_Time'] < window_end)
    ]

    # Align spike times to stimulus onset
    aligned_spike_times = trial_spikes['Spike_Time'] - window_start

    for spike in aligned_spike_times:
        idx = int(spike / dt)
        start_idx = idx - len(kernel) // 2
        end_idx = start_idx + len(kernel)

        if start_idx >= 0 and end_idx < len(sdf):
            sdf[start_idx:end_idx] += kernel

# Average across trials
sdf /= len(stim_windows)

# --- PLOT ---
plt.figure(figsize=(10, 4))
plt.plot(time_vector, sdf, color='black', label='All Units (Averaged)')
plt.xlabel("Time relative to stimulus onset (s)")
plt.ylabel("Spike Density (a.u.)")
plt.title("Spike Density Function (All Units, Stimulus Only)")
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()