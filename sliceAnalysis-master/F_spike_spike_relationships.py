import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict
import math
import os

# Parameters
fs = 10000  # Hz
bin_size_ms = 1
half_width_ms = 100
bin_size = bin_size_ms / 1000
half_width = half_width_ms / 1000

stim_times_control = [160, 260, 360, 460, 560]
stim_times_ach = [1495, 1595, 1695, 1795, 1895]
window = 10  # sec

# Load data
fpath = r'C:\Users\James\Desktop\sliceAnalysis-master\spykingcircus_output04_09\sorter_output\recording\recording_sc.GUI\\'
spike_times = np.load(fpath + 'spike_times.npy') / fs  # convert to seconds
spike_clusters = np.load(fpath + 'spike_clusters.npy')

# Assign spike times per unit
units = np.unique(spike_clusters)
unit_spikes = {unit: spike_times[spike_clusters == unit] for unit in units}
print(f"Loaded spike data for {len(units)} units")

# Function to assign spikes to condition
def filter_by_condition(spike_times, stim_times, window):
    condition_spikes = []
    for stim in stim_times:
        mask = (spike_times >= stim) & (spike_times < stim + window)
        condition_spikes.append(spike_times[mask] - stim)  # align to stim
    return np.concatenate(condition_spikes) if condition_spikes else np.array([])

# Compute CCG
def compute_ccg(spike_times1, spike_times2, bin_size, half_width):
    edges = np.arange(-half_width, half_width + bin_size, bin_size)  # len = 2*half_width/bin_size + 1
    if len(spike_times1) == 0 or len(spike_times2) == 0:
        return np.zeros(len(edges)-1)
    diffs = np.concatenate([spike_times2 - t for t in spike_times1])
    hist, _ = np.histogram(diffs, bins=edges)
    return hist

# Containers for CCGs
ccgs_control = defaultdict(dict)
ccgs_ach = defaultdict(dict)
lags = np.arange(-half_width_ms, half_width_ms + bin_size_ms, bin_size_ms)  # e.g. -100 to 100 in ms

print("Computing CCGs for all unit pairs...")
for u1, u2 in combinations(units, 2):
    s1_ctrl = filter_by_condition(unit_spikes[u1], stim_times_control, window)
    s2_ctrl = filter_by_condition(unit_spikes[u2], stim_times_control, window)
    s1_ach = filter_by_condition(unit_spikes[u1], stim_times_ach, window)
    s2_ach = filter_by_condition(unit_spikes[u2], stim_times_ach, window)

    ccgs_control[u1][u2] = compute_ccg(s1_ctrl, s2_ctrl, bin_size, half_width)
    ccgs_ach[u1][u2] = compute_ccg(s1_ach, s2_ach, bin_size, half_width)
print("CCGs computed.")

# Plot example CCG for one pair
example_u1, example_u2 = units[0], units[1]
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(lags[:-1], ccgs_control[example_u1][example_u2], width=bin_size_ms)
plt.title(f"Control CCG: Unit {example_u1} ↔ Unit {example_u2}")
plt.xlabel("Lag (ms)")
plt.ylabel("Coincidences")

plt.subplot(1, 2, 2)
plt.bar(lags[:-1], ccgs_ach[example_u1][example_u2], width=bin_size_ms, color='r')
plt.title(f"ACh CCG: Unit {example_u1} ↔ Unit {example_u2}")
plt.xlabel("Lag (ms)")
plt.tight_layout()
#plt.show()

# === New: Generate heatmaps of cross-correlation peak values ===

units = np.array(sorted(units))
n_units = len(units)
metric_window_ms = 10  # window around zero lag to find peak
bin_centers = (lags[:-1] + lags[1:]) / 2
metric_indices = np.where((bin_centers >= -metric_window_ms) & (bin_centers <= metric_window_ms))[0]

ccg_metric_control = np.zeros((n_units, n_units))
ccg_metric_ach = np.zeros((n_units, n_units))

print("Computing cross-correlation peak metrics for heatmaps...")

for i, u1 in enumerate(units):
    for j, u2 in enumerate(units):
        if u1 == u2:
            ccg_metric_control[i, j] = 0
            ccg_metric_ach[i, j] = 0
        else:
            # Access the CCGs regardless of order, default zeros if missing
            if u2 in ccgs_control.get(u1, {}):
                ctrl = ccgs_control[u1][u2]
                ach = ccgs_ach[u1][u2]
            elif u1 in ccgs_control.get(u2, {}):
                ctrl = ccgs_control[u2][u1]
                ach = ccgs_ach[u2][u1]
            else:
                ctrl = np.zeros(len(bin_centers))
                ach = np.zeros(len(bin_centers))

            ccg_metric_control[i, j] = ctrl[metric_indices].max()
            ccg_metric_ach[i, j] = ach[metric_indices].max()

    if (i+1) % 10 == 0 or i == n_units-1:
        print(f"Processed {i+1} / {n_units} units")

print("Plotting heatmaps...")

import seaborn as sns

# Heatmap matrices
n_units = len(units)
ccg_metric_control = np.full((n_units, n_units), np.nan)
ccg_metric_ach = np.full((n_units, n_units), np.nan)

# Indices for computing CCG strength around 0 lag
center = len(lags) // 2
window_bins = int(10 / bin_size_ms)  # ±10 ms
metric_indices = slice(center - window_bins, center + window_bins + 1)

# Compute raw mean CCG around 0 lag for each unit pair
for i, u1 in enumerate(units):
    for j, u2 in enumerate(units):
        if u1 == u2 or u2 not in ccgs_control[u1]:
            continue
        ccg_ctrl = ccgs_control[u1][u2]
        ccg_ach = ccgs_ach[u1][u2]
        ccg_metric_control[i, j] = np.mean(ccg_ctrl[metric_indices])
        ccg_metric_ach[i, j] = np.mean(ccg_ach[metric_indices])

# Shared color scale for comparability
vmin_ctrl = np.nanmin(ccg_metric_control)
vmax_ctrl = np.nanmax(ccg_metric_control)

vmin_ach = np.nanmin(ccg_metric_ach)
vmax_ach = np.nanmax(ccg_metric_ach)

vmin_all = np.nanmin([ccg_metric_control, ccg_metric_ach])
vmax_all = np.nanmax([ccg_metric_control, ccg_metric_ach])

# Combined figure with 2 rows: one for separate scales, one for shared
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# --- Row 1: Independent scales ---
sns.heatmap(ccg_metric_control, ax=axs[0, 0], cmap="viridis",
            vmin=vmin_ctrl, vmax=vmax_ctrl,
            xticklabels=units, yticklabels=units, cbar=True, square=True)
axs[0, 0].set_title("Control (Independent scale)", fontsize=10)

sns.heatmap(ccg_metric_ach, ax=axs[0, 1], cmap="viridis",
            vmin=vmin_ach, vmax=vmax_ach,
            xticklabels=units, yticklabels=units, cbar=True, square=True)
axs[0, 1].set_title("ACh (Independent scale)", fontsize=10)

# --- Row 2: Shared scale ---
sns.heatmap(ccg_metric_control, ax=axs[1, 0], cmap="viridis",
            vmin=vmin_all, vmax=vmax_all,
            xticklabels=units, yticklabels=units, cbar=True, square=True)
axs[1, 0].set_title("Control (Shared scale)", fontsize=10)

sns.heatmap(ccg_metric_ach, ax=axs[1, 1], cmap="viridis",
            vmin=vmin_all, vmax=vmax_all,
            xticklabels=units, yticklabels=units, cbar=True, square=True)
axs[1, 1].set_title("ACh (Shared scale)", fontsize=10)

# Axis formatting
for ax in axs.flat:
    ax.set_xlabel("Unit", fontsize=4)
    ax.set_ylabel("Unit", fontsize=4)
    ax.tick_params(labelsize=3)

plt.tight_layout()
plt.savefig("ccg_heatmaps_both_styles.png", dpi=150)
plt.close()
print("✅ Combined heatmap figure saved as 'ccg_heatmaps_both_styles.png'")