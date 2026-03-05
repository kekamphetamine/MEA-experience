import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import circmean, circstd
import matplotlib.pyplot as plt
import pandas as pd
import math

# Parameters
stim_times_set1 = [160, 260, 360, 460, 560]        # first stimulus series
stim_times_set2 = [1495, 1595, 1695, 1795, 1895]   # second stimulus series
baseline_offset = 100000  # 10 seconds before stimulus
post_stim_window = 10  # seconds
fs = 10000  # Hz
gamma_band = (43, 63)

# Welch params
nperseg = 8192
noverlap = nperseg // 2

start_sec = 140
lfp_offset_sec = 140  # because lfp_signal starts at 140s
end_sec = 2000
start_sample = int(start_sec * fs)  # should be 1,600,000
end_sample = int(end_sec * fs)      # should be 6,600,000

fpath = (r'C:\Users\James\Desktop\sliceAnalysis-master\spykingcircus_output04_09\sorter_output\\')
lfp_data = np.load(fpath + 'recording.npy')
lfp_raw = lfp_data[start_sample:end_sample, 0]  # time x channel

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)

lfp_signal = bandpass_filter(lfp_raw, gamma_band[0], gamma_band[1], fs)

# Load spike times (in samples)
spike_times = np.load(r'C:\Users\James\Desktop\sliceAnalysis-master\spykingcircus_output04_09\sorter_output\recording\recording_sc.GUI\spike_times.npy')

# Convert to seconds
fs = 10000  # Hz
spike_times_sec = spike_times / fs

# Assign to condition
def assign_spikes_to_condition(spike_times_sec, stim_times, window):
    condition_spikes = []
    for stim in stim_times:
        mask = (spike_times_sec >= stim) & (spike_times_sec < stim + window)
        condition_spikes.extend(spike_times_sec[mask])
    return np.array(condition_spikes)


def extract_gamma_rich_spikes(lfp, stim_times, spike_times, fs, window_sec=10, sd_thresh=2, lfp_offset_sec=140):
    gamma_spikes = []

    for stim_time in stim_times:
        # Align indexing with lfp that starts at lfp_offset_sec
        start = int((stim_time - lfp_offset_sec) * fs)
        end = int((stim_time - lfp_offset_sec + window_sec) * fs)

        if start < 0 or end > len(lfp):
            print(f"Skipping stim_time={stim_time} due to bounds")
            continue

        lfp_segment = lfp[start:end]

        # Hilbert envelope of gamma-filtered LFP
        gamma_envelope = np.abs(hilbert(lfp_segment))

        # Thresholding
        threshold = np.mean(gamma_envelope) + sd_thresh * np.std(gamma_envelope)
        gamma_mask = gamma_envelope > threshold

        # Spike selection
        spikes_in_window = spike_times[(spike_times >= stim_time) & (spike_times < stim_time + window_sec)]
        spike_indices = ((spikes_in_window - stim_time) * fs).astype(int)
        spike_indices = spike_indices[spike_indices < len(gamma_mask)]  # avoid out-of-bounds

        gamma_spikes_segment = spikes_in_window[gamma_mask[spike_indices]]
        gamma_spikes.extend(gamma_spikes_segment)

    return np.array(gamma_spikes)

spikes_control = assign_spikes_to_condition(spike_times_sec, stim_times_set1, post_stim_window)
spikes_ach = assign_spikes_to_condition(spike_times_sec, stim_times_set2, post_stim_window)

print(f"Total spikes loaded: {len(spike_times_sec)}")
print(f"Spikes in Control window: {len(spikes_control)}")
print(f"Spikes in ACh window:     {len(spikes_ach)}")


def get_lfp_phase_at_spikes(lfp_signal, spike_times_sec, fs):
    analytic = hilbert(lfp_signal)
    phase_signal = np.angle(analytic)
    spike_indices = (np.array(spike_times_sec) * fs).astype(int)
    valid = (spike_indices >= 0) & (spike_indices < len(phase_signal))
    return phase_signal[spike_indices[valid]]

def analyze_phase_locking(spike_times_sec, lfp_signal, fs):
    phases = get_lfp_phase_at_spikes(lfp_signal, spike_times_sec, fs)
    plv = np.abs(np.mean(np.exp(1j * phases)))
    mean_angle = circmean(phases, high=np.pi, low=-np.pi)
    std_angle = circstd(phases, high=np.pi, low=-np.pi)
    return phases, plv, mean_angle, std_angle

# Plot
#time_axis = np.arange(start_sample, end_sample) / fs
#plt.figure(figsize=(16, 5))
#plt.plot(time_axis, lfp_gamma, linewidth=0.5)
#plt.xlabel("Time (s)")
#plt.ylabel("Gamma-filtered LFP")
#plt.title("Gamma-filtered LFP from 160 to 660 seconds")
#plt.tight_layout()
#plt.show()

gamma_spikes_control = extract_gamma_rich_spikes(
    lfp_signal, stim_times_set1, spike_times_sec, fs, window_sec=10, sd_thresh=2, lfp_offset_sec=140
)

gamma_spikes_ach = extract_gamma_rich_spikes(
    lfp_signal, stim_times_set2, spike_times_sec, fs, window_sec=10, sd_thresh=2, lfp_offset_sec=140
)



print(f"Gamma spikes (Control): {len(gamma_spikes_control)}")
print(f"Gamma spikes (ACh):     {len(gamma_spikes_ach)}")

pct_control = 100 * len(gamma_spikes_control) / len(spikes_control)
pct_ach = 100 * len(gamma_spikes_ach) / len(spikes_ach)

print(f"Control: {pct_control:.1f}% of spikes during gamma")
print(f"ACh:     {pct_ach:.1f}% of spikes during gamma")


# Control
phases_control, plv_control, mean_control, std_control = analyze_phase_locking(
    spikes_control, lfp_signal, fs
)

# ACh
phases_ach, plv_ach, mean_ach, std_ach = analyze_phase_locking(
    spikes_ach, lfp_signal, fs
)

# Print results
print("\n=== Total Spike–LFP Phase Locking ===")
print(f"[Control] n={len(phases_control)} spikes — PLV: {plv_control:.3f}, Mean Phase: {mean_control:.2f} rad")
print(f"[ACh]     n={len(phases_ach)} spikes — PLV: {plv_ach:.3f}, Mean Phase: {mean_ach:.2f} rad")

fig, axs = plt.subplots(1, 2, subplot_kw=dict(polar=True), figsize=(10, 5))
axs[0].hist(phases_control, bins=30, density=True, color='blue', alpha=0.7)
axs[0].set_title("Control")

axs[1].hist(phases_ach, bins=30, density=True, color='red', alpha=0.7)
axs[1].set_title("ACh")

plt.suptitle("Total Spike–LFP Gamma Phase Locking")
plt.tight_layout()
#plt.show()

spike_clusters = np.load(r'C:\Users\James\Desktop\sliceAnalysis-master\spykingcircus_output04_09\sorter_output\recording\recording_sc.GUI\spike_clusters.npy')
# Function to mask spike times and clusters together
def filter_spikes_by_window(spike_times, spike_clusters, stim_times, window):
    """
    Returns spike times and corresponding cluster labels within stimulus windows.
    """
    stim_mask = np.zeros_like(spike_times, dtype=bool)
    for stim in stim_times:
        stim_mask |= (spike_times >= stim * fs) & (spike_times < (stim + window) * fs)
    return spike_times[stim_mask], spike_clusters[stim_mask]

# Filter spikes to include only those in control or ACh windows
filtered_spike_times_ctrl, filtered_clusters_ctrl = filter_spikes_by_window(
    spike_times, spike_clusters, stim_times_set1, post_stim_window
)
filtered_spike_times_ach, filtered_clusters_ach = filter_spikes_by_window(
    spike_times, spike_clusters, stim_times_set2, post_stim_window
)

# Get unique units from both sets
units_ctrl = np.unique(filtered_clusters_ctrl)
units_ach = np.unique(filtered_clusters_ach)
all_units = np.unique(np.concatenate((units_ctrl, units_ach)))

min_spikes = 20
valid_units = []

print("Checking spike counts...\n")

# Step 1: Filter valid units based on spike count
for unit in all_units:
    unit_spikes_ctrl = filtered_spike_times_ctrl[filtered_clusters_ctrl == unit]
    unit_spikes_ach = filtered_spike_times_ach[filtered_clusters_ach == unit]

    n_ctrl = len(unit_spikes_ctrl)
    n_ach = len(unit_spikes_ach)

    if n_ctrl >= min_spikes and n_ach >= min_spikes:
        valid_units.append(unit)
        print(f"Unit {unit}: OK (Ctrl: {n_ctrl}, ACh: {n_ach})")
    else:
        print(f"Unit {unit}: Skipped (Ctrl: {n_ctrl}, ACh: {n_ach})")

print(f"\nValid units (≥{min_spikes} spikes per condition): {len(valid_units)}\n")

# Step 2: PLV analysis with Rayleigh correction
results = []
print("Starting PLV and Rayleigh Z analysis...\n")

for i, unit in enumerate(valid_units):
    print(f"Processing unit {unit} ({i+1}/{len(valid_units)})...")

    unit_spikes_ctrl = filtered_spike_times_ctrl[filtered_clusters_ctrl == unit] / fs
    unit_spikes_ach = filtered_spike_times_ach[filtered_clusters_ach == unit] / fs

    n_ctrl = len(unit_spikes_ctrl)
    n_ach = len(unit_spikes_ach)

    # Analyze PLV
    _, plv_ctrl, _, _ = analyze_phase_locking(unit_spikes_ctrl, lfp_signal, fs)
    _, plv_ach, _, _ = analyze_phase_locking(unit_spikes_ach, lfp_signal, fs)

    # Rayleigh Z and p
    z_ctrl = n_ctrl * plv_ctrl**2
    z_ach = n_ach * plv_ach**2
    p_ctrl = math.exp(-z_ctrl)
    p_ach = math.exp(-z_ach)

    print(f"  Control PLV: {plv_ctrl:.3f} | Z: {z_ctrl:.2f}, p: {p_ctrl:.2e}")
    print(f"  ACh     PLV: {plv_ach:.3f} | Z: {z_ach:.2f}, p: {p_ach:.2e}")

    results.append((unit, plv_ctrl, z_ctrl, p_ctrl, plv_ach, z_ach, p_ach))

print("\nPLV analysis complete.")

# Step 3: Convert to structured array and print table
results_array = np.array(results, dtype=[
    ('unit', int),
    ('plv_control', float), ('z_control', float), ('p_control', float),
    ('plv_ach', float), ('z_ach', float), ('p_ach', float)
])

# Table output
print("\n=== Final PLV + Rayleigh Z Table (≥20 Spikes/Condition) ===")
print(f"{'Unit':<6} {'PLV_Ctrl':<10} {'Z_Ctrl':<10} {'p_Ctrl':<10} {'PLV_ACh':<10} {'Z_ACh':<10} {'p_ACh':<10}")
for r in results_array:
    print(f"{r['unit']:<6} {r['plv_control']:<10.3f} {r['z_control']:<10.2f} {r['p_control']:<10.2e} "
          f"{r['plv_ach']:<10.3f} {r['z_ach']:<10.2f} {r['p_ach']:<10.2e}")

# Build a DataFrame directly from results_array
df = pd.DataFrame({
    'Unit': results_array['unit'],
    'PLV_Ctrl': results_array['plv_control'],
    'Z_Ctrl': results_array['z_control'],
    'p_Ctrl': results_array['p_control'],
    'PLV_ACh': results_array['plv_ach'],
    'Z_ACh': results_array['z_ach'],
    'p_ACh': results_array['p_ach'],
})

# Save to CSV
csv_filename = 'plv_zscore_table.csv'
df.to_csv(csv_filename, index=False)
print(f"\nSaved table to: {csv_filename}")

units_of_interest = [1, 7, 16, 22, 24, 32, 43, 48, 50, 56, 58, 61, 72, 76, 78, 81, 85, 88, 91, 95]

unit_results = {}

for unit in units_of_interest:
    # Get spike times for this unit
    unit_mask = spike_clusters == unit
    unit_spike_times_sec = spike_times[unit_mask] / fs

    # Assign spikes to conditions
    spikes_ctrl = assign_spikes_to_condition(unit_spike_times_sec, stim_times_set1, post_stim_window)
    spikes_ach = assign_spikes_to_condition(unit_spike_times_sec, stim_times_set2, post_stim_window)

    # Analyze phase locking
    phases_ctrl, plv_ctrl, mean_ctrl, std_ctrl = analyze_phase_locking(spikes_ctrl, lfp_signal, fs)
    phases_ach, plv_ach, mean_ach, std_ach = analyze_phase_locking(spikes_ach, lfp_signal, fs)

    unit_results[unit] = {
        'control': {
            'phases': phases_ctrl,
            'plv': plv_ctrl,
            'mean_phase': mean_ctrl,
            'std_phase': std_ctrl,
            'n_spikes': len(phases_ctrl)
        },
        'ach': {
            'phases': phases_ach,
            'plv': plv_ach,
            'mean_phase': mean_ach,
            'std_phase': std_ach,
            'n_spikes': len(phases_ach)
        }
    }

for unit in units_of_interest:
    data = unit_results[unit]
    fig, axs = plt.subplots(1, 2, subplot_kw=dict(polar=True), figsize=(10, 4))

    axs[0].hist(data['control']['phases'], bins=30, density=True, color='blue', alpha=0.6)
    axs[0].set_title(f'Unit {unit} - Control\nPLV: {data["control"]["plv"]:.2f}')

    axs[1].hist(data['ach']['phases'], bins=30, density=True, color='red', alpha=0.6)
    axs[1].set_title(f'Unit {unit} - ACh\nPLV: {data["ach"]["plv"]:.2f}')

    plt.suptitle(f'Spike–LFP Gamma Phase Locking — Unit {unit}')
    plt.tight_layout()
    #plt.savefig(f"unit_{unit}_phase_locking.png", dpi=300)
    plt.close()

import numpy as np
import matplotlib.pyplot as plt

# Collect all raw phases without alignment
all_phases_ctrl = []
all_phases_ach = []

for unit in units_of_interest:
    data = unit_results[unit]
    all_phases_ctrl.append(data['control']['phases'])
    all_phases_ach.append(data['ach']['phases'])

# Flatten into one array per condition
all_phases_ctrl = np.concatenate(all_phases_ctrl)
all_phases_ach = np.concatenate(all_phases_ach)

# Compute PLV for unaligned phases
def compute_plv(phases):
    return np.abs(np.mean(np.exp(1j * phases)))

plv_ctrl = compute_plv(all_phases_ctrl)
plv_ach = compute_plv(all_phases_ach)

# Plotting
fig, axs = plt.subplots(1, 2, subplot_kw=dict(polar=True), figsize=(12, 5))

axs[0].hist(all_phases_ctrl, bins=30, density=True, color='blue', alpha=0.7)
axs[0].set_title(f'Population Control (unaligned)\nPLV: {plv_ctrl:.3f}')

axs[1].hist(all_phases_ach, bins=30, density=True, color='red', alpha=0.7)
axs[1].set_title(f'Population ACh (unaligned)\nPLV: {plv_ach:.3f}')

plt.suptitle('Unaligned Spike–LFP Gamma Phase Locking Across Units')
plt.tight_layout()
plt.savefig("population_phase_locking_unaligned.png", dpi=300)
plt.show()


#PLV MEAN/MODE PHASE ALIGNED CIRCULAR HISTOGRAM

from scipy.stats import circmean

# Helper function for circular subtraction and wrapping to [-pi, pi]
def align_phases(phases, mean_phase):
    aligned = phases - mean_phase
    aligned = (aligned + np.pi) % (2 * np.pi) - np.pi
    return aligned

# Collect all aligned phases
all_aligned_phases_ctrl = []
all_aligned_phases_ach = []

for unit in units_of_interest:
    data = unit_results[unit]

    # Align phases for control and ACh
    mean_phase_ctrl = data['control']['mean_phase']
    mean_phase_ach = data['ach']['mean_phase']

    aligned_ctrl = align_phases(data['control']['phases'], mean_phase_ctrl)
    aligned_ach = align_phases(data['ach']['phases'], mean_phase_ach)

    all_aligned_phases_ctrl.append(aligned_ctrl)
    all_aligned_phases_ach.append(aligned_ach)

# Concatenate all units
all_aligned_phases_ctrl = np.concatenate(all_aligned_phases_ctrl)
all_aligned_phases_ach = np.concatenate(all_aligned_phases_ach)

# Compute population PLV for aligned spikes
def compute_plv(phases):
    return np.abs(np.mean(np.exp(1j * phases)))

pop_plv_ctrl = compute_plv(all_aligned_phases_ctrl)
pop_plv_ach = compute_plv(all_aligned_phases_ach)

# Plot combined summary
fig, axs = plt.subplots(1, 2, subplot_kw=dict(polar=True), figsize=(12, 5))

axs[0].hist(all_aligned_phases_ctrl, bins=30, density=True, color='blue', alpha=0.7)
axs[0].set_title(f'Population Control\nPLV: {pop_plv_ctrl:.3f}')

axs[1].hist(all_aligned_phases_ach, bins=30, density=True, color='red', alpha=0.7)
axs[1].set_title(f'Population ACh\nPLV: {pop_plv_ach:.3f}')

plt.suptitle('Population Spike–LFP Gamma Phase Locking (Mean Phase Aligned)')
plt.tight_layout()
plt.show()

def circular_mode(phases, bins=60):
    """Estimate the circular mode using histogram peak."""
    counts, bin_edges = np.histogram(phases, bins=bins, range=(-np.pi, np.pi))
    max_bin_idx = np.argmax(counts)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers[max_bin_idx]

all_aligned_phases_ctrl = []
all_aligned_phases_ach = []

for unit in units_of_interest:
    data = unit_results[unit]

    # Use circular mode for alignment
    mode_phase_ctrl = circular_mode(data['control']['phases'])
    mode_phase_ach = circular_mode(data['ach']['phases'])

    aligned_ctrl = align_phases(data['control']['phases'], mode_phase_ctrl)
    aligned_ach = align_phases(data['ach']['phases'], mode_phase_ach)

    all_aligned_phases_ctrl.append(aligned_ctrl)
    all_aligned_phases_ach.append(aligned_ach)

# Concatenate all aligned phases across units
all_aligned_phases_ctrl = np.concatenate(all_aligned_phases_ctrl)
all_aligned_phases_ach = np.concatenate(all_aligned_phases_ach)

# Compute population PLV
def compute_plv(phases):
    return np.abs(np.mean(np.exp(1j * phases)))

pop_plv_ctrl = compute_plv(all_aligned_phases_ctrl)
pop_plv_ach = compute_plv(all_aligned_phases_ach)

# Plot
fig, axs = plt.subplots(1, 2, subplot_kw=dict(polar=True), figsize=(12, 5))

axs[0].hist(all_aligned_phases_ctrl, bins=30, density=True, color='blue', alpha=0.7)
axs[0].set_title(f'Population Control\nPLV: {pop_plv_ctrl:.3f}')

axs[1].hist(all_aligned_phases_ach, bins=30, density=True, color='red', alpha=0.7)
axs[1].set_title(f'Population ACh\nPLV: {pop_plv_ach:.3f}')

plt.suptitle('Population Spike–LFP Gamma Phase Locking (Mode-Aligned)')
plt.tight_layout()
plt.show()


#PPC STUFF=================================================================================
def compute_ppc(phases):
    """
    Computes pairwise phase consistency (PPC) from a vector of spike–LFP phases.
    Returns NaN if fewer than 2 spikes.
    """
    n = len(phases)
    if n < 2:
        return np.nan

    diffs = phases[:, None] - phases[None, :]  # NxN matrix
    upper_triangle = np.triu_indices(n, k=1)
    cos_diffs = np.cos(diffs[upper_triangle])
    return np.mean(cos_diffs)

for unit in units_of_interest:
    ctrl_phases = unit_results[unit]['control']['phases']
    ach_phases = unit_results[unit]['ach']['phases']

    unit_results[unit]['control']['ppc'] = compute_ppc(ctrl_phases)
    unit_results[unit]['ach']['ppc'] = compute_ppc(ach_phases)

ppc_table = pd.DataFrame([
    {
        'Unit': unit,
        'n_spikes_ctrl': unit_results[unit]['control']['n_spikes'],
        'PPC_ctrl': unit_results[unit]['control']['ppc'],
        'n_spikes_ach': unit_results[unit]['ach']['n_spikes'],
        'PPC_ach': unit_results[unit]['ach']['ppc'],
    }
    for unit in units_of_interest
])

#ppc_table.to_csv('ppc_unitwise_summary.csv', index=False)
#print("Saved PPC results to 'ppc_unitwise_summary.csv'")

labels = [str(unit) for unit in units_of_interest]
ppc_ctrl = [unit_results[u]['control']['ppc'] for u in units_of_interest]
ppc_ach  = [unit_results[u]['ach']['ppc'] for u in units_of_interest]

x = np.arange(len(units_of_interest))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, ppc_ctrl, width, label='Control', color='skyblue')
bars2 = ax.bar(x + width/2, ppc_ach, width, label='ACh', color='salmon')

ax.set_ylabel('PPC')
ax.set_title('PPC by Unit and Condition')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.legend()
plt.tight_layout()
#plt.savefig('ppc_comparison_barplot.png')
plt.show()