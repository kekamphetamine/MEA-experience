import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load spike data
spike_clusters = np.load(
    r"C:\Users\James\Desktop\sliceAnalysis-master\spykingcircus_output\sorter_output\recording\recording_sc.GUI\spike_clusters.npy")
spike_times = np.load(
    r"C:\Users\James\Desktop\sliceAnalysis-master\spykingcircus_output\sorter_output\recording\recording_sc.GUI\spike_times.npy")

# Define sampling rate (Hz)
sampling_rate = 10000  # 10 kHz

# Convert spike times to seconds
spike_times_sec = spike_times / sampling_rate

# Load cluster group information
clustergroup_path = r"C:\Users\James\Desktop\sliceAnalysis-master\spykingcircus_output\sorter_output\recording\recording_sc.GUI\cluster_group.tsv"
cluster_df = pd.read_csv(clustergroup_path, sep='\t')

# Filter for 'good' units
target_units = cluster_df[cluster_df['group'] == 'good']['cluster_id'].tolist()

print("Target units labeled 'good':", target_units)


# Define stimulus sequence (HS set)
stim_labels_HS = ["N", "C", "N", "S1", "N", "S2", "N", "S3", "N", "D"]
stim_durations = 10  # Each stimulus lasts 10s

# Stimulus set occurrences (start times) for "HS"
stim_start_times_HS = [160, 260, 360, 460, 560, 1495, 1595, 1695, 1795, 1895]

# New "HC" stimulus set
stim_labels_HC = ["N", "C", "N", "S1", "N", "S2", "N", "S3", "N", "D"]
stim_start_times_HC = [727, 827, 927, 1027, 1127, 2102, 2202, 2302, 2402, 2502]

# Combine both sets
stim_labels = stim_labels_HS + stim_labels_HC
stim_start_times = stim_start_times_HS + stim_start_times_HC

# Define stimulus colors
stim_colors = {"C": "red", "S1": "blue", "S2": "green", "S3": "purple", "D": "orange", "N": "gray"}

# Prepare to store results
spike_data = []

# Process each target unit
for unit in target_units:
    unit_spike_times = spike_times_sec[spike_clusters == unit]

    for spike_time in unit_spike_times:
        # Find which stimulus was active at this spike time
        stimulus_label = None
        stimulus_set = None
        for set_start in stim_start_times_HS:
            for i, label in enumerate(stim_labels_HS):
                stim_start_time = set_start + i * stim_durations
                stim_end_time = stim_start_time + stim_durations

                if stim_start_time <= spike_time < stim_end_time:
                    stimulus_label = label
                    stimulus_set = "HS"
                    break
            if stimulus_label:
                break

        if not stimulus_label:
            for set_start in stim_start_times_HC:
                for i, label in enumerate(stim_labels_HC):
                    stim_start_time = set_start + i * stim_durations
                    stim_end_time = stim_start_time + stim_durations

                    if stim_start_time <= spike_time < stim_end_time:
                        stimulus_label = label
                        stimulus_set = "HC"
                        break
                if stimulus_label:
                    break

        # Combine the set name and stimulus label
        combined_label = f"{stimulus_set}_{stimulus_label}"

        # Store (unit, spike_time, stimulus_label)
        spike_data.append([unit, spike_time, combined_label])

# Convert to DataFrame
spike_df = pd.DataFrame(spike_data, columns=["Unit", "Spike_Time (s)", "Stimulus"])

# Save to CSV
csv_path = r"C:\Users\James\Desktop\sliceAnalysis-master\spykingcircus_output\10_28spike_timing_with_stimulus.csv"
spike_df.to_csv(csv_path, index=False)

print(f"Saved spike timing data with stimulus information to: {csv_path}")
