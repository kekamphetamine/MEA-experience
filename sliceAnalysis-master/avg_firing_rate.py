import pandas as pd
import numpy as np

# Load the CSV file with spike data (which already has IFR calculated)
spike_csv_path = r"C:\Users\James\Desktop\sliceAnalysis-master\spykingcircus_output04_09\04_09spike_timing_with_stimulus.csv"
df = pd.read_csv(spike_csv_path)

# Load cluster group information
clustergroup_path = r"C:\Users\James\Desktop\sliceAnalysis-master\spykingcircus_output04_09\sorter_output\recording\recording_sc.GUI\cluster_group.tsv"
cluster_df = pd.read_csv(clustergroup_path, sep='\t')

# Filter for 'good' units
target_units = cluster_df[cluster_df['group'] == 'good']['cluster_id'].tolist()

# Load the condition table (stimulus start times in P1 to P5)
condition_csv_path = r"C:\Users\James\Desktop\sliceAnalysis-master\spykingcircus_output04_09\exp_timeline.xlsx"
conditions_df = pd.read_excel(condition_csv_path)

# Initialize a list to store results
results = []

# Loop over each condition and stimulus
for index, row in conditions_df.iterrows():
    result_row = {"Condition": row["Condition"], "Stimulus": row["Stim"], "Odor": row["Odor"]}

    for unit in target_units:
        unit_spike_times = df[df["Unit"] == unit]["Spike_Time (s)"].values
        total_spikes = 0  # For summing spikes across P1–P5

        # Loop over the P1 to P5 columns
        for p in range(1, 6):
            start_time = row[f"P{p}"]
            end_time = start_time + 10
            spikes_in_window = (unit_spike_times >= start_time) & (unit_spike_times < end_time)
            num_spikes = np.sum(spikes_in_window)
            #result_row[f"IFR_P{p} (unit {unit})"] = num_spikes / 10  # IFR
            total_spikes += num_spikes

        # Add the total spike count across P1–P5
        result_row[f"Total_Spikes_P1toP5 (unit {unit})"] = total_spikes

    results.append(result_row)

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
output_csv_path = r"C:\Users\James\Desktop\sliceAnalysis-master\spykingcircus_output04_09\04_09spike_sum_P1toP5_per_unit.csv"
results_df.to_csv(output_csv_path, index=False)

print("Saved detailed CSV with separate IFR values for P1–P5 per unit.")


# Define bins
start_time = 0
end_time = 150
bin_size = 10
time_bins = np.arange(start_time, end_time + bin_size, bin_size)  # [0, 10, ..., 150]

# Initialize list for bin-wise spike counts
bin_results = []

# Use the same target units
for unit in target_units:
    unit_spike_times = df[df["Unit"] == unit]["Spike_Time (s)"].values
    unit_result = {"Unit": unit}

    for i in range(len(time_bins) - 1):
        bin_start = time_bins[i]
        bin_end = time_bins[i + 1]
        spikes_in_bin = (unit_spike_times >= bin_start) & (unit_spike_times < bin_end)
        num_spikes = np.sum(spikes_in_bin)
        unit_result[f"{int(bin_start)}–{int(bin_end)}s"] = num_spikes

    bin_results.append(unit_result)

# Save the 10s bin spike count data
bin_df = pd.DataFrame(bin_results)
bin_output_path = r"C:\Users\James\Desktop\sliceAnalysis-master\spykingcircus_output04_09\04_09spike_totals_by_10s_bin.csv"
bin_df.to_csv(bin_output_path, index=False)

print("Saved spike totals in 10-second bins per unit.")