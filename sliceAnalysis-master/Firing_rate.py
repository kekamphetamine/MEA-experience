import pandas as pd
import numpy as np

# Load the CSV file
csv_path = r"C:\Users\James\Desktop\sliceAnalysis-master\spykingcircus_output04_09\04_09spike_timing_with_stimulus.csv"  # Update with your actual file path
df = pd.read_csv(csv_path)

# Load cluster group information
clustergroup_path = r"C:\Users\James\Desktop\sliceAnalysis-master\spykingcircus_output04_09\sorter_output\recording\recording_sc.GUI\cluster_group.tsv"
cluster_df = pd.read_csv(clustergroup_path, sep='\t')

# Filter for 'good' units
target_units = cluster_df[cluster_df['group'] == 'good']['cluster_id'].tolist()

# Define the time range for whole second timepoints
start_time = 0
end_time = 2631
timepoints = np.arange(np.floor(start_time), np.ceil(end_time))  # Whole second timepoints

# Initialize a list to store the result data
result_data = []

# Loop over each timepoint and calculate IFR for each target unit
for timepoint in timepoints:
    # Create a row for the current timepoint
    row = {"time (s)": timepoint}

    # Calculate IFR for each target unit
    for unit in target_units:
        # Get the spike times for the given unit
        unit_spike_times = df[df["Unit"] == unit]["Spike_Time (s)"].values

        # Count the spikes that occur in the current timepoint (1-second window)
        spikes_in_window = (unit_spike_times >= timepoint) & (unit_spike_times < timepoint + 1)
        num_spikes = np.sum(spikes_in_window)

        # Calculate IFR (spikes per second)
        ifr_value = num_spikes  # Since we're using whole seconds, this is just the count of spikes
        row[f"IFR (unit {unit})"] = ifr_value

    # Append the row to the result data
    result_data.append(row)

# Convert the result data to a DataFrame
result_df = pd.DataFrame(result_data)

# Save the result to a new CSV
output_csv_path = r"C:\Users\James\Desktop\sliceAnalysis-master\spykingcircus_output04_09\04_09IFR_by_time_and_unit.csv"  # Update with your desired output path
result_df.to_csv(output_csv_path, index=False)

print("Updated CSV saved with IFR by time and unit!")