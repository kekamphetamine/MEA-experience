import h5py
import matplotlib.pyplot as plt

file_path = "C:\quetzalcoatl\workspace\James_q\Experiments/20240604slice1Cont_merged_npconv/20240604slice1Cont_merged_npconvtimes.clusters.hdf5"

#For all electrodes

with h5py.File(file_path, 'r') as f:
    electrode_count = len(f['electrodes'])

    spike_data = []
    for i in range(1, electrode_count):  # Iterate through all electrodes
        times_key = f'/times_{i}'
        if times_key in f:
            spike_times = f[times_key][:].flatten()

    # Plotting raster plot with shared time axis
    plt.figure(figsize=(12, 8))
    for idx, (electrode_idx, spike_times) in enumerate(spike_data):
        plt.eventplot(spike_times, lineoffsets=electrode_idx, linelengths=0.5, label=f'Electrode {electrode_idx}')

    plt.title('Raster Plot of Spike Data')
    plt.xlabel('Time')
    plt.ylabel('Electrode')
    plt.tight_layout()
    plt.show()