import matplotlib.pyplot as plt
import h5py
import numpy as np

filepath = r'T:\James_q\Experiments\ACh\20250409m1slice4\Merge_2025-04-09T06-51-36McsRecording.h5'
#example - r'T:\James_q\Experiments\ACh\20250321m1slice2\2025-03-21T13-10-19McsRecording.h5'
sampling_rate = 10000  # Set your actual sample rate

###INSPECT H5 FILE

import h5py


# Load the HDF5 file
with h5py.File(filepath, "r") as f:
    def print_structure(name, obj):
        print(name)  # Print the dataset/group structure
    f.visititems(print_structure)

'''
###ELECTRODE DATA CAN BE FOUND HERE
with h5py.File(filepath, "r") as f:
    # Navigate to the dataset
    channel_data = f["Data/Recording_0/AnalogStream/Stream_2/ChannelData"]

    # Convert to a NumPy array (could be large, so slice if needed)
    data = np.array(channel_data)

    # Print shape: (num_channels, num_samples)
    print("Shape of ChannelData:", data.shape)

    # Select a single channel to plot (e.g., channel 0)
    channel_idx = 0
    signal = data[channel_idx, :]

    # Plot the voltage trace
    plt.figure(figsize=(10, 4))
    plt.plot(signal[:10000])  # Plot first 10,000 samples
    plt.xlabel("Sample Index")
    plt.ylabel("Voltage (uV)")
    plt.title(f"Raw Voltage Trace - Channel {channel_idx}")
    plt.show()

'''

import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.extractors as ex
import probeinterface as pi


# Load the .prb file (likely returns a ProbeGroup)
probe_group = pi.io.read_prb(r'T:\James_q\the_best_probe.prb')
probe = probe_group.probes[0]


###PREPROCESSING
recording = ex.read_mcsh5(filepath, stream_id=2)

recording = recording.set_probe(probe)

#recording_filtered = spre.bandpass_filter(recording, freq_min=300, freq_max=3000)

sorter_name = "spykingcircus"
output_folder = "spykingcircus_output04_09"


# Run sorter
sorting = ss.run_sorter(
    sorter_name,
    recording,
    output_folder=output_folder,
    remove_existing_folder=True,  # Overwrites if folder exists
    verbose=True,
    with_output=True
)

# Print results
print(f"Sorting completed! Found {len(sorting.get_unit_ids())} units.")

# Save sorted spikes
sorting.save("sorted_spikes")
