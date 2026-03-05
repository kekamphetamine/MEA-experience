import numpy as np
import sys
from tqdm import tqdm
from numpy.lib.format import open_memmap
import h5py
from pathlib import Path

def convert_mergedh5_to_np(in_filename, out_filename, chunks=1e9):
    """
    Converts an H5 file to a NumPy file for Spyking Circus.
    Automatically detects the correct stream with 120 channels.
    """
    with h5py.File(in_filename, 'r') as f:
        # Auto-detect correct stream with 120 channels
        base_path = "Data/Recording_0/AnalogStream"
        correct_stream = None
        for stream_name in f[base_path]:
            stream = f[f"{base_path}/{stream_name}/ChannelData"]
            if stream.shape[0] == 120:
                correct_stream = stream
                break

        if correct_stream is None:
            raise ValueError("No stream with 120 channels found.")

        num_electrodes = correct_stream.shape[0]
        num_samples = correct_stream.shape[1]

        itemsize = np.array([0.0], dtype=np.float32).nbytes
        n_items = int(chunks // itemsize)
        total_n = num_electrodes * num_samples

        pbar = tqdm(total=total_n * itemsize, file=sys.stdout, unit_scale=True, unit='bytes')

        mmap_array = open_memmap(out_filename, mode='w+', dtype=np.float32, shape=(num_samples, num_electrodes))

        for k in range(num_electrodes):
            signal = correct_stream[k, :]
            i = 0
            while i * n_items < num_samples:
                items = np.array(signal[i * n_items:min((i + 1) * n_items, num_samples)], dtype=np.float32)
                mmap_array[i * n_items:i * n_items + len(items), k] = items
                pbar.update(len(items) * itemsize)
                i += 1

        pbar.close()

# Specify path to your existing data
filepath_i = Path(r"C:\Users\James\Desktop\Interim\Assist\20260130m2slice2\Merge_2026-01-30T17-02-21McsRecording.h5")
# filepath_i = Path(r'T:\James_q\Experiments\ACh\20250321m1slice2\2025-03-21MergeMcsRecording.h5')
filepath_o = Path(r"C:\Users\James\Desktop\Interim\Assist\20260130m2slice2\2026-01-30Merge_npconv.npy")
# filepath_o = Path(r'T:\James_q\Experiments\ACh\20250321m1slice2/2025-03-21Merge_npconv.npy')

convert_mergedh5_to_np(filepath_i, filepath_o)