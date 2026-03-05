from spikeinterface.extractors import read_mcsh5
from pathlib import Path
import numpy as np
from kilosort import io
import matplotlib.pyplot as plt
import h5py


# Specify the path where the data will be copied to, and where Kilosort 4
# results will be saved.
DATA_DIRECTORY = Path(r'T:\James_q\Kilosort')
# Create path if it doesn't exist
DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

# Specify path to your existing data
filepath = Path(r'T:\James_q\Experiments\ACh\20250321m1slice2\2025-03-21MergeMcsRecording.h5')   # NOTE: You must change this
# Load existing data with spikeinterface

#       `read_nwb_recording`, such as `electrical_series_name`. Any required
#       arguments should be clearly spelled out by an error message.
recording = read_mcsh5(filepath, stream_id=2)

# NOTE: Data will be saved as np.int16 by default since that is the standard
#       for ephys data. If you need a different data type for whatever reason
#       such as `np.uint16`, be sure to update this.
dtype = np.int32
filename, N, c, s, fs, probe_path = io.spikeinterface_to_binary(
    recording, DATA_DIRECTORY, data_name='0321slice1ACh.bin', dtype=dtype,
    chunksize=60000, export_probe=False)
