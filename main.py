"""
Author: Paul Chung
Last Update: Mar 4, 2025 5:58PM
"""

import matplotlib.pyplot as plt
import MEAutility as MEA
import numpy as np
import pandas as pd
import pyyaml as yaml
import scipy as sp

data = np.load("20260130m1slice2/spike_times.npy")
print(pd.DataFrame(data))

print(f"Shape: {data.shape}")
print(f"Data type: {data.dtype}")
