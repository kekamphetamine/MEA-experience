import numpy as np
import pandas as pd
from pathlib import Path

# -------- paths --------
base_path = Path(
    r"C:\Users\James\Desktop\Interim\Assist\20260130m2slice2\2026-01-30Merge_npconv\2026-01-30Merge_npconvsc_.GUI"
)

spike_clusters_path = base_path / "spike_clusters.npy"
spike_times_path = base_path / "spike_times.npy"

cluster_group_path = base_path / "cluster_group.tsv"
cluster_info_path = base_path / "cluster_info.tsv"

# -------- load spike data --------
spike_clusters = np.load(spike_clusters_path)
spike_times = np.load(spike_times_path).squeeze()

# -------- load cluster labels --------
if cluster_group_path.exists():
    cluster_df = pd.read_csv(cluster_group_path, sep="\t")
    good_clusters = cluster_df.loc[
        cluster_df["group"] == "good", "cluster_id"
    ].values

elif cluster_info_path.exists():
    cluster_df = pd.read_csv(cluster_info_path, sep="\t")
    good_clusters = cluster_df.loc[
        cluster_df["group"] == "good", "cluster_id"
    ].values

else:
    raise FileNotFoundError(
        "No cluster_group.tsv or cluster_info.tsv found. "
        "Cannot identify noise/MUA."
    )

print("Total spikes:", spike_times.shape[0])
print("Unique clusters in spikes:", len(np.unique(spike_clusters)))
print("Cluster label columns:", cluster_df.columns)
print(cluster_df.head())

# -------- filter spikes --------
mask = np.isin(spike_clusters, good_clusters)

filtered_clusters = spike_clusters[mask]
filtered_times = spike_times[mask]

# -------- create raster dataframe --------
raster_df = pd.DataFrame({
    "unit_id": filtered_clusters,
    "spike_time_samples": filtered_times
})

# Optional: sort nicely
raster_df = raster_df.sort_values(
    by=["unit_id", "spike_time_samples"]
)

# -------- save CSV --------
out_path = base_path / "raster_good_units.csv"
raster_df.to_csv(out_path, index=False)

print(f"Saved raster CSV with {raster_df.unit_id.nunique()} units")
print(f"Output: {out_path}")

import matplotlib.pyplot as plt
import numpy as np

# -------- convert to seconds --------
fs = 10000  # sampling rate (Hz)
times_sec = filtered_times / fs

# -------- prepare raster rows --------
units = np.unique(filtered_clusters)
unit_to_row = {unit: i for i, unit in enumerate(units)}
y_vals = np.array([unit_to_row[u] for u in filtered_clusters])

# -------- plot --------
plt.figure(figsize=(12, 6))

plt.scatter(
    times_sec,
    y_vals,
    s=2,
    marker="|"
)

plt.yticks(
    ticks=np.arange(len(units)),
    labels=units
)

plt.xlabel("Time (s)")
plt.ylabel("Unit ID")
plt.title("Spike Raster (All Unlabeled Units)")

plt.tight_layout()
plt.show()
