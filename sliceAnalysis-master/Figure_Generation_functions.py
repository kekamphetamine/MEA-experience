import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

def plot_spike_raster_with_kde(X, time_range=(0, 30), bw_adjust=0.1, seed=42):
    """
    Plots a raster plot of spike times and a Kernel Density Estimate (KDE) of the spike timings.

    Parameters:
    - X: int, the number of neurons
    - time_range: tuple, (min_time, max_time), the time range for spike times (default: (0, 30))
    - bw_adjust: float, bandwidth adjustment for KDE (default: 0.1)
    - seed: int, random seed for reproducibility (default: 42)
    """
    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Generate random spike times for each neuron within the specified time range
    spike_times = [np.sort(np.random.uniform(time_range[0], time_range[1], size=np.random.randint(5, 30))) for _ in range(X)]

    # Flatten the list of spike times for KDE
    all_spikes = np.concatenate(spike_times)

    # Plotting
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # 1. Raster Plot
    for i, neuron_spikes in enumerate(spike_times):
        ax[0].vlines(neuron_spikes, i + 0.5, i + 1.5, color='black')  # Spike times for each neuron

    ax[0].set_ylabel('Neuron Index')
    ax[0].set_title('Raster Plot')
    ax[0].set_ylim(0, X + 1)
    ax[0].set_xlim(time_range)

    # 2. Kernel Density Estimate (KDE) Plot
    # Using seaborn to plot the KDE with specified bandwidth adjustment
    sns.kdeplot(all_spikes, ax=ax[1], shade=True, color="blue", bw_adjust=bw_adjust)  # Narrowing the KDE

    # Alternatively, use scipy's gaussian_kde for more control over bandwidth
    kde = gaussian_kde(all_spikes, bw_method=bw_adjust)  # Smaller bw_method narrows the KDE
    x = np.linspace(time_range[0], time_range[1], 1000)

    # Customize plot
    ax[1].set_ylabel('Density')
    ax[1].set_xlabel('Time (seconds)')
    ax[1].set_title('Spike Timing KDE')

    plt.tight_layout()
    plt.show()

# Example usage of the function
plot_spike_raster_with_kde(X=50, time_range=(0, 30), bw_adjust=0.1)

