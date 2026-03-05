import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil, sqrt
import matplotlib.animation as animation
from tqdm import tqdm
from matplotlib.patches import Rectangle

# Paths to the .npy files
spike_times_path = r"C:\quetzalcoatl\workspace\James_q\kilosort\spike_times.npy"
spike_positions_path = r"C:\quetzalcoatl\workspace\James_q\kilosort\spike_positions.npy"  # Adjust the path as needed

# Define electrode positions
electrode_geometry = {
        0: [+0.0, +0.0],  # A4
        1: [+0.0, -200.0],
        2: [+0.0, -400.0],
        3: [+0.0, -600.0],
        4: [+0.0, -800.0],
        5: [+0.0, -1000.0],
        6: [+200.0, +200.0],  # B3
        7: [+200.0, +0.0],
        8: [+200.0, -200.0],
        9: [+200.0, -400.0],
        10: [+200.0, -600.0],
        11: [+200.0, -800.0],
        12: [+200.0, -1000.0],
        13: [+200.0, -1200.0],
        14: [+400.0, +400.0],  # C2
        15: [+400.0, +200.0],
        16: [+400.0, 0.0],
        17: [+400.0, -200.0],
        18: [+400.0, -400.0],
        19: [+400.0, -600.0],
        20: [+400.0, -800.0],
        21: [+400.0, -1000.0],
        22: [+400.0, -1200.0],
        23: [+400.0, -1400.0],
        24: [+600.0, +600.0],  # D1
        25: [+600.0, +400.0],
        26: [+600.0, +200.0],
        27: [+600.0, 0.0],
        28: [+600.0, -200.0],
        29: [+600.0, -400.0],
        30: [+600.0, -600.0],
        31: [+600.0, -800.0],
        32: [+600.0, -1000.0],
        33: [+600.0, -1200.0],
        34: [+600.0, -1400.0],
        35: [+600.0, -1600.0],
        36: [+800.0, +600.0],  # E1
        37: [+800.0, +400.0],
        38: [+800.0, +200.0],
        39: [+800.0, 0.0],
        40: [+800.0, -200.0],
        41: [+800.0, -400.0],
        42: [+800.0, -600.0],
        43: [+800.0, -800.0],
        44: [+800.0, -1000.0],
        45: [+800.0, -1200.0],
        46: [+800.0, -1400.0],
        47: [+800.0, -1600.0],
        48: [+1000.0, +600.0],  # F1
        49: [+1000.0, +400.0],
        50: [+1000.0, +200.0],
        51: [+1000.0, 0.0],
        52: [+1000.0, -200.0],
        53: [+1000.0, -400.0],
        54: [+1000.0, -600.0],
        55: [+1000.0, -800.0],
        56: [+1000.0, -1000.0],
        57: [+1000.0, -1200.0],
        58: [+1000.0, -1400.0],
        59: [+1000.0, -1600.0],
        60: [+1200.0, +600.0],  # G1
        61: [+1200.0, +400.0],
        62: [+1200.0, +200.0],
        63: [+1200.0, 0.0],
        64: [+1200.0, -200.0],
        65: [+1200.0, -400.0],
        66: [+1200.0, -600.0],
        67: [+1200.0, -800.0],
        68: [+1200.0, -1000.0],
        69: [+1200.0, -1200.0],
        70: [+1200.0, -1400.0],
        71: [+1200.0, -1600.0],
        72: [+1400.0, +600.0],  # H1
        73: [+1400.0, +400.0],
        74: [+1400.0, +200.0],
        75: [+1400.0, 0.0],
        76: [+1400.0, -200.0],
        77: [+1400.0, -400.0],
        78: [+1400.0, -600.0],
        79: [+1400.0, -800.0],
        80: [+1400.0, -1000.0],
        81: [+1400.0, -1200.0],
        82: [+1400.0, -1400.0],
        83: [+1400.0, -1600.0],
        84: [+1600.0, +600.0],  # J1
        85: [+1600.0, +400.0],
        86: [+1600.0, +200.0],
        87: [+1600.0, 0.0],
        88: [+1600.0, -200.0],
        89: [+1600.0, -400.0],
        90: [+1600.0, -600.0],
        91: [+1600.0, -800.0],
        92: [+1600.0, -1000.0],
        93: [+1600.0, -1200.0],
        94: [+1600.0, -1400.0],
        95: [+1600.0, -1600.0],
        96: [+1800.0, +400.0],  # K2
        97: [+1800.0, +200.0],
        98: [+1800.0, 0.0],
        99: [+1800.0, -200.0],
        100: [+1800.0, -400.0],
        101: [+1800.0, -600.0],
        102: [+1800.0, -800.0],
        103: [+1800.0, -1000.0],
        104: [+1800.0, -1200.0],
        105: [+1800.0, -1400.0],
        106: [+2000.0, +200.0],  # L3
        107: [+2000.0, 0.0],
        108: [+2000.0, -200.0],
        109: [+2000.0, -400.0],
        110: [+2000.0, -600.0],
        111: [+2000.0, -800.0],
        112: [+2000.0, -1000.0],
        113: [+2000.0, -1200.0],
        114: [+2200.0, 0.0],  # M4
        115: [+2200.0, -200.0],
        116: [+2200.0, -400.0],
        117: [+2200.0, -600.0],
        118: [+2200.0, -800.0],
        119: [+2200.0, -1000.0]
    }

# Load the data
spike_times = np.load(spike_times_path, allow_pickle=True)
# Define the sampling frequency (Hz)
sampling_frequency = 30000  # This is typically the value for your specific data

# Convert spike times to seconds
spike_times_seconds = spike_times / sampling_frequency

# Print the converted spike times for verification
print("First 10 spike times in seconds:", spike_times_seconds[:10])
print("Last 10 spike times in seconds:", spike_times_seconds[-10:])

spike_positions = np.load(spike_positions_path, allow_pickle=True)
spike_templates = np.load(r'C:\quetzalcoatl\workspace\James_q\kilosort\spike_templates.npy', allow_pickle=True)

# Get unique spike template values
unique_templates = np.unique(spike_templates)
print("Unique spike templates:", unique_templates)

# Check the loaded data
print("Spike times shape:", spike_times_seconds.shape)
print("Spike positions shape:", spike_positions.shape)

# Ensure both arrays have the same length
if spike_times.shape[0] != spike_positions.shape[0]:
    raise ValueError("The number of spike times and spike positions do not match.")

# Ensure spike_positions is a 2D array
if spike_positions.ndim != 2 or spike_positions.shape[1] != 2:
    raise ValueError("Spike positions should be a 2D array with shape (n, 2).")


# Extract x and y positions
x_positions, y_positions = zip(*electrode_geometry.values())

# Create DataFrame for electrode positions
electrode_positions = pd.DataFrame({
    'channel': list(electrode_geometry.keys()),
    'x': x_positions,
    'y': y_positions
})

# Define a color map
def create_color_map(num_colors):
    # Use a colormap from matplotlib
    cmap = plt.get_cmap('tab20', num_colors)  # Adjust colormap as needed
    return cmap

# Map unique spike templates to colors
num_unique_templates = len(unique_templates)
color_map = create_color_map(num_unique_templates)
template_to_color = {template: color_map(i) for i, template in enumerate(unique_templates)}

def plot_spikes_within_time_window(start_time, end_time):
    # Filter spikes based on the time window
    mask = (spike_times_seconds >= start_time) & (spike_times_seconds <= end_time)
    filtered_spike_positions = spike_positions[mask]
    filtered_spike_templates = spike_templates[mask]

    # Create a scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(x_positions, y_positions, c='black', label='Electrodes')

    # Plot each spike with color based on its template
    for template in unique_templates:
        template_mask = filtered_spike_templates == template
        plt.scatter(filtered_spike_positions[template_mask, 0],
                    filtered_spike_positions[template_mask, 1],
                    c=[template_to_color[template]],
                    label=f'Template {template}',
                    alpha=0.5)  # Set transparency for visibility

    # Add labels for the electrodes
    for index, row in electrode_positions.iterrows():
        plt.text(row['x'], row['y'], row['channel'], fontsize=9, ha='right')

    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Spikes from {start_time} to {end_time}')
    plt.show()

def plot_spikes_in_time_windows(window_size):
    num_windows = (spike_times_seconds.max() // window_size) + 1
    num_cols = 3  # Number of columns in the subplot grid
    num_rows = ceil(num_windows / num_cols)  # Calculate number of rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), constrained_layout=True)

    # Flatten axes array for easy iteration
    axes = axes.flatten()

    for i in range(num_windows):
        start_time = i * window_size
        end_time = start_time + window_size - 1

        # Filter spikes based on the time window
        mask = (spike_times_seconds >= start_time) & (spike_times_seconds <= end_time)
        filtered_spike_positions = spike_positions[mask]
        filtered_spike_templates = spike_templates[mask]

        # Plotting on the current subplot
        ax = axes[i]
        #ax.scatter(x_positions, y_positions, c='black', label='Electrodes')

        # Plot each spike with color based on its template
        for template in unique_templates:
            template_mask = filtered_spike_templates == template
            ax.scatter(filtered_spike_positions[template_mask, 0],
                       filtered_spike_positions[template_mask, 1],
                       c=[template_to_color[template]],
                       label=f'Template {template}',
                       alpha=0.5)  # Set transparency for visibility

        # Add labels for the electrodes
        for index, row in electrode_positions.iterrows():
            ax.text(row['x'], row['y'], row['channel'], fontsize=9, ha='right')

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'Spikes from {start_time} to {end_time}')
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')

    # Turn off unused axes
    for j in range(num_windows, len(axes)):
        fig.delaxes(axes[j])

    plt.show()

#plot spikes in a single time window
plot_spikes_within_time_window(0, 300001)

# Plot spikes in multiple equal time windows
window_size = 10
#plot_spikes_in_time_windows(window_size)

# Get unique spike templates
unique_templates = np.unique(spike_templates)


num_unique_templates = len(unique_templates)
color_map = create_color_map(num_unique_templates)
template_to_color = {template: color_map(i) for i, template in enumerate(unique_templates)}

# Extract x and y positions for electrodes
x_positions, y_positions = zip(*electrode_geometry.values())
electrode_positions = pd.DataFrame({
    'channel': list(electrode_geometry.keys()),
    'x': x_positions,
    'y': y_positions
})

# Parameters for real-time plotting
fade_duration = 1  # Duration for fade-in and fade-out in seconds
fade_frames = int(fade_duration * 15)  # Assuming 30 FPS

# Set up the progress bar
#num_frames = (spike_times_seconds.max() // window_size + 1) * frames_per_window
#progress_bar = tqdm(total=num_frames, desc='Creating Animation', unit='frame')

# Parameters
fps = 5  # Frames per second
time_window = 1  # Window size in seconds for visualization
interval = 1000 / fps  # milliseconds per frame
fade_duration = 0.5  # Duration for fade-in and fade-out in seconds
fade_frames = int(fade_duration * fps)
frame_interval = int(time_window * fps)

# Initialize plot
fig, (ax_main, ax_timebar) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [8, 1]})
fig.subplots_adjust(hspace=0.4)

# Main plot for spikes
ax_main.scatter(x_positions, y_positions, c='black', label='Electrodes')
scat_spikes = ax_main.scatter([], [], c=[], alpha=0, label='Spikes')  # Empty initially

# Time bar settings
ax_timebar.set_xlim(0, 1)
ax_timebar.set_ylim(0, 1)
ax_timebar.axis('off')
timebar = ax_timebar.add_patch(Rectangle((0, 0), 0, 1, color='blue', alpha=0.5))

def update(frame):
    ax_main.cla()  # Clear the axis instead of using ax_main.clear()

    # Add electrodes
    ax_main.scatter(x_positions, y_positions, c='black', label='Electrodes')

    # Calculate current time window
    current_time = frame * (time_window / frame_interval)
    start_time = current_time
    end_time = start_time + time_window

    # Print debug information
    print(f"Frame: {frame}, Start Time: {start_time}, End Time: {end_time}")

    # Filter spikes based on the time window
    mask = (spike_times_seconds >= start_time) & (spike_times_seconds <= end_time)
    filtered_spike_positions = spike_positions[mask]
    filtered_spike_templates = spike_templates[mask]

    # Check how many spikes are being filtered
    print(f"Number of spikes in current window: {len(filtered_spike_positions)}")

    # Calculate fade effect
    fade_in = np.clip(frame % fade_frames / fade_frames, 0, 1)
    fade_out = np.clip(1 - (frame - fade_frames) / fade_frames, 0, 1)

    # Plot each spike with color based on its template
    for template in unique_templates:
        template_mask = filtered_spike_templates == template
        alpha_value = fade_in if frame < fade_frames else fade_out
        ax_main.scatter(filtered_spike_positions[template_mask, 0],
                        filtered_spike_positions[template_mask, 1],
                        c=[template_to_color[template]],
                        label=f'Template {template}',
                        alpha=alpha_value)  # Set transparency for fading effect

    # Add labels for the electrodes
    for index, row in electrode_positions.iterrows():
        ax_main.text(row['x'], row['y'], row['channel'], fontsize=9, ha='right')

    ax_main.set_xlabel('X Position')
    ax_main.set_ylabel('Y Position')
    ax_main.set_title(f'Spike Activity')
    ax_main.grid(True)
    ax_main.set_aspect('equal', adjustable='box')

    # Update the time bar
    progress = (current_time - spike_times_seconds.min()) / (spike_times_seconds.max() - spike_times_seconds.min())
    timebar.set_width(progress)

# Create the animation
#num_frames = int((spike_times_seconds.max() - spike_times_seconds.min()) / (time_window / frame_interval) * fps)
#print(f"Total frames: {num_frames}")

#ani = animation.FuncAnimation(
#    fig, update, frames=num_frames, repeat=False, interval=interval, blit=False
#)

#plt.show()