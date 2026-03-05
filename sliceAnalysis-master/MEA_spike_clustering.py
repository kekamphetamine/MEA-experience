import pandas as pd
import numpy as np
from matplotlib.widgets import PolygonSelector

electrode_geometry = {
    'D12': (120.0, 0.0),
    'E12': (160.0, 0.0),
    'F12': (200.0, 0.0),
    'G12': (240.0, 0.0),
    'H12': (280.0, 0.0),
    'J12': (320.0, 0.0),
    'C11': (80.0, 40.0),
    'D11': (120.0, 40.0),
    'E11': (160.0, 40.0),
    'F11': (200.0, 40.0),
    'G11': (240.0, 40.0),
    'H11': (280.0, 40.0),
    'J11': (320.0, 40.0),
    'K11': (360.0, 40.0),
    'B10': (40.0, 80.0),
    'C10': (80.0, 80.0),
    'D10': (120.0, 80.0),
    'E10': (160.0, 80.0),
    'F10': (200.0, 80.0),
    'G10': (240.0, 80.0),
    'H10': (280.0, 80.0),
    'J10': (320.0, 80.0),
    'K10': (360.0, 80.0),
    'L10': (400.0, 80.0),
    'A9': (0.0, 120.0),
    'B9': (40.0, 120.0),
    'C9': (80.0, 120.0),
    'D9': (120.0, 120.0),
    'E9': (160.0, 120.0),
    'F9': (200.0, 120.0),
    'G9': (240.0, 120.0),
    'H9': (280.0, 120.0),
    'J9': (320.0, 120.0),
    'K9': (360.0, 120.0),
    'L9': (400.0, 120.0),
    'M9': (440.0, 120.0),
    'A8': (0.0, 160.0),
    'B8': (40.0, 160.0),
    'C8': (80.0, 160.0),
    'D8': (120.0, 160.0),
    'E8': (160.0, 160.0),
    'F8': (200.0, 160.0),
    'G8': (240.0, 160.0),
    'H8': (280.0, 160.0),
    'J8': (320.0, 160.0),
    'K8': (360.0, 160.0),
    'L8': (400.0, 160.0),
    'M8': (440.0, 160.0),
    'A7': (0.0, 200.0),
    'B7': (40.0, 200.0),
    'C7': (80.0, 200.0),
    'D7': (120.0, 200.0),
    'E7': (160.0, 200.0),
    'F7': (200.0, 200.0),
    'G7': (240.0, 200.0),
    'H7': (280.0, 200.0),
    'J7': (320.0, 200.0),
    'K7': (360.0, 200.0),
    'L7': (400.0, 200.0),
    'M7': (440.0, 200.0),
    'A6': (0.0, 240.0),
    'B6': (40.0, 240.0),
    'C6': (80.0, 240.0),
    'D6': (120.0, 240.0),
    'E6': (160.0, 240.0),
    'F6': (200.0, 240.0),
    'G6': (240.0, 240.0),
    'H6': (280.0, 240.0),
    'J6': (320.0, 240.0),
    'K6': (360.0, 240.0),
    'L6': (400.0, 240.0),
    'M6': (440.0, 240.0),
    'A5': (0.0, 280.0),
    'B5': (40.0, 280.0),
    'C5': (80.0, 280.0),
    'D5': (120.0, 280.0),
    'E5': (160.0, 280.0),
    'F5': (200.0, 280.0),
    'G5': (240.0, 280.0),
    'H5': (280.0, 280.0),
    'J5': (320.0, 280.0),
    'K5': (360.0, 280.0),
    'L5': (400.0, 280.0),
    'M5': (440.0, 280.0),
    'A4': (0.0, 320.0),
    'B4': (40.0, 320.0),
    'C4': (80.0, 320.0),
    'D4': (120.0, 320.0),
    'E4': (160.0, 320.0),
    'F4': (200.0, 320.0),
    'G4': (240.0, 320.0),
    'H4': (280.0, 320.0),
    'J4': (320.0, 320.0),
    'K4': (360.0, 320.0),
    'L4': (400.0, 320.0),
    'M4': (440.0, 320.0),
    'B3': (40.0, 360.0),
    'C3': (80.0, 360.0),
    'D3': (120.0, 360.0),
    'E3': (160.0, 360.0),
    'F3': (200.0, 360.0),
    'G3': (240.0, 360.0),
    'H3': (280.0, 360.0),
    'J3': (320.0, 360.0),
    'K3': (360.0, 360.0),
    'L3': (400.0, 360.0),
    'C2': (80.0, 400.0),
    'D2': (120.0, 400.0),
    'E2': (160.0, 400.0),
    'F2': (200.0, 400.0),
    'G2': (240.0, 400.0),
    'H2': (280.0, 400.0),
    'J2': (320.0, 400.0),
    'K2': (360.0, 400.0),
    'D1': (120.0, 440.0),
    'E1': (160.0, 440.0),
    'F1': (200.0, 440.0),
    'G1': (240.0, 440.0),
    'H1': (280.0, 440.0),
    'J1': (320.0, 440.0)
}


# Extract x and y positions
x_positions, y_positions = zip(*electrode_geometry.values())

# Create DataFrame
electrode_positions = pd.DataFrame({
    'channel': list(electrode_geometry.keys()),
    'x': x_positions,
    'y': y_positions
})

# Load spike positions from the specified path
file_path = r'C:\quetzalcoatl\workspace\James_q\kilosort\spike_positions.npy'
spike_positions = np.load(file_path)
spike_df = pd.DataFrame(spike_positions, columns=['x', 'y'])

import matplotlib.pyplot as plt

# Create the plot
fig, ax = plt.subplots(figsize=(12, 10))
scatter_electrodes = ax.scatter(electrode_positions['x'], electrode_positions['y'], color='blue', label='Electrodes', s=50)
#scatter_spikes = ax.scatter(spike_df['x'], spike_df['y'], color='red', label='Spikes', s=20, alpha=0.6)

# Add electrode labels
for i, row in electrode_positions.iterrows():
    ax.text(row['x'], row['y'], row['channel'], fontsize=9, ha='right', va='bottom', color='black')

# Customize the plot
ax.set_title('Electrode Array with Spike Locations')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.legend()
ax.grid(True)
ax.set_aspect('equal', adjustable='box')

# Polygon selection handler
class PolygonSelectorHandler:
    def __init__(self, ax, electrode_positions):
        self.ax = ax
        self.electrode_positions = electrode_positions
        self.poly = None
        self.xys = None
        self.selected_electrodes = []

    def onselect(self, verts):
        if self.poly is not None:
            self.poly.remove()
        self.poly = plt.Polygon(verts, edgecolor='black', facecolor='none')
        self.ax.add_patch(self.poly)
        self.ax.figure.canvas.draw()

        # Convert polygon vertices to path
        from matplotlib.path import Path
        path = Path(verts)

        # Check which electrodes are inside the polygon
        self.selected_electrodes = self.electrode_positions[path.contains_points(self.electrode_positions[['x', 'y']])]

        # Print selected electrodes
        print("Selected Electrodes:")
        for _, row in self.selected_electrodes.iterrows():
            print(f"('channel': {row['channel']}, 'x': {row['x']}, 'y': {row['y']}),")

# Initialize the polygon selector
#polygon_selector = PolygonSelector(ax, PolygonSelectorHandler(ax, electrode_positions).onselect)

# Show the plot
plt.show()