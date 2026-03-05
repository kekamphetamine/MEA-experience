import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector, Cursor
from matplotlib.path import Path
from collections import defaultdict
import string

# Define electrode geometry
electrode_geometry = {
    'D12': (120.0, 0.0, 0),
    'E12': (160.0, 0.0, 1),
    'F12': (200.0, 0.0, 2),
    'G12': (240.0, 0.0, 3),
    'H12': (280.0, 0.0, 4),
    'J12': (320.0, 0.0, 5),
    'C11': (80.0, 40.0, 6),
    'D11': (120.0, 40.0, 7),
    'E11': (160.0, 40.0, 8),
    'F11': (200.0, 40.0, 9),
    'G11': (240.0, 40.0, 10),
    'H11': (280.0, 40.0, 11),
    'J11': (320.0, 40.0, 12),
    'K11': (360.0, 40.0, 13),
    'B10': (40.0, 80.0, 14),
    'C10': (80.0, 80.0, 15),
    'D10': (120.0, 80.0, 16),
    'E10': (160.0, 80.0, 17),
    'F10': (200.0, 80.0, 18),
    'G10': (240.0, 80.0, 19),
    'H10': (280.0, 80.0, 20),
    'J10': (320.0, 80.0, 21),
    'K10': (360.0, 80.0, 22),
    'L10': (400.0, 80.0, 23),
    'A9': (0.0, 120.0, 24),
    'B9': (40.0, 120.0, 25),
    'C9': (80.0, 120.0, 26),
    'D9': (120.0, 120.0, 27),
    'E9': (160.0, 120.0, 28),
    'F9': (200.0, 120.0, 29),
    'G9': (240.0, 120.0, 30),
    'H9': (280.0, 120.0, 31),
    'J9': (320.0, 120.0, 32),
    'K9': (360.0, 120.0, 33),
    'L9': (400.0, 120.0, 34),
    'M9': (440.0, 120.0, 35),
    'A8': (0.0, 160.0, 36),
    'B8': (40.0, 160.0, 37),
    'C8': (80.0, 160.0, 38),
    'D8': (120.0, 160.0, 39),
    'E8': (160.0, 160.0, 40),
    'F8': (200.0, 160.0, 41),
    'G8': (240.0, 160.0, 42),
    'H8': (280.0, 160.0, 43),
    'J8': (320.0, 160.0, 44),
    'K8': (360.0, 160.0, 45),
    'L8': (400.0, 160.0, 46),
    'M8': (440.0, 160.0, 47),
    'A7': (0.0, 200.0, 48),
    'B7': (40.0, 200.0, 49),
    'C7': (80.0, 200.0, 50),
    'D7': (120.0, 200.0, 51),
    'E7': (160.0, 200.0, 52),
    'F7': (200.0, 200.0, 53),
    'G7': (240.0, 200.0, 54),
    'H7': (280.0, 200.0, 55),
    'J7': (320.0, 200.0, 56),
    'K7': (360.0, 200.0, 57),
    'L7': (400.0, 200.0, 58),
    'M7': (440.0, 200.0, 59),
    'A6': (0.0, 240.0, 60),
    'B6': (40.0, 240.0, 61),
    'C6': (80.0, 240.0, 62),
    'D6': (120.0, 240.0, 63),
    'E6': (160.0, 240.0, 64),
    'F6': (200.0, 240.0, 65),
    'G6': (240.0, 240.0, 66),
    'H6': (280.0, 240.0, 67),
    'J6': (320.0, 240.0, 68),
    'K6': (360.0, 240.0, 69),
    'L6': (400.0, 240.0, 70),
    'M6': (440.0, 240.0, 71),
    'A5': (0.0, 280.0, 72),
    'B5': (40.0, 280.0, 73),
    'C5': (80.0, 280.0, 74),
    'D5': (120.0, 280.0, 75),
    'E5': (160.0, 280.0, 76),
    'F5': (200.0, 280.0, 77),
    'G5': (240.0, 280.0, 78),
    'H5': (280.0, 280.0, 79),
    'J5': (320.0, 280.0, 80),
    'K5': (360.0, 280.0, 81),
    'L5': (400.0, 280.0, 82),
    'M5': (440.0, 280.0, 83),
    'A4': (0.0, 320.0, 84),
    'B4': (40.0, 320.0, 85),
    'C4': (80.0, 320.0, 86),
    'D4': (120.0, 320.0, 87),
    'E4': (160.0, 320.0, 88),
    'F4': (200.0, 320.0, 89),
    'G4': (240.0, 320.0, 90),
    'H4': (280.0, 320.0, 91),
    'J4': (320.0, 320.0, 92),
    'K4': (360.0, 320.0, 93),
    'L4': (400.0, 320.0, 94),
    'M4': (440.0, 320.0, 95),
    'B3': (40.0, 360.0, 96),
    'C3': (80.0, 360.0, 97),
    'D3': (120.0, 360.0, 98),
    'E3': (160.0, 360.0, 99),
    'F3': (200.0, 360.0, 100),
    'G3': (240.0, 360.0, 101),
    'H3': (280.0, 360.0, 102),
    'J3': (320.0, 360.0, 103),
    'K3': (360.0, 360.0, 104),
    'L3': (400.0, 360.0, 105),
    'C2': (80.0, 400.0, 106),
    'D2': (120.0, 400.0, 107),
    'E2': (160.0, 400.0, 108),
    'F2': (200.0, 400.0, 109),
    'G2': (240.0, 400.0, 110),
    'H2': (280.0, 400.0, 111),
    'J2': (320.0, 400.0, 112),
    'K2': (360.0, 400.0, 113),
    'D1': (120.0, 440.0, 114),
    'E1': (160.0, 440.0, 115),
    'F1': (200.0, 440.0, 116),
    'G1': (240.0, 440.0, 117),
    'H1': (280.0, 440.0, 118),
    'J1': (320.0, 440.0, 119),
}

# Extract x and y positions
x_positions, y_positions = zip(*electrode_geometry.values())

# Create DataFrame for electrodes
electrode_positions = pd.DataFrame({
    'channel': list(electrode_geometry.keys()),
    'x': x_positions,
    'y': y_positions
})


class PolygonSelectorHandler:
    def __init__(self, ax, electrode_positions):
        self.ax = ax
        self.electrode_positions = electrode_positions
        self.poly = None
        self.selected_electrodes = []
        self.unit_electrodes = {}  # Dictionary to track unit assignments
        self.current_unit = None
        self.current_principal_electrode = None
        self.path = None

    def onselect(self, verts):
        if self.poly is not None:
            self.poly.remove()
        self.poly = plt.Polygon(verts, edgecolor='black', facecolor='none')
        self.ax.add_patch(self.poly)
        self.ax.figure.canvas.draw()

        # Convert polygon vertices to path
        self.path = Path(verts)

        # Check which electrodes are inside the polygon
        selected_electrodes_df = self.electrode_positions[
            self.path.contains_points(self.electrode_positions[['x', 'y']])]

        # Update selected electrodes
        self.selected_electrodes = selected_electrodes_df
        self.print_selected_electrodes()

    def print_selected_electrodes(self):
        if self.current_unit is None:
            return

        print(f"\nSelected Electrodes for Unit '{self.current_unit}':")
        for _, row in self.selected_electrodes.iterrows():
            print(f"('channel': {row['channel']}, 'x': {row['x']}, 'y': {row['y']}),")

        if self.current_unit:
            if self.current_unit not in self.unit_electrodes:
                self.unit_electrodes[self.current_unit] = []
            self.unit_electrodes[self.current_unit].extend(self.selected_electrodes['channel'].tolist())

        # Display unit assignments
        self.display_unit_assignments()

    def display_unit_assignments(self):
        print("\nCurrent Unit Assignments:")
        for unit_id, channels in self.unit_electrodes.items():
            print(f"Unit '{unit_id}': {channels}")


def create_plot():
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter_electrodes = ax.scatter(electrode_positions['x'], electrode_positions['y'], color='blue',
                                    label='Electrodes', s=50)

    # Add electrode labels
    for i, row in electrode_positions.iterrows():
        ax.text(row['x'], row['y'], row['channel'], fontsize=9, ha='right', va='bottom', color='black')

    # Customize the plot
    ax.set_title('Electrode Array')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')

    return fig, ax


def save_to_file(unit_assignments, filename, trial_condition, num_units):
    with open(filename, 'w') as f:
        # Write 'Info' section
        f.write("Info\n")
        f.write(f"Trial Condition: {trial_condition}\n")
        f.write(f"Number of Units: {num_units}\n\n")

        # Write 'Unit Electrodes' section
        f.write("Unit Electrodes\n")
        for unit_id, channels in unit_assignments.items():
            f.write(f"Unit '{unit_id}': {channels}\n")

def main():
    num_units = int(input("Enter the number of units: "))

    unit_assignments = {}
    principal_electrode_counter = defaultdict(int)  # To track how many units use the same principal electrode

    for _ in range(num_units):
        # Create and display plot
        fig, ax = create_plot()
        polygon_selector_handler = PolygonSelectorHandler(ax, electrode_positions)

        # Get principal electrode number from user
        while True:
            try:
                principal_electrode_number = int(input("Enter a number (0-120) for the principal electrode: "))
                if 0 <= principal_electrode_number <= 120:
                    principal_electrode_name = f"{principal_electrode_number}"
                    break
                else:
                    print("Number must be between 0 and 120. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")

        # Generate a unique unit name
        counter = principal_electrode_counter[principal_electrode_name]
        suffix = string.ascii_lowercase[counter]
        polygon_selector_handler.current_unit = f"{principal_electrode_name}{suffix}"
        principal_electrode_counter[principal_electrode_name] += 1

        polygon_selector = PolygonSelector(ax, polygon_selector_handler.onselect)

        print(f"\nSelect electrodes for Unit '{polygon_selector_handler.current_unit}'. Draw a polygon around the electrodes.")
        plt.title(f"Select electrodes for Unit '{polygon_selector_handler.current_unit}'")
        plt.show()

        unit_assignments[polygon_selector_handler.current_unit] = polygon_selector_handler.unit_electrodes.get(polygon_selector_handler.current_unit, [])

    print("\nAll units have been processed.")
    print("\nFinal Unit Assignments:")
    for unit_id, channels in unit_assignments.items():
        print(f"Unit '{unit_id}': {channels}")

    # Ask for trial condition and export confirmation
    trial_condition = input("Enter the trial condition (Cont, CCh, W1, W2, W3, Other): ").strip()
    export = input("Do you want to export the final unit assignments to a text file? (yes/no): ").strip().lower()
    if export in ['yes', 'y']:
        filename = input("Enter the filename (e.g., 'unit_assignments.txt'): ").strip()
        save_to_file(unit_assignments, filename, trial_condition, num_units)
        print(f"Unit assignments have been saved to '{filename}'.")

if __name__ == "__main__":
    main()