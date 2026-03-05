import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Path to your NPY file
file_path = (r'C:\Users\James\Desktop\Kilosort stuff\Kilosort Example\kilosort4\templates.npy')

# Load the NPY file
data = np.load(file_path, allow_pickle=True)

# Print the shape and data type
print(f'Shape: {data.shape}')
print(f'Data type: {data.dtype}')

# Slice the data to disregard the third dimension (keep the first 2D slice as an example)
data_sliced = data[:, :, 0]  # Selecting the first 2D slice

# Get the number of columns
num_columns = data_sliced.shape[1]

# Create a range for the x-axis (0 to num_columns-1)
x = np.arange(num_columns)

# Plot the data
plt.figure(figsize=(12, 6))  # Adjust the size as needed

# Plot each row
for i in range(data_sliced.shape[0]):  # Plot all rows
    plt.plot(x, data_sliced[i, :], label=f'Row {i+1}')

# Add labels and title
plt.xlabel('Column Index')
plt.ylabel('Value')
plt.title('Plot of All Rows')
plt.legend()  # Show the legend to identify each line
plt.grid(True)  # Add a grid for better readability

# Show the plot
plt.show()
