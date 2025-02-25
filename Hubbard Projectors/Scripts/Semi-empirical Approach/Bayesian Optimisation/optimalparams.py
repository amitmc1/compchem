import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize, LinearSegmentedColormap

# Define the file path
file_path = r'C:\Users\amit_\OneDrive\PhD Python\DFTdevelopment\FinalBayesianTesting\Benchmarking\optimised_results_50000_nouncertainty.txt'

# Define column names based on your description
columns = ['iteration', 'U', 'c1', 'c2', 'BG', 'v0', '3d', '2p', 'Status1', 'Status2', 'Status3', 'J']

# Read the data into a DataFrame
df = pd.read_csv(file_path, sep=' ', header=None, names=columns, skipinitialspace=True)

# Filter the DataFrame based on specified conditions
filtered_df = df[
    (df['Status1'] == 'Stable') &
    (df['Status2'] == 'Nonzero') &
    (df['Status3'] == 'Sensible')
]

# Extract required columns for plotting
x = filtered_df['U']
y = filtered_df['c1']
z = filtered_df['c2']
colors = filtered_df['J']  # Color according to column 'J'

# Set the range of values for the color scale
vmin = colors.min()  # Minimum value of 'J' in the filtered data
vmax = 600  # Custom maximum value for color scale

# Create a normalization instance to scale data values to the range [0, 1]
norm = Normalize(vmin=vmin, vmax=vmax)

# Define custom colormap from red to orange to green
colors_custom = ['green', 'orange', 'red']
cmap_custom = LinearSegmentedColormap.from_list('custom_cmap', colors_custom, N=256)

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with color coding based on the 'J' column and custom color scale
scatter = ax.scatter(x, y, z, c=colors, cmap=cmap_custom, norm=norm)
colorbar = fig.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.05)  # Set pad to adjust the distance from the plot
colorbar.set_label(r'$J^{SP}$', fontsize=12)

# Add labels and title
ax.set_xlabel('Ti 3$d$ Hubbard $U$ Value (eV)')
ax.set_ylabel('Ti 3$d$ Hubbard Projector $c_{1}$')
ax.set_zlabel('Ti 3$d$ Hubbard Projector $c_{2}$')
#ax.set_title('3D Scatter Plot')

# Add color bar with custom limits and label
#cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', fraction=0.05, pad=0.05)
#cbar.set_label('J(SE)')
#ticks=(np.linspace(vmin, vmax, num=5))  # Customize color bar ticks
#cbar.set_ticks(np.round(ticks).astype(int))

# Show the plot
plt.show()