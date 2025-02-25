import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
csv_file = '10materials.csv'

# Read the CSV file
df = pd.read_csv(csv_file, header=None)

# The first row contains materials, the second row contains variables
materials = df.iloc[0, :].values
variables = df.iloc[1, :].values

# Create a DataFrame with the correct structure
data = df.iloc[2:, :].reset_index(drop=True)
data.columns = pd.MultiIndex.from_arrays([materials, variables])

# Convert all data to numeric
data = data.apply(pd.to_numeric, errors='coerce')

# Get the unique materials
unique_materials = data.columns.levels[0]

# Define material groups
groups = {
    'Group 1': ['Anatase', 'Rutile', 'CeO2', 'Cu2O'],
    'Group 2': ['WO3', 'MoO3', 'ZrO2', 'Y2O3'],
    'Group 3': ['LiCoO2', 'LiFePO4']
}

# Create an empty DataFrame to store results
results = pd.DataFrame(columns=['Material', 'U', 'c1', 'c2', 'J2'])

# Function to find the family of solutions for a given material
def find_family_of_solutions(material):
    global results
    if material not in data.columns.levels[0]:
        raise ValueError(f"Material '{material}' not found in the dataset")

    material_data = data[material]

    # Group by U and find the c1, c2 that minimize J2
    grouped = material_data.groupby('U').apply(lambda df: df.loc[df['J2'].idxmin()])

    # Print out the family of solutions for each material
    print(f"Family of solutions for material {material}:")
    for idx, row in grouped.iterrows():
        print(f"Material={material}, U={row['U']}, c1={row['c1']}, c2={row['c2']}, J2={row['J2']}")

        # Create a DataFrame for the solution and append to results
        solution_df = pd.DataFrame({
            'Material': [material],
            'U': [row['U']],
            'c1': [row['c1']],
            'c2': [row['c2']],
            'J2': [row['J2']]
        })
        results = pd.concat([results, solution_df], ignore_index=True)

# Loop through all unique materials and find the family of solutions
for material in unique_materials:
    find_family_of_solutions(material)

# Save results to a text file
results.to_csv('family_of_solutions.txt', sep='\t', index=False)

print("Family of solutions has been saved to 'family_of_solutions.txt'")

# Plotting the family of solutions for each group
n_groups = len(groups)
fig, axs = plt.subplots(n_groups, 2, figsize=(16, 5 * n_groups), constrained_layout=False)

# Define colors for each material
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_materials)))

# Define material labels with LaTeX formatting
def format_label(material):
    if material == 'Anatase':
        return r'a-TiO$_2$'
    elif material == 'Rutile':
        return r'r-TiO$_2$'
    elif material == 'Cu2O':
        return r'Cu$_2$O'
    elif material == 'CeO2':
        return r'CeO$_2$'
    elif material == 'WO3':
        return r'WO$_3$'
    elif material == 'MoO3':
        return r'MoO$_3$'
    elif material == 'ZrO2':
        return r'ZrO$_2$'
    elif material == 'Y2O3':
        return r'Y$_2$O$_3$'
    elif material == 'LiCoO2':
        return r'LiCoO$_2$'
    elif material == 'LiFePO4':
        return r'LiFePO$_4$'
    else:
        return material

# Collect handles and labels for the legend
handles = []
labels = []

# Plot each group
for i, (group_name, materials_list) in enumerate(groups.items()):
    for material in materials_list:
        if material in unique_materials:
            material_data = results[results['Material'] == material]
            color = colors[unique_materials.tolist().index(material)]  # Get color for the material

            # Plot U vs c1
            sc1 = axs[i, 0].scatter(material_data['U'], material_data['c1'], color=color, label=format_label(material))
            axs[i, 0].set_xlabel('Hubbard $U$ Value (eV)')
            axs[i, 0].set_ylabel('Hubbard Projector $c_{1}$')
            #axs[i, 0].set_title(f'{group_name}: U vs $c_{1}$')

            # Plot U vs c2
            sc2 = axs[i, 1].scatter(material_data['U'], material_data['c2'], color=color, label=format_label(material))
            axs[i, 1].set_xlabel('Hubbard $U$ Value (eV)')
            axs[i, 1].set_ylabel('Hubbard Projector $c_{2}$')
            #axs[i, 1].set_title(f'{group_name}: U vs $c_{2}$')

            # Collect handles and labels for the legend
            if len(handles) == 0 or format_label(material) not in labels:  # Only add new handles and labels
                handles.append(sc1)
                labels.append(format_label(material))

# Add a single horizontal legend above the entire figure
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=len(labels)//2)

# Adjust layout
plt.show()

