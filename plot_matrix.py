import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os


def plot_matrix(matrix, output_path, xlabel='Frame index (Video X)', ylabel='Frame index (Video Y)'):
    """
    Visualizes a single matrix with a color map and labeled axes.
    """

    plt.figure(figsize=(6, 4))
    # plt.figure(figsize=(36, 16))

    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["white", "#006600"])
    # cmap = 'viridis'
    
    # Plot the matrix
    plt.imshow(matrix, aspect='auto', cmap=cmap)
    
    # Move x-axis to the top
    plt.gca().xaxis.tick_top()

    # Set ticks at intervals of 2 for both axes
    plt.xticks(np.arange(0, matrix.shape[1], 2))  # X-axis tick positions
    plt.yticks(np.arange(0, matrix.shape[0], 2))  # Y-axis tick positions

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # ### Add text values in each cell
    # for i in range(matrix.shape[0]):
    #     for j in range(matrix.shape[1]):
    #         value = matrix[i, j]
    #         if value>0:
    #             # Display the value at the correct position
    #             plt.text(j, i, f'{value:.2f}', ha='center', va='center', color='white', fontsize=8)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# for batch_idx in [9, 22, 33]:
for batch_idx in range(69):
    # mattype="Ck"
    mattype="T"
    # mattype="afternnz"
    matrix = np.load(f"(LAV){mattype}_npyfiles/{mattype}_{batch_idx}.npy")
    matrix = matrix.squeeze()


    filtered_matrix = np.zeros_like(matrix)  # Initialize with zeros (purple)

    ### Keep only max values of each row
    # max_indices = matrix.argmax(axis=1)  # Get max index for each row
    # for i, j in enumerate(max_indices):
    #     filtered_matrix[i, j] = matrix[i, j]  # Retain only max values
    
    ### Keep only max values of each col
    max_indices = matrix.argmax(axis=0)  # Get max index for each col
    for i, x in enumerate(max_indices):
        # filtered_matrix[x, i] = matrix[x, i]  # Retain only max values
        filtered_matrix[x, i] = 1  # Retain only max values and set the max value to 1 so matrix only has 2 colors


    os.makedirs(f"{mattype}_matrix_plots", exist_ok=True)
    path = f"{mattype}_matrix_plots/{batch_idx}_{mattype}_plot.png"
    plot_matrix(matrix=filtered_matrix, output_path=path)