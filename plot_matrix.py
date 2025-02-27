import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os


def plot_matrix(matrix, output_path, xlabel='Frame index (Video X)', ylabel='Frame index (Video Y)'):
    """
    Visualizes a single matrix with a color map and labeled axes.
    """

    # plt.figure(figsize=(6, 4))
    plt.figure(figsize=(10, 10))

    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["white", "#006600"])
    
    # Plot the matrix
    plt.pcolormesh(matrix, cmap=cmap)
    
    # Move x-axis to the top
    plt.gca().xaxis.tick_top()

    # Hide ticks
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    # Set ticks at intervals of 2 for both axes
    # plt.xticks(np.arange(0, matrix.shape[1], 2))  # X-axis tick positions
    # plt.yticks(np.arange(0, matrix.shape[0], 2))  # Y-axis tick positions

    plt.gca().xaxis.set_label_position('top')
    plt.xlabel(xlabel, fontsize=22)
    plt.ylabel(ylabel, fontsize=22)

    # ### Add text values in each cell
    # for i in range(matrix.shape[0]):
    #     for j in range(matrix.shape[1]):
    #         value = matrix[i, j]
    #         if value>0:
    #             # Display the value at the correct position
    #             plt.text(j, i, f'{value:.2f}', ha='center', va='center', color='white', fontsize=8)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()

for batch_idx in range(69):
    # model = "(VAOT)"
    model = "(LAV)"
    
    ## When you create the folder using store_for_viz.py, make sure to rename it with the model at the start
    matrix = np.load(f"{model}npy_T-matrix/T_{batch_idx}.npy")
    matrix = matrix.squeeze()


    filtered_matrix = np.zeros_like(matrix)  # Initialize with zeros

    ### Keep only max values of each row
    # max_indices = matrix.argmax(axis=1)  # Get max index for each row
    # for i, j in enumerate(max_indices):
    #     filtered_matrix[i, j] = matrix[i, j]  # Retain only max values
    
    ### Keep only max values of each col
    max_indices = matrix.argmax(axis=0)  # Get max index for each col
    for i, x in enumerate(max_indices):
        # filtered_matrix[x, i] = matrix[x, i]  # Retain only max values
        filtered_matrix[x, i] = 1  # Retain only max values and set the max value to 1 so matrix only has 2 colors


    os.makedirs(f"{model}T_matrix_plots", exist_ok=True)
    path = f"{model}T_matrix_plots/{batch_idx}_T_plot.pdf"
    plot_matrix(matrix=filtered_matrix, output_path=path)