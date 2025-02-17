import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # Use Matplotlib for loading images
from matplotlib.patches import Rectangle
import random

def load_ap_logs(log_dir):
    """Load and extract AP@5 values from all log files."""
    ap_data = []
    
    for filename in os.listdir(log_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(log_dir, filename)
            with open(filepath, "r") as f:
                log = json.load(f)
                for query_path, details in log.get("AP@5", {}).items():
                    ap_data.append({
                        "query_path": query_path,
                        "ap5": details["AP@5"],
                        "neighbors": details["neighbours"],
                        "neighbor_labels": details["neighbours labels"],
                        "query_label": details["query frame label"]
                    })
    
    return ap_data  # Return all queries, no sorting or slicing


def load_image(image_path):
    """Load an image using Matplotlib, return a blank white image if not found."""
    if os.path.exists(image_path):
        return mpimg.imread(image_path)
    else:
        return np.ones((128, 128, 3))  # White placeholder if image not found


def visualize_high_low_queries(logs_high_dir, logs_low_dir, N=5, output_path="combined_queries.png"):
    """ 
    Visualizes N query frames with two rows of neighbors: 
    - Top row from AP-LogsHigh (highest AP@5)
    - Bottom row from AP-LogsLow (lowest AP@5)
    """
    
    # Load logs from both folders
    queries_high = load_ap_logs(logs_high_dir)
    queries_low = load_ap_logs(logs_low_dir)
    
    # Convert to dictionaries for quick lookup by query path
    high_dict = {q["query_path"]: q for q in queries_high}
    low_dict = {q["query_path"]: q for q in queries_low}
    
    # Find common queries in both logs
    common_queries = list(set(high_dict.keys()) & set(low_dict.keys()))
    
####
    # Sort by highest AP@5 in High and lowest AP@5 in Low
    # sorted_queries = sorted(
    #     common_queries, 
    #     key=lambda q: (high_dict[q]["ap5"], -low_dict[q]["ap5"]), 
    #     reverse=True  # Sort descending for High, ascending for Low
    # )[:N]
####

####
    # Filter queries where AP@5 is 1 in High and 0 in Low
    valid_queries = [
        q for q in common_queries
        if high_dict[q]["ap5"] == 1.0 and low_dict[q]["ap5"] == 0.0
    ]

    # Randomly select N queries (ensure we don't exceed available data)
    sorted_queries = random.sample(valid_queries, min(N, len(valid_queries)))
####

    # Plot each query with two rows of neighbors
    fig, axes = plt.subplots(N * 2, 6, figsize=(12, 2 * N * 2))  # Double rows for each query
    


    for row_idx, query_path in enumerate(sorted_queries):
        query_data_high = high_dict[query_path]
        query_data_low = low_dict[query_path]

        # Load images
        query_img = load_image(query_path)
        neighbors_high = [load_image(p) for p in query_data_high["neighbors"]]
        neighbors_low = [load_image(p) for p in query_data_low["neighbors"]]
        
        # Query Frame (1st Column)
        axes[row_idx * 2, 0].imshow(query_img)
        axes[row_idx * 2, 0].set_title(query_path, fontsize=5, wrap=True)
        axes[row_idx * 2, 0].axis("off")
        axes[row_idx * 2 + 1, 0].axis("off")  # Empty space in bottom row for query
        
        # Neighbors from High (Top Row)
        for col_idx, neighbor_img in enumerate(neighbors_high):
            axes[row_idx * 2, col_idx + 1].imshow(neighbor_img)
            axes[row_idx * 2, col_idx + 1].axis("off")

            # Check if label matches query label
            correct = query_data_high["neighbor_labels"][col_idx] == query_data_high["query_label"]
            symbol = "✔" if correct else "✖"
            color = "green" if correct else "red"

            # Add tick/cross annotation
            axes[row_idx * 2, col_idx + 1].text(5, 10, symbol, fontsize=12, color=color, weight="bold")

        
        # Neighbors from Low (Bottom Row)
        for col_idx, neighbor_img in enumerate(neighbors_low):
            axes[row_idx * 2 + 1, col_idx + 1].imshow(neighbor_img)
            axes[row_idx * 2 + 1, col_idx + 1].axis("off")

            # Check if label matches query label
            correct = query_data_low["neighbor_labels"][col_idx] == query_data_low["query_label"]
            symbol = "✔" if correct else "✖"
            color = "green" if correct else "red"

            # Add tick/cross annotation
            axes[row_idx * 2 + 1, col_idx + 1].text(5, 10, symbol, fontsize=12, color=color, weight="bold")


    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

def visualize_single_combined_query(query_path, logs_high, logs_low, output_path="single_query.png"):
    """ 
    Visualizes a single query frame with two rows of neighbors:
    - Top row from logs_high (High AP@5)
    - Bottom row from logs_low (Low AP@5)
    """
    
    # Load logs
    queries_high = load_ap_logs(logs_high)
    queries_low = load_ap_logs(logs_low)

    # Convert to dictionaries for lookup
    high_dict = {q["query_path"]: q for q in queries_high}
    low_dict = {q["query_path"]: q for q in queries_low}

    # Ensure the query exists in both logs
    if query_path not in high_dict or query_path not in low_dict:
        print(f"Query {query_path} not found in the logs!")
        return

    query_data_high = high_dict[query_path]
    query_data_low = low_dict[query_path]

    # Load images
    query_img = load_image(query_path)
    neighbors_high = [load_image(p) for p in query_data_high["neighbors"]]
    neighbors_low = [load_image(p) for p in query_data_low["neighbors"]]

    # Create figure (2 rows: High on top, Low on bottom)
    # fig, axes = plt.subplots(2, 6, figsize=(12, 4))
    fig, axes = plt.subplots(2, 6, figsize=(12, 4), gridspec_kw={'hspace': 0.1, 'wspace': 0.02})

    # Query frame (1st column)
    axes[0, 0].imshow(query_img, extent=[0, 1, 0, 1], aspect='auto')
    axes[0, 0].set_title(query_path, fontsize=5, wrap=True)
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")  # Empty space in bottom row for query

    # Neighbors from High (Top Row)
    for col_idx, neighbor_img in enumerate(neighbors_high):
        axes[0, col_idx + 1].imshow(neighbor_img, extent=[0, 1, 0, 1], aspect='auto')
        axes[0, col_idx + 1].axis("off")

    # Neighbors from Low (Bottom Row)
    for col_idx, neighbor_img in enumerate(neighbors_low):
        axes[1, col_idx + 1].imshow(neighbor_img, extent=[0, 1, 0, 1], aspect='auto')
        axes[1, col_idx + 1].axis("off")

    #### BORDERS
    # Define border coordinates
    top_border_x = axes[0, 1].get_position().x0  # Leftmost neighbor (top row)
    bottom_border_x = axes[1, 1].get_position().x0  # Leftmost neighbor (bottom row)

    top_border_width = axes[0, 5].get_position().x1 - top_border_x  # Covers all 5 neighbors
    bottom_border_width = axes[1, 5].get_position().x1 - bottom_border_x  # Covers all 5 neighbors

    top_border_y = axes[0, 1].get_position().y0  # Bottom Y of the top row
    bottom_border_y = axes[1, 1].get_position().y0  # Bottom Y of the bottom row

    top_border_height = axes[0, 1].get_position().y1 - top_border_y  # Height of the top row
    bottom_border_height = axes[1, 1].get_position().y1 - bottom_border_y  # Height of the bottom row

    # Create a blue border around the top row's 5 neighbors
    fig.patches.append(Rectangle((top_border_x, top_border_y), top_border_width, top_border_height, 
                                transform=fig.transFigure, edgecolor="blue", linewidth=2, facecolor="none"))

    # Create a red border around the bottom row's 5 neighbors
    fig.patches.append(Rectangle((bottom_border_x, bottom_border_y), bottom_border_width, bottom_border_height, 
                                transform=fig.transFigure, edgecolor="red", linewidth=2, facecolor="none"))
    ####

    ####Text Labels
    # Define label properties
    label_props = {
        "fontsize": 8,
        "fontweight": "bold",
        "color": "black",
        "verticalalignment": "center",
        "horizontalalignment": "left"
    }

    # Add rectangle and label to the first neighbor of the top row
    axes[0, 1].add_patch(Rectangle((0, 0), 0.25, 0.08, transform=axes[0, 1].transAxes, 
                                color="white", lw=0))
    axes[0, 1].text(0.02, 0.035, "VAOT", transform=axes[0, 1].transAxes, **label_props)

    # Add rectangle and label to the first neighbor of the bottom row
    axes[1, 1].add_patch(Rectangle((0, 0), 0.18, 0.08, transform=axes[1, 1].transAxes, 
                                color="white", lw=0))
    axes[1, 1].text(0.02, 0.035, "LAV", transform=axes[1, 1].transAxes, **label_props)

    # Add rectangle and label to the Query frame
    axes[0, 0].add_patch(Rectangle((0, 0), 0.28, 0.08, transform=axes[0, 0].transAxes, 
                                color="white", lw=0))
    axes[0, 0].text(0.02, 0.035, "Query", transform=axes[0, 0].transAxes, **label_props)
    ####

    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def visualize_single_query(query_path, log_dir):
    """Visualize only one query frame and its 5 neighbors based on the provided query path."""
    all_queries = load_ap_logs(log_dir)
    
    # Find the matching query data
    query_data = next((q for q in all_queries if q["query_path"] == query_path), None)
    
    if not query_data:
        print(f"Query frame '{query_path}' not found in logs.")
        return
    
    # Set up a single-row plot
    fig, axes = plt.subplots(1, 6, figsize=(12, 2))  # 1 row, 6 images
    
    # Load query and neighbor images
    query_img = load_image(query_data["query_path"])
    neighbor_imgs = [load_image(p) for p in query_data["neighbors"]]
    neighbor_labels = query_data["neighbor_labels"]
    query_label = query_data["query_label"]
    
    # Display query frame
    axes[0].imshow(query_img)
    axes[0].set_title(os.path.basename(query_data["query_path"]), fontsize=8, wrap=True)
    axes[0].axis("off")

    # Display neighbor frames with tick or cross
    for col_idx, (neighbor_img, neighbor_label) in enumerate(zip(neighbor_imgs, neighbor_labels)):
        axes[col_idx + 1].imshow(neighbor_img)
        axes[col_idx + 1].axis("off")

        # Add tick or cross above the image
        match_symbol = "✓" if neighbor_label == query_label else "✗"
        axes[col_idx + 1].set_title(match_symbol, fontsize=12, color="green" if match_symbol == "✓" else "red")

    plt.tight_layout()
    plt.savefig("single-query.png", dpi=300)

if __name__ == "__main__":
    # query_frame_path = "Data_Test/Test/baseball_swing/0168/000001.jpg"  # Replace with actual query path
    # visualize_single_query(query_frame_path, log_dir)

    # visualize_high_low_queries("AP-LogsHigh", "AP-LogsLow", N=10)

    query_frame_path = "./Data_Test/Test/baseball_swing/0187/000005.jpg"
    visualize_single_combined_query(query_frame_path, "AP-LogsHigh", "AP-LogsLow", output_path="combined_single_query.png")