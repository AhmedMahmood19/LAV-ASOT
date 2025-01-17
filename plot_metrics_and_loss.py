import pandas as pd
import matplotlib.pyplot as plt

########################################
# SET THE MODE BEFORE RUNNING THE CODE #
########################################
mode = "val"
########################################

loss_csv_path = "./LOGS/lightning_logs/version_0/metrics.csv"
if mode=="val":
    # Metrics of val set:
    metrics_csv_path = "./log_val/evaluation_results.csv"
elif mode=="train":
    # Metrics of train set:
    metrics_csv_path = "./log_train/evaluation_results.csv"

# Read the CSV files into DataFrames
metrics_df = pd.read_csv(metrics_csv_path)
loss_df = pd.read_csv(loss_csv_path)

# Extract steps and train_loss
train_steps = loss_df['step']
train_loss = loss_df['train_loss']

# Extract the step column and metrics from the metrics CSV
val_steps = metrics_df['step']
metrics = metrics_df.columns[1:]  # All columns except 'step'

# Create a dual y-axes plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot metrics on the primary y-axis
for metric in metrics:
    line, = ax1.plot(val_steps, metrics_df[metric], label=metric, alpha=0.7, linestyle='--')  # Plot metric and get line object
    
    # Extract the color of the current line
    line_color = line.get_color()

    # Find max value and corresponding step
    max_val = metrics_df[metric].max()
    max_step = val_steps[metrics_df[metric].idxmax()]

    # Add a marker and annotation
    ax1.scatter(max_step, max_val, color=line_color, s=10, zorder=5)  # Highlight max point with the same color as the line
    
    # Annotate max value with the same color as the line
    ax1.text(
        max_step + 500,  # Move the text 500 units to the right of the point (adjust as needed)
        max_val,         # Keep the y-position the same as the max value
        f"{max_val:.2f}",
        fontsize=8,
        va='center',
        color=line_color,  # Use the same color as the line
        bbox=dict(
            facecolor='white',  # Background color (white)
            alpha=0.5,          # Transparency (0 = fully transparent, 1 = fully opaque)
            edgecolor='none',   # No border around the box
            boxstyle='round,pad=0.2'  # Round corners with padding
        )
    )

ax1.set_xlabel("Step")
ax1.set_ylabel("Metric axis")
ax1.set_yticks([i / 10 for i in range(11)])  # Y-ticks from 0 to 1 in increments of 0.1
ax1.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

ax1.legend(loc='upper center', ncol=8)

# Create a secondary y-axis for train_loss
ax2 = ax1.twinx()
ax2.plot(train_steps, train_loss, label='Train Loss', color='red')
ax2.set_ylabel("Train Loss axis")
ax2.set_ylim(0, 400)
ax2.legend(loc='lower left')

# Add a title
plt.title(f"Metrics and Train Loss Over {train_steps.values[-1]} Training Steps")

# Save the figure
if mode=="val":
    # Path for val set:
    output_path = "plot_of_metrics_and_loss-VALSET.png"
elif mode=="train":
    # Path for train set:
    output_path = "plot_of_metrics_and_loss-TRAINSET.png"

plt.tight_layout()  # Adjust layout
plt.savefig(output_path)
plt.close()

print(f"Graph saved as {output_path}")