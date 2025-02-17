import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Set before running
batch_idx = 30  # Change this based on which file you want to visualize
output_path = f"{batch_idx}_tsne.png"
# model = "(LAV)"
model = "(VAOT)"
perp=6

# Load features Shape (20,128)
features_X = np.load(f'{model}features_npyfiles/features_X_{batch_idx}.npy').squeeze(0)
features_Y = np.load(f'{model}features_npyfiles/features_Y_{batch_idx}.npy').squeeze(0) 

# Apply t-SNE (reducing 128D to 2D) on combined data
tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
combined_features = np.vstack((features_X, features_Y))
combined_2d = tsne.fit_transform(combined_features)

# Split transformed features back
vidembs_2d_X = combined_2d[:len(features_X)]
vidembs_2d_Y = combined_2d[len(features_X):]

# Plot the embeddings
fig, ax = plt.subplots(figsize=(5, 5))

# Plot features_X with autumn colormap
sc_X = ax.scatter(vidembs_2d_X[:, 0], vidembs_2d_X[:, 1], 
                  c=range(len(features_X)), cmap='Greens', alpha=0.8, label='X', vmax=25, vmin=-5)


# Plot features_Y with winter colormap
sc_Y = ax.scatter(vidembs_2d_Y[:, 0], vidembs_2d_Y[:, 1], 
                  c=range(len(features_Y)), cmap='Reds', alpha=0.8, label='Y', vmax=25, vmin=-5)

# Remove x and y axes
ax.set_xticks([])
ax.set_yticks([])

# Save the plot
plt.savefig(f"{batch_idx}-perp{perp}-{model}-test.png")
