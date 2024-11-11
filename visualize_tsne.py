import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from train import AlignNet


'''
Can run the following code for just loading and printing centroids from model file
'''
# from train import AlignNet  # Replace with your actual module path

# # Load the checkpoint
# checkpoint_path = '/workspace/LAV-ASOT/LOGS2/CKPTS/STEPS/modellAV_epoch=214_step=15000.ckpt'
# model = AlignNet.load_from_checkpoint(checkpoint_path)

# # Print the centroids
# print("Cluster centroids:")
# print(model.clusters.data)

# # This gives [6,128] meaning 6 cluster centroids of size 128 (same as embedding size)
# print(model.clusters.data.shape)



# Load the centroids
n_clusters=5
checkpoint_path = '/workspace/LAV-ASOT/best_step_logs/exp4/modellAV_epoch=171_step=12000.ckpt'
model = AlignNet.load_from_checkpoint(checkpoint_path)
centroids = model.clusters.data
centroids = centroids.cpu().numpy()

# Load all the embeddings and labels
DEST_VAL = "/workspace/LAV-ASOT/best_step_logs/exp4/eval_step_12000/val_pouring_embs.npy"
val_embeddings = np.load(DEST_VAL, allow_pickle=True).tolist()

# Load the embeddings for 2 vids
VID_i1=0
VID_i2=1
embeddings_video1 = val_embeddings["embs"][VID_i1]
embeddings_video2 = val_embeddings["embs"][VID_i2]

labels_vid1 = val_embeddings["labels"][VID_i1]
labels_vid2 = val_embeddings["labels"][VID_i2]
labels_vid1 = np.array(labels_vid1)
labels_vid2 = np.array(labels_vid2)

'''
Assuming centroids, embeddings_video1 and embeddings_video2 are numpy arrays with shapes:
centroids: (5, 128)
embeddings_video1: (n_frames_vid1, 128)
embeddings_video2: (n_frames_vid2, 128)
Assuming labels_vid1 and labels_vid2 are arrays of integers with shapes:
(n_frames_vid1,)
(n_frames_vid2,)
'''

# Combine all data for t-SNE
combined_embeddings = np.concatenate([embeddings_video1, embeddings_video2, centroids], axis=0)

# Run t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(combined_embeddings)

# Extract t-SNE results for plotting
tsne_vid1 = tsne_results[:len(embeddings_video1)]
tsne_vid2 = tsne_results[len(embeddings_video1):len(embeddings_video1) + len(embeddings_video2)]
tsne_centroids = tsne_results[-n_clusters:]

# Define custom colors for each unique label
custom_colors = ['violet', 'indigo', 'dodgerblue', 'limegreen', 'darkorange']  # Customize these colors

# Plotting
plt.figure(figsize=(12, 8))

# Plot video 1 frames with unfilled circles and colors based on labels
unique_labels = np.unique(labels_vid1)
for label in unique_labels:
    idx = labels_vid1 == label
    plt.scatter(tsne_vid1[idx, 0], tsne_vid1[idx, 1], 
                label=f'Vid1 Label {label}', marker='o', s=20, 
                facecolors='none', edgecolor=custom_colors[label])

# Plot video 2 frames with filled markers '1' and colors based on labels
for label in unique_labels:
    idx = labels_vid2 == label
    plt.scatter(tsne_vid2[idx, 0], tsne_vid2[idx, 1], 
                label=f'Vid2 Label {label}', marker='x', s=20, 
                facecolors=custom_colors[label])

# # Plot centroids with a distinct color and larger marker
# plt.scatter(tsne_centroids[:, 0], tsne_centroids[:, 1], c='red', marker='*', s=100, label='Centroids')
# Plot centroids with colors matching the corresponding label colors
for i in range(n_clusters):
    plt.scatter(tsne_centroids[i, 0], tsne_centroids[i, 1], 
                color=custom_colors[i], marker='*', s=200, 
                label=f'Centroid {i+1}')


plt.title('t-SNE Visualization of Frame Embeddings and Cluster Centroids')
plt.legend()

# Save the figure
plt.savefig('tsne_visualization.png')  # Save the plot as a PNG file