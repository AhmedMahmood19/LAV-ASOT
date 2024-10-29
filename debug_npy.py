import numpy as np

'''
Code to check val embeddings for debugging

val_activity_embs.npy and it's train version are exactly the same files
it's a dict with keys ['embs', 'names', 'labels', 'frame_paths']
all of these are lists of len=num of val vids
embs is a list of nparrays, shape of embs[i]=(numframes of vid i, 128)
names is a list of strings, names[i]=filename of vid i
labels is a list of lists of ints, len of labels[i]=numframes of vid i
frame_paths is a list of lists of strings, len of frame_paths[i]=numframes of vid i
'''

def check_embs():
    DEST_VAL="/workspace/video-alignment/LAV-ASOT/log/eval_step_9000/val_kallax_shelf_drawer_embs.npy"
    val_embeddings = np.load(DEST_VAL, allow_pickle=True).tolist()
    val_embs, val_labels, val_names, val_frame_paths = val_embeddings['embs'], val_embeddings['labels'], val_embeddings['names'], val_embeddings['frame_paths']

    # Check the number of vids used and if they all have the same length
    print(len(val_embs), len(val_names), len(val_labels), len(val_frame_paths))

    # Check which videos were used
    print(val_names)

    # Fix for an old issue
    for i in range(len(val_names)):
        if len(val_embs[i]) != len(val_frame_paths[i]):
            # make the no. of framepaths same as the no. of embs 
            end = min(len(val_embs[i]), len(val_frame_paths[i]))
            val_embs[i]=val_embs[i][:end]
            val_frame_paths[i]=val_frame_paths[i][:end]
            print(f"\nVideo {val_names[i]} had mismatched lengths of framepaths and embeddings, so it was fixed")