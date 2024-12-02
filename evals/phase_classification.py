import numpy as np
import os, json
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, confusion_matrix


def fit_svm(train_embs, train_labels):
    train_embs = np.concatenate(train_embs)
    train_labels = np.concatenate(train_labels)

    svm_model = SVC(decision_function_shape='ovo')
    svm_model.fit(train_embs, train_labels)
    train_acc = svm_model.score(train_embs, train_labels)

    return svm_model, train_acc


def evaluate_svm(svm, val_embs, val_labels):

    val_preds = []
    for vid_embs in val_embs:
        vid_preds = svm.predict(vid_embs)
        val_preds.append(vid_preds)

    # concatenate labels and preds in one array
    val_preds = np.concatenate(val_preds)
    val_labels = np.concatenate(val_labels)

    # calculate accuracy and confusion matrix
    val_acc = accuracy_score(val_labels, val_preds)
    conf_mat = confusion_matrix(val_labels, val_preds)

    return val_acc, conf_mat


def evaluate_phase_classification(ckpt_step, train_embs, train_labels, val_embs, val_labels, act_name, CONFIG,  writer=None, verbose=False, csv_dict=None):

    for frac in CONFIG.EVAL.CLASSIFICATION_FRACTIONS:
        N_Vids = max(1, int(len(train_embs) * frac))
        embs = train_embs[:N_Vids]
        labs = train_labels[:N_Vids]

        if verbose:
            print(f'Fraction = {frac}, Total = {len(train_embs)}, Used = {len(embs)}')

        svm_model, train_acc = fit_svm(embs, labs)
        val_acc, conf_mat = evaluate_svm(svm_model, val_embs, val_labels)

        if verbose:
            print('\n-----------------------------')
            print('Fraction: ', frac)
            print('Train-Acc: ', train_acc)
            print('Val-Acc: ', val_acc)
            print('Conf-Mat: ', conf_mat)


        writer.add_scalar(f'classification/train_{act_name}_{frac}', train_acc, global_step=ckpt_step)
        writer.add_scalar(f'classification/val_{act_name}_{frac}', val_acc, global_step=ckpt_step)
        
        # print(f'classification/train_{act_name}_{frac}', train_acc, f"global_step={ckpt_step}")
        csv_dict[f"PC ({frac})"]=val_acc
        print(f'IMPORTANT!!!         classification/val_{act_name}_{frac}', val_acc, f"global_step={ckpt_step}")

    return train_acc, val_acc


def _compute_ap(query_frames, support_frames, query_labels, support_labels, query_frame_paths=None, support_frame_paths=None, log=False):
    # Initialize AP sums for k = 5, 10, 15
    query_AP_sums = {5: 0, 10: 0, 15: 0}

    frame_logs = {f'AP@{k}': {} for k in [5, 10, 15]} if log else None
    
    for k in [5, 10, 15]:

        # Train the NN model on the embeddings from the support set
        NN = NearestNeighbors(n_neighbors=k).fit(support_frames)
        # Find the k nearest neighbours of the embeddings from the query set
        dists, indices = NN.kneighbors(query_frames)

        # Calculate frame retrieval accuracy for each query frame i from the query set
        for i in range(query_frames.shape[0]):
            # true label of frame i, repeated k times
            a = np.array([query_labels[i]] * k)
            # using the indices of the k nearest frames of frame i, find the labels of these k nearest frames
            b = support_labels[indices[i]]
            # calculate the ratio of the retrieved frames with the same label as the query frame i 
            val = (a==b).sum()/k
            # Accumulate AP scores
            query_AP_sums[k] += val


            if log:
                frame_logs[f'AP@{k}'][query_frame_paths[i]] = {
                    'neighbours': support_frame_paths[indices[i]].tolist(),
                    'neighbours labels': b.tolist(),
                    'query frame label': query_labels[i],
                    f'AP@{k}': val
                }


    # Return the AP sums for the query set
    return query_AP_sums, frame_logs

def compute_ap(embeddings, labels, names, frame_paths, log=False):
    # Initialize AP sums for k = 5, 10, 15
    all_AP_sums = {5: 0, 10: 0, 15: 0}
    num_vids = len(embeddings)

    if log:
        os.makedirs("AP-Logs", exist_ok=True)

    # For each video, use its frames as the query set while treating frames from all other videos as the support set
    for i in range(num_vids):

        # Create query and support arrays
        query_frames = embeddings[i]  # Frame embeddings for current video
        query_labels = np.array(labels[i])  # Frame labels for current video
        query_frame_paths = np.array(frame_paths[i]) if log else None

        support_frames = np.vstack([embeddings[j] for j in range(num_vids) if j != i])  # Frame embeddings for all other videos
        support_labels = np.hstack([labels[j] for j in range(num_vids) if j != i])  # Frame labels for all other videos
        support_frame_paths = np.hstack([frame_paths[j] for j in range(num_vids) if j != i]) if log else None


        # Get AP scores for the query set
        query_AP_sums, frame_logs = _compute_ap(query_frames, support_frames, query_labels, support_labels, query_frame_paths, support_frame_paths, log)

        # Accumulate AP scores
        for k in [5, 10, 15]:
            all_AP_sums[k] += query_AP_sums[k]


        if log:
            with open(os.path.join("AP-Logs", f"framelogs-{names[i]}.json"), 'w') as json_file:
                json.dump(frame_logs, json_file, indent=4)

    # Calculate the mean AP@5, AP@10, and AP@15 scores across all frames in the dataset
    num_frames_in_dataset = sum(len(embs) for embs in embeddings)
    mean_AP = {f"AP@{k}": (all_AP_sums[k]/num_frames_in_dataset) for k in all_AP_sums}
    
    return [mean_AP['AP@5'], mean_AP['AP@10'], mean_AP['AP@15']]
