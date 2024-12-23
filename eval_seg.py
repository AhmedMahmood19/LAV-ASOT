import os
from pyexpat import features

import numpy as np

import utils
import align_dataset_test as align_dataset
from config import CONFIG

from evals.phase_classification import evaluate_phase_classification, compute_ap
from evals.kendalls_tau import evaluate_kendalls_tau
from evals.phase_progression import evaluate_phase_progression

from train import AlignNet
import torch
from torch.utils.tensorboard import SummaryWriter

import random
import argparse
import glob
from natsort import natsorted
import segmentation_asot
from evals.metrics import ClusteringMetrics, indep_eval_metrics
import torch.nn.functional as F
import torch.nn as nn

def _partition_and_sample(n_samples, n_frames):
    if n_samples is None:
        indices = np.arange(n_frames)
        mask = np.full(n_frames, 1, dtype=bool)
    elif n_samples < n_frames:
        if random:
            boundaries = np.linspace(0, n_frames - 1, n_samples + 1).astype(int)
            indices = np.random.randint(low=boundaries[:-1], high=boundaries[1:])
        else:
            indices = np.linspace(0, n_frames - 1, n_samples).astype(int)
        mask = np.full(n_samples, 1, dtype=bool)
    else:
        indices = np.concatenate((np.arange(n_frames), np.full(n_samples - n_frames, n_frames - 1)))
        mask = np.concatenate((np.full(n_frames, 1, dtype=bool), np.zeros(n_samples - n_frames, dtype=bool)))
    return indices, mask

def get_embeddings(model, data, mappings, args, CONFIG):
    embeddings = []
    labels = []
    frame_paths = []
    names = []
    masks = []
    device = f"cuda:{args.device}"
    print("Here")
    for act_iter in iter(data):

        for i, seq_iter in enumerate(act_iter):

            seq_embs = []
            seq_fpaths = []
            original = 0
            for j, batch in enumerate(seq_iter):

                a_X, a_name, a_frames = batch
                a_X = a_X.to(device).unsqueeze(0)
                original = a_X.shape[1] // 2

                b = a_X[:, -1].clone()
                try:
                    b = torch.stack([b] * ((args.num_frames * 2) - a_X.shape[1]), axis=1).to(device)
                except:
                    b = torch.from_numpy(np.array([])).float().to(device)
                a_X = torch.concat([a_X, b], axis=1)
                a_emb = model(a_X)[:, :original, :]

                if args.verbose:
                    print(f'Seq: {i}, ', a_emb.shape)

                seq_embs.append(a_emb.squeeze(0).detach().cpu().numpy())
                seq_fpaths.extend(a_frames)
                # print(j)


            seq_embs = np.concatenate(seq_embs, axis=0)

            name = str(a_name).split('/')[-1]

            # lab = labels_npy[name]['labels']
            data_path = './Data_Test/'
            gt = [line.rstrip() for line in open(os.path.join(data_path, 'groundTruth', name))]
            inds, mask = _partition_and_sample(CONFIG.EVAL.NUM_FRAMES, len(gt))
            gt = torch.Tensor([mappings[gt[ind]] for ind in inds]).long()
            # This deals with the issue when there's a length mismatch in labels and frames
            end = min(seq_embs.shape[0], len(gt))
            lab = gt[:end]
            seq_embs = seq_embs[:end]
            # make the no. of framepaths same as the no. of embs
            seq_fpaths = seq_fpaths[:end]

            embeddings.append(seq_embs[:end])
            frame_paths.append(seq_fpaths)
            names.append(a_name)
            labels.append(lab)
            masks.append(mask)
            print("calculating embs")
            # print(len(seq_embs[:end]))
            # print(len(mask))
            # print(len(a_name))
            # print(len(lab))
            # print(len(seq_fpaths))
            # print(seq_fpaths)

    return embeddings, names, labels, frame_paths, masks

def prep(x):
    i, nm = x.rstrip().split(' ')
    return nm, int(i)

def evaluate(features, mask, clusters, CONFIG):
    device = torch.device("cuda")
    # features = torch.tensor(features)
    # features = features.to(device)
    # mask = torch.tensor(mask)
    # mask = mask.to(device)
    T = features.shape[0]

    # print(CONFIG.N_CLUSTERS)
    temp_prior_segmentation = segmentation_asot.temporal_prior(T, CONFIG.N_CLUSTERS, CONFIG.RHO, device)
    cost_matrix_segmentation = 1. - features @ clusters.T.unsqueeze(0)
    # print(cost_matrix_segmentation.shape)
    # print(features.shape)
    # print(temp_prior_segmentation.shape)
    cost_matrix_segmentation  += temp_prior_segmentation
    # mask = mask.unsqueeze(0)
    # print(mask.shape)
    # print(cost_matrix_segmentation.shape)

    segmentation, _ = segmentation_asot.segment_asot(cost_matrix_segmentation, mask,
                                                                 eps= CONFIG.EPS_EVAL, alpha=CONFIG.ALPHA_EVAL,
                                                                 radius=CONFIG.RADIUS_GW,
                                                                 ub_frames=CONFIG.UB_FRAMES, ub_actions=CONFIG.UB_ACTIONS,
                                                                 lambda_frames=CONFIG.LAMBDA_FRAMES_EVAL,
                                                                 lambda_actions=CONFIG.LAMBDA_ACTIONS_EVAL,
                                                                 n_iters=CONFIG.N_OT_EVAL, step_size=CONFIG.STEP_SIZE)
    segments = segmentation.argmax(dim=2)
    return segments

def main(ckpts, args, CONFIG):
    summary_dest = os.path.join(args.dest, 'eval_logs')
    os.makedirs(summary_dest, exist_ok=True)

    d = CONFIG.DTWALIGNMENT.EMBEDDING_SIZE
    clusters = nn.parameter.Parameter(data=F.normalize(torch.randn(CONFIG.N_CLUSTERS, d, device=args.device), dim=-1), requires_grad=True)
    mof = ClusteringMetrics(metric='mof')
    f1 = ClusteringMetrics(metric='f1')
    miou = ClusteringMetrics(metric='miou')
    for ckpt in ckpts:
        writer = SummaryWriter(summary_dest, filename_suffix='eval_logs')

        print(f"\n\nStarting Evaluation On Checkpoint: {ckpt}\n\n")

        # get ckpt-step from the ckpt name
        _, ckpt_step = ckpt.split('.')[0].split('_')[-2:]
        ckpt_step = int(ckpt_step.split('=')[1])
        DEST = os.path.join(args.dest, 'eval_step_{}'.format(ckpt_step))

        device = f"cuda:{args.device}"
        model = AlignNet.load_from_checkpoint(ckpt, map_location=device)
        model.to(device)
        model.eval()

        # grad off
        torch.set_grad_enabled(False)

        if args.num_frames:
            CONFIG.TRAIN.NUM_FRAMES = args.num_frames
            CONFIG.EVAL.NUM_FRAMES = args.num_frames

        # CONFIG.update(model.hparams.config)

        if args.data_path:
            data_path = args.data_path
        else:
            data_path = CONFIG.DATA_PATH
        data_path = './Data_Test/'

        train_path = os.path.join(data_path, 'Test')
        val_path = os.path.join(data_path, 'Test')
        # lab_name = "pouring" + "_val"
        # labels = np.load(f"./npyrecords/{lab_name}.npy", allow_pickle=True).item()
        action_mapping =list(map(prep, open(os.path.join(data_path, 'mapping/mapping.txt'))))
        action_mapping = dict(action_mapping)
        # create dataset
        _transforms = utils.get_transforms(augment=False)

        random.seed(0)
        train_data = align_dataset.AlignData(train_path, args.batch_size, CONFIG.DATA, transform=_transforms,
                                             flatten=False)
        val_data = align_dataset.AlignData(val_path, args.batch_size, CONFIG.DATA, transform=_transforms, flatten=False)

        mof_score = []
        f1_score = []
        miou_score = []

        for i_action in range(train_data.n_classes):
            # print("HERE",i_action)
            train_data.set_action_seq(i_action)
            val_data.set_action_seq(i_action)

            train_act_name = train_data.get_action_name(i_action)
            val_act_name = val_data.get_action_name(i_action)

            assert train_act_name == val_act_name

            if args.verbose:
                print(f'Getting embeddings for {train_act_name}...')

            val_embs, val_names, val_labels, val_frame_paths, val_masks = get_embeddings(model, val_data, action_mapping, args, CONFIG)
            # train and val are the exact same data now
            train_embs, train_names, train_labels, train_frame_paths, train_masks = val_embs, val_names, val_labels, val_frame_paths, val_masks

            # # save embeddings
            # os.makedirs(DEST, exist_ok=True)
            # DEST_TRAIN = os.path.join(DEST, f'train_{train_act_name}_embs.npy')
            # DEST_VAL = os.path.join(DEST, f'val_{val_act_name}_embs.npy')
            #
            # np.save(DEST_TRAIN, {'embs': train_embs, 'names': train_names, 'labels': train_labels,
            #                      'frame_paths': train_frame_paths, 'masks': train_masks})
            # np.save(DEST_VAL,
            #         {'embs': val_embs, 'names': val_names, 'labels': val_labels, 'frame_paths': val_frame_paths, 'masks': val_masks})
            #
            # train_embeddings = np.load(DEST_TRAIN, allow_pickle=True).tolist()
            # val_embeddings = np.load(DEST_VAL, allow_pickle=True).tolist()
            #
            # train_embs, train_labels, train_names, train_frame_paths, train_masks= train_embeddings['embs'], train_embeddings[
            #     'labels'], train_embeddings['names'], train_embeddings['frame_paths'], train_embeddings['masks']
            # val_embs, val_labels, val_names, val_frame_paths, val_masks = val_embeddings['embs'], val_embeddings['labels'], val_embeddings['names'], val_embeddings['frame_paths'], val_embeddings['masks']

            print("########In Evaluation######")
            for i in range(len(val_embs)):
                feature = torch.tensor(val_embs[i])
                feature = feature.to(device)
                feature = feature.unsqueeze(0)
                mask = torch.tensor(val_masks[i])
                mask = mask.to(device)
                mask = mask.unsqueeze(0)
                gt = val_labels[i]
                gt = gt.to(device)
                gt = gt.unsqueeze(0)
                segments = evaluate(feature, mask, clusters, CONFIG)
                mof.update(segments, gt, mask)
                f1.update(segments, gt, mask)
                miou.update(segments, gt, mask)

                m = indep_eval_metrics(segments, gt, mask, ['mof', 'f1', 'miou'],  exclude_cls=None)
                mof_score.append(m['mof'])
                f1_score.append(m['f1'])
                miou_score.append(m['miou'])

        all_mof = np.mean(mof_score)
        all_f1 = np.mean(f1_score)
        all_miou = np.mean(miou_score)

        print("mof score: ", all_mof)
        print("f1 score: ", all_f1)
        print("miou score: ", all_miou)

        mof.reset()
        f1.reset()
        miou.reset()
        # writer.flush()
        #
        # writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--dest', type=str, default='./')

    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--visualize', dest='visualize', action='store_true')
    parser.add_argument('--device', type=int, default=0, help='Cuda device to be used')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--num_frames', type=int, default=256, help='Path to dataset')

    parser.add_argument('--alpha-eval', '-ae', type=float, default=0.6,
                        help='weighting of KOT term on frame features in OT')
    parser.add_argument('--n-ot-eval', '-no', type=int, nargs='+', default=[25, 1],
                        help='number of outer and inner iterations for ASOT solver (eval)')
    parser.add_argument('--eps-eval', '-ee', type=float, default=0.04,
                        help='entropy regularization for OT during val/test')
    parser.add_argument('--radius-gw', '-r', type=float, default=0.02,
                        help='Radius parameter for GW structure loss')  # original 0.02
    parser.add_argument('--ub-frames', '-uf', action='store_true',
                        help='relaxes balanced assignment assumption over frames, i.e., each frame is assigned')
    parser.add_argument('--ub-actions', '-ua', action='store_true',
                        help='relaxes balanced assignment assumption over actions, i.e., each action is uniformly represented in a video')
    parser.add_argument('--lambda-frames-eval', '-lfe', type=float, default=0.05,
                        help='penalty on balanced frames assumption for test')
    parser.add_argument('--lambda-actions-eval', '-lae', type=float, default=0.01,
                        help='penalty on balanced actions assumption for test')
    parser.add_argument('--rho', type=float, default=0.25,
                        help='Factor for global structure weighting term')  # original was 0.25, 0.2 yield better results
    parser.add_argument('--n-clusters', '-c', type=int, default=22,
                        help='number of actions/clusters')
    parser.add_argument('--step-size', '-ss', type=float, default=None,
                        help='Step size/learning rate for ASOT solver. Worth setting manually if ub-frames && ub-actions')

    args = parser.parse_args()

    if args.alpha_eval:
        CONFIG.ALPHA_EVAL = args.alpha_eval
    if args.n_ot_eval:
        CONFIG.N_OT_EVAL = args.n_ot_eval
    if args.eps_eval:
        CONFIG.EPS_EVAL = args.eps_eval
    if args.radius_gw:
        CONFIG.RADIUS_GW = args.radius_gw
    if args.lambda_frames_eval:
        CONFIG.LAMBDA_FRAMES_EVAL = args.lambda_frames_eval
    if args.lambda_actions_eval:
        CONFIG.LAMBDA_ACTIONS_EVAL = args.lambda_actions_eval
    if args.rho:
        CONFIG.RHO = args.rho
    if args.n_clusters:
        CONFIG.N_CLUSTERS = args.n_clusters

    # NOTE: the default values of these args cause the if condition to fail, so we wont use the if condition for them
    CONFIG.STEP_SIZE = args.step_size
    CONFIG.UB_FRAMES = args.ub_frames
    CONFIG.UB_ACTIONS = args.ub_actions
    # print(CONFIG.N_CLUSTERS)
    if os.path.isdir(args.model_path):
        ckpts = natsorted(glob.glob(os.path.join(args.model_path, '*')))
    else:
        ckpts = [args.model_path]

    ckpt_mul = args.device
    main(ckpts, args, CONFIG)