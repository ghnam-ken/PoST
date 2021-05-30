import numpy as np

import os
import glob
import argparse
import cv2
import pandas as pd

from metric import spatial_accuracy, temporal_accuracy
from utils import denormalize, normalize, draw_pnt, sample_cnt

IMAGE_FOLDER = 'images'
POINT_FOLDER = 'points'
INDEX_FNAME = 'indices.txt'

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--data_path', default='./data/PoST', type=str, help='directory to dataset')
    args.add_argument('--result_path', required=True, type=str, help='directory to result')
    args.add_argument('--output_path', default='./outputs/PoST', type=str, help='directory to outputs')
    args.add_argument('--threshs', nargs='+', default=[0.16, 0.08, 0.04], type=float, help='thresholds to evaluate')
    args.add_argument('--visualize', action='store_true', help='visualize the results')

    opt = args.parse_args()

    data_path = opt.data_path
    result_path = opt.result_path
    visualize = opt.visualize
    threshs = opt.threshs
    output_path = opt.output_path

    exp_name = os.path.basename(result_path)
    output_path = os.path.join(output_path, exp_name)
    os.makedirs(output_path, exist_ok=True)

    seqs = [d for d in os.listdir(data_path) if not d.startswith('.')]
    seqs = sorted(seqs)
    sp_acc_all = {}
    tp_acc_all = {}
    for i, seq_name in enumerate(seqs):
        print(f'({i+1:02d}/{len(seqs):02d}) processing sequence:{seq_name}...')
        if visualize:
            viz_path = os.path.join(output_path, 'visuals', seq_name)
            os.makedirs(viz_path, exist_ok=True)

        # setting accuracy dictionaries
        sp_acc_all[seq_name] = {f'th:{thresh}':[] for thresh in threshs}
        tp_acc_all[seq_name] = {f'th:{thresh}':[] for thresh in threshs}

        # setting path 
        seq_path = os.path.join(data_path, seq_name)
        img_dir = os.path.join(seq_path, IMAGE_FOLDER)
        pnt_dir = os.path.join(seq_path, POINT_FOLDER)
        idx_path = os.path.join(seq_path, INDEX_FNAME)
        res_dir = os.path.join(result_path, seq_name)

        # image and point list 
        img_list = sorted(os.listdir(img_dir))
        pnt_list = sorted(os.listdir(pnt_dir))
        idx = np.loadtxt(idx_path, dtype=np.long)

        # anchor image
        img0_path = os.path.join(img_dir, img_list[0])
        img0 = cv2.imread(img0_path)
        height, width = img0.shape[:2]

        # path to init points
        init_path = os.path.join(seq_path, 'init.txt')
        gt_path = os.path.join(pnt_dir, pnt_list[0])
        res_path = os.path.join(res_dir, pnt_list[0])

        # set inital points
        init = np.loadtxt(init_path)
        gt = np.loadtxt(gt_path)
        if os.path.exists(res_path):
            res = np.loadtxt(res_path)
        else:
            res = np.loadtxt(gt_path)

        # if new index is needed, preprocess it 
        if exp_name == 'Ours':
            idx = sample_cnt(init, idx, len(res))

        # to calculate temporal accuracy
        prev_gt = None
        prev_res = None
        prev_keep = None

        # eval
        for pnt_fname in pnt_list[1:]:
            gt_path = os.path.join(pnt_dir, pnt_fname)
            res_path = os.path.join(res_dir, pnt_fname)

            # load gt and build mask for keep (-1 for no point to eval)
            gt = np.loadtxt(gt_path)
            keep = gt[...,0] >= 0
            
            # load result
            if os.path.exists(res_path):
                res = np.loadtxt(res_path)
            else:
                print(f'[WARNING] {res_path} is not found, using previous point to calculate')
                if res is None:
                    continue

            # sample points to eval
            if len(res) != len(gt):
                res = res[idx]

            # normalize
            gt = normalize(gt, width=width, height=height)
            if np.sum(res > 1.) > 0:
                res = normalize(res, width=width, height=height)

            # calculate accuracy for each threshold in the sequence
            for thresh in threshs:
                sp_acc = spatial_accuracy(gt[keep], res[keep], thresh)
                sp_acc_all[seq_name][f'th:{thresh}'].append(sp_acc)
                if prev_gt is not None:
                    keep_both = np.logical_and(keep, prev_keep)
                    tp_acc = temporal_accuracy(gt[keep_both], res[keep_both], 
                                               prev_gt[keep_both], prev_res[keep_both], 
                                               thresh)

                    tp_acc_all[seq_name][f'th:{thresh}'].append(tp_acc)
           
            # to calculate temporal accuracy 
            prev_gt = gt
            prev_res = res
            prev_keep = keep
            
            # visualization
            if visualize:
                img_fname = pnt_fname.replace('.txt', '.jpg')
                img_path = os.path.join(img_dir, img_fname)
                img = cv2.imread(img_path)
                height, width = img.shape[:2]

                res_denorm = denormalize(res, width, height)
                img = draw_pnt(img, res_denorm)
                save_path = os.path.join(viz_path, img_fname)
                cv2.imwrite(save_path, img)

        # calculate mean of each sequence
        for thresh in threshs:
            sp_acc_all[seq_name][f'th:{thresh}'] = np.mean(sp_acc_all[seq_name][f'th:{thresh}'])
            tp_acc_all[seq_name][f'th:{thresh}'] = np.mean(tp_acc_all[seq_name][f'th:{thresh}'])
    
    # calculate mean for all
    sp_acc_all['mean'] = {}
    tp_acc_all['mean'] = {}
    for thresh in threshs:
        sp_accs = [ sp_acc_all[seq_name][f'th:{thresh}'] for seq_name in sp_acc_all if seq_name != 'mean']
        tp_accs = [ tp_acc_all[seq_name][f'th:{thresh}'] for seq_name in tp_acc_all if seq_name != 'mean']
        sp_acc_all['mean'][f'th:{thresh}'] = np.mean(sp_accs)
        tp_acc_all['mean'][f'th:{thresh}'] = np.mean(tp_accs)
    
    sp_acc_df = pd.DataFrame(sp_acc_all).round(3)
    tp_acc_df = pd.DataFrame(tp_acc_all).round(3)

    print(sp_acc_df)
    print(tp_acc_df)

    sp_save_path = os.path.join(output_path, 'spatial_accuracy.csv')
    tp_save_path = os.path.join(output_path, 'temporal_accuracy.csv')

    sp_acc_df.to_csv(sp_save_path)
    tp_acc_df.to_csv(tp_save_path)
