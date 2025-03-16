#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse

import numpy as np
import pandas as pd

from src.io import load_data
from src.scaling import DataScaler
from src.metrics import (
    calculate_pairwise_distances,
    calculate_inertial_radii,
    calculate_inertial_radii_alt,
)
from src.rcca import regularized_cca
from src.dnn import DNN
from src.trainer import (
    train_model_with_validation,
    train_model,
)
from src.optimizer import optimization, validation
from src.device import DeviceContextManager, move_to_device, prepare_for_gpu

def main(args):
    folder_path = args.data_dir
    filename_for_X = args.file_x
    filename_for_Y = args.file_y

    data_list = load_data(os.path.join(folder_path, filename_for_X))
    entry_list = [f'entry{i}' for i in range(1, len(data_list) + 1)]
    atom_list = ['OA', 'OB', 'OC', 'OD', 'OE', 'OF', 'OG', 'OH', 'OI', 'OJ', 'OK', 'OL', 'N']
    pairwise_list = [f'{first}-{second}'
                     for idx, first in enumerate(atom_list)
                     for second in atom_list[idx+1:]]
    
    all_pairwise_distances = calculate_pairwise_distances(data_list, entry_list, pairwise_list)
    selected_pairwise_list = [
        'OB-OC', 'OB-OD', 'OB-OE', 'OB-OJ', 'OB-OL', 'OB-N',
        'OC-OE', 'OC-OJ', 'OC-OL', 'OC-N',
        'OD-OE', 'OD-OJ', 'OD-OL', 'OD-N',
        'OE-OJ', 'OE-OL', 'OE-N',
        'OJ-OL', 'OJ-N'
    ]
    selected_pairwise_distances = all_pairwise_distances.loc[:, selected_pairwise_list]
    
    inertial_radii = calculate_inertial_radii(data_list, entry_list)
    # For verification, one could use:
    # inertial_radii_alt = calculate_inertial_radii_alt(all_pairwise_distances, atom_list)

    deltaE = pd.read_csv(os.path.join(folder_path, filename_for_Y), header=0, index_col=0)

    X_for_CCA = selected_pairwise_distances.values[:60, :]
    Y_for_CCA = deltaE.values[:60, :]
    
    canonical_corr_ini, A_ini, B_ini = regularized_cca(
        X_for_CCA, Y_for_CCA, lambda1=0, lambda2=args.lambda2, n_components=2
    )
    canonical_corr, A, B = regularized_cca(
        X_for_CCA, Y_for_CCA, lambda1=args.lambda1, lambda2=args.lambda2, n_components=2
    )
    shrinkage_ratio = np.where(np.abs(A_ini[:, 0]) != 0, 
        np.abs(A[:, 0]) / np.abs(A_ini[:, 0]), 0)

    df_canonical_corr = pd.DataFrame(
        canonical_corr,
        index = ['Component 1', 'Component 2'],
        columns = ['rho']
    )
    df_A = pd.DataFrame(
        A,
        columns = ['Component 1', 'Component 2'],
        index = selected_pairwise_list
    )
    df_B = pd.DataFrame(
        B,
        columns = ['Component 1', 'Component 2'],
        index = ['hydroxo-oxo', 'oxyl-oxo']
    )
    df_shrinkage = pd.DataFrame({
        'Feature': selected_pairwise_list,
        'Initial Weight |a|': abs(A_ini[:, 0]),
        'Final Weight |a|': abs(A[:, 0]),
        'Shrinkage Ratio': shrinkage_ratio
    })
    df_shrinkage.sort_values('Shrinkage Ratio', ascending = False, inplace = True)

    print('\n=== Canonical Correlations ===')
    print(df_canonical_corr.to_string())
    print('\n=== Canonical Weight Vectors for X ===')
    print(df_A.to_string())
    print('\n=== Canonical Weight Vectors for Y ===')
    print(df_B.to_string())
    print(f'\n=== Feature-Specific Shrinkage Ratios for Component 1 ===')
    print(df_shrinkage.to_string(index = False))
    
    scaler = DataScaler(args.scale_method, args.all_elements)
    X, Y = scaler.scale_data(selected_pairwise_distances, deltaE)
    X, Y = X.astype('float32'), Y.astype('float32')
    
    with DeviceContextManager(args.device):
        X, Y = move_to_device(X, Y, device=args.device)
        if args.mode == 'optimization':
            optimization(X, Y, args.n_splits, args.n_epochs, args.n_trials, args.seed, args.device)
        elif args.mode == 'validation':
            validation(X, Y, args.n_epochs, args.seed, args.device)
        else:
            print("Invalid mode. Choose either 'optimization' or 'validation'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNN & CCA analysis')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data folder path')
    parser.add_argument('--file_x', type=str, default='raw_data_for_X.txt', help='Filename for X')
    parser.add_argument('--file_y', type=str, default='raw_data_for_Y.txt', help='Filename for Y')
    parser.add_argument('--lambda1', type=float, default=0.5306122, help='Regularization parameter lambda1 (default: 0.5306122)')
    parser.add_argument('--lambda2', type=float, default=0.008979592, help='Regularization parameter lambda2 (default: 0.008979592)')
    parser.add_argument('--scale_method', type=str, default='minmax', choices=['minmax', 'zscore'])
    parser.add_argument('--all_elements', type=lambda s: s.lower() in ['true', '1', 'yes', 't'], default=True,
                        help='Apply uniform scaling to all elements. Specify "False" to disable (default: True).')
    parser.add_argument('--n_splits', type=int, default=10, help='Number of K-fold splits')
    parser.add_argument('--n_epochs', type=int, default=10000, help='Number of epochs')
    parser.add_argument('--n_trials', type=int, default=10000, help='Number of Optuna trials')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--device', type=int, default=-1, help='GPU device id (or -1 for CPU)')
    parser.add_argument('--mode', type=str, default='optimization', choices=['optimization', 'validation'],
                        help='Mode of execution')
    args = parser.parse_args()
    main(args)

