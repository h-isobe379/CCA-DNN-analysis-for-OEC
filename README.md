# CCA-DNN-analysis-for-OEC

This project demonstrates an integrated framework for data loading, preprocessing, regularized CCA analysis, DNN training/validation, and hyperparameter optimization with Optuna. The primary objective is to elucidate the relationship between the collective motion within the primary coordination sphere and the catalytic function of the oxygen-evolving complex in Photosystem II.

## Usage

1. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the main script with the desired configuration:
    ```bash
    python scripts/main.py \
    --data_dir ./data \
    --file_x raw_data_for_X.txt \
    --file_y raw_data_for_Y.txt \
    --lambda1 0.5306122 \
    --lambda2 0.008979592 \
    --scale_method minmax \
    --all_elements True \
    --n_splits 10 \
    --n_epochs 10000 \
    --n_trials 10000 \
    --seed 123 \
    --device -1 \
    --mode optimization
    ```

## Command-Line Arguments

- `--data_dir`: Data folder path. Default: `./data`
- `--file_x`: Filename for X. Default: `raw_data_for_X.txt`
- `--file_y`: Filename for Y. Default: `raw_data_for_Y.txt`
- `--lambda1`: Regularization parameter lambda1 for CCA. Default: `0.5306122`
- `--lambda2`: Regularization parameter lambda2 for CCA. Default: `0.008979592`
- `--scale_method`: Scaling method for DNN (`minmax` or `zscore`). Default: `minmax`
- `--all_elements`: Boolean flag to apply the same scaling to all elements. Default: `True`
- `--n_splits`: Number of K-fold splits. Default: `10`
- `--n_epochs`: Number of training epochs. Default: `10000`
- `--n_trials`: Number of Optuna trials. Default: `10000`
- `--seed`: Random seed. Default: `None`
- `--device`: GPU device ID (`-1` for CPU). Default: `-1`
- `--mode`: Execution mode (`optimization` or `validation`). Default: `optimization`

