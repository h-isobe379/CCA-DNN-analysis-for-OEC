import numpy as np
import optuna
import chainer.functions as F
from chainer import using_config, cuda
from sklearn.model_selection import KFold
from src.dnn import DNN
from src.trainer import train_model
from src.device import prepare_for_gpu, move_to_device
import chainer.optimizers as O
import chainer.optimizer_hooks as oph

def objective(trial, device, X, Y, n_splits, n_epochs, seed):
    n_hidden1 = trial.suggest_int('n_hidden1', 3, 120, step=1)
    n_hidden2 = trial.suggest_int('n_hidden2', 3, 120, step=1)
    n_hidden3 = trial.suggest_int('n_hidden3', 3, 120, step=1)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    momentum = trial.suggest_float('momentum', 0.5, 1.0)
    rate = trial.suggest_float('rate', 1e-6, 1e-3, log=True)
    
    net = DNN(X.shape[1], n_hidden1, n_hidden2, n_hidden3, Y.shape[1])
    optimizer = O.MomentumSGD(lr, momentum)
    optimizer.setup(net)
    optimizer.add_hook(oph.WeightDecay(rate))
    prepare_for_gpu(net, optimizer, device)

    kf = KFold(n_splits, shuffle = True, random_state = seed)
    loss_train_list, loss_val_list = [], []
    for k, (train_index, val_index) in enumerate(kf.split(X)):
        x_train, x_val = X[train_index, :], X[val_index, :]
        t_train, t_val = Y[train_index, :], Y[val_index, :]
        train_model(net, optimizer, n_epochs, x_train, t_train, device)
        # train_model(net, optimizer, n_epochs, x_train, t_train, device, 39)
        with using_config('train', False), using_config('enable_backprop', False):
            loss_train = F.mean_absolute_error(net(x_train), t_train)
            loss_val = F.mean_absolute_error(net(x_val), t_val)
        loss_train_list.append(cuda.to_cpu(loss_train.array).item())
        loss_val_list.append(cuda.to_cpu(loss_val.array).item())
        print(f'#{k+1} in {n_splits}-fold split. '
            f'MAE (train): {loss_train.array.item():.4f}, '
            f'MAE (val): {loss_val.array.item():.4f}')
    
    loss_train_ave = np.mean(loss_train_list)
    loss_val_ave = np.mean(loss_val_list)
    print(f'averaged MAE (train): {loss_train_ave:.4f}, '
        f'averaged MAE (val): {loss_val_ave:.4f}')
    return loss_val_ave

def optimization(X, Y, n_splits, n_epochs, n_trials, seed, device):
    print('\n=== Optimization Process Started ===')
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, device, X, Y, n_splits, n_epochs, seed),
                   n_trials=n_trials)
    print(f'Best score: {study.best_value:.4f}\nBest params: {study.best_params}')

def validation(X, Y, n_epochs, seed, device):
    print('\n=== Validation Process Started ===')
    net = DNN(X.shape[1], 13, 21, 16, Y.shape[1])
    optimizer = O.MomentumSGD(lr=0.007588980090391012, momentum=0.9300360126385203)
    optimizer.setup(net)
    optimizer.add_hook(oph.WeightDecay(rate=5.299335393846766e-06))
    prepare_for_gpu(net, optimizer, device)
    from sklearn.model_selection import train_test_split
    x_train, x_val, t_train, t_val = train_test_split(X, Y, test_size=0.2, random_state=seed)
    from src.trainer import train_model_with_validation
    train_model_with_validation(net, optimizer, n_epochs, x_train, t_train, x_val, t_val, device)

