import chainer
import chainer.functions as F
from chainer import using_config, cuda
from tqdm import tqdm
from src.device import move_to_device
from src.visualization import plot_results

def train_step(net, optimizer, x_batch, t_batch):
    net.cleargrads()
    y_batch = net(x_batch)
    loss_batch = F.mean_squared_error(y_batch, t_batch)
    loss_batch.backward()
    optimizer.update()
    return loss_batch

def train_model(net, optimizer, epochs, x_train, t_train, device=0, batch_size=None):
    x_train, t_train = move_to_device(x_train, t_train, device=device)
    if batch_size:
        n_batches = (len(x_train) // batch_size) + (1 if len(x_train) % batch_size > 0 else 0)
    with tqdm(range(epochs), desc='training') as progress_bar:
        for _ in progress_bar:
            if batch_size:
                for i in range(n_batches):
                    x_batch = x_train[i * batch_size:(i + 1) * batch_size]
                    t_batch = t_train[i * batch_size:(i + 1) * batch_size]
                    loss_batch = train_step(net, optimizer, x_batch, t_batch)
                    progress_bar.set_postfix({'MSE': loss_batch.array.item()})
            else:
                loss_train_batch = train_step(net, optimizer, x_train, t_train)
                progress_bar.set_postfix({'MSE': loss_train_batch.array.item()})

def train_model_with_validation(net, optimizer, epochs, x_train, t_train, x_test, t_test, device=0):
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        loss_train_batch = train_step(net, optimizer, x_train, t_train)
        with using_config('train', False), using_config('enable_backprop', False):
            train_loss = F.mean_absolute_error(net(x_train), t_train)
            test_loss = F.mean_absolute_error(net(x_test), t_test)
        train_losses.append(float(cuda.to_cpu(train_loss.array)))
        test_losses.append(float(cuda.to_cpu(test_loss.array)))
        print(f'epoch: {epoch+1}/{epochs}, loss (train): {train_loss.array.item():.4f}, loss (test): {test_loss.array.item():.4f}')
    
    with using_config('train', False), using_config('enable_backprop', False):
        y_pred_train = net(x_train)
        y_pred_test = net(x_test)
    y_pred_train = cuda.to_cpu(y_pred_train.array)
    y_pred_test = cuda.to_cpu(y_pred_test.array)
    plot_results(train_losses, test_losses, t_train, y_pred_train, t_test, y_pred_test)

