import matplotlib.pyplot as plt

def plot_results(train_losses, test_losses, t_train, y_pred_train, t_test, y_pred_test):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.plot(train_losses, label='train', color='blue', linestyle='-', linewidth=2)
    ax1.plot(test_losses, label='test', color='orange', linestyle='--', linewidth=2)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss (MAE)')
    ax1.set_title('Training and Test Loss over Epochs')
    ax1.legend()
    
    ax2.scatter(t_train[:, 0], y_pred_train[:, 0], label='train, hydroxo', alpha=0.6, color='blue')
    ax2.scatter(t_train[:, 1], y_pred_train[:, 1], label='train, oxyl', alpha=0.6, color='orange')
    ax2.scatter(t_test[:, 0], y_pred_test[:, 0], label='test, hydroxo', alpha=0.6, color='green')
    ax2.scatter(t_test[:, 1], y_pred_test[:, 1], label='test, oxyl', alpha=0.6, color='red')
    
    min_val = min(t_train.min(), t_test.min())
    max_val = max(t_train.max(), t_test.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='ideal')
    ax2.set_xlabel('DFT driving force')
    ax2.set_ylabel('DNN-predicted driving force')
    ax2.set_title('Predicted vs DFT Driving Force')
    ax2.legend()

    plt.tight_layout()
    plt.show()

