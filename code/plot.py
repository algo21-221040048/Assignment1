import matplotlib.pyplot as plt
import pickle
import gzip


# This function is used to read and split the particular data
def read_data(part_num: int) -> ():
    x_name = 'train_losses_{}.pkl.gz'.format(part_num)
    y_name = 'valid_losses_{}.pkl.gz'.format(part_num)
    z_name = 'avg_valid_losses_{}.pkl.gz'.format(part_num)
    x = pickle.loads(gzip.decompress(open(x_name, 'rb').read()))
    y = pickle.loads(gzip.decompress(open(y_name, 'rb').read()))
    z = pickle.loads(gzip.decompress(open(z_name, 'rb').read()))
    return x, y, z


# Visualizing the Loss and the Early Stopping Checkpoint
# visualize the loss as the network trained
def plot_loss(t_l: list, v_l: list) -> ():
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(t_l) + 1), t_l, label='Training Loss, BS={}'.format(1000))
    plt.plot(range(1, len(v_l) + 1), v_l, label='Validation Loss, BS={}'.format(2000))
    plt.xlabel('batches')
    plt.ylabel('loss')
    plt.xlim(0, len(t_l) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('loss_plot.png', bbox_inches='tight')


def plot_early_stop(avg_v_l: list) -> ():
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(avg_v_l) + 1), avg_v_l, label='Average Validation Loss Per Epoch')

    # find position of lowest validation loss
    min_poss = avg_v_l.index(min(avg_v_l)) + 1
    plt.axvline(min_poss, linestyle='--', color='r', label='Early Stopping Checkpoint')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xlim(0, len(avg_v_l) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('loss_plot_epochs.png', bbox_inches='tight')


train_losses, valid_losses, avg_valid_losses = read_data(1)
plot_loss(train_losses, valid_losses)
num, avg_valid_losses = zip(*avg_valid_losses)
plot_early_stop(avg_valid_losses)


# shap.initjs()
# explainer = shap.Explainer(model)
# shap_values = explainer.shap_values(X)