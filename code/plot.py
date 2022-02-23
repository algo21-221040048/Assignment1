import matplotlib.pyplot as plt


# Visualizing the Loss and the Early Stopping Checkpoint
# visualize the loss as the network trained
def plot_loss(train_losses, valid_losses):
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss, BS={}'.format(BATCH_SIZE))
    plt.plot(range(1, len(valid_losses)+1), valid_losses, label='Validation Loss, BS={}'.format(BATCH_SIZE))

    plt.xlabel('batches')
    plt.ylabel('loss')
    plt.xlim(0, len(train_losses)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('loss_plot.png', bbox_inches='tight')


def plot_early_stop(avg_valid_losses):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(avg_valid_losses) + 1), avg_valid_losses, label='Average Validation Loss Per Epoch')

    # find position of lowest validation loss
    minposs = avg_valid_losses.index(min(avg_valid_losses)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xlim(0, len(avg_valid_losses) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('loss_plot_epochs.png', bbox_inches='tight')


# shap.initjs()
# explainer = shap.Explainer(model)
# shap_values = explainer.shap_values(X)