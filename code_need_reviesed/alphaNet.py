from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from original_data_preprocessing import *
from pytorchtools import EarlyStopping
import matplotlib.pyplot as plt
from model import *

EPOCHS = 20
LR = 0.0001
PATIENCE = 10
BATCH_SIZE = 1000


def read_data_new():
    x_name = 'data/dict_train_valid_x.pkl.gz'
    y_name = 'data/dict_train_valid_y.pkl.gz'
    data_dict_x = pickle.loads(gzip.decompress(open(x_name, 'rb').read()))
    data_dict_y = pickle.loads(gzip.decompress(open(y_name, 'rb').read()))
    x_train, x_valid, y_train, y_valid = data_dict_x['x_train'], data_dict_x['x_valid'], data_dict_y['y_train'], data_dict_y['y_valid']
    x_train, x_valid, y_train, y_valid = map(torch.tensor, (x_train, x_valid, y_train, y_valid))
    return x_train, x_valid, y_train, y_valid


x_train, x_valid, y_train, y_valid = read_data_new()
print("Data Ready!")
train_data_dataset = TensorDataset(x_train, y_train)
valid_data_dataset = TensorDataset(x_valid, y_valid)
print("\ntrain数据集有{}个数据，每个数据的大小为{}和{};valid数据集有{}个数据，每个数据的大小为{}和{}".format(len(train_data_dataset),
                                                                           train_data_dataset.__getitem__(0)[
                                                                               0].unsqueeze(
                                                                               0).shape,
                                                                           train_data_dataset.__getitem__(0)[
                                                                               1].unsqueeze(
                                                                               0).shape, len(valid_data_dataset),
                                                                           valid_data_dataset.__getitem__(0)[
                                                                               0].unsqueeze(
                                                                               0).shape,
                                                                           valid_data_dataset.__getitem__(0)[
                                                                               1].unsqueeze(
                                                                               0).shape))


def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=False),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


def get_model():
    model_init = AlphaNet_v1()
    print(torch.cuda.is_available())
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_init.to(dev)
    return model_init, optim.RMSprop(model_init.parameters(), lr=LR), dev


def preprocess(x, y):
    return x.view(-1, 1, 9, 30).to(dev), y.to(dev)


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield self.func(*b)


train_data_dataloader, valid_data_dataloader = get_data(train_data_dataset, valid_data_dataset, BATCH_SIZE)
print("训练集分组数为{}, 验证级分组数为{}".format(len(train_data_dataloader), len(valid_data_dataloader)))
train_data_dataloader = WrappedDataLoader(train_data_dataloader, preprocess)
valid_data_dataloader = WrappedDataLoader(valid_data_dataloader, preprocess)


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb).float(), yb.float())

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()  # whether to adjust it? no, it's true!

    return loss.item(), len(xb)


def fit_all(epochs, model, loss_func, opt, train_dl, valid_dl, patience):
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(epochs):
        model.train()
        for batch_idx, (xb, yb) in enumerate(train_dl):
            # print("========================")
            # print("输入x的大小为：", xb.shape)
            # print("输入y的大小为：", yb.shape)
            loss, xb_len = loss_batch(model, loss_func, xb, yb, opt)
            if (batch_idx + 1) % (len(train_data_dataloader) + 1) != 0:
                batch_idx = batch_idx + 1
                print('Train Epoch: {} [{}/{} {:.2f}%]\tLoss: {:.6f}'.format(epoch + 1, batch_idx * xb_len, len(train_data_dataset), (100 * batch_idx * xb_len) / len(train_data_dataset), loss))
            else:
                batch_idx = batch_idx + 1
                print('Train Epoch: {} [{}/{} {:.0f}%]\tLoss: {:.6f}'.format(epoch + 1, (batch_idx - 1) * 1000 + xb_len,
                                                                             len(train_data_dataset),
                                                                             (100 * ((batch_idx - 1) * 1000 + xb_len)) / len(train_data_dataset), loss))
            train_losses.append(loss)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
            valid_losses += losses

        valid_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        avg_valid_losses.append((epoch+1, valid_loss))

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load('checkpoint.pt'))
    return train_losses, valid_losses, avg_valid_losses


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


# This is main function
loss_func = torch.nn.MSELoss(reduction='sum')
model, opt, dev = get_model()
model.to(dev)
for name, each in model.named_parameters():
    print(name, each, each.shape, each.requires_grad)
train_losses, valid_losses, avg_valid_losses = fit_all(EPOCHS, model, loss_func, opt, train_data_dataloader, valid_data_dataloader, PATIENCE)
for name, each in model.named_parameters():
    print(name, each, each.shape, each.requires_grad)
plot_loss(train_losses, valid_losses)
plot_early_stop(avg_valid_losses)


# shap.initjs()
# explainer = shap.Explainer(model)
# shap_values = explainer.shap_values(X)

