# This part of code is the main body, which used to train the data and make validation
import torch.utils.data
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch import optim
from pytorchtools import EarlyStopping
from model import *
import pickle
import gzip


# Hyper parameters
EPOCHS = 10
LR = 0.0001
PATIENCE = 10
BATCH_SIZE = 1000
TRAIN_SIZE = 0.5


# This function is used to read and split the particular data
def read_data(part_num: int) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    x_name = '../Data_preprocessing/data_x_part_{}.pkl.gz'.format(part_num)
    y_name = '../Data_preprocessing/data_y_part_{}.pkl.gz'.format(part_num)
    x = pickle.loads(gzip.decompress(open(x_name, 'rb').read()))
    y = pickle.loads(gzip.decompress(open(y_name, 'rb').read()))
    x_train, x_valid, y_train, y_valid = train_test_split(np.array(x), np.array(y), train_size=TRAIN_SIZE, shuffle=False)
    x_train, x_valid, y_train, y_valid = map(torch.tensor, (x_train, x_valid, y_train, y_valid))
    return x_train, x_valid, y_train, y_valid


# This function is used to reconstruct batches through `Dataloader` method
def get_data(train_ds: torch.utils.data.TensorDataset, valid_ds: torch.utils.data.TensorDataset, bs: int)\
        -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    return DataLoader(train_ds, batch_size=bs, shuffle=False), DataLoader(valid_ds, batch_size=bs * 2)


# This function is used to build the model and optimizer
def get_model() -> (AlphaNet_v1, optim, torch.device):
    model_init = AlphaNet_v1()
    print(torch.cuda.is_available())
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_init.to(dev)
    return model_init, optim.RMSprop(model_init.parameters(), lr=LR), dev


# This function is used to reshape the training data
def preprocess(x: torch.tensor, y: torch.tensor, dev: torch.device) -> (torch.tensor, torch.tensor):
    return x.view(-1, 1, 9, 30).to(dev), y.to(dev)


# This function is to calculate the loss of each batch
def loss_batch(model: AlphaNet_v1, loss_func: torch.nn.MSELoss, xb: torch.tensor, yb: torch.tensor, opt=None) -> (torch.tensor, int):
    loss = loss_func(model(xb).float(), yb.float())

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()  # whether to adjust it? no, it's true!

    return loss.item(), len(xb)


#  This function is used to train the model and make the validation
def fit(epochs: int, model: AlphaNet_v1, loss_func: torch.nn.MSELoss, opt: torch.optim.RMSprop, train_dl: torch.utils.data.DataLoader, valid_dl: torch.utils.data.DataLoader, train_ds: torch.utils.data.Dataset, patience: int):
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(epochs):
        model.train()
        for batch_idx, (xb, yb) in enumerate(train_dl):
            loss, xb_len = loss_batch(model, loss_func, xb, yb, opt)
            batch_idx = batch_idx + 1
            if batch_idx % len(train_dl) != 0:
                print('Train Epoch: {} [{}/{} {:.2f}%]\tLoss: {:.6f}'.format(epoch + 1, batch_idx * xb_len, len(train_ds), (100 * batch_idx * xb_len) / len(train_ds), loss))
            else:
                print('Train Epoch: {} [{}/{} {:.0f}%]\tLoss: {:.6f}'.format(epoch + 1, (batch_idx - 1) * BATCH_SIZE + xb_len, len(train_ds), 100, loss))
            train_losses.append(loss)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
            for each in losses:
                valid_losses.append(each)

        valid_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        avg_valid_losses.append((epoch+1, valid_loss))

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load('checkpoint.pt'))
    return train_losses, valid_losses, avg_valid_losses


# This class is used to preprocess each batch
class WrappedDataLoader:
    def __init__(self, dl, func, dev):
        self.dl = dl
        self.func = func
        self.dev = dev

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield self.func(*b, self.dev)


# This is the main function
def main(part_num: int):
    PATH_train_losses = 'train_losses_{}.pkl.gz'.format(part_num)
    PATH_valid_losses = 'valid_losses_{}.pkl.gz'.format(part_num)
    PATH_avg_valid_losses = 'avg_valid_losses_{}.pkl.gz'.format(part_num)
    x_train, x_valid, y_train, y_valid = read_data(part_num)
    train_data_dataset = TensorDataset(x_train, y_train)
    valid_data_dataset = TensorDataset(x_valid, y_valid)
    train_data_dataloader, valid_data_dataloader = get_data(train_data_dataset, valid_data_dataset, BATCH_SIZE)
    model, opt, device = get_model()
    model.to(device)
    loss_func = torch.nn.MSELoss(reduction='sum')
    train_data_dataloader = WrappedDataLoader(train_data_dataloader, preprocess, device)
    valid_data_dataloader = WrappedDataLoader(valid_data_dataloader, preprocess, device)
    print("\ntrain数据集有{}个数据，每个数据的大小为{}和{};valid数据集有{}个数据，每个数据的大小为{}和{}".format(
        len(train_data_dataset),
        train_data_dataset.__getitem__(0)[0].unsqueeze(0).shape,
        train_data_dataset.__getitem__(0)[1].unsqueeze(0).shape,
        len(valid_data_dataset),
        valid_data_dataset.__getitem__(0)[0].unsqueeze(0).shape,
        valid_data_dataset.__getitem__(0)[1].unsqueeze(0).shape))
    print("训练集分组数为{}, 验证级分组数为{}".format(len(train_data_dataloader), len(valid_data_dataloader)))

    train_losses, valid_losses, avg_valid_losses = fit(EPOCHS, model, loss_func, opt, train_data_dataloader,
                                                       valid_data_dataloader, train_data_dataset, PATIENCE)
    open(PATH_train_losses, 'wb').write(gzip.compress(pickle.dumps(train_losses)))
    open(PATH_valid_losses, 'wb').write(gzip.compress(pickle.dumps(valid_losses)))
    open(PATH_avg_valid_losses, 'wb').write(gzip.compress(pickle.dumps(avg_valid_losses)))


if __name__ == '__main__':
    for i in range(1, 8):
        main(i)
