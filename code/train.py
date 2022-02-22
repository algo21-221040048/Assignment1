# This part of code is the main body, which used to train the data and make validation
import torch.utils.data
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch import optim
from pytorchtools import EarlyStopping
from plot import *
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
def preprocess(x: torch.tensor, y: torch.tensor) -> (torch.tensor, torch.tensor):
    return x.view(-1, 1, 9, 30).to(dev), y.to(dev)


# This class is used to preprocess each batch
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


x_train, x_valid, y_train, y_valid = read_data(1)
train_data_dataset = TensorDataset(x_train, y_train)
valid_data_dataset = TensorDataset(x_valid, y_valid)
train_data_dataloader, valid_data_dataloader = get_data(train_data_dataset, valid_data_dataset, BATCH_SIZE)
model, opt, dev = get_model()
model.to(dev)
loss_func = torch.nn.MSELoss(reduction='sum')
train_data_dataloader = WrappedDataLoader(train_data_dataloader, preprocess)
valid_data_dataloader = WrappedDataLoader(valid_data_dataloader, preprocess)


print("\ntrain数据集有{}个数据，每个数据的大小为{}和{};valid数据集有{}个数据，每个数据的大小为{}和{}".format(
      len(train_data_dataset),
      train_data_dataset.__getitem__(0)[0].unsqueeze(0).shape,
      train_data_dataset.__getitem__(0)[1].unsqueeze(0).shape,
      len(valid_data_dataset),
      valid_data_dataset.__getitem__(0)[0].unsqueeze(0).shape,
      valid_data_dataset.__getitem__(0)[1].unsqueeze(0).shape))
print("训练集分组数为{}, 验证级分组数为{}".format(len(train_data_dataloader), len(valid_data_dataloader)))



