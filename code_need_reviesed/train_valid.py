# This part is to read the data and separate the training set and validation set
from pathlib import Path
from original_data_preprocessing import *
from sklearn.model_selection import train_test_split


def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def read_data():
    DATA_PATH = Path("data")
    data_dict_x = {}
    data_dict_y = {}
    for i in range(5):
        x_name = 'data_dict_x_part_{}.pkl.gz'.format(i+1)
        y_name = 'data_dict_y_part_{}.pkl.gz'.format(i+1)
        PATH_X = DATA_PATH / x_name
        PATH_Y = DATA_PATH / y_name
        data_dict_x_part = pickle.loads(gzip.decompress(open(PATH_X, 'rb').read()))
        data_dict_y_part = pickle.loads(gzip.decompress(open(PATH_Y, 'rb').read()))
        print("第{}部分中x和y数据集分别使用的交易日天数为：{}, {}".format(i+1, len(data_dict_x_part), len(data_dict_y_part)))
        data_dict_x = Merge(data_dict_x, data_dict_x_part)
        data_dict_y = Merge(data_dict_y, data_dict_y_part)
    print("x和y数据集分别使用的交易日天数为：{}, {}".format(len(data_dict_x), len(data_dict_y)))

    # handle x
    data_numpy_x = np.empty(shape=(0, 270))
    for each in data_dict_x.values():
        data_numpy_x = np.concatenate((data_numpy_x, each.reshape(each.shape[0], -1)), axis=0)
        # print(data_numpy_x.shape)
    print('part 1 finish!')

    # handle y
    data_numpy_y = np.array([])
    for each in data_dict_y.values():
        data_numpy_y = np.concatenate((data_numpy_y, each.reshape(-1)), axis=0)
    print('part 2 finish!')

    # handle nan in y data
    nan_list = list(map(lambda x: x[0], np.argwhere(np.isnan(data_numpy_y)).tolist()))
    data_numpy_x = np.delete(data_numpy_x, nan_list, axis=0)
    data_numpy_y = np.delete(data_numpy_y, nan_list, axis=0)
    print('part 3 finish!')

    # handle over value data in y data
    over_list = list(map(lambda x: x[0], np.argwhere((data_numpy_y > 10) | (data_numpy_y < -10)).tolist()))
    data_numpy_x = np.delete(data_numpy_x, over_list, axis=0)
    data_numpy_y = np.delete(data_numpy_y, over_list, axis=0)
    print('part 4 finish!')

    assert data_numpy_x.shape[0] == data_numpy_y.shape[0]
    print("x数据的大小为：{}".format(data_numpy_x.shape))
    print("y数据的大小为：{}".format(data_numpy_y.shape))

    x_train, x_test, y_train, y_test = train_test_split(data_numpy_x, data_numpy_y, train_size=0.5, shuffle=False)
    # x_train, x_test, y_train, y_test = map(torch.tensor, (x_train, x_test, y_train, y_test))
    return x_train, x_test, y_train, y_test


read_data_x = pickle.loads(gzip.decompress(open('dict_train_valid_x.pkl.gz', 'rb').read()))
read_data_y = pickle.loads(gzip.decompress(open('dict_train_valid_y.pkl.gz', 'rb').read()))
print(read_data_x)
print(read_data_y)