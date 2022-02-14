# This part of code is used to cleaning and handling the original data
# input: original data
# output: data picture and y
import pandas as pd
import numpy as np
import pickle
import gzip


def calculate_return1(wind_code):
    df = data[data['wind_code'] == wind_code]
    df_new = df.copy()  # returning a view versus a copy
    df_new['close_adj'] = df_new.apply(lambda x: x['close_price'] * x['adj_factor'], axis=1).copy()
    df_new['return1'] = df_new['close_adj'].pct_change()
    df_new = df_new.drop(['close_adj', 'adj_factor'], axis='columns')
    return df_new
    # stock_code = data['wind_code'].unique().tolist()
    # data_with_return1 = pd.concat([calculate_return1(each) for each in stock_code], axis=0, ignore_index=False)
    # data_with_return1.sort_values(by='trading_date', inplace=True)
    # assert data_with_return1.shape == data.shape
    # data_with_return1.to_csv("data_with_return1.csv", header=True, index=True)
    # print("Finish!")


def calculate_return5(wind_code):
    df = y_data[y_data['wind_code'] == wind_code]
    df_new = df.copy()  # returning a view versus a copy
    df_new['close_adj'] = df_new.apply(lambda x: x['close_price'] * x['adj_factor'], axis=1).copy()
    df_new['return1'] = df_new['close_adj'].pct_change()
    df_new['std'] = df_new['return1'].rolling(5).std(ddof=0)  # this could be modified
    # df_new['std'] = df_new['std'].shift(periods=1)
    df_new['return5'] = df_new['close_adj'].pct_change(periods=5)
    df_new['std_return5'] = df_new['return5']/df_new['std']
    df_new = df_new.drop(['close_adj', 'adj_factor', 'return5', 'return1', 'std', 'close_price'], axis='columns')
    return df_new
    # stock_code = y_data['wind_code'].unique().tolist()
    # data_y_with_return5 = pd.concat([calculate_return5(each) for each in stock_code], axis=0, ignore_index=False)
    # data_y_with_return5.sort_values(by='trading_date', inplace=True)
    # assert data_y_with_return5.shape == data.shape
    # data_y_with_return5.to_csv("data_y_with_return5.csv", header=True, index=True)
    # print("Finish!")


def preprocessing():
    trading_list = pd.read_csv('Trading_date.csv', header=0, index_col=False)
    trading_list = trading_list['trading_date'].unique().tolist()

    data = pd.read_csv('original_data_table.csv', header=0, index_col='trading_date')
    if 'Unnamed: 0' in data.columns:
        data = data.drop(['Unnamed: 0'], axis='columns')

    filter_medium = pd.read_csv('filter_data_table.csv', header=0, index_col='trading_date', low_memory=False)
    if 'Unnamed: 0' in filter_medium.columns:
        filter_medium = filter_medium.drop(['Unnamed: 0'], axis='columns')

    data_with_return1 = pd.read_csv('data_with_return1.csv', header=0, index_col='trading_date', low_memory=False)
    if 'Unnamed: 0' in data_with_return1.columns:
        data_with_return1 = data_with_return1.drop(['Unnamed: 0'], axis='columns')

    y = pd.read_csv('original_y_data_table.csv', header=0, index_col='trading_date', low_memory=False)
    if 'Unnamed: 0' in y.columns:
        y = y.drop(['Unnamed: 0'], axis='columns')

    data_y = pd.read_csv('data_y_with_return5.csv', header=0, index_col='trading_date', low_memory=False)
    if 'Unnamed: 0' in data_y.columns:
        data_y = data_y.drop(['Unnamed: 0'], axis='columns')

    delist = pd.read_csv('delist_date_table.csv', header=0, low_memory=False)
    if 'Unnamed: 0' in delist.columns:
        delist = delist.drop(['Unnamed: 0'], axis='columns')

    return trading_list, data, filter_medium, data_with_return1, y, data_y, delist


def processing(start_date, end_date, trading_input, data_input, y_date):
    assert start_date in trading_input
    assert end_date in trading_input
    assert y_date in trading_input
    # end_date = trading_input[trading_input.index(end_date) - 1]

    data_medium = data_input.loc[start_date:end_date]
    trade_status_medium = trade_status.loc[start_date:end_date]
    st_medium = st.loc[start_date:end_date]
    st_medium_stock = pd.unique(st_medium.values).tolist()
    stock_code_needed_truncated = pd.unique(trade_status_medium.values).tolist()  # no trading during this period
    stock_code_needed_truncated += trade_status_no_data  # no trade data during this period
    stock_code_needed_truncated += st_medium_stock  # st during this period
    stock_code_needed_truncated += stock_delist_list

    for each in pd.unique(stock_code_needed_truncated).tolist():
        data_medium = data_medium[data_medium['wind_code'] != each]
    stock_code = data_medium['wind_code'].unique().tolist()
    for each in stock_code:
        if data_medium[data_medium['wind_code'] == each].shape[0] != 30:   # no full 30 day data during this period
            # print(each)
            data_medium = data_medium[data_medium['wind_code'] != each]
        elif data_medium[data_medium['wind_code'] == each].isnull().any().any():  # data picture of this code has none during this period
            # print(each)
            data_medium = data_medium[data_medium['wind_code'] != each]
        else:
            continue
    len_after = data_medium.shape[0]
    assert len_after % 30 == 0

    data_medium.sort_values(by=['wind_code', 'trading_date'], inplace=True)
    data_picture = data_medium.drop(['wind_code'], axis='columns')
    num = data_picture.shape[0] // 30
    data_numpy = np.empty((num, 9, 30))
    data_y_numpy = np.empty((num, 1))
    for i in range(num):
        # print(i)
        # handle x
        medium_array = data_picture.iloc[30 * i: 30 * (i + 1)].values
        assert data_numpy[i].shape == (9, 30)
        data_numpy[i] = medium_array.T

        # handle y
        medium = data_medium.iloc[30 * i: 30 * (i + 1)]
        assert len(medium['wind_code'].unique().tolist()) == 1
        stock_code_medium = medium['wind_code'].unique().tolist()[0]
        # print(stock_code_medium)
        _ = data_y_with_return5.loc[y_date]
        # print(_[_['wind_code'] == stock_code_medium])
        medium_y_array = _[_['wind_code'] == stock_code_medium]['std_return5'].values
        data_y_numpy[i] = medium_y_array
    return data_medium, data_numpy, data_y_numpy


def getting_data_pictures_dict(start, end):
    # get the data_picture and compress it into local
    data_dict_x = {}
    data_dict_y = {}
    for each in trading_date[start: end]:
        end_date_index = trading_date.index(each)
        if end_date_index % 2 != 0:
            continue
        y_date = trading_date[end_date_index + 5]  # this could be modified
        start_date = trading_date[end_date_index - 29]
        data_after_handle, data_picture_single_day, data_y_single_day = processing(start_date, each, trading_date, data_with_return1, y_date)
        print(data_picture_single_day.shape, data_y_single_day.shape)
        print('====={}to{}=========={}======='.format(start_date, each, "Finish!"))
        data_dict_x[each] = data_picture_single_day
        data_dict_y[each] = data_y_single_day
    return data_dict_x, data_dict_y


def save_and_read():
    for i in range(4, 5):
        x_name = 'data_dict_x_part_{}.pkl.gz'.format(i+1)
        y_name = 'data_dict_y_part_{}.pkl.gz'.format(i+1)
        if i != 4:
            print("=====Handling data from {} to {}=====".format(trading_date[30 + 500 * i], trading_date[30 + 500 * (i + 1) - 1]))
            data_dict_x, data_dict_y = getting_data_pictures_dict(30 + 500 * i, 30 + 500 * (i+1))
        else:
            print("=====Handling data from {} to {}=====".format(trading_date[30 + 500 * 4], trading_date[-7]))
            data_dict_x, data_dict_y = getting_data_pictures_dict(30 + 500 * 4, -7)
        open(x_name, 'wb').write(gzip.compress(pickle.dumps(data_dict_x)))
        open(y_name, 'wb').write(gzip.compress(pickle.dumps(data_dict_y)))
        print("Save Finish!")
        # read_data_x = pickle.loads(gzip.decompress(open(x_name, 'rb').read()))
        # read_data_y = pickle.loads(gzip.decompress(open(y_name, 'rb').read()))
        # print("Read Finish!")


if __name__ == '__main__':
    trading_date, data, filter_data, data_with_return1, y_data, data_y_with_return5, de_list = preprocessing()
    trading_sep_date = list(filter(None, [trading_date[1:][i] if i % 30 == 0 else None for i in range(len(trading_date[1:]))]))

    # select out the stocks which need to be filtered
    st = filter_data[filter_data['is_st'] == 1.0]
    st = st['wind_code']
    max_up_or_down = filter_data[(filter_data['max_up_or_down'] == 1) | (filter_data['max_up_or_down'] == -1)]
    max_up_or_down = max_up_or_down['wind_code']
    trade_status = filter_data[filter_data['trade_status'] != '交易']
    trade_status = trade_status['wind_code']
    trade_status_no_data = (filter_data[filter_data['trade_status'].isnull()])['wind_code'].unique().tolist()  # no data

    # these are PT
    stock_delist_date = de_list[de_list['delist_date'] != '1899-12-30']
    stock_delist_date = stock_delist_date[(stock_delist_date['delist_date'] <= '2020-05-29') & (stock_delist_date['delist_date'] >= '2011-01-31')]
    stock_delist_list = stock_delist_date['wind_code'].unique().tolist()

    save_and_read()







































