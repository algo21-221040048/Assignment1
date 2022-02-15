# This part of code is used to cleaning and handling the original data
# input: original data
# output: data picture and y
import pandas as pd
import numpy as np
import pickle
import gzip


def read():
    trading_list = pd.read_csv('../OriginalData_csv/Trading_date.csv', header=0, index_col=False)
    trading_list = trading_list['trading_date'].tolist()

    x_data_without_return1 = pd.read_csv('../OriginalData_csv/x_data_without_return1.csv', header=0,
                                         index_col='trading_date')
    if 'Unnamed: 0' in x_data_without_return1.columns:
        x_data_without_return1 = x_data_without_return1.drop(['Unnamed: 0'], axis='columns')
    x_data_without_return1.dropna(how='any', inplace=True)

    x_y_calculateReturn_data = pd.read_csv('../OriginalData_csv/x_y_calculateReturn_data.csv', header=0,
                                           index_col='trading_date')
    if 'Unnamed: 0' in x_y_calculateReturn_data.columns:
        x_y_calculateReturn_data = x_y_calculateReturn_data.drop(['Unnamed: 0'], axis='columns')
    x_y_calculateReturn_data.dropna(how='any', inplace=True)

    filter_data = pd.read_csv('../OriginalData_csv/filter_data.csv', header=0, index_col='trading_date',
                              low_memory=False)
    if 'Unnamed: 0' in filter_data.columns:
        filter_data = filter_data.drop(['Unnamed: 0'], axis='columns')

    delist_data = pd.read_csv('../OriginalData_csv/delist_data.csv', header=0, low_memory=False)
    if 'Unnamed: 0' in delist_data.columns:
        delist_data = delist_data.drop(['Unnamed: 0'], axis='columns')
    delist_data.dropna(how='any', inplace=True)

    return trading_list, x_data_without_return1, x_y_calculateReturn_data, filter_data, delist_data


def calculate_return1(df: pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()  # returning a view versus a copy
    df_new['close_adj'] = df_new.groupby('trading_date').apply(lambda x: x['close_price'] * x['adj_factor']).reset_index(drop=True)
    df_new['return1'] = df_new['close_adj'].pct_change()
    df_new = df_new.drop(['close_adj', 'adj_factor'], axis='columns')
    return df_new


if __name__ == '__main__':
    trading_list, x_data_without_return1, x_y_calculateReturn_data, filter_data, delist_data = read()
    data = pd.merge(x_data_without_return1.reset_index(), x_y_calculateReturn_data.reset_index(), how='left',
                    on=['trading_date', 'wind_code', 'close_price', 'adj_factor'])
    data = pd.merge(data, filter_data.reset_index(), how='left', on=['trading_date', 'wind_code'])
