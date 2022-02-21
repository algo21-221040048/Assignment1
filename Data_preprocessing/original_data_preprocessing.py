# This part of code is used to cleaning and handling the original data
# Input: original data
# Output: data picture and y
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as f
import pickle
import gzip


# Hyper parameters
TRAIN_UPDATE_DAYS = 122
FREQUENCY = 2
PERIOD = 5


# This function is to read in the original data
def read():
    trading_list = pd.read_csv('../OriginalData_csv/Trading_date.csv', header=0, index_col=False)
    trading_list = trading_list['trading_date'].tolist()

    x_data_without_return1 = pd.read_csv('../OriginalData_csv/x_data_without_return1.csv', header=0,
                                         index_col='trading_date')
    if 'Unnamed: 0' in x_data_without_return1.columns:
        x_data_without_return1 = x_data_without_return1.drop(['Unnamed: 0'], axis='columns')

    filter_data = pd.read_csv('../OriginalData_csv/filter_data.csv', header=0, index_col='trading_date',
                              low_memory=False)
    if 'Unnamed: 0' in filter_data.columns:
        filter_data = filter_data.drop(['Unnamed: 0'], axis='columns')

    delist_data = pd.read_csv('../OriginalData_csv/delist_data.csv', header=0, low_memory=False)
    if 'Unnamed: 0' in delist_data.columns:
        delist_data = delist_data.drop(['Unnamed: 0'], axis='columns')
    delist_data.dropna(how='any', inplace=True)

    return trading_list, x_data_without_return1, filter_data, delist_data


# This function is to calculate the column return1
def calculate_return1(df: pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()  # returning a view versus a copy
    df_new['close_adj'] = df_new.groupby('trading_date').apply(lambda x: x['close_price'] * x['adj_factor']).reset_index(drop=True)
    df_new['return1'] = (df_new.groupby('wind_code'))['close_adj'].apply(lambda x: x.pct_change(fill_method=None)).reset_index(drop=True)
    df_new = df_new.drop(['adj_factor'], axis='columns')
    return df_new


# This function must be used after `calculate_return1` function
def calculate_return_bn(df: pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()
    df_new['return5'] = (df_new.groupby('wind_code'))['close_adj'].apply(lambda x: x.pct_change(periods=PERIOD, fill_method=None)).reset_index(drop=True)
    df_new['return5'] = (df_new.groupby('wind_code'))['return5'].shift(periods=-PERIOD)
    df_new['return_bn'] = (df_new.groupby('trading_date'))['return5'].transform(lambda x: (x - x.mean())/x.std(ddof=0))
    return df_new


# This function is used to filter the ST and PT stocks
def data_cleaning(df: pd.DataFrame, delist: pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()
    # find st
    st = df_new[df_new['is_st'] == 1.0]['wind_code'].unique().tolist()

    # find pt
    stock_delist_date = delist[delist['delist_date'] != '1899-12-30']
    stock_delist_date = stock_delist_date[
        (stock_delist_date['delist_date'] <= '2020-05-29') & (stock_delist_date['delist_date'] >= '2011-01-31')]
    pt = stock_delist_date['wind_code'].unique().tolist()

    # delete st, pt stocks
    df_new = df_new[~df_new['wind_code'].isin(set(st + pt))]
    return df_new.reset_index(drop=True)


# This function is used to fill NaN by forward filling method
def data_fillingNa(df: pd.DataFrame) -> pd.DataFrame:
    medium = df[['wind_code', 'open_price', 'high_price', 'low_price', 'close_price', 'vwap']]
    medium = medium.groupby('wind_code').fillna(method='ffill', axis=0)
    return df.fillna(medium)


# This function is used to handle special case: '停牌一天'
def sp_handle(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[((df['trade_status'] == '停牌一天') & (~df['close_price'].isna())), "open_price":"close_price"] = np.nan
    return df


# This function is used to handle with the NaN in return1 and return5 group by wind_code
def processing(x: pd.DataFrame) -> pd.DataFrame:
    x = x.drop(['susp_days', 'susp_reason'], axis='columns')
    x = x.fillna(-1000)
    # x = x[~((x['return1'].isna()) | (x['return_bn'].isna()))].reset_index(drop=True)
    x = x.reset_index(drop=True)
    return x


def split_and_judging(x: pd.DataFrame) -> pd.DataFrame:
    pass


if __name__ == '__main__':
    trading_list, x_data_without_return1, filter_data, delist_data = read()
    data = pd.merge(x_data_without_return1.reset_index(), filter_data.reset_index(), how='left', on=['trading_date', 'wind_code'])

    # data = data_fillingNa(data)  # This step isn't necessary
    # # Test if the fillNaN method is reasonable
    # # data[data['wind_code'] == "000001.SZ"].to_csv("000001.SZ.fill.csv", header=True, index=True)

    # Data preparing
    data = data_cleaning(data, delist_data)
    data = sp_handle(data)
    data = calculate_return1(data)
    data = calculate_return_bn(data)
    data = data.groupby('wind_code').apply(lambda x: processing(x))
    # data = data.drop(['wind_code'], axis='columns')

    train_start_list = trading_list[:-1500:TRAIN_UPDATE_DAYS]
    train_end_list = []
    for each in train_start_list:
        train_end_list.append(trading_list[trading_list.index(each) + 1500])

    for i in range(len(train_start_list)):
        PATH_X = 'data_x_part_{}.pkl.gz'.format(i+1)
        PATH_Y = 'data_y_part_{}.pkl.gz'.format(i+1)
        data_x = torch.empty((0, 270))
        data_y = torch.tensor([])
        medium = data[(data['trading_date'] >= train_start_list[i]) & (data['trading_date'] < train_end_list[i])]
        for each in medium['wind_code'].unique().tolist():
            sub_df = medium.loc[each]
            sub_df_length = sub_df.shape[0]
            if sub_df_length < 30:
                continue
            else:
                # Handle x
                sub_df_x = np.array(sub_df[['open_price', 'high_price', 'low_price', 'close_price', 'vwap', 'volume', 'return1', 'turn', 'free_turn']]).T
                sub_df_x = torch.from_numpy(sub_df_x)

                # Split x
                split_sub_df_x = f.unfold(sub_df_x.unsqueeze(0).unsqueeze(0), kernel_size=(9, 30), stride=(1, FREQUENCY))
                B, W, L = split_sub_df_x.size()
                split_sub_df_x = split_sub_df_x.permute(0, 2, 1)
                split_sub_df_x = split_sub_df_x.squeeze(0)

                # Handle y
                sub_df_y = np.array(sub_df.iloc[29::FREQUENCY]['return5'])
                split_sub_df_y = torch.from_numpy(sub_df_y)

                assert split_sub_df_x.shape[0] == split_sub_df_y.shape[0]
                # filter x,y where x contains -1000
                truncate_index = np.unique(np.where(split_sub_df_x == -1000)[0])
                split_sub_df_x = np.delete(split_sub_df_x, truncate_index, 0)
                split_sub_df_y = np.delete(split_sub_df_y, truncate_index, 0)
                # filter x,y where y contains -1000
                truncate_index = np.unique(np.where(split_sub_df_y == -1000)[0])
                split_sub_df_x = np.delete(split_sub_df_x, truncate_index, 0)
                split_sub_df_y = np.delete(split_sub_df_y, truncate_index, 0)
                data_x = torch.cat((data_x.double(), split_sub_df_x), 0)
                data_y = torch.cat((data_y.double(), split_sub_df_y), 0)

        open(PATH_X, 'wb').write(gzip.compress(pickle.dumps(data_x)))
        open(PATH_Y, 'wb').write(gzip.compress(pickle.dumps(data_y)))

        print("Part {} finish !".format(i+1))






