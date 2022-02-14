# This part of code is used to download the original data from the database
import json
from typing import Dict
from urllib.parse import parse_qs, urlparse
import pandas as pd
import requests


host = '192.168.1.114'
port = 30300


def get_data(params: Dict, url=f'http://{host}:{port}/tqmain/equity_daily', res=None) -> pd.DataFrame:
    '''
    主要获取DB中的数据
    :param params: GET方法需要的query string，字典参数
    :param url: 资源的url地址
    :param res: 循环获取分页时会用到
    :return:
    '''
    if not res:
        res = []
    assert not urlparse(url).params, 'url中不要放入query string'

    r = requests.get(
        url,
        params=params)
    assert r.status_code == 200, f'请求失败： {r.status_code} {r.text}'
    j = json.loads(r.text)

    res.extend(j['results'])

    if j['next']:  # 如果有下一页，递归获取直至无下一页
        query = urlparse(j['next']).query
        page = parse_qs(query).get('page')[0]
        params['page'] = page
        return get_data(
            url=url,
            res=res,
            params=params
        )

    res = pd.DataFrame(res)
    return res


def get_tick(params: Dict,  url: str) -> pd.DataFrame:  # url 默认获取股票日度表
    '''
    主要获取以文件形式存放的tick数据
    :param params: GET方法需要的query string，字典参数
    :param url: 资源的url地址
    :return:
    '''
    assert not urlparse(url).params, 'url中不要放入query string'

    r = requests.get(
        url,
        params=params)
    assert r.status_code == 200, f'请求失败： {r.status_code} {r.text}'
    j = json.loads(r.text)
    columns = j['columns'].split(',')
    if not j['data']:
        return pd.DataFrame()
    data = j['data'].split('\n')
    data = [i.split(',') for i in data]

    res = pd.DataFrame(data, columns=columns)
    return res


def get_token(usr: str, pwd: str) -> str:
    """获取用户token"""
    url = f'http://{host}:{port}/tq_user/login'
    r = requests.post(url, data={
        "username": usr,
        "password": pwd
    })
    j = json.loads(r.text)
    assert 'error' not in j, f"error: {j['error']}"
    assert 'token' in j, f"error: {j}"
    return j['token']


def is_trading_date(dt='2021-01-01', exchange_cn='上海证券交易所') -> bool:
    url = f"http://{host}:{port}/tqmain/trading_date_info/"
    params = {
        "nature_date": dt,
        "exchange_cn": exchange_cn,
    }
    r = requests.get(url, params=params)
    j = json.loads(r.text)
    return j['results'][0]['is_trading_date']


def get_authenticate_data(token: str) -> pd.DataFrame:
    '''用token访问tq_model数据'''
    url = f'http://{host}:{port}/tq_model/lgt_factor/?trading_date=2021-04-08'
    header = {'Authorization': f'Token {token}'}  # 把token放入header中
    r = requests.get(url, headers=header)
    assert r.status_code == 200, f'error: {r.status_code}; {r.text}'
    j = json.loads(r.text)
    df = pd.DataFrame(j['results'])
    return df


def get_user_id_by_token(token: str) -> Dict:
    url = f'http://{host}:{port}/tq_user/validate_token'
    data = {'token': token}
    r = requests.post(url, data=data)
    assert r.status_code == 200, f'error: {r.status_code}; {r.text}'
    j = json.loads(r.text)
    return j


def get_original_data(start_date: str, end_date: str, url=f'http://{host}:{port}/tqmain/equity_daily', wind_code=None) -> pd.DataFrame:
    params = {}
    params['wind_code'] = wind_code
    params['start_date'] = start_date
    params['end_date'] = end_date
    params['fields'] = 'trading_date,wind_code,open_price,high_price,low_price,close_price,vwap,volume,turn,free_turn,adj_factor'
    df = get_data(params=params, url=url)
    df.sort_values(by='trading_date', inplace=True)
    df = df.reset_index()
    if 'index' in df.columns:
        df = df.drop(['index'], axis='columns')
    # df.dropna(how='any', inplace=True)
    return df


def get_filter_data(start_date: str, end_date: str, url=f'http://{host}:{port}/tqmain/equity_daily', wind_code=None) -> pd.DataFrame:
    params = {}
    params['wind_code'] = wind_code
    params['start_date'] = start_date
    params['end_date'] = end_date
    params['fields'] = 'trading_date,wind_code,is_st,trade_status,susp_days,susp_reason,max_up_or_down'
    df = get_data(params=params, url=url)
    df.sort_values(by='trading_date', inplace=True)
    df = df.reset_index()
    if 'index' in df.columns:
        df = df.drop(['index'], axis='columns')
    return df


def get_return_data(start_date: str, end_date: str, url=f'http://{host}:{port}/tqmain/equity_daily', wind_code=None) -> pd.DataFrame:
    params = {}
    params['wind_code'] = wind_code
    params['start_date'] = start_date
    params['end_date'] = end_date
    params['fields'] = 'trading_date,wind_code,close_price,adj_factor'
    df = get_data(params=params, url=url)
    df.sort_values(by='trading_date', inplace=True)
    df = df.reset_index()
    if 'index' in df.columns:
        df = df.drop(['index'], axis='columns')
    return df


def get_delist_data(delist_date=None, url=f'http://{host}:{port}/tqmain/equity_header', wind_code=None):
    params = {}
    params['wind_code'] = wind_code
    params['delist_date'] = delist_date
    params['fields'] = 'wind_code,delist_date'
    df = get_data(params=params, url=url)
    return df


if __name__ == '__main__':
    trading_list = pd.read_csv('Trading_date.csv', header=0, index_col=False)
    trading_list = trading_list['trading_date'].tolist()

    get_original_data(trading_list[0], trading_list[-1]).to_csv("x_data_without_return1.csv", header=True, index=True)
    print("Finish!")

    get_filter_data(trading_list[0], trading_list[-1]).to_csv("filter_data.csv", header=True, index=True)
    print("Finish!")

    get_return_data(trading_list[0], trading_list[-1]).to_csv("x_y_calculateReturn_data.csv", header=True, index=True)
    print("Finish!")

    get_delist_data().to_csv("delist_data.csv", header=True, index=True)
    print("Finish!")





