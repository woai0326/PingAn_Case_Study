import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date
from  datetime import time
from datetime import timedelta
from main import calculate_log_return
from trend_deviation_factor import validate_market_data
from trend_deviation_factor import validate_stock_data
from trend_deviation_factor import duplicate_market_data


def calculate_ratio_with_market(stock_data, market_data):
    print('crwm')
    a = validate_market_data(market_data)
    b = validate_stock_data(stock_data)
    b = b.reindex(range(b.shape[0]))
    c = duplicate_market_data(a, b.shape[0] // a.shape[0])
    b['ratio_with_market'] = b['log_return'] / c['log_return']
    return b


def calculate_mean_std(data, columns, duration):
    mean = np.mean(data[str(date.today()-timedelta(days=duration)):str(date.today())][columns])
    std = np.std(data[str(date.today()-timedelta(days=duration)):str(date.today())][columns])
    return mean, std


def calculate_strange_time_ratio_factor(stock_data, columns, lamda):
    print('start strange time')
    valid_data = pd.DataFrame()
    strange_time_ratio = {}
    stock_category = stock_data.drop_duplicates(['stock_symbol'])
    for i in stock_category['stock_symbol']:
        for j in range(stock_data[stock_data['stock_symbol'] == i].shape[0]):
            row_data = stock_data[stock_data['stock_symbol'] == i].iloc[j, ]
            if (row_data[columns] > lamda) | (row_data[columns] < 0) == True:
                valid_data = valid_data.append(row_data)
        if stock_data[stock_data['stock_symbol'] == i].shape[0] != 0:
            result = valid_data.shape[0] / stock_data[stock_data['stock_symbol'] == i].shape[0]
        else:
            result = np.nan
        strange_time_ratio[i] = result
        valid_data = pd.DataFrame()
    return strange_time_ratio


def calculate_strange_volume_ratio_factor(stock_data, columns, lamda):
    print('start strange volume')
    valid_data = pd.DataFrame()
    strange_volume_ratio = {}
    stock_category = stock_data.drop_duplicates(['stock_symbol'])
    for i in stock_category['stock_symbol']:
        for j in range(stock_data[stock_data['stock_symbol'] == i].shape[0]):
            row_data = stock_data[stock_data['stock_symbol'] == i].iloc[j, ]
            if (row_data[columns] > lamda) | (row_data[columns] < 0) == True:
                valid_data = valid_data.append(row_data)
        if (sum(stock_data[stock_data['stock_symbol'] == i]['volume']) == 0) | (valid_data.empty) == True:
            result = np.nan
        else:
            result = sum(valid_data['volume']) / sum(stock_data[stock_data['stock_symbol'] == i]['volume'])
        strange_volume_ratio[i] = result
        valid_data = pd.DataFrame()
    return strange_volume_ratio


def calculate_strange_return_ratio_factor(stock_data, columns, lamda):
    print('start strange return')
    valid_data = pd.DataFrame()
    strange_return_ratio = {}
    stock_category = stock_data.drop_duplicates(['stock_symbol'])
    for i in stock_category['stock_symbol']:
        for j in range(stock_data[stock_data['stock_symbol'] == i].shape[0]):
            row_data = stock_data[stock_data['stock_symbol'] == i].iloc[j, ]
            if (row_data[columns] > lamda) | (row_data[columns] < 0) == True:
                valid_data = valid_data.append(row_data)
        if (sum(np.abs(stock_data[stock_data['stock_symbol'] == i]['log_return'])) == 0) | (valid_data.empty) == True:
            result = np.nan
        else:
            result = sum(np.abs(valid_data['log_return'])) / sum(
                np.abs(stock_data[stock_data['stock_symbol'] == i]['log_return']))
        strange_return_ratio[i] = result
        valid_data = pd.DataFrame()
    return strange_return_ratio


def run():
    stock_data = pd.read_csv('/Users/ScShen/PycharmProjects/PingAn_Case_Study/test/train_high_frequency_data.csv',
                             converters={'stock_symbol': str, 'date': str})
    market_data = pd.read_csv('/Users/ScShen/PycharmProjects/PingAn_Case_Study/data/hs300_15min_2018-12-24.csv')

    stock_data = calculate_ratio_with_market(stock_data, market_data)
    stock_data = stock_data.dropna()

    strange_time_ratio = calculate_strange_time_ratio_factor(stock_data, 'ratio_with_market', 1.47)
    strange_volume_ratio = calculate_strange_volume_ratio_factor(stock_data, 'ratio_with_market', 1.47)
    strange_return_ratio = calculate_strange_return_ratio_factor(stock_data, 'ratio_with_market', 1.47)


if __name__ == "__main__":
    run()
