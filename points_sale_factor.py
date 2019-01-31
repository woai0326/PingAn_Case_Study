import pandas as pd
import numpy as np
from trend_deviation_factor import validate_market_data
from trend_deviation_factor import validate_stock_data
from trend_deviation_factor import duplicate_market_data


def points_sale_volume_factor(stock_data, market_data):
    print('start points sale volume factor')
    a = validate_market_data(market_data)
    b = validate_stock_data(stock_data)
    b = b.reindex(range(b.shape[0]))
    c = duplicate_market_data(a, b.shape[0] // a.shape[0])
    b['points_sale_volume_factor'] = b['volume'] / c['volume']
    return b


def points_sale_amount_factor(stock_data, market_data):
    print('start points sale amount factor')
    a = validate_market_data(market_data)
    b = validate_stock_data(stock_data)
    b = b.reindex(range(b.shape[0]))
    c = duplicate_market_data(a, b.shape[0] // a.shape[0])
    b['points_sale_amount_factor'] = b['amount'] / c['volume']
    return b


def run():
    stock_data = pd.read_csv('/Users/ScShen/PycharmProjects/PingAn_Case_Study/test/train_high_frequency_data.csv',
                             converters={'stock_symbol': str, 'date': str})
    market_data = pd.read_csv('/Users/ScShen/PycharmProjects/PingAn_Case_Study/data/hs300_15min_2018-12-24.csv')

    stock_data = points_sale_volume_factor(stock_data, market_data)
    stock_data = points_sale_amount_factor(stock_data, market_data)


if __name__ == "__main__":
    run()
