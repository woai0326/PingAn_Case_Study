import pandas as pd
import numpy as np
from trend_deviation_factor import validate_market_data
from trend_deviation_factor import validate_stock_data
from trend_deviation_factor import duplicate_market_data
from sklearn.linear_model import LinearRegression
from pandas import Series


def calculate_lr_with_market_beta(stock_data, market_data):
    print('clwmb')
    stock_data = validate_stock_data(stock_data)
    market_data = validate_market_data(market_data)
    beta_factor = {}
    # alpha_factor = {}
    stock_category = stock_data.drop_duplicates(['stock_symbol'])
    for i in stock_category['stock_symbol']:
        current_df = stock_data[stock_data['stock_symbol'] == i]
        X = Series.as_matrix(market_data['log_return'])
        y = Series.as_matrix(current_df['log_return'])
        lr = LinearRegression()
        lr.fit(X.reshape(-1, 1), y)
        beta_factor[i] = lr.coef_[0]
        # alpha_factor[i] = lr.intercept_
    return beta_factor


def calculate_lr_with_market_alpha(stock_data, market_data):
    print('clwmb')
    stock_data = validate_stock_data(stock_data)
    market_data = validate_market_data(market_data)
    alpha_factor = {}
    stock_category = stock_data.drop_duplicates(['stock_symbol'])
    for i in stock_category['stock_symbol']:
        current_df = stock_data[stock_data['stock_symbol'] == i]
        X = Series.as_matrix(market_data['log_return'])
        y = Series.as_matrix(current_df['log_return'])
        lr = LinearRegression()
        lr.fit(X.reshape(-1, 1), y)
        alpha_factor[i] = lr.intercept_
    return alpha_factor


def run():
    stock_data = pd.read_csv('/Users/ScShen/PycharmProjects/PingAn_Case_Study/test/train_high_frequency_data.csv',
                             converters={'stock_symbol': str, 'date': str})
    market_data = pd.read_csv('/Users/ScShen/PycharmProjects/PingAn_Case_Study/data/hs300_15min_2018-12-24.csv')

    beta_factor = calculate_lr_with_market_beta(stock_data, market_data)
    alpha_factor = calculate_lr_with_market_alpha(stock_data, market_data)


if __name__ == "__main__":
    run()