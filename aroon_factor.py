import pandas as pd
import numpy as np


def calculate_aroon_up_factor(stock_data):
    stock_category = stock_data.drop_duplicates(['stock_symbol'])
    aroon_up_factor = []
    for i in stock_category['stock_symbol']:
        current_df = stock_data[stock_data['stock_symbol'] == i]
        for j in range(current_df.shape[0]):
            aroon_day = current_df.shape[0] - (j - np.where(current_df == np.max(current_df['close']))[0][0])
            if current_df.shape[0] != 0:
                value = aroon_day / current_df.shape[0]
            else:
                value = np.nan
            aroon_up_factor.append(value)
    stock_data['aroon_up_factor'] = aroon_up_factor
    return stock_data


def calculate_aroon_down_factor(stock_data):
    stock_category = stock_data.drop_duplicates(['stock_symbol'])
    aroon_down_factor = []
    for i in stock_category['stock_symbol']:
        current_df = stock_data[stock_data['stock_symbol'] == i]
        for j in range(current_df.shape[0]):
            aroon_day = current_df.shape[0] - (j - np.where(current_df == np.min(current_df['close']))[0][0])
            if current_df.shape[0] != 0:
                value = aroon_day / current_df.shape[0]
            else:
                value = np.nan
            aroon_down_factor.append(value)
    stock_data['aroon_down_factor'] = aroon_down_factor
    return stock_data


def run():
    stock_data = pd.read_csv('/Users/ScShen/PycharmProjects/PingAn_Case_Study/test/train_high_frequency_data.csv',
                             converters={'stock_symbol': str, 'date': str})
    stock_data = stock_data.dropna()

    stock_data = calculate_aroon_up_factor(stock_data)
    stock_data = calculate_aroon_down_factor(stock_data)


if __name__ == "__main__":
    run()