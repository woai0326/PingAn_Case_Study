import pandas as pd
import numpy as np
from trend_deviation_factor import validate_stock_data


def calculate_fluctuation_factor(stock_data):
    stock_data = validate_stock_data(stock_data)
    fluctuation_factor = {}
    stock_category = stock_data.drop_duplicates(['stock_symbol'])
    for i in stock_category['stock_symbol']:
        current_df = stock_data[stock_data['stock_symbol'] == i]
        std = np.std(current_df['log_return'])
        fluctuation_factor[i] = std
    return fluctuation_factor


def run():
    stock_data = pd.read_csv('/Users/ScShen/PycharmProjects/PingAn_Case_Study/test/train_high_frequency_data.csv',
                             converters={'stock_symbol': str, 'date': str})

    fluctuation_factor = calculate_fluctuation_factor(stock_data)



if __name__ == "__main__":
    run()
