import pandas as pd
import numpy as np
import sys
import getopt
from main import calculate_log_return

# def calculate_stock_pearson(data,market):
#     print('cal pearson')
#     calculate_log_return(data)
#     n = len(data)
#     sum1 = 0
#     sum2 = 0
#     sum1_pow = 0
#     sum2_pow = 0
#     p_sum = 0
#     final_result = {}
#     stock_category = data.drop_duplicates(['stock_symbol'])
#     for i in stock_category['stock_symbol']:
#         for j in range(data[data['stock_symbol'] == i].shape[0]):
#             row_data = data[data['stock_symbol'] == i].iloc[j,]
#             # simple sums & square
#             sum1 += row_data['log_return']
#             sum2 += market['log_return']
#             sum1_pow += np.power(row_data['log_return'], 2)
#             sum2_pow += np.power(market['log_return'], 2)
#             # sum up the products
#             p_sum += row_data['log_return'] * market['log_return']
#         num = p_sum - (sum1 * sum2 / n)
#         den = np.sqrt((sum1_pow - np.power(sum1, 2) / n) * (sum2_pow - np.power(sum2, 2) / n))
#         if den == 0:
#             return 0
#         else:
#             final_result[i] = num / den
#     return final_result


# jieximinlinghang ??????
def order_name(order):
    opts, argv = getopt.getopt(sys.argv[1:], 'n:t:s:e:h', ['start=', 'end=', 'help'])
    for opts_name, opts_value in opts:
        if opts_name == '-n':
            code = opts_value
        if opts_name == '-t':
            frequency = opts_value
        if opts_name in ['-s', '--start']:
            starttime = opts_value
        if opts_name in ['-e', '--end']:
            endtime = opts_value
        if opts_name in ['-h', '--help']:
            print('python factor.py -n hs300')
            sys.exit()


# data clean & calculate log_return in stock data
def validate_stock_data(stock_data):
    calculate_log_return(stock_data)
    stock_data = stock_data.fillna(value=stock_data.loc[np.where(pd.isna(stock_data))[0][0], 'close'] / stock_data.loc[
        np.where(pd.isna(stock_data))[0][0], 'open'])
    frequency_dict = {
        '1min': 240,
        '5min': 48,
        '10min': 24,
        '15min': 16,
        '60min': 4
    }
    drop_list = []
    stock_set = set(stock_data['stock_symbol'])
    stock_list = list(stock_data['stock_symbol'])
    for stock_symbol in stock_set:
        if stock_list.count(stock_symbol) != frequency_dict['15min']:
            drop_list.append(stock_symbol)
    result = stock_data[list(map(lambda x: False if x in drop_list else True, stock_data['stock_symbol']))]
    # if (np.any(result['log_return'].isna()) is True) == 1:
    #     result.loc[np.where(pd.isna(result))[0][0], 'log_return'] = result.loc[np.where(
    #         pd.isna(result))[0][0], 'close'] / result.loc[np.where(pd.isna(result))[0][0], 'open']
    return result


# data clean & calculate log_return in market data
def validate_market_data(market_data):
    market_data['log_return'] = np.log(market_data['close'] / market_data['close'].shift(-1))
    market_data = market_data.fillna(value=market_data.loc[np.where(pd.isna(market_data))[0][0], 'close'] /
                                            market_data.loc[np.where(pd.isna(market_data))[0][0], 'open'])
    before_dup = market_data[list(map(lambda x: True if x.find('2018-12-25') >= 0 else False, market_data['date']))]
    return before_dup


# duplicate market data
def duplicate_market_data(before_dup, round_number):
    matched_market_data = pd.concat([before_dup for i in range(round_number)], axis=0, ignore_index=True)
    return matched_market_data


def calculate_pearson(stock_data, market_data):
    print('cal pearson')
    result = np.corrcoef(stock_data, market_data)
    return result[0, 1]


def run():
    stock_data = pd.read_csv('/Users/ScShen/PycharmProjects/PingAn_Case_Study/test/train_high_frequency_data.csv',
                             converters={'stock_symbol': str, 'date': str})
    market_data = pd.read_csv('/Users/ScShen/PycharmProjects/PingAn_Case_Study/data/hs300_15min_2018-12-24.csv')

    a = validate_stock_data(stock_data)
    b = validate_market_data(market_data)
    round_number = a.shape[0] // b.shape[0]
    c = duplicate_market_data(b, round_number)
    pearson_coef = calculate_pearson(a['log_return'], c['log_return'])


if __name__ == "__main__":
    run()
