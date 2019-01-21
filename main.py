import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


def calculate_log_return(data):
    print('clr')
    data['pre_close'] = data['close'].shift(-1).copy()
    last_stock = data.loc[0, 'stock_symbol']
    for row in range(data.shape[0]):
        # if row % 1000 == 0:
        # print('the current row is %d' % row)
        if data.loc[row, 'stock_symbol'] != last_stock:
            data.loc[row - 1, 'pre_close'] = data.loc[row - 1, 'open']
            last_stock = data.loc[row, 'stock_symbol']
    data['log_return'] = np.log(data['close'] / data['pre_close'])
    return data


def calculate_upward_fluctuation_ratio(data):
    print('start cufr')
    calculate_log_return(data)
    upper = 0
    lower = 0
    final_result = {}
    stock_categoty = data.drop_duplicates(['stock_symbol'])
    for i in stock_categoty['stock_symbol']:
        for j in range(data[data['stock_symbol'] == i].shape[0]):
            row_data = data[data['stock_symbol'] == i].iloc[j,]
            lower += np.power(row_data['log_return'], 2)
            if row_data['log_return'] > 0:
                upper += np.power(row_data['log_return'], 2)
        if lower != 0:
            result = upper / lower
        else:
            result = np.nan
        final_result[i] = result
        upper = 0
        lower = 0
    return final_result


def calculate_volume_ratio(data):
    print('start cvr')
    upper = 0
    lower = 0
    final_result = {}
    stock_categoty = data.drop_duplicates(['stock_symbol'])
    for i in stock_categoty['stock_symbol']:
        for j in range(data[data['stock_symbol'] == i].shape[0]):
            row_data = data[data['stock_symbol'] == i].iloc[j,]
            if row_data['time'] in ['09:45:00', '10:00:00']:
                upper += row_data['volume']
            if row_data['time'] in ['13:15:00', '13:30:00']:
                lower += row_data['volume']
        if lower != 0:
            result = upper / lower
        else:
            result = np.nan
        final_result[i] = result
        upper = 0
        lower = 0
    return final_result


def calculate_rskew(data):
    print('start rskew')
    data = calculate_log_return(data)
    upper = 0
    lower = 0
    final_result = {}
    stock_categoty = data.drop_duplicates(['stock_symbol'])
    for i in stock_categoty['stock_symbol']:
        ave_list = data[data['stock_symbol'] == i]['log_return']
        average = np.mean(ave_list)
        for j in range(data[data['stock_symbol'] == i].shape[0]):
            row_data = data[data['stock_symbol'] == i].iloc[j,]
            upper += 4 * np.power(row_data['log_return'] - average, 3)
            lower += np.power(row_data['log_return'] - average, 2)
        if lower != 0:
            result = upper / np.power(lower, 3 / 2)
        else:
            result = np.nan
        final_result[i] = result
        upper = 0
        lower = 0
    return final_result


def add_factor(factor_dictionary, dataframe, factor_name):
    print('add_factor')
    for row in range(dataframe.shape[0]):
        try:
            dataframe.ix[row, factor_name] = factor_dictionary[dataframe.ix[row, 'stock_symbol']]
        except:
            dataframe.ix[row, factor_name] = 'none'
    return dataframe



def run():
    train_data = pd.read_csv('train_high_frequency_data.csv', converters={'stock_symbol': str, 'date': str})
    test_data = pd.read_csv('test_high_frequency_data.csv', converters={'stock_symbol': str, 'date': str})
    train_output = pd.read_csv('train_output.csv', converters={'stock_symbol': str})
    test_output = pd.read_csv('test_output_template.csv', converters={'stock_symbol': str})

    factor1_dictionary_train = calculate_upward_fluctuation_ratio(train_data)
    train_output = add_factor(factor1_dictionary_train, train_output, 'factor1')
    factor1_dictionary_test = calculate_upward_fluctuation_ratio(test_data)
    test_output = add_factor(factor1_dictionary_test, test_output, 'factor1')

    factor2_dictionary_train = calculate_volume_ratio(train_data)
    train_output = add_factor(factor2_dictionary_train, train_output, 'factor2')
    factor2_dictionary_test = calculate_volume_ratio(test_data)
    test_output = add_factor(factor2_dictionary_test, test_output, 'factor2')

    factor3_dictionary_train = calculate_rskew(train_data)
    train_output = add_factor(factor3_dictionary_train, train_output, 'factor3')
    factor3_dictionary_test = calculate_rskew(test_data)
    test_output = add_factor(factor3_dictionary_test, test_output, 'factor3')

    train_output.dropna(axis=0, inplace=True)
    train_data_X = train_output[['factor1', 'factor2', 'factor3']]
    train_data_y = train_output[['label']].values.reshape((-1, ))

    # RF
    param_grid_rf = dict(
        n_estimators=[10, 50, 100, 200, 500],
        max_depth=[1, 2, 3],
        min_samples_leaf=[5, 10, 20]
    )

    grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=10, scoring='accuracy', n_jobs=-1)
    grid_rf.fit(train_data_X, train_data_y)
    print(grid_rf.best_estimator_)
    cv_result_rf = pd.DataFrame.from_dict(grid_rf.cv_results_)
    cv_result_rf.to_csv('cv_result_rf.csv', index=None)

    # SVC
    param_grid_svc = dict(
        C=[0.1, 1, 10],
        gamma=[1, 0.1, 0.01]
    )

    grid_svc = GridSearchCV(SVC(), param_grid_svc, cv=10, scoring='accuracy', n_jobs=-1)
    grid_svc.fit(train_data_X, train_data_y)
    print(grid_svc.best_estimator_)
    cv_result_svc = pd.DataFrame.from_dict(grid_svc.cv_results_)
    cv_result_svc.to_csv('cv_result_svc.csv', index=None)

    # MLP
    param_grid_mlp = dict(
        solver=['adam','lbfgs','sgd'],
        alpha=[0.01, 0.001, 0.0001],
        hidden_layer_sizes = [(10,10),(20,10),(20,20)]
    )

    grid_mlp = GridSearchCV(MLPClassifier(), param_grid_mlp, cv=10, scoring='accuracy', n_jobs=-1)
    grid_mlp.fit(train_data_X, train_data_y)
    print(grid_mlp.best_estimator_)
    cv_result_mlp = pd.DataFrame.from_dict(grid_mlp.cv_results_)
    cv_result_mlp.to_csv('cv_result_mlp.csv', index=None)

    # XGB
    param_grid_xgb = dict(
        max_depth=[2, 4, 6],
        min_child_weight=[3, 4, 5]
    )

    grid_xgb = GridSearchCV(XGBClassifier(objective="multi:softprob"), param_grid_xgb, cv=10, scoring='accuracy', n_jobs=-1)
    grid_xgb.fit(train_data_X, train_data_y)
    print(grid_xgb.best_estimator_)
    cv_result_xgb = pd.DataFrame.from_dict(grid_xgb.cv_results_)
    cv_result_xgb.to_csv('cv_result_xgb.csv', index=None)


    # best model prediction
    test_data_X = test_output[['factor1', 'factor2', 'factor3']]
    test_data_X['factor1'].fillna(test_data_X['factor1'].median(), inplace=True)
    test_data_X['factor2'].fillna(test_data_X['factor2'].median(), inplace=True)
    test_data_X['factor3'].fillna(test_data_X['factor3'].median(), inplace=True)

    # best model
    grid_best_model = XGBClassifier(objective='multi:softprob', min_child_weight=4,max_depth=2,num_class = 3)
    grid_best_model.fit(train_data_X,train_data_y)
    prediction_best_model= grid_best_model.predict(test_data_X)

    # save result to csv
    test_output['label'] = prediction_best_model
    test_output['probability'] = prediction_best_model
    test_output.to_csv('test_output_result.csv', index=None)



if __name__ == "__main__":
    run()
