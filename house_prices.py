import pandas as pd
from house_prices_functions import *
from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

full_train_data = pd.read_csv(r'C:\Users\M\OneDrive - Carleton University\Documents\my_stuff\Projects\Kaggle_House_Prices\train.csv')
test_data = pd.read_csv(r'C:\Users\M\OneDrive - Carleton University\Documents\my_stuff\Projects\Kaggle_House_Prices\test.csv')

full_features = full_train_data.drop(['Id', 'SalePrice'], axis=1)
test_data = test_data.drop(['Id'], axis=1)
target = full_train_data.SalePrice

# model = GradientBoostingRegressor(random_state=42)  ## tried a dew basic models, this performed best, this will act as baseline

# model =  MLPRegressor(random_state=42, hidden_layer_sizes=(10000,5000,1000,500,100,20), max_iter=100000, solver='adam', alpha=0.0001, activation='relu', learning_rate='adaptive', learning_rate_init=0.0001)


xgb_params = dict(
    max_depth=6,           # maximum depth of each tree - try 2 to 10
    learning_rate=0.01,    # effect of each tree - try 0.0001 to 0.1
    n_estimators=1000,     # number of trees (that is, boosting rounds) - try 1000 to 8000
    min_child_weight=1,    # minimum number of houses in a leaf - try 1 to 10
    colsample_bytree=0.7,  # fraction of features (columns) per tree - try 0.2 to 1.0
    subsample=0.7,         # fraction of instances (rows) per tree - try 0.2 to 1.0
    reg_alpha=0.5,         # L1 regularization (like LASSO) - try 0.0 to 10.0
    reg_lambda=1.0,        # L2 regularization (like Ridge) - try 0.0 to 10.0
    num_parallel_tree=1,   # set > 1 for boosted random forests
)

model = XGBRegressor(**xgb_params)


#############################################################################


# train_set_components = preprocess_function(full_features)
# test_set_components = preprocess_function(test_data)

## this does about the same as old preprocessing func, a little better depending on model
train_set_components = alternate_preprocess(full_features)
test_set_components = alternate_preprocess(test_data)


good_num_features = feature_selection(train_set_components[0], train_set_components[1], target)


make_sklearn_pipeline(train_set_components, target, test_set_components, good_num_features, model, n_xvalid=5, train_or_test=0)

# to get test data predictions, just need to change that last 0 to a 1, to put it in testmode
# test_predictions = make_sklearn_pipeline(train_set_components, target, test_set_components, good_num_features, model, n_xvalid=5, train_or_test=1)

# print (test_predictions.shape)
# print (test_predictions)

# test_predictions.to_csv(r'C:\Users\M\OneDrive - Carleton University\Documents\my_stuff\Projects\Kaggle_House_Prices\predicted_prices_submission_attempt4_xgbregressor.csv')


