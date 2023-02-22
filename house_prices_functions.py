import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression


## don't use, doesnt improve performance
def alternate_preprocess(data_set):  ## just return all categorical to use OH encoding    
        ## columns with missing values
        cols_with_na = [col for col in data_set.columns if data_set[col].isnull().any()]
       
        removed_cols = []
       
        ## checking percentages of missing values per column, removing ones with > 50% missing, won't use those
        for col in cols_with_na:
            frac = data_set[col].isnull().sum()/len(data_set[col])
            # print ('{}: {}'.format(col, frac))
           
            if frac > 0.6: ## note this fraction is chosen such that it removes the same set of columns from both train & test data sets
                data_set = data_set.drop(col, axis=1)
                removed_cols.append(col)
                # print (col)
        
        ## update col_with_na
        cols_with_na = list(set(cols_with_na) - set(removed_cols))
           
       
        ## identifying all categorical, and numerical columns
        categorical_cols = [col for col in data_set.columns if data_set[col].dtype == "object"]
       
        numerical_cols = [col for col in data_set.columns if data_set[col].dtype in ['int64', 'float64']]
        numerical_cols = list(set(numerical_cols) - set(removed_cols))
       
        ## columns with ordinal categorical data, determined manually (so might contain some deleted colums) based on what makes sense as ordinal, specific to this dataset
        ordinal_cat_cols = []
       
        OH_cat_cols = categorical_cols
       
       
        ## getting rid of high cardinality OH categorical columns
        high_cardinality_cols = [col for col in OH_cat_cols if data_set[col].nunique() > 10]
        
        data_set = data_set.drop(high_cardinality_cols, axis=1)
        
        OH_cat_cols = list(set(OH_cat_cols) - set(high_cardinality_cols))
        
        ## update col_with_na
        cols_with_na = list(set(cols_with_na) - set(high_cardinality_cols))
           
        preprocess_returns = [data_set, numerical_cols, ordinal_cat_cols, OH_cat_cols]
        
        return preprocess_returns
       
       
       
       



def preprocess_function(data_set): ## this function is specifically defined for this housing data set, can take in features dataset, and test dataset
    ## columns with missing values
    cols_with_na = [col for col in data_set.columns if data_set[col].isnull().any()]
    
    removed_cols = []
    
    ## checking percentages of missing values per column, removing ones with > 50% missing, won't use those
    for col in cols_with_na:
        frac = data_set[col].isnull().sum()/len(data_set[col])
        # print ('{}: {}'.format(col, frac))
        
        if frac > 0.6: ## note this fraction is chosen such that it removes the same set of columns from both train & test data sets
            data_set = data_set.drop(col, axis=1)
            removed_cols.append(col)
            # print (col)
     
    ## update col_with_na
    cols_with_na = list(set(cols_with_na) - set(removed_cols))
        
    
    ## identifying all categorical, and numerical columns
    categorical_cols = [col for col in data_set.columns if data_set[col].dtype == "object"]
    
    numerical_cols = [col for col in data_set.columns if data_set[col].dtype in ['int64', 'float64']]
    numerical_cols = list(set(numerical_cols) - set(removed_cols))
    
    
    ## columns with ordinal categorical data, determined manually (so might contain some deleted colums) based on what makes sense as ordinal, specific to this dataset
    ordinal_cat_cols = ['Utilities', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 'KitchenQual', 'Functional', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence']
    
    ## One Hot categorical columns
    OH_cat_cols = list(set(categorical_cols) - set(ordinal_cat_cols))
    
    
    ## removing the deleted columns
    ordinal_cat_cols = list(set(ordinal_cat_cols) - set(removed_cols))
    
    ## now need to determine the ordinal relationship between the labels in each of these columns, unfortunately, this has to be done manually using the data_description file
    ## will make a list of dictionaries, for each of the above columns (without the removed columns)
    ordinal_dicts = [{'AllPub': 4, 'NoSewr': 3, 'NoSeWa':2, 'ELO':1}, {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}, {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}, {'Ex': 5, 'Gd': 4, 'TA':3, 'Fa': 2, 'Po': 1, 'NA': 0}, {'Ex': 5, 'Gd': 4, 'TA':3, 'Fa': 2, 'Po': 1, 'NA': 0}, {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}, {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf':1, 'NA': 0}, {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf':1, 'NA': 0}, {'Ex': 5, 'Gd': 4, 'TA':3, 'Fa': 2, 'Po': 1}, {'N': 0, 'Y': 1}, {'Ex': 5, 'Gd': 4, 'TA':3, 'Fa': 2, 'Po': 1}, {'Typ': 8, 'Min1': 7, 'Min2': 6, 'Mod': 5, 'Maj1': 4, 'Maj2': 3, 'Sev': 2, 'Sal': 1}, {'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0}, {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}, {'Ex': 5, 'Gd': 4, 'TA':3, 'Fa': 2, 'Po': 1, 'NA': 0}]         
    
    
    ## ordinal encoding, replacing the corresponding columns
    for i in range(len(ordinal_cat_cols)):
        data_set[ordinal_cat_cols[i]] = data_set[ordinal_cat_cols[i]].map(ordinal_dicts[i])
        
    
    
    ## getting rid of high cardinality OH categorical columns
    high_cardinality_cols = [col for col in OH_cat_cols if data_set[col].nunique() > 10]
    
    data_set = data_set.drop(high_cardinality_cols, axis=1)
    
    OH_cat_cols = list(set(OH_cat_cols) - set(high_cardinality_cols))
    
    ## update col_with_na
    cols_with_na = list(set(cols_with_na) - set(high_cardinality_cols))

    preprocess_returns = [data_set, numerical_cols, ordinal_cat_cols, OH_cat_cols]
    
    return preprocess_returns



def feature_selection(features, numerical_cols, target):  ## pass features dataset to this func, after preprocess function, 
    my_imputer = SimpleImputer(strategy='mean')
    
    imputed_num_features = pd.DataFrame(my_imputer.fit_transform(features[numerical_cols]))
    
    imputed_num_features.columns = numerical_cols
    
    fit_data = SelectKBest(mutual_info_regression, k= 30).fit(imputed_num_features, target)

    good_num_features = fit_data.get_feature_names_out()
    
    # print (good_num_features)

    return good_num_features


## after data is cleaned, we can make the pipeline with preprocessing and model


def make_sklearn_pipeline(features_data_set_components, target_data_set, test_data_set_components, selected_num_features, model, n_xvalid, train_or_test):
    
    
    ## removing the "bad" numerical features, though seems to alwasy get worse for some reason, to avoid using this set k = 'all' in feature_selection func
    remove_these_features = list(set(features_data_set_components[1]) - set(selected_num_features))
    
    features_data_set_components[0] = features_data_set_components[0].drop(remove_these_features, axis=1)
    test_data_set_components[0] = test_data_set_components[0].drop(remove_these_features, axis=1)
    
    ## updatenumerical cols to be used below
    features_data_set_components[1] = selected_num_features
    
    
    
    ## imputer for numerical data
    numerical_transformer = SimpleImputer(strategy='mean')
    
    ## imputer for categorical data
    ordinal_transformer = SimpleImputer(strategy='most_frequent')   ## encoding already done for these
    
    OH_transformer = Pipeline(steps= [('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    
    ## Bundle imputers for numerical and categorical data, can use the same peprocessor for both train and test datasets, becasue they have the same column names
    preprocessor = ColumnTransformer(transformers= [('num', numerical_transformer, features_data_set_components[1]), ('ordinal', ordinal_transformer, features_data_set_components[2]), ('OH', OH_transformer, features_data_set_components[3])])
      
    
    ## Bundle preprocessing and model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    
    if train_or_test == 0:
    
        # Multiply by -1 since sklearn calculates *negative* MAE
        scores = -1 * cross_val_score(pipeline, features_data_set_components[0], target_data_set, cv = n_xvalid, scoring='neg_mean_absolute_error')
        
        print('MAE scores:\n', scores)
        print('Average MAE score (across experiments) for {}:'.format(type(model).__name__))
        print(scores.mean())
        print ('----------------------------------------------------------')
        
    if train_or_test == 1:
        pipeline.fit(features_data_set_components[0], target_data_set)
        
        preds = pipeline.predict(test_data_set_components[0])
        
        preds = pd.DataFrame(preds, index = [i+1461 for i in np.arange(len(preds))], columns = ['SalePrice'])
        preds.index.name = 'Id'
        
        return preds




    
