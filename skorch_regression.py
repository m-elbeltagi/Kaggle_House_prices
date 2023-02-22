import torch
from torch import nn
import pandas as pd
from skorch import NeuralNetClassifier
from house_prices_functions import *


full_train_data = pd.read_csv(r'C:\Users\M\OneDrive - Carleton University\Documents\my_stuff\Projects\Kaggle_House_Prices\train.csv')
test_data = pd.read_csv(r'C:\Users\M\OneDrive - Carleton University\Documents\my_stuff\Projects\Kaggle_House_Prices\test.csv')

full_features = full_train_data.drop(['Id', 'SalePrice'], axis=1)
test_data = test_data.drop(['Id'], axis=1)
target = full_train_data.SalePrice


device = "cuda" if torch.cuda.is_available() else "cpu"
print("using {}".format(device))


## defining the pytorch neural network
class TorchRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
          nn.Linear(72, 140),
          nn.ReLU(),
          nn.Linear(140, 72),
          nn.ReLU(),
          nn.Linear(72, 36),
          nn.ReLU(),
          nn.Linear(36, 1)
        )
        


    def forward(self, x):
        return self.layers(x)
    

skorch_regressor = NeuralNetClassifier(
    TorchRegressor,
    max_epochs=100,
    criterion=nn.CrossEntropyLoss(),
    lr=0.1,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
)


model = skorch_regressor

train_set_components = preprocess_function(full_features)
test_set_components = preprocess_function(test_data)

make_sklearn_pipeline(train_set_components, target, test_set_components, model, n_xvalid=5, train_or_test=0)


## Doesnt work, I don't think it can handle the OH encoding because it changes the dim of input data, I thougt the wrapper could handle this, but maybe not