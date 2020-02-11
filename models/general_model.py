import sys
import inspect

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

sys.path.append('../')
from dataset_utils import create_lists_pairs    
import model_utilities

class GeneralModel(object): 
    def __init__(self, n_neighbors, dataset_uri, feature_cols, target_cols, prediction_index, random_seed=-1):
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.prediction_index = prediction_index
        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = model_utilities.split_dataset_train_val_test(dataset_uri, feature_cols, target_cols, prediction_index)
        self.x_train, self.x_val, self.x_test = model_utilities.scale(self.x_train, self.x_val, self.x_test)

        self.data_set_parts = {
            'train': {
                'x': self.x_train,
                'y': self.y_train
            },
            'val': {
                'x': self.x_val,
                'y': self.y_val
            },
            'test': {
                'x': self.x_test,
                'y': self.y_test
            }
        }
    def train(self, model):
        model_utilities.train_model(model, self.x_train, self.y_train)

    def evaluate(self, model, dataset_part='test'):
        return model_utilities.evaluate(model, self.data_set_parts, dataset_part)

    def get_hyperparameters(self):
        return model_utilities.get_hyperparameters(self)