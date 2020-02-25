import os
import sys
import inspect
from abc import abstractmethod
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

#sys.path.append('../')
sys.path.insert(0, "D:/Thesis draft/code/UCF_ML")
from dataset_utils import handle_data
from dataset_utils import create_lists_pairs
from dataset_utils import create_sequence_from_flat_data

class GeneralModel(object):
    def __init__(self, dataset_uri, feature_cols, target_cols, prediction_index):
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.prediction_index = prediction_index
        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = \
            self.split_dataset_train_val_test(dataset_uri, feature_cols, target_cols, prediction_index)

        self.scaler = StandardScaler()
        self.x_train, self.x_val, self.x_test = self.scale(self.x_train, self.x_val, self.x_test)
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

    def train(self, include_val=False):
        x, y = self.x_train, self.y_train
        if include_val:
            x, y = np.concatenate((x, self.x_val), axis=0), np.concatenate((y, self.y_val), axis=0)
        self.train_on(x, y)

    @abstractmethod
    def train_on(self, x, y):
        pass

    def predict(self, x, scale=False):
        if scale:
            x = self.scaler.transform(x)
        return self.predict_on(x)

    @abstractmethod
    def predict_on(self, x):
        pass
    

    @abstractmethod
    def get_restoring_path(self):
        pass

    def split_dataset_train_val_test(self, dataset_uri, feature_cols, target_cols, prediction_index):
        x, y = handle_data(dataset_uri, feature_cols, target_cols, prediction_index, as_numpy=True)
        #x, y = create_lists_pairs(dataset_uri, feature_cols, target_cols, prediction_index, as_numpy=True)
        x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val,
            y_train_val,
            test_size=0.125,
            shuffle=False
        )
        return x_train, x_val, x_test, y_train, y_val, y_test

    def scale(self, train, val, test):
        self.scaler.fit(train)
        train = self.scaler.transform(train)
        val = self.scaler.transform(val)
        test = self.scaler.transform(test)
        return train, val, test

    def evaluate(self, dataset_part, flat=True):
        if flat:
            x, y = self.data_set_parts[dataset_part]['x'], self.data_set_parts[dataset_part]['y']
        else:
            x = create_sequence_from_flat_data(self.data_set_parts[dataset_part]['x'], self.prediction_index)
            y = self.data_set_parts[dataset_part]['y'][self.prediction_index:,:]
        y_hat = self.predict(x)
        return mean_squared_error(y, y_hat, multioutput='raw_values')

    def get_hyperparameters(self):
        attributes = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        return list(a for a in attributes if not (a[0].startswith('__')
                                                  and a[0].endswith('__'))
                    and not (a[0].startswith('data_set_parts'))
                    and not (a[0].startswith('x'))
                    and not (a[0].startswith('y'))
                    )

    def save(self):
        if not os.path.exists(self.get_restoring_path()):
            os.makedirs(os.path.dirname(self.get_restoring_path()))
        file_handler = open(self.get_restoring_path(), 'wb')
        pickle.dump(self, file_handler)

    def load_latest(self):
        return GeneralModel.load(restore_path=self.get_restoring_path())

    @staticmethod
    def load(restore_path):
        file_handler = open(restore_path, 'rb')
        return pickle.load(file_handler)