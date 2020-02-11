import os
import sys

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

sys.path.append('../../')
sys.path.append('../')
# print(sys.path)
from dataset_utils import create_lists_pairs
import model_utilities


class SVMConfig(object):
    def __init__(self, dataset_uri, feature_cols, target_cols, prediction_index):
        self.model = svm.SVR()

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

    def train(self):
        model_utilities.train_model(self.model, self.x_train, self.y_train)

    def evaluate(self, dataset_part='test'):
        return model_utilities.evaluate(self.model, self.data_set_parts, dataset_part)

    def get_hyperparameters(self):
        return model_utilities.get_hyperparameters(self)

if __name__ == '__main__':
    X = [[0, 0], [2, 2]]
    y = [0.5, 2.5]
    clf = svm.SVR()
    clf.fit(X, y)

    res = clf.predict([[1, 1]])
    print(res)

    print('-------------------------')

    import settings
    dataset_uri = os.path.join(settings.PROJECT_ROOT_ADDRESS, "data/2_5_2020_random_actions_1h_every_60s.csv")
    feature_cols = [
        "OUT_T[*C]",
        "OUT_T[*C]",
        "OUT_H[%]",
        "T6[*C]",
        "H6[%]",
        "T12[*C]",
        "H12[%]",
        "T18[*C]",
        "H18[%]",
        "T19[*C]",
        "H19[%]",
        "T24[*C]",
        "H24[%]",
        "T25[*C]",
        "H25[%]",
        "T26[*C]",
        "H26[%]",
        "LAMP_STATE",
        "FAN_STATE",
        "AC_STATE",
        "HEATER_STATE",
        "M0",
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
        "M6",
        "M8",
        "M9",
        "M10",
        "M11",
        "M12",
        "M13",
        "M14",
        "M15",
    ]

    target_cols = ["T6[*C]", "T12[*C]", "T18[*C]", "T19[*C]", "T24[*C]", "T25[*C]", "T26[*C]"]
    prediction_index = 8

    svn_config = SVMConfig(
        dataset_uri=dataset_uri,
        feature_cols=feature_cols,
        target_cols=target_cols,
        prediction_index=prediction_index
    )

    svn_config.train()
    train_error = svn_config.evaluate(dataset_part='train')
    val_error = svn_config.evaluate(dataset_part='val')
    test_error = svn_config.evaluate(dataset_part='test')

    print(f'Train Error: {train_error}')
    print(f'Validation Error: {val_error}')
    print(f'Test Error: {test_error}')