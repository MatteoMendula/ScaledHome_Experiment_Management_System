import os
import sys

import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from models.general_model import GeneralModel

sys.path.append('../../')
sys.path.append('../')
# print(sys.path)
from dataset_utils import create_lists_pairs


class SVRConfig(GeneralModel):
    def __init__(self, dataset_uri, feature_cols, target_cols, prediction_index):
        super(SVRConfig, self).__init__(dataset_uri, feature_cols, target_cols, prediction_index)

        self.models = [svm.SVR() for i in range(len(target_cols))]

    def train_on(self, x, y):
        for i in range(len(self.target_cols)):
            self.models[i].fit(x, y[:, i])

    def predict(self, x):
        y_hat = np.zeros(shape=(x.shape[0], len(self.target_cols)))
        for i in range(len(self.target_cols)):
            y_hat[:, i] = self.models[i].predict(x)

        return y_hat


if __name__ == '__main__':
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

    svn_config = SVRConfig(
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