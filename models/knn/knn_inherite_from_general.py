# from _curses import KEY_NEXT
import os
import sys

from sklearn.neighbors import KNeighborsRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error

sys.path.append('../../')
sys.path.append('../')
from dataset_utils import create_lists_pairs
import model_utilities
import general_model

class KNNConfig(object):
    def __init__(self, n_neighbors, dataset_uri, feature_cols, target_cols, prediction_index, random_seed=-1):
        if random_seed != -1:
            self.random_seed = random_seed

        # print(n_neighbors,dataset_uri)
        # print()
        # print(dataset_uri, feature_cols, target_cols, prediction_index)

        self.n_neighbors = n_neighbors
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.prediction_index = prediction_index

        self.model = KNeighborsRegressor(
            n_neighbors=n_neighbors
        )
        # x, y = create_lists_pairs(dataset_uri, feature_cols, target_cols, prediction_index, as_numpy=True)
        # x_train, self.x_test, y_train, self.y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
        # self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
        #     x_train,
        #     y_train,
        #     test_size=0.125,
        #     shuffle=False
        # )
        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = model_utilities.split_dataset_train_val_test(dataset_uri, feature_cols, target_cols, prediction_index)
        # self.scaler = StandardScaler()
        # self.scaler.fit(self.x_train)
        # self.x_train = self.scaler.transform(self.x_train)
        # self.x_val = self.scaler.transform(self.x_val)
        # self.x_test = self.scaler.transform(self.x_test)
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

    # def train(self):
    #     self.model.fit(self.x_train, self.y_train)

    def train(self):
        model_utilities.train_model(self.model, self.x_train, self.y_train)

    # def evaluate(self, dataset_part='test'):
    #     if dataset_part == 'train':
    #         x, y = self.x_train, self.y_train
    #     elif dataset_part == 'val':
    #         x, y = self.x_val, self.y_val
    #     elif dataset_part == 'test':
    #         x, y = self.x_test, self.y_test
    #     else:
    #         raise Exception('The dataset part should be train, val or test.')

    #     y_hat = self.model.predict(x)
    #     return mean_squared_error(y, y_hat, multioutput='raw_values')


    def evaluate(self, dataset_part='test'):
        return model_utilities.evaluate(self.model, self.data_set_parts, dataset_part)

    # def get_hyperparameters(self):
    #     return {
    #         'n_neighbors': self.n_neighbors,
    #         'feature_cols': self.feature_cols,
    #         'target_cols': self.target_cols,
    #         'prediction_index': self.prediction_index
    #     }

    def get_hyperparameters(self):
        return model_utilities.get_hyperparameters(self)


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

    knn_config = KNNConfig(
        n_neighbors=3,
        dataset_uri=dataset_uri,
        feature_cols=feature_cols,
        target_cols=target_cols,
        prediction_index=prediction_index
    )

    knn_config.train()
    train_error = knn_config.evaluate(dataset_part='train')
    val_error = knn_config.evaluate(dataset_part='val')
    test_error = knn_config.evaluate(dataset_part='test')

    print(f'Train Error: {train_error}')
    print(f'Validation Error: {val_error}')
    print(f'Test Error: {test_error}')
