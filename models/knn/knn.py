# from _curses import KEY_NEXT
import os
import sys

from sklearn.neighbors import KNeighborsRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error

sys.path.append('../../')
sys.path.append('../')
from models.general_model import GeneralModel


class KNNConfig(GeneralModel):
    def __init__(self, dataset_uri, feature_cols, target_cols, prediction_index, n_neighbors):
        super(KNNConfig, self).__init__(dataset_uri, feature_cols, target_cols, prediction_index)
        self.n_neighbors = n_neighbors
        self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors)

    def train_on(self, x, y):
        self.model.fit(x, y)

    def predict_on(self, x):
        return self.model.predict(x)

    def get_restoring_path(self):
        dir_name = os.path.dirname(__file__)
        return os.path.join(
            dir_name,
            'saved_models',
            f'KNNConfig_k{self.n_neighbors}'
        )

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
