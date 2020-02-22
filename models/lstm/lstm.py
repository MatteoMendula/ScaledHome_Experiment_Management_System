import os
import sys

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

sys.path.append('../')
sys.path.insert(0, "D:/Thesis draft/code/UCF_ML/models")
from general_model import GeneralModel



class LSTMConfig(GeneralModel):
    def __init__(self, dataset_uri, feature_cols, target_cols, prediction_index):
        super(LSTMConfig, self).__init__(dataset_uri, feature_cols, target_cols, prediction_index)
        self.model = Sequential()
        look_back = 1
        self.model.add(LSTM(4, input_shape=(len(feature_cols), look_back)))
        self.model.add(Dense(len(target_cols)))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def train_on(self, x, y):
        #print (x.shape)
        x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
        #print (x.shape)
        y = np.reshape(y, (y.shape[0], 1, y.shape[1]))
        self.model.fit(x, y, epochs=500, batch_size=1, verbose=2)

    def predict_on(self, x):
        return self.model.predict(x)

    def get_restoring_path(self):
        dir_name = os.path.dirname(__file__)
        return os.path.join(
            dir_name,
            'saved_models',
            f'NNConfig_k'
        )
            
if __name__ == '__main__':
    import settings
    dataset_uri = os.path.join(settings.PROJECT_ROOT_ADDRESS, "data/2_5_2020_random_actions_1h_every_60s.csv")
    feature_cols = [
        "TIME",
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

    target_cols = [
        "T6[*C]",
        "T12[*C]",
        "T18[*C]",
        "T19[*C]",
        "T24[*C]",
        "T25[*C]",
        "T26[*C]",
    ]
    prediction_index = 8

    lstm_config = LSTMConfig(
        dataset_uri=dataset_uri,
        feature_cols=feature_cols,
        target_cols=target_cols,
        prediction_index=prediction_index
    )

    lstm_config.train()
    train_error = lstm_config.evaluate(dataset_part='train')
    val_error = lstm_config.evaluate(dataset_part='val')
    test_error = lstm_config.evaluate(dataset_part='test')

    print(f'Train Error: {train_error}')
    print(f'Validation Error: {val_error}')
    print(f'Test Error: {test_error}')