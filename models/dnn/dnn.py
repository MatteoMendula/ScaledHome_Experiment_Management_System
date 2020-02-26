import os
import sys

from keras.models import Sequential
from keras.layers import Dense

sys.path.append('../')
sys.path.insert(0, "D:/Thesis draft/code/UCF_ML/models")
from general_model import GeneralModel



class DNNConfig(GeneralModel):
    def __init__(self, dataset_uri, feature_cols, target_cols, prediction_index, loss_function,n_layers, n_epochs, batch_size):
        super(DNNConfig, self).__init__(dataset_uri, feature_cols, target_cols, prediction_index)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.model = Sequential()
        self.loss_function = loss_function
        self.model.add(Dense(128, input_dim=len(feature_cols), kernel_initializer='normal', activation='relu'))
        for _ in range(n_layers):
            self.model.add(Dense(128, kernel_initializer='normal', activation='relu'))
        self.model.add(Dense(len(target_cols), kernel_initializer='normal'))
        #self.model.compile(loss='mean_squared_error', optimizer='adam')
        #self.model.compile(loss = huber_loss, optimizer='adam')
        self.model.compile(loss = self.loss_function, optimizer='adam')

    def train_on(self, x, y):
        self.model.fit(x, y, epochs=self.n_epochs, batch_size=self.batch_size)

    def predict_on(self, x):
        return self.model.predict(x)

    def get_restoring_path(self):
        dir_name = os.path.dirname(__file__)
        return os.path.join(
            dir_name,
            'saved_models',
            f'DNNConfig'
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

    target_cols = [
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
    prediction_index = 8

    nn_config = DNNConfig(
        dataset_uri=dataset_uri,
        feature_cols=feature_cols,
        target_cols=target_cols,
        prediction_index=prediction_index
    )

    nn_config.train()
    train_error = nn_config.evaluate(dataset_part='train')
    val_error = nn_config.evaluate(dataset_part='val')
    test_error = nn_config.evaluate(dataset_part='test')

    print(f'Train Error: {train_error}')
    print(f'Validation Error: {val_error}')
    print(f'Test Error: {test_error}')