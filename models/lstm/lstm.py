import os
import sys

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

sys.path.append('../')
sys.path.insert(0, "D:/Thesis draft/code/UCF_ML/models")
sys.path.insert(0, "D:/Thesis draft/code/UCF_ML")
from dataset_utils import create_sequence_from_flat_data
import settings

from general_model import GeneralModel



class LSTMConfig(GeneralModel):
    def __init__(self, dataset_uri, feature_cols, target_cols, prediction_index, window_size, n_layers, n_epochs, n_neurons, loss_function, batch_size):
        super(LSTMConfig, self).__init__(dataset_uri, feature_cols, target_cols, prediction_index, window_size)
        self.prediction_index = prediction_index
        self.window_size = window_size
        self.n_layers = n_layers
        self.n_epochs = n_epochs 
        self.n_neurons = n_neurons 
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.change_data_into_sequences()
        self.model = Sequential()
        self.model.add(LSTM(self.n_neurons, input_shape=(self.window_size, len(feature_cols))))
        self.model.add(Dense(len(target_cols)))
        self.model.compile(loss=self.loss_function, optimizer='adam')

    def train_on(self, x, y):
        #print(x.shape, y.shape)
        #x1 = create_sequence_from_flat_data(x, self.prediction_index)
        #print(x1.shape, y[self.prediction_index:,:].shape)
        #self.model.fit(x1, y[:y.shape[0]-self.prediction_index,:], epochs=100, batch_size=1, verbose=2)
        #self.model.fit(x1, y[self.prediction_index:,:], epochs=100, batch_size=1, verbose=2)
        self.model.fit(x, y, epochs=self.n_epochs, batch_size=self.batch_size, verbose=2)

    def predict_on(self, x):
        #x1 = create_sequence_from_flat_data(x, self.prediction_index)
        #print('predict', x.shape)
        return self.model.predict(x)

    def get_restoring_path(self):
        dir_name = os.path.dirname(__file__)
        return os.path.join(
            dir_name,
            'saved_models',
            f'LSTMConfig'
        )
            
if __name__ == '__main__':
    import settings
    dataset_uri = os.path.join(settings.PROJECT_ROOT_ADDRESS, "data/2_5_2020_random_actions_1h_every_60s.csv")
    feature_cols = settings.INPUT_FEATURE_NAMES
    if 'TIME' in settings.INPUT_FEATURE_NAMES:
        feature_cols.remove('TIME')
    target_cols = settings.TARGET_FEATURE_NAMES
    if 'TIME' in settings.TARGET_FEATURE_NAMES:
        target_cols.remove('TIME')
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