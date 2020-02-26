import sys
import os
import itertools

sys.path.append('../')
import settings


class HyperParameterSearcher(object):
    def __init__(self, algorithm_class, config):
        self.config = config
        self.algorithm_class = algorithm_class
        self.best = {
        # it refers to validation
            'model': None,
            'error': list() 
        }

    def update_bests(self, val_error, knn_config):
        if self.best['model'] is None or sum(self.best['error']) > sum(val_error):
            self.best['model'] = knn_config
            self.best['error'] = val_error


    def create_config_conbinations_as_list_of_dicts(self):
        parameters = self.config.keys()
        ll = list()
        for key in parameters:
            if type(self.config[key])!=list:
                p_list = list()
                p_list.append(self.config[key])
            else:
                p_list = self.config[key]
            ll.append(p_list)
        permutations = list (itertools.product(*ll))
        ld = list()
        for p in permutations:
            d = {}
            index = 0
            for k in parameters:
                d[k] = p[index]
                index += 1
            ld.append(d)

        return ld


    def search(self, mode='only best'):
        configurations = self.create_config_conbinations_as_list_of_dicts()
        for config_i in configurations:
            model_config = self.algorithm_class(
                **config_i
            )
            model_config.train()
            # train_error = model_config.evaluate(dataset_part='train')
            val_error = model_config.evaluate(dataset_part='val')[0]
            # print(config_i, val_error)
            # test_error = model_config.evaluate(dataset_part='test')

            if (mode == 'only best'):
                self.update_bests(val_error, model_config)
            elif (mode == 'all models'):
                pass


    def get_best(self):
        return self.best


def search_knn():
    from models.knn.knn import KNNConfig
    # HyperParameterSearcher(KNNConfig, )
    dataset_uri = os.path.join(settings.PROJECT_ROOT_ADDRESS, "data/2_5_2020_random_actions_1h_every_60s.csv")
    # dataset_uri = os.path.join(settings.PROJECT_ROOT_ADDRESS, "data/random_actions_every_60s_full_data.csv")

    n_neighbors = [3, 11, 21]
    feature_cols = [settings.INPUT_FEATURE_NAMES]
    if 'TIME' in settings.INPUT_FEATURE_NAMES:
        feature_cols[0].remove('TIME')
    target_cols = [settings.TARGET_FEATURE_NAMES]
    if 'TIME' in settings.TARGET_FEATURE_NAMES:
        target_cols[0].remove('TIME')
    distance_metrics = ['euclidean','manhattan']
    prediction_index = [6]

    common_config = {
        'dataset_uri': dataset_uri,
        'n_neighbors': n_neighbors,
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'distance_metric': distance_metrics,
        'prediction_index': prediction_index
    }

    knn_searcher = HyperParameterSearcher(KNNConfig, common_config)

    knn_searcher.search()

    bests = knn_searcher.get_best()
    bests['model'].save()
    path = bests['model'].get_restoring_path()
    knn_loaded = KNNConfig.load(path)
    knn_loaded.train(include_val=True)
    
    print('----------------------------KNN MODEL----------------------------')
    print()
    print('BEST PARAMS')
    print(bests['model'].get_hyperparameters())
    print()
    print('[TEST error - without validation]')
    print('row mse list:',bests['model'].evaluate(dataset_part='test')[0])
    print('accuracy',bests['model'].evaluate(dataset_part='test')[1])
    print()
    print('[TEST error - with validation]')
    print('row mse list:',knn_loaded.evaluate(dataset_part='test')[0])
    print('accuracy',knn_loaded.evaluate(dataset_part='test')[1])
    print()


def search_svr():
    from models.svm.svm import SVRConfig
    dataset_uri = os.path.join(settings.PROJECT_ROOT_ADDRESS, "data/2_5_2020_random_actions_1h_every_60s.csv")
    # dataset_uri = os.path.join(settings.PROJECT_ROOT_ADDRESS, "data/random_actions_every_60s_full_data.csv")
    feature_cols = [settings.INPUT_FEATURE_NAMES]
    if 'TIME' in settings.INPUT_FEATURE_NAMES:
        feature_cols[0].remove('TIME')
    target_cols = [settings.TARGET_FEATURE_NAMES]
    if 'TIME' in settings.TARGET_FEATURE_NAMES:
        target_cols[0].remove('TIME')
    prediction_index = [6]

    common_config = {
        'dataset_uri': dataset_uri,
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'prediction_index': prediction_index
    }

    svr_searcher = HyperParameterSearcher(SVRConfig, common_config)

    svr_searcher.search()

    bests = svr_searcher.get_best()
    bests['model'].save()
    path = bests['model'].get_restoring_path()
    svr_loaded = SVRConfig.load(path)
    svr_loaded.train(include_val=True)

    print('----------------------------SVR MODEL----------------------------')
    print()
    print('BEST PARAMS')
    print(bests['model'].get_hyperparameters())
    print()
    print('[TEST error - without validation]')
    print('row mse list:',bests['model'].evaluate(dataset_part='test')[0])
    print('accuracy',bests['model'].evaluate(dataset_part='test')[1])
    print()
    print('[TEST error - with validation]')
    print('row mse list:',svr_loaded.evaluate(dataset_part='test')[0])
    print('accuracy',svr_loaded.evaluate(dataset_part='test')[1])
    print()


def search_dnn():
    from keras.losses import mean_squared_error, huber_loss
    from models.dnn.dnn import DNNConfig
    # HyperParameterSearcher(KNNConfig, )
    dataset_uri = os.path.join(settings.PROJECT_ROOT_ADDRESS, "data/2_5_2020_random_actions_1h_every_60s.csv")
    # dataset_uri = os.path.join(settings.PROJECT_ROOT_ADDRESS, "data/random_actions_every_60s_full_data.csv")

    feature_cols = [settings.INPUT_FEATURE_NAMES]
    if 'TIME' in settings.INPUT_FEATURE_NAMES:
        feature_cols[0].remove('TIME')
    target_cols = [settings.TARGET_FEATURE_NAMES]
    if 'TIME' in settings.TARGET_FEATURE_NAMES:
        target_cols[0].remove('TIME')
    prediction_index = [6]
    loss_function = [mean_squared_error, huber_loss]
    n_layers = [5]
    n_epochs = [500]
    batch_size = [5]
    common_config = {
        'dataset_uri': dataset_uri,
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'loss_function': loss_function,
        'n_layers': n_layers,
        'prediction_index': prediction_index,
        'n_epochs': n_epochs,
        'batch_size': batch_size
    }

    dnn_searcher = HyperParameterSearcher(DNNConfig, common_config)

    dnn_searcher.search()

    bests = dnn_searcher.get_best()
    bests['model'].save()
    path = bests['model'].get_restoring_path()
    dnn_loaded = DNNConfig.load(path)
    dnn_loaded.train(include_val=True)

    print('----------------------------DNN MODEL----------------------------')
    print()
    print('BEST PARAMS')
    print(bests['model'].get_hyperparameters())
    print()
    print('[TEST error - without validation]')
    print('row mse list:',bests['model'].evaluate(dataset_part='test')[0])
    print('accuracy',bests['model'].evaluate(dataset_part='test')[1])
    print()
    print('[TEST error - with validation]')
    print('row mse list:',dnn_loaded.evaluate(dataset_part='test')[0])
    print('accuracy',dnn_loaded.evaluate(dataset_part='test')[1])
    print()

def search_lstm():
    from keras.losses import mean_squared_error, huber_loss
    from models.lstm.lstm import LSTMConfig
    # HyperParameterSearcher(KNNConfig, )
    dataset_uri = os.path.join(settings.PROJECT_ROOT_ADDRESS, "data/2_5_2020_random_actions_1h_every_60s.csv")
    # dataset_uri = os.path.join(settings.PROJECT_ROOT_ADDRESS, "data/random_actions_every_60s_full_data.csv")

    feature_cols = [settings.INPUT_FEATURE_NAMES]
    if 'TIME' in settings.INPUT_FEATURE_NAMES:
        feature_cols[0].remove('TIME')
    target_cols = [settings.TARGET_FEATURE_NAMES]
    if 'TIME' in settings.TARGET_FEATURE_NAMES:
        target_cols[0].remove('TIME')
    prediction_index = [6]
    loss_function = [mean_squared_error, huber_loss]
    n_layers = [1]
    n_epochs = [500]
    n_neurons = [16]
    batch_size = [5]

    common_config = {
        'dataset_uri': dataset_uri,
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'loss_function': loss_function,
        'n_layers': n_layers,
        'prediction_index': prediction_index,
        'n_epochs': n_epochs,
        'n_neurons': n_neurons,
        'batch_size': batch_size
    }

    lstm_searcher = HyperParameterSearcher(LSTMConfig, common_config)

    lstm_searcher.search()

    bests = lstm_searcher.get_best()
    bests['model'].save()
    path = bests['model'].get_restoring_path()
    lstm_loaded = LSTMConfig.load(path)
    lstm_loaded.train(include_val=True)

    print('----------------------------LSTM MODEL----------------------------')
    print()
    print('BEST PARAMS')
    print(bests['model'].get_hyperparameters())
    print()
    print('[TEST error - without validation]')
    print('row mse list:',bests['model'].evaluate(dataset_part='test')[0])
    print('accuracy',bests['model'].evaluate(dataset_part='test')[1])
    print()
    print('[TEST error - with validation]')
    print('row mse list:',lstm_loaded.evaluate(dataset_part='test')[0])
    print('accuracy',lstm_loaded.evaluate(dataset_part='test')[1])
    print()

if __name__ == '__main__':
    # search_knn()
    # search_svr()
    # search_dnn()
    search_lstm()

