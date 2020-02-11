import sys
import os
import itertools

sys.path.append('../')
import settings


class HyperParameterSearcher(object):
    # '''
    def __init__(self, algorithm_class, config):
        self.config = config
        self.algorithm_class = algorithm_class
        self.best = {
        # it refers to validation
            'model': None,
            'error': list() 
        }
    # '''

    def update_bests(self, val_error, knn_config):

        if self.best['model'] is None or sum(self.best['error']) > sum(val_error):
            self.best['model'] = knn_config
            self.best['error'] = val_error
            # print('Best hyperparameters updated')
            # print(self.best,'\n')

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

    # '''
    def search(self, mode='only best'):
        list_of_dicts = self.create_config_conbinations_as_list_of_dicts()
        for config_i in list_of_dicts:
            knn_config = self.algorithm_class(
                **config_i
            )
            knn_config.train()
            # train_error = knn_config.evaluate(dataset_part='train')
            val_error = knn_config.evaluate(dataset_part='val')
            # print(config_i, val_error)
            # test_error = knn_config.evaluate(dataset_part='test')

            if (mode == 'only best'):
                self.update_bests(val_error, knn_config)
            elif (mode == 'all models'):
                pass
    # '''

    def get_best(self):
        return self.best


def search_knn():
    from models.knn.knn import KNNConfig
    # HyperParameterSearcher(KNNConfig, )
    dataset_uri = os.path.join(settings.PROJECT_ROOT_ADDRESS, "data/2_5_2020_random_actions_1h_every_60s.csv")

    n_neighbors = [3, 11, 21]
    feature_cols = [
        settings.INPUT_FEATURE_NAMES
    ]
    target_cols = [
        settings.TARGET_FEATURE_NAMES
    ]
    prediction_index = [6]

    common_config = {
        'dataset_uri': dataset_uri,
        'n_neighbors': n_neighbors,
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'prediction_index': prediction_index
    }

    knn_searcher = HyperParameterSearcher(KNNConfig, common_config)

    knn_searcher.search()

    bests = knn_searcher.get_best()

    print()
    print(bests['model'].get_hyperparameters())
    print(bests['model'].evaluate(dataset_part='test'))

    bests['model'].train(include_val=True)
    print(bests['model'].evaluate(dataset_part='test'))


def search_svr():
    from models.svm.svm import SVRConfig
    dataset_uri = os.path.join(settings.PROJECT_ROOT_ADDRESS, "data/2_5_2020_random_actions_1h_every_60s.csv")
    feature_cols = [
        settings.INPUT_FEATURE_NAMES
    ]
    target_cols = [
        settings.TARGET_FEATURE_NAMES
    ]
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

    print()
    print(bests['model'].get_hyperparameters())

    print(bests['model'].evaluate(dataset_part='test'))

    bests['model'].train(include_val=True)
    print(bests['model'].evaluate(dataset_part='test'))


if __name__ == '__main__':
    search_knn()
    search_svr()


