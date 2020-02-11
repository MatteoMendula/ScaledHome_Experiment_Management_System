import sys
import os
import itertools

# from itertools import combinations 

sys.path.append('../')
sys.path.append('./knn')
from knn import KNNConfig 
import settings

class HyperParameterSearcher(object):
    # def __init__(algorithm_class, *args, **kwargs):
    # '''
    def __init__(self, algorithm_class, config):
        self.config = config
        self.algorithm_class = algorithm_class
        self.best = {
        # it refers to validation
            'knn': None,
            'error': list() 
        }
    # '''

    '''
    def __init__(self, dataset_uri, n_neighbors, input_features, output_features, prediction_index):
        self.dataset_uri = dataset_uri
        self.n_neighbors = n_neighbors
        self.input_features = input_features
        self.output_features = output_features
        self.prediction_index = prediction_index
        self.best = {
            # it refers to validation
            'knn': None,
            'error': None 
        }
    # '''

    def update_bests(self, val_error, knn_config):
        # print('update_bests -- val error',val_error)
        # print('update_bests -- sum val error',sum(val_error))
        # print('update_bests -- best error',self.best['error'])
        # if (len(self.best['error']) > 0):
        #     print('update_bests -- sum best error',sum(self.best['error']))
        # else:
        #     print('update_bests -- sum best error', type(self.best['error']))
        # print()
        
        if (self.best['knn'] == None or sum(self.best['error']) > sum(val_error)):
            self.best['knn'] = knn_config
            self.best['error'] = val_error
            print('Best hyperparameters updated')
            print(self.best,'\n')

    '''
    def search(self, mode='only best'):
        for n_neighbors_i in self.n_neighbors:
            for in_f_i in self.input_features:
                for out_f_i in self.output_features:
                    for prediction_index_i in self.prediction_index:
                        knn_config = KNNConfig(
                            n_neighbors=n_neighbors_i,
                            dataset_uri=self.dataset_uri,
                            feature_cols=in_f_i,
                            target_cols=out_f_i,
                            prediction_index=prediction_index_i
                        )
                        knn_config.train()
                        # train_error = knn_config.evaluate(dataset_part='train')
                        val_error = knn_config.evaluate(dataset_part='val')
                        print(knn_config, val_error)
                        # test_error = knn_config.evaluate(dataset_part='test')

                        if (mode == 'only best'):
                            self.update_bests( val_error, knn_config)
                        elif (mode == 'all models'):
                            pass
    # '''


    '''
    def search(self, mode='only best'):
        for in_f_i in self.config['feature_cols']:
            for out_f_i in self.config['target_cols']:
                for prediction_index_i in self.config['prediction_index']:
                    config_i = {
                        'dataset_uri': self.config['dataset_uri'],
                        'feature_cols': in_f_i,
                        'target_cols': out_f_i,
                        'prediction_index': prediction_index_i
                    }
                    if (self.algorithm_class == KNNConfig):
                        for n_neighbors_i in self.config['n_neighbors']:
                            config_i['n_neighbors'] = n_neighbors_i

                            # print('.....................')
                            # print(config_i)
                            # print('....................')

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






if __name__ == '__main__':
    from models.knn.knn import KNNConfig
    # HyperParameterSearcher(KNNConfig, )


    dataset_uri = os.path.join(settings.PROJECT_ROOT_ADDRESS, "data/2_5_2020_random_actions_1h_every_60s.csv")

    n_neighbors = [3, 5]
    feature_cols = [
        ["OUT_T[*C]", "T6[*C]"],
        ["OUT_T[*C]", "T12[*C]"],
    ]
    target_cols = [
        ["T12[*C]", "T18[*C]"]
    ]
    prediction_index = [8, 12]

    ''' 
    knn_searcher = HyperParameterSearcher(
            dataset_uri = dataset_uri,
            n_neighbors = n_neighbors,
            input_features = feature_cols,
            output_features = target_cols,
            prediction_index = prediction_index
        )
    # '''
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
    print(bests['knn'].get_hyperparameters())



