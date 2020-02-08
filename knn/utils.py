# from numpy import genfromtxt
import numpy as np

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
import settings

import pandas as pd
def read_data_pandas(data_path, cols_to_drop):
    data_frame = pd.read_csv(data_path)
    #data_frame = data_frame.set_index(pd.DatetimeIndex(data_frame['TIME']))
    data_frame = data_frame.drop(columns=cols_to_drop)
    return data_frame


def readFileFromCSVtoNpArray(path_to_file, file_name, cols_to_drop=[]): 
    file_uri = os.path.join(path_to_file, file_name)
    df = read_data_pandas(file_uri, cols_to_drop)
    my_np_array = np.array(df)
    # my_data = np.genfromtxt(file_uri, delimiter=',')
    return my_np_array

def readFileFromCSVtoDictionary(path_to_file, file_name, cols_to_drop=[]): 
    file_uri = os.path.join(path_to_file, file_name)
    df = read_data_pandas(file_uri, cols_to_drop)
    dictionary = dict()
    for key in df.keys():
        dictionary[key] = list(df[key])
    return dictionary

def createPairs(path_to_file, file_name, col1, col2):
    out_list = list()
    dictionary = readFileFromCSVtoDictionary(path_to_file, file_name)
    # print(len(dictionary.items()))
    # print(len(dictionary[col1]))
    for index in range(len(dictionary[col1])):
        out_list.append((dictionary[col1][index], dictionary[col2][index]))
    return out_list

if __name__ == '__main__':
    # cols_to_drop = ["TIME"]
    # np_array = readFileFromCSVtoNpArray(settings.PROJECT_ROOT_ADDRESS, "data/2_5_2020_random_actions_1h_every_60s.csv")
    # dictionary = readFileFromCSVtoDictionary(settings.PROJECT_ROOT_ADDRESS, "data/2_5_2020_random_actions_1h_every_60s.csv")
    list_of_pairs = createPairs(settings.PROJECT_ROOT_ADDRESS, "data/2_5_2020_random_actions_1h_every_60s.csv", "TIME", "AC_STATE")
    print(list_of_pairs[:10])
    # print(np_array)
