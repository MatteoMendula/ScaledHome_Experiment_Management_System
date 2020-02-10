import os

import numpy as np
import pandas as pd

import settings


def read_data_pandas(data_path, cols_to_drop):
    data_frame = pd.read_csv(data_path)
    # data_frame = data_frame.set_index(pd.DatetimeIndex(data_frame['TIME']))
    data_frame = data_frame.drop(columns=cols_to_drop)
    return data_frame


def read_file_from_csv_to_np_array(path_to_file, file_name, cols_to_drop=None):
    if cols_to_drop is None:
        cols_to_drop = []
    file_uri = os.path.join(path_to_file, file_name)
    df = read_data_pandas(file_uri, cols_to_drop)
    my_np_array = np.array(df)
    # my_data = np.genfromtxt(file_uri, delimiter=',')
    return my_np_array


def read_file_from_csv_to_dictionary(file_uri, cols_to_drop=None):
    if cols_to_drop is None:
        cols_to_drop = []
    df = read_data_pandas(file_uri, cols_to_drop)
    dictionary = dict()
    for key in df.keys():
        dictionary[key] = list(df[key])
    return dictionary


def create_lists_pairs(file_uri, col_list1, col_list2, prediction_index=0, as_numpy=False):
    out_list = list()
    dictionary = read_file_from_csv_to_dictionary(file_uri)
    # print(len(dictionary.items()))
    # print(len(dictionary[col1]))
    for index in range(len(dictionary[col_list1[0]]) - prediction_index):
        list1 = list()
        list2 = list()
        for col1 in col_list1:
            list1.append(dictionary[col1][index])
        for col2 in col_list2:
            list2.append(dictionary[col2][index + prediction_index])
        out_list.append((list1, list2))

    if as_numpy:
        x = list()
        y = list()
        for pair in out_list:
            x.append(pair[0])
            y.append(pair[1])

        return np.array(x), np.array(y)

    return out_list




if __name__ == '__main__':
    # cols_to_drop = ["TIME"]
    # np_array = readFileFromCSVtoNpArray(settings.PROJECT_ROOT_ADDRESS, "data/2_5_2020_random_actions_1h_every_60s.csv")
    # dictionary = readFileFromCSVtoDictionary(settings.PROJECT_ROOT_ADDRESS, "data/2_5_2020_random_actions_1h_every_60s.csv")
    file_uri = os.path.join(settings.PROJECT_ROOT_ADDRESS, "data/2_5_2020_random_actions_1h_every_60s.csv")
    list_of_pairs = create_lists_pairs(file_uri, ["TIME", "OUT_T[*C]"], ["T6[*C]"], 8, as_numpy=True)
    print(list_of_pairs[:10])
    # print(np_array)
