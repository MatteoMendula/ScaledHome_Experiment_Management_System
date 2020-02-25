import os

import numpy as np
import pandas as pd

import settings


def read_data_pandas(data_path, cols_to_drop=None):
    if cols_to_drop is None:
        cols_to_drop = []
    data_frame = pd.read_csv(data_path)
    # data_frame = data_frame.set_index(pd.DatetimeIndex(data_frame['TIME']))
    data_frame = data_frame.drop(columns=cols_to_drop)
    return data_frame


def read_file_from_csv_to_np_array(file_uri, features, targets):
    df = read_data_pandas(file_uri)
    cols_to_drop = [item for item in list(df.keys()) if item not in (features+targets)]
    df_filtered = df.drop(columns=cols_to_drop)
    x_indexies = []
    y_indexies = []
    for index, key in enumerate(df_filtered.keys()):
        if key in features:
            x_indexies.append(index)
        if key in targets:
            y_indexies.append(index)
    my_np_array = np.array(df_filtered)
    return my_np_array, x_indexies, y_indexies


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

def handle_data_no_cross(file_uri, col_list1, col_list2,prediction_index=3, as_numpy=True):
    data_np, x_indexies, y_indexies = read_file_from_csv_to_np_array(file_uri, features=col_list1, targets=col_list2)
    dataX, dataY = [], []
    sequence_x = []
    for row in range(data_np.shape[0]-prediction_index):
        temp_x = []
        temp_y = []
        for col in range(data_np.shape[1]):
            if (col in x_indexies):
                temp_x.append(data_np[row,col])
            elif (col in y_indexies):
                temp_y.append(data_np[row,col])
        if row == 0:
            print(temp_x, temp_y)
        dataY.append(temp_y)
        sequence_x.append(temp_x)
        if (row+1) % prediction_index == 0:
            dataX.append(sequence_x)
            if row == 0:
                print('ASD',dataX, sequence_x)
            sequence_x = []
            if row == 0:
                print('ASD2',dataX, sequence_x)

def handle_data(file_uri, feature_list, target_list,prediction_index=3, as_numpy=True, flat=True):
    data_np, x_indexies, y_indexies = read_file_from_csv_to_np_array(file_uri, features=feature_list, targets=target_list)
    dataX, dataY = [], []
    sequence_x = []
    for row in range(data_np.shape[0]-prediction_index):
        for index in range(prediction_index):
            #temp_x = np.take(data_np[row+index], x_indexies)
            sequence_x.append(np.take(data_np[row+index], x_indexies))
        temp_y = np.take(data_np[row+prediction_index], y_indexies)
        dataY.append(temp_y)
        if flat:
            dataX.append(sequence_x[0])
        else:
            dataX.append(sequence_x)
        #print(row, sequence_x)
        sequence_x = []
    return np.array(dataX), np.array(dataY)

if __name__ == '__main__':
    # cols_to_drop = ["TIME"]
    # np_array = readFileFromCSVtoNpArray(settings.PROJECT_ROOT_ADDRESS, "data/2_5_2020_random_actions_1h_every_60s.csv")
    # dictionary = readFileFromCSVtoDictionary(settings.PROJECT_ROOT_ADDRESS, "data/2_5_2020_random_actions_1h_every_60s.csv")
    file_uri = os.path.join(settings.PROJECT_ROOT_ADDRESS, "data/2_5_2020_random_actions_1h_every_60s.csv")
    #x,y = handle_data_no_cross(file_uri, ["TIME", "OUT_T[*C]"], ["T6[*C]"], 4, as_numpy=True)
    x,y = handle_data(file_uri, ["TIME", "OUT_T[*C]"], ["T6[*C]"], 4, as_numpy=True)
    
    print(type(x))
    print(y)

    print(x.shape)
    print(y.shape)
    # print(np_array)
