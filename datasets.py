import os
from copy import deepcopy

import pandas as pd
import numpy as np

import settings


def read_dataset_from_csv(dataset_address, intervals):
    mydateparser = lambda x: pd.datetime.strptime(x, "%Y/%m/%d %H:%M")
    df = pd.read_csv(dataset_address, date_parser=mydateparser, index_col=1, header=None)
    df.columns = ('Year', 'Temp')
    df = df.drop(columns=['Year'])
    if dataset_address.endswith('data/mi_meteo_2001.csv'):
        df = df.iloc[18:618, :]

    df = df.iloc[::intervals, :]
    # df = df.set_index('Time')
    # df.index = pd.to_datetime(df.index, format="%Y/%m/%d %H:%M")
    return df


def convert_dataset_to_scale_home_temps(dataset_address, min_temp_home, max_temp_home, desired_temp=25, intervals=3):
    """All temperatures are based on celsius"""
    df = read_dataset_from_csv(dataset_address, intervals=intervals)
    pd.set_option('display.max_rows', None)

    df_with_max_min_each_day = deepcopy(df)

    df_with_max_min_each_day['min'] = np.repeat(
        df.groupby([df.index.month, df.index.day]).agg(['min']).values, 24 // intervals
    )
    df_with_max_min_each_day['max'] = np.repeat(
        df.groupby([df.index.month, df.index.day]).agg(['max']).values, 24 // intervals
    )

    def convert(x):
        # x[1] is the min of the day and x[2] is the max of the day.
        x[1] = min(x[1], desired_temp)
        x[2] = max(x[2], desired_temp)
        a = (max_temp_home - min_temp_home) / (x[2] - x[1])
        b = min_temp_home - a * x[1]
        return a * x + b

    df = df_with_max_min_each_day.apply(convert, axis=1)
    # pd.set_option('display.max_rows', None)
    # print(df)

    return zip(list(df.index.values), list(df.values[:, 0]))


def find_max_diff(intervals):
    dataset_address = os.path.join(settings.PROJECT_ROOT_ADDRESS, 'data/mi_meteo_2001.csv')
    df = read_dataset_from_csv(dataset_address, intervals)
    print('max')
    print(df.max())

    print('min')
    print(df.min())

    # print(df.head())
    diff_data = df.diff()

    # print(diff_data.head())

    print('max change')
    print(diff_data.max())

    print('min')
    print(diff_data.min())


if __name__ == '__main__':
    # find_max_diff(intervals=3)
    dataset_address = os.path.join(settings.PROJECT_ROOT_ADDRESS, 'data/mi_meteo_2001.csv')
    ds = convert_dataset_to_scale_home_temps(dataset_address, 23, 34)
    for time_step, temp in ds:
        print(f'{time_step}: {temp}')
