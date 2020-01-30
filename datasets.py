import os

import pandas as pd

import settings


def read_dataset_from_csv(dataset_address, intervals):
    mydateparser = lambda x: pd.datetime.strptime(x, "%Y/%m/%d %H:%M")
    df = pd.read_csv(dataset_address, date_parser=mydateparser)
    df.columns = ('Year', 'Time', 'Temp')
    df = df.drop(columns=['Year'])
    if dataset_address.endswith('data/mi_meteo_2001.csv'):
        df = df.iloc[17:-6, :]

    df = df.iloc[::intervals, :]
    df = df.set_index('Time')
    return df


def convert_dataset_to_scale_home_temps(dataset_address, min_temp_home, max_temp_home, intervals=3):
    df = read_dataset_from_csv(dataset_address, intervals=intervals)

    print(df.head())
    max_temp_data = df.max()
    min_temp_data = df.min()

    a = (max_temp_home - min_temp_home) / (max_temp_data - min_temp_data)
    b = min_temp_home - a * min_temp_data

    df = df.apply(lambda x: a * x + b, axis=1)
    pd.set_option('display.max_rows', None)
    print(df)


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
    convert_dataset_to_scale_home_temps(dataset_address, 22, 29)
