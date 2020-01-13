import os

import pandas

import settings


def find_max_diff(intervals):

    dataset_address = os.path.join(settings.PROJECT_ROOT_ADDRESS, 'data/mi_meteo_2001.csv')
    df = pandas.read_csv(dataset_address)
    df.columns = ('Year', 'Time', 'Temp')

    df = df.drop(columns=['Year'])
    df = df.iloc[17:-6, :]
    df = df.iloc[::intervals, :]

    df = df.set_index('Time')
    # print(df.head())

    diff_data = df.diff()

    # print(diff_data.head())

    print('max')
    print(diff_data.max())

    print('min')
    print(diff_data.min())


if __name__ == '__main__':
    find_max_diff(intervals=3)
