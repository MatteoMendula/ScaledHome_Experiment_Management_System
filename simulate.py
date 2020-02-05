import os

import simulations
import datasets
import settings

if __name__ == '__main__':
    dataset_address = os.path.join(settings.PROJECT_ROOT_ADDRESS, 'data/mi_meteo_2001.csv')

    steps_per_day = 8
    intervals = 24 // steps_per_day

    ds = datasets.convert_dataset_to_scale_home_temps(dataset_address, 23, 38, intervals=intervals)

    dict_of_days_temperatures = datasets.unzipAndCreateDictOfLists(ds,steps_per_day)

    # print(len(dict_of_days_temperatures.keys()))

    for day,temperatures in dict_of_days_temperatures.items():
        print("Day {0} - temperatures: {1}".format(day,temperatures))
        simulations.simulateFromList(temperatures, settings.SIMULATION_INTERVAL)