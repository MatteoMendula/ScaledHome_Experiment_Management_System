import settings
import utils
import time
import json
import os
import datasets

import numpy as np

def increaseTemp(lamp_state, fan_state):
    # print("Increasing temperature")
    if lamp_state == 0:
        utils.handleLamp(1)
    if fan_state == 1:
        utils.handleFan(0)

def decreaseTemp(lamp_state, fan_state):
    # print("Decreasing temperature")
    if lamp_state == 1:
        utils.handleLamp(0)
    if fan_state == 0:
        utils.handleFan(1)

def getOutTempLampAndFanState():
    parsed_json = json.loads(utils.getLastState())
    outside_temperature = round(float(parsed_json["sensors"]["outside"]["temperature"]))
    lamp_state = int(parsed_json["lamp"])
    fan_state = int(parsed_json["fan"])
    return outside_temperature, lamp_state, fan_state

# from typing import List
# def simulateFromList(temperatures_list: List[int], interval: int) -> None:

# this function takes as input a list of different temperatures to reach and 
# the time (in seconds) interval that can be used to reach the temperatures 
def simulateFromList(temperatures_list, interval):
    # print(json.loads(utils.getLastState())["sensors"]["outside"]["temperature"])
    for temperature in temperatures_list:
        print("Simulating temperature {0} in list {1}".format(str(temperature),str(temperatures_list)))
        index = 0
        seconds = 0
        outside_temperature,_,_ = getOutTempLampAndFanState()
        temperature_rounded = round(temperature)
        control = round(temperature) == outside_temperature
        while (seconds < interval or (index == 0 and not control)):
            outside_temperature, lamp_state, fan_state = getOutTempLampAndFanState()
            # print("Got temperature ",outside_temperature)
            if (outside_temperature <= temperature_rounded - settings.HYSTERESIS_VALUE):
                increaseTemp(lamp_state, fan_state)
            elif (outside_temperature >= temperature_rounded + settings.HYSTERESIS_VALUE):
                decreaseTemp(lamp_state, fan_state)
            # control = round(temperature) in range(outside_temperature-settings.HYSTERESIS_VALUE,outside_temperature+settings.HYSTERESIS_VALUE+1)
            control = temperature_rounded == outside_temperature
            time.sleep(1)
            seconds += 1    
        index += 1

def simulateRealDataTemperatures():
    dataset_address = os.path.join(settings.PROJECT_ROOT_ADDRESS, 'data/mi_meteo_2001.csv')

    steps_per_day = 8
    intervals = 24 // steps_per_day

    ds = datasets.convert_dataset_to_scale_home_temps(dataset_address, 23, 38, intervals=intervals)

    dict_of_days_temperatures = datasets.unzipAndCreateDictOfLists(ds,steps_per_day)

    # print(len(dict_of_days_temperatures.keys()))

    for day,temperatures in dict_of_days_temperatures.items():
        print("Day {0} - temperatures: {1}".format(day,temperatures))
        simulateFromList(temperatures, settings.SIMULATION_INTERVAL)

def generateRandomActuatorsActions(n_actuators):
    return np.random.randint(2, size=n_actuators)

actuators_switcher = {
    0: lambda state: utils.handleLamp(state),
    1: lambda state: utils.handleFan(state),
    2: lambda state: utils.handleAc(state),
    3: lambda state: utils.handleHeater(state),
    4: lambda state: utils.handleMotor(state, 0),
    5: lambda state: utils.handleMotor(state, 1),
    6: lambda state: utils.handleMotor(state, 2),
    7: lambda state: utils.handleMotor(state, 3),
    8: lambda state: utils.handleMotor(state, 4),
    9: lambda state: utils.handleMotor(state, 5),
    10: lambda state: utils.handleMotor(state, 6),
    11: lambda state: utils.handleMotor(state, 8),
    12: lambda state: utils.handleMotor(state, 9),
    13: lambda state: utils.handleMotor(state, 10),
    14: lambda state: utils.handleMotor(state, 11),
    15: lambda state: utils.handleMotor(state, 12),
    16: lambda state: utils.handleMotor(state, 13),
    17: lambda state: utils.handleMotor(state, 14),
    18: lambda state: utils.handleMotor(state, 15),
}

def runRandomActions(random_actions):
    for index, state in enumerate(generateRandomActuatorsActions(random_actions)):
        handleActuator = actuators_switcher.get(index, lambda: "Invalid month")
        handleActuator(state)

def performRandomActionsNtimes(n_times, interval):
    for _ in range(0,n_times):
        runRandomActions(19)
        time.sleep(interval)

def performRandomActionsFixedDuration(simulation_duration, interval):
    seconds = 0
    while ((seconds+interval) < simulation_duration):
        runRandomActions(19)
        time.sleep(interval)
        seconds += interval

if __name__ == '__main__':
    # runRandomActions(19)
    performRandomActionsNtimes(3,10)