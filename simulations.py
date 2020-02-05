import settings
import utils
import time
import json

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