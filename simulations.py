import settings
import utils
import time
import json

def increaseTemp(lamp_state, fan_state):
    print("Increasing temperature")
    if lamp_state == 0:
        utils.handleLamp(1)
    if fan_state == 1:
        utils.handleFan(0)

def decreaseTemp(lamp_state, fan_state):
    print("Decreasing temperature")
    if lamp_state == 1:
        utils.handleLamp(0)
    if fan_state == 0:
        utils.handleFan(1)

# from typing import List
# def simulateFromList(temperatures_list: List[int], interval: int) -> None:

# this function takes as input a list of different temperatures to reach and 
# the time (in seconds) interval that can be used to reach the temperatures 
def simulateFromList(temperatures_list, interval):
    # print(json.loads(utils.getLastState())["sensors"]["outside"]["temperature"])
    for temperature in temperatures_list:
        seconds = 0
        while (seconds < interval):
            parsed_json = (json.loads(utils.getLastState()))
            lamp_state = int(parsed_json["lamp"])
            fan_state = int(parsed_json["fan"])
            # print(parsed_json)
            outside_temperature = float(parsed_json["sensors"]["outside"]["temperature"])
            print("Got temperature ",outside_temperature)
            if (outside_temperature <= temperature - settings.HYSTERESIS_VALUE):
                increaseTemp(lamp_state, fan_state)
            elif (outside_temperature >= temperature + settings.HYSTERESIS_VALUE):
                decreaseTemp(lamp_state, fan_state)
            time.sleep(1)
            seconds += 1    