import settings
import utils
import time
import json

# this function takes as input a list of different temperatures to reach and 
# the time (in seconds) interval that can be used to reach the temperatures 

def increaseTemp():
    print("Increasing temperature")
    utils.handleLamp(1)
    utils.handleFan(0)

def decreaseTemp():
    print("Decreasing temperature")
    utils.handleLamp(0)
    utils.handleFan(1)

def simulateFromList(temperatures_list, interval):
    # print(json.loads(utils.getLastState())["sensors"]["outside"]["temperature"])
    for temperature in temperatures_list:
        seconds = 0
        while (seconds < interval):
            parsed_json = (json.loads(utils.getLastState()))
            outside_temperature = float(parsed_json["sensors"]["outside"]["temperature"])
            print("Got temperature ",outside_temperature)
            if (outside_temperature <= temperature - settings.HYSTERESIS_VALUE):
                increaseTemp()
            elif (outside_temperature >= temperature + settings.HYSTERESIS_VALUE):
                decreaseTemp()
            time.sleep(1)
            seconds += 1    