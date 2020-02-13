import requests 
import json

import planner_settings

api_endpoint = planner_settings.API_URL
api_key = planner_settings.API_KEY

def get_sh_initial_state():
    return {
        'time': 'initial_state',
        'sensors': {
            'outside':{
                'temperature': 'initial_state',
                'humidity': 'initial_state'
            },
            'sensor_6':{
                'temperature': 'initial_state',
                'humidity': 'initial_state'
            },
            'sensor_12':{
                'temperature': 'initial_state',
                'humidity': 'initial_state'
            },
            'sensor_18':{
                'temperature': 'initial_state',
                'humidity': 'initial_state'
            },
            'sensor_19':{
                'temperature': 'initial_state',
                'humidity': 'initial_state'
            },
            'sensor_24':{
                'temperature': 'initial_state',
                'humidity': 'initial_state'
            },
            'sensor_25':{
                'temperature': 'initial_state',
                'humidity': 'initial_state'
            },
            'sensor_26':{
                'temperature': 'initial_state',
                'humidity': 'initial_state'
            }
        },
        'motors':{
            'motor0': 'initial_state',
            'motor1': 'initial_state',
            'motor2': 'initial_state',
            'motor3': 'initial_state',
            'motor4': 'initial_state',
            'motor5': 'initial_state',
            'motor6': 'initial_state',
            'motor8': 'initial_state',
            'motor9': 'initial_state',
            'motor10': 'initial_state',
            'motor11': 'initial_state',
            'motor12': 'initial_state',
            'motor13': 'initial_state',
            'motor14': 'initial_state',
            'motor15': 'initial_state'
        },
        'lamp': 'initial_state',
        'fan': 'initial_state',
        'ac': 'initial_state',
        'heater': 'initial_state'
    }

# state = 0 -> turn off
# state = 1 -> turn on
def handle_lamp(state):
    value = "lamp on" if state==1 else "lamp off"
    data = {    
        "key": api_key, 
        "type": "cmd",
        "value": value
    }
    response = requests.post(url = api_endpoint, data = data)  
    pastebin_url = response.text 
    print("[Handle lamp action: {0}] Reponse is: {1}".format(value,pastebin_url)) 

# state = 0 -> turn off
# state = 1 -> turn on
def handle_fan(state):
    value = "fan on" if state==1 else "fan off"
    data = {    
        "key": api_key, 
        "type": "cmd",
        "value": value
    }
    response = requests.post(url = api_endpoint, data = data)  
    pastebin_url = response.text 
    print("[Handle fan action: {0}] Reponse is: {1}".format(value,pastebin_url)) 

# state = 0 -> turn off
# state = 1 -> turn on
def handle_ac(state):
    value = "ac on" if state==1 else "ac off"
    data = {    
        "key": api_key, 
        "type": "cmd",
        "value": value
    }
    response = requests.post(url = api_endpoint, data = data)  
    pastebin_url = response.text 
    print("[Handle facan action: {0}] Reponse is: {1}".format(value,pastebin_url)) 

# state = 0 -> turn off
# state = 1 -> turn on
def handle_heater(state):
    value = "heater on" if state==1 else "heater off"
    data = {    
        "key": api_key, 
        "type": "cmd",
        "value": value
    }
    response = requests.post(url = api_endpoint, data = data)  
    pastebin_url = response.text 
    print("[Handle heater action: {0}] Reponse is: {1}".format(value,pastebin_url)) 

# state = 0 -> close motor
# state = 1 -> open motor
# id: identifier of a specific motor -> id E [0-6,8-15]U{"all"} where [0,6] are windows and [8-15] are doors 
def handle_motor(state, id):
    # value = "heater on" if state==1 else "heater off"
    value = ("open" if state==1 else "close") + " motor " + str(id)
    data = {    
        "key": api_key, 
        "type": "cmd",
        "value": value
    }
    response = requests.post(url = api_endpoint, data = data)  
    pastebin_url = response.text 
    print("[Handle motors action: {0}] Reponse is: {1}".format(value,pastebin_url)) 

# this function returns the last record collected by SH_middlware as a dict file
# eg outside_temperature = round(float(parsed_json["sensors"]["outside"]["temperature"]))
def get_last_state():
    data = {    
            "key": api_key, 
            "type": "request",
            "value": "last record"
        }
    # print(data)
    response = requests.post(url = api_endpoint, data = data)  
    pastebin_url = response.text 
    # print("[GetLastState action] Reponse is: {0}".format(pastebin_url)) 
    parsed_json = json.loads(pastebin_url)
    return parsed_json

# this function returns all the records collected by SH_middleware since it has been activated
def get_all_records_collected():
    data = {    
                "key": api_key, 
                "type": "request",
                "value": "all records collected as string"
            }
    response = requests.post(url = api_endpoint, data = data)  
    pastebin_url = response.text 
    # print("[getAllRecordsCollected action] Reponse is: {0}".format(pastebin_url)) 
    return pastebin_url