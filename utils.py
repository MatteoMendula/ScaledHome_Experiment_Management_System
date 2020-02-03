import requests

import settings

api_endpoint = settings.API_ENDPOINT
api_key = settings.API_KEY

# state = 0 -> turn off
# state = 1 -> turn on
def handleLamp(state):
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
def handleFan(state):
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
def handleAc(state):
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
def handleHeater(state):
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
def handleMotor(state, id):
    # value = "heater on" if state==1 else "heater off"
    value = ("open " if state==1 else "close ") + id
    data = {    
        "key": api_key, 
        "type": "cmd",
        "value": value
    }
    response = requests.post(url = api_endpoint, data = data)  
    pastebin_url = response.text 
    print("[Handle motors action: {0}] Reponse is: {1}".format(value,pastebin_url)) 

# this function returns the last record collected by SH_middlware
def getLastState():
    data = {    
            "key": api_key, 
            "type": "request",
            "value": "last record"
        }
    print(data)
    response = requests.post(url = api_endpoint, data = data)  
    pastebin_url = response.text 
    print("[GetLastState action] Reponse is: {0}".format(pastebin_url)) 

# this function returns all the records collected by SH_middleware since it has been activated
def getAllRecordsCollected():
    data = {    
                "key": api_key, 
                "type": "request",
                "value": "all records collected as string"
            }
    response = requests.post(url = api_endpoint, data = data)  
    pastebin_url = response.text 
    print("[getAllRecordsCollected action] Reponse is: {0}".format(pastebin_url)) 


