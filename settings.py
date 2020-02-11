from my_settings import *


TIME_TO_SIMULATE_ONE_INTERVAL = 20

# defining the api-endpoint  
API_ENDPOINT = "http://localhost:3000/pythonAPI"
API_KEY = "scaledHomeUcf"

HYSTERESIS_VALUE = 1

# in seconds (eg 20min*60 = 1200)
# (7min*60 = 420)
# (10min*60 = 600)
SIMULATION_INTERVAL = 600


INPUT_FEATURE_NAMES = [
    "OUT_T[*C]",
    "OUT_H[%]",
    "T6[*C]",
    "H6[%]",
    "T12[*C]",
    "H12[%]",
    "T18[*C]",
    "H18[%]",
    "T19[*C]",
    "H19[%]",
    "T24[*C]",
    "H24[%]",
    "T25[*C]",
    "H25[%]",
    "T26[*C]",
    "H26[%]",
    "LAMP_STATE",
    "FAN_STATE",
    "AC_STATE",
    "HEATER_STATE",
    "M0",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M8",
    "M9",
    "M10",
    "M11",
    "M12",
    "M13",
    "M14",
    "M15",
]


TARGET_FEATURE_NAMES = ["T6[*C]", "T12[*C]", "T18[*C]", "T19[*C]", "T24[*C]", "T25[*C]", "T26[*C]"]