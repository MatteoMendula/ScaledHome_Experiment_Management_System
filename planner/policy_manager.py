import planner_utils
from datetime import datetime, timedelta
import time

#  TODO
# 0- Loop over these for t seconds
# 1- Get the state of the house
# 2- Decide on what to do: If current temp < desired temp -> turn on the heater and vice versa for AC.
# 3- Send the decision to the house
# 4- After t seconds, return the amount of time that heater and AC were on and difference between desired temp
# and achieved temp at every timestep we collected data.

class PolicyManager(object):
    #def __init__(self, model_as_file_location, policy):
    def __init__(self):
        #self.model_as_file_location = model_as_file_location
        self.sh_state = planner_utils.get_sh_initial_state()
        self.target = 22


    def update_sh_last_state(self):
        self.sh_state = planner_utils.get_last_state()

    # state = 0 -> turn off
    # state = 1 -> turn on
    # actuators= [fan, heater, ac, lamp]
    def handle_actuators(self, actuator_id, next_state):
        if 'motor' in actuator_id and self.sh_state['motors'][actuator_id] != next_state:
            planner_utils.handle_motor(next_state, actuator_id.split('motor')[1])
            self.sh_state['motors'][actuator_id] = next_state
        else:
            if actuator_id == 'lamp' and self.sh_state[actuator_id] != next_state:
                planner_utils.handle_lamp(next_state)
            elif actuator_id == 'fan' and self.sh_state[actuator_id] != next_state:
                planner_utils.handle_fan(next_state)
            elif actuator_id == 'ac' and self.sh_state[actuator_id] != next_state:
                planner_utils.handle_ac(next_state)
            elif actuator_id == 'heater' and self.sh_state[actuator_id] != next_state:
                planner_utils.handle_heater(next_state)
            self.sh_state[actuator_id] = next_state


    def run(self, min_duration):
        sleep_time = 1  # seconds
        begin_time = datetime.now()
        ac_energy_consumption = 0
        heater_energy_consumption = 0
        planner_utils.handle_ac(0)
        planner_utils.handle_heater(0)
        while True:
            current_time = datetime.now()
            if current_time - begin_time > timedelta(minutes=min_duration):
                break
            self.update_sh_last_state()
            if float(self.sh_state['sensors']['sensor_6']['temperature']) < self.target:
                self.handle_actuators('motor15', 0)
                self.handle_actuators('ac', 0)
                self.handle_actuators('heater', 1)
            else:
                self.handle_actuators('motor15', 1)
                self.handle_actuators('ac', 1)
                self.handle_actuators('heater', 0)

                
            if self.sh_state['heater'] == 1:
                heater_energy_consumption += sleep_time
            if self.sh_state['ac'] == 1:
                ac_energy_consumption += sleep_time

            time.sleep(sleep_time)
        print(ac_energy_consumption)
        print(heater_energy_consumption)

if __name__ == '__main__':
    pm = PolicyManager()
    pm.run(1)