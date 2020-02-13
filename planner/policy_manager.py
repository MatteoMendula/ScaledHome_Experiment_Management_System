import planner_utils


#  TODO
# 0- Loop over these for t seconds
# 1- Get the state of the house
# 2- Decide on what to do: If current temp < desired temp -> turn on the heater and vice versa for AC.
# 3- Send the decision to the house
# 4- After t seconds, return the amount of time that heater and AC were on and difference between desired temp
# and achieved temp at every timestep we collected data.

class PolicyManager(object):
    def __init__(self, model_as_file_location, policy):
        self.model_as_file_location = model_as_file_location
        self.sh_state = planner_utils.get_sh_initial_state()
        self.policy = policy

        ''' 
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            EG POLICY
            {
                mode: keep_temperature_still,
                target: 27 celsius degree,
                duration: 10 min
            }
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # '''


    def update_sh_last_state(self):
        self.sh_state = planner_utils.get_last_state()


    ''' 
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    actions should be [0,1]
    keys have to be ['fan', 'heater', 'lamp', 'ac', 'motorX']
    eg 
    {
        'fan': 1
        'motor11':0
    }
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ''' 
    def perform_actions_by_dictionary(self, actions_dictionary):
        for action_key in actions_dictionary:
            action = actions_dictionary[action_key]
            if 'lamp' in action_key:
                planner_utils.handle_lamp(action)
            elif 'ac' in action_key:
                planner_utils.handle_ac(action)
            elif 'heater' in action_key:
                planner_utils.handle_heater(action)
            elif 'fan' in action_key:
                planner_utils.handle_fan(action)
            elif 'motor' in action_key:
                planner_utils.handle_motor(action, action_key.plit('motor')[1])

    def run(self):
        if self.policy['mode'] == 'keep_temperature_still':
            # 0. load the model
            # 1. read sh state
            # 2. create new dictionary of actions
            # 3. perform dictionary of actions with perform_actions_by_dictionary
            # 4. if simulation time < self.policy['duration'] go to 1 