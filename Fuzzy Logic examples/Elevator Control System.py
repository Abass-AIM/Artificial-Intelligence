import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt

# Define antecedents and consequent
waiting_passengers = ctrl.Antecedent(np.arange(0, 11, 1), 'waiting_passengers')
current_load = ctrl.Antecedent(np.arange(0, 7, 1), 'current_load')
priority = ctrl.Consequent(np.arange(0, 11, 1), 'priority')

# Membership functions for waiting_passengers
waiting_passengers['few'] = fuzz.trimf(waiting_passengers.universe, [0, 0, 5])
waiting_passengers['some'] = fuzz.trimf(waiting_passengers.universe, [0, 5, 10])
waiting_passengers['many'] = fuzz.trimf(waiting_passengers.universe, [5, 10, 10])

# Membership functions for current_load
current_load['light'] = fuzz.trimf(current_load.universe, [0, 0, 3])
current_load['moderate'] = fuzz.trimf(current_load.universe, [0, 3, 6])
current_load['heavy'] = fuzz.trimf(current_load.universe, [3, 6, 6])

# Membership functions for priority
priority['low'] = fuzz.trimf(priority.universe, [0, 0, 5])
priority['medium'] = fuzz.trimf(priority.universe, [0, 5, 10])
priority['high'] = fuzz.trimf(priority.universe, [5, 10, 10])

# Define the fuzzy rules
rule1 = ctrl.Rule(waiting_passengers['few'] & current_load['light'], priority['low'])
rule2 = ctrl.Rule(waiting_passengers['few'] & current_load['moderate'], priority['low'])
rule3 = ctrl.Rule(waiting_passengers['few'] & current_load['heavy'], priority['medium'])
rule4 = ctrl.Rule(waiting_passengers['some'] & current_load['light'], priority['medium'])
rule5 = ctrl.Rule(waiting_passengers['some'] & current_load['moderate'], priority['medium'])
rule6 = ctrl.Rule(waiting_passengers['some'] & current_load['heavy'], priority['high'])
rule7 = ctrl.Rule(waiting_passengers['many'] & current_load['light'], priority['high'])
rule8 = ctrl.Rule(waiting_passengers['many'] & current_load['moderate'], priority['high'])
rule9 = ctrl.Rule(waiting_passengers['many'] & current_load['heavy'], priority['high'])

# Create the control system
priority_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
priority_sim = ctrl.ControlSystemSimulation(priority_ctrl)

# Set input values
priority_sim.input['waiting_passengers'] = 7  # Example: 7 waiting passengers
priority_sim.input['current_load'] = 2       # Example: Current load is 2

# Compute the output
priority_sim.compute()

# Print the output priority
print("Priority:", priority_sim.output['priority'])

# Visualize the results
waiting_passengers.view(sim=priority_sim)
current_load.view(sim=priority_sim)
priority.view(sim=priority_sim)