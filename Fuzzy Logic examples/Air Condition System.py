import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt

# Define antecedents and consequent
room_temperature = ctrl.Antecedent(np.arange(10, 40, 1), 'room_temperature')
humidity_level = ctrl.Antecedent(np.arange(30, 90, 1), 'humidity_level')
ac_power = ctrl.Consequent(np.arange(0, 101, 1), 'ac_power')

# Define membership functions for pedestrian_count
room_temperature['low'] = fuzz.trimf(room_temperature.universe, [10, 10, 25])
room_temperature['medium'] = fuzz.trimf(room_temperature.universe, [20, 25, 30])
room_temperature['high'] = fuzz.trimf(room_temperature.universe, [25, 40, 40])

# Define membership functions for traffic_density
humidity_level['low'] = fuzz.trimf(humidity_level.universe, [30, 30, 60])
humidity_level['medium'] = fuzz.trimf(humidity_level.universe, [45, 60, 75])
humidity_level['high'] = fuzz.trimf(humidity_level.universe, [60, 90, 90])

# Define membership functions for crossing_time
ac_power['low'] = fuzz.trimf(ac_power.universe, [0, 0, 50])
ac_power['medium'] = fuzz.trimf(ac_power.universe, [25, 50, 75])
ac_power['high'] = fuzz.trimf(ac_power.universe, [50, 100, 100])

# Define rules
rule1 = ctrl.Rule(room_temperature['low'] & humidity_level['low'], ac_power['low'])
rule2 = ctrl.Rule(room_temperature['medium'] | humidity_level['medium'], ac_power['medium'])
rule3 = ctrl.Rule(room_temperature['high'] | humidity_level['high'], ac_power['high'])
rule4 = ctrl.Rule(room_temperature['low'] & humidity_level['high'], ac_power['medium'])
rule5 = ctrl.Rule(room_temperature['high'] & humidity_level['low'], ac_power['medium'])

# Create control system
ac_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])

# Create control system simulation
ac_simulation = ctrl.ControlSystemSimulation(ac_control)

# Provide inputs
ac_simulation.input['room_temperature'] = 32
ac_simulation.input['humidity_level'] = 70

# Compute the output
ac_simulation.compute()

# Print the resulting crossing time
print(f"AC Power: {ac_simulation.output['ac_power']:.2f} seconds")

# Visualize membership functions and output
room_temperature.view()
humidity_level.view()
ac_power.view(sim=ac_simulation)
plt.show()