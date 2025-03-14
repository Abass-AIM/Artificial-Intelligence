import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt

# Define antecedents and consequent
pedestrian_count = ctrl.Antecedent(np.arange(0, 31, 1), 'pedestrian_count')
traffic_density = ctrl.Antecedent(np.arange(0, 6, 1), 'traffic_density')
crossing_time = ctrl.Consequent(np.arange(0, 61, 1), 'crossing_time')

# Define membership functions for pedestrian_count
pedestrian_count['low'] = fuzz.trimf(pedestrian_count.universe, [0, 0, 15])
pedestrian_count['medium'] = fuzz.trimf(pedestrian_count.universe, [10, 15, 25])
pedestrian_count['high'] = fuzz.trimf(pedestrian_count.universe, [15, 30, 30])

# Define membership functions for traffic_density
traffic_density['low'] = fuzz.trimf(traffic_density.universe, [0, 0, 3])
traffic_density['medium'] = fuzz.trimf(traffic_density.universe, [2, 3, 4])
traffic_density['high'] = fuzz.trimf(traffic_density.universe, [3, 5, 5])

# Define membership functions for crossing_time
crossing_time['short'] = fuzz.trimf(crossing_time.universe, [0, 0, 30])
crossing_time['medium'] = fuzz.trimf(crossing_time.universe, [15, 30, 45])
crossing_time['long'] = fuzz.trimf(crossing_time.universe, [30, 60, 60])

# Define rules
rule1 = ctrl.Rule(pedestrian_count['low'] & traffic_density['low'], crossing_time['short'])
rule2 = ctrl.Rule(pedestrian_count['medium'] | traffic_density['medium'], crossing_time['medium'])
rule3 = ctrl.Rule(pedestrian_count['high'] & traffic_density['high'], crossing_time['long'])
rule4 = ctrl.Rule(pedestrian_count['low'] & traffic_density['high'], crossing_time['medium'])
rule5 = ctrl.Rule(pedestrian_count['high'] & traffic_density['low'], crossing_time['medium'])

# Create control system
crossing_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])

# Create control system simulation
crossing_simulation = ctrl.ControlSystemSimulation(crossing_control)

# Provide inputs
crossing_simulation.input['pedestrian_count'] = 5
crossing_simulation.input['traffic_density'] = 1

# Compute the output
crossing_simulation.compute()

# Print the resulting crossing time
print(f"Crossing time: {crossing_simulation.output['crossing_time']:.2f} seconds")

# Visualize membership functions and output
pedestrian_count.view()
traffic_density.view()
crossing_time.view(sim=crossing_simulation)
plt.show()