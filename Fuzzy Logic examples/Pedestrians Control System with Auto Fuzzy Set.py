import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Define the fuzzy input and output variables
pedestrian_count = ctrl.Antecedent(np.arange(1, 31, 1), 'pedestrian_count')  # Input: Number of pedestrians
traffic_density = ctrl.Antecedent(np.arange(1, 6, 1), 'traffic_density')     # Input: Number of cars in traffic
crossing_time = ctrl.Consequent(np.arange(1, 61, 1), 'crossing_time')        # Output: Crossing time in seconds

# Automatically generate three fuzzy sets (poor, average, good)
pedestrian_count.automf(3, names=['poor', 'average', 'good'])
traffic_density.automf(3, names=['low', 'average', 'high'])
crossing_time.automf(3, names=['poor', 'average', 'good'])

# Define fuzzy rules
#rule1 = ctrl.Rule(pedestrian_count['poor'] & traffic_density['low'], crossing_time['poor'])   # Few pedestrians and low traffic -> Short crossing time
#rule2 = ctrl.Rule(pedestrian_count['average'] | traffic_density['average'], crossing_time['average'])  # Moderate pedestrians and traffic -> Medium crossing time

rule1 = ctrl.Rule(pedestrian_count['poor'], crossing_time['poor']) # Few pedestrians -> Short crossing time
rule2 = ctrl.Rule(pedestrian_count['average'], crossing_time['average']) # Moderatepedestrians -> Medium crossing time
rule3 = ctrl.Rule(pedestrian_count['good'], crossing_time['good']) # Many pedestrians -> Long crossing time
rule4 = ctrl.Rule(pedestrian_count['good'] & traffic_density['high'], crossing_time['good'])   # Many pedestrians and high traffic -> Long crossing time

# Create the fuzzy control system
crossing_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4])

# Create the fuzzy control system simulation
crossing_simulation = ctrl.ControlSystemSimulation(crossing_control)

# Input a test value for pedestrian count and traffic density
crossing_simulation.input['pedestrian_count'] = 30
crossing_simulation.input['traffic_density'] = 6

# Compute the fuzzy logic output
crossing_simulation.compute()
print(f"Crossing time: {crossing_simulation.output['crossing_time']:.2f} seconds")
crossing_time.view(sim=crossing_simulation)