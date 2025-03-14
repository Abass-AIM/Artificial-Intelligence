import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
#We define three fuzzy variables: temperature (input), fan speed (output), and their membership functions.
# Define fuzzy variables
temperature = ctrl.Antecedent(np.arange(0, 101, 1), 'temperature')
fan_speed = ctrl.Consequent(np.arange(0, 101, 1), 'fan_speed')
# Define membership functions
temperature['cold'] = fuzz.trimf(temperature.universe, [0, 0, 50])
temperature['warm'] = fuzz.trimf(temperature.universe, [20, 50, 80])
temperature['hot'] = fuzz.trimf(temperature.universe, [50, 100, 100])
fan_speed['low'] = fuzz.trimf(fan_speed.universe, [0, 0, 50])
fan_speed['medium'] = fuzz.trimf(fan_speed.universe, [20, 50, 80])
fan_speed['high'] = fuzz.trimf(fan_speed.universe, [50, 100, 100])
#We create simple if-then rules for the fuzzy system:
rule1 = ctrl.Rule(temperature['cold'], fan_speed['low'])
rule2 = ctrl.Rule(temperature['warm'], fan_speed['medium'])
rule3 = ctrl.Rule(temperature['hot'], fan_speed['high'])
#Now, we create a control system and simulate it
fan_control = ctrl.ControlSystem([rule1, rule2, rule3])
fan_simulation = ctrl.ControlSystemSimulation(fan_control)
# Test the system with a specific temperature value
fan_simulation.input['temperature'] = 40
fan_simulation.compute()
print(f"Fan speed: {fan_simulation.output['fan_speed']:.2f}%")
#To visualize how the fuzzy membership functions behave, we use matplotlib:
import matplotlib.pyplot as plt
temperature.view()
fan_speed.view()
plt.show()