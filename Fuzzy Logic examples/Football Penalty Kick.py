#Importing Necessary Packages
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

#Defining the Fuzzy Range from a speed of 30 to 90
x = np.arange(30, 80, 0.1)
#Defining the triangular membership functions
slow = fuzz.trimf(x, [30, 30, 50])
medium = fuzz.trimf(x, [30, 50, 70])
medium_fast = fuzz.trimf(x, [50, 60, 80])
full_speed = fuzz.trimf(x, [60, 80, 80])
#Plotting the Membership Functions Defined
plt.figure()
plt.plot(x, full_speed, 'b', linewidth=1.5, label='Full Speed')
plt.plot(x, medium_fast, 'k', linewidth=1.5, label='Medium Fast')
plt.plot(x, medium, 'm', linewidth=1.5, label='Medium Powered')
plt.plot(x, slow, 'r', linewidth=1.5, label='Slow')
plt.title('Penalty Kick Fuzzy')
plt.ylabel('Membership')
plt.xlabel("Speed (Miles Per Hour)")
plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5),
ncol=1, fancybox=True, shadow=True)
plt.show()

