import numpy as np
import skfuzzy as sk
#Defining the Numpy array for Tip Quality
x_qual = np.arange(0, 11, 1)
#Defining the Numpy array for Triangular membership functions
qual_lo_1 = sk.trimf(x_qual, [0, 5, 10])

#Defining the Numpy array for Trapezoidal membership functions
qual_lo_2 = sk.trapmf(x_qual, [0, 3, 6,10])

#Defining the Numpy array for Gaussian membership functions
qual_lo_3 = sk.gaussmf(x_qual, np.mean(x_qual), np.std(x_qual))

#Defining the Numpy array for Generalized Bell membership functions
qual_lo_4 = sk.gbellmf(x_qual, 0.5, 0.5, 0.5)

#Defining the Numpy array for Sigmoid membership functions
qual_lo_5 = sk.sigmf(x_qual, 0.5,0.5)