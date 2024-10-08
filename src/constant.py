import numpy as np 
import scipy.linalg as LA


# number of zones in the HVAC system
n= 4 
# the range for the R_ij parameter 
R_min = 0.4; R_max= 0.6 
# the range for the M matrix
M_min = 950; M_max = 1050

# time constant for discretization
tav = 360

# number of time steps that you want your model goes ahead in time
K = 10



#initial condition vector, we need this at the end for simulation
x0_min = 10; x0_max= 25




sigma_v = np.diag(np.array([16, 16, 100, 100]))


