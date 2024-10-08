import  numpy as np
import scipy.linalg as LA








def costume_noise(s, noise_len):
    R_sigma = LA.cholesky(s)
    N = R_sigma.T @ np.random.randn(noise_len,1)
    return N
