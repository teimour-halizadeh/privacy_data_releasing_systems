
import numpy as np
import scipy.linalg as LA



class SystemModel():

    def __init__(self, n, R_min, R_max, M_min, M_max, tav, x0_min, x0_max, K):
            # The adjacency graph subroutine, here it is a star graph
        In = np.eye(n)
        e1 = In[:,[0]]
        em1 = np.vstack((0, R_min + (R_max-R_min)*np.random.rand(n-1,1)))
        Adj_room = e1 @ em1.T + em1 @ e1.T
        D = Adj_room @ np.ones((n,1))
        Lap_room =  np.diag(D[:,0]) - Adj_room  
        # M matrix in the model
        D = M_min + (M_max-M_min)*np.random.rand(n,1)
        M = np.diag(D[:,0])

        # The final A matrix in linear system
        self.A = np.eye(n) - tav * LA.inv(M) @ Lap_room

        # system outputs, they measure the temperature of the first and the last zone.
        self.C = np.vstack((In[[0],:], In[[n-1],:]))

        # initial condition 
        self.x_0 = x0_min + (x0_max-x0_min)*np.random.rand(n,1)

        # observability matrix
        self.O_K = np.vstack([self.C] + [self.C @ np.linalg.matrix_power(self.A, i)
                                    for i in range(1, K)])
        # observability gramian
        self.W_O = (self.O_K.T) @ self.O_K 

        













