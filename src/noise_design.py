
import numpy as np
import scipy.linalg as LA
import cvxpy as cp






class NoiseDesign():


    def __init__(self, A, C, K):
        self.n = np.size(A,0)
        self.p = np.size(C, 0)
        self.K = K
    
        
        self.O_K = np.vstack([C] + [C @ np.linalg.matrix_power(A, i)
                                    for i in range(1, K)])
        self.W_O = (self.O_K.T) @ self.O_K 


    def solve_cvx(self, sigma_v):

        #these two matrices come from the note,
        #we need them for toptimization
        M = self.O_K @ LA.inv(self.W_O) @ self.O_K.T
        NTN = self.O_K @ LA.inv(self.W_O) @ LA.inv(sigma_v) @ LA.inv(self.W_O) @ self.O_K.T

        # Define variables R, and beta
        R = cp.Variable((self.p*self.K, self.p*self.K), symmetric=True)
        beta = cp.Variable(1)

        # Form objective.
        obj = cp.Maximize(beta)

        # Form the constraints
        constraints = [M @ R @ M - R - NTN + beta * np.eye(self.p*self.K)<<0]

        # Form and solve problem.
        prob = cp.Problem(obj, constraints)
        prob.solve(solver='MOSEK', eps=1e-6) 
    
        #the value of Sigma inverse
        sigma_inv =   (-M @ (R.value) @ M + (R.value) + NTN)
        #The covariance of the noise
        sigma = LA.inv (sigma_inv)

        return prob.status, beta.value, sigma

    def solve_sdp_diff_entropy(self,  Eps_err):
        # Define variables S and W
        S = cp.Variable((self.p * self.K, self.p * self.K), symmetric=True)
        W = cp.Variable((self.n, self.n), symmetric=True)

        # Define the objective: maximize(log_det(W))
        obj = cp.Maximize(cp.log_det(W))

        # Define the constraints
        Z = cp.bmat([[W, np.zeros((self.n, (self.p * self.K) - self.n))],
                    [np.zeros(((self.p * self.K) - self.n, self.n)),
                      np.zeros(((self.p * self.K) - self.n, (self.p * self.K) - self.n))]])

        constraints = [
            Z - S << 0,                          # Z - S <= 0
            -W + 0.005 * np.eye(self.n) << 0,           # W > 0
            -S + 0.005 * np.eye(self.p * self.K) << 0,         # S > 0
            cp.trace(S) <= Eps_err               # trace(S) <= Eps_err
        ]

        # Define and solve the problem
        prob = cp.Problem(obj, constraints)
        prob.solve(solver='MOSEK', eps=1e-6)

        return prob.status,  S.value









