import os
import numpy as np
import pandas as pd

"""
    Generate a linear standardized VAR system
    input:
        N: time series length 
        D: dimension of system 
        lag: lag for VAR system
    
    output:
        S: simulated VAR system (N * D)
        A: adjacent matrix indicating the true connection     
           column: candidate (j)
           row: target (i)
           A[i,j] = 1: j Granger causes i
           A[i,j] = 0: j does not Granger causes i
"""

def standardized_var_system(N, D, lag=1):
    
    def stationary_var(beta, D, lag, radius):
        bottom = np.hstack((np.eye(D * (lag-1)), np.zeros((D * (lag - 1), D))))  
        beta_tilde = np.vstack((beta,bottom))
        eig = np.linalg.eigvals(beta_tilde)
        maxeig = max(np.absolute(eig))
        not_stationary = maxeig >= radius
    
        return beta * 0.95, not_stationary
    
    if D  == 5:
        sparsity = 0.5
    
    if D == 30:
        sparsity = 0.2
    
    beta_value = 5
    sd_e = 2.0
    radius = 0.97
    beta = np.eye(D) * beta_value
    A = np.zeros((D,D))

    # Set dependencies for each component
    num_nonzero = int(D * sparsity) - 1
    for i in range(D):
        choice = np.random.choice(D - 1, size = num_nonzero, replace = False)
        choice[choice >= i] += 1
        beta[i, choice] = beta_value
        A[i, choice] = 1

    # Create full beta matrix
    beta_full = beta
    for i in range(1, lag):
        beta_full = np.hstack((beta_full, beta))

    not_stationary = True
    while not_stationary:
        beta_full, not_stationary = stationary_var(beta_full, D, lag, radius)
    
    # create VAR model
    errors = np.random.normal(loc = 0, scale = sd_e, size = (D, N))
    S = np.zeros((D, N))
    S[:, range(lag)] = errors[:, range(lag)]
    for i in range(lag, N):
        S[:, i] = np.dot(beta_full, S[:, range(i - lag, i)].flatten(order = 'F')) + errors[:, i]
    
    return S.T, A


"""
    Generate a nonlinear driver-response Henon system 
    input:
        N: time series length 
        D: dimension of system 
    
    output:
        S: simulated Henon system (N * D)
        A: adjacent matrix indicating the true connection     
           column: candidate (j)
           row: target (i)
           A[i,j] = 1: j Granger causes i
           A[i,j] = 0: j does not Granger causes i
"""

def henon_system(N, D):
    
    sd_e = 1.0
    
    # create Henon syste
    S = np.random.uniform(0,1,(D,N))

    # head and end
    for d in [0,D-1]:
        for t in range(2, N):
            S[d, t] = 1.4 - np.square(S[d, t-1]) + 0.3*S[d, t-2]
    
    C = 1.0 # coupling strength 
    
    for d in range(1,D-1):
        for t in range(2,N):
            S[d, t] = 1.4 - np.square(0.5*C*(S[d-1,t-1] + S[d+1, t-1]) + (1-C)*S[d, t-1]) + 0.3*S[d, t-2]
    
    A = np.zeros((D,D))
    for d in range(1, D-1):
        A[d, d-1] = 1
        A[d, d+1] = 1
    
    return S.T, A


"""
    Save simulated system
    input:
        S: simulated system
        A: adjacent matrix indicating the true connection
        _dir: target directory to save the simulated system 
    
"""

def saveSystem(S, A, _dir):
    N,D = S.shape

    col_names = ['Z_' + str(d) for d in range(1, D+1)]
    S = pd.DataFrame(S, columns=col_names)
    A = pd.DataFrame(A, columns = col_names, dtype = "int32")
    
    N = str(N)
    D = str(D)
    S.to_csv(_dir+"/system"+ "_" + D + "_" + N + ".csv", index = False)
    A.to_csv(_dir+"/A" + "_true_" + D + "_" + N + ".csv", index = False)


if __name__ == "__main__":
    
    N_set = [1000] # length of time series 
    D_set = [5, 30] # dimension of time series 
    
    # generate simulated systems          
    np.random.seed(654)    
    
    dirs = ['var_system', 'henon_system']
    GC_analyzers = ["bivariateGC", "conditionalGC", "groupLassoGC", "mlpGC"]
    
    for d in dirs:
        data_d = "data/" + d
        if (not os.path.exists(data_d)):
            os.makedirs(data_d)
            
        for gc_analyzer in GC_analyzers:
            result_d_gc = "results/" + d + "/" + gc_analyzer
            if (not os.path.exists(result_d_gc)):
                os.makedirs(result_d_gc)
            
        if d == "var_system":
            for N in N_set:
                for D in D_set:
                    S, A = standardized_var_system(N,D)
                    saveSystem(S, A, data_d)
            
        elif d == "henon_system":
            for N in N_set:
                for D in D_set:
                    S, A = henon_system(N, D)
                    saveSystem(S, A, data_d) 
        
        else:
            raise ValueError("invalid system name")
    



    