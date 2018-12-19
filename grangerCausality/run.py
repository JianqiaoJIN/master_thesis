import torch
import pandas as pd
import numpy as np
from Python.mlpGC_analyzer import MLPGCAnalyzer

def oneFileProcess(d, D, N, lag=5):

    file_name = "data/" + d + "/system_" + str(D) + "_" +str(N) +".csv"
    
    S = pd.read_csv(file_name) 
    S = S.values
    
    print ("***---" + file_name + "---***")
    print ("Multilayer perceptron Granger causality analysis")
    mlpGC_instance = MLPGCAnalyzer(S, d, lag)
    mlpGC_instance.analyze()
    mlpGC_instance.saveResults()
    print ("File " + file_name + " processing complete.")


if __name__ == "__main__":

    torch.manual_seed(12345)
        
    
    dirs = ['var_system','henon_system']
    N_set = [1000]
    D_set = [5, 30]
    
    for d in dirs:
        for D in D_set:
            for N in N_set:
                oneFileProcess(d, D, N)


