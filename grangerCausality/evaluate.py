import pandas as pd
import numpy as np
from sklearn import metrics
import json, pickle, copy

def writeJSON(filePath, data):
    """Dumps data to a nicely formatted json at filePath."""
    with open(filePath, "w") as outFile:
        outFile.write(json.dumps(data,
                                 sort_keys=True,
                                 indent=4,
                                 separators=(',', ': ')))




"""
    drop A[i,i]
"""

def preProcess(A_true, A_est):
    A_true = A_true.values
    A_est = A_est.values
    
    
    r,c = A_true.shape
    for i in range(r):
        A_true[i, i] = -1
        A_est[i, i] = -1
    
    A_true_temp = np.reshape(A_true, newshape=(1, r*c))[0]
    A_est_temp = list(np.reshape(A_est, newshape=(1, r*c))[0])
    
    A_true_temp = list(filter(lambda a: a != -1, A_true_temp))
    A_est_temp = list(filter(lambda a: a != -1, A_est_temp))
    
    return A_true_temp, A_est_temp


"""
    calculate AUC 
"""
def compute_AUC(A_true, A_est):

    A_true, A_est = preProcess(A_true, A_est)
    
    fpr, tpr, thresholds = metrics.roc_curve(A_true, A_est)
    
    return metrics.auc(fpr, tpr)

"""
    calculate F1-score
"""
    
def compute_F1Score(A_true, A_est):

    A_true, A_est = preProcess(A_true, A_est)
    
    #print ("F1: %f" % f1_score(A_est, A_true, average= "binary"))
    return f1_score(A_est, A_true, average= "binary")

if __name__ == "__main__":
    
    GC_types = ["bivariateGC","conditionalGC","groupLassoGC", "mlpGC"]
    dirs = ['var_system', 'henon_system']
    N_set = [1000]
    D_set = [5, 30]

    
    results_AUC = {}
    for d in dirs:
        D_GC_AUC = {}
        for D in D_set:
            N_D_GC = {}
            for N in N_set:
                A_true = pd.read_csv("data/"+d+"/A_true_"+str(D) + "_" + str(N)+".csv")
                N_D_GC_AUC = {}
                for GC in GC_types:  
                    file_name = "results/"+d+"/"+GC+"/A_est_"+ str(D) + "_" + str(N) +".csv"
                    A_est = pd.read_csv(file_name)
                    AUC = compute_AUC(A_true, A_est)
                    N_D_GC_AUC[GC] = AUC
                N_D_GC[N] = N_D_GC_AUC
            D_GC_AUC[D] = N_D_GC
        
        results_AUC[d] = D_GC_AUC
    
    writeJSON("results_auc.json", results_AUC)
    

                
