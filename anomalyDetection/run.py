import pandas as pd
from opuad_detector import OPUAD

if __name__ == "__main__":
    """
        The format of the data must follows: [timestamp][value]
    """
    exp = pd.read_csv("data/example.csv", parse_dates=["timestamp"]) 
    probationaryPeriod = 1000
    T = 0.9
    
    opuad_instance = OPUAD(T, probationaryPeriod)
    
    for i, row in exp.iterrows():
        inputData = row.to_dict()  
        alertAnomaly = opuad_instance.handleRecord(inputData)
        if alertAnomaly:
            print ("Alert: " + str(inputData["timestamp"]))
        