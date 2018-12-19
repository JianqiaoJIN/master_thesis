import os,math
import pandas as pd
from os import listdir
from opuad_detector import OPUAD


def oneFileDetection(f, T=0.9, SPATIAL_TOLERANCE = 0.02):
    
    print ("Process " + f + " now...")
    
    data = pd.read_csv("data/" + f, parse_dates=["timestamp"]) 
    probationaryPeriod = min(math.floor(0.15 * data.shape[0]),750)
    opuad_instance = OPUAD(T, SPATIAL_TOLERANCE, probationaryPeriod)
    
    headers = list(data.columns.values) + ['raw_score', 'anomaly_score'] 
    rows = []
    
    for i, row in data.iterrows():
        inputData = row.to_dict()  
        raw_score, anomaly_score = opuad_instance.handleRecord(inputData)
        
        outputRow = list(row) + [raw_score, anomaly_score]
        rows.append(outputRow)
        
    results = pd.DataFrame(rows, columns=headers)
    results.to_csv('results/opuad_'+f)
    
    print ("Process ends \n")


if __name__ == "__main__":
    """
        The format of the data must follows: [timestamp][value]
    """
    data_dir = 'data'
    files = os.listdir(data_dir)
    
    for f in files:
        oneFileDetection(f)
    
    
    
    
    

        