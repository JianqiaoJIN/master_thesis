import math, copy
import numpy as np
import pandas as pd
from scipy.stats import norm
from nupic.algorithms import anomaly_likelihood

class OPUAD():
    def __init__(self, T, probationaryPeriod):
        # number of prototypes
        self.K = 5

        # number of data points that has been seen so far 
        self.record = 0
        
        # threshold
        self.T = T

        # probationaryPeriod
        self.probationaryPeriod = probationaryPeriod
        
        # initialize the anomaly likelihood object
        numentaLearningPeriod = int(math.floor(self.probationaryPeriod / 2.0))
        self.anomalyLikelihood = anomaly_likelihood.AnomalyLikelihood(
            learningPeriod=numentaLearningPeriod,
            estimationSamples=self.probationaryPeriod-numentaLearningPeriod,
            reestimationPeriod=100
        )
        
        # keep track of valid range for spatial anomaly detection
        self.SPATIAL_TOLERANCE = 0.05
        self.minVal = None
        self.maxVal = None
        
    def handleRecord(self, inputData):
        
        self.record += 1
        value = inputData['value']
        timestamp = inputData['timestamp']
                
        if (self.record == 1):
            """set prototypes"""
            self.q = np.repeat(value, self.K)
            self.sigma = abs(value * 0.1)
        
        """compute anomay score"""
        rawScore = self.calculateAnomalyScore(value)
    
        """compute anomaly likelihood"""     
        anomalyScore = self.anomalyLikelihood.anomalyProbability(
                value, rawScore, timestamp)

        logScore = self.anomalyLikelihood.computeLogLikelihood(anomalyScore)
        finalScore = logScore
    
        """check spatial anomaly for univariate time series"""
        # check if there is a spatial anomaly
        # update max and min
        spatialAnomaly = False
            
        if self.minVal != self.maxVal:
            tolerance = (self.maxVal - self.minVal) * self.SPATIAL_TOLERANCE
            maxExpected = self.maxVal + self.SPATIAL_TOLERANCE
            minExpected = self.minVal - self.SPATIAL_TOLERANCE

            if value > maxExpected or value < minExpected:
                spatialAnomaly = True
        
        if self.maxVal is None or value > self.maxVal:
            self.maxVal = value
        if self.minVal is None or value < self.minVal:
            self.minVal = value
            
        if spatialAnomaly:
            finalScore = 1.0
        
        if value == 0:
            finalScore = 1.0
            
        """ report anomaly """
        alertAnomaly = 0
        if self.record > self.probationaryPeriod and finalScore >= self.T:
            alertAnomaly = 1

        """ update prototypes """
        if (self.record > 1):
            self.updateParameter(value)

        return alertAnomaly
    
    def calculateAnomalyScore(self, value):

        density_sum = 0

        for i in range(self.K):
            density_sum = density_sum + norm.pdf(value, self.q[i], self.sigma) / self.K
        
        if density_sum == 0:
            anomalyScore = 100
        else:
            anomalyScore = - math.log(density_sum)

        return anomalyScore
    
    def updateParameter(self, value):    
        C = []
        B = []

        # l -> l-th prototype

        for l in range(self.K):
            tmp_C = []
            tmp_B = 0

            # k -> k-th prototype

            for k in range(self.K):
                tmp = (self.q[k] - self.q[l])*(self.q[k]-self.q[l])
                C_lk = (1-tmp/(2*self.sigma*self.sigma))*np.exp(-tmp/(4*self.sigma*self.sigma))
                tmp_C.append(C_lk)
                tmp_B = tmp_B + tmp*np.exp(-tmp/(4*self.sigma*self.sigma))

            C.append(tmp_C)
            B_l = (value-self.q[l])*np.exp(-(value-self.q[l])*(value-self.q[l])/(4*self.sigma*self.sigma))-tmp_B/self.K
            B_l = B_l * self.K / (self.record)
            B.append(B_l)

        # solve the linear equation
        try:
            delta_q = np.linalg.lstsq(C, B, rcond=None)[0]
        except np.linalg.linalg.LinAlgError:
            print "Singular Matrix"
        self.q = self.q + delta_q
