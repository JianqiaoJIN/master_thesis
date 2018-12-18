Anomaly Detection and Exploratory Causal Analysis for SAP HANA 
-----------------------------
Welcome. This repository contains the codes and implementations of my master thesis. My master thesis is mainly concentrated on two topics: anomaly detection for streaming data and Granger causality analysis among multivariate time series data. 


### Anomaly Detection
An online algorithm called OPUAD (Online Prototypes Update Anomaly Detetion) is proposed to detect anomalies for streaming applications in an unsupervised, automated fashion without supervison, which makes it self-adaptive and computationally efficient. The algorithm is evaluated by the Numenta Anomaly Benchmark (NAB) https://github.com/numenta/NAB. 

#### Dependencies
Python is chosen for implementation and the following dependencies are required to implement the OPUAD algorithm:
- Python 2.7
- numpy
- scikit-learn
- pandas
- nupic 

#### Scoreboard

The NAB scores are normalized such that the maximum possible is 100.0 (i.e. the perfect detector), and a baseline of 0.0 is determined by the "null" detector (which makes no detections).

| Detector      | Standard Profile | Reward Low FP | Reward Low FN |
|---------------|------------------|---------------|---------------|
| Perfect       | 100.0            | 100.0         | 100.0         |
| [Numenta HTM](https://github.com/numenta/nupic) | 70.5-69.7     | 62.6-61.7     | 75.2-74.2     |
| [CAD OSE](https://github.com/smirmik/CAD) | 69.9          | 67.0          | 73.2          |
| [OPUAD](https://github.com/JianqiaoJIN/master_thesis/tree/master/anomalyDetection)* | 64.5      | 59.5        | 68.0          |
| Null          | 0.0              | 0.0           | 0.0           |

\* The OPUAD algorithm takes the third place, here only shows other two outstanding algorithms, the full scoreborad you can see https://github.com/numenta/NAB

\* If you want to test the OPUAD algorithm by NAB, please adapt it to the requirements of the NAB at first. 
