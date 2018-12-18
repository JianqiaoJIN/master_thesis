Anomaly Detection and Exploratory Causal Analysis for SAP HANA 
-----------------------------
Welcome. This repository contains the codes and implementations of my master thesis. My master thesis is mainly concentrated on two topics: anomaly detection for streaming data and Granger causality analysis among multivariate time series data. 


#### Anomaly Detection
An online algorithm called OPUAD (Online Prototypes Update Anomaly Detetion) is proposed to detect anomalies for streaming applications in an unsupervised, automated fashion without supervison, which makes it self-adaptive and computationally efficient. The algorithm is evaluated by the Numenta Anomaly Benchmark (NAB) https://github.com/numenta/NAB. 

##### Dependencies
Python is chosen for implementation and the following dependencies are required to implement the OPUAD algorithm:
- Python 2.7
- numpy
- scikit-learn
- pandas
- nupic 

##### Scoreboard

The NAB scores are normalized such that the maximum possible is 100.0 (i.e. the perfect detector), and a baseline of 0.0 is determined by the "null" detector (which makes no detections).

| Detector      | Standard Profile | Reward Low FP | Reward Low FN |
|---------------|------------------|---------------|---------------|
| Perfect       | 100.0            | 100.0         | 100.0         |
| [Numenta HTM](https://github.com/numenta/nupic) | 70.5-69.7     | 62.6-61.7     | 75.2-74.2     |
| [CAD OSE](https://github.com/smirmik/CAD) | 69.9          | 67.0          | 73.2          |
| [OPUAD](https://github.com/JianqiaoJIN/master_thesis/blob/master/anomalyDetection/opuad_detector.py)* | 64.5      | 59.5        | 68.0          |
| [KNN CAD](https://github.com/numenta/NAB/tree/master/nab/detectors/knncad) | 58.0     | 43.4  | 64.8     |
| [Relative Entropy](http://www.hpl.hp.com/techreports/2011/HPL-2011-8.pdf) | 54.6 | 47.6 | 58.8 |
| [Random Cut Forest](http://proceedings.mlr.press/v48/guha16.pdf)| 51.7 | 38.4 | 59.7 |
| [Twitter ADVec v1.0.0](https://github.com/twitter/AnomalyDetection)| 47.1             | 33.6          | 53.5          |
| [Windowed Gaussian](https://github.com/numenta/NAB/blob/master/nab/detectors/gaussian/windowedGaussian_detector.py) | 39.6             | 20.9         | 47.4          |
| [Etsy Skyline](https://github.com/etsy/skyline) | 35.7             | 27.1          | 44.5          |
| Bayesian Changepoint         | 17.7              | 3.2           | 32.2           |
|  [EXPoSE](https://arxiv.org/abs/1601.06602v3)   | 16.4     | 3.2  | 26.9     |
| Null          | 0.0              | 0.0           | 0.0           |

\* If you want to test the OPUAD algorithm by NAB, please adapt it to the requirements of the NAB at first.Here no details will be given. 
