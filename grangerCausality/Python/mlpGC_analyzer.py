import os
import pickle
import pandas as pd
import numpy as np
import copy
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from Python.base import NeuralGCAnalyzer

class MLPGCAnalyzer(NeuralGCAnalyzer):
    
    """ ---- initial MLPGCAnalyzer ---- """
    " # normalize time series (z-score)"
    " # prepare train data "
    " # set up network for each time series "
    
    def __init__(self, S, d, lag):
        
        super(MLPGCAnalyzer, self).__init__(S, d)
        
        # normalize time series 
        self.S = self.normalize()

        # prepare train data 
        self.N, self.D = self.S.shape # N: length, D: dimension
        self.lag = lag # network considered lag
        X_train, Y_train = self.format_ts_data()
        self.X_var = Variable(torch.from_numpy(X_train).float())  
        self.Y_var = [Variable(torch.from_numpy(Y_train[:, target][:, np.newaxis]).float()) for target in range(self.D)]
        
        # set up network for each time series 
        self.hidden_units = [10]
        self.nonlinearity = 'sigmoid'
        self.layers = list(zip([self.D * self.lag] + self.hidden_units[:-1], self.hidden_units))
        self.sequentials = [self.setNetwork() for _ in range(self.D)]
        self.sequential_copy = copy.deepcopy(self.sequentials[0])
        
        # initialize parameters for calculating loss function
        self.loss_fn = nn.MSELoss() # loss function 
        self.weight_decay = 0.01  # weight_decay for ridge penalty
        self.penalty_type = "group_lasso" # lasso penalty type 
        self.lam = 0.1 # weight decay for lasso penalty
        
    def analyze(self):
        
        verbose = True 
        
        nepoch =  1000 # number of training epochs
        loss_check = 50 # interval for checking loss
        nchecks = max(int(nepoch / loss_check), 1)
        
        # Prepare for training
        #train_loss = np.zeros((nchecks, self.D))
        train_objective = np.zeros((1, self.D))
        #counter = 0
        improvement = True
        epoch = 0
        last_error = 100 # initial error
        
        # Begin training
        while epoch < nepoch and improvement:
            improvement = self.train()
            train_objective[0,:] = self._objective()
            error = np.mean(train_objective)
              
            # Check progress
            if (epoch+1) % loss_check == 0:
                # save results 
                #train_loss[counter, :] = self._loss()
                #train_objective[counter,:] = self._objective()
   
                # Print results
                if verbose:
                    print('----------')
                    print('epoch %d' % (epoch+1))
                    print('train loss = %e' % error)
                    print('----------')

            # termination or no ? 
            
            #if last_error - error < 0.000001:
                #break
            
            last_error = error     
            epoch += 1
        
        if verbose:
            print('Done training')

        weights = self.get_weights()
        weights_est = [np.linalg.norm(np.reshape(w, newshape = (self.hidden_units[0] * self.lag, self.D), order = 'F'), axis = 0) for w in weights]
        
        with open("results/" +  self.d + "/" +  "mlpGC/" + "weights_est_" + str(self.D) + "_" + str(self.N), 'wb') as f:
            pickle.dump(weights_est, f)
    
    
    def saveResults(self):
        with open("results/" + self.d + "/" + "mlpGC/" + "weights_est_" + str(self.D) + "_" + str(self.N) , 'rb') as f:
            weights_est = pickle.load(f)
        
        weights_est = np.array(weights_est)
        for i in range(self.D):
            weights_est[i,i] = 0.0

        # get A_true
        file_name = "data/" + self.d + "/" + "/A_true_" + str(self.D) + "_" + str(self.N) + ".csv"
        A_true = pd.read_csv(file_name).values

        # get the optimal q which can maximize the AUC (evaluation criteria)
        q = self.optimizeAUC(weights_est, A_true)
        
        thres = np.quantile(weights_est, q)
        weights_est[weights_est < thres] = 0.0
        
        for i in range(self.D):
            weights_est[i,i] = 0.0
            
        A_ = np.zeros((self.D, self.D))
        A_[weights_est > 0.0] = 1

        col_names = ['Z_' + str(d) for d in range(1, self.D+1)]
        A_  = pd.DataFrame(A_, columns = col_names, dtype = "int32")
        W = pd.DataFrame(weights_est, columns = col_names)

        A_.to_csv("results/"+ self.d + "/mlpGC/A_est_" + str(self.D) + "_" + str(self.N) + ".csv", index = False)
        W.to_csv("results/"+ self.d + "/mlpGC/W_est_" + str(self.D) + "_" + str(self.N) + ".csv", index = False)
            
    
    def train(self):
        
        "calculate loss"
        loss = self._loss()
        ridge = self._ridge()

        total_loss = sum(loss) + sum(ridge)
        
        " calculate lasso penalty "
        penalty = self._lasso()
       
        [net.zero_grad() for net in self.sequentials]
        total_loss.backward()
        
        " line search"
        t = 0.3
        s = 0.8
        min_lr = 1e-18

        # Return value, to indicate whether improvements have been made
        return_value = False
        
        # a new network 
        new_net = self.sequential_copy
        new_net_params = list(new_net.parameters())
        
        # Set up initial learning rate (step size t(k)), objective function value to beat
        self.lr = 1
        for target, net in enumerate(self.sequentials):

            original_objective = loss[target] + ridge[target] + penalty[target]
            original_net_params = list(net.parameters())
            
            
            while self.lr > min_lr:
                # Take gradient step in new params
                for params, o_params in zip(new_net_params, original_net_params):
                    params.data = o_params.data - o_params.grad.data * self.lr
                
                
                # group lasso -> Apply proximal operator to new params (update params)
                self.prox_operator(new_net_params[0])
                
                # Compute objective function using new params
                Y_pred = new_net(self.X_var)
                new_objective = self.loss_fn(Y_pred, self.Y_var[target]) # lasso
                new_objective += self.weight_decay * torch.sum(new_net_params[2]**2) #ridge 
                new_objective += self.lam * self.apply_penalty(new_net_params[0]) # lasso
                
                
                diff_squared = sum([torch.sum((o_params.data - params.data)**2) for (params, o_params) in zip(new_net_params, original_net_params)])
                
                diff_squared = diff_squared.float()
                                
                if new_objective.data.numpy() < original_objective.data.numpy() - t * self.lr * diff_squared.data.numpy():
                    # Replace parameter values
                    for params, o_params in zip(new_net_params, original_net_params):
                        o_params.data = params.data 
                    
                    return_value = True
                    break
                    
                else:
                    # Try a lower learning rate
                    self.lr *= s
                
            # Update initial learning rate for next training iteration
            self.lr = np.sqrt(self.lr * self.lr)

        return return_value
           
    def get_weights(self, p = None):
        if p is None:
            return [list(net.parameters())[0].data.numpy().copy() for net in self.sequentials]
        else:
            return list(self.sequentials[p].parameters())[0].data.numpy().copy()
    
    def _loss(self):
        " calculate loss (MSE)"
        Y_pred = [net(self.X_var) for net in self.sequentials]
        return [self.loss_fn(Y_pred[target], self.Y_var[target]) for target in range(self.D)]
    
    def _ridge(self):
        " calculate ridge penalty "
        return [self.weight_decay * torch.sum(list(net.parameters())[2]**2) for net in self.sequentials]
    
    def _lasso(self):
        "calculate lasso penalty"
        return [self.lam * self.apply_penalty(list(net.parameters())[0]) for net in self.sequentials]
    
    def _objective(self):
        loss = self._loss()
        ridge = self._ridge()
        penalty = self._lasso()
        return [l + p + r for (l, p, r) in zip(loss, penalty, ridge)]
             
    def setNetwork(self):
        net = nn.Sequential()
        
        for i, (d_in, d_out) in enumerate(self.layers):
            net.add_module('fc%d' % i, nn.Linear(d_in, d_out, bias = True))
            if self.nonlinearity == 'relu':
                net.add_module('relu%d' % i, nn.ReLU())
            elif self.nonlinearity == 'sigmoid':
                net.add_module('sigmoid%d' % i, nn.Sigmoid())
            elif self.nonlinearity is not None:
                raise ValueError('nonlinearity must be "relu" or "sigmoid"')
        net.add_module('out', nn.Linear(self.hidden_units[-1], 1, bias = True))
        
        return net
    
        