import torch
import copy
import numpy as np
from sklearn import metrics


class NeuralGCAnalyzer(object):
    
    def __init__(self, S, d):
        self.S = S # analyzed time series 
        self.d = d # directory name
        
        
    def normalize(self):
        S_centered = self.S - np.mean(self.S, axis = 0)
        sigma = np.sqrt(np.var(self.S, axis = 0))
        
        return np.divide(S_centered, sigma)
    
    def format_ts_data(self):
        N_train = self.N - self.lag
        
        X_train = np.zeros((N_train, self.D * self.lag))
        Y_train = np.zeros((N_train, self.D))

        for t in range(self.lag, self.N):
            X_train[t - self.lag, :] = self.S[range(t - self.lag, t), :].flatten(order = 'F')
            Y_train[t - self.lag, :] = self.S[t, :]
        
        return X_train, Y_train
    
    
    def tensorize_sequence(self,X, window = 20, stride = None):
        if stride is None:
            stride = window
    
        sequence_list = []
        T, p = X.shape
        start = 0
        end = window
        while end < T:
            tensor = np.zeros((window, 1, p))
            tensor[:, 0, :] = X[range(start, end), :]
            sequence_list.append(tensor)
    
            start += stride
            end += stride
    
        return np.concatenate(sequence_list, axis = 1)
    
    def apply_penalty(self, W):
        
        if self.penalty_type == 'group_lasso':
            group_loss = [torch.norm(W[:, (i * self.lag):((i + 1) * self.lag)], p = 2) for i in range(self.D)]  
            total = sum(group_loss)
        elif self.penalty_type == 'hierarchical':
            hierarchical_loss = [torch.norm(W[:, (i * self.lag):((i + 1) * self.lag - j)], p = 2) for i in range(self.D) for j in range(self.lag)]  
            total = sum(hierarchical_loss)
        elif self.penalty_type == 'stacked':
            column_loss = [torch.norm(W[:, i], p = 2) for i in range(self.lag * self.D)]  
            group_loss = [torch.norm(W[:, (i * self.lag):((i + 1) * self.lag)], p = 2) for i in range(self.D)]  
            total = sum(column_loss) + sum(group_loss)
        else:
            raise ValueError('unsupported penalty')

        return total
    
    def prox_operator(self, W):
        if self.penalty_type == 'group_lasso':
            self._prox_group_lasso(W)
            
        elif self.penalty_type == 'hierarchical':
            self._prox_hierarchical(W)
            
        elif self.penalty == 'stacked':
            self._prox_stacked(W)
            
        else:
            raise ValueError('unsupported penalty')
    
    def _prox_group_lasso(self, W):
        '''
            Apply prox operator
        '''
        C = W.data.numpy()
        h, l = C.shape
        C = np.reshape(C, newshape = (self.lag * h, self.D), order = 'F')
        C = self._prox_update(C)
        C = np.reshape(C, newshape = (h, l), order = 'F')

        W.data = torch.from_numpy(C)  

    def _prox_hierarchical(self, W):
        ''' 
            Apply prox operator
        '''
        C = W.data.numpy()
        
        h, l = C.shape
        C = np.reshape(C, newshape = (self.lag * h, self.D), order = 'F')
        
        for i in range(1, self.lag + 1):
            end = i * h
            temp = C[range(end), :]
            C[range(end), :] = self._prox_update(temp)
        
        C = np.reshape(C, newshape = (h, l), order = 'F')
    
        W.data = torch.from_numpy(C)  


    def _prox_stacked(self, W):
        '''
            Apply prox operator
        '''
        C = W.data.numpy()
    
        h, l = C.shape
        C = np.reshape(C, newshape = (self.lag * h, self.D), order = 'F')
        for i in range(self.lag):
            start = i * h
            end = (i + 1) * h
            temp = C[range(start, end), :]
            C[range(start, end), :] = self._prox_update(temp)

        C = self._prox_update(C)
        C = np.reshape(C, newshape = (h, l), order = 'F')
    
        W.data = torch.from_numpy(C)

    def _prox_update(self, W):
        '''
            Apply prox operator to a matrix, where columns each have group lasso penalty
        '''
        norm_value = np.linalg.norm(W, axis = 0, ord = 2)
        norm_value_gt = norm_value >= (self.lam * self.lr)

        W[:, np.logical_not(norm_value_gt)] = 0.0
        W[:, norm_value_gt] = W[:, norm_value_gt] * (1 - np.divide(self.lam * self.lr, norm_value[norm_value_gt][np.newaxis, :]))
        
        return W
    
    
    def optimizeAUC(self, weights_est, A_true):
        pastCalls = {}
        delta = 0.1
        q = np.arange(0,1,0.05) 
        
        " calculate initial AUC "
        best_q = q[0]
        best_AUC = self.computeAUC(weights_est, q[0], A_true)
        
        for i in q:
            AUC = self.computeAUC(weights_est, i, A_true)
            if AUC > best_AUC:
                best_AUC = AUC
                best_q = i

            #print ("AUC %f" %AUC)
        
        #print (best_q)
        
        return best_q     
                    
    def computeAUC(self, weights_est, q, A_true):
        
        w_est = copy.deepcopy(weights_est)
    
        thres = np.quantile(w_est, q)

        w_est[w_est < thres] = 0.0

        for d in range(self.D):
            w_est[d,d] = 0.0
                
        A_est = np.zeros((self.D, self.D))
        A_est[w_est > 0.0] = 1
        
        for d in range(self.D):
            A_true[d, d] = -1
            A_est[d, d] = -1
    
        A_true_temp = np.reshape(A_true, newshape=(1, self.D*self.D))[0]
        A_est_temp = list(np.reshape(A_est, newshape=(1, self.D*self.D))[0])
    
        A_true_temp = list(filter(lambda a: a != -1, A_true_temp))
        A_est_temp = list(filter(lambda a: a != -1, A_est_temp))
        
        fpr, tpr, thresholds = metrics.roc_curve(A_true_temp, A_est_temp)
    
        return metrics.auc(fpr, tpr)
        
        
        
        
