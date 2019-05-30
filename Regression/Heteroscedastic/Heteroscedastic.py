"""
Created on wed Nov 2018

@author: Yibo Yang
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from models import Regression
from plotting import newfig, savefig
    
if __name__ == "__main__":
    
    N = 200
    X_dim = 1
    Y_dim = 1
    Z_dim = 1
    err_var = 0.5
    
    lb = 0.0*np.ones((1,X_dim))
    ub = 1.0*np.ones((1,X_dim)) 
    
    # Generate training data
    X = np.linspace(-2., 2.,N)[:,None]
    def f(X):
        return np.log(10.0*(abs(X-0.03)+0.03))*np.sin(np.pi*(abs(X-0.03)+0.03))
   
    error = 1.0/np.exp(2.0*(abs(X-0.03)+0.03))*np.random.normal(0,err_var,X.size)[:,None]

    Y = f(X) + error
    
    # Generate test data
    N_star = 2000
    X_star = lb + (ub-lb)*np.linspace(-2.0,2.,N_star)[:,None]
    Y_star = f(X_star)
    
    # Model creation
    layers_P = np.array([X_dim+Z_dim,100,100,100,Y_dim])
    layers_Q = np.array([X_dim+Y_dim,100,100,100,Z_dim])  
    layers_T = np.array([X_dim+Y_dim,100,100,1])       
    model = Regression(X, Y, layers_P, layers_Q, layers_T, lam = 2.5)
        
    model.train(nIter = 20000, batch_size = N)
        
    # Prediction
    plt.figure(1)
    N_samples = 500
    samples_mean = np.zeros((X_star.shape[0], N_samples))
    for i in range(0, N_samples):
        samples_mean[:,i:i+1] = model.generate_sample(X_star)
        plt.plot(X_star, samples_mean[:,i:i+1],'k.', alpha = 0.005)
    plt.plot(X, Y, 'r*',alpha = 0.2, label = '%d training data' % N)
        
    mu_pred = np.mean(samples_mean, axis = 1)    
    Sigma_pred = np.var(samples_mean, axis = 1)
#

    plt.figure(2, figsize=(6, 5.5), facecolor='w', edgecolor='k')  
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.plot(X_star, Y_star, 'b-', label = "Exact", linewidth=2)
    plt.plot(X_star, mu_pred, 'r--', label = "Prediction", linewidth=2)
    lower = mu_pred - 2.0*np.sqrt(Sigma_pred)
    upper = mu_pred + 2.0*np.sqrt(Sigma_pred)
    plt.fill_between(X_star.flatten(), lower.flatten(), upper.flatten(), 
                     facecolor='orange', alpha=0.5, label="Two std band")
    plt.plot(X, Y, 'kx', alpha = 0.5, label = 'Training Data', markersize = 4)
    plt.xlabel('$x$',fontsize=18)
    plt.ylabel('$f(x)$',fontsize=18)
    plt.legend(loc='upper left', frameon=False, prop={'size': 10.5, 'weight': 'extra bold'})
    plt.savefig('./Heter.png', dpi = 600)
    

    
   
    
    