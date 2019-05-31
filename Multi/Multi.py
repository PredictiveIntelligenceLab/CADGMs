"""
Created on wed Nov 2018

@author: Yibo Yang
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from models import CADGM_Multi
from plotting import newfig, savefig

np.random.seed(1234)

if __name__ == "__main__":

    # Load the training data
    data = np.load('MuFi_data.npy').item()
    X = data['X']
    f_H = data['f_H']
    f_L = data['f_L']

    Y_dim = 1
    X_dim = 1
    Z_dim = 1

    idx = [0, 80, 120, 200]
    N_H = X.shape[1] * len(idx)

    X_H = X[idx, :]
    Y_H = f_H[idx, :]
    Y_L = f_L[idx, :]
    X_H = X_H.reshape(-1, 1)
    Y_H = Y_H.reshape(-1, 1)
    Y_L = Y_L.reshape(-1, 1)

    # High fidelity
    def ff_H(x):
        return (6.0*x-2.0)**2 * np.sin(12.*x-4.0)
    
    # Low fidelity
    def ff_L(x):
        return 0.5*ff_H(x) + 10.0*(x-0.5) - 5.0

    X_star = X[:,0:1]
    X_L = X[:,0:1]
    Y_star_H = ff_H(X_L)
    Y_star_L = ff_L(X_L)

    # Model creation
    layers_P = np.array([X_dim+Z_dim+Y_dim,100,100,100,Y_dim])
    layers_Q = np.array([X_dim+Y_dim+Y_dim,100,100,100,Z_dim])  
    layers_T = np.array([X_dim+Y_dim,100,100,1])       
    model = CADGM_Multi(X_H, Y_H, Y_L, layers_P, layers_Q, layers_T, lam = 1.5)
        
    model.train(nIter = 20000, batch_size = N_H)
        
    # Prediction
    plt.figure(1)
    N_samples = 500
    samples_mean = np.zeros((X_star.shape[0], N_samples))
    for i in range(0, N_samples):
        idx = np.random.permutation(f_L.shape[1])[0]
        samples_mean[:,i:i+1] = model.generate_sample(X_star, f_L[:,idx:idx+1])
        plt.plot(X_star, samples_mean[:,i:i+1],'k.', alpha = 0.005)
    plt.plot(X_H, Y_H, 'r*',alpha = 0.2, label = '%d training data' % N_H)
        
    mu_pred = np.mean(samples_mean, axis = 1)    
    Sigma_pred = np.var(samples_mean, axis = 1)

    # Reference mean and variance
    F_H = samples_mean[:,0:50]
    Ref_mean = np.mean(f_H, axis = 1)
    Ref_std = np.var(f_H, axis = 1)


    plt.figure(10, figsize=(6, 5), facecolor='w', edgecolor='k')  
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.plot(X_star, Ref_mean, 'b-', label = "Exact", linewidth=2)
    lower = Ref_mean - 2.0*np.sqrt(Ref_std)
    upper = Ref_mean + 2.0*np.sqrt(Ref_std)
    plt.fill_between(X_star.flatten(), lower.flatten(), upper.flatten(), 
                     facecolor='green', alpha=0.2, label="Real Two std band")
    plt.plot(X_star, mu_pred, 'r--', label = "Prediction", linewidth=2)
    lower = mu_pred - 2.0*np.sqrt(Sigma_pred)
    upper = mu_pred + 2.0*np.sqrt(Sigma_pred)
    plt.fill_between(X_star.flatten(), lower.flatten(), upper.flatten(), 
                     facecolor='orange', alpha=0.5, label="Predicted Two std band")
    plt.xlabel('$x$',fontsize=13)
    plt.ylabel('$f(x)$',fontsize=13)
    plt.legend(loc='upper left', frameon=False, prop={'size': 13})
    plt.savefig('./Multi.png', dpi = 600)
    
########## Compare the prediction ###########

    plt.figure(5, figsize=(6, 5), facecolor='w', edgecolor='k')  
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.plot(X, f_H, 'b-')
    plt.plot(X[0,0], f_H[0,0], 'b-', label = "Exact samples", alpha = 1.)
    plt.plot(X_H, Y_H, 'rx', markersize = 4)
    plt.plot(X_H[0,0], Y_H[0,0], 'rx', label = "Training samples", alpha = 1.)
    plt.xlabel('$x$',fontsize=13)
    plt.ylabel('$f(x)$',fontsize=13)
    plt.legend(loc='upper left', frameon=False, prop={'size': 13})
    plt.savefig('./Multi_com1.png', dpi = 600)


    plt.figure(6, figsize=(6, 5), facecolor='w', edgecolor='k')  
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.plot(X, f_H, 'b-', alpha = 0.5)
    plt.plot(X[0,0], f_H[0,0], 'b-', label = "Exact samples", alpha = 0.5)
    plt.plot(X, F_H, 'r-', alpha = 0.5)
    plt.plot(X[0,0], F_H[0,0], 'r-', label = "Generated samples", alpha = 0.5)
    plt.xlabel('$x$',fontsize=13)
    plt.ylabel('$f(x)$',fontsize=13)
    plt.legend(loc='upper left', frameon=False, prop={'size': 13})
    plt.savefig('./Multi_com2.png', dpi = 600)




