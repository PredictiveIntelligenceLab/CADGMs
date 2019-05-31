"""
Created on wed Nov 2018

@author: Yibo Yang
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
plt.switch_backend('agg')
from models import CADGM_HD
from plotting import newfig, savefig
import h5py 

np.random.seed(1234)
    
if __name__ == "__main__":

    # Load training data
    train = np.load('DataBG64.npy').item()
    Y_u = train['Y']
    T_u = train['T']

    # Load testing data
    test = np.load('DataBG256.npy').item() 
    Y_t = test['Y']
    T_t = test['T']

    # Model creation
    model = CADGM_HD(Y_u, T_u, lam = 1.5, beta = 0.5)
    model.train(nIter = 30000)

    X = np.linspace(-7, 3, 128)[:,None]
    torch.save(model, 'model.pkl')

    # Domain bounds
    lb, ub = X.min(0), X.max(0)

    # Plot
    X_star = X

    idx = np.random.permutation(T_t.shape[2])[0]
    idx = 139

    t = T_t[idx * 100 + 50, :, :]
    print(t)
    N_samples = 1000
    uuu = np.zeros((N_samples, 1, 128))

    for i in range(0, N_samples):
        uuu[i:i+1,:] = model.predict(t)

    uuu_mu_pred = np.mean(uuu, axis = 0)
    uuu_Sigma_pred = np.var(uuu, axis = 0)
    Ref_mean = np.mean(Y_t[idx * 100: (idx+1) * 100,:,:], axis = 0)
    Ref_std = np.var(Y_t[idx * 100: (idx+1) * 100,:,:], axis = 0)

    U_mean = np.reshape(uuu_mu_pred, [128, 1])
    U_std = np.reshape(uuu_Sigma_pred, [128, 1])
    Ref_mean = np.reshape(Ref_mean, [128, 1])
    Ref_std = np.reshape(Ref_std, [128, 1])

    new_samples = np.reshape(uuu[0:10].T, (128, 10))
    # Compare the uncertainty versus the truth
    plt.figure(2, figsize=(6, 5), facecolor='w', edgecolor='k')  
    plt.plot(X_star, new_samples, 'k-')
    plt.plot(X_star, Ref_mean, 'b-', label = "Real")
    lower = Ref_mean - 2.0*np.sqrt(Ref_std)
    upper = Ref_mean + 2.0*np.sqrt(Ref_std)
    plt.fill_between(X_star.flatten(), lower.flatten(), upper.flatten(), 
                     facecolor='green', alpha=0.2, label="Two std band")
    plt.plot(X_star, U_mean, 'r--', label = "Prediction")
    lower = U_mean - 2.0*np.sqrt(U_std)
    upper = U_mean + 2.0*np.sqrt(U_std)
    plt.fill_between(X_star.flatten(), lower.flatten(), upper.flatten(), 
                     facecolor='orange', alpha=0.5, label="Two std band")
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend(loc='upper left')
    plt.savefig('./prediction.png', dpi = 600)



    # Load the reference data (1000 x 256 x 128 refer to number of realization, number of snap shot and number of spatial discretization)
    f = h5py.File('./Burgers.mat','r') 

    x = f.get('x').value
    t = f.get('t').value
    U = f.get('U').value

    skip = 1
    idx_t = np.arange(0, t.shape[0], skip)
    num_step = idx_t.shape[0]

    num_sample = 10
    T = t[idx_t].T
    T = np.tile(T, (num_sample, 1))
    T = T.reshape(num_sample * num_step,1,1, order = 'F')

    Y = np.zeros((num_sample * num_step, 1, U.shape[1]))

    for k in range(num_step):
        idx_u = np.random.permutation(U.shape[2])[0: num_sample]
        Y[k * num_sample: (k+1)*num_sample, 0, :] = U[idx_t[k], :, idx_u]
    x = np.linspace(-7,3, Y.shape[2])[:,None]
    

    # Plot
    uuu = np.zeros((num_sample * 256, 1, 128))
    for m in range(0, 256):
        for n in range(0, num_sample):
            t = T_t[m * 100 + 50, :, :]
            uuu[m*num_sample + n:m*num_sample + n+1,:] = model.predict(t)


    # Exact trajectories
    nmodes = 16
    x = np.linspace(-7,3, Y.shape[2])[:,None]

    fig = plt.figure(2, facecolor= None, frameon = False)
    ax = fig.add_subplot(111, projection='3d')
    for k in range(num_sample):
        step_index = 256 // nmodes * num_sample
        index = np.arange(0,nmodes) * step_index + k
        Y_temp1 = Y[index, :, :]
        Y_temp1 = Y_temp1.reshape(nmodes, 128)
        index = np.arange(0,nmodes) / nmodes * T[-1,0,0]
        XX, II = np.meshgrid(x,index)
        X_star = np.hstack((XX.flatten()[:,None], II.flatten()[:,None]))
        Y_star1 = Y_temp1.flatten()[:,None]
        Phi_plot1 = griddata(X_star, Y_star1.flatten(), (XX, II), method='cubic')
        ax.plot_wireframe(XX,II,Phi_plot1, rstride = 1, cstride = 0, color = 'k', alpha = 0.2)
        
    ax.plot_wireframe(XX,II,Phi_plot1, rstride = 1, cstride = 0, color = 'k', label = "Exact", alpha = 0.2, linewidth=1.0)
    ax.view_init(40, 60)
    plt.legend(loc='upper right', frameon=False, prop={'size': 13})
    ax.set_xlabel('$x$', fontsize=13)
    ax.set_ylabel('$t$', fontsize=13)
    ax.set_zlabel('$u$', fontsize=13)
    plt.savefig('./Exact_Burgers.png', dpi = 600)

    # Predicted trajectories
    nmodes = 16
    x = np.linspace(-7,3, Y.shape[2])[:,None]

    fig = plt.figure(3, facecolor= None, frameon = False)
    ax = fig.add_subplot(111, projection='3d')
    for k in range(num_sample):
        step_index = 256 // nmodes * num_sample
        index = np.arange(0,nmodes) * step_index + k
        Y_temp2 = uuu[index, :, :]
        Y_temp2 = Y_temp2.reshape(nmodes, 128)
        index = np.arange(0,nmodes) / nmodes * T[-1,0,0]
        XX, II = np.meshgrid(x,index)
        X_star = np.hstack((XX.flatten()[:,None], II.flatten()[:,None]))
        Y_star2 = Y_temp2.flatten()[:,None]
        Phi_plot2 = griddata(X_star, Y_star2.flatten(), (XX, II), method='cubic')
        ax.plot_wireframe(XX,II,Phi_plot2, rstride = 1, cstride = 0, color = 'r', alpha = 0.2, linewidth=1.0)
        
    ax.plot_wireframe(XX,II,Phi_plot2, rstride = 1, cstride = 0, color = 'r', label = "Prediction", alpha = 0.2)
    ax.view_init(40, 60)
    plt.legend(loc='upper right', frameon=False, prop={'size': 13})
    ax.set_xlabel('$x$', fontsize=13)
    ax.set_ylabel('$t$', fontsize=13)
    ax.set_zlabel('$u$', fontsize=13)
    plt.savefig('./Predicted_Burgers.png', dpi = 600)

	

