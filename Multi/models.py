"""
Created on wed Nov 2018

@author: Yibo Yang
"""


import tensorflow as tf
import numpy as np
import timeit

class CADGM_Multi:
    # Initialize the class
    def __init__(self, X, Y_H, Y_L, layers_P, layers_Q, layers_T, lam = 1.0):
                
        # Normalize data
        self.Xmean, self.Xstd = X.mean(0), X.std(0)
        self.Y_H_mean, self.Y_H_std = Y_H.mean(0), Y_H.std(0)
        self.Y_L_mean, self.Y_L_std = Y_L.mean(0), Y_L.std(0)
        X = (X - self.Xmean) / self.Xstd
        Y_H = (Y_H - self.Y_H_mean) / self.Y_H_std
        Y_L = (Y_L - self.Y_L_mean) / self.Y_L_std
     
        self.X = X
        self.Y_H = Y_H
        self.Y_L = Y_L
        
        self.layers_P = layers_P
        self.layers_Q = layers_Q
        self.layers_T = layers_T
        
        self.X_dim = X.shape[1]
        self.Y_dim = Y_H.shape[1]
        self.Z_dim = layers_Q[-1]
        self.lam = lam
        self.k1 = 1   #1
        self.k2 = 5   #5

        # Initialize network weights and biases        
        self.weights_P, self.biases_P = self.initialize_NN(layers_P)
        self.weights_Q, self.biases_Q = self.initialize_NN(layers_Q)
        self.weights_T, self.biases_T = self.initialize_NN(layers_T)
        
        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        
        # Define placeholders and computational graph
        self.X_tf = tf.placeholder(tf.float32, shape=(None, self.X_dim))
        self.Y_H_tf = tf.placeholder(tf.float32, shape=(None, self.Y_dim))
        self.Y_L_tf = tf.placeholder(tf.float32, shape=(None, self.Y_dim))
        self.Z_tf = tf.placeholder(tf.float32, shape=(None, self.Z_dim))
        
        # Generator loss
        self.G_loss, self.KL_loss, self.recon_loss  = self.compute_KL_loss(self.X_tf, self.Y_H_tf, self.Y_L_tf, self.Z_tf)
        
        # Discriminator loss
        self.T_loss  = self.compute_T_loss(self.X_tf, self.Y_H_tf, self.Y_L_tf, self.Z_tf)
        
        self.sample = self.sample_generator(self.X_tf, self.Z_tf, self.Y_L_tf)

        # Define optimizer        
        self.optimizer_KL = tf.train.AdamOptimizer(1e-4)
        self.optimizer_T = tf.train.AdamOptimizer(1e-4)
        
        # Define train Ops
        self.train_op_KL = self.optimizer_KL.minimize(self.G_loss, 
                                                      var_list = [self.weights_P, self.biases_P,
                                                                  self.weights_Q, self.biases_Q])
                                                                    
        self.train_op_T = self.optimizer_T.minimize(self.T_loss,
                                                    var_list = [self.weights_T, self.biases_T])

        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    
    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers):      
        # Xavier initialization
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
            return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev, dtype=tf.float32)   
        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
           
           
    # Evaluates the forward pass
    def forward_pass(self, H, layers, weights, biases):
        num_layers = len(layers)
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H
    
    
    # Decoder: p(y|x,z)
    def net_P(self, X, Z, Y_L):
        Y_H = self.forward_pass(tf.concat([X, Z, Y_L], 1),
                              self.layers_P,
                              self.weights_P,
                              self.biases_P)
        return Y_H
    
    # Encoder: q(z|x,y)
    def net_Q(self, X, Y_H, Y_L):
        Z = self.forward_pass(tf.concat([X, Y_H, Y_L], 1),
                              self.layers_Q,
                              self.weights_Q,
                              self.biases_Q)
        return Z
    
    # Discriminator
    def net_T(self, X, Y_H):
        T = self.forward_pass(tf.concat([X, Y_H], 1),
                              self.layers_T,
                              self.weights_T,
                              self.biases_T)        
        return T
    
    
    def compute_KL_loss(self, X, Y_H, Y_L, Z):  
        # Prior: p(z)
        z_prior = Z
        # Decoder: p(y|x,z)
        Y_pred = self.net_P(X, z_prior, Y_L)        
        # Encoder: q(z|x,y)
        z_encoder = self.net_Q(X, Y_pred, Y_L)
        # Discriminator loss
        T_pred = self.net_T(X, Y_pred)
        
        KL = tf.reduce_mean(T_pred)
        log_q = -tf.reduce_mean(tf.square(z_prior-z_encoder))

        loss = KL + (1.0-self.lam)*log_q
        
        return loss, KL, log_q
    
    
    def compute_T_loss(self, X, Y_H, Y_L, Z): 
        # Prior: p(z)
        z_prior = Z
        # Decoder: p(y|x,z)
        Y_pred = self.net_P(X, z_prior, Y_L)                
        
        # Discriminator loss
        T_real = self.net_T(X, Y_H)
        T_fake = self.net_T(X, Y_pred)
        
        T_real = tf.sigmoid(T_real)
        T_fake = tf.sigmoid(T_fake)
        
        T_loss = -tf.reduce_mean(tf.log(1.0 - T_real + 1e-8) + \
                                 tf.log(T_fake + 1e-8)) 
        
        return T_loss
           
    
    # Fetches a mini-batch of data
    def fetch_minibatch(self,X, Y_H, Y_L, N_batch):
        N = X.shape[0]
        idx = np.random.choice(N, N_batch, replace=False)
        X_batch = X[idx,:]
        Y_H_batch = Y_H[idx,:]
        Y_L_batch = Y_L[idx,:]
        return X_batch, Y_H_batch, Y_L_batch
    
    
    # Trains the model
    def train(self, nIter = 10000, batch_size = 100): 

        start_time = timeit.default_timer()        

        for it in range(nIter):     
            # Fetch a mini-batch of data
            X_batch, Y_H_batch, Y_L_batch = self.fetch_minibatch(self.X, self.Y_H, self.Y_L, batch_size)

            Z = np.random.randn(X_batch.shape[0], 1)
            # Define a dictionary for associating placeholders with data
            tf_dict = {self.X_tf: X_batch, self.Y_H_tf: Y_H_batch, self.Y_L_tf: Y_L_batch, self.Z_tf: Z}  
            
            # Run the Tensorflow session to minimize the loss
            for i in range(self.k1):
                self.sess.run(self.train_op_T, tf_dict)
            for j in range(self.k2):
                self.sess.run(self.train_op_KL, tf_dict)

            # Print
            if it % 100 == 0:
                elapsed = timeit.default_timer() - start_time
                loss_KL_value, reconv = self.sess.run([self.KL_loss, self.recon_loss], tf_dict)
                loss_T_value = self.sess.run(self.T_loss, tf_dict)
                
                print('It: %d, KL_loss: %.2e, Recon_loss: %.2e, T_loss: %.2e, Time: %.2f' % 
                      (it, loss_KL_value, reconv, loss_T_value, elapsed))
                start_time = timeit.default_timer()
               

    def sample_generator(self, X, Z, Y_L):        
        # Prior: p(z)
        z_prior = Z       
        # Decoder: p(y|x,z)
        Y_pred = self.net_P(X, z_prior, Y_L)      
        return Y_pred
    
    
    def generate_sample(self, X_star, Y_L):
        X_star = (X_star - self.Xmean) / self.Xstd
        Y_L = (Y_L - self.Y_L_mean) / self.Y_L_std

        Z = np.random.randn(1, 1) 
        Z = Z.repeat(X_star.shape[0], 1)
        Z = Z.T
        tf_dict = {self.X_tf: X_star, self.Y_L_tf: Y_L, self.Z_tf: Z}      
        Y_star = self.sess.run(self.sample, tf_dict) 
        # De-normalize outputs
        Y_star = Y_star * self.Y_H_std + self.Y_H_mean
        return Y_star

