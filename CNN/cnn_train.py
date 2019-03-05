# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 14:57:43 2019

@author: sunhu
"""

import pickle 
import numpy as np
import solver
import cnn


with open('mnist.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')
print(data[0][0].shape)

N = np.shape(data[0][0])[0]
N_val = np.shape(data[1][0])[0]
N = 50000
N_val = 2000

x_train = np.reshape(data[0][0][0:N,:], (N,1,28,28))
x_val =np.reshape(data[1][0][0:N_val,:], (N_val,1,28,28))

N_test = np.shape(data[2][0])[0]
x_test = np.reshape(data[2][0], (N_test,1,28,28))
y_test =data[2][1]

# CNN
data_input = {
    'X_train': x_train,
    'y_train': data[0][1][0:N],
    'X_val':  x_val,
    'y_val': data[1][1][0:N_val]
}

model = cnn.ConvNet(input_dim=(1, 28, 28), num_filters = 16, filter_size=5,
                                  hidden_dim=128, num_classes=10, reg = 0.7)
solver = solver.Solver(model, data_input,
                update_rule='sgd_momentum',
                optim_config={
                  'learning_rate': 0.08,
                  "momentum" : 0.8,
                  'velocity' : 0
                },
                lr_decay=0.97,
                num_epochs=20, batch_size=25,
                print_every=10)

solver.train()
print(solver.check_accuracy(x_test, y_test, num_samples=None, batch_size=40))
#print(solver.best_params)

