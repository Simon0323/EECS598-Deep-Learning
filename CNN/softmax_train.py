# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 22:20:07 2019

@author: sunhu
"""
import pickle 
import numpy as np
import solver
import softmax
import cnn


with open('mnist.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')

print(data[0][0].shape)

x_test = data[2][0]
y_test = data[2][1]
data_input = {
    'X_train': data[0][0],
    'y_train': data[0][1],
    'X_val':  data[1][0],
    'y_val': data[1][1]# validation labels
} 

#model = softmax.SoftmaxClassifier(input_dim=784, num_classes=10,reg=0.02)
model = softmax.SoftmaxClassifier(input_dim=784, hidden_dim=256, num_classes=10,reg=0.02)
solver = solver.Solver(model, data_input,
                update_rule='sgd_momentum',
                optim_config={
                  'learning_rate': 4e-1,
                  "momentum" : 0.8,
                  'velocity' : 0
                },
                lr_decay=0.98,
                num_epochs=20, batch_size=100,
                print_every=1000)

solver.train()
print(solver.check_accuracy(x_test, y_test, num_samples=None, batch_size=40))
#print(solver.best_params))