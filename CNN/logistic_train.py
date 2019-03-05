# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 23:09:26 2019

@author: sunhu
"""
import pickle
import numpy as np
import solver
import logistic
import svm
import softmax

with open('data.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')
print(np.shape(data[0][0:500, :]))


x_test = data[0][750:, :]
y_test = data[1][750:]

data_input = {
    'X_train': data[0][0:500, :],
    'y_train': data[1][0:500],
    'X_val':  data[0][500:750, :],
    'y_val': data[1][500:750]# validation labels
}

model = logistic.LogisticClassifier(input_dim=20, reg=0.12)   #lr=2
#model = logistic.LogisticClassifier(input_dim=20, hidden_dim=16, reg=0.08) # lr=0.8
solver = solver.Solver(model, data_input,
                update_rule='sgd',
                optim_config={
                  'learning_rate': 0.8,
                },
                lr_decay=0.98,
                num_epochs=800, batch_size=40,
                print_every=2000)
solver.train()
print(solver.check_accuracy(x_test, y_test, num_samples=None, batch_size=40))
#print(solver.best_params)

