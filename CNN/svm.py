import numpy as np

from layers import *

class SVM(object):
  """
  A binary SVM classifier with optional hidden layers.
  
  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=100, hidden_dim=None, weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the model. Weights            #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases (if any) using the keys 'W2' and 'b2'.                        #
    ############################################################################
    if hidden_dim!=None:
        w1 = np.random.randn(input_dim, hidden_dim)*weight_scale
        b1 = np.zeros(hidden_dim)
        w2 = np.random.randn(hidden_dim,1)*weight_scale
        b2 = np.array([0.0])
        self.params={'W1':w1, 'b1':b1, 'W2':w2, 'b2':b2}
    else:    
        #without hidden layer 
        w1 = np.random.randn(input_dim,1)*weight_scale
        b1 = np.array([0.0])
        self.params={'W1':w1, 'b1':b1}
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, D)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N,) where scores[i] represents the classification 
    score for X[i].
    
    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the model, computing the            #
    # scores for X and storing them in the scores variable.                    #
    ############################################################################
    param = self.params
    if len(param) == 4:
        #with hidden layer
        w1,b1,w2,b2 = param['W1'], param['b1'], param['W2'], param['b2']
        y_temp, cachefc1 = fc_forward(X, w1, b1)
        y1, cacheReLU = relu_forward(y_temp)
        scores, cachefc2 = fc_forward(y1, w2, b2)
    elif len(param) == 2:
        #without hidden layer 
        w1, b1 = param['W1'], param['b1']
        scores, cachefc = fc_forward(X, w1, b1)
    scores = np.reshape(2*scores-1, -1)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the model. Store the loss          #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss and make sure that grads[k] holds the gradients for self.params[k]. #
    # Don't forget to add L2 regularization.                                   #
    #                                                                          #
    ############################################################################
    reg = self.reg
    N = np.shape(X)[0] 
    temp_y = np.copy(y)
    y[temp_y<=0] = -1;
    if len(param) == 4:
        #with hidden layer
        loss, dy = svm_loss(scores, y)
        loss += reg * (np.sum(w1*w1)+np.sum(w2*w2))/N 
        dx2, dw2, db2_temp = fc_backward(dy, cachefc2)  # 
        dw2 += reg * w2 / N
        db2 = db2_temp
        dy1 = relu_backward(dx2, cacheReLU)
        dx1, dw1, db1 = fc_backward(dy1, cachefc1)
        dw1 += reg * w1 / N
        grads = {'W1':dw1, 'b1':db1, 'W2':dw2, 'b2':db2}
    elif len(param) == 2:
        #without hidden layer 
        loss, dy = svm_loss(scores, y)
        loss += reg * np.matmul(w1.T,w1)/N 
        dx, dw, db_temp = fc_backward(dy, cachefc)
        dw += reg * w1 / N
        db = db_temp
        grads = {'W1':dw, 'b1':db}
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
