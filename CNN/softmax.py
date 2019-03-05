import numpy as np
from layers import *


class SoftmaxClassifier(object):
    """
    A fully-connected neural network with
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecture should be fc - softmax if no hidden layer.
    The architecture should be fc - relu - fc - softmax if one hidden layer

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=28*28, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with fc weights                                  #
        # and biases using the keys 'W' and 'b'                                    #
        ############################################################################
        if hidden_dim!=None:
            w1 = np.random.randn(input_dim, hidden_dim)*weight_scale
            b1 = np.zeros(hidden_dim)
            w2 = np.random.randn(hidden_dim, num_classes)*weight_scale
            b2 = np.zeros(num_classes)
            self.params={'W1':w1, 'b1':b1, 'W2':w2, 'b2':b2}
        else:    
            #without hidden layer 
            w1 = np.random.randn(input_dim, num_classes)*weight_scale
            b2 = np.zeros(num_classes)
            self.params={'W1':w1, 'b1':b1}
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the one-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
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
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the one-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        reg = self.reg
        N = np.shape(X)[0] 
        if len(param) == 4:
            #compute the loss
            loss, dy = softmax_loss(scores, y)
            loss += reg * (np.sum(w1*w1)+np.sum(w2*w2))/N 
            # compute the gradient for the second layer 
            dx2, dw2, db2 = fc_backward(dy, cachefc2)  # 
            dw2 += reg * w2 / N
            # compute the gradient for the first layer 
            dy1 = relu_backward(dx2, cacheReLU)
            dx1, dw1, db1 = fc_backward(dy1, cachefc1)
            dw1 += reg * w1 / N
            # return the result 
            grads = {'W1':dw1, 'b1':db1, 'W2':dw2, 'b2':db2}
        elif len(param) == 2:
            #without hidden layer 
            loss, dy = softmax_loss(scores, y)
            loss += reg * np.matmul(w1.T,w1)/N 
            dx, dw, db_temp = fc_backward(dy, cachefc)
            dw += reg * w1 / N
            db = db_temp[0,0]
            grads = {'W1':dw, 'b1':db}
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
