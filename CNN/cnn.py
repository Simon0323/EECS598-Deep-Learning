import numpy as np

from layers import *


class ConvNet(object):
  """
  A convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - fc - softmax

  You may also consider adding dropout layer or batch normalization layer. 
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(1, 28, 28), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    self.bn_param ={}
    self.flag_bn = 0
    self.flag_drop = 0
    
    w1 = np.random.randn(num_filters, 1, filter_size, filter_size)*weight_scale
    b1 = np.zeros(num_filters)
    len1 = input_dim[1]-(filter_size-1)
    len2 = len1/2
    len2 = len2**2 * num_filters
    
    w2 = np.random.randn(int(len2), hidden_dim)*np.sqrt(2/hidden_dim)
    b2 = np.zeros(hidden_dim)
    
    w3 = np.random.randn(hidden_dim, num_classes)*np.sqrt(2/(num_classes+hidden_dim))
    b3 = np.zeros(num_classes)
    
    D_after_cov = int(len1*len1*num_filters)
    if (self.flag_bn == 1):
        gamma = np.ones(D_after_cov)
        beta = np.zeros(D_after_cov)
        self.bn_param = {'mode':'train', 'eps':1e-5, 'momentum':0.8, 'running_mean':0,
                         'running_var':0}
        self.params={'W1':w1, 'b1':b1, 'W2':w2, 'b2':b2,  'W3':w3, 'b3':b3, 
                     'gamma':gamma, 'beta':beta}
    else:
        self.params={'W1':w1, 'b1':b1, 'W2':w2, 'b2':b2,  'W3':w3, 'b3':b3}
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
        
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    N = X.shape[0]
    y_cov, cachecv = conv_forward(X, W1)
    
    if (self.flag_bn == 1):
        gamma, beta = self.params['gamma'], self.params['beta']
        y_cov_shape = y_cov.shape;
        y_batch_after, cachebn = batchnorm_forward(np.reshape(y_cov, (N,-1)), gamma, beta, self.bn_param)
        self.bn_param = cachebn[5]
        y_cov = y_batch_after.reshape(y_cov_shape)
    
    y1, cacheReLU1 = relu_forward(y_cov)
    y1_pool, cachepool = max_pool_forward(y1, pool_param)
    y1_vector = np.reshape(y1_pool, (N,-1))
    if (self.flag_drop == 1):
        dropout_param = {'p':0.8, 'mode':'train'}
        y1_pool, cache_drop = dropout_forward(y1_vector, dropout_param)
    y2, cachefc2 = fc_forward(y1_pool, W2, b2)
    y2, cacheReLU2 = relu_forward(y2)
    
    scores, cachefc3 = fc_forward(y2, W3, b3)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    reg = self.reg
    N = np.shape(X)[0] 
    input_dim = np.shape(X)[2]
    num_filter, filter_size = np.shape(W1)[0], np.shape(W1)[2]
    len1 = input_dim-(filter_size-1)
    len1 = len1//2
    

    # compute the loss
    loss, dy = softmax_loss(scores, y)
    loss += reg * (np.sum(W1*W1)+np.sum(W2*W2) + np.sum(W3*W3))
    loss /= N
    # copute the gradient for the third layer 
    dx3, dw3, db3 = fc_backward(dy, cachefc3)   
    dw3 += reg * W3 / N
    # copute the gradient for the third layer 
    dy2 = relu_backward(dx3, cacheReLU2)
    dx2, dw2, db2 = fc_backward(dy2, cachefc2)   
    dw2 += reg * W2 / N
    if (self.flag_drop == 1):
        dx2 = dropout_backward(dx2, cache_drop)
    # copute the gradient for the third layer
    dx2_prepool = np.reshape(dx2, (N, num_filter, len1, len1))
    dx2_pool = max_pool_backward(dx2_prepool, cachepool)
    dy1 = relu_backward(dx2_pool, cacheReLU1)
    
    if (self.flag_bn):
        dy1_shape = dy1.shape
        dy1_after_bn, dgamma, dbeta = batchnorm_backward(dy1.reshape(N,-1), cachebn)
        dy1 = dy1_after_bn.reshape(dy1_shape)
        loss+=(np.sum(beta*beta)+np.sum(gamma*gamma))
        dgamma+=reg*gamma/N
        dbeta += reg*beta/N
    
    dx1, dw1 = conv_backward(dy1, cachecv)
    dw1 += reg * W1 / N
    db1 = 0.0
    if (self.flag_bn == 1):
        grads = {'W1':dw1, 'b1':db1, 'W2':dw2, 'b2':db2, 'W3':dw3, 'b3':db3, 
                 'gamma':dgamma, 'beta':dbeta}
    else:
        grads = {'W1':dw1, 'b1':db1, 'W2':dw2, 'b2':db2, 'W3':dw3, 'b3':db3}
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
