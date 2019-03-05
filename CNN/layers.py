from builtins import range
import numpy as np
from scipy import signal
import skimage.measure


def fc_forward(x, w, b):
    """
    Computes the forward pass for a fully-connected layer.

    The input x has shape (N, Din, Din2) and contains a minibatch of N
    examples, where each example x[i] has shape (Din,Din2).

    Inputs:
    - x: A numpy array containing input data, of shape (N, Din, Din2)
    - w: A numpy array of weights, of shape (Din*Din2, Dout)
    - b: A numpy array of biases, of shape (Dout,)

    Returns a tuple of:
    - out: output, of shape (N, Dout)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the forward pass. Store the result in out.              #
    ###########################################################################
    N = np.shape(x)[0]
    out = np.matmul(x.reshape(N,-1),w)
    out += b.reshape(1,-1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def fc_backward(dout, cache):
    """
    Computes the backward pass for a fully_connected layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, Dout)
    - cache: Tuple of:
      - x: Input data, of shape (N, Din, Din2)
      - w: Weights, of shape (Din*Din2, Dout)
      - b: Biases, of shape (Dout,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, Din, Din2)
    - dw: Gradient with respect to w, of shape (Din*Din2, Dout)
    - db: Gradient with respect to b, of shape (Dout,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    N = np.shape(x)[0]
    x_shape = np.shape(x)
    dout = dout.reshape(N,-1)
    dx = np.matmul(dout, w.T).reshape(x_shape)
    dw = np.matmul(x.reshape(N,-1).T, dout)
    db = np.matmul(dout.T,np.ones((N,1))).reshape(-1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.copy(x)
    out[x<0] = 0
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = np.copy(dout)
    dx[x<0] = 0
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        sample_mean = np.mean(x, axis = 0)
        sample_var = np.var(x, axis = 0)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        std = np.sqrt(sample_var+eps)
        xhat = (x - sample_mean)/std
        out = xhat*gamma.T + beta.T
        bn_param['running_mean'] = running_mean
        bn_param['running_var'] = running_var
        cache = (x, xhat, gamma, std, sample_mean, bn_param)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        out = ((x - running_mean)/running_var)*gamma.T/beta.T
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    x, xhat, gamma, std, sample_mean, _ = cache
    N,D = np.shape(dout)
    dxhat = dout*gamma.T
    dsigma = np.sum((x - sample_mean)*dxhat, axis = 0)*(-1/2)*std**(-3)
    dmu = np.sum(dxhat,axis = 0)*(-1/std)
    
    dx = dxhat/std + (2 * dsigma * (x-sample_mean) + dmu)/N 
    dgamma = np.sum(dout*xhat,axis = 0)
    dbeta = np.sum(dout, axis = 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Implement the vanilla version of dropout.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        out = np.random.rand(* np.shape(x))
        mask = out<p
        out = mask * x
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x*p
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = mask*dout
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward(x, w):
    """
    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = H - HH + 1
      W' = W - WW + 1
    - cache: (x, w)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    N, C, H, W = np.shape(x)
    w_shape = np.shape(w)
    F, HH, WW = w_shape[0],  w_shape[2], w_shape[3]
    H_prime, W_prime = H - HH + 1, W - WW + 1
    out = np.zeros((N, F, H_prime, W_prime))
    w_present = np.flip(w, axis = 1)
    for i in range(N):
        for f in range(F):
            out[i,f,:,:] = signal.convolve(x[i,:,:,:], w_present[f,:,:,:], mode = 'valid')
    """
    w_present = np.copy(w)
    w_present = np.flip(np.flip(w_present, axis = 2), axis = 3)  
    out_test = np.zeros((N, F, H_prime, W_prime))  
    for i in range(N):
        for j in range(H_prime):
            for k in range(W_prime):
                x_start, x_end = j, j + HH
                y_start, y_end = k, k + WW
                conv_window = x[i, :, x_start:x_end, y_start:y_end];
                temp_result = conv_window * w_present;
                out_test[i,:,j,k] = np.sum(temp_result, axis=(1,2,3));
    """
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w)
    return out, cache


def conv_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives.(N, F, H', W')
    - cache: A tuple of (x, w) as in conv_forward
    - w: Filter weights of shape (F, C, HH, WW)

    Returns a tuple of:
    - dx: Gradient with respect to x (N, C, H, W)
    - dw: Gradient with respect to w (F, C, HH, WW)
    """
    dx, dw = None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w = cache
    x_shape = np.shape(x)
    w_shape = np.shape(w)
    H,W = x_shape[2],x_shape[3]
    H_w, W_w = w_shape[2],w_shape[3]
    F, C, HH, WW =  w_shape[0], w_shape[1], w_shape[2], w_shape[3]
    dout_shape = np.shape(dout)
    H_prime, W_prime = dout_shape[2], dout_shape[3]
    dx = np.zeros(x_shape)
    dw = np.zeros(w_shape)
    dx_temp = np.zeros(x_shape)
    dw_temp = np.zeros(w_shape)
    dx_temp2 = np.zeros((H,W))
    dw_temp2 = np.zeros((H_w,W_w))
    N = x_shape[0]
    w_present = np.copy(w)
    w_present = np.flip(np.flip(w_present, axis = 2), axis = 3)    
    for i in range(N):
        for j in range(H_prime):
            for k in range(W_prime):
                x_start, x_end = j, j + HH
                y_start, y_end = k, k + WW
                for f in range(F):
                    dx[i, :, x_start:x_end, y_start:y_end] += w_present[f] * dout[i,f,j,k]
                    dw[f, :, :, :] += x[i, :, x_start:x_end, y_start:y_end] * dout[i,f,j,k]
    dw = np.flip(np.flip(dw, axis = 2), axis = 3)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw


def max_pool_forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    N, C, H, W = np.shape(x)
    H_prime = int(1 + (H - pool_height) / stride)
    W_prime = int(1 + (W - pool_width) / stride)
    out = np.zeros((N, C, H_prime, W_prime))
#    out[:,:,:,:] = skimage.measure.block_reduce(x[:,:,:,:],(1, 1, stride, stride),func = np.max)
    for i in range(N):
        for j in range(H_prime):
            for k in range(W_prime):
                h_start, h_end, w_start, w_end = j*stride, j*stride+pool_height, k*stride, k*stride+pool_width
                out[i, :, j, k] = np.max(x[i, :, h_start:h_end, w_start:w_end], axis =(1,2))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    N, C, H_prime, W_prime = np.shape(dout)
    x, pool_param = cache
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    H = int((H_prime - 1)*stride + pool_height)
    W = int((W_prime - 1)*stride + pool_width)
    dx = np.zeros((N, C, H, W))
    for i in range(N):
        for j in range(H_prime):
            for k in range(W_prime):
                h_start, h_end = stride * j, stride * j + pool_height
                w_start, w_end = stride * k, stride * k + pool_width
                pooling_window = x[i, :, h_start:h_end, w_start:w_end]
                for c in range(C):
                    """
                    index = np.unravel_index(pooling_window[c].argmax(), pooling_window.shape)
                    index_h = index[0] + h_start
                    index_w = index[1] + w_start
                    dx[i, c, index_h, index_w] = dout[i, c, j, k]
                    """
                    index = np.unravel_index(np.argmax(pooling_window[c]), pooling_window[c].shape)
                    index_h = index[0]
                    index_w = index[1]
                    dx[i, c, h_start+index_h, w_start+index_w] += dout[i, c, j, k]
    
#                index = pooling_window.argmax(axis = 1,2)
#                index_h = 
                
                
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient for binary SVM classification.
  Inputs:
  - x: Input data, of shape (N,) where x[i] is the score for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i]
  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = np.shape(x)[0]
#  y = y.reshape(N,-1)
  temp = np.copy(1-y*x)
  index = temp<0 
  temp[index] = 0
  loss = np.sum(temp)/N
  dx = np.copy(-y)/N
  dx[index] = 0
  return loss, dx


def logistic_loss(x, y):
  """
  Computes the loss and gradient for binary classification with logistic 
  regression.
  Inputs:
  - x: Input data, of shape (N,) where x[i] is the logit for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i]
  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  
  N = np.shape(x)[0]
  #x = x.reshape(N,-1)
  #y = np.reshape(y,(N,-1))
  loss = sum(np.log(1+np.exp(-x*y)))/N
  dx = -y/(1+np.exp(x*y))/N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.
  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C
  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  temp = np.exp(x)
  N = np.shape(x)[0]
  index = np.arange(N)
  temp_sum = np.sum(temp, axis = 1)
  temp2 = temp[index, y]
  temp3 = temp2.reshape(N,1)/np.reshape(temp_sum, (N,1))
  temp4= -np.log(temp3)
  loss = np.sum(temp4) / N
  dx = temp / np.reshape(temp_sum, (N,1))
  dx[index, y] -= 1
  dx/=N
  return loss, dx













