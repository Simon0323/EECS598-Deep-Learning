"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""
import numpy as np

def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.
    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.
    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)
    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """

    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    next_h = np.tanh(prev_h.dot(Wh)+x.dot(Wx)+b.reshape(1,-1))
    cache = (x, Wx, Wh, next_h, prev_h)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.
    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass
    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """

    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    x, Wx, Wh, h, pre_h = cache
    dtemp = (1-h**2)*dnext_h  # N*H
    dprev_h = dtemp.dot(Wh.T)   # (N*H) (H*H) = N*H
    dWh = pre_h.T.dot(dtemp)
    dWx = x.T.dot(dtemp)
    dx = dtemp.dot(Wx.T)
    db = np.sum(dtemp, axis = 0)
    db = db.reshape(-1)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.
    Inputs:
    - x: Input data for the entire timeseries, of shape (T, N, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)
    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (T, N, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    T, N, D = x.shape
    H = h0.shape[1]
    h = np.zeros((T,N,H))
    h[0,:,:], cache_temp = rnn_step_forward(x[0,:,:], h0, Wx, Wh, b)
    cachelist_temp = []
    cachelist_temp.append(cache_temp)
    for i in range(1,T):
        h[i,:,:], cache_temp = rnn_step_forward(x[i,:,:], h[i-1,:,:], Wx, Wh, b)
        cachelist_temp.append(cache_temp)
    cache = cachelist_temp
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.
    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (T, N, H)
    Returns a tuple of:
    - dx: Gradient of inputs, of shape (T, N, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    T, N, H = dh.shape
    dx_temp, dh_next, dWx, dWh, db = rnn_step_backward(dh[T-1], cache[T-1])
    D = dx_temp.shape[1]
    dx = np.zeros((T,N,D))
    dx[T-1,:,:] = dx_temp
    for i in reversed(range(T-1)):
        dx[i,:,:], dh_next, dWx_temp, dWh_temp, db_temp = rnn_step_backward(dh[i]+dh_next, cache[i])
        dWx += dWx_temp
        dWh += dWh_temp 
        db += db_temp
    dh0 = dh_next
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.
    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.
    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)
    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    cache = None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    # For Wx of shape D x 4H, you may assume they are the sequence of parameters#
    # for forget gate, input gate, concurrent input, output gate. Wh and b also #
    # follow the same order.                                                    #
    #############################################################################
    N = 1
    if (len(x.shape) == 2): N, D = x.shape
    H = prev_h.shape[1]
    f_t = sigmoid(x.dot(Wx[:,:H])+prev_h.dot(Wh[:,:H])+b[:H].reshape(1,-1))  # N, H
    i_t = sigmoid(x.dot(Wx[:,H:2*H])+prev_h.dot(Wh[:,H:2*H])+b[H:2*H].reshape(1,-1))  # N H
    C_hat = np.tanh(x.dot(Wx[:,2*H:3*H])+prev_h.dot(Wh[:,2*H:3*H])+b[2*H:3*H].reshape(1,-1))  # N H
    next_c = f_t*prev_c + i_t*C_hat  # N H 
    o_t = sigmoid(x.dot(Wx[:,3*H:])+prev_h.dot(Wh[:,3*H:])+b[3*H:].reshape(1,-1))  # N H 
    next_h = o_t * np.tanh(next_c)
    cache = (prev_c, prev_h, Wx, Wh, x, C_hat, f_t, i_t, next_c, o_t)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.
    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh, dc, dWx, dWh, db, dprev_h, dprev_c = None, None, None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    prev_c, prev_h, Wx, Wh, x, c_hat, f_t, i_t, c_t, o_t = cache
    dh = dnext_h
    dc = dnext_c + dnext_h*o_t*(1-np.tanh(c_t)**2)
    dprev_c = f_t*dc
    dtemp1 = (1-f_t)*f_t*prev_c*dc # N H
    dtemp2 = (1-i_t)*i_t*dc*c_hat # N H
    dtemp3 = (1-c_hat**2)*dc*i_t # N H
    dtemp4 = dh*np.tanh(c_t)*(1-o_t)*o_t  # N H
    dtemp = np.hstack((dtemp1, dtemp2, dtemp3, dtemp4))   # N 4H
    dWx = x.T.dot(dtemp)
    dWh = prev_h.T.dot(dtemp)
    dprev_h = dtemp.dot(Wh.T)
    dx = dtemp.dot(Wx.T)
    db = np.sum(dtemp, axis = 0)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.
    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.
    Inputs:
    - x: Input data of shape (T, N, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)
    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (T, N, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    T, N, D = x.shape
    H = h0.shape[1]
    h = np.zeros((T,N,H))
    h[0,:,:], next_c, cache_temp = lstm_step_forward(x[0,:,:], h0, np.zeros_like(h0), Wx, Wh, b)
    cachelist_temp = []
    cachelist_temp.append(cache_temp)
    for i in range(1,T):
        h[i,:,:], next_c, cache_temp = lstm_step_forward(x[i,:,:], h[i-1,:,:], next_c, Wx, Wh, b)
        cachelist_temp.append(cache_temp)
    cache = cachelist_temp
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]
    Inputs:
    - dh: Upstream gradients of hidden states, of shape (T, N, H)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient of input data of shape (T, N, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    T, N, H = dh.shape
    dx_temp, dh_next, dc_next, dWx, dWh, db = lstm_step_backward(dh[T-1], np.zeros_like(dh[0]), cache[T-1])
    D = dx_temp.shape[1]
    dx = np.zeros((T,N,D))
    dx[T-1,:,:] = dx_temp
    for i in reversed(range(T-1)):
        dx[i,:,:], dh_next, dc_next, dWx_temp, dWh_temp, db_temp = lstm_step_backward(dh[i]+dh_next, dc_next, cache[i])
        dWx += dWx_temp
        dWh += dWh_temp 
        db += db_temp
    dh0 = dh_next

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.
    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x must be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.
    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    out = W[x, :]
    cache = x, W
    return out, cache

def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.
    HINT: Look up the function np.add.at
    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass
    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    x, W = cache
    dW = np.zeros_like(W)
    np.add.at(dW, x, dout)
    return dW



def temporal_fc_forward(x, w, b):
    """
    Forward pass for a temporal fully-connected layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.
    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)
    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N = 1
    if (len(x.shape)==3): N, T, D = x.shape
    else: T, D = x.shape
    M = w.shape[1]
    out = x.reshape(N*T, D).dot(w)+b.reshape(1,-1)
    if (N!=1): out = out.reshape(N, T, M)
    else: out = out.reshape(T, M)
    cache = (x, w)
    return out, cache
    
    
def temporal_fc_backward(dout, cache):
    """
    Backward pass for temporal fully-connected layer.
    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass
    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w = cache
    N = 1
    if (len(dout.shape)==3): N, T, M = dout.shape
    else: T, M = dout.shape
    
    D = x.shape[2]
    if (N!=1): dx = dout.reshape(N*T, M).dot(w.T).reshape(N,T,D)
    else:  dx = dout.reshape(N*T, M).dot(w.T).reshape(T,D)
    dw = x.reshape(N*T, D).T.dot(dout.reshape(N*T, M))
    db = np.sum(dout, axis = (0,1))
    return (dx,dw,db)



def temporal_softmax_loss(x, y, mask):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.
    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.
    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.
    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """
    # V is the probability for every vocabulary 
    N, T, V = x.shape
    x_exp = np.exp(x.reshape(N*T, V))
    x_exp_sum = np.sum(x_exp, axis = 1)
    
    index1 = np.arange(N*T)
    loss_temp = x_exp[index1, y.reshape(-1)[:]]/ x_exp_sum
    
    loss_final = -np.log(loss_temp.reshape(N,T))*mask
    loss = np.sum(loss_final)/N
    
    #the shape is (N*T V)
    dx = x_exp/x_exp_sum.reshape(-1,1)
    dx[index1, y.reshape(-1)[:]] -= 1  
    dx *= mask.reshape(N*T)[:, None]
    dx = dx.reshape(N,T,V)/N
    return loss, dx
