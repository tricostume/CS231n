import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]
  exp_scores = np.zeros((X.shape[0],W.shape[1]))
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # With the MAX we try to improve numerical stability
  for i in range(num_train):
      exp_scores = np.exp(X[i,:].dot(W)-np.max(X[i,:].dot(W)))
      for j in range(num_classes):
          dW[:,j] += (exp_scores[j]/np.sum(exp_scores))*X[i,:]
          if j == y[i]:
              dW[:,j] += -X[i,:]
      loss += -np.log(exp_scores[y[i]]/np.sum(exp_scores))
  
  loss /= num_train
  dW /= num_train
  
  loss += reg*np.sum(W*W)  
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
   # With the AMAX we try to improve numerical stability
  probs = np.exp(X.dot(W)-np.amax(X.dot(W),1)[:,np.newaxis])/np.sum(np.exp(X.dot(W)-np.amax(X.dot(W),1)[:,np.newaxis]),1)[:,np.newaxis]
  loss = np.sum(-np.log(probs[range(num_train),y]))
  
  probs[range(num_train),y] += -1
  dW = np.dot(X.T,probs)
  
  loss /= num_train
  dW /= num_train
  
  loss += reg*np.sum(W*W)  
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

