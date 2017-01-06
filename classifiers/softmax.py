import numpy as np
from random import shuffle

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
  num_samples = X.shape[0]
  num_dim = W.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  # forward pass
  # f = np.zeros((num_samples, num_classes))	#init activation function

  for i in xrange(num_samples):	
	f = X[i].dot(W)
	# numerical stability trick: see http://cs231n.github.io/linear-classify/#softmax
	f -= np.max(f)
	fyi_exp = np.exp(f[y[i]])
	f_exp = np.exp(f)
	breuk =  fyi_exp / np.sum(f_exp)
	loss += -np.log(breuk)

	# gradient calc
	for j in xrange(num_classes):
		# a = exp(fyi), b= sum(exp(fj))
		da = fyi_exp * X[i]
		# db = num_classes * np.reshape(X[i],(num_dim, 1)).dot(np.reshape(f_exp, (1, num_classes)))
		db = X[i] * np.sum(f_exp)
		# print da.shape, db.shape
		dW[:,j] += (breuk * db - da)* 1/fyi_exp

  loss /= num_samples	#mean across all samples
  loss += 0.5 * reg * np.sum(W * W)	#add regularization term

  dW /= num_samples		#mean across all samples
  dW += reg * W 		#add regularization term

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

