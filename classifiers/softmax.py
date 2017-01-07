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
	f = X[i].dot(W)		# 1xC matrix
	# numerical stability trick: see http://cs231n.github.io/linear-classify/#softmax
	f -= np.max(f)
	fyi_exp = np.exp(f[y[i]])
	f_exp = np.exp(f)
	p = lambda j: np.exp(f[j])/np.sum(np.exp(f))
	breuk =  fyi_exp / np.sum(f_exp)
	breuk2 = p(y[i])
	if breuk != breuk2:
		print 'p not correct'
	loss += -np.log(breuk)

	# gradient calc
	for j in xrange(num_classes):
		# dW derived from Li = -fyi + log (sum(efj))
		dW[:,j] += X[i] * (p(j) - (j == y[i]))

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
  num_samples = X.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # loss L = 1/N * Li
  # Li = (-fyi + log (sum(efj)))
  f = X.dot(W)			# NxC matrix
  # normalization trick for numerical stability
  f -= np.max(f)
  loss = np.sum( -f[np.arange(num_samples), y] + np.log( np.sum(np.exp(f),axis=1) ), axis=0 )

  loss /= num_samples	#mean across all samples
  loss += 0.5 * reg * np.sum(W * W)	#add regularization term

  # print np.exp(f).shape, np.sum(np.exp(f), axis=1).shape
  p = np.exp(f)/np.sum(np.exp(f), axis=1).reshape(num_samples,1)		# NxC matrix
  # print p.shape
  dW = X.T.dot(p)
  # for j==y[i]: the gradient must be corrected with -X[i]
  # matrix multiplication trick to select and sum the right sample values for each class label C
  A = np.zeros_like(p.T)	# CxN matrix 
  # print A.shape
  for j in xrange(num_classes):
  	A[j,y==j] = 1
  Xmask = A.dot(X)  	# CxD matrix summing D for all samples for each label C
  # print Xmask.T.shape
  dW += -Xmask.T 	

  # alternative found online (https://github.com/MyHumbleSelf/cs231n/blob/master/assignment1/cs231n/classifiers/softmax.py) that is slightly (0.005s) faster
  # this one uses the indicator function (different dimensions for inputs so code slightly adapted here!!!)
  # Gradient: dw_j = 1/num_train * \sum_i[x_i * (p(y_i = j)-Ind{y_i = j} )]
  # p = np.exp(f)/np.sum(np.exp(f), axis=1).reshape(num_samples,1)
  # ind = np.zeros(p.T.shape)
  # # print y.shape, ind.shape
  # ind[y, range(num_samples)] = 1
  # dW = np.dot(X.T, (p-ind.T))
  # dW /= num_train  

  dW /= num_samples		#mean across all samples
  dW += reg * W 		#add regularization term  

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

