import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
	scores = X[i].dot(W)
	correct_class_score = scores[y[i]]
	diff_count = 0 	# count number of j's where j != y_j
	for j in xrange(num_classes):
		if j == y[i]:
			# for index, value in enumerate(scores): 
			# 	marg = value - correct_class_score + 1
			# 	if (index != y[i]) & (marg > 0):
			# 		dW[i,j] += -marg
			# dW[i,j] *= X[i,j]
			continue
		margin = scores[j] - correct_class_score + 1
		if margin > 0:
			diff_count += 1
			loss += margin
			dW[:,j] += X[i]	# gradient update for j != y_j
	# gradient update for j == y_j
	dW[:, y[i]] += -diff_count * X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  dW += reg * W 	# gradient for regularization term

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

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

  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  # see https://github.com/bruceoutdoors/CS231n/blob/master/assignment1/cs231n/classifiers/linear_svm.py

  # create y mask to put all y_j values to zero
  # y = y.reshape(y.shape[0], 1) 
  mask = np.ones((y.shape[0], W.shape[1]))
  c = 0
  for item in y:
	mask[item,c] = 0
	c+=1

  matmul = X.dot(W)
  data_loss = 1/X.shape[0] * np.sum((matmul - matmul[y])* mask + 1)  #delta = 1

  regu_loss = 0.5 * reg * np.sum(W * W)

  loss = data_loss + regu_loss

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  dW = 0
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
