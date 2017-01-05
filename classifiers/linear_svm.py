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
		# alternative code 
		# accumulate loss for the i-th example
    	# loss_i += max(0, scores[j] - correct_class_score + delta)
		margin = scores[j] - correct_class_score + 1
		if margin > 0:
			diff_count += 1
			loss += margin
			dW[:,j] += X[i]	# gradient update for j != y_j
	# gradient update for j == y_j
	dW[:,y[i]] += -diff_count * X[i]

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

  num_classes = W.shape[1]
  num_train = X.shape[0]
  delta = 1.0

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  # half-vectorized from http://cs231n.github.io/linear-classify/#svm
  # delta = 1.0
  # scores = W.dot(x)
  # # compute the margins for all classes in one vector operation
  # margins = np.maximum(0, scores - scores[y] + delta)
  # # on y-th position scores[y] - scores[y] canceled and gave delta. We want
  # # to ignore the y-th position and only consider margin on max wrong class
  # margins[y] = 0
  # loss_i = np.sum(margins)
  # return loss_i

  # inspired by tutorial https://github.com/bruceoutdoors/CS231n/blob/master/assignment1/cs231n/classifiers/linear_svm.py

  scores = X.dot(W)
  correct_class_scores = scores[np.arange(num_train),y]
  margins = np.maximum(0, scores - correct_class_scores[:, np.newaxis] + delta)
  margins[np.arange(num_train),y] = 0	# on y-th position: set margin to 0 (sum excludes j == y_j)

  loss = np.sum(margins)
  loss /= num_train	#get mean across samples
  loss += 0.5 * reg * np.sum(W * W)		#regularization

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
  
  # # Semi-vectorized version... It's not any faster than the loop version.
  # num_classes = W.shape[1]
  # incorrect_counts = np.sum(margins > 0, axis=1)
  # for k in xrange(num_classes):
	 #  # use the indices of the margin array as a mask.
	 #  wj = np.sum(X[margins[:, k] > 0], axis=0)
	 #  wy = np.sum(-incorrect_counts[y == k][:, np.newaxis] * X[y == k], axis=0)
	 #  dW[:, k] = wj + wy
  # dW /= num_train # average out weights

  # fully vectorized: rougjly 10x faster
  # rows map to samples, colums map to class
  # a value v in X_mask[i,j] adds row sample i to class j with multiple of v (1 except where j==y_j)
  X_mask = np.zeros(margins.shape)
  X_mask[margins > 0] = 1
  # for each sample, find total number of classes where margin > 0
  incorrectsamples_count = np.sum(X_mask, axis=1)
  # adapt weight where j == y_j
  X_mask[np.arange(num_train), y] -= incorrectsamples_count

  dW = X.T.dot(X_mask) / num_train
  # regularize the weights
  dW += reg*W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
