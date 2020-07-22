#!/usr/bin/env python

import numpy as np
import random

from utils.gradcheck import gradcheck_naive, grad_tests_softmax, grad_tests_negsamp
from utils.utils import normalizeRows, softmax


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """

    s = 1 / (1 + np.exp(-x))
    return s


def naiveSoftmaxLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset
):
    """ Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between a center word's 
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models.

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding #
                    in shape (word vector length, )
                    (v_c in the pdf handout)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors is
                    in shape (num words in vocab, word vector length) 
                    for all words in vocab (U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """

    # centerWordVec  1,D
    # gradCenterVec  1,D
    # outsideWordIdx  int
    # outsideVectors V,D
    # gradOutsideVecs  V,D

    # 새로 배운 것들
    # (V, ) = (V x d) (d, ) 
    # (K X 1) * (K X m) = (K X m) 
    # matrix[row].shape is 1D : 하나만 부를 시 무조건 1D

    # V X 1 = (V x d) (d x 1)
    normed = normalizeRows(outsideVectors.dot(centerWordVec.reshape(-1,1)))
    y_hat = softmax(normed)
    loss = -np.log(y_hat[outsideWordIdx]) 
   
    # V X 1 
    # dL/da . 
    # u_w == u_o : (a - 1) 
    # u_w != u_o : a
    y_hat[outsideWordIdx] -= 1
    
    # 1 X d = (V X 1).T (V X d)
    # (yhat - y).T * U
    gradCenterVec = y_hat.T.dot(outsideVectors).flatten()
    # V, d = V, 1 * 1, d
    # (yhat - y) * vc.T
    gradOutsideVecs = y_hat.dot(centerWordVec.reshape(1,-1))


    ### Please use the provided softmax function (imported earlier in this file)
    ### This numerically stable implementation helps you avoid issues pertaining
    ### to integer overflow. 

    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs

# get words that doesn't exist in the window
def getNegativeSamples(outsideWordIdx, dataset, K):
    """ Samples K indexes which are not the outsideWordIdx """

    # None K개를 갖는 list
    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def negSamplingLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset,
    K=10
):
    """ Negative sampling loss function for word2vec models

    Implement the negative sampling loss and gradients for a centerWordVec
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if # 이 부분이 좀 어렵네....
    it was samspled three times, and so forth.

    Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient
    """ 
    gradCenterVec = np.zeros(centerWordVec.shape)
    gradOutsideVecs = np.zeros(outsideVectors.shape)

    negSampleWordIndices  = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices 

    loss = 0.0

    
    scores = outsideVectors.dot(centerWordVec.reshape(-1,1))

    # V X 1 
    z1 = sigmoid(scores)
    z2 = sigmoid(-scores)
    # scalar
    p_outside = z1[outsideWordIdx]
    # (K, d)
    p_negatives = z2[negSampleWordIndices] 

    loss = -np.log(p_outside) - np.sum(np.log(p_negatives))

    # dJ / dv_c
    gradCenterVec += (p_outside -1) * outsideVectors[outsideWordIdx] + np.sum((1 - p_negatives) * outsideVectors[negSampleWordIndices])
    # dJ / du_o 
    gradOutsideVecs[outsideWordIdx] += z1[outsideWordIdx] * centerWordVec
    # dJ / du_k 
    for i, n_idx in enumerate(negSampleWordIndices):
        gradOutsideVecs[n_idx] += (1 - p_negatives[i]) * centerWordVec
        # gradOutsideVecs[n_idx] += np.sum((1 - p_negatives).dot(centerWordVec(1,-1))) 
    return loss, gradCenterVec, gradOutsideVecs


def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,
             centerWordVectors, outsideVectors, dataset,
             word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentCenterWord -- a string of the current center word
    windowSize -- integer, contextcontext window size
    outsideWords -- list of no more than 2*windowSize strings, the outside words
    word2Ind -- a dictionary that maps words to their indices in
              the word vector list
    centerWordVectors -- center word vectors (as rows) is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (V in pdf handout)
    outsideVectors -- outside vectors is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (U in the pdf handout)
    word2vecLossAndGradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """
    _, D = centerWordVectors.shape

    loss = 0.0
    gradCenterVec = np.zeros(D) # 1, d
    gradOutsideVectors = np.zeros(outsideVectors.shape) # V, d

    center_idx = word2Ind[currentCenterWord]
    v_c = centerWordVectors[center_idx]
    outside_indices = [word2Ind[o_word] for o_word in outsideWords]

    # scores = outsideVectors.dot(v_c.T)
    # window = outsideVectors[np.max(center_idx - windowSize,0) : np.min(center_idx + windowSize, V),:]


    for o_idx in outside_indices:
        l, gradC, gradO = word2vecLossAndGradient(v_c, o_idx, outsideVectors, dataset)
        loss += l
        gradCenterVec += gradC
        gradOutsideVectors += gradO
        

    # for i in range(windowSize*2):
    #     if window[i] == currentCenterWord:
    #         continue; 
    #     l, gradC, gradO = word2vecLossAndGradient(currentCenterWord, window[i], outsideVectors, dataset)
    #     loss += l
    #     gradCentervec += gradC
    #     gradOutsideVectors += gradO


 
    return loss, gradCenterVec, gradOutsideVectors


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, word2Ind, wordVectors, dataset, 
                         windowSize,
                         word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[:int(N/2),:]
    outsideVectors = wordVectors[int(N/2):,:]
    for i in range(batchsize):
        windowSize1 = random.randint(1, windowSize)
        centerWord, context = dataset.getRandomContext(windowSize1)

        c, gin, gout = word2vecModel(
            centerWord, windowSize1, context, word2Ind, centerWordVectors,
            outsideVectors, dataset, word2vecLossAndGradient
        )
        loss += c / batchsize
        grad[:int(N/2), :] += gin / batchsize
        grad[int(N/2):, :] += gout / batchsize

    return loss, grad


def test_word2vec():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])

    print("==== Gradient check for skip-gram with naiveSoftmaxLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, naiveSoftmaxLossAndGradient),
        dummy_vectors, "naiveSoftmaxLossAndGradient Gradient")
    grad_tests_softmax(skipgram, dummy_tokens, dummy_vectors, dataset)

    print("==== Gradient check for skip-gram with negSamplingLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingLossAndGradient),
        dummy_vectors, "negSamplingLossAndGradient Gradient")

    grad_tests_negsamp(skipgram, dummy_tokens, dummy_vectors, dataset, negSamplingLossAndGradient)


if __name__ == "__main__":
    test_word2vec()

