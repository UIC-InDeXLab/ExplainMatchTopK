import random
import heapq
import math
import topk


def approximateShapleyInTopK(vectors, evaluationFunction, m, k, j, algorithm='GeneralPurpose'):
    scores = [0 for x in range(len(vectors[0]))]
    attributes = [x for x in range(len(vectors[0]))]
    if algorithm == 'Threshold':
        attributeLists = topk.preProcess(vectors)
    for mi in range(m):
        permutation = attributes.copy()
        random.shuffle(attributes)
        for position in range(len(permutation)):
            if algorithm == 'Threshold':
                prev = topk.computeTopKThreshold(vectors, attributeLists, evaluationFunction, permutation[:position], k)
                curr = topk.computeTopKThreshold(vectors, attributeLists, evaluationFunction, permutation[:position+1], k)
                prevScore = 0 if position == 0 else (1 if j in prev else 0)
                currScore = 1 if j in curr else 0
            else:
                prevTuples = topk.generateTuples(vectors, evaluationFunction, permutation, position)
                currTuples = topk.generateTuples(vectors, evaluationFunction, permutation, position+1)
                prevScore = 0 if position == 0 else (1 if topk.computeInTopK(prevTuples, k, j) else 0)
                currScore = 1 if topk.computeInTopK(currTuples, k, j) else 0
            scores[permutation[position]] = scores[permutation[position]] + (currScore - prevScore)/m
    return scores

def approximateShapleyNotInTopK(vectors, evaluationFunction, m, k, j, algorithm='GeneralPurpose'):
    scores = [0 for x in range(len(vectors[0]))]
    attributes = [x for x in range(len(vectors[0]))]
    if algorithm == 'Threshold':
        attributeLists = topk.preProcess(vectors)
    for mi in range(m):
        permutation = attributes.copy()
        random.shuffle(attributes)
        for position in range(len(permutation)):
            if algorithm == 'Threshold':
                prev = topk.computeTopKThreshold(vectors, attributeLists, evaluationFunction, permutation[:position], k)
                curr = topk.computeTopKThreshold(vectors, attributeLists, evaluationFunction, permutation[:position+1], k)
                prevScore = 0 if position == 0 else (1 if not j in prev else 0)
                currScore = 1 if not j in curr else 0
            else:
                prevTuples = topk.generateTuples(vectors, evaluationFunction, permutation, position)
                currTuples = topk.generateTuples(vectors, evaluationFunction, permutation, position+1)
                prevScore = 0 if position == 0 else (1 if not topk.computeInTopK(prevTuples, k, j) else 0)
                currScore = 1 if not topk.computeInTopK(currTuples, k, j) else 0
            scores[permutation[position]] = scores[permutation[position]] + (currScore - prevScore)/m
    return scores

def approximateShapleyTopKLookLikeThis(vectors, evaluationFunction, m, k, algorithm='GeneralPurpose'):
    scores = [0 for x in range(len(vectors[0]))]
    attributes = [x for x in range(len(vectors[0]))]
    if algorithm == 'Threshold':
        attributeLists = topk.preProcess(vectors)
    initialTuples = topk.generateTuples(vectors, evaluationFunction, [x for x in range(len(vectors[0]))], len(vectors[0]))
    initialTopK = topk.computeTopK(initialTuples, k)
    setInitialTopK = set(initialTopK)
    for mi in range(m):
        permutation = attributes.copy()
        random.shuffle(attributes)
        for position in range(len(permutation)):
            if algorithm == 'Threshold':
                topKPrev = topk.computeTopKThreshold(vectors, attributeLists, evaluationFunction, permutation[:position], k)
                topKCurr = topk.computeTopKThreshold(vectors, attributeLists, evaluationFunction, permutation[:position+1], k)
            else:
                prevTuples = topk.generateTuples(vectors, evaluationFunction, permutation, position)
                currTuples = topk.generateTuples(vectors, evaluationFunction, permutation, position+1)
                topKPrev = topk.computeTopK(prevTuples, k)
                topKCurr = topk.computeTopK(currTuples, k)
            setPrevTopK = set(topKPrev)
            setCurrTopK = set(topKCurr)
            IoUPrev = 0 if position == 0 else len(setInitialTopK.intersection(setPrevTopK))/len(setInitialTopK.union(setPrevTopK))
            IoUCurr = len(setInitialTopK.intersection(setCurrTopK))/len(setInitialTopK.union(setCurrTopK))
            scores[permutation[position]] = scores[permutation[position]] + (IoUCurr - IoUPrev)/m 
    return scores

def approximateWhyInTheseTopKs(vectors, evaluationFunctions, m, k, j, algorithm='GeneralPurpose'):
    scores = [0 for x in range(len(vectors[0]))]
    attributes = [x for x in range(len(vectors[0]))]
    if algorithm == 'Threshold':
        attributeLists = topk.preProcess(vectors)
    initialTopKs = set()
    for evaluationFunction in range(len(evaluationFunctions)):
        initialTuples = topk.generateTuples(vectors, evaluationFunctions[evaluationFunction], [x for x in range(len(vectors[0]))], len(vectors[0]))
        if topk.computeInTopK(initialTuples, k, j):
            initialTopKs.add(evaluationFunction)
    for mi in range(m):
        permutation = attributes.copy()
        random.shuffle(attributes)
        for position in range(len(permutation)):
            prevTopKs = set()
            currTopKs = set()
            for evaluationFunction in range(len(evaluationFunctions)):
                if algorithm == 'Threshold':
                    if j in topk.computeTopKThreshold(vectors, attributeLists, evaluationFunctions[evaluationFunction], permutation[:position], k):
                        prevTopKs.add(evaluationFunction)
                    if j in topk.computeTopKThreshold(vectors, attributeLists, evaluationFunctions[evaluationFunction], permutation[:position+1], k):
                        currTopKs.add(evaluationFunction)
                else:
                    prevTuples = topk.generateTuples(vectors, evaluationFunctions[evaluationFunction], permutation, position)
                    currTuples = topk.generateTuples(vectors, evaluationFunctions[evaluationFunction], permutation, position+1)
                    if topk.computeInTopK(prevTuples, k, j):
                        prevTopKs.add(evaluationFunction)
                    if topk.computeInTopK(currTuples, k, j):
                        currTopKs.add(evaluationFunction)
            IoUPrev = 0 if position == 0 else len(initialTopKs.intersection(prevTopKs))/len(initialTopKs.union(prevTopKs))
            IoUCurr = len(initialTopKs.intersection(currTopKs))/len(initialTopKs.union(currTopKs))
            scores[permutation[position]] = scores[permutation[position]] + (IoUCurr - IoUPrev)/m 
    return scores   


vectors = [[5,3,1],[2,4,4],[3,1,2],[4,1,3],[1,2,5]]
weights = [80,90,4]
weights2 = [90,10,5]
weights3 = [50,60,10]
print(approximateShapleyInTopK(vectors,lambda e:(sum([(weights[x]*e[x]) for x in range(len(weights))])), 5000, 2, 0))
print(approximateShapleyInTopK(vectors,lambda e:(sum([(weights[x]*e[x]) for x in range(len(weights))])), 5000, 2, 0, 'Threshold'))
print(approximateShapleyNotInTopK(vectors,lambda e:(sum([(weights[x]*e[x]) for x in range(len(weights))])), 5000, 2, 2))
print(approximateShapleyNotInTopK(vectors,lambda e:(sum([(weights[x]*e[x]) for x in range(len(weights))])), 5000, 2, 2, 'Threshold'))
print(approximateShapleyTopKLookLikeThis(vectors,lambda e:(sum([(weights[x]*e[x]) for x in range(len(weights))])), 5000, 2))
print(approximateShapleyTopKLookLikeThis(vectors,lambda e:(sum([(weights[x]*e[x]) for x in range(len(weights))])), 5000, 2, 'Threshold'))
print(approximateWhyInTheseTopKs(vectors[:3],[lambda e:(sum([(weights[x]*e[x]) for x in range(len(weights))])), lambda e:(sum([(weights2[x]*e[x]) for x in range(len(weights))])), lambda e:(sum([(weights3[x]*e[x]) for x in range(len(weights))]))], 5000, 2, 0))
print(approximateWhyInTheseTopKs(vectors[:3],[lambda e:(sum([(weights[x]*e[x]) for x in range(len(weights))])), lambda e:(sum([(weights2[x]*e[x]) for x in range(len(weights))])), lambda e:(sum([(weights3[x]*e[x]) for x in range(len(weights))]))], 5000, 2, 0, 'Threshold'))

#print(approximateShapleyNotInTopK(preProcess(vectors, weights), 3, 10000, 2, 0))
