import math
import heapq
import itertools
import topk

def ComputeShapleyInTopK(vectors, evaluationFunction, k, j, algorithm='GeneralPurpose'):
    scores = [0 for x in range(len(vectors[0]))]
    if algorithm == 'Threshold':
        attributeLists = topk.preProcess(vectors)
    for permutation in itertools.permutations(range(len(vectors[0]))):
        for position in range(len(vectors[0])):
            if algorithm == 'Threshold':
                prevTopK = topk.computeTopKThreshold(vectors, attributeLists, evaluationFunction, permutation[:position], k)
                currTopK = topk.computeTopKThreshold(vectors, attributeLists, evaluationFunction, permutation[:position+1], k)
                prevScore = 0 if position == 0 else (1 if j in prevTopK else 0)
                currScore = 1 if j in currTopK else 0
            else:
                prevTuples = topk.generateTuples(vectors, evaluationFunction, permutation, position)
                currTuples = topk.generateTuples(vectors, evaluationFunction, permutation, position+1)
                prevScore = 0 if position == 0 else (1 if topk.computeInTopK(prevTuples, k, j) else 0)
                currScore = 1 if topk.computeInTopK(currTuples, k, j) else 0
            scores[permutation[position]] = scores[permutation[position]] + (currScore - prevScore)/math.factorial(len(vectors[0]))
    return scores
                    
            
def ComputeShapleyNotInTopK(vectors, evaluationFunction, k, j, algorithm='GeneralPurpose'):
    scores = [0 for x in range(len(vectors[0]))]
    if algorithm == 'Threshold':
        attributeLists = topk.preProcess(vectors)
    for permutation in itertools.permutations(range(len(vectors[0]))):
        for position in range(len(vectors[0])):
            if algorithm == 'Threshold':
                prevTopK = topk.computeTopKThreshold(vectors, attributeLists, evaluationFunction, permutation[:position], k)
                currTopK = topk.computeTopKThreshold(vectors, attributeLists, evaluationFunction, permutation[:position+1], k)
                prevScore = 0 if position == 0 else (1 if not j in prevTopK else 0)
                currScore = 1 if not j in currTopK else 0
            else:
                prevTuples = topk.generateTuples(vectors, evaluationFunction, permutation, position)
                currTuples = topk.generateTuples(vectors, evaluationFunction, permutation, position+1)
                prevScore = 0 if position == 0 else (1 if not topk.computeInTopK(prevTuples, k, j) else 0)
                currScore = 1 if not topk.computeInTopK(currTuples, k, j) else 0
            scores[permutation[position]] = scores[permutation[position]] + (currScore - prevScore)/math.factorial(len(vectors[0]))
    return scores        
                
   
def ComputeShapleyTopKLookLikeThis(vectors, evaluationFunction, k, algorithm='GeneralPurpose'):
    scores = [0 for x in range(len(weights))]
    if algorithm == 'Threshold':
        attributeLists = topk.preProcess(vectors)
    initialTuples = topk.generateTuples(vectors, evaluationFunction, [x for x in range(len(vectors[0]))], len(vectors[0]))
    initialTopK = topk.computeTopK(initialTuples, k)
    setInitialTopK = set(initialTopK)
    for permutation in itertools.permutations(range(len(vectors[0]))):
        for position in range(len(vectors[0])):
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
            scores[permutation[position]] = scores[permutation[position]] + (IoUCurr - IoUPrev)/math.factorial(len(vectors[0])) 
    return scores

def ComputeWhyInTheseTopKs(vectors, evaluationFunctions, k, j, algorithm='GeneralPurpose'):
    scores = [0 for x in range(len(weights))]
    if algorithm == 'Threshold':
        attributeLists = topk.preProcess(vectors)
    initialTopKs = set()
    for evaluationFunction in range(len(evaluationFunctions)):
        initialTuples = topk.generateTuples(vectors, evaluationFunctions[evaluationFunction], [x for x in range(len(vectors[0]))], len(vectors[0]))
        if topk.computeInTopK(initialTuples, k, j):
            initialTopKs.add(evaluationFunction)
    for permutation in itertools.permutations(range(len(vectors[0]))):
        for position in range(len(vectors[0])):
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
            scores[permutation[position]] = scores[permutation[position]] + (IoUCurr - IoUPrev)/math.factorial(len(vectors[0])) 
    return scores   
    
vectors = [[5,3,1],[2,4,4],[3,1,2],[4,1,3],[1,2,5]]
weights = [80,90,4]
weights2 = [90,10,5]
weights3 = [50,60,10]
print(ComputeShapleyInTopK(vectors,lambda e, attributes:(sum([(weights[x]*e[x]) for x in range(len(weights))])), 2, 0))
print(ComputeShapleyInTopK(vectors,lambda e, attributes:(sum([(weights[x]*e[x]) for x in range(len(weights))])), 2, 0, 'Threshold'))
print(ComputeShapleyNotInTopK(vectors,lambda e, attributes:(sum([(weights[x]*e[x]) for x in range(len(weights))])), 2, 2))
print(ComputeShapleyNotInTopK(vectors,lambda e, attributes:(sum([(weights[x]*e[x]) for x in range(len(weights))])), 2, 2, 'Threshold'))
print(ComputeShapleyTopKLookLikeThis(vectors,lambda e, attributes:(sum([(weights[x]*e[x]) for x in range(len(weights))])), 2))
print(ComputeShapleyTopKLookLikeThis(vectors,lambda e, attributes:(sum([(weights[x]*e[x]) for x in range(len(weights))])), 2, 'Threshold'))
print(ComputeWhyInTheseTopKs(vectors[:3],[lambda e, attributes:(sum([(weights[x]*e[x]) for x in range(len(weights))])), lambda e:(sum([(weights2[x]*e[x]) for x in range(len(weights))])), lambda e:(sum([(weights3[x]*e[x]) for x in range(len(weights))]))], 2, 0))
print(ComputeWhyInTheseTopKs(vectors[:3],[lambda e, attributes:(sum([(weights[x]*e[x]) for x in range(len(weights))])), lambda e:(sum([(weights2[x]*e[x]) for x in range(len(weights))])), lambda e:(sum([(weights3[x]*e[x]) for x in range(len(weights))]))], 2, 0, 'Threshold'))
#[0.5, 0.5, 0.0]
#Using Threshold Algorithm...
#[0.5, 0.5, 0.0]
#[0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
#Using Threshold Algorithm...
#[0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
#[0.1111111111111111, 0.7777777777777779, 0.1111111111111111]
#Using Threshold Algorithm...
#[0.1111111111111111, 0.7777777777777779, 0.1111111111111111]
#[0.5, 0.5, 0.0]
#Using Threshold Algorithm...
#[0.5, 0.5, 0.0]
