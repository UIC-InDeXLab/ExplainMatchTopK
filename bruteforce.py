import math
import heapq
import itertools
import topk
import utility

def ComputeShapleyInTopK(vectors, evaluationFunction, k, j, d, unWrapFunction, algorithm='GeneralPurpose'):
    scores = [0 for x in range(d)]
    previousSeen = {}
    dFactorial = math.factorial(d)
    if algorithm == 'Threshold':
        attributeLists = topk.preProcess(vectors, evaluationFunction)
    for permutation in itertools.permutations(range(d)):
        currHash = 0
        for position in range(d):
            prevHash = currHash
            currHash = currHash | (1 << permutation[position])
            if algorithm == 'Threshold':
                if prevHash not in previousSeen:
                    prevTopK = topk.computeTopKThreshold(vectors, attributeLists, evaluationFunction, permutation[:position], k)
                    prevScore = 0 if position == 0 else (1 if j in prevTopK else 0)
                    previousSeen[prevHash] = prevScore
                else:
                    prevScore = previousSeen[prevHash]
                if currHash not in previousSeen:
                    currTopK = topk.computeTopKThreshold(vectors, attributeLists, evaluationFunction, permutation[:position+1], k)
                    currScore = 1 if j in currTopK else 0
                    previousSeen[currHash] = currScore
                else:
                    currScore = previousSeen[currHash]
            else:
                if prevHash not in previousSeen:
                    prevTuples = topk.generateTuples(vectors, evaluationFunction, permutation, position, unWrapFunction)
                    prevScore = 0 if position == 0 else (1 if topk.computeInTopK(prevTuples, k, j) else 0)
                    previousSeen[prevHash] = prevScore
                else:
                    prevScore = previousSeen[prevHash]
                if currHash not in previousSeen:
                    currTuples = topk.generateTuples(vectors, evaluationFunction, permutation, position+1, unWrapFunction)
                    currScore = 1 if topk.computeInTopK(currTuples, k, j) else 0
                    previousSeen[currHash] = currScore
                else:
                    currScore = previousSeen[currHash]     
            scores[permutation[position]] = scores[permutation[position]] + (currScore - prevScore)/dFactorial
    return scores
                    
            
def ComputeShapleyNotInTopK(vectors, evaluationFunction, k, j, d, unWrapFunction, algorithm='GeneralPurpose'):
    scores = [0 for x in range(d)]
    previousSeen = {}
    dFactorial = math.factorial(d)
    if algorithm == 'Threshold':
        attributeLists = topk.preProcess(vectors, evaluationFunction)
    for permutation in itertools.permutations(range(d)):
        currHash = 0
        for position in range(d):
            prevHash = currHash
            currHash = currHash | (1 << permutation[position])
            if algorithm == 'Threshold':
                if prevHash not in previousSeen:
                    prevTopK = topk.computeTopKThreshold(vectors, attributeLists, evaluationFunction, permutation[:position], k)
                    prevScore = 0 if position == 0 else (1 if not j in prevTopK else 0)
                    previousSeen[prevHash] = prevScore
                else:
                    prevScore = previousSeen[prevHash]
                if currHash not in previousSeen:
                    currTopK = topk.computeTopKThreshold(vectors, attributeLists, evaluationFunction, permutation[:position+1], k)
                    currScore = 1 if not j in currTopK else 0
                    previousSeen[currHash] = currScore
                else:
                    currScore = previousSeen[currHash]
            else:
                if prevHash not in previousSeen:
                    prevTuples = topk.generateTuples(vectors, evaluationFunction, permutation, position, unWrapFunction)
                    prevScore = 0 if position == 0 else (1 if not topk.computeInTopK(prevTuples, k, j) else 0)
                    previousSeen[prevHash] = prevScore
                else:
                    prevScore = previousSeen[prevHash]
                if currHash not in previousSeen:
                    currTuples = topk.generateTuples(vectors, evaluationFunction, permutation, position+1, unWrapFunction)
                    currScore = 1 if not topk.computeInTopK(currTuples, k, j) else 0
                    previousSeen[currHash] = currScore
                else:
                    currScore = previousSeen[currHash]
            scores[permutation[position]] = scores[permutation[position]] + (currScore - prevScore)/dFactorial
    return scores        
                
   
def ComputeShapleyTopKLookLikeThis(vectors, evaluationFunction, k, d, unWrapFunction, algorithm='GeneralPurpose'):
    scores = [0 for x in range(d)]
    dFactorial = math.factorial(d)
    previousSeen = {}
    if algorithm == 'Threshold':
        attributeLists = topk.preProcess(vectors, evaluationFunction)
    initialTuples = topk.generateTuples(vectors, evaluationFunction, [x for x in range(len(vectors[0]))], len(vectors[0]), unWrapFunction)
    initialTopK = topk.computeTopK(initialTuples, k)
    setInitialTopK = set(initialTopK)
    for permutation in itertools.permutations(range(d)):
        currHash = 0
        for position in range(d):
            prevHash = currHash
            currHash = currHash | (1 << permutation[position])
            if algorithm == 'Threshold':
                topKPrev = topk.computeTopKThreshold(vectors, attributeLists, evaluationFunction, permutation[:position], k)
                topKCurr = topk.computeTopKThreshold(vectors, attributeLists, evaluationFunction, permutation[:position+1], k)
                setPrevTopK = set(topKPrev)
                setCurrTopK = set(topKCurr)
                #buggy lines, check for when sets are both empty (divide by zero)
                IoUPrev = 0 if position == 0 else len(setInitialTopK.intersection(setPrevTopK))/len(setInitialTopK.union(setPrevTopK))
                IoUCurr = len(setInitialTopK.intersection(setCurrTopK))/len(setInitialTopK.union(setCurrTopK))
            else:
                if prevHash not in previousSeen:
                    prevTuples = topk.generateTuples(vectors, evaluationFunction, permutation, position, unWrapFunction)
                    topKPrev = topk.computeTopK(prevTuples, k)
                    setPrevTopK = set(topKPrev)
                    IoUPrev = 0 if position == 0 else 1 if (len(setInitialTopK) == 0 and len(setPrevTopK) == 0) else len(setInitialTopK.intersection(setPrevTopK))/len(setInitialTopK.union(setPrevTopK))
                    previousSeen[prevHash] = IoUPrev
                else:
                    IoUPrev = previousSeen[prevHash]
                if currHash not in previousSeen:
                    currTuples = topk.generateTuples(vectors, evaluationFunction, permutation, position+1, unWrapFunction)
                    topKCurr = topk.computeTopK(currTuples, k)
                    setCurrTopK = set(topKCurr)
                    IoUCurr = 1 if (len(setInitialTopK) == 0 and len(setCurrTopK) == 0) else len(setInitialTopK.intersection(setCurrTopK))/len(setInitialTopK.union(setCurrTopK))
                    previousSeen[currHash] = IoUCurr
                else:
                    IoUCurr = previousSeen[currHash]
            scores[permutation[position]] = scores[permutation[position]] + (IoUCurr - IoUPrev)/dFactorial 
    return scores

def ComputeWhyInTheseTopKs(vectors, evaluationFunctions, k, j, d, unWrapFunction, algorithm='GeneralPurpose'):
    scores = [0 for x in range(d)]
    dFactorial = math.factorial(d)
    if algorithm == 'Threshold':
        attributeLists = topk.preProcess(vectors)
    initialTopKs = set()
    previousSeen = {}
    for evaluationFunction in range(d):
        initialTuples = topk.generateTuples(vectors, evaluationFunctions[evaluationFunction], [x for x in range(len(vectors[0]))], len(vectors[0]), unWrapFunction)
        if topk.computeInTopK(initialTuples, k, j):
            initialTopKs.add(evaluationFunction)
    for permutation in itertools.permutations(range(d)):
        currHash = 0
        for position in range(d):
            prevHash = currHash
            currHash = currHash | (1 << permutation[position])
            prevTopKs = set()
            currTopKs = set()
            if prevHash not in previousSeen:
                for evaluationFunction in range(len(evaluationFunctions)):
                    if algorithm == 'Threshold':
                        if j in topk.computeTopKThreshold(vectors, attributeLists, evaluationFunctions[evaluationFunction], permutation[:position], k):
                            prevTopKs.add(evaluationFunction)
                    else:
                        prevTuples = topk.generateTuples(vectors, evaluationFunctions[evaluationFunction], permutation, position, unWrapFunction)
                        if topk.computeInTopK(prevTuples, k, j):
                            prevTopKs.add(evaluationFunction)
                IoUPrev = 0 if position == 0 else 1 if (len(prevTopKs) == 0 and len(initialTopKs) == 0) else len(initialTopKs.intersection(prevTopKs))/len(initialTopKs.union(prevTopKs))
                previousSeen[prevHash] = IoUPrev
            else:
                IoUPrev = previousSeen[prevHash]
            if currHash not in previousSeen:
                for evaluationFunction in range(len(evaluationFunctions)):
                    if algorithm == 'Threshold':
                        if j in topk.computeTopKThreshold(vectors, attributeLists, evaluationFunctions[evaluationFunction], permutation[:position+1], k):
                            currTopKs.add(evaluationFunction)
                    else:
                        currTuples = topk.generateTuples(vectors, evaluationFunctions[evaluationFunction], permutation, position+1, unWrapFunction)
                        if topk.computeInTopK(currTuples, k, j):
                            currTopKs.add(evaluationFunction)
                IoUCurr = 1 if (len(initialTopKs) == 0 and len(currTopKs) == 0) else len(initialTopKs.intersection(currTopKs))/len(initialTopKs.union(currTopKs))
                previousSeen[currHash] = IoUCurr
            else:
                IoUCurr = previousSeen[currHash]
            scores[permutation[position]] = scores[permutation[position]] + (IoUCurr - IoUPrev)/dFactorial
    return scores   
    
vectors = [[5,3,1],[2,4,4],[3,1,2],[4,1,3],[1,2,5]]
weights = [80,90,4]
weights2 = [90,10,5]
weights3 = [50,60,10]
print(ComputeShapleyInTopK(vectors,lambda e:(sum([(weights[x]*e[x]) if e[x] is not None else 0 for x in range(len(weights))])), 2, 0, 3, None))
print(ComputeShapleyNotInTopK(vectors,lambda e:(sum([(weights[x]*e[x]) if e[x] is not None else 0 for x in range(len(weights))])), 2, 2, 3, None))
print(ComputeShapleyTopKLookLikeThis(vectors,lambda e:(sum([(weights[x]*e[x]) if e[x] is not None else 0 for x in range(len(weights))])), 2, 3, None))
print(ComputeWhyInTheseTopKs(vectors[:3],[lambda e:(sum([(weights[x]*e[x]) if e[x] is not None else 0 for x in range(len(weights))])), lambda e:(sum([(weights2[x]*e[x]) if e[x] is not None else 0 for x in range(len(weights))])), lambda e:(sum([(weights3[x]*e[x]) if e[x] is not None else 0 for x in range(len(weights))]))], 2, 0, 3, None))
#[0.5, 0.5, 0.0]
#[0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
#[0.1111111111111111, 0.7777777777777779, 0.1111111111111111]
#[0.5, 0.5, 0.0]
