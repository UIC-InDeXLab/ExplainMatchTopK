import random
import heapq
import math

def preProcess(vectors, weights):
    results = {}
    weightedVectors = []
    for vector in vectors:
        weightedVector = []
        for i in range(len(weights)):
            weightedVector.append(vector[i] * weights[i])
        weightedVectors.append(weightedVector)
    results['Weighted Vectors'] = weightedVectors
    attributeLists = []
    for a in range(len(weights)):
        attributeList = [(weightedVectors[x][a], x) for x in range(len(weightedVectors))]
        attributeList.sort(reverse=True)
        attributeLists.append(attributeList)
    results['Attribute Lists'] = attributeLists
    return results

def getTopK(preprocessed, attributes, k):
    if (len(attributes) == 0):
        return []
    threshold = float('inf')
    seenVectors = set()
    vectorValues = {}
    position = 0
    while len([vector for vector in seenVectors if vectorValues[vector] >= threshold]) < k:
        currentValues = [attributeList[position] for attributeList in [preprocessed['Attribute Lists'][attribute] for attribute in attributes]]
        threshold = sum([x[0] for x in currentValues])
        for vector in [x[1] for x in currentValues]:
            if vector not in seenVectors:
                currentVector = preprocessed['Weighted Vectors'][vector]
                vectorValues[vector] = sum([currentVector[x] for x in attributes])
                seenVectors.add(vector)
        position = position + 1
    resultVectors = [(vectorValues[vector] * -1, vector) for vector in seenVectors]
    heapq.heapify(resultVectors)
    topK = []
    for x in range(k):
        topK.append(heapq.heappop(resultVectors)[1])
    return topK
    


def approximateShapleyInTopK(preprocessed, d, m, k, j):
    scores = [0 for x in range(d)]
    attributes = [x for x in range(d)]
    for mi in range(m):
        permutation = attributes.copy()
        random.shuffle(attributes)
        for ai in range(len(permutation)):
            prev = getTopK(preprocessed, permutation[:ai], k)
            curr = getTopK(preprocessed, permutation[:ai+1], k)
            currScore = 1 if j in curr else 0
            prevScore = 0 if ai == 0 else (1 if j in prev else 0)
            scores[permutation[ai]] = scores[permutation[ai]] + math.comb(d-1, ai)*d/2**(d-1)*(currScore - prevScore)/m
    return scores

def approximateShapleyNotInTopK(preprocessed, d, m, k, j):
    scores = [0 for x in range(d)]
    attributes = [x for x in range(d)]
    for mi in range(m):
        permutation = attributes.copy()
        random.shuffle(attributes)
        for ai in range(len(permutation)):
            prev = getTopK(preprocessed, permutation[:ai], k)
            curr = getTopK(preprocessed, permutation[:ai+1], k)
            currScore = 1 if j not in curr else 0
            prevScore = 0 if ai == 0 else (1 if j not in prev else 0)
            scores[permutation[ai]] = scores[permutation[ai]] + math.comb(d-1, ai)*d/2**(d-1)*(currScore - prevScore)/m
    return scores

vectors = [[5,3,1],[2,4,4],[3,1,2],[4,1,3],[1,2,5]]
weights = [80,90,4]
print(preProcess(vectors, weights))
print(approximateShapleyInTopK(preProcess(vectors, weights), 3, 10000, 2, 0))
print(approximateShapleyNotInTopK(preProcess(vectors, weights), 3, 10000, 2, 0))
