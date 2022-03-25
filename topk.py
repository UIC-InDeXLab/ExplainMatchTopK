import heapq
import sortedcontainers

def generateTuples(vectors, evaluationFunction, permutation, position):
    tuples = [[0, x] for x in range(len(vectors))]
    for vector in range(len(vectors)):
        tuples[vector][0] = tuples[vector][0] + evaluationFunction([vectors[vector][x] if x in permutation[:position] else 0 for x in range(len(vectors[vector]))])
    return [(x[0]*-1, x[1]) for x in tuples]

def computeInTopK(tuples, k, j):
    value = tuples[j][0]
    greater = 0
    for t in tuples:
        if t[0] < value:
            greater = greater + 1
    return greater < k

def computeTopK(tuples, k):
    heapq.heapify(tuples)
    topK = []
    for x in range(k):
        topK.append(heapq.heappop(tuples)[1])
    return topK


def updateCurrentValues(attributes, position, evaluationFunction, attributeLists, maskedValue, currentValues):
    for attribute in attributes:
        currentValues.append(attributeLists[attribute][position])   

def computeAboveThreshold(threshold, vectorsGreaterThanThreshold, vectorValues, vectorHeap):
    while len(vectorHeap) > 0 and vectorValues[vectorHeap[0]] > threshold:
        vectorsGreaterThanThreshold[vectorHeap[0]] = True
        heapq.heappop(vectorHeap)

def checkCurrentValuesInThreshold(currentValues, vectorValues, vectors, attributes, maskedValue, evaluationFunction, threshold, vectorsGreaterThanThreshold, vectorHeap):
    for vector in [x[1] for x in currentValues]:
        if vector not in vectorValues:
            for attribute in attributes:
                maskedValue[attribute] = vectors[vector][attribute]
            vectorValues[vector] = evaluationFunction(maskedValue)
            if vectorValues[vector] > threshold:
                vectorsGreaterThanThreshold[vector] = True
            else:
                heapq.heappush(vectorHeap, vector)

def computeTopKThreshold(vectors, attributeLists, evaluationFunction, attributes, k): 
    if (len(attributes) == 0):
        return []
    threshold = float('inf')
    vectorValues = {}
    position = 0
    vectorsGreaterThanThreshold = {}
    vectorHeap = []
    i = 0
    while len(vectorsGreaterThanThreshold) < k:
        i += 1
        maskedValue = [0 for x in range(len(vectors[0]))]
        currentValues = []
        updateCurrentValues(attributes, position, evaluationFunction, attributeLists, maskedValue, currentValues)
        threshold = sum([x[0] for x in currentValues])
        computeAboveThreshold(threshold, vectorsGreaterThanThreshold, vectorValues, vectorHeap)
        checkCurrentValuesInThreshold(currentValues, vectorValues, vectors, attributes, maskedValue, evaluationFunction, threshold, vectorsGreaterThanThreshold, vectorHeap)
        position = position + 1
    print(i)
    resultVectors = [(vectorValues[vector] * -1, vector) for vector in vectorsGreaterThanThreshold.keys()]
    return computeTopK(resultVectors, k)

def preProcess(vectors, evaluationFunction=None):
    attributeLists = []
    if evaluationFunction != None:
        for attribute in range(len(vectors[0])):
            maskedVector = [0 for x in range(len(vectors[0]))]
            attributeList = []
            for vector in range(len(vectors)):
                maskedVector[attribute] = vectors[vector][attribute]
                attributeList.append((evaluationFunction(maskedVector), vector))
            attributeList.sort(reverse=True)
            attributeLists.append(attributeList)
    else:
        for attribute in range(len(vectors[0])):
            attributeList = [(vectors[vector][attribute], vector) for vector in range(len(vectors))]
            attributeList.sort(reverse=True)
            attributeLists.append(attributeList)
    return attributeLists
