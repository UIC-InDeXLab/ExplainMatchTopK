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
        maskedValue[attribute] = attributeLists[attribute][position][0]
        currentValues.append((evaluationFunction(maskedValue), attributeLists[attribute][position][1]))
        maskedValue[attribute] = 0

def computeAboveThreshold(vectorsAboveThreshold, threshold, vectorsGreaterThanThreshold, vectorValues):
    templist =[]
    for vector in vectorsAboveThreshold:
        if vectorValues[vector] > threshold:
            templist.append(vector)
            vectorsGreaterThanThreshold.append(vector)
        else:
            break
    for elem in templist:
        vectorsAboveThreshold.remove(elem)

def checkCurrentValuesInThreshold(currentValues, vectorValues, vectors, attributes, maskedValue, evaluationFunction, threshold, vectorsAboveThreshold, vectorsGreaterThanThreshold):
    for vector in [x[1] for x in currentValues]:
        if vector not in vectorValues:
            for attribute in attributes:
                maskedValue[attribute] = vectors[vector][attribute]
            vectorValues[vector] = evaluationFunction(maskedValue)
            if vectorValues[vector] > threshold:
                vectorsGreaterThanThreshold.append(vector)
            else:
                vectorsAboveThreshold.add(vector)

def computeTopKThreshold(vectors, attributeLists, evaluationFunction, attributes, k): 
    if (len(attributes) == 0):
        return []
    threshold = float('inf')
    vectorValues = {}
    position = 0
    vectorsAboveThreshold = sortedcontainers.SortedList([])
    vectorsGreaterThanThreshold = []
    while len(vectorsGreaterThanThreshold) < k:
        maskedValue = [0 for x in range(len(vectors[0]))]
        currentValues = []
        updateCurrentValues(attributes, position, evaluationFunction, attributeLists, maskedValue, currentValues)
        '''currentValues = [(evaluationFunction([attributeLists[attribute][position][0] if x == attribute else 0 for x in range(len(vectors[0]))]), attributeLists[attribute][position][1]) for attribute in attributes]'''
        threshold = sum([x[0] for x in currentValues])
        computeAboveThreshold(vectorsAboveThreshold, threshold, vectorsGreaterThanThreshold, vectorValues)
        checkCurrentValuesInThreshold(currentValues, vectorValues, vectors, attributes, maskedValue, evaluationFunction, threshold, vectorsAboveThreshold, vectorsGreaterThanThreshold)
        position = position + 1
    resultVectors = [(vectorValues[vector] * -1, vector) for vector in vectorValues.keys()]
    return computeTopK(resultVectors, k)

def preProcess(vectors):
    attributeLists = []
    for attribute in range(len(vectors[0])):
        attributeList = [(vectors[vector][attribute], vector) for vector in range(len(vectors))]
        attributeList.sort(reverse=True)
        attributeLists.append(attributeList)
    return attributeLists
