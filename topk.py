import heapq

def generateTuples(vectors, evaluationFunction, permutation, position, unWrapFunction):
    tuples = [[0, x] for x in range(len(vectors))]
    subarrayDict = {}
    for x in unWrapFunction(permutation[:position]) if unWrapFunction is not None else permutation[:position]:
        subarrayDict[x] = True
    for vector in range(len(vectors)):
        maskedValue = [vectors[vector][x] if x in subarrayDict else None for x in range(len(vectors[vector]))]
        tuples[vector][0] = tuples[vector][0] + evaluationFunction(maskedValue)
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

def checkCurrentValuesInThreshold(currentValues, vectorValues, vectors, attributes, maskedValue, evaluationFunction, threshold, vectorsGreaterThanThreshold, vectorHeap, unWrapFunction):
    for vector in [x[1] for x in currentValues]:
        if vector not in vectorValues:
            for attribute in unWrapFunction(attributes) if unWrapFunction is not None else maskedValue:
                maskedValue[attribute] = vectors[vector][attribute]
            vectorValues[vector] = evaluationFunction(maskedValue)
            if vectorValues[vector] > threshold:
                vectorsGreaterThanThreshold[vector] = True
            else:
                heapq.heappush(vectorHeap, vector)

def computeTopKThreshold(vectors, attributeLists, evaluationFunction, attributes, k, unWrapFunction=None):
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
        maskedValue = [None for x in range(len(vectors[0]))]
        currentValues = []
        updateCurrentValues(attributes, position, evaluationFunction, attributeLists, maskedValue, currentValues)
        threshold = sum([x[0] for x in currentValues])
        computeAboveThreshold(threshold, vectorsGreaterThanThreshold, vectorValues, vectorHeap)
        checkCurrentValuesInThreshold(currentValues, vectorValues, vectors, attributes, maskedValue, evaluationFunction, threshold, vectorsGreaterThanThreshold, vectorHeap, unWrapFunction)
        position = position + 1
    print(i)
    resultVectors = [(vectorValues[vector] * -1, vector) for vector in vectorsGreaterThanThreshold.keys()]
    return computeTopK(resultVectors, k)

#dead codee, delete
def preProcess(vectors, evaluationFunction=None):
    return
    # attributeLists = []
    # if evaluationFunction != None:
    #     for attribute in range(len(vectors[0])):
    #         maskedVector = [None for x in range(len(vectors[0]))]
    #         attributeList = []
    #         for vector in range(len(vectors)):
    #             for attr in unWrapFunction([attribute]) if unWrapFunction is not None else [attribute]:
    #                 maskedVector[attr] = vectors[vector][attr]
    #             attributeList.append((evaluationFunction(maskedVector), vector))
    #         attributeList.sort(reverse=True)
    #         attributeLists.append(attributeList)
    # else:
    #     for attribute in range(len(vectors[0])):
    #         attributeList = [(vectors[vector][attribute], vector) for vector in range(len(vectors))]
    #         attributeList.sort(reverse=True)
    #         attributeLists.append(attributeList)
    # return attributeLists
