import heapq

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


def computeTopKThreshold(vectors, attributeLists, evaluationFunction, attributes, k): 
    if (len(attributes) == 0):
        return []
    threshold = float('inf')
    seenVectors = set()
    vectorValues = {}
    position = 0
    while len([vector for vector in seenVectors if vectorValues[vector] >= threshold]) < k:
        currentValues = [(evaluationFunction([attributeLists[attribute][position][0] if x == attribute else 0 for x in range(len(vectors[0]))]), attributeLists[attribute][position][1]) for attribute in attributes]
        threshold = sum([x[0] for x in currentValues])
        for vector in [x[1] for x in currentValues]:
            if vector not in seenVectors:
                vectorValues[vector] = evaluationFunction([vectors[vector][x] if x in attributes else 0 for x in range(len(vectors[0]))])
                seenVectors.add(vector)
        position = position + 1
    resultVectors = [(vectorValues[vector] * -1, vector) for vector in seenVectors]
    return computeTopK(resultVectors, k)

def preProcess(vectors):
    attributeLists = []
    for attribute in range(len(vectors[0])):
        attributeList = [(vectors[vector][attribute], vector) for vector in range(len(vectors))]
        attributeList.sort(reverse=True)
        attributeLists.append(attributeList)
    return attributeLists
