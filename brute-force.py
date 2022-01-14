import math
import heapq

def ComputeShapleyInTopK(vectors, weights, k, j):
    scores = [0 for x in range(len(weights))]
    for c in range(len(weights)):        
        for a in range(2**len(weights)):
            if math.floor(a/(2**c))%(2) == 0:
                continue
            prev = [[0, x] for x in range(len(vectors))]
            for b in range(len(weights)):
                if math.floor(a/(2**b))%(2) == 0:
                    for vector in range(len(vectors)):
                        prev[vector][0] = prev[vector][0] + vectors[vector][b] * weights[b]
            curr = [x[:] for x in prev[:]]
            for vector in range(len(vectors)):
                curr[vector][0] = curr[vector][0] + vectors[vector][c] * weights[c]
            prevTuples = [(x[0]*-1, x[1]) for x in prev]
            currTuples = [(x[0]*-1, x[1]) for x in curr]
            heapq.heapify(prevTuples)
            heapq.heapify(currTuples)
            topKPrev = []
            topKCurr = []
            for x in range(k):
                topKPrev.append(heapq.heappop(prevTuples)[1])
                topKCurr.append(heapq.heappop(currTuples)[1])
            currScore = 1 if j in topKCurr else 0
            prevScore = 0 if prev[j][0] == 0 else (1 if j in topKPrev else 0)
            scores[c] = scores[c] + (currScore - prevScore)/(2**(len(weights)-1))
    return scores
                    
            
def ComputeShapleyNotInTopK(vectors, weights, k, j):
    scores = [0 for x in range(len(weights))]
    for c in range(len(weights)): # c - dimensions       
        for a in range(2**len(weights)): # a - various attribute combinations
            if math.floor(a/(2**c))%(2) == 0:
                continue
            prev = [[0, x] for x in range(len(vectors))]
            for b in range(len(weights)):
                if math.floor(a/(2**b))%(2) == 0:
                    for vector in range(len(vectors)):
                        prev[vector][0] = prev[vector][0] + vectors[vector][b] * weights[b]
            curr = [x[:] for x in prev[:]]
            for vector in range(len(vectors)):
                curr[vector][0] = curr[vector][0] + vectors[vector][c] * weights[c]
            prevTuples = [(x[0]*-1, x[1]) for x in prev]
            currTuples = [(x[0]*-1, x[1]) for x in curr]
            heapq.heapify(prevTuples)
            heapq.heapify(currTuples)
            topKPrev = []
            topKCurr = []
            for x in range(k):
                topKPrev.append(heapq.heappop(prevTuples)[1])
                topKCurr.append(heapq.heappop(currTuples)[1])
            currScore = 1 if j not in topKCurr else 0
            prevScore = 0 if prev[j][0] == 0 else (1 if j not in topKPrev else 0)
            scores[c] = scores[c] + (currScore - prevScore)/(2**(len(weights)-1))
    return scores        
                
   
def ComputeShapleyTopKLookLikeThis(vectors, weights, k):
    scores = [0 for x in range(len(weights))]
    for c in range(len(weights)): # c - dimensions       
        for a in range(2**len(weights)): # a - various attribute combinations
            if math.floor(a/(2**c))%(2) == 0:
                continue
            prev = [[0, x] for x in range(len(vectors))]
            for b in range(len(weights)):
                if math.floor(a/(2**b))%(2) == 0:
                    for vector in range(len(vectors)):
                        prev[vector][0] = prev[vector][0] + vectors[vector][b] * weights[b]
            curr = [x[:] for x in prev[:]]
            for vector in range(len(vectors)):
                curr[vector][0] = curr[vector][0] + vectors[vector][c] * weights[c]
            prevTuples = [(x[0]*-1, x[1]) for x in prev]
            currTuples = [(x[0]*-1, x[1]) for x in curr]
            heapq.heapify(prevTuples)
            heapq.heapify(currTuples)
            topKPrev = []
            topKCurr = []
            for x in range(k):
                topKPrev.append(heapq.heappop(prevTuples)[1])
                topKCurr.append(heapq.heappop(currTuples)[1])
            setCurrTopK = set(topKCurr)
            setPrevTopK = set(topKPrev)
            IoU = 1-len(setCurrTopK.intersection(setPrevTopK))/len(setCurrTopK.union(setPrevTopK)) # Jaccard similarity
            scores[c] = scores[c] + (IoU)/(2**(len(weights)-1)) 
    return scores                  
    
vectors = [[5,3,1],[2,4,4],[3,1,2],[4,1,3],[1,2,5]]
weights = [80,90,4]
print(ComputeShapleyInTopK(vectors,weights, 2, 0))
print(ComputeShapleyNotInTopK(vectors,weights, 2, 0))
print(ComputeShapleyTopKLookLikeThis(vectors,weights, 2))
