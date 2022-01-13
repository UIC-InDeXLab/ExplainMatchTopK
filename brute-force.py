import math

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
            print(prev)
            print(curr)
            if j in [x[1] for x  in sorted(curr, key=lambda e: e[0], reverse=True)[:k]] and j not in [x[1] for x in sorted(prev, key=lambda e: e[0], reverse=True)[:k]]:
                scores[c] = scores[c] + 1/(2**(len(weights)-1))
    return scores
                    
            
                
    
vectors = [[5,3,1],[2,4,4],[3,1,2],[4,1,3],[1,2,5]]
weights = [80,90,4]
print(ComputeShapleyInTopK(vectors, weights, 2, 0))
