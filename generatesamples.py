import random
import numpy as np

def generateSamples(n, d, p=1, zipfA=1.5):
    results = {'Functions':[]}
    # results['Tuples'] = [[random.random() for x in range(d)] for y in range(n)]
    # results['Tuples'] = [[x/np.linalg.norm(y, 1) for x in y] for y in results['Tuples']]
    for a in range(n):
        results['Functions'].append(lambda e, weights=np.random.zipf(zipfA, size=(d)).astype(float).tolist(), a=d: sum([(0 if e[x] == None else (abs(e[x])**p)*weights[x]) for x in range(a)]))
    return results

def generateNonLinearSamples(n, d):
    return generateSamples(n, d, p=2)
# def generateNonLinearSamples(n, d):
#     results = {}
#     results['Tuples'] = [[random.random() for x in range(d)] for y in range(n)]
#     results['Tuples'] = [[x/np.linalg.norm(y, 1) for x in y] for y in results['Tuples']]
#     results['Functions'] = []
#     for a in range(n):
#         results['Functions'].append(lambda e, weights=[random.random() for x in range(d)], a=d: sum([e[x]**2*weights[x] for x in range(a)]))
#     return results
