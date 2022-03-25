import random
import numpy as np

def generateSamples(n, d):
    results = {}
    results['Tuples'] = [[random.random() for x in range(d)] for y in range(n)]
    results['Tuples'] = [[x/np.linalg.norm(y, 1) for x in y] for y in results['Tuples']]
    results['Functions'] = []
    for a in range(n):
        results['Functions'].append(lambda e, weights=[random.random() for x in range(d)], a=d: sum([e[x]*weights[x] for x in range(a)]))
    return results
