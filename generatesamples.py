import random

def generateSamples(n, d):
    results = {}
    results['Tuples'] = [[random.random() for x in range(d)] for y in range(n)]
    results['Functions'] = []
    for a in range(n):
        weights = [random.random() for x in range(d)]
        results['Functions'].append(lambda e:(sum([e[x]*weights[x] for x in range(d)])))
    return results
