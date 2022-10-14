import dill
import topk
import experiments

def inTopK(t, tuples, functions, k, d, unWrapFunction):
    for f in range(len(functions)):
        evaluatedTuples = topk.generateTuples(tuples, functions[f], [x for x in range(d)], d, unWrapFunction)
        if t in topk.computeTopK(evaluatedTuples, k):
            return f
    return False

def borderLineTopK(t, tuples, functions, k, d, unWrapFunction, skipList=[]):
    for f in range(len(functions)):
        if f in skipList:
            continue
        evaluatedTuples = topk.generateTuples(tuples, functions[f], [x for x in range(d)], d, unWrapFunction)
        if t not in topk.computeTopK(evaluatedTuples, k) and t in topk.computeTopK(evaluatedTuples, k+1):
            return f
    return False

def tInXTopKs(tuples, t, functions, k, minim, maxim, d, unWrapFunction):
    count = 0

    for function in functions:
        evaluatedTuples = topk.generateTuples(tuples, function, [x for x in range(d)], d, unWrapFunction)
        topK = topk.computeTopK(evaluatedTuples, k)
        if t in topK:
            count = count + 1

    return count >= minim and count <= maxim


def findQueryPoint(tuples, k, functions, d, unWrapFunction, minim, maxim):
    while True:
        for t in range(len(tuples)):
            if not tInXTopKs(tuples, t, functions, k, minim, maxim, d, unWrapFunction):
                continue
            topK = inTopK(t, tuples, functions, k, d, unWrapFunction)
            borderline = borderLineTopK(t, tuples, functions, k, d, unWrapFunction)
            print(t, topK, borderline)
            if borderline is not False  :
                return t, topK, borderline
        maxim = maxim + 1

def fullAttributesCandidates():
    results = {}

    datasets = dill.load(open('Candidates-Dataset.dill', 'rb'))
    functions = dill.load(open('Revised-Candidate-Functions.dill', 'rb'))

    t, topkFunc, borderlineFunc = findQueryPoint(datasets['Candidates'], 5, functions['HRs'], 22, None, 3, 6)

    inTopKResults = {}
    notInTopKResults = {}
    whyThisTopKResults = {}
    whyInTheseTopKResults = {}

    inTopKResults['Approximate'] = experiments.approximateInTopK(datasets['Candidates'], functions['HRs'][topkFunc], 200, 5, t, 22, [0.0 for x in range(22)], None)
    notInTopKResults['Approximate'] = experiments.approximateNotInTopK(datasets['Candidates'], functions['HRs'][borderlineFunc], 200, 5, t, 22, [0.0 for x in range(22)], None)
    whyThisTopKResults['Approximate'] = experiments.approximateWhyThisTopK(datasets['HRs'], functions['Candidates'][t], 200, 5, 22, [0.0 for x in range(22)], None)
    whyInTheseTopKResults['Approximate'] = experiments.approximateWhyInTheseTopK(datasets['Candidates'], functions['HRs'], 200, 5, t, 22, [0.0 for x in range(22)], None)

    results['Query Point'] = (t, topkFunc, borderlineFunc)
    results['InTopK'] = inTopKResults
    results['NotInTopK'] = notInTopKResults
    results['WhyThisTopK'] = whyThisTopKResults
    results['WhyInTheseTopKs'] = whyInTheseTopKResults

    dill.dump(results, open('case_study_experiment.dill', 'wb'))
    return


if __name__ == "__main__":
    fullAttributesCandidates()