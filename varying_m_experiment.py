import dill
import experiments
import topk

def tInXTopKs(tuples, t, functions, k, minim, maxim, d, unWrapFunction):
    count = 0

    for function in functions:
        evaluatedTuples = topk.generateTuples(tuples, function, [x for x in range(d)], d, unWrapFunction)
        topK = topk.computeTopK(evaluatedTuples, k)
        if t in topK:
            count = count + 1

    return count >= minim and count <= maxim

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

def varyingMExperimentWhyThese(tuples, functions, reverseTuples, reverseFunctions, d, unWrapFunction, minim, maxim, k, secondUnwrap=None, secondD=None):
    mTested = [25,50,75,100,125,150,175,200,225,250]

    results = {}

    t, topkFunc, borderlineFunc = findQueryPoint(tuples, k, functions, d, unWrapFunction, minim, maxim)

    whyInTheseTopKResults = {}

    results['Query Points'] = (t, topkFunc, borderlineFunc)

    whyInTheseTopKResults['BruteForce'] = experiments.bruteForceWhyInTheseTopK(tuples, functions, k, t, d, unWrapFunction)

    whyInTheseTopKResults['Approximate'] = {}

    for m in mTested:
        whyInTheseTopKResults['Approximate'][m] = experiments.approximateWhyInTheseTopK(tuples, functions, m, k, t, d, whyInTheseTopKResults['BruteForce']['ShapleyValues'], unWrapFunction)

    results['WhyInTheseTopKs'] = whyInTheseTopKResults

    return results

def findQueryPointOld(tuples, k, functions, d, unWrapFunction):
    for t in range(len(tuples)):
        topK = inTopK(t, tuples, functions, k, d, unWrapFunction)
        borderline = borderLineTopK(t, tuples, functions, k, d, unWrapFunction)
        if topK is not False and borderline is not False and tInXTopKs(tuples, t, functions, k, 3, 6, d,
                                                                       unWrapFunction) is not False:
            return t, topK, borderline

def varyingMExperiment(tuples, functions, reverseTuples, reverseFunctions, d, unWrapFunction):
    k = 5
    mTested = [25,50,75,100,125,150,175,200,225,250]

    results = {}

    t, topkFunc, borderlineFunc = findQueryPointOld(tuples, k, functions, d, unWrapFunction)

    inTopKResults = {}
    notInTopKResults = {}
    whyThisTopKResults = {}

    results['Query Points'] = (t, topkFunc, borderlineFunc)

    inTopKResults['BruteForce'] = experiments.bruteForceInTopK(tuples, functions[topkFunc], k, t, d, unWrapFunction)
    notInTopKResults['BruteForce'] = experiments.bruteForceNotInTopK(tuples, functions[borderlineFunc], k, t, d, unWrapFunction)
    whyThisTopKResults['BruteForce'] = experiments.bruteForceWhyThisTopK(reverseTuples, reverseFunctions[t], k, d, unWrapFunction)

    inTopKResults['Approximate'] = {}
    notInTopKResults['Approximate'] = {}
    whyThisTopKResults['Approximate'] = {}

    for m in mTested:
        inTopKResults['Approximate'][m] = experiments.approximateInTopK(tuples, functions[topkFunc], m, k, t, d, inTopKResults['BruteForce']['ShapleyValues'], unWrapFunction)
        notInTopKResults['Approximate'][m] = experiments.approximateNotInTopK(tuples, functions[borderlineFunc], m, k, t, d, notInTopKResults['BruteForce']['ShapleyValues'], unWrapFunction)
        whyThisTopKResults['Approximate'][m] = experiments.approximateWhyThisTopK(reverseTuples, reverseFunctions[t], m, k, d, whyThisTopKResults['BruteForce']['ShapleyValues'], unWrapFunction)

    results['InTopK'] = inTopKResults
    results['NotInTopK'] = notInTopKResults
    results['WhyThisTopK'] = whyThisTopKResults

    return results

def SyntheticExperiment():
    datasets = dill.load(open('data/a_z_l_2_varying_d.dill', 'rb'))
    whyInTheseAZL = varyingMExperimentWhyThese(datasets[9][0], datasets[9][1], datasets[9][2], datasets[9][3], 9, None, 3, 6, 5)
    datasets = dill.load(open('data/c_z_l_2_varying_d.dill', 'rb'))
    whyInTheseCZL = varyingMExperimentWhyThese(datasets[9][0], datasets[9][1], datasets[9][2], datasets[9][3], 9, None, 3, 6, 5)
    datasets = dill.load(open('data/i_z_l_2_varying_d.dill', 'rb'))
    whyInTheseIZL = varyingMExperimentWhyThese(datasets[9][0], datasets[9][1], datasets[9][2], datasets[9][3], 9, None, 3, 6, 5)
    datasets = dill.load(open('data/a_z_nl_2_varying_d.dill', 'rb'))
    whyInTheseAZNL = varyingMExperimentWhyThese(datasets[9][0], datasets[9][1], datasets[9][2], datasets[9][3], 9, None, 3, 6, 5)
    datasets = dill.load(open('data/c_z_nl_2_varying_d.dill', 'rb'))
    whyInTheseCZNL = varyingMExperimentWhyThese(datasets[9][0], datasets[9][1], datasets[9][2], datasets[9][3], 9, None, 3, 6, 5)
    datasets = dill.load(open('data/i_z_nl_2_varying_d.dill', 'rb'))
    whyInTheseIZNL = varyingMExperimentWhyThese(datasets[9][0], datasets[9][1], datasets[9][2], datasets[9][3], 9, None, 3, 6, 5)
    datasets = dill.load(open('data/a_z_l_varying_d.dill', 'rb'))
    AZL = varyingMExperiment(datasets[9][0], datasets[9][1], datasets[9][2], datasets[9][3], 9, None)
    datasets = dill.load(open('data/c_z_l_varying_d.dill', 'rb'))
    CZL = varyingMExperiment(datasets[9][0], datasets[9][1], datasets[9][2], datasets[9][3], 9, None)
    datasets = dill.load(open('data/i_z_l_varying_d.dill', 'rb'))
    IZL = varyingMExperiment(datasets[9][0], datasets[9][1], datasets[9][2], datasets[9][3], 9, None)
    datasets = dill.load(open('data/a_z_nl_varying_d.dill', 'rb'))
    AZNL = varyingMExperiment(datasets[9][0], datasets[9][1], datasets[9][2], datasets[9][3], 9, None)
    datasets = dill.load(open('data/c_z_nl_varying_d.dill', 'rb'))
    CZNL = varyingMExperiment(datasets[9][0], datasets[9][1], datasets[9][2], datasets[9][3], 9, None)
    datasets = dill.load(open('data/i_z_nl_varying_d.dill', 'rb'))
    IZNL = varyingMExperiment(datasets[9][0], datasets[9][1], datasets[9][2], datasets[9][3], 9, None)

    AZL['WhyInTheseTopKs'] = whyInTheseAZL['WhyInTheseTopKs']
    CZL['WhyInTheseTopKs'] = whyInTheseCZL['WhyInTheseTopKs']
    IZL['WhyInTheseTopKs'] = whyInTheseIZL['WhyInTheseTopKs']
    AZNL['WhyInTheseTopKs'] = whyInTheseAZNL['WhyInTheseTopKs']
    CZNL['WhyInTheseTopKs'] = whyInTheseCZNL['WhyInTheseTopKs']
    IZNL['WhyInTheseTopKs'] = whyInTheseIZNL['WhyInTheseTopKs']
    AZL['WhyInTheseTopKsQueryPoints'] = whyInTheseAZL['Query Points']
    CZL['WhyInTheseTopKsQueryPoints'] = whyInTheseCZL['Query Points']
    IZL['WhyInTheseTopKsQueryPoints'] = whyInTheseIZL['Query Points']
    AZNL['WhyInTheseTopKsQueryPoints'] = whyInTheseAZNL['Query Points']
    CZNL['WhyInTheseTopKsQueryPoints'] = whyInTheseCZNL['Query Points']
    IZNL['WhyInTheseTopKsQueryPoints'] = whyInTheseIZNL['Query Points']

    dill.dump(AZL, open('SyntheticAZL-Clean.dill', 'wb'))
    dill.dump(CZL, open('SyntheticCZL-Clean.dill', 'wb'))
    dill.dump(IZL, open('SyntheticIZL-Clean.dill', 'wb'))
    dill.dump(AZNL, open('SyntheticAZNL-Clean.dill', 'wb'))
    dill.dump(CZNL, open('SyntheticCZNL-Clean.dill', 'wb'))
    dill.dump(IZNL, open('SyntheticIZNL-Clean.dill', 'wb'))


if __name__ == "__main__":
    SyntheticExperiment()