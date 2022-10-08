import dill
from varying_m_experiment import findQueryPoint
import experiments
from model import ModelGenerator

def UnwrapCandidate(attributes):
    unwrapped = [[0],[1],[2],[3,4,10,11],[5,6,7,8,9,12],[13,14,15,16],[17],[18,19,20],[21]]

    res = []
    for a in attributes:
        res.extend(unwrapped[a])

    return res

def UnwrapHR(attributes):
    unwrapped = [[0],[1],[2],[3,4,10,11],[5,6,7,8,9,12],[13,14,15,16],[17],[21]]

    res = []
    for a in attributes:
        res.extend(unwrapped[a])

    return res

def varyingMExperimentCandidates(tuples, functions, reverseTuples, reverseFunctions, d, unWrapFunction, minim, maxim, k, secondUnwrap=None, secondD=None):
    mTested = [25,50,75,100,125,150,175,200,225,250]

    results = {}

    t, topkFunc, borderlineFunc = findQueryPoint(tuples, k, functions, d, unWrapFunction, minim, maxim)

    inTopKResults = {}
    notInTopKResults = {}
    whyThisTopKResults = {}
    whyInTheseTopKResults = {}

    results['Query Points'] = (t, topkFunc, borderlineFunc)

    inTopKResults['BruteForce'] = experiments.bruteForceInTopK(tuples, functions[topkFunc], k, t, d, unWrapFunction)
    notInTopKResults['BruteForce'] = experiments.bruteForceNotInTopK(tuples, functions[borderlineFunc], k, t, d, unWrapFunction)
    whyThisTopKResults['BruteForce'] = experiments.bruteForceWhyThisTopK(reverseTuples, reverseFunctions[t], k, d if secondD is None else secondD, secondUnwrap if secondUnwrap is not None else unWrapFunction)
    whyInTheseTopKResults['BruteForce'] = experiments.bruteForceWhyInTheseTopK(tuples, functions, k, t, d, unWrapFunction)

    inTopKResults['Approximate'] = {}
    notInTopKResults['Approximate'] = {}
    whyThisTopKResults['Approximate'] = {}
    whyInTheseTopKResults['Approximate'] = {}

    inTopKResults['SHAP'] = {}
    notInTopKResults['SHAP'] = {}
    whyThisTopKResults['SHAP'] = {}
    whyInTheseTopKResults['SHAP'] = {}

    inTopKModel = ModelGenerator()
    inTopKModel.database(tuples).eval_func(functions[topkFunc]).k(k).unwrap_func(unWrapFunction).target(t)

    notInTopKModel = ModelGenerator()
    notInTopKModel.database(tuples).eval_func(functions[borderlineFunc]).k(k).unwrap_func(unWrapFunction).target(t)

    whyThisTopKModel = ModelGenerator()
    whyThisTopKModel.database(reverseTuples).eval_func(reverseFunctions[t]).k(k).unwrap_func(secondUnwrap).setup_top_k()

    whyInTheseTopKModel = ModelGenerator()
    whyInTheseTopKModel.database(tuples).eval_funcs(functions).k(k).target(t).unwrap_func(unWrapFunction).setup_top_ks()

    for m in mTested:
        inTopKResults['Approximate'][m] = experiments.approximateInTopK(tuples, functions[topkFunc], m, k, t, d, inTopKResults['BruteForce']['ShapleyValues'], unWrapFunction)
        notInTopKResults['Approximate'][m] = experiments.approximateNotInTopK(tuples, functions[borderlineFunc], m, k, t, d, notInTopKResults['BruteForce']['ShapleyValues'], unWrapFunction)
        whyThisTopKResults['Approximate'][m] = experiments.approximateWhyThisTopK(reverseTuples, reverseFunctions[t], m, k, secondD, whyThisTopKResults['BruteForce']['ShapleyValues'], secondUnwrap)
        whyInTheseTopKResults['Approximate'][m] = experiments.approximateWhyInTheseTopK(tuples, functions, m, k, t, d, whyInTheseTopKResults['BruteForce']['ShapleyValues'], unWrapFunction)

        inTopKResults['SHAP'][m] = experiments.shapInTopK(inTopKModel, d, m*d, inTopKResults['BruteForce']['ShapleyValues'])
        notInTopKResults['SHAP'][m] = experiments.shapNotInTopK(notInTopKModel, d, m*d, notInTopKResults['BruteForce']['ShapleyValues'])
        whyThisTopKResults['SHAP'][m] = experiments.shapWhyThisTopK(whyThisTopKModel, d, m*secondD, whyThisTopKResults['BruteForce']['ShapleyValues'])
        whyInTheseTopKResults['SHAP'][m] = experiments.shapWhyTheseTopKs(whyInTheseTopKModel, d, m*d, whyInTheseTopKResults['BruteForce']['ShapleyValues'])

    results['InTopK'] = inTopKResults
    results['NotInTopK'] = notInTopKResults
    results['WhyThisTopK'] = whyThisTopKResults
    results['WhyInTheseTopKs'] = whyInTheseTopKResults

    return results

def CandidatesExperiment():
    datasets = dill.load(open('Candidates-Dataset.dill', 'rb'))
    functions = dill.load(open('Revised-Candidate-Functions.dill', 'rb'))
    dill.dump(varyingMExperimentCandidates(datasets['Candidates'], functions['HRs'], datasets['HRs'], functions['Candidates'], 9,
                                 UnwrapCandidate, 3, 6, 5, secondUnwrap=UnwrapHR, secondD=8), open('VaryingMCandidates-Clean    .dill', 'wb'))


if __name__ == "__main__":
    CandidatesExperiment()