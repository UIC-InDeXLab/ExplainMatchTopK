import itertools
import pickle

import dill
import bruteforce
import approximate
import time
import topk
import numpy as np
import multiprocessing
import threading
import concurrent.futures
from itertools import chain, combinations


def bruteForceInTopK(tuples, evalFunc, k, j, d, unWrapFunction):
    results = {}

    start_time = time.process_time_ns()
    shapley = bruteforce.ComputeShapleyInTopK(tuples, evalFunc, k, j, d, unWrapFunction)
    runtime = time.process_time_ns() - start_time

    results['RuntimeNS'] = runtime
    results['ShapleyValues'] = shapley

    return results

def bruteForceNotInTopK(tuples, evalFunc, k, j, d, unWrapFunction):
    results = {}

    start_time = time.process_time_ns()
    shapley = bruteforce.ComputeShapleyNotInTopK(tuples, evalFunc, k, j, d, unWrapFunction)
    runtime = time.process_time_ns() - start_time

    results['RuntimeNS'] = runtime
    results['ShapleyValues'] = shapley

    return results

def bruteForceWhyThisTopK(tuples, evalFunc, k, d, unWrapFunction):
    results = {}

    start_time = time.process_time_ns()
    shapley = bruteforce.ComputeShapleyTopKLookLikeThis(tuples, evalFunc, k, d, unWrapFunction)
    runtime = time.process_time_ns() - start_time

    results['RuntimeNS'] = runtime
    results['ShapleyValues'] = shapley

    return results

def bruteForceWhyInTheseTopK(tuples, evalFuncs, k, j, d, unWrapFunction):
    results = {}

    start_time = time.process_time_ns()
    shapley = bruteforce.ComputeWhyInTheseTopKs(tuples, evalFuncs, k, j, d, unWrapFunction)
    runtime = time.process_time_ns() - start_time

    results['RuntimeNS'] = runtime
    results['ShapleyValues'] = shapley

    return results

def approximateInTopK(tuples, evalFunc, m, k, j, d, bruteForce, unWrapFunction):
    results = {}

    start_time = time.process_time_ns()
    shapley = approximate.approximateShapleyInTopK(tuples, evalFunc, m, k, j, d, unWrapFunction)
    runtime = time.process_time_ns() - start_time

    results['RuntimeNS'] = runtime
    results['ShapleyValues'] = shapley
    X = np.asarray(shapley)
    Y = np.asarray(bruteForce)
    results['AverageDiffernce'] = np.mean(np.abs(X - Y))
    results['StdDifference'] = np.std(np.abs(X-Y))

    return results

def approximateNotInTopK(tuples, evalFunc, m, k, j, d, bruteForce, unWrapFunction):
    results = {}

    start_time = time.process_time_ns()
    shapley = approximate.approximateShapleyNotInTopK(tuples, evalFunc, m, k, j, d, unWrapFunction)
    runtime = time.process_time_ns() - start_time

    results['RuntimeNS'] = runtime
    results['ShapleyValues'] = shapley
    X = np.asarray(shapley)
    Y = np.asarray(bruteForce)
    results['AverageDiffernce'] = np.mean(np.abs(X - Y))
    results['StdDifference'] = np.std(np.abs(X-Y))

    return results

def approximateWhyThisTopK(tuples, evalFunc, m, k, d, bruteForce, unWrapFunction):
    results = {}

    start_time = time.process_time_ns()
    shapley = approximate.approximateShapleyTopKLookLikeThis(tuples, evalFunc, m, k, d, unWrapFunction)
    runtime = time.process_time_ns() - start_time

    results['RuntimeNS'] = runtime
    results['ShapleyValues'] = shapley
    X = np.asarray(shapley)
    Y = np.asarray(bruteForce)
    results['AverageDiffernce'] = np.mean(np.abs(X - Y))
    results['StdDifference'] = np.std(np.abs(X-Y))

    return results

def approximateWhyInTheseTopK(tuples, evalFuncs, m, k, j, d, bruteForce, unWrapFunction):
    results = {}

    start_time = time.process_time_ns()
    shapley = approximate.approximateWhyInTheseTopKs(tuples, evalFuncs, m, k, j, d, unWrapFunction)
    runtime = time.process_time_ns() - start_time

    results['RuntimeNS'] = runtime
    results['ShapleyValues'] = shapley
    X = np.asarray(shapley)
    Y = np.asarray(bruteForce)
    results['AverageDiffernce'] = np.mean(np.abs(X - Y))
    results['StdDifference'] = np.std(np.abs(X-Y))

    return results


def individualInRangeOfTopKs(tuples, functions, minim, maxim, k, unWrapFunction):
    count = [0 for i in range(len(tuples))]

    for function in functions:
        evaluatedTuples = topk.generateTuples(tuples, function, [x for x in range(len(tuples[0]))], len(tuples[0]), unWrapFunction)
        topK = topk.computeTopK(evaluatedTuples, k)
        for j in topK:
            count[j] = count[j] + 1

    for c in range(len(count)):
        if count[c] >= minim and count[c] <= maxim:
            return c


def varyingMExperiment(tuples, functions, reverseTuples, reverseFunctions, d, unWrapFunction, minim, maxim, k):
    mTested = [25,50,75,100,125,150,175,200,225,250]

    results = {}

    t, topkFunc, borderlineFunc = findQueryPoint(tuples, k, functions, d, unWrapFunction, minim, maxim)

    inTopKResults = {}
    notInTopKResults = {}
    whyThisTopKResults = {}
    whyInTheseTopKResults = {}

    results['Query Points'] = (t, topkFunc, borderlineFunc)

    inTopKResults['BruteForce'] = bruteForceInTopK(tuples, functions[topkFunc], k, t, d, unWrapFunction)
    notInTopKResults['BruteForce'] = bruteForceNotInTopK(tuples, functions[borderlineFunc], k, t, d, unWrapFunction)
    whyThisTopKResults['BruteForce'] = bruteForceWhyThisTopK(reverseTuples, reverseFunctions[t], k, d, unWrapFunction)
    whyInTheseTopKResults['BruteForce'] = bruteForceWhyInTheseTopK(tuples, functions, k, t, d, unWrapFunction)

    inTopKResults['Approximate'] = {}
    notInTopKResults['Approximate'] = {}
    whyThisTopKResults['Approximate'] = {}
    whyInTheseTopKResults['Approximate'] = {}

    for m in mTested:
        inTopKResults['Approximate'][m] = approximateInTopK(tuples, functions[topkFunc], m, k, t, d, inTopKResults['BruteForce']['ShapleyValues'], unWrapFunction)
        notInTopKResults['Approximate'][m] = approximateNotInTopK(tuples, functions[borderlineFunc], m, k, t, d, notInTopKResults['BruteForce']['ShapleyValues'], unWrapFunction)
        whyThisTopKResults['Approximate'][m] = approximateWhyThisTopK(reverseTuples, reverseFunctions[t], m, k, d, whyThisTopKResults['BruteForce']['ShapleyValues'], unWrapFunction)
        whyInTheseTopKResults['Approximate'][m] = approximateWhyInTheseTopK(tuples, functions, m, k, t, d, whyInTheseTopKResults['BruteForce']['ShapleyValues'], unWrapFunction)

    results['InTopK'] = inTopKResults
    results['NotInTopK'] = notInTopKResults
    results['WhyThisTopK'] = whyThisTopKResults
    results['WhyInTheseTopKs'] = whyInTheseTopKResults

    return results

def varyingDExperiment(datasets, unWrapFunction, minim, maxim, k):
    resultsFinal = {}

    executor = concurrent.futures.ThreadPoolExecutor()
    skipFutureTopK = False
    skipFutureNotTopK = False
    skipFutureWhyThisTopK = False
    skipFutureWhyTheseTopKs = False
    apprxSkipFutureTopK = False
    apprxSkipFutureNotTopK = False
    apprxSkipFutureWhyThisTopK = False
    apprxSkipFutureWhyTheseTopKs = False

    for index in sorted(datasets.keys()):
        tuples, functions, reverseTuples, reverseFunctions = datasets[index]

        d = index
        t, topkFunc, borderlineFunc = findQueryPoint(tuples, k, functions, d, unWrapFunction, minim, maxim)

        results = {}

        results['Query Points'] = (t, topkFunc, borderlineFunc)

        inTopKResults = {}
        notInTopKResults = {}
        whyThisTopKResults = {}
        whyInTheseTopKResults = {}

        if not skipFutureTopK:
            try:
                inTopKResults['BruteForce'] = executor.submit(bruteForceInTopK, tuples, functions[topkFunc], k, t, d, unWrapFunction).result(timeout=3600)
            except concurrent.futures.TimeoutError:
                inTopKResults['BruteForce'] = 'Too long!'
                skipFutureTopK = True
        else:
            inTopKResults['BruteForce'] = 'Too long!'

        if not skipFutureNotTopK:
            try:
                notInTopKResults['BruteForce'] = executor.submit(bruteForceNotInTopK, tuples, functions[borderlineFunc], k, t, d, unWrapFunction).result(timeout=3600)
            except concurrent.futures.TimeoutError:
                notInTopKResults['BruteForce'] = 'Too long!'
                skipFutureNotTopK = True
        else:
            notInTopKResults['BruteForce'] = 'Too long!'

        if not skipFutureWhyThisTopK:
            try:
                whyThisTopKResults['BruteForce'] = executor.submit(bruteForceWhyThisTopK, reverseTuples, reverseFunctions[t], k, d, unWrapFunction).result(timeout=3600)
            except concurrent.futures.TimeoutError:
                whyThisTopKResults['BruteForce'] = 'Too long!'
                skipFutureWhyThisTopK = True
        else:
            whyThisTopKResults['BruteForce'] = 'Too long!'


        if not skipFutureWhyTheseTopKs:
            try:
                whyInTheseTopKResults['BruteForce'] = executor.submit(bruteForceWhyInTheseTopK, tuples, functions, k, t, d, unWrapFunction).result(timeout=3600)
            except concurrent.futures.TimeoutError:
                whyInTheseTopKResults['BruteForce'] = 'Too long!'
                skipFutureWhyTheseTopKs = True
        else:
            whyInTheseTopKResults['BruteForce'] = 'Too long!'

        if not apprxSkipFutureTopK:
            try:
                inTopKResults['Approximate'] = executor.submit(approximateInTopK, tuples, functions[topkFunc], 100, k, t, d, inTopKResults['BruteForce']['ShapleyValues'] if type(inTopKResults['BruteForce']) is dict else [0.0 for x in range(d)], unWrapFunction).result(timeout=3600)
            except concurrent.futures.TimeoutError:
                inTopKResults['Approximate'] = 'Too long!'
                apprxSkipFutureTopK = True
        else:
            inTopKResults['Approximate'] = 'Too long!'

        if not apprxSkipFutureNotTopK:
            try:
                notInTopKResults['Approximate'] = executor.submit(approximateInTopK, tuples, functions[borderlineFunc], 100, k, t, d, notInTopKResults['BruteForce']['ShapleyValues'] if type(notInTopKResults['BruteForce']) is dict else [0.0 for x in range(d)], unWrapFunction).result(timeout=3600)
            except concurrent.futures.TimeoutError:
                notInTopKResults['Approximate'] = 'Too long!'
                apprxSkipFutureNotTopK = True
        else:
            notInTopKResults['Approximate'] = 'Too long!'

        if not apprxSkipFutureWhyThisTopK:
            try:
                whyThisTopKResults['Approximate'] = executor.submit(approximateWhyThisTopK, reverseTuples, reverseFunctions[t], 100, k, d, whyThisTopKResults['BruteForce']['ShapleyValues'] if type(whyThisTopKResults['BruteForce']) is dict else [0.0 for x in range(d)], unWrapFunction).result(timeout=3600)
            except concurrent.futures.TimeoutError:
                whyThisTopKResults['Approximate'] = 'Too long!'
                apprxSkipFutureWhyThisTopK = True
        else:
            whyThisTopKResults['Approximate'] = 'Too long!'

        if not apprxSkipFutureWhyTheseTopKs:
            try:
                whyInTheseTopKResults['Approximate'] = executor.submit(approximateWhyInTheseTopK, tuples, functions, 100, k, t, d, whyInTheseTopKResults['BruteForce']['ShapleyValues'] if type(whyInTheseTopKResults['BruteForce']) is dict else [0.0 for x in range(d)], unWrapFunction).result(timeout=3600)
            except concurrent.futures.TimeoutError:
                whyInTheseTopKResults['Approximate'] = 'Too long!'
                apprxSkipFutureWhyTheseTopKs = True
        else:
            whyInTheseTopKResults['Approximate'] = 'Too long!'

        results['InTopK'] = inTopKResults
        results['NotInTopK'] = notInTopKResults
        results['WhyThisTopK'] = whyThisTopKResults
        results['WhyInTheseTopKs'] = whyInTheseTopKResults
        resultsFinal[index] = results

    return resultsFinal

def newVaryingD(d, unWrapFunction):
    datasets = dill.load(open('Varying-D.dill', 'rb'))
    prev_results = dill.load(open('UpdatedExperimentDResults3.dill', 'rb'))
    k = 5
    executor = concurrent.futures.ThreadPoolExecutor()

    skip_future_top_k = False
    skip_future_not_top_k = False
    apprx_skip_future_top_k = False
    apprx_skip_future_not_top_k = False

    for index in sorted(datasets.keys()):
        dataset = datasets[index]

        evaluated_tuples = topk.generateTuples(dataset['Tuples'], dataset['Functions'][0],
                                              [x for x in range(len(dataset['Tuples'][0]))], len(dataset['Tuples'][0]), unWrapFunction)
        top_k = topk.computeTopK(evaluated_tuples, k)
        top_k_plus_one = topk.computeTopK(evaluated_tuples, k+1)

        in_top_k_results: dict[any, any] = {}
        not_in_top_k_results: dict[any, any] = {}

        if not skip_future_top_k:
            try:
                in_top_k_results['BruteForce'] = executor.submit(bruteForceInTopK, dataset['Tuples'],
                                                              dataset['Functions'][0], k, top_k[k - 1], d, unWrapFunction).result(
                    timeout=3600)
            except concurrent.futures.TimeoutError:
                in_top_k_results['BruteForce'] = 'Too long!'
                skip_future_top_k = True
        else:
            in_top_k_results['BruteForce'] = 'Too long!'
        if not skip_future_not_top_k:
            try:
                not_in_top_k_results['BruteForce'] = executor.submit(bruteForceNotInTopK, dataset['Tuples'], dataset['Functions'][0], k, top_k_plus_one[k], d, unWrapFunction).result(timeout=3600)
            except concurrent.futures.TimeoutError:
                not_in_top_k_results['BruteForce'] = 'Too long!'
                skip_future_not_top_k = True
        else:
            not_in_top_k_results['BruteForce'] = 'Too long!'
        if not apprx_skip_future_top_k:
            try:
                in_top_k_results['Approximate'] = executor.submit(approximateInTopK, dataset['Tuples'],
                                                               dataset['Functions'][0], 100, k, top_k[k - 1], d,
                                                               in_top_k_results['BruteForce']['ShapleyValues'] if type(
                                                                   in_top_k_results['BruteForce']) is dict else [0.0 for x in
                                                                                                              range(
                                                                                                                  len(
                                                                                                                      dataset[
                                                                                                                          'Tuples'][
                                                                                                                          0]))], unWrapFunction).result(
                    timeout=3600)
            except concurrent.futures.TimeoutError:
                in_top_k_results['Approximate'] = 'Too long!'
                apprx_skip_future_top_k = True
        else:
            in_top_k_results['Approximate'] = 'Too long!'

        if not apprx_skip_future_not_top_k:
            try:
                not_in_top_k_results['Approximate'] = executor.submit(approximateInTopK, dataset['Tuples'],
                                                                  dataset['Functions'][0], 100, k, top_k_plus_one[k],
                                                                      d, not_in_top_k_results['BruteForce']['ShapleyValues'] if type(
                                                                      not_in_top_k_results['BruteForce']) is dict else [0.0 for
                                                                                                                    x in
                                                                                                                    range(
                                                                                                                        len(
                                                                                                                            dataset[
                                                                                                                                'Tuples'][
                                                                                                                                0]))], unWrapFunction).result(
                    timeout=3600)
            except concurrent.futures.TimeoutError:
                not_in_top_k_results['Approximate'] = 'Too long!'
                apprx_skip_future_not_top_k = True
        else:
            not_in_top_k_results['Approximate'] = 'Too long!'
        prev_results[index]['InTopK'] = in_top_k_results
        prev_results[index]['NotInTopK'] = not_in_top_k_results

    dill.dump(prev_results, open('UpdatedExperimentDResults4.dill', 'wb'))

def newVaryingM(d, unWrapFunction):
    dataset = dill.load(open('Varying-D.dill', 'rb'))[8]
    prev_results = dill.load(open('ExperimentMResults8.dill', 'rb'))
    k = 5

    evaluated_tuples = topk.generateTuples(dataset['Tuples'], dataset['Functions'][0],
                                          [x for x in range(len(dataset['Tuples'][0]))], len(dataset['Tuples'][0]), unWrapFunction)
    topK = topk.computeTopK(evaluated_tuples, k)
    topKPlusOne = topk.computeTopK(evaluated_tuples, k + 1)

    inTopKResults = {}
    notInTopKResults = {}

    inTopKResults['BruteForce'] = bruteForceInTopK(dataset['Tuples'], dataset['Functions'][0], k, topK[k - 1], d, unWrapFunction)
    notInTopKResults['BruteForce'] = bruteForceNotInTopK(dataset['Tuples'], dataset['Functions'][0], k, topKPlusOne[k], d, unWrapFunction)

    inTopKResults['Approximate'] : dict[any,any] = {}
    notInTopKResults['Approximate'] : dict[any,any] = {}

    mTested = [25,50,75,100,125,150,175,200,225,250]
    for m in mTested:
        inTopKResults['Approximate'][m] = approximateInTopK(dataset['Tuples'], dataset['Functions'][0], m, k, topK[k-1], d, inTopKResults['BruteForce']['ShapleyValues'], unWrapFunction)
        notInTopKResults['Approximate'][m] = approximateNotInTopK(dataset['Tuples'], dataset['Functions'][0], m, k, topKPlusOne[k], d, notInTopKResults['BruteForce']['ShapleyValues'], unWrapFunction)

    prev_results['InTopK'] = inTopKResults
    prev_results['NotInTopK'] = notInTopKResults

    dill.dump(prev_results, open('UpdatedExperimentMResults.dill', 'wb'))

def computeMaxShapleyValues(ShapleyValues):
     return [tup[1] for tup in sorted([(ShapleyValues[x], x) for x in range(len(ShapleyValues))])[-1:]]

def maskTuples(tuples, attributes, unWrapFunction):
    return [[tpl[x] if x not in (unWrapFunction(attributes) if unWrapFunction is not None else attributes) else None for x in range(len(tpl)) ] for tpl in tuples]

def inTopK(t, tuples, functions, k, d, unWrapFunction):
    for f in range(len(functions)):
        evaluatedTuples = topk.generateTuples(tuples, functions[f], [x for x in range(d)], d, unWrapFunction)
        if t in topk.computeTopK(evaluatedTuples, k):
            return f
    return False

def borderLineTopK(t, tuples, functions, k, d, unWrapFunction):
    for f in range(len(functions)):
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

def findTInTopKs(tuples, functions, minim, maxim, k, d, unWrapFunction):
    while True:
        for t in range(len(tuples)):
            if tInXTopKs(tuples, t, functions, k, minim, maxim, d, unWrapFunction):
                return t
        maxim = maxim + 1

def findInTopK(tuples, functions, k, d, unWrapFunction):
    for t in range(len(tuples)):
        topK = inTopK(t, tuples, functions, k, d, unWrapFunction)
        if topK is not False:
            return t, topK

def findBorderLineTopK(tuples, functions, k, d, unWrapFunction):
    for t in range(len(tuples)):
        borderline = borderLineTopK(t, tuples, functions, k, d, unWrapFunction)
        if borderline is not False:
            return t, borderline

def findQueryPointIndependent(tuples, k, functions, d, unWrapFunction, minim, maxim):
    return (findInTopK(tuples, functions, k, d, unWrapFunction), findBorderLineTopK(tuples, functions, k, d, unWrapFunction),
            findTInTopKs(tuples, functions, minim, maxim, k, d, unWrapFunction))


def removeAttributesExperiment(datasets, trialResults, k, unWrapFunction):
    inTopKScore = 0
    notInTopKScore = 0
    whyThisTopKScore = 0
    whyInTheseTopKsScore = 0
    apprxInTopKScore = 0
    apprxNotInTopKScore = 0
    apprxWhyThisTopKScore = 0
    apprxWhyInTheseTopKsScore = 0

    for x in range(len(datasets)):
        dataset = datasets[x]
        trialResult = trialResults[x]

        t, topkFunc, borderlineFunc = findQueryPoint(dataset['Tuples'], k, dataset['Functions'], 6, None, 3, 6)

        topKFuncEvaluatedTuples = topk.generateTuples(dataset['Tuples'], dataset['Functions'][topkFunc], [x for x in range(len(dataset['Tuples'][0]))], len(dataset['Tuples'][0]), unWrapFunction)
        topK = topk.computeTopK(topKFuncEvaluatedTuples, k)

        theseTopKs = set()
        for f in range(len(dataset['Functions'])):
            function = dataset['Functions'][f]
            evaluatedTuples = topk.generateTuples(dataset['Tuples'], function,
                                                  [x for x in range(len(dataset['Tuples'][0]))],
                                                  len(dataset['Tuples'][0]), unWrapFunction)
            tempTopK = topk.computeTopK(evaluatedTuples, k)
            if t in tempTopK:
                theseTopKs.add(f)

        computeFailure = all(
            [abs(ShapleyValue - 1 / len(trialResult['InTopK']['BruteForce']['ShapleyValues'])) < .05 for ShapleyValue
             in trialResult['InTopK']['BruteForce']['ShapleyValues']])

        if computeFailure:
            attributes = range(len(dataset['Tuples'][0]))
            powerSet = chain.from_iterable(combinations(attributes, r) for r in range(len(attributes) + 1))
            for s in powerSet:
                if len(s) == len(attributes):
                    continue
                inTopKTuples = maskTuples(dataset['Tuples'], s, unWrapFunction)
                evaluatedTuples = topk.generateTuples(inTopKTuples, dataset['Functions'][topkFunc],
                                                      [x for x in range(len(dataset['Tuples'][0]))],
                                                      len(dataset['Tuples'][0]), unWrapFunction)
                newTopk = topk.computeTopK(evaluatedTuples, k)
                if t in newTopk:
                    inTopKScore = inTopKScore + 1 / (len(datasets) * 2 ** len(dataset['Tuples'][0]))
        else:
            inTopKTuples = maskTuples(dataset['Tuples'],
                                      computeMaxShapleyValues(trialResult['InTopK']['BruteForce']['ShapleyValues']),
                                      unWrapFunction)
            evaluatedTuples = topk.generateTuples(inTopKTuples, dataset['Functions'][topkFunc], [x for x in range(len(dataset['Tuples'][0]))], len(dataset['Tuples'][0]), unWrapFunction)
            newTopk = topk.computeTopK(evaluatedTuples, k)
            if t not in newTopk:
                inTopKScore = inTopKScore + 1/len(datasets)


        computeFailure = all(
            [abs(ShapleyValue - 1 / len(trialResult['InTopK']['Approximate']['ShapleyValues'])) < .05 for ShapleyValue
             in trialResult['InTopK']['Approximate']['ShapleyValues']])

        if computeFailure:
            attributes = range(len(dataset['Tuples'][0]))
            powerSet = chain.from_iterable(combinations(attributes, r) for r in range(len(attributes) + 1))
            for s in powerSet:
                if len(s) == len(attributes):
                    continue
                inTopKTuples = maskTuples(dataset['Tuples'], s, unWrapFunction)
                evaluatedTuples = topk.generateTuples(inTopKTuples, dataset['Functions'][topkFunc],
                                                      [x for x in range(len(dataset['Tuples'][0]))],
                                                      len(dataset['Tuples'][0]))
                newTopk = topk.computeTopK(evaluatedTuples, k)
                if t in newTopk:
                    apprxInTopKScore = apprxInTopKScore + 1 / (len(datasets) * 2 ** len(dataset['Tuples'][0]))
        else:
            apprxInTopKTuples = maskTuples(dataset['Tuples'], computeMaxShapleyValues(
                trialResult['InTopK']['Approximate']['ShapleyValues']), unWrapFunction)
            evaluatedTuples = topk.generateTuples(apprxInTopKTuples, dataset['Functions'][topkFunc], [x for x in range(len(dataset['Tuples'][0]))], len(dataset['Tuples'][0]), unWrapFunction)
            newTopk = topk.computeTopK(evaluatedTuples, k)
            if t not in newTopk:
                apprxInTopKScore = apprxInTopKScore + 1/len(datasets)


        computeFailure = all([abs(ShapleyValue - 1/len(trialResult['NotInTopK']['BruteForce']['ShapleyValues'])) < .05 for ShapleyValue in trialResult['NotInTopK']['BruteForce']['ShapleyValues']])

        if computeFailure:
            attributes = range(len(dataset['Tuples'][0]))
            powerSet = chain.from_iterable(combinations(attributes, r) for r in range(len(attributes) + 1))
            for s in powerSet:
                if len(s) == len(attributes):
                    continue
                notInTopKTuples = maskTuples(dataset['Tuples'], s, unWrapFunction)
                evaluatedTuples = topk.generateTuples(notInTopKTuples, dataset['Functions'][borderlineFunc],
                                                      [x for x in range(len(dataset['Tuples'][0]))],
                                                      len(dataset['Tuples'][0]), unWrapFunction)
                newTopk = topk.computeTopK(evaluatedTuples, k)
                if t not in newTopk:
                    notInTopKScore = notInTopKScore + 1 / (len(datasets) * 2 ** len(dataset['Tuples'][0]))
        else:
            notInTopKTuples = maskTuples(dataset['Tuples'], computeMaxShapleyValues(
                trialResult['NotInTopK']['BruteForce']['ShapleyValues']), unWrapFunction)
            evaluatedTuples = topk.generateTuples(notInTopKTuples, dataset['Functions'][borderlineFunc],
                                                  [x for x in range(len(dataset['Tuples'][0]))],
                                                  len(dataset['Tuples'][0]), unWrapFunction)
            newTopk = topk.computeTopK(evaluatedTuples, k)
            if t in newTopk:
                notInTopKScore = notInTopKScore + 1 / len(datasets)

        computeFailure = all([abs(ShapleyValue - 1/len(trialResult['NotInTopK']['Approximate']['ShapleyValues'])) < .05 for ShapleyValue in trialResult['NotInTopK']['Approximate']['ShapleyValues']])

        if computeFailure:
            attributes = range(len(dataset['Tuples'][0]))
            powerSet = chain.from_iterable(combinations(attributes, r) for r in range(len(attributes) + 1))
            for s in powerSet:
                if len(s) == len(attributes):
                    continue
                apprxNotInTopKTuples = maskTuples(dataset['Tuples'], s, unWrapFunction)
                evaluatedTuples = topk.generateTuples(apprxNotInTopKTuples, dataset['Functions'][borderlineFunc],
                                                      [x for x in range(len(dataset['Tuples'][0]))],
                                                      len(dataset['Tuples'][0]), unWrapFunction)
                newTopk = topk.computeTopK(evaluatedTuples, k)
                if t not in newTopk:
                    apprxNotInTopKScore = apprxNotInTopKScore + 1 / (len(datasets) * 2 ** len(dataset['Tuples'][0]))

        else:
            apprxNotInTopKTuples = maskTuples(dataset['Tuples'], computeMaxShapleyValues(
                trialResult['NotInTopK']['Approximate']['ShapleyValues']), unWrapFunction)
            evaluatedTuples = topk.generateTuples(apprxNotInTopKTuples, dataset['Functions'][borderlineFunc], [x for x in range(len(dataset['Tuples'][0]))], len(dataset['Tuples'][0]), unWrapFunction)
            newTopk = topk.computeTopK(evaluatedTuples, k)
            if t in newTopk:
                apprxNotInTopKScore = apprxNotInTopKScore + 1/len(datasets)




        whyThisTopKTuples = maskTuples(dataset['Tuples'], computeMaxShapleyValues(
            trialResult['WhyThisTopK']['BruteForce']['ShapleyValues']), unWrapFunction)
        evaluatedTuples = topk.generateTuples(whyThisTopKTuples, dataset['Functions'][t], [x for x in range(len(dataset['Tuples'][0]))], len(dataset['Tuples'][0]), unWrapFunction)
        newTopk = topk.computeTopK(evaluatedTuples, k)
        whyThisTopKScore = whyThisTopKScore + (1 - len((set(newTopk).intersection(set(topK))))/len(set(newTopk).union(set(topK))))/len(datasets)

        apprxWhyThisTopKTuples = maskTuples(dataset['Tuples'], computeMaxShapleyValues(
            trialResult['WhyThisTopK']['Approximate']['ShapleyValues']), unWrapFunction)
        evaluatedTuples = topk.generateTuples(apprxWhyThisTopKTuples, dataset['Functions'][t], [x for x in range(len(dataset['Tuples'][0]))], len(dataset['Tuples'][0]), None)
        newTopk = topk.computeTopK(evaluatedTuples, k)
        apprxWhyThisTopKScore = apprxWhyThisTopKScore + (1 - len((set(newTopk).intersection(set(topK))))/len(set(newTopk).union(set(topK))))/len(datasets)


        whyTheseTopKsTuples = maskTuples(dataset['Tuples'], computeMaxShapleyValues(
            trialResult['WhyInTheseTopKs']['BruteForce']['ShapleyValues']), unWrapFunction)
        newTheseTopKs = set()
        for f in range(len(dataset['Functions'])):
            function = dataset['Functions'][f]
            evaluatedTuples = topk.generateTuples(whyTheseTopKsTuples, function,
                                                  [x for x in range(len(dataset['Tuples'][0]))],
                                                  len(dataset['Tuples'][0]), None)
            tempTopK = topk.computeTopK(evaluatedTuples, k)
            if t in tempTopK:
                newTheseTopKs.add(f)

        whyInTheseTopKsScore = whyInTheseTopKsScore + (1 - len((newTheseTopKs.intersection(theseTopKs)))/len(newTheseTopKs.union(theseTopKs)))/len(datasets)


        apprxWhyTheseTopKsTuples = maskTuples(dataset['Tuples'], computeMaxShapleyValues(
            trialResult['WhyInTheseTopKs']['Approximate']['ShapleyValues']), unWrapFunction)
        newTheseTopKs = set()
        for f in range(len(dataset['Functions'])):
            function = dataset['Functions'][f]
            evaluatedTuples = topk.generateTuples(apprxWhyTheseTopKsTuples, function,
                                                  [x for x in range(len(dataset['Tuples'][0]))],
                                                  len(dataset['Tuples'][0]), None)
            tempTopK = topk.computeTopK(evaluatedTuples, k)
            if t in tempTopK:
                newTheseTopKs.add(f)


        apprxWhyInTheseTopKsScore = apprxWhyInTheseTopKsScore + (1 - len((newTheseTopKs.intersection(theseTopKs)))/len(newTheseTopKs.union(theseTopKs)))/len(datasets)

    return [('Why In Top K Score: ', ('Brute Force', inTopKScore), ('Approximate', apprxInTopKScore)),
               ('Why Not In Top K Score: ', ('Brute Force', notInTopKScore), ('Approximate', apprxNotInTopKScore)),
               ('Why This Top K Score', ('Brute Force', whyThisTopKScore), ('Approximate', apprxWhyThisTopKScore)),
               ('Why In These Top Ks Score', ('Brute Force', whyInTheseTopKsScore), ('Approximate', apprxWhyInTheseTopKsScore))]
     #
    # results = {}
    # prev = dill.load(open('ExperimentMResults8.dill', 'rb'))
    #
    # inTopK = [y[1] for y in sorted([(results['InTopK']['BruteForce']['ShapleyValues'][x], x) for x in results['InTopK']['BruteForce']['ShapleyValues']])[-2:]]
    # newEvaluatedTuples =
    #
    # results['NotInTopK'] = notInTopKResults
    # results['WhyThisTopK'] = whyThisTopKResults
    # results['WhyInTheseTopKs'] = whyInTheseTopKResults

def datasetExperiment(dataset, d, unWrapFunction, k):

    results = {}

    t, topkFunc, borderlineFunc = findQueryPoint(dataset['Tuples'], k, dataset['Functions'], 6, None, 3, 6)

    inTopKResults = {}
    notInTopKResults = {}
    whyThisTopKResults = {}
    whyInTheseTopKResults = {}

    inTopKResults['BruteForce'] = bruteForceInTopK(dataset['Tuples'], dataset['Functions'][topkFunc], k, t, d, unWrapFunction)
    notInTopKResults['BruteForce'] = bruteForceNotInTopK(dataset['Tuples'], dataset['Functions'][borderlineFunc], k, t, d, unWrapFunction)
    whyThisTopKResults['BruteForce'] = bruteForceWhyThisTopK(dataset['Tuples'], dataset['Functions'][t], k, d, unWrapFunction)
    whyInTheseTopKResults['BruteForce'] = bruteForceWhyInTheseTopK(dataset['Tuples'], dataset['Functions'], k, t, d, unWrapFunction)
    inTopKResults['Approximate'] = approximateInTopK(dataset['Tuples'], dataset['Functions'][topkFunc], 200, k, t, d, inTopKResults['BruteForce']['ShapleyValues'] if type(inTopKResults['BruteForce']) is dict else [0.0 for x in range(len(dataset['Tuples'][0]))], unWrapFunction)
    notInTopKResults['Approximate'] = approximateNotInTopK(dataset['Tuples'], dataset['Functions'][borderlineFunc], 200, k, t, d, notInTopKResults['BruteForce']['ShapleyValues'] if type(notInTopKResults['BruteForce']) is dict else [0.0 for x in range(len(dataset['Tuples'][0]))], unWrapFunction)
    whyThisTopKResults['Approximate'] = approximateWhyThisTopK(dataset['Tuples'], dataset['Functions'][t], 200, k, d, whyThisTopKResults['BruteForce']['ShapleyValues'] if type(whyThisTopKResults['BruteForce']) is dict else [0.0 for x in range(len(dataset['Tuples'][0]))], unWrapFunction)
    whyInTheseTopKResults['Approximate'] = approximateWhyInTheseTopK(dataset['Tuples'], dataset['Functions'], 200, k, t, d, whyInTheseTopKResults['BruteForce']['ShapleyValues'] if type(whyInTheseTopKResults['BruteForce']) is dict else [0.0 for x in range(len(dataset['Tuples'][0]))], unWrapFunction)

    results['Query Point'] = (t, topkFunc, borderlineFunc)
    results['InTopK'] = inTopKResults
    results['NotInTopK'] = notInTopKResults
    results['WhyThisTopK'] = whyThisTopKResults
    results['WhyInTheseTopKs'] = whyInTheseTopKResults

    return results

def UnwrapCandidate(attributes):
    unwrapped = [[0],[1],[2],[3,4,10,11],[5,6,7,8,9,12],[13,14,15,16],[17],[18,19,20],[21]]

    res = []
    for a in attributes:
        res.extend(unwrapped[a])

    return res

def CandidatesExperiment():
    datasets = dill.load(open('Candidates-Dataset.dill', 'rb'))
    functions = dill.load(open('Candidates-Functions.dill', 'rb'))
    dill.dump(varyingMExperiment(datasets['Candidates'], functions['HRs'], datasets['HRs'], functions['Candidates'], 9,
                                 UnwrapCandidate, 3, 6, 5), open('VaryingMCandidates.dill', 'wb'))

def fullAttributesCandidates():
    results = {}

    datasets = dill.load(open('Candidates-Dataset.dill', 'rb'))
    functions = dill.load(open('Candidates-Functions.dill', 'rb'))

    t, topkFunc, borderlineFunc = findQueryPoint(datasets['Candidates'], 5, functions['HRs'], 22, None, 3, 6)

    inTopKResults = {}
    notInTopKResults = {}
    whyThisTopKResults = {}
    whyInTheseTopKResults = {}

    inTopKResults['Approximate'] = approximateInTopK(datasets['Candidates'], functions['HRs'][topkFunc], 200, 5, t, 22, [0.0 for x in range(22)], None)
    notInTopKResults['Approximate'] = approximateNotInTopK(datasets['Candidates'], functions['HRs'][topkFunc], 200, 5, t, 22, [0.0 for x in range(22)], None)
    whyThisTopKResults['Approximate'] = approximateWhyThisTopK(datasets['HRs'], functions['Candidates'][t], 200, 5, 22, [0.0 for x in range(22)], None)
    whyInTheseTopKResults['Approximate'] = approximateWhyInTheseTopK(datasets['Candidates'], functions['HRs'], 200, 5, t, 22, [0.0 for x in range(22)], None)

    results['Query Point'] = (t, topkFunc, borderlineFunc)
    results['InTopK'] = inTopKResults
    results['NotInTopK'] = notInTopKResults
    results['WhyThisTopK'] = whyThisTopKResults
    results['WhyInTheseTopKs'] = whyInTheseTopKResults

    dill.dump(results, open('Candidates22ApproximateReslts.dill', 'wb'))


def SyntheticExperiment():
    # datasets = dill.load(open('data/a_z_l_varying_d.dill', 'rb'))
    # dill.dump(varyingMExperiment(datasets[9][0], datasets[9][1], datasets[9][2], datasets[9][3], 9, None, 3, 6, 5), open('SyntheticMAZL.dill'))
    # datasets = dill.load(open('data/c_z_l_varying_d.dill', 'rb'))
    # dill.dump(varyingMExperiment(datasets[9][0], data sets[9][1], datasets[9][2], datasets[9][3], 9, None, 3, 6, 5), open('SyntheticMCZL.dill'))
    # datasets = dill.load(open('data/i_z_l_varying_d.dill', 'rb'))
    # dill.dump(varyingMExperiment(datasets[9][0], datasets[9][1], datasets[9][2], datasets[9][3], 9, None, 3, 6, 5), open('SyntheticMIZL.dill'))
    # datasets = dill.load(open('data/a_z_nl_varying_d.dill', 'rb'))
    # dill.dump(varyingMExperiment(datasets[9][0], datasets[9][1], datasets[9][2], datasets[9][3], 9, None, 3, 6, 5), open('SyntheticMAZNL.dill'))
    # datasets = dill.load(open('data/c_z_nl_varying_d.dill', 'rb'))
    # dill.dump(varyingMExperiment(datasets[9][0], datasets[9][1], datasets[9][2], datasets[9][3], 9, None, 3, 6, 5), open('SyntheticMCZNL.dill'))
    # datasets = dill.load(open('data/i_z_nl_varying_d.dill', 'rb'))
    # dill.dump(varyingMExperiment(datasets[9][0], datasets[9][1], datasets[9][2], datasets[9][3], 9, None, 3, 6, 5), open('SyntheticMIZNL.dill'))

    # dill.dump(varyingDExperiment(dill.load(open('data/a_z_l_varying_d.dill', 'rb')), None, 3, 6, 5), open('SyntheticDAZL.dill', 'wb'))
    # dill.dump(varyingDExperiment(dill.load(open('data/c_z_l_varying_d.dill', 'rb')), None, 3, 6, 5), open('SyntheticDCZL.dill', 'wb'))
    # dill.dump(varyingDExperiment(dill.load(open('data/i_z_l_varying_d.dill', 'rb')), None, 3, 6, 5), open('SyntheticDIZL.dill', 'wb'))
    dill.dump(varyingDExperiment(dill.load(open('data/a_z_nl_varying_d.dill', 'rb')), None, 3, 6, 5), open('SyntheticDAZNL.dill', 'wb'))
    dill.dump(varyingDExperiment(dill.load(open('data/c_z_nl_varying_d.dill', 'rb')), None, 3, 6, 5), open('SyntheticDCZNL.dill', 'wb'))
    dill.dump(varyingDExperiment(dill.load(open('data/i_z_nl_varying_d.dill', 'rb')), None, 3, 6, 5), open('SyntheticDIZNL.dill', 'wb'))

def RunningExampleExperiment():
    dataset = pickle.load(open('Running-Example.pickle', 'rb'))
    functions = dill.load(open('Running-Example-Functions2.dill', 'rb'))

    pickle.dump(
        varyingMExperiment(dataset['Candidates'], functions['HRs'], dataset['HRs'], functions['Candidates'], 4, None, 3,
                           3, 2),
        open('Running-Example-Results2.pickle', 'wb'))

def generateMLData():
    a = dill.load(open('data/a_u_300_5_9999.dill', 'rb'))
    c = dill.load(open('data/c_u_300_5_9999.dill', 'rb'))
    i = dill.load(open('data/i_u_300_5_9999.dill', 'rb'))
    for x in range(10):
        functions = pickle.load(open('data/functions-'+str(x)+'.dill', 'rb'))['Functions']
        res = []
        for ds in range(100*x, 100*(x+1)):
            fncs = functions
            print(x*3000+(ds%100)*300)
            print(x*3000+(ds%100+1)*300)
            (t1, topkFunc), (t2, borderlineFunc), t3 = findQueryPointIndependent(i[ds], 5, functions[(ds%100)*300:(ds%100+1)*300], 5, None, 3, 10)
            results = {}
            results['InTopK'] = bruteForceInTopK(a[ds], functions[(ds%100)*300:(ds%100+1)*300][topkFunc], 5, t1, 5, None)
            results['NotInTopK'] = bruteForceNotInTopK(a[ds], functions[(ds%100)*300:(ds%100+1)*300][borderlineFunc], 5, t2, 5, None)
            results['WhyThisTopK'] = bruteForceWhyThisTopK(a[ds], functions[(ds%100)*300:(ds%100+1)*300][0], 3, 5, None)
            results['WhyInTheseTopKs'] = bruteForceWhyInTheseTopK(a[ds], functions[(ds%100)*300:(ds%100+1)*300], 3, t3, 5, None)
            res.append(results)
        dill.dump(res, open('data/ml2-'+str(x)+'.dill', 'wb'))
    for x in range(10, 20):
        functions = pickle.load(open('data/functions-' + str(x) + '.dill', 'rb'))['Functions']
        res = []
        for ds in range(100 * (x-10), 100 * (x -9)):
            print(x*3000+(ds%100)*300)
            print(x*3000+(ds%100+1)*300)
            (t1, topkFunc), (t2, borderlineFunc), t3 = findQueryPointIndependent(i[ds], 5, functions[(ds%100)*300:(ds%100+1)*300], 5, None, 3, 10)
            results = {}
            results['InTopK'] = bruteForceInTopK(c[ds], functions[(ds%100)*300:(ds%100+1)*300][topkFunc], 5, t1, 5, None)
            results['NotInTopK'] = bruteForceNotInTopK(c[ds], functions[(ds%100)*300:(ds%100+1)*300][borderlineFunc], 5, t2, 5, None)
            results['WhyThisTopK'] = bruteForceWhyThisTopK(c[ds], functions[(ds%100)*300:(ds%100+1)*300][0], 3, 5, None)
            results['WhyInTheseTopKs'] = bruteForceWhyInTheseTopK(c[ds], functions[(ds%100)*300:(ds%100+1)*300], 3, t3, 5, None)
            res.append(results)
        dill.dump(res, open('data/ml2-' + str(x) + '.dill', 'wb'))
    for x in range(20, 30):
        functions = pickle.load(open('data/functions-' + str(x) + '.dill', 'rb'))['Functions']
        res = []
        for ds in range(100 * (x - 20), 100 * (x - 19)):
            print(x*3000+(ds%100)*300)
            print(x*3000+(ds%100+1)*300)
            (t1, topkFunc), (t2, borderlineFunc), t3 = findQueryPointIndependent(i[ds], 5, functions[(ds%100)*300:(ds%100+1)*300], 5, None, 3, 10)
            results = {}
            results['InTopK'] = bruteForceInTopK(i[ds], functions[(ds%100)*300:(ds%100+1)*300][topkFunc], 5, t1, 5, None)
            results['NotInTopK'] = bruteForceNotInTopK(i[ds], functions[(ds%100)*300:(ds%100+1)*300][borderlineFunc], 5, t2, 5, None)
            results['WhyThisTopK'] = bruteForceWhyThisTopK(i[ds], functions[(ds%100)*300:(ds%100+1)*300][0], 3, 5, None)
            results['WhyInTheseTopKs'] = bruteForceWhyInTheseTopK(i[ds], functions[(ds%100)*300:(ds%100+1)*300], 3, t3, 5, None)
            res.append(results)
        dill.dump(res, open('data/ml2-' + str(x) + '.dill', 'wb'))


def main():
    #CandidatesExperiment()
    #SyntheticExperiment()
    #RunningExampleExperiment()
    generateMLData()
    # fullAttributesCandidates()

    #datasets = dill.load(open('1000x100-5-samples', 'rb'))
    #results = []
    #for dataset in datasets:
    #   results.append(datasetExperiment(dataset))
    #dill.dump(results, open('MultipleSamplesExperiment200', 'wb'))

    # res = []
    # dataset = {}
    # functions = dill.load(open('Removing-Functions-Linear.dill', 'rb'))['Functions']
    # for tuples in dill.load(open('data/a_u_100_6_9.dill', 'rb')):
    #     dataset['Tuples'] = tuples
    #     dataset['Functions'] = functions[:100]
    #     functions = functions[100:]
    #     res.append(datasetExperiment(dataset, 6, None, 5))
    # dill.dump(res, open('data/remove_a_u_l.dill', 'wb'))
    # res = []
    # for tuples in dill.load(open('data/i_u_100_6_9.dill', 'rb')):
    #     dataset['Tuples'] = tuples
    #     dataset['Functions'] = functions[:100]
    #     functions = functions[100:]
    #     res.append(datasetExperiment(dataset, 6, None, 5))
    # dill.dump(res, open('data/remove_i_u_l.dill', 'wb'))
    # res = []
    # for tuples in dill.load(open('data/c_u_100_6_9.dill', 'rb')):
    #     dataset['Tuples'] = tuples
    #     dataset['Functions'] = functions[:100]
    #     functions = functions[100:]
    #     res.append(datasetExperiment(dataset, 6, None, 5))
    # dill.dump(res, open('data/remove_c_u_l.dill', 'wb'))
    # res = []
    # for tuples in dill.load(open('data/a_z_100_6_9.dill', 'rb')):
    #     dataset['Tuples'] = tuples
    #     dataset['Functions'] = functions[:100]
    #     functions = functions[100:]
    #     res.append(datasetExperiment(dataset, 6, None, 5))
    # dill.dump(res, open('data/remove_a_z_l.dill', 'wb'))
    # res = []
    # for tuples in dill.load(open('data/i_z_100_6_9.dill', 'rb')):
    #     dataset['Tuples'] = tuples
    #     dataset['Functions'] = functions[:100]
    #     functions = functions[100:]
    #     res.append(datasetExperiment(dataset, 6, None, 5))
    # dill.dump(res, open('data/remove_i_z_l.dill', 'wb'))
    # res = []
    # for tuples in dill.load(open('data/c_z_100_6_9.dill', 'rb')):
    #     dataset['Tuples'] = tuples
    #     dataset['Functions'] = functions[:100]
    #     functions = functions[100:]
    #     res.append(datasetExperiment(dataset, 6, None, 5))
    # dill.dump(res, open('data/remove_c_z_l.dill', 'wb'))
    # res = []
    # functions = dill.load(open('Removing-Functions-Nonlinear.dill', 'rb'))['Functions']
    # for tuples in dill.load(open('data/a_u_100_6_9.dill', 'rb')):
    #     dataset['Tuples'] = tuples
    #     dataset['Functions'] = functions[:100]
    #     functions = functions[100:]
    #     res.append(datasetExperiment(dataset, 6, None, 5))
    # dill.dump(res, open('data/remove_a_u_nl.dill', 'wb'))
    # res = []
    # for tuples in dill.load(open('data/i_u_100_6_9.dill', 'rb')):
    #     dataset['Tuples'] = tuples
    #     dataset['Functions'] = functions[:100]
    #     functions = functions[100:]
    #     res.append(datasetExperiment(dataset, 6, None, 5))
    # dill.dump(res, open('data/remove_i_u_nl.dill', 'wb'))
    # res = []
    # for tuples in dill.load(open('data/c_u_100_6_9.dill', 'rb')):
    #     dataset['Tuples'] = tuples
    #     dataset['Functions'] = functions[:100]
    #     functions = functions[100:]
    #     res.append(datasetExperiment(dataset, 6, None, 5))
    # dill.dump(res, open('data/remove_c_u_nl.dill', 'wb'))
    # res = []
    # for tuples in dill.load(open('data/a_z_100_6_9.dill', 'rb')):
    #     dataset['Tuples'] = tuples
    #     dataset['Functions'] = functions[:100]
    #     functions = functions[100:]
    #     res.append(datasetExperiment(dataset, 6, None, 5))
    # dill.dump(res, open('data/remove_a_z_nl.dill', 'wb'))
    # res = []
    # for tuples in dill.load(open('data/i_z_100_6_9.dill', 'rb')):
    #     dataset['Tuples'] = tuples
    #     dataset['Functions'] = functions[:100]
    #     functions = functions[100:]
    #     res.append(datasetExperiment(dataset, 6, None, 5))
    # dill.dump(res, open('data/remove_i_z_nl.dill', 'wb'))
    # res = []
    # for tuples in dill.load(open('data/c_z_100_6_9.dill', 'rb')):
    #     dataset['Tuples'] = tuples
    #     dataset['Functions'] = functions[:100]
    #     functions = functions[100:]
    #     res.append(datasetExperiment(dataset, 6, None, 5))
    # dill.dump(res, open('data/remove_c_z_nl.dill', 'wb'))

    # functions = dill.load(open('Removing-Functions-Linear.dill', 'rb'))['Functions']
    # datasets = []
    # for tuples in dill.load(open('data/a_u_100_6_9.dill', 'rb')):
    #     dataset = {}
    #     dataset['Tuples'] = tuples
    #     dataset['Functions'] = functions[:100]
    #     functions = functions[100:]
    #     datasets.append(dataset)
    # dill.dump(removeAttributesExperiment(datasets, dill.load(open('data/remove_a_u_l.dill', 'rb')), 5, None), open('data/remove_results_a_u_l.dill', 'wb'))
    # datasets = []
    # for tuples in dill.load(open('data/i_u_100_6_9.dill', 'rb')):
    #     dataset = {}
    #     dataset['Tuples'] = tuples
    #     dataset['Functions'] = functions[:100]
    #     functions = functions[100:]
    #     datasets.append(dataset)
    # dill.dump(removeAttributesExperiment(datasets, dill.load(open('data/remove_i_u_l.dill', 'rb')), 5, None), open('data/remove_results_i_u_l.dill', 'wb'))
    # datasets = []
    # for tuples in dill.load(open('data/c_u_100_6_9.dill', 'rb')):
    #     dataset = {}
    #     dataset['Tuples'] = tuples
    #     dataset['Functions'] = functions[:100]
    #     functions = functions[100:]
    #     datasets.append(dataset)
    # dill.dump(removeAttributesExperiment(datasets, dill.load(open('data/remove_c_u_l.dill', 'rb')), 5, None), open('data/remove_results_c_u_l.dill', 'wb'))
    # datasets = []
    # for tuples in dill.load(open('data/a_z_100_6_9.dill', 'rb')):
    #     dataset = {}
    #     dataset['Tuples'] = tuples
    #     dataset['Functions'] = functions[:100]
    #     functions = functions[100:]
    #     datasets.append(dataset)
    # dill.dump(removeAttributesExperiment(datasets, dill.load(open('data/remove_a_z_l.dill', 'rb')), 5, None), open('data/remove_results_a_z_l.dill', 'wb'))
    # datasets = []
    # for tuples in dill.load(open('data/i_z_100_6_9.dill', 'rb')):
    #     dataset = {}
    #     dataset['Tuples'] = tuples
    #     dataset['Functions'] = functions[:100]
    #     functions = functions[100:]
    #     datasets.append(dataset)
    # dill.dump(removeAttributesExperiment(datasets, dill.load(open('data/remove_i_z_l.dill', 'rb')), 5, None), open('data/remove_results_i_z_l.dill', 'wb'))
    # datasets = []
    # for tuples in dill.load(open('data/c_z_100_6_9.dill', 'rb')):
    #     dataset = {}
    #     dataset['Tuples'] = tuples
    #     dataset['Functions'] = functions[:100]
    #     functions = functions[100:]
    #     datasets.append(dataset)
    # dill.dump(removeAttributesExperiment(datasets, dill.load(open('data/remove_c_z_l.dill', 'rb')), 5, None), open('data/remove_results_c_z_l.dill', 'wb'))
    # datasets = []
    # functions = dill.load(open('Removing-Functions-Nonlinear.dill', 'rb'))['Functions']
    # for tuples in dill.load(open('data/a_u_100_6_9.dill', 'rb')):
    #     dataset = {}
    #     dataset['Tuples'] = tuples
    #     dataset['Functions'] = functions[:100]
    #     functions = functions[100:]
    #     datasets.append(dataset)
    # dill.dump(removeAttributesExperiment(datasets, dill.load(open('data/remove_a_u_nl.dill', 'rb')), 5, None), open('data/remove_results_a_u_nl.dill', 'wb'))
    # datasets = []
    # for tuples in dill.load(open('data/i_u_100_6_9.dill', 'rb')):
    #     dataset = {}
    #     dataset['Tuples'] = tuples
    #     dataset['Functions'] = functions[:100]
    #     functions = functions[100:]
    #     datasets.append(dataset)
    # dill.dump(removeAttributesExperiment(datasets, dill.load(open('data/remove_i_u_nl.dill', 'rb')), 5, None), open('data/remove_results_i_u_nl.dill', 'wb'))
    # datasets = []
    # for tuples in dill.load(open('data/c_u_100_6_9.dill', 'rb')):
    #     dataset = {}
    #     dataset['Tuples'] = tuples
    #     dataset['Functions'] = functions[:100]
    #     functions = functions[100:]
    #     datasets.append(dataset)
    # dill.dump(removeAttributesExperiment(datasets, dill.load(open('data/remove_c_u_nl.dill', 'rb')), 5, None), open('data/remove_results_c_u_nl.dill', 'wb'))
    # datasets = []
    # for tuples in dill.load(open('data/a_z_100_6_9.dill', 'rb')):
    #     dataset = {}
    #     dataset['Tuples'] = tuples
    #     dataset['Functions'] = functions[:100]
    #     functions = functions[100:]
    #     datasets.append(dataset)
    # dill.dump(removeAttributesExperiment(datasets, dill.load(open('data/remove_a_z_nl.dill', 'rb')), 5, None), open('data/remove_results_a_z_nl.dill', 'wb'))
    # datasets = []
    # for tuples in dill.load(open('data/i_z_100_6_9.dill', 'rb')):
    #     dataset = {}
    #     dataset['Tuples'] = tuples
    #     dataset['Functions'] = functions[:100]
    #     functions = functions[100:]
    #     datasets.append(dataset)
    # dill.dump(removeAttributesExperiment(datasets, dill.load(open('data/remove_i_z_nl.dill', 'rb')), 5, None), open('data/remove_results_i_z_nl.dill', 'wb'))
    # datasets = []
    # for tuples in dill.load(open('data/c_z_100_6_9.dill', 'rb')):
    #     dataset = {}
    #     dataset['Tuples'] = tuples
    #     dataset['Functions'] = functions[:100]
    #     functions = functions[100:]
    #     datasets.append(dataset)
    # dill.dump(removeAttributesExperiment(datasets, dill.load(open('data/remove_c_z_nl.dill', 'rb')), 5, None), open('data/remove_results_c_z_nl.dill', 'wb'))

    #removeAttributesExperiment()
    #newVaryingD()
    #newVaryingM()
    #
    # res = varyingMExperiment()
    # dill.dump(res, open('ExperimentM-NLResult.dill', 'wb'))
    #res = varyingDExperiment()
    #dill.dump(res, open('ExperimentD-NLResult.dill', 'wb'))


if __name__ == "__main__":
    main()

