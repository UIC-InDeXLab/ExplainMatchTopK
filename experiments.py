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


def bruteForceInTopK(tuples, evalFunc, k, j):
    results = {}
    
    start_time = time.process_time_ns()
    shapley = bruteforce.ComputeShapleyInTopK(tuples, evalFunc, k, j)
    runtime = time.process_time_ns() - start_time

    results['RuntimeNS'] = runtime
    results['ShapleyValues'] = shapley

    return results

def bruteForceNotInTopK(tuples, evalFunc, k, j):
    results = {}
    
    start_time = time.process_time_ns()
    shapley = bruteforce.ComputeShapleyNotInTopK(tuples, evalFunc, k, j)
    runtime = time.process_time_ns() - start_time

    results['RuntimeNS'] = runtime
    results['ShapleyValues'] = shapley

    return results

def bruteForceWhyThisTopK(tuples, evalFunc, k):
    results = {}
    
    start_time = time.process_time_ns()
    shapley = bruteforce.ComputeShapleyTopKLookLikeThis(tuples, evalFunc, k)
    runtime = time.process_time_ns() - start_time

    results['RuntimeNS'] = runtime
    results['ShapleyValues'] = shapley

    return results

def bruteForceWhyInTheseTopK(tuples, evalFuncs, k, j):
    results = {}
    
    start_time = time.process_time_ns()
    shapley = bruteforce.ComputeWhyInTheseTopKs(tuples, evalFuncs, k, j)
    runtime = time.process_time_ns() - start_time

    results['RuntimeNS'] = runtime
    results['ShapleyValues'] = shapley

    return results

def approximateInTopK(tuples, evalFunc, m, k, j, bruteForce):
    results = {}
    
    start_time = time.process_time_ns()
    shapley = approximate.approximateShapleyInTopK(tuples, evalFunc, m, k, j)
    runtime = time.process_time_ns() - start_time

    results['RuntimeNS'] = runtime
    results['ShapleyValues'] = shapley
    X = np.asarray(shapley)
    Y = np.asarray(bruteForce)
    results['AverageDiffernce'] = np.mean(np.abs(X - Y))
    results['StdDifference'] = np.std(np.abs(X-Y))

    return results

def approximateNotInTopK(tuples, evalFunc, m, k, j, bruteForce):
    results = {}
    
    start_time = time.process_time_ns()
    shapley = approximate.approximateShapleyNotInTopK(tuples, evalFunc, m, k, j)
    runtime = time.process_time_ns() - start_time

    results['RuntimeNS'] = runtime
    results['ShapleyValues'] = shapley
    X = np.asarray(shapley)
    Y = np.asarray(bruteForce)
    results['AverageDiffernce'] = np.mean(np.abs(X - Y))
    results['StdDifference'] = np.std(np.abs(X-Y))

    return results

def approximateWhyThisTopK(tuples, evalFunc, m, k, bruteForce):
    results = {}
    
    start_time = time.process_time_ns()
    shapley = approximate.approximateShapleyTopKLookLikeThis(tuples, evalFunc, m, k)
    runtime = time.process_time_ns() - start_time

    results['RuntimeNS'] = runtime
    results['ShapleyValues'] = shapley
    X = np.asarray(shapley)
    Y = np.asarray(bruteForce)
    results['AverageDiffernce'] = np.mean(np.abs(X - Y))
    results['StdDifference'] = np.std(np.abs(X-Y))

    return results

def approximateWhyInTheseTopK(tuples, evalFuncs, m, k, j, bruteForce):
    results = {}
    
    start_time = time.process_time_ns()
    shapley = approximate.approximateWhyInTheseTopKs(tuples, evalFuncs, m, k, j)
    runtime = time.process_time_ns() - start_time

    results['RuntimeNS'] = runtime
    results['ShapleyValues'] = shapley
    X = np.asarray(shapley)
    Y = np.asarray(bruteForce)
    results['AverageDiffernce'] = np.mean(np.abs(X - Y))
    results['StdDifference'] = np.std(np.abs(X-Y))

    return results


def individualInRangeOfTopKs(tuples, functions, minim, maxim, k):
    count = [0 for i in range(len(tuples))]

    for function in functions:
        evaluatedTuples = topk.generateTuples(tuples, function, [x for x in range(len(tuples[0]))], len(tuples[0]))
        topK = topk.computeTopK(evaluatedTuples, k)
        for j in topK:
            count[j] = count[j] + 1

    for c in range(len(count)):
        if count[c] >= minim and count[c] <= maxim:
            return c


def varyingMExperiment():
    dataset = dill.load(open('Varying-D.dill', 'rb'))[8]
    k = 5
    mTested = [25,50,75,100,125,150,175,200,225,250]

    results = {}

    evaluatedTuples = topk.generateTuples(dataset['Tuples'], dataset['Functions'][0], [x for x in range(len(dataset['Tuples'][0]))], len(dataset['Tuples'][0]))
    topK = topk.computeTopK(evaluatedTuples, k)
    topKPlusOne = topk.computeTopK(evaluatedTuples, k+1)
    inXTopKs = individualInRangeOfTopKs(dataset['Tuples'], dataset['Functions'], 5, 5, k)

    inTopKResults = {}
    notInTopKResults = {}
    whyThisTopKResults = {}
    whyInTheseTopKResults = {}

    inTopKResults['BruteForce'] = bruteForceInTopK(dataset['Tuples'], dataset['Functions'][0], k, topK[k-1])
    notInTopKResults['BruteForce'] = bruteForceNotInTopK(dataset['Tuples'], dataset['Functions'][0], k, topKPlusOne[k])
    whyThisTopKResults['BruteForce'] = bruteForceWhyThisTopK(dataset['Tuples'], dataset['Functions'][0], k)
    whyInTheseTopKResults['BruteForce'] = bruteForceWhyInTheseTopK(dataset['Tuples'], dataset['Functions'], k, inXTopKs)

    inTopKResults['Approximate'] = {}
    notInTopKResults['Approximate'] = {}
    whyThisTopKResults['Approximate'] = {}
    whyInTheseTopKResults['Approximate'] = {}

    for m in mTested:   
        inTopKResults['Approximate'][m] = approximateInTopK(dataset['Tuples'], dataset['Functions'][0], m, k, topK[k-1], inTopKResults['BruteForce']['ShapleyValues'])
        notInTopKResults['Approximate'][m] = approximateNotInTopK(dataset['Tuples'], dataset['Functions'][0], m, k, topKPlusOne[k], notInTopKResults['BruteForce']['ShapleyValues'])
        whyThisTopKResults['Approximate'][m] = approximateWhyThisTopK(dataset['Tuples'], dataset['Functions'][0], m, k, whyThisTopKResults['BruteForce']['ShapleyValues'])
        whyInTheseTopKResults['Approximate'][m] = approximateWhyInTheseTopK(dataset['Tuples'], dataset['Functions'], m, k, inXTopKs,  whyInTheseTopKResults['BruteForce']['ShapleyValues'])

    results['InTopK'] = inTopKResults
    results['NotInTopK'] = notInTopKResults
    results['WhyThisTopK'] = whyThisTopKResults
    results['WhyInTheseTopKs'] = whyInTheseTopKResults

    return results

def varyingDExperiment():
    datasets = dill.load(open('Varying-D.dill', 'rb'))
    k = 5
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
        dataset = datasets[index]
        results = {}
        evaluatedTuples = topk.generateTuples(dataset['Tuples'], dataset['Functions'][0], [x for x in range(len(dataset['Tuples'][0]))], len(dataset['Tuples'][0]))
        topK = topk.computeTopK(evaluatedTuples, k)
        topKPlusOne = topk.computeTopK(evaluatedTuples, k)
        inXTopKs = individualInRangeOfTopKs(dataset['Tuples'], dataset['Functions'], 5, 8, k)

        inTopKResults = {}
        notInTopKResults = {}
        whyThisTopKResults = {}
        whyInTheseTopKResults = {}

        if not skipFutureTopK:
            try:
                inTopKResults['BruteForce'] = executor.submit(bruteForceInTopK, dataset['Tuples'], dataset['Functions'][0], k, topK[k-1]).result(timeout=3600)
            except concurrent.futures.TimeoutError:
                inTopKResults['BruteForce'] = 'Too long!'
                skipFutureTopK = True
        else:
            inTopKResults['BruteForce'] = 'Too long!'
            
        if not skipFutureNotTopK:
            try:
                notInTopKResults['BruteForce'] = executor.submit(bruteForceNotInTopK, dataset['Tuples'], dataset['Functions'][0], k, topKPlusOne[k]).result(timeout=3600)
            except concurrent.futures.TimeoutError:
                notInTopKResults['BruteForce'] = 'Too long!'
                skipFutureNotTopK = True
        else:
            notInTopKResults['BruteForce'] = 'Too long!'
            
        if not skipFutureWhyThisTopK:
            try:
                whyThisTopKResults['BruteForce'] = executor.submit(bruteForceWhyThisTopK, dataset['Tuples'], dataset['Functions'][0], k).result(timeout=3600)
            except concurrent.futures.TimeoutError:
                whyThisTopKResults['BruteForce'] = 'Too long!'
                skipFutureWhyThisTopK = True
        else:
            whyThisTopKResults['BruteForce'] = 'Too long!'
            

        if not skipFutureWhyTheseTopKs:
            try:
                whyInTheseTopKResults['BruteForce'] = executor.submit(bruteForceWhyInTheseTopK, dataset['Tuples'], dataset['Functions'], k, inXTopKs).result(timeout=3600)
            except concurrent.futures.TimeoutError:
                whyInTheseTopKResults['BruteForce'] = 'Too long!'
                skipFutureWhyTheseTopKs = True
        else:
            whyInTheseTopKResults['BruteForce'] = 'Too long!'

        if not apprxSkipFutureTopK:
            try:
                inTopKResults['Approximate'] = executor.submit(approximateInTopK, dataset['Tuples'], dataset['Functions'][0], 100, k, topK[k-1], inTopKResults['BruteForce']['ShapleyValues'] if type(inTopKResults['BruteForce']) is dict else [0.0 for x in range(len(dataset['Tuples'][0]))]).result(timeout=3600)
            except concurrent.futures.TimeoutError:
                inTopKResults['Approximate'] = 'Too long!'
                apprxSkipFutureTopK = True
        else:
            inTopKResults['Approximate'] = 'Too long!'

        if not apprxSkipFutureNotTopK:
            try:
                notInTopKResults['Approximate'] = executor.submit(approximateInTopK, dataset['Tuples'], dataset['Functions'][0], 100, k, topKPlusOne[k], notInTopKResults['BruteForce']['ShapleyValues'] if type(notInTopKResults['BruteForce']) is dict else [0.0 for x in range(len(dataset['Tuples'][0]))]).result(timeout=3600)
            except concurrent.futures.TimeoutError:
                notInTopKResults['Approximate'] = 'Too long!'
                apprxSkipFutureNotTopK = True
        else:
            notInTopKResults['Approximate'] = 'Too long!'

        if not apprxSkipFutureWhyThisTopK:
            try:
                whyThisTopKResults['Approximate'] = executor.submit(approximateWhyThisTopK, dataset['Tuples'], dataset['Functions'][0], 100, k, whyThisTopKResults['BruteForce']['ShapleyValues'] if type(whyThisTopKResults['BruteForce']) is dict else [0.0 for x in range(len(dataset['Tuples'][0]))]).result(timeout=3600)
            except concurrent.futures.TimeoutError:
                whyThisTopKResults['Approximate'] = 'Too long!'
                apprxSkipFutureWhyThisTopK = True
        else:
            whyThisTopKResults['Approximate'] = 'Too long!'

        if not apprxSkipFutureWhyTheseTopKs:
            try:
                whyInTheseTopKResults['Approximate'] = executor.submit(approximateWhyInTheseTopK, dataset['Tuples'], dataset['Functions'], 100, k, inXTopKs, whyInTheseTopKResults['BruteForce']['ShapleyValues'] if type(whyInTheseTopKResults['BruteForce']) is dict else [0.0 for x in range(len(dataset['Tuples'][0]))]).result(timeout=3600)
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

def newVaryingD():
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
                                              [x for x in range(len(dataset['Tuples'][0]))], len(dataset['Tuples'][0]))
        top_k = topk.computeTopK(evaluated_tuples, k)
        top_k_plus_one = topk.computeTopK(evaluated_tuples, k+1)

        in_top_k_results: dict[any, any] = {}
        not_in_top_k_results: dict[any, any] = {}

        if not skip_future_top_k:
            try:
                in_top_k_results['BruteForce'] = executor.submit(bruteForceInTopK, dataset['Tuples'],
                                                              dataset['Functions'][0], k, top_k[k - 1]).result(
                    timeout=3600)
            except concurrent.futures.TimeoutError:
                in_top_k_results['BruteForce'] = 'Too long!'
                skip_future_top_k = True
        else:
            in_top_k_results['BruteForce'] = 'Too long!'
        if not skip_future_not_top_k:
            try:
                not_in_top_k_results['BruteForce'] = executor.submit(bruteForceNotInTopK, dataset['Tuples'], dataset['Functions'][0], k, top_k_plus_one[k]).result(timeout=3600)
            except concurrent.futures.TimeoutError:
                not_in_top_k_results['BruteForce'] = 'Too long!'
                skip_future_not_top_k = True
        else:
            not_in_top_k_results['BruteForce'] = 'Too long!'
        if not apprx_skip_future_top_k:
            try:
                in_top_k_results['Approximate'] = executor.submit(approximateInTopK, dataset['Tuples'],
                                                               dataset['Functions'][0], 100, k, top_k[k - 1],
                                                               in_top_k_results['BruteForce']['ShapleyValues'] if type(
                                                                   in_top_k_results['BruteForce']) is dict else [0.0 for x in
                                                                                                              range(
                                                                                                                  len(
                                                                                                                      dataset[
                                                                                                                          'Tuples'][
                                                                                                                          0]))]).result(
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
                                                                  not_in_top_k_results['BruteForce']['ShapleyValues'] if type(
                                                                      not_in_top_k_results['BruteForce']) is dict else [0.0 for
                                                                                                                    x in
                                                                                                                    range(
                                                                                                                        len(
                                                                                                                            dataset[
                                                                                                                                'Tuples'][
                                                                                                                                0]))]).result(
                    timeout=3600)
            except concurrent.futures.TimeoutError:
                not_in_top_k_results['Approximate'] = 'Too long!'
                apprx_skip_future_not_top_k = True
        else:
            not_in_top_k_results['Approximate'] = 'Too long!'
        prev_results[index]['InTopK'] = in_top_k_results
        prev_results[index]['NotInTopK'] = not_in_top_k_results

    dill.dump(prev_results, open('UpdatedExperimentDResults4.dill', 'wb'))

def newVaryingM():
    dataset = dill.load(open('Varying-D.dill', 'rb'))[8]
    prev_results = dill.load(open('ExperimentMResults8.dill', 'rb'))
    k = 5

    evaluated_tuples = topk.generateTuples(dataset['Tuples'], dataset['Functions'][0],
                                          [x for x in range(len(dataset['Tuples'][0]))], len(dataset['Tuples'][0]))
    topK = topk.computeTopK(evaluated_tuples, k)
    topKPlusOne = topk.computeTopK(evaluated_tuples, k + 1)

    inTopKResults = {}
    notInTopKResults = {}

    inTopKResults['BruteForce'] = bruteForceInTopK(dataset['Tuples'], dataset['Functions'][0], k, topK[k - 1])
    notInTopKResults['BruteForce'] = bruteForceNotInTopK(dataset['Tuples'], dataset['Functions'][0], k, topKPlusOne[k])

    inTopKResults['Approximate'] : dict[any,any] = {}
    notInTopKResults['Approximate'] : dict[any,any] = {}

    mTested = [25,50,75,100,125,150,175,200,225,250]
    for m in mTested:
        inTopKResults['Approximate'][m] = approximateInTopK(dataset['Tuples'], dataset['Functions'][0], m, k, topK[k-1], inTopKResults['BruteForce']['ShapleyValues'])
        notInTopKResults['Approximate'][m] = approximateNotInTopK(dataset['Tuples'], dataset['Functions'][0], m, k, topKPlusOne[k], notInTopKResults['BruteForce']['ShapleyValues'])

    prev_results['InTopK'] = inTopKResults
    prev_results['NotInTopK'] = notInTopKResults

    dill.dump(prev_results, open('UpdatedExperimentMResults.dill', 'wb'))

def computeMaxShapleyValues(ShapleyValues):
     return [tup[1] for tup in sorted([(ShapleyValues[x], x) for x in range(len(ShapleyValues))])[-1:]]

def maskTuples(tuples, attributes):
    return [[tpl[x] if x not in attributes else 0 for x in range(len(tpl)) ] for tpl in tuples]

def removeAttributesExperiment():
    datasets = dill.load(open('1000x100-5-samples', 'rb'))
    trialResults = dill.load(open('MultipleSamplesExperiment2', 'rb'))
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

        k=5

        evaluatedTuples = topk.generateTuples(dataset['Tuples'], dataset['Functions'][0], [x for x in range(len(dataset['Tuples'][0]))], len(dataset['Tuples'][0]))
        topK = topk.computeTopK(evaluatedTuples, k)
        topKPlusOne = topk.computeTopK(evaluatedTuples, k + 1)
        inXTopKs = individualInRangeOfTopKs(dataset['Tuples'], dataset['Functions'], 3, 6, k)

        theseTopKs = set()
        for f in range(len(dataset['Functions'])):
            function = dataset['Functions'][f]
            evaluatedTuples = topk.generateTuples(dataset['Tuples'], function,
                                                  [x for x in range(len(dataset['Tuples'][0]))],
                                                  len(dataset['Tuples'][0]))
            tempTopK = topk.computeTopK(evaluatedTuples, k)
            if inXTopKs in tempTopK:
                theseTopKs.add(f)

        computeFailure = all(
            [abs(ShapleyValue - 1 / len(trialResult['InTopK']['BruteForce']['ShapleyValues'])) < .05 for ShapleyValue
             in ['InTopK']['BruteForce']['ShapleyValues']])

        if computeFailure:
            attributes = range(len(dataset['Tuples'][0]))
            powerSet = chain.from_iterable(combinations(attributes, r) for r in range(len(attributes) + 1))
            for s in powerSet:
                if len(s) == len(attributes):
                    continue
                inTopKTuples = maskTuples(dataset['Tuples'], s)
                evaluatedTuples = topk.generateTuples(inTopKTuples, dataset['Functions'][0],
                                                      [x for x in range(len(dataset['Tuples'][0]))],
                                                      len(dataset['Tuples'][0]))
                newTopk = topk.computeTopK(evaluatedTuples, k)
                if topK[k-1] in newTopk:
                    inTopKScore = inTopKScore + 1 / (len(datasets) * 2 ** len(dataset['Tuples'][0]))
        else:
            inTopKTuples = maskTuples(dataset['Tuples'], computeMaxShapleyValues(trialResult['InTopK']['BruteForce']['ShapleyValues']))
            evaluatedTuples = topk.generateTuples(inTopKTuples, dataset['Functions'][0], [x for x in range(len(dataset['Tuples'][0]))], len(dataset['Tuples'][0]))
            newTopk = topk.computeTopK(evaluatedTuples, k)
            if topK[0] not in newTopk:
                inTopKScore = inTopKScore + 1/len(datasets)


        computeFailure = all(
            [abs(ShapleyValue - 1 / len(trialResult['InTopK']['Approximate']['ShapleyValues'])) < .05 for ShapleyValue
             in ['InTopK']['Approximate']['ShapleyValues']])

        if computeFailure:
            attributes = range(len(dataset['Tuples'][0]))
            powerSet = chain.from_iterable(combinations(attributes, r) for r in range(len(attributes) + 1))
            for s in powerSet:
                if len(s) == len(attributes):
                    continue
                inTopKTuples = maskTuples(dataset['Tuples'], s)
                evaluatedTuples = topk.generateTuples(inTopKTuples, dataset['Functions'][0],
                                                      [x for x in range(len(dataset['Tuples'][0]))],
                                                      len(dataset['Tuples'][0]))
                newTopk = topk.computeTopK(evaluatedTuples, k)
                if topK[k-1] in newTopk:
                    apprxInTopKScore = apprxInTopKScore + 1 / (len(datasets) * 2 ** len(dataset['Tuples'][0]))
        else:
            apprxInTopKTuples = maskTuples(dataset['Tuples'], computeMaxShapleyValues(trialResult['InTopK']['Approximate']['ShapleyValues']))
            evaluatedTuples = topk.generateTuples(apprxInTopKTuples, dataset['Functions'][0], [x for x in range(len(dataset['Tuples'][0]))], len(dataset['Tuples'][0]))
            newTopk = topk.computeTopK(evaluatedTuples, k)
            if topK[0] not in newTopk:
                apprxInTopKScore = apprxInTopKScore + 1/len(datasets)


        computeFailure = all([abs(ShapleyValue - 1/len(trialResult['NotInTopK']['BruteForce']['ShapleyValues'])) < .05 for ShapleyValue in ['NotInTopK']['BruteForce']['ShapleyValues']])

        if computeFailure:
            attributes = range(len(dataset['Tuples'][0]))
            powerSet = chain.from_iterable(combinations(attributes, r) for r in range(len(attributes) + 1))
            for s in powerSet:
                if len(s) == len(attributes):
                    continue
                notInTopKTuples = maskTuples(dataset['Tuples'], s)
                evaluatedTuples = topk.generateTuples(notInTopKTuples, dataset['Functions'][0],
                                                      [x for x in range(len(dataset['Tuples'][0]))],
                                                      len(dataset['Tuples'][0]))
                newTopk = topk.computeTopK(evaluatedTuples, k)
                if topKPlusOne[k] not in newTopk:
                    notInTopKScore = notInTopKScore + 1 / (len(datasets) * 2 ** len(dataset['Tuples'][0]))
        else:
            notInTopKTuples = maskTuples(dataset['Tuples'], computeMaxShapleyValues(
                trialResult['NotInTopK']['BruteForce']['ShapleyValues']))
            evaluatedTuples = topk.generateTuples(notInTopKTuples, dataset['Functions'][0],
                                                  [x for x in range(len(dataset['Tuples'][0]))],
                                                  len(dataset['Tuples'][0]))
            newTopk = topk.computeTopK(evaluatedTuples, k)
            if topKPlusOne[k] in newTopk:
                notInTopKScore = notInTopKScore + 1 / len(datasets)

        computeFailure = all([abs(ShapleyValue - 1/len(trialResult['NotInTopK']['Approximate']['ShapleyValues'])) < .05 for ShapleyValue in ['NotInTopK']['Approximate']['ShapleyValues']])

        if computeFailure:
            attributes = range(len(dataset['Tuples'][0]))
            powerSet = chain.from_iterable(combinations(attributes, r) for r in range(len(attributes) + 1))
            for s in powerSet:
                if len(s) == len(attributes):
                    continue
                apprxNotInTopKTuples = maskTuples(dataset['Tuples'], s)
                evaluatedTuples = topk.generateTuples(apprxNotInTopKTuples, dataset['Functions'][0],
                                                      [x for x in range(len(dataset['Tuples'][0]))],
                                                      len(dataset['Tuples'][0]))
                newTopk = topk.computeTopK(evaluatedTuples, k)
                if topKPlusOne[k] not in newTopk:
                    apprxNotInTopKScore = notInTopKScore + 1 / (len(datasets) * 2 ** len(dataset['Tuples'][0]))

        else:
            apprxNotInTopKTuples = maskTuples(dataset['Tuples'], computeMaxShapleyValues(trialResult['NotInTopK']['Approximate']['ShapleyValues']))
            evaluatedTuples = topk.generateTuples(apprxNotInTopKTuples, dataset['Functions'][0], [x for x in range(len(dataset['Tuples'][0]))], len(dataset['Tuples'][0]))
            newTopk = topk.computeTopK(evaluatedTuples, k)
            if topKPlusOne[k] in newTopk:
                apprxNotInTopKScore = apprxNotInTopKScore + 1/len(datasets)




        whyThisTopKTuples = maskTuples(dataset['Tuples'], computeMaxShapleyValues(trialResult['WhyThisTopK']['BruteForce']['ShapleyValues']))
        evaluatedTuples = topk.generateTuples(whyThisTopKTuples, dataset['Functions'][0], [x for x in range(len(dataset['Tuples'][0]))], len(dataset['Tuples'][0]))
        newTopk = topk.computeTopK(evaluatedTuples, k)
        whyThisTopKScore = whyThisTopKScore + (1 - len((set(newTopk).intersection(set(topK))))/len(set(newTopk).union(set(topK))))/len(datasets)

        apprxWhyThisTopKTuples = maskTuples(dataset['Tuples'], computeMaxShapleyValues(trialResult['WhyThisTopK']['Approximate']['ShapleyValues']))
        evaluatedTuples = topk.generateTuples(apprxWhyThisTopKTuples, dataset['Functions'][0], [x for x in range(len(dataset['Tuples'][0]))], len(dataset['Tuples'][0]))
        newTopk = topk.computeTopK(evaluatedTuples, k)
        apprxWhyThisTopKScore = apprxWhyThisTopKScore + (1 - len((set(newTopk).intersection(set(topK))))/len(set(newTopk).union(set(topK))))/len(datasets)


        whyTheseTopKsTuples = maskTuples(dataset['Tuples'], computeMaxShapleyValues(trialResult['WhyInTheseTopKs']['BruteForce']['ShapleyValues']))
        newTheseTopKs = set()
        for f in range(len(dataset['Functions'])):
            function = dataset['Functions'][f]
            evaluatedTuples = topk.generateTuples(whyTheseTopKsTuples, function,
                                                  [x for x in range(len(dataset['Tuples'][0]))],
                                                  len(dataset['Tuples'][0]))
            tempTopK = topk.computeTopK(evaluatedTuples, k)
            if inXTopKs in tempTopK:
                newTheseTopKs.add(f)

        whyInTheseTopKsScore = whyInTheseTopKsScore + (1 - len((newTheseTopKs.intersection(theseTopKs)))/len(newTheseTopKs.union(theseTopKs)))/len(datasets)


        apprxWhyTheseTopKsTuples = maskTuples(dataset['Tuples'], computeMaxShapleyValues(trialResult['WhyInTheseTopKs']['Approximate']['ShapleyValues']))
        newTheseTopKs = set()
        for f in range(len(dataset['Functions'])):
            function = dataset['Functions'][f]
            evaluatedTuples = topk.generateTuples(apprxWhyTheseTopKsTuples, function,
                                                  [x for x in range(len(dataset['Tuples'][0]))],
                                                  len(dataset['Tuples'][0]))
            tempTopK = topk.computeTopK(evaluatedTuples, k)
            if inXTopKs in tempTopK:
                newTheseTopKs.add(f)


        apprxWhyInTheseTopKsScore = apprxWhyInTheseTopKsScore + (1 - len((newTheseTopKs.intersection(theseTopKs)))/len(newTheseTopKs.union(theseTopKs)))/len(datasets)

    dill.dump([('Why In Top K Score: ', ('Brute Force', inTopKScore), ('Approximate', apprxInTopKScore)),
               ('Why Not In Top K Score: ', ('Brute Force', notInTopKScore), ('Approximate', apprxNotInTopKScore)),
               ('Why This Top K Score', ('Brute Force', whyThisTopKScore), ('Approximate', apprxWhyThisTopKScore)),
               ('Why In These Top Ks Score', ('Brute Force', whyInTheseTopKsScore), ('Approximate', apprxWhyInTheseTopKsScore))]
              , open('RemoveAttributeExperiments.dill', 'wb'))
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

def datasetExperiment(dataset):
    k = 5

    results = {}
    
    evaluated_tuples = topk.generateTuples(dataset['Tuples'], dataset['Functions'][0], [x for x in range(len(dataset['Tuples'][0]))], len(dataset['Tuples'][0]))
    topK = topk.computeTopK(evaluated_tuples, k)
    topKPlusOne = topk.computeTopK(evaluated_tuples, k + 1)
    inXTopKs = individualInRangeOfTopKs(dataset['Tuples'], dataset['Functions'], 3, 6, k)

    inTopKResults = {}
    notInTopKResults = {}
    whyThisTopKResults = {}
    whyInTheseTopKResults = {}

    inTopKResults['BruteForce'] = bruteForceInTopK(dataset['Tuples'], dataset['Functions'][0], k, topK[k-1])
    notInTopKResults['BruteForce'] = bruteForceNotInTopK(dataset['Tuples'], dataset['Functions'][0], k, topKPlusOne[k])
    whyThisTopKResults['BruteForce'] = bruteForceWhyThisTopK(dataset['Tuples'], dataset['Functions'][0], k)
    whyInTheseTopKResults['BruteForce'] = bruteForceWhyInTheseTopK(dataset['Tuples'], dataset['Functions'], k, inXTopKs)
    inTopKResults['Approximate'] = approximateInTopK(dataset['Tuples'], dataset['Functions'][0], 100, k, topK[k-1], inTopKResults['BruteForce']['ShapleyValues'] if type(inTopKResults['BruteForce']) is dict else [0.0 for x in range(len(dataset['Tuples'][0]))])
    notInTopKResults['Approximate'] = approximateNotInTopK(dataset['Tuples'], dataset['Functions'][0], 100, k, topKPlusOne[k], notInTopKResults['BruteForce']['ShapleyValues'] if type(notInTopKResults['BruteForce']) is dict else [0.0 for x in range(len(dataset['Tuples'][0]))])
    whyThisTopKResults['Approximate'] = approximateWhyThisTopK(dataset['Tuples'], dataset['Functions'][0], 100, k, whyThisTopKResults['BruteForce']['ShapleyValues'] if type(whyThisTopKResults['BruteForce']) is dict else [0.0 for x in range(len(dataset['Tuples'][0]))])
    whyInTheseTopKResults['Approximate'] = approximateWhyInTheseTopK(dataset['Tuples'], dataset['Functions'], 100, k, inXTopKs, whyInTheseTopKResults['BruteForce']['ShapleyValues'] if type(whyInTheseTopKResults['BruteForce']) is dict else [0.0 for x in range(len(dataset['Tuples'][0]))])

    results['InTopK'] = inTopKResults
    results['NotInTopK'] = notInTopKResults
    results['WhyThisTopK'] = whyThisTopKResults
    results['WhyInTheseTopKs'] = whyInTheseTopKResults

    return results


# datasets = dill.load(open('100x100-5-samples', 'rb'))
# results = []
# for dataset in datasets:
#    results.append(datasetExperiment(dataset))
# dill.dump(results, open('MultipleSamplesExperiment2', 'wb'))

removeAttributesExperiment()
#newVaryingD()
#newVaryingM()
