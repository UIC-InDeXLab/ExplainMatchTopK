import dill
import bruteforce
import approximate
import time
import topk
import numpy as np
import multiprocessing
import threading
import concurrent.futures

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
    notInTopK = 0
    while notInTopK in topK:
        notInTopK = notInTopK + 1
    inXTopKs = individualInRangeOfTopKs(dataset['Tuples'], dataset['Functions'], 5, 5, k)

    inTopKResults = {}
    notInTopKResults = {}
    whyThisTopKResults = {}
    whyInTheseTopKResults = {}

    inTopKResults['BruteForce'] = bruteForceInTopK(dataset['Tuples'], dataset['Functions'][0], k, topK[0])
    notInTopKResults['BruteForce'] = bruteForceNotInTopK(dataset['Tuples'], dataset['Functions'][0], k, notInTopK)
    whyThisTopKResults['BruteForce'] = bruteForceWhyThisTopK(dataset['Tuples'], dataset['Functions'][0], k)
    whyInTheseTopKResults['BruteForce'] = bruteForceWhyInTheseTopK(dataset['Tuples'], dataset['Functions'], k, inXTopKs)

    inTopKResults['Approximate'] = {}
    notInTopKResults['Approximate'] = {}
    whyThisTopKResults['Approximate'] = {}
    whyInTheseTopKResults['Approximate'] = {}

    for m in mTested:   
        inTopKResults['Approximate'][m] = approximateInTopK(dataset['Tuples'], dataset['Functions'][0], m, k, topK[0], inTopKResults['BruteForce']['ShapleyValues'])
        notInTopKResults['Approximate'][m] = approximateNotInTopK(dataset['Tuples'], dataset['Functions'][0], m, k, notInTopK, notInTopKResults['BruteForce']['ShapleyValues'])
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
        notInTopK = 0
        while notInTopK in topK:
            notInTopK = notInTopK + 1
        inXTopKs = individualInRangeOfTopKs(dataset['Tuples'], dataset['Functions'], 5, 8, k)

        inTopKResults = {}
        notInTopKResults = {}
        whyThisTopKResults = {}
        whyInTheseTopKResults = {}

        if not skipFutureTopK:
            try:
                inTopKResults['BruteForce'] = executor.submit(bruteForceInTopK, dataset['Tuples'], dataset['Functions'][0], k, topK[0]).result(timeout=3600)
            except concurrent.futures.TimeoutError:
                inTopKResults['BruteForce'] = 'Too long!'
                skipFutureTopK = True
        else:
            inTopKResults['BruteForce'] = 'Too long!'
            
        if not skipFutureNotTopK:
            try:
                notInTopKResults['BruteForce'] = executor.submit(bruteForceNotInTopK, dataset['Tuples'], dataset['Functions'][0], k, notInTopK).result(timeout=3600)
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
                inTopKResults['Approximate'] = executor.submit(approximateInTopK, dataset['Tuples'], dataset['Functions'][0], 100, k, topK[0], inTopKResults['BruteForce']['ShapleyValues'] if type(inTopKResults['BruteForce']) is dict else [0.0 for x in range(len(dataset['Tuples'][0]))]).result(timeout=3600)
            except concurrent.futures.TimeoutError:
                inTopKResults['Approximate'] = 'Too long!'
                apprxSkipFutureTopK = True
        else:
            inTopKResults['Approximate'] = 'Too long!'

        if not apprxSkipFutureNotTopK:
            try:
                notInTopKResults['Approximate'] = executor.submit(approximateInTopK, dataset['Tuples'], dataset['Functions'][0], 100, k, notInTopK, notInTopKResults['BruteForce']['ShapleyValues'] if type(notInTopKResults['BruteForce']) is dict else [0.0 for x in range(len(dataset['Tuples'][0]))]).result(timeout=3600)
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
                whyThisTopKResults['Approximate'] = 'Too long!'
                apprxSkipFutureWhyTheseTopKs = True
        else:
            whyThisTopKResults['Approximate'] = 'Too long!'
        
        results['InTopK'] = inTopKResults
        results['NotInTopK'] = notInTopKResults
        results['WhyThisTopK'] = whyThisTopKResults
        results['WhyInTheseTopKs'] = whyInTheseTopKResults
        resultsFinal[index] = results

    return resultsFinal


res = varyingDExperiment()
dill.dump(res, open('ExperimentDResults.dill', 'wb'))
