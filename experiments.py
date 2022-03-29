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

def newVaryingD():
    datasets = dill.load(open('Varying-D.dill', 'rb'))
    prevResults = dill.load(open('ExperimentDResults', 'rb'))
    k = 5

    for index in sorted(datasets.keys()):
        dataset = datasets[index]
        evaluatedTuples = topk.generateTuples(dataset['Tuples'], dataset['Functions'][0],
                                              [x for x in range(len(dataset['Tuples'][0]))], len(dataset['Tuples'][0]))
        topK = topk.computeTopK(evaluatedTuples, k)
        notInTopK = 0
        while notInTopK in topK:
            notInTopK = notInTopK + 1

        prevResults = prevResults[index]

        notInTopKResults = prevResults['NotInTopK']

        try:
            prevResults['NotInTopK']['Approximate'] = executor.submit(approximateNotInTopK, dataset['Tuples'],
                                                              dataset['Functions'][0], 100, k, notInTopK,
                                                              notInTopKResults['BruteForce']['ShapleyValues'] if type(
                                                                  notInTopKResults['BruteForce']) is dict else [0.0 for
                                                                                                                x in
                                                                                                                range(
                                                                                                                    len(
                                                                                                                        dataset[
                                                                                                                            'Tuples'][
                                                                                                                            0]))]).result(
                timeout=3600)
        except concurrent.futures.TimeoutError:
            notInTopKResults['Approximate'] = 'Too long!'

    dill.dump(prevResults, open('UpdatedExperimentDResults', 'wb'))

def computeMaxShapleyValues(ShapleyValues):
     return [tup[1] for tup in sorted([(ShapleyValues[x], x) for x in range(len(ShapleyValues))])[-2:]]

def maskTuples(tuples, attributes):
    return [[tpl[x] if x not in attributes else 0 for x in range(len(tpl)) ] for tpl in tuples]

def removeAttributesExperiment():
    datasets = dill.load(open('1000-8-samples.dill', 'rb'))
    trialResults = dill.load(open('MultipleSamplesExperiment', 'rb'))
    inTopKScore = 0
    notInTopKScore = 0
    whyThisTopKScore = 0
    whyInTheseTopKsScore = 0

    for x in range(len(datasets)):
        dataset = datasets[x]
        trialResult = trialResults[x]

        k=5

        evaluatedTuples = topk.generateTuples(dataset['Tuples'], dataset['Functions'][0], [x for x in range(len(dataset['Tuples'][0]))], len(dataset['Tuples'][0]))
        topK = topk.computeTopK(evaluatedTuples, k)
        inXTopKs = individualInRangeOfTopKs(dataset['Tuples'], dataset['Functions'], 5, 5, k)
        notInTopK = 0
        while notInTopK in topK:
            notInTopK = notInTopK + 1

        theseTopKs = set()
        for f in range(len(dataset['Functions'])):
            function = dataset['Functions'][f]
            evaluatedTuples = topk.generateTuples(dataset['Tuples'], function,
                                                  [x for x in range(len(dataset['Tuples'][0]))],
                                                  len(dataset['Tuples'][0]))
            tempTopK = topk.computeTopK(evaluatedTuples, k)
            if inXTopKs in tempTopK:
                theseTopKs.add(f)

        inTopKTuples = maskTuples(dataset['Tuples'], computeMaxShapleyValues(trialResult['InTopK']['BruteForce']['ShapleyValues']))
        evaluatedTuples = topk.generateTuples(inTopKTuples, dataset['Functions'][0], [x for x in range(len(dataset['Tuples'][0]))], len(dataset['Tuples'][0]))
        newTopk = topk.computeTopK(evaluatedTuples, k)
        if topK[0] not in newTopk:
            inTopKScore = inTopKScore + 1/len(datasets)

        notInTopKTuples = maskTuples(dataset['Tuples'], computeMaxShapleyValues(trialResult['NotInTopK']['BruteForce']['ShapleyValues']))
        evaluatedTuples = topk.generateTuples(notInTopKTuples, dataset['Functions'][0], [x for x in range(len(dataset['Tuples'][0]))], len(dataset['Tuples'][0]))
        newTopk = topk.computeTopK(evaluatedTuples, k)
        if notInTopK in newTopk:
            notInTopKScore = notInTopKScore + 1/len(datasets)

        whyThisTopKTuples = maskTuples(dataset['Tuples'], computeMaxShapleyValues(trialResult['WhyThisTopK']['BruteForce']['ShapleyValues']))
        evaluatedTuples = topk.generateTuples(whyThisTopKTuples, dataset['Functions'][0], [x for x in range(len(dataset['Tuples'][0]))], len(dataset['Tuples'][0]))
        newTopk = topk.computeTopK(evaluatedTuples, k)
        whyThisTopKScore = whyThisTopKScore + (1 - len((set(newTopk).intersection(set(topK))))/len(set(newTopk).union(set(topK))))/len(datasets)

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

    dill.dump([('Why In Top K Score: ', inTopKScore), ('Why Not In Top K Score: ', notInTopKScore), ('Why This Top K Score', whyThisTopKTuples), ('Why In These Top Ks Score', whyInTheseTopKsScore)], open('RemoveAttributeExperiments.dill', 'wb'))
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

    results['InTopK'] = inTopKResults
    results['NotInTopK'] = notInTopKResults
    results['WhyThisTopK'] = whyThisTopKResults
    results['WhyInTheseTopKs'] = whyInTheseTopKResults

    return results

#datasets = dill.load(open('1000-8-samples.dill', 'rb'))
#results = []
#for dataset in datasets:
#    results.append(datasetExperiment(dataset))
#dill.dump(results, open('MultipleSamplesExperiment', 'rb'))

#removeAttributesExperiment()
newVaryingD()