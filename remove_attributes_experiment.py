import sys

import dill
import experiments
import topk
from itertools import chain, combinations


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
            topK = inTopK(t, tuples, functions, k, d, unWrapFunction)
            borderline = borderLineTopK(t, tuples, functions, k, d, unWrapFunction)
            if topK is not False and borderline is not False and tInXTopKs(tuples, t, functions, k, minim, maxim, d,
                                                                           unWrapFunction) is not False:
                return t, topK, borderline
        maxim = maxim + 1

def transformToRanks(tuples, function, d, t):
    ranks = []
    for x in range(d):
        evaluated_tuples = topk.generateTuples(tuples, function, [d], 1, None)
        rank = 1
        for evaluated_tuple in evaluated_tuples:
            if evaluated_tuple[0] > evaluated_tuples[t][0]:
                rank = rank + 1
        ranks.append(rank)
    return ranks

def transformToJaccards(tuples, function, d, k, originalTopk):
    jaccards = []
    for x in range(d):
        evaluated_tuples = topk.generateTuples(tuples, function, [d], 1, None)
        newTopK = topk.computeTopK(evaluated_tuples, k)
        jaccards.append(len(set(newTopK).intersection(set(originalTopk)))/len(set(newTopK).union(set(originalTopk))))
    return jaccards

def transformToJaccards2(tuples, functions, d, t, k, originalTopk):
    jaccards = []
    for x in range(d):
        currentSet = set()
        for f in range(len(functions)):
            function = functions[f]
            evaluated_tuples = topk.generateTuples(tuples, function, [d], 1, None)
            newTopK = topk.computeTopK(evaluated_tuples, k)
            if t in newTopK:
                currentSet.add(f)
        jaccards.append(len(set(currentSet).intersection(set(originalTopk)))/len(set(currentSet).union(set(originalTopk))))
    return jaccards

def maskTuples(tuples, attributes, unWrapFunction):
    return [[tpl[x] if x not in (unWrapFunction(attributes) if unWrapFunction is not None else attributes) else None for x in range(len(tpl)) ] for tpl in tuples]


def computeMaxShapleyValues(ShapleyValues):
    return [tup[1] for tup in sorted([(ShapleyValues[x], x) for x in range(len(ShapleyValues))])[-1:]]


def remove_attributes(datasets, trialResults, k, unWrapFunction):
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
        evaluatedTuples = topk.generateTuples(apprxWhyThisTopKTuples, dataset['Functions'][t], [x for x in range(len(dataset['Tuples'][0]))], len(dataset['Tuples'][0]), unWrapFunction)
        newTopk = topk.computeTopK(evaluatedTuples, k)
        apprxWhyThisTopKScore = apprxWhyThisTopKScore + (1 - len((set(newTopk).intersection(set(topK))))/len(set(newTopk).union(set(topK))))/len(datasets)


        whyTheseTopKsTuples = maskTuples(dataset['Tuples'], computeMaxShapleyValues(
            trialResult['WhyInTheseTopKs']['BruteForce']['ShapleyValues']), unWrapFunction)
        newTheseTopKs = set()
        for f in range(len(dataset['Functions'])):
            function = dataset['Functions'][f]
            evaluatedTuples = topk.generateTuples(whyTheseTopKsTuples, function,
                                                  [x for x in range(len(dataset['Tuples'][0]))],
                                                  len(dataset['Tuples'][0]))
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
                                                  len(dataset['Tuples'][0]))
            tempTopK = topk.computeTopK(evaluatedTuples, k)
            if t in tempTopK:
                newTheseTopKs.add(f)


        apprxWhyInTheseTopKsScore = apprxWhyInTheseTopKsScore + (1 - len((newTheseTopKs.intersection(theseTopKs)))/len(newTheseTopKs.union(theseTopKs)))/len(datasets)


    return [('Why In Top K Score: ', ('Brute Force', inTopKScore), ('Approximate', apprxInTopKScore)),
               ('Why Not In Top K Score: ', ('Brute Force', notInTopKScore), ('Approximate', apprxNotInTopKScore)),
               ('Why This Top K Score', ('Brute Force', whyThisTopKScore), ('Approximate', apprxWhyThisTopKScore)),
               ('Why In These Top Ks Score', ('Brute Force', whyInTheseTopKsScore), ('Approximate', apprxWhyInTheseTopKsScore))]

def generate_dataset(dataset, d, unWrapFunction, k):

    results = {}

    t, topkFunc, borderlineFunc = findQueryPoint(dataset['Tuples'], k, dataset['Functions'], 6, None, 3, 6)

    inTopKResults = {}
    notInTopKResults = {}
    whyThisTopKResults = {}
    whyInTheseTopKResults = {}

    inTopKResults['BruteForce'] = experiments.bruteForceInTopK(dataset['Tuples'], dataset['Functions'][topkFunc], k, t, d, unWrapFunction)
    notInTopKResults['BruteForce'] = experiments.bruteForceNotInTopK(dataset['Tuples'], dataset['Functions'][borderlineFunc], k, t, d, unWrapFunction)
    whyThisTopKResults['BruteForce'] = experiments.bruteForceWhyThisTopK(dataset['Tuples'], dataset['Functions'][t], k, d, unWrapFunction)
    whyInTheseTopKResults['BruteForce'] = experiments.bruteForceWhyInTheseTopK(dataset['Tuples'], dataset['Functions'], k, t, d, unWrapFunction)
    inTopKResults['Approximate'] = experiments.approximateInTopK(dataset['Tuples'], dataset['Functions'][topkFunc], 200, k, t, d, inTopKResults['BruteForce']['ShapleyValues'] if type(inTopKResults['BruteForce']) is dict else [0.0 for x in range(len(dataset['Tuples'][0]))], unWrapFunction)
    notInTopKResults['Approximate'] = experiments.approximateNotInTopK(dataset['Tuples'], dataset['Functions'][borderlineFunc], 200, k, t, d, notInTopKResults['BruteForce']['ShapleyValues'] if type(notInTopKResults['BruteForce']) is dict else [0.0 for x in range(len(dataset['Tuples'][0]))], unWrapFunction)
    whyThisTopKResults['Approximate'] = experiments.approximateWhyThisTopK(dataset['Tuples'], dataset['Functions'][t], 200, k, d, whyThisTopKResults['BruteForce']['ShapleyValues'] if type(whyThisTopKResults['BruteForce']) is dict else [0.0 for x in range(len(dataset['Tuples'][0]))], unWrapFunction)
    whyInTheseTopKResults['Approximate'] = experiments.approximateWhyInTheseTopK(dataset['Tuples'], dataset['Functions'], 200, k, t, d, whyInTheseTopKResults['BruteForce']['ShapleyValues'] if type(whyInTheseTopKResults['BruteForce']) is dict else [0.0 for x in range(len(dataset['Tuples'][0]))], unWrapFunction)

    results['Query Point'] = (t, topkFunc, borderlineFunc)
    results['InTopK'] = inTopKResults
    results['NotInTopK'] = notInTopKResults
    results['WhyThisTopK'] = whyThisTopKResults
    results['WhyInTheseTopKs'] = whyInTheseTopKResults

    return results

def remove_attributes_heuristics(datasets, k, unWrapFunction):
    inTopKScoreWeights = 0
    inTopKScoreRank = 0
    notInTopKScoreWeights = 0
    notInTopKScoreRank = 0
    whyThisTopKScoreWeights = 0
    whyThisTopKScoreJaccard = 0
    whyInTheseTopKsScoreWeight = 0
    whyInTheseTopKsScoreJaccard = 0

    for x in range(len(datasets)):
        dataset = datasets[x]

        t, topkFunc, borderlineFunc = findQueryPoint(dataset['Tuples'], k, dataset['Functions'], 6, None, 3, 6)

        topKFuncEvaluatedTuples = topk.generateTuples(dataset['Tuples'], dataset['Functions'][t], [x for x in range(len(dataset['Tuples'][0]))], len(dataset['Tuples'][0]), unWrapFunction)
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

        maxWeightInTopK = dataset['Weights'][topkFunc].index(max(dataset['Weights'][topkFunc]))
        inTopKRanks = transformToRanks(dataset['Tuples'], dataset['Functions'][topkFunc], 6, t)
        maxRankInTopK = inTopKRanks.index(max(inTopKRanks))

        maxWeightInTopKTuples = maskTuples(dataset['Tuples'],
                                  [maxWeightInTopK],
                                  unWrapFunction)
        evaluatedTuples = topk.generateTuples(maxWeightInTopKTuples, dataset['Functions'][topkFunc], [x for x in range(len(dataset['Tuples'][0]))], len(dataset['Tuples'][0]), unWrapFunction)
        newTopk = topk.computeTopK(evaluatedTuples, k)
        if t not in newTopk:
            inTopKScoreWeights = inTopKScoreWeights + 1/len(datasets)

        maxRankInTopKTuples = maskTuples(dataset['Tuples'],
                                  [maxRankInTopK],
                                  unWrapFunction)
        evaluatedTuples = topk.generateTuples(maxRankInTopKTuples, dataset['Functions'][topkFunc], [x for x in range(len(dataset['Tuples'][0]))], len(dataset['Tuples'][0]), unWrapFunction)
        newTopk = topk.computeTopK(evaluatedTuples, k)
        if t not in newTopk:
            inTopKScoreRank = inTopKScoreRank + 1/len(datasets)

        maxWeightNotInTopK = dataset['Weights'][borderlineFunc].index(max(dataset['Weights'][borderlineFunc]))
        notInTopKRanks = transformToRanks(dataset['Tuples'], dataset['Functions'][borderlineFunc], 6, t)
        minRankNotInTopK = notInTopKRanks.index(min(notInTopKRanks))

        maxWeightNotInTopKTuples = maskTuples(dataset['Tuples'],
                                  [maxWeightNotInTopK],
                                  unWrapFunction)
        evaluatedTuples = topk.generateTuples(maxWeightNotInTopKTuples, dataset['Functions'][borderlineFunc],
                                              [x for x in range(len(dataset['Tuples'][0]))],
                                              len(dataset['Tuples'][0]), unWrapFunction)
        newTopk = topk.computeTopK(evaluatedTuples, k)
        if t in newTopk:
            notInTopKScoreWeights = notInTopKScoreWeights + 1 / len(datasets)

        minRankNotInTopKTuples = maskTuples(dataset['Tuples'],
                                         [minRankNotInTopK],
                                         unWrapFunction)
        evaluatedTuples = topk.generateTuples(minRankNotInTopKTuples, dataset['Functions'][borderlineFunc],
                                              [x for x in range(len(dataset['Tuples'][0]))], len(dataset['Tuples'][0]),
                                              unWrapFunction)
        newTopk = topk.computeTopK(evaluatedTuples, k)
        if t in newTopk:
            notInTopKScoreRank = notInTopKScoreRank + 1 / len(datasets)

        maxWeightThisTopK = dataset['Weights'][t].index(max(dataset['Weights'][t]))
        thisTopKJaccards = transformToJaccards(dataset['Tuples'], dataset['Functions'][t], 6, k, topK)
        maxJaccardThisTopK = thisTopKJaccards.index(max(thisTopKJaccards))

        maxWeightThisTopKTuples = maskTuples(dataset['Tuples'],
                                           [maxWeightThisTopK],
                                           unWrapFunction)
        evaluatedTuples = topk.generateTuples(maxWeightThisTopKTuples, dataset['Functions'][t],
                                              [x for x in range(len(dataset['Tuples'][0]))], len(dataset['Tuples'][0]),
                                              unWrapFunction)
        newTopk = topk.computeTopK(evaluatedTuples, k)
        whyThisTopKScoreWeights = whyThisTopKScoreWeights + (1 - len((set(newTopk).intersection(set(topK))))/len(set(newTopk).union(set(topK))))/len(datasets)

        maxJaccardThisTopKTuples = maskTuples(dataset['Tuples'],
                                           [maxJaccardThisTopK],
                                           unWrapFunction)
        evaluatedTuples = topk.generateTuples(maxJaccardThisTopKTuples, dataset['Functions'][t], [x for x in range(len(dataset['Tuples'][0]))], len(dataset['Tuples'][0]), None)
        newTopk = topk.computeTopK(evaluatedTuples, k)
        whyThisTopKScoreJaccard = whyThisTopKScoreJaccard + (1 - len((set(newTopk).intersection(set(topK))))/len(set(newTopk).union(set(topK))))/len(datasets)

        avgWeights = [0 for x in range(6)]
        for weights in dataset['Weights']:
            for weight in range(len(weights)):
                avgWeights[weight] = avgWeights[weight] + weights[weight]/len(dataset['Weights'])
        maxAvgWeightTheseTopKs = avgWeights.index(max(avgWeights))
        theseTopKJaccards = transformToJaccards2(dataset['Tuples'], dataset['Functions'], 6, t, k, theseTopKs)
        maxJaccardTheseTopKs = theseTopKJaccards.index(max(theseTopKJaccards))

        maxAvgWeightTheseTopKsTuples = maskTuples(dataset['Tuples'],
                                           [maxAvgWeightTheseTopKs],
                                           unWrapFunction)
        newTheseTopKs = set()
        for f in range(len(dataset['Functions'])):
            function = dataset['Functions'][f]
            evaluatedTuples = topk.generateTuples(maxAvgWeightTheseTopKsTuples, function,
                                                  [x for x in range(len(dataset['Tuples'][0]))],
                                                  len(dataset['Tuples'][0]), None)
            tempTopK = topk.computeTopK(evaluatedTuples, k)
            if t in tempTopK:
                newTheseTopKs.add(f)

        whyInTheseTopKsScoreWeight = whyInTheseTopKsScoreWeight + (1 - len((newTheseTopKs.intersection(theseTopKs)))/len(newTheseTopKs.union(theseTopKs)))/len(datasets)

        maxJaccardTheseTopKTuples = maskTuples(dataset['Tuples'],
                                           [maxJaccardTheseTopKs],
                                           unWrapFunction)
        newTheseTopKs = set()
        for f in range(len(dataset['Functions'])):
            function = dataset['Functions'][f]
            evaluatedTuples = topk.generateTuples(maxJaccardTheseTopKTuples, function,
                                                  [x for x in range(len(dataset['Tuples'][0]))],
                                                  len(dataset['Tuples'][0]), None)
            tempTopK = topk.computeTopK(evaluatedTuples, k)
            if t in tempTopK:
                newTheseTopKs.add(f)

        whyInTheseTopKsScoreJaccard = whyInTheseTopKsScoreJaccard + (1 - len((newTheseTopKs.intersection(theseTopKs)))/len(newTheseTopKs.union(theseTopKs)))/len(datasets)


    return [('Why In Top K Score: ', ('Weight', inTopKScoreWeights), ('Max Rank', inTopKScoreRank)),
               ('Why Not In Top K Score: ', ('Weight', notInTopKScoreWeights), ('Min Rank', notInTopKScoreRank)),
               ('Why This Top K Score', ('Weight', whyThisTopKScoreWeights), ('Jaccard', whyThisTopKScoreJaccard)),
               ('Why In These Top Ks Score', ('Weight', whyInTheseTopKsScoreWeight), ('Jaccard', whyInTheseTopKsScoreJaccard))]

def remove_attributes_experiment(experiments):
    res = []
    datasets = []

    dataset = {}
    funcsFile = dill.load(open('Removing-Functions-Linear.dill', 'rb'))
    functions = funcsFile['Functions']
    weights = funcsFile['Weights']

    if 'AUL' in experiments:
        for tuples in dill.load(open('data/a_u_100_6_9.dill', 'rb')):
            dataset['Tuples'] = tuples
            dataset['Functions'] = functions[:100]
            dataset['Weights'] = weights[:100]
            functions = functions[100:]
            weights = weights[100:]
            res.append(generate_dataset(dataset, 6, None, 5))
            datasets.append(dataset)

        remove_attributes_results = remove_attributes(datasets, res, 5, None)
        remove_attributes_heuristics_results = remove_attributes_heuristics(datasets, 5, None)
        dill.dump((remove_attributes_results, remove_attributes_heuristics_results), open('remove_results_a_u_l.dill', 'wb'))

    res = []
    datasets = []

    if 'IUL' in experiments:
        for tuples in dill.load(open('data/i_u_100_6_9.dill', 'rb')):
            dataset['Tuples'] = tuples
            dataset['Functions'] = functions[:100]
            dataset['Weights'] = weights[:100]
            functions = functions[100:]
            weights = weights[100:]
            res.append(generate_dataset(dataset, 6, None, 5))
            datasets.append(dataset)
        remove_attributes_results = remove_attributes(datasets, res, 5, None)
        remove_attributes_heuristics_results = remove_attributes_heuristics(datasets, 5, None)
        dill.dump((remove_attributes_results, remove_attributes_heuristics_results), open('remove_results_i_u_l.dill', 'wb'))


    res = []
    datasets = []

    if 'CUL' in experiments:
        for tuples in dill.load(open('data/c_u_100_6_9.dill', 'rb')):
            dataset['Tuples'] = tuples
            dataset['Functions'] = functions[:100]
            dataset['Weights'] = weights[:100]
            functions = functions[100:]
            weights = weights[100:]
            res.append(generate_dataset(dataset, 6, None, 5))
            datasets.append(dataset)
        remove_attributes_results = remove_attributes(datasets, res, 5, None)
        remove_attributes_heuristics_results = remove_attributes_heuristics(datasets, 5, None)
        dill.dump((remove_attributes_results, remove_attributes_heuristics_results), open('remove_results_c_u_l.dill', 'wb'))

    res = []
    datasets = []

    if 'AZL' in experiments:
        for tuples in dill.load(open('data/a_z_100_6_9.dill', 'rb')):
            dataset['Tuples'] = tuples
            dataset['Functions'] = functions[:100]
            dataset['Weights'] = weights[:100]
            functions = functions[100:]
            weights = weights[100:]
            res.append(generate_dataset(dataset, 6, None, 5))
            datasets.append(dataset)
        remove_attributes_results = remove_attributes(datasets, res, 5, None)
        remove_attributes_heuristics_results = remove_attributes_heuristics(datasets, 5, None)
        dill.dump((remove_attributes_results, remove_attributes_heuristics_results), open('remove_results_a_z_l.dill', 'wb'))

    res = []
    datasets = []

    if 'IZL' in experiments:
        for tuples in dill.load(open('data/i_z_100_6_9.dill', 'rb')):
            dataset['Tuples'] = tuples
            dataset['Functions'] = functions[:100]
            dataset['Weights'] = weights[:100]
            functions = functions[100:]
            weights = weights[100:]
            res.append(generate_dataset(dataset, 6, None, 5))
            datasets.append(dataset)
        remove_attributes_results = remove_attributes(datasets, res, 5, None)
        remove_attributes_heuristics_results = remove_attributes_heuristics(datasets, 5, None)
        dill.dump((remove_attributes_results, remove_attributes_heuristics_results), open('remove_results_i_z_l.dill', 'wb'))

    res = []
    datasets = []

    if 'CZL' in experiments:
        for tuples in dill.load(open('data/c_z_100_6_9.dill', 'rb')):
            dataset['Tuples'] = tuples
            dataset['Functions'] = functions[:100]
            dataset['Weights'] = weights[:100]
            functions = functions[100:]
            weights = weights[100:]
            res.append(generate_dataset(dataset, 6, None, 5))
            datasets.append(dataset)
        remove_attributes_results = remove_attributes(datasets, res, 5, None)
        remove_attributes_heuristics_results = remove_attributes_heuristics(datasets, 5, None)
        dill.dump((remove_attributes_results, remove_attributes_heuristics_results), open('remove_results_c_z_l.dill', 'wb'))

    res = []
    datasets = []

    funcsFile = dill.load(open('Removing-Functions-Nonlinear.dill', 'rb'))
    functions = funcsFile['Functions']
    weights = funcsFile['Weights']
    if 'AUNL' in experiments:
        for tuples in dill.load(open('data/a_u_100_6_9.dill', 'rb')):
            dataset['Tuples'] = tuples
            dataset['Functions'] = functions[:100]
            dataset['Weights'] = weights[:100]
            functions = functions[100:]
            weights = weights[100:]
            res.append(generate_dataset(dataset, 6, None, 5))
            datasets.append(dataset)
        remove_attributes_results = remove_attributes(datasets, res, 5, None)
        remove_attributes_heuristics_results = remove_attributes_heuristics(datasets, 5, None)
        dill.dump((remove_attributes_results, remove_attributes_heuristics_results), open('remove_results_a_u_nl.dill', 'wb'))

    res = []
    datasets = []

    if 'IUNL' in experiments:
        for tuples in dill.load(open('data/i_u_100_6_9.dill', 'rb')):
            dataset['Tuples'] = tuples
            dataset['Functions'] = functions[:100]
            dataset['Weights'] = weights[:100]
            functions = functions[100:]
            weights = weights[100:]
            res.append(generate_dataset(dataset, 6, None, 5))
            datasets.append(dataset)
        remove_attributes_results = remove_attributes(datasets, res, 5, None)
        remove_attributes_heuristics_results = remove_attributes_heuristics(datasets, 5, None)
        dill.dump((remove_attributes_results, remove_attributes_heuristics_results), open('remove_results_i_u_nl.dill', 'wb'))

    res = []
    datasets = []

    if 'CUNL' in experiments:
        for tuples in dill.load(open('data/c_u_100_6_9.dill', 'rb')):
            dataset['Tuples'] = tuples
            dataset['Functions'] = functions[:100]
            dataset['Weights'] = weights[:100]
            functions = functions[100:]
            weights = weights[100:]
            res.append(generate_dataset(dataset, 6, None, 5))
            datasets.append(dataset)
        remove_attributes_results = remove_attributes(datasets, res, 5, None)
        remove_attributes_heuristics_results = remove_attributes_heuristics(datasets, 5, None)
        dill.dump((remove_attributes_results, remove_attributes_heuristics_results), open('remove_results_c_u_nl.dill', 'wb'))

    res = []
    datasets = []

    if 'AZNL' in experiments:
        for tuples in dill.load(open('data/a_z_100_6_9.dill', 'rb')):
            dataset['Tuples'] = tuples
            dataset['Functions'] = functions[:100]
            dataset['Weights'] = weights[:100]
            functions = functions[100:]
            weights = weights[100:]
            res.append(generate_dataset(dataset, 6, None, 5))
            datasets.append(dataset)
        remove_attributes_results = remove_attributes(datasets, res, 5, None)
        remove_attributes_heuristics_results = remove_attributes_heuristics(datasets, 5, None)
        dill.dump((remove_attributes_results, remove_attributes_heuristics_results), open('remove_results_a_z_nl.dill', 'wb'))

    res = []
    datasets = []

    if 'IZNL' in experiments:
        for tuples in dill.load(open('data/i_z_100_6_9.dill', 'rb')):
            dataset['Tuples'] = tuples
            dataset['Functions'] = functions[:100]
            dataset['Weights'] = weights[:100]
            functions = functions[100:]
            weights = weights[100:]
            res.append(generate_dataset(dataset, 6, None, 5))
            datasets.append(dataset)
        remove_attributes_results = remove_attributes(datasets, res, 5, None)
        remove_attributes_heuristics_results = remove_attributes_heuristics(datasets, 5, None)
        dill.dump((remove_attributes_results, remove_attributes_heuristics_results), open('remove_results_i_z_nl.dill', 'wb'))

    res = []
    datasets = []

    if 'CZNL' in experiments:
        for tuples in dill.load(open('data/c_z_100_6_9.dill', 'rb')):
            dataset['Tuples'] = tuples
            dataset['Functions'] = functions[:100]
            dataset['Weights'] = weights[:100]
            functions = functions[100:]
            weights = weights[100:]
            res.append(generate_dataset(dataset, 6, None, 5))
            datasets.append(dataset)
        remove_attributes_results = remove_attributes(datasets, res, 5, None)
        remove_attributes_heuristics_results = remove_attributes_heuristics(datasets, 5, None)
        dill.dump((remove_attributes_results, remove_attributes_heuristics_results), open('remove_results_c_z_nl.dill', 'wb'))

if __name__ == "__main__":
    remove_attributes_experiment(sys.argv[1:])