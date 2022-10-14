import sys
import dill
import topk
import experiments

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


def computeRanking(attributes, flag=False):
    ranks = []
    for curr in attributes:
        for attribute in attributes:
            rank = 1
            if attribute > curr + (.01 if flag else 0):
                rank = rank + 1
        ranks.append(rank)
    return ranks


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

def calculate_top_attribute_scores(datasets, k, unWrapFunction):
    inTopKScoreWeights = 0
    inTopKScoreRank = 0
    inTopKScoreApprox = 0
    notInTopKScoreWeights = 0
    notInTopKScoreRank = 0
    notInTopKScoreApprox = 0
    whyThisTopKScoreWeights = 0
    whyThisTopKScoreJaccard = 0
    whyThisTopKScoreApprox = 0
    whyInTheseTopKsScoreJaccard = 0
    whyInTheseTopKsScoreApprox = 0

    whyTheseTopKMaxShapley = []

    for x in range(len(datasets)):
        dataset = datasets[x]

        result = experiments.datasetExperiment(dataset, 6, None, 5)

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

        inTopKMaxShapley = [x for x in range(len(result['InTopK']['BruteForce']['ShapleyValues'])) if abs(result['InTopK']['BruteForce']['ShapleyValues'][x] - max(result['InTopK']['BruteForce']['ShapleyValues'])) < .00001]

        maxWeightInTopK = dataset['Weights'][topkFunc].index(max(dataset['Weights'][topkFunc]))
        if maxWeightInTopK in inTopKMaxShapley:
            inTopKScoreWeights = inTopKScoreWeights + 1/len(datasets)

        inTopKRanks = transformToRanks(dataset['Tuples'], dataset['Functions'][topkFunc], 6, t)
        maxRankInTopK = inTopKRanks.index(max(inTopKRanks))
        if maxRankInTopK in inTopKMaxShapley:
            inTopKScoreRank = inTopKScoreRank + 1/len(datasets)


        maxApproxInTopK = result['InTopK']['Approximate']['ShapleyValues'].index(max(result['InTopK']['Approximate']['ShapleyValues']))
        if maxApproxInTopK in inTopKMaxShapley:
            inTopKScoreApprox = inTopKScoreApprox + 1/len(datasets)


        #---------------------------------------------------------------------#

        notInTopKMaxShapley = [x for x in range(len(result['NotInTopK']['BruteForce']['ShapleyValues'])) if abs(result['NotInTopK']['BruteForce']['ShapleyValues'][x] - max(result['NotInTopK']['BruteForce']['ShapleyValues'])) < .00001]

        maxWeightNotInTopK = dataset['Weights'][borderlineFunc].index(max(dataset['Weights'][borderlineFunc]))
        if maxWeightNotInTopK in notInTopKMaxShapley:
            notInTopKScoreWeights = notInTopKScoreWeights + 1/len(datasets)

        notInTopKRanks = transformToRanks(dataset['Tuples'], dataset['Functions'][borderlineFunc], 6, t)
        maxRankNotInTopK = notInTopKRanks.index(max(notInTopKRanks))
        if maxRankNotInTopK in notInTopKMaxShapley:
            notInTopKScoreRank = notInTopKScoreRank + 1/len(datasets)


        maxApproxNotInTopK = result['NotInTopK']['Approximate']['ShapleyValues'].index(max(result['NotInTopK']['Approximate']['ShapleyValues']))
        if maxApproxNotInTopK in notInTopKMaxShapley:
            notInTopKScoreApprox = notInTopKScoreApprox + 1/len(datasets)


        #---------------------------------------------------------------------#

        thisTopKKMaxShapley = [x for x in range(len(result['WhyThisTopK']['BruteForce']['ShapleyValues'])) if abs(result['WhyThisTopK']['BruteForce']['ShapleyValues'][x] - max(result['WhyThisTopK']['BruteForce']['ShapleyValues'])) < .00001]

        maxWeightThisTopK = dataset['Weights'][t].index(max(dataset['Weights'][t]))
        if maxWeightThisTopK in thisTopKKMaxShapley:
            whyThisTopKScoreWeights = whyThisTopKScoreWeights + 1/len(datasets)

        thisTopJaccard = transformToJaccards(dataset['Tuples'], dataset['Functions'][t], 6, t, topK)
        maxJaccardThisTopK = thisTopJaccard.index(max(thisTopJaccard))
        if maxJaccardThisTopK in thisTopKKMaxShapley:
            whyThisTopKScoreJaccard = whyThisTopKScoreJaccard + 1/len(datasets)

        maxApproxThisTopK = result['WhyThisTopK']['Approximate']['ShapleyValues'].index(max(result['WhyThisTopK']['Approximate']['ShapleyValues']))
        if maxApproxThisTopK in thisTopKKMaxShapley:
            whyThisTopKScoreApprox = whyThisTopKScoreApprox + 1/len(datasets)


        #---------------------------------------------------------------------#

        theseTopKMaxShapley = [x for x in range(len(result['WhyInTheseTopKs']['BruteForce']['ShapleyValues'])) if abs(result['WhyInTheseTopKs']['BruteForce']['ShapleyValues'][x] - max(result['WhyInTheseTopKs']['BruteForce']['ShapleyValues'])) < .00001]

        theseTopKJaccards = transformToJaccards2(dataset['Tuples'], dataset['Functions'], 6, t, k, theseTopKs)
        maxTheseTopKJaccards = theseTopKJaccards.index(max(theseTopKJaccards))
        if maxTheseTopKJaccards in theseTopKMaxShapley:
            whyInTheseTopKsScoreJaccard = whyInTheseTopKsScoreJaccard + 1/len(datasets)

        maxApproxTheseTopKs = result['WhyInTheseTopKs']['Approximate']['ShapleyValues'].index(max(result['WhyInTheseTopKs']['Approximate']['ShapleyValues']))
        if maxApproxTheseTopKs in theseTopKMaxShapley:
            whyInTheseTopKsScoreApprox = whyInTheseTopKsScoreApprox + 1/len(datasets)

        whyTheseTopKMaxShapley.append(theseTopKMaxShapley)




    return [('Why In Top K Score: ', ('Approx', inTopKScoreApprox), ('Weight', inTopKScoreWeights), ('Max Rank', inTopKScoreRank)),
               ('Why Not In Top K Score: ', ('Approx', notInTopKScoreApprox), ('Weight', notInTopKScoreWeights), ('Min Rank', notInTopKScoreRank)),
               ('Why This Top K Score', ('Approx', whyThisTopKScoreApprox), ('Weight', whyThisTopKScoreWeights), ('Jaccard', whyThisTopKScoreJaccard)),
               ('Why In These Top Ks Score', ('Approx', whyInTheseTopKsScoreApprox), ('Jaccard', whyInTheseTopKsScoreJaccard))]

def top_attribute_experiment(experiments):
    funcsFile = dill.load(open('Removing-Functions-Linear.dill', 'rb'))
    functions = funcsFile['Functions']
    weights = funcsFile['Weights']
    datasets = []
    if 'AUL' in experiments:
        for tuples in dill.load(open('data/a_u_100_6_9.dill', 'rb')):
            dataset = {}
            dataset['Tuples'] = tuples
            dataset['Functions'] = functions[:100]
            dataset['Weights'] = weights[:100]
            weights = weights[100:]
            functions = functions[100:]
            datasets.append(dataset)
        dill.dump(calculate_top_attribute_scores(datasets, 5, None), open('top_attribute_results_a_u_l.dill', 'wb'))

    datasets = []
    if 'IUL' in experiments:
        for tuples in dill.load(open('data/i_u_100_6_9.dill', 'rb')):
            dataset = {}
            dataset['Tuples'] = tuples
            dataset['Functions'] = functions[:100]
            dataset['Weights'] = weights[:100]
            weights = weights[100:]
            functions = functions[100:]
            datasets.append(dataset)
        dill.dump(calculate_top_attribute_scores(datasets, 5, None), open('top_attribute_results_i_u_l.dill', 'wb'))
    datasets = []
    if 'CUL' in experiments:
        for tuples in dill.load(open('data/c_u_100_6_9.dill', 'rb')):
            dataset = {}
            dataset['Tuples'] = tuples
            dataset['Functions'] = functions[:100]
            dataset['Weights'] = weights[:100]
            weights = weights[100:]
            functions = functions[100:]
            datasets.append(dataset)
        dill.dump(calculate_top_attribute_scores(datasets, 5, None), open('top_attribute_results_c_u_l.dill', 'wb'))
    datasets = []
    if 'AZL' in experiments:
        for tuples in dill.load(open('data/a_z_100_6_9.dill', 'rb')):
            dataset = {}
            dataset['Tuples'] = tuples
            dataset['Functions'] = functions[:100]
            dataset['Weights'] = weights[:100]
            weights = weights[100:]
            functions = functions[100:]
            datasets.append(dataset)
        dill.dump(calculate_top_attribute_scores(datasets, 5, None), open('top_attribute_results_a_z_l.dill', 'wb'))
    datasets = []
    if 'IZL' in experiments:
        for tuples in dill.load(open('data/i_z_100_6_9.dill', 'rb')):
            dataset = {}
            dataset['Tuples'] = tuples
            dataset['Functions'] = functions[:100]
            dataset['Weights'] = weights[:100]
            weights = weights[100:]
            functions = functions[100:]
            datasets.append(dataset)
        dill.dump(calculate_top_attribute_scores(datasets, 5, None), open('top_attribute_results_i_z_l.dill', 'wb'))
    datasets = []
    if 'CZL' in experiments:
        for tuples in dill.load(open('data/c_z_100_6_9.dill', 'rb')):
            dataset = {}
            dataset['Tuples'] = tuples
            dataset['Functions'] = functions[:100]
            dataset['Weights'] = weights[:100]
            weights = weights[100:]
            functions = functions[100:]
            datasets.append(dataset)
        dill.dump(calculate_top_attribute_scores(datasets, 5, None), open('top_attribute_results_c_z_l.dill', 'wb'))
    datasets = []
    funcsFile2 = dill.load(open('Removing-Functions-Nonlinear.dill', 'rb'))
    functions = funcsFile2['Functions']
    weights = funcsFile2['Weights']
    if 'AUNL' in experiments:
        for tuples in dill.load(open('data/a_u_100_6_9.dill', 'rb')):
            dataset = {}
            dataset['Tuples'] = tuples
            dataset['Functions'] = functions[:100]
            dataset['Weights'] = weights[:100]
            weights = weights[100:]
            functions = functions[100:]
            datasets.append(dataset)
        dill.dump(calculate_top_attribute_scores(datasets, 5, None), open('top_attribute_results_a_u_nl.dill', 'wb'))
    datasets = []
    if 'IUNL' in experiments:
        for tuples in dill.load(open('data/i_u_100_6_9.dill', 'rb')):
            dataset = {}
            dataset['Tuples'] = tuples
            dataset['Functions'] = functions[:100]
            dataset['Weights'] = weights[:100]
            weights = weights[100:]
            functions = functions[100:]
            datasets.append(dataset)
        dill.dump(calculate_top_attribute_scores(datasets, 5, None), open('top_attribute_results_i_u_nl.dill', 'wb'))
    datasets = []
    if 'CUNL' in experiments:
        for tuples in dill.load(open('data/c_u_100_6_9.dill', 'rb')):
            dataset = {}
            dataset['Tuples'] = tuples
            dataset['Functions'] = functions[:100]
            dataset['Weights'] = weights[:100]
            weights = weights[100:]
            functions = functions[100:]
            datasets.append(dataset)
        dill.dump(calculate_top_attribute_scores(datasets, 5, None), open('top_attribute_results_c_u_nl.dill', 'wb'))
    datasets = []
    if 'AZNL' in experiments:
        for tuples in dill.load(open('data/a_z_100_6_9.dill', 'rb')):
            dataset = {}
            dataset['Tuples'] = tuples
            dataset['Functions'] = functions[:100]
            dataset['Weights'] = weights[:100]
            weights = weights[100:]
            functions = functions[100:]
            datasets.append(dataset)
        dill.dump(calculate_top_attribute_scores(datasets, 5, None), open('top_attribute_results_a_z_nl.dill', 'wb'))
    datasets = []
    if 'IZNL' in experiments:
        for tuples in dill.load(open('data/i_z_100_6_9.dill', 'rb')):
            dataset = {}
            dataset['Tuples'] = tuples
            dataset['Functions'] = functions[:100]
            dataset['Weights'] = weights[:100]
            weights = weights[100:]
            functions = functions[100:]
            datasets.append(dataset)
        dill.dump(calculate_top_attribute_scores(datasets, 5, None), open('top_attribute_results_i_z_nl.dill', 'wb'))
    datasets = []
    if 'CZNL' in experiments:
        for tuples in dill.load(open('data/c_z_100_6_9.dill', 'rb')):
            dataset = {}
            dataset['Tuples'] = tuples
            dataset['Functions'] = functions[:100]
            dataset['Weights'] = weights[:100]
            weights = weights[100:]
            functions = functions[100:]
            datasets.append(dataset)
        dill.dump(calculate_top_attribute_scores(datasets, 5, None), open('top_attribute_results_c_z_nl.dill', 'wb'))


if __name__ == "__main__":
    top_attribute_experiment(sys.argv[1:])