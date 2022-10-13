import dill
import experiments
import topk

def brute_force_experiment(tuples, functions, reverseTuples, reverseFunctions, d, k):
    results = {}

    inTopKResults = {}
    notInTopKResults = {}
    whyThisTopKResults = {}
    whyInTheseTopKResults = {}

    inTopKResults['BruteForce'] = experiments.bruteForceInTopK(tuples, functions[8], k, 2, d, None)
    notInTopKResults['BruteForce'] = experiments.bruteForceNotInTopK(tuples, functions[9], k, 2, d, None)
    whyThisTopKResults['BruteForce'] = experiments.bruteForceWhyThisTopK(reverseTuples, reverseFunctions[2], k, d, None)
    whyInTheseTopKResults['BruteForce'] = experiments.bruteForceWhyInTheseTopK(tuples, functions, k, 2, d, None)

    results['InTopK'] = inTopKResults
    results['NotInTopK'] = notInTopKResults
    results['WhyThisTopK'] = whyThisTopKResults
    results['WhyInTheseTopKs'] = whyInTheseTopKResults

    return results


def running_example_tables():
    dataset = dill.load(open('RE-Dataset.dill', 'rb'))
    functions = generate_functions(dataset)

    dill.dump(
        brute_force_experiment(dataset['Candidates'], functions['HRs'], dataset['HRs'], functions['Candidates'], 4, 2),
        open('Running-Example-Tables.dill', 'wb'))


def generate_functions(dataset):
    functionWeights = dill.load(open('RE-Function-Weights.dill', 'rb'))

    def generate_scoring_function(data, weights):
        return lambda e: sum([(0 if e[x] is None else (1 - abs(e[x] - data[x])) * weights[x]) for x in range(4)])

    functions = {}
    functions['HRs'] = []
    functions['Candidates'] = []
    for x in range(10):
        functions['HRs'].append(generate_scoring_function(dataset['HRs'][x], functionWeights['HRs'][x]))
        functions['Candidates'].append(
            generate_scoring_function(dataset['Candidates'][x], functionWeights['Candidates'][x]))
    return functions


def running_example_matching():
    dataset = dill.load(open('RE-Dataset.dill', 'rb'))
    functions = generate_functions(dataset)

    topKs = {}
    topKs['Candidates'] = []
    topKs['HRs'] = []

    for x in range(10):
        hrTuples = topk.generateTuples(dataset['HRs'], functions['Candidates'][x], [x for x in range(4)], 4, None)
        topKs['Candidates'].append(topk.computeTopK(hrTuples, 2))
        candidateTuples = topk.generateTuples(dataset['Candidates'], functions['HRs'][x], [x for x in range(4)], 4, None)
        topKs['HRs'].append(topk.computeTopK(candidateTuples, 2))

    dill.dump(topKs, open('Running-Example-Matching.dill', 'wb'))



if __name__ == "__main__":
    running_example_tables()
    running_example_matching()