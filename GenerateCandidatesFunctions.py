import pickle
import random

import numpy as np


def generateHRsFunction(org, coefficient):
    values = pickle.load(open('Candidates-Values.pickle', 'rb'))

    overallWeights = np.random.zipf(coefficient, 7)
    overallWeights = overallWeights / overallWeights.sum()

    degreeWeights = np.random.zipf(coefficient, len(values['Degrees']))
    degreeWeights = degreeWeights / degreeWeights.max()

    streamWeights = np.random.zipf(coefficient, len(values['Stream']))
    streamWeights = streamWeights / streamWeights.max()

    intYears = [int(year) for year in list(values['Graduation Years'])]

    skillSum = sum([(x + 1 / 3) if x is not None else 1 / 3 for x in org[:13]])
    skillWeights = [((x + 1 / 3) if x is not None else 1 / 3) / skillSum for x in org[:13]]

    performanceSum = sum([(x + 1 / 3) if x is not None else 1 / 3 for x in org[13:17]])
    performanceWeights = [((x + 1 / 3) if x is not None else 1 / 3) / performanceSum for x in org[13:17]]

    degreeList = list(values['Degrees'])
    streamList = list(values['Stream'])

    return (lambda e, original=org:
            overallWeights[0] * sum([(skillWeights[x] * e[x]) if e[x] is not None else 0 for x in range(13)]) +
            overallWeights[1] * sum(
                [(performanceWeights[x] * e[x + 13]) if e[x] is not None else 0 for x in range(4)]) +
            ((overallWeights[2] * len([skill for skill in original[17] if skill in e[17]]) /
              len(original[17]))
             if original[17] is not None and e[17] is not None else 0) +
            ((overallWeights[3] * degreeWeights[degreeList.index(e[18])])
             if e[18] is not None else 0) +
            ((overallWeights[4] * streamWeights[streamList.index(e[19])])
             if e[19] is not None else 0) +
            ((overallWeights[5] * ((max(intYears)) - int(e[20])) / (max(intYears) - min(intYears)))
             if e[20] is not None else 0) +
            ((overallWeights[6] * (1 if e[21] == original[21] else 0))
             if original[21] is not None and e[21] is not None else 0),

            overallWeights, skillWeights, performanceWeights, degreeWeights, streamWeights)


def generateCandidatesFunction(org, coefficient):
    values = pickle.load(open('Candidates-Values.pickle', 'rb'))

    overallWeights = np.random.zipf(coefficient, 4)
    overallWeights = overallWeights / overallWeights.sum()

    cityWeights = np.random.zipf(coefficient, len(values['Cities']))
    cityWeights = cityWeights / cityWeights.max()

    skillSum = sum([x + 1 if x is not None else 1 for x in org[:13]])
    skillWeights = [((x + 1) if x is not None else 1) / skillSum for x in org[:13]]

    performanceSum = sum([x + 20 if x is not None else 20 for x in org[13:17]])
    performanceWeights = [((x + 20) if x is not None else 20) / performanceSum for x in org[13:17]]

    return (lambda e, original=org:
            overallWeights[0] * sum([((skillWeights[x] * (1 - e[x])) if e[x] is not None else 0) for x in range(13)]) +
            overallWeights[1] * sum(
                [(performanceWeights[x] * (1 - e[x + 13]) if e[x + 13] is not None else 0) for x in range(4)]) +
            ((overallWeights[2] * len([skill for skill in original[17]]) / len(original[17])) if
             (original[17] is not None and e[17] is not None) else 0) +
            ((overallWeights[3] * (1 if e[21] == original[21] else 0))
             if (original[21] is not None and e[21] is not None) else 0),

            overallWeights, skillWeights, performanceWeights, cityWeights)


def generateRunningExampleFunctions(org, coefficient):
    overallWeights = np.random.zipf(coefficient, 4)
    overallWeights = overallWeights / overallWeights.sum()

    return (lambda e, original=org: sum(
        [(overallWeights[x] * (1 - abs(original[x] - e[x]))) if original[x] is not None and e[x] is not None else 0 for
         x in range(4)]), overallWeights)
