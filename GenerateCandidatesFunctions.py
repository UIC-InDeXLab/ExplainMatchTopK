import pickle
import numpy as np

def generateCandidatesFunction():
    values = pickle.load(open('Candidates-Values.pickle', 'rb'))

    overallWeights = np.random.zipf(1.2, 22)
    overallWeights = overallWeights / overallWeights.sum()

    skillsetWeights = np.random.zipf(1.2, len(values['SkillSet']))
    skillsetWeights = skillsetWeights / skillsetWeights.sum()

    degreeWeights = np.random.zipf(1.2, len(values['Degrees']))
    degreeWeights = degreeWeights / degreeWeights.max()

    streamWeights = np.random.zipf(1.2, len(values['Stream']))
    streamWeights = streamWeights / streamWeights.max()

    graduationWeights = np.random.zipf(1.2, len(values['Graduation Years']))
    graduationWeights = graduationWeights / graduationWeights.max()

    cityWeights = np.random.zipf(1.2, len(values['Cities']))
    cityWeights = cityWeights / cityWeights.max()

    skillsetList = list(values['SkillSet'])
    degreeList = list(values['Degrees'])
    streamList = list(values['Stream'])
    yearsList = list(values['Graduation Years'])
    cityList = list(values['Cities'])

    return lambda e: sum([overallWeights[x] * e[x] for x in range(17)]) + sum([skillsetWeights[skillsetList.index(skill)] for skill in e[17]]) * overallWeights[17] + overallWeights[18] * degreeWeights[degreeList.index(e[18])] + overallWeights[19] * streamWeights[streamList.index((e[19]))] + overallWeights[20] * graduationWeights[yearsList.index(e[20])] + overallWeights[21] * cityWeights[cityList.index(e[21])]