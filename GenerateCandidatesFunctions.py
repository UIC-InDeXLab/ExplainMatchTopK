import pickle
import numpy as np


def generateCandidatesFunction(original, coefficient):
    values = pickle.load(open('Candidates-Values.pickle', 'rb'))

    overallWeights = np.random.zipf(coefficient, 22)
    overallWeights = overallWeights / overallWeights.sum()

    skillsetWeights = np.random.zipf(coefficient, len(values['SkillSet']))
    skillsetWeights = skillsetWeights / skillsetWeights.sum()

    degreeWeights = np.random.zipf(coefficient, len(values['Degrees']))
    degreeWeights = degreeWeights / degreeWeights.max()

    streamWeights = np.random.zipf(coefficient, len(values['Stream']))
    streamWeights = streamWeights / streamWeights.max()

    graduationWeights = np.random.zipf(coefficient, len(values['Graduation Years']))
    graduationWeights = graduationWeights / graduationWeights.max()

    cityWeights = np.random.zipf(coefficient, len(values['Cities']))
    cityWeights = cityWeights / cityWeights.max()

    skillsetList = list(values['SkillSet'])
    degreeList = list(values['Degrees'])
    streamList = list(values['Stream'])
    yearsList = list(values['Graduation Years'])
    cityList = list(values['Cities'])

    return (lambda e: sum([(overallWeights[x] * (1 - abs(original[x] - e[x]))) if original[x] is not None and e[x] is not None else 0 for x in range(17)]) +
                      (overallWeights[17] * ((sum([(abs(skillsetWeights[skillsetList.index(skill)] -
                                                      skillsetWeights[skillsetList.index(skill)])) for skill in e[17] if
                                                  skill in original[17]]) /
                                             (sum([abs(skillsetWeights[skillsetList.index(skill)] - skillsetWeights[
                                                 skillsetList.index(skill)])
                                                  for skill in e[17] if skill in original[17]]) +
                                             sum([skillsetWeights[skillsetList.index(skill)] for skill in original[17]
                                                  if
                                                  skill not in e[17]]) +
                                             sum([skillsetWeights[skillsetList.index(skill)] for skill in e[17] if
                                                  skill not in original[17]]))))
                                            if len(e[17]) + len(original[17]) > 0 else 1)
                                                if original[17] is not None and e[17] is not None else 0+
                      (overallWeights[18] * (1 - abs(degreeWeights[degreeList.index(e[18])] -
                                                    degreeWeights[degreeList.index(original[18])])))
                                    if original[18] is not None and e[18] is not None else 0 +
                      (overallWeights[19] * (1 - abs(streamWeights[streamList.index((e[19]))] -
                                                    streamWeights[streamList.index(original[19])])))
                                    if original[19] is not None and e[19] is not None else 0 +
                      (overallWeights[20] * (1 - abs(graduationWeights[yearsList.index(e[20])] -
                                                    graduationWeights[yearsList.index(original[20])])))
                                    if original[20] is not None and e[20] is not None else 0 +
                      (overallWeights[21] * (1 - abs(cityWeights[cityList.index(e[21])] -
                                                    cityWeights[cityList.index(original[21])])))
                                    if original[21] is not None and e[21] is not None else 0,

            overallWeights, skillsetWeights, degreeWeights, streamWeights, graduationWeights, cityWeights)
