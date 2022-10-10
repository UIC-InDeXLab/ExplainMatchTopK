import concurrent.futures
import experiments
import dill
import varying_m_experiment
import sys
from model import ModelGenerator

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
    shapSkipFutureTopK = False
    shapSkipFutureNotTopK = False
    shapSkipFutureWhyThisTopK = False
    shapSkipFutureWhyTheseTopKs = False

    for index in sorted(datasets.keys()):
        tuples, functions, reverseTuples, reverseFunctions = datasets[index]

        d = index
        t, topkFunc, borderlineFunc = varying_m_experiment.findQueryPoint(tuples, k, functions, d, unWrapFunction, minim, maxim)

        results = {}

        results['Query Points'] = (t, topkFunc, borderlineFunc)

        inTopKResults = {}
        notInTopKResults = {}
        whyThisTopKResults = {}
        whyInTheseTopKResults = {}

        inTopKModel = ModelGenerator()
        inTopKModel.database(tuples).eval_func(functions[topkFunc]).k(k).target(t)

        notInTopKModel = ModelGenerator()
        notInTopKModel.database(tuples).eval_func(functions[borderlineFunc]).k(k).target(t)

        whyThisTopKModel = ModelGenerator()
        whyThisTopKModel.database(reverseTuples).eval_func(reverseFunctions[t]).k(k).setup_top_k()

        whyInTheseTopKModel = ModelGenerator()
        whyInTheseTopKModel.database(tuples).eval_funcs(functions).k(k).target(t).setup_top_ks()

        if not skipFutureTopK:
            try:
                inTopKResults['BruteForce'] = executor.submit(experiments.bruteForceInTopK, tuples, functions[topkFunc], k, t, d, unWrapFunction).result(timeout=3600)
            except concurrent.futures.TimeoutError:
                inTopKResults['BruteForce'] = 'Too long!'
                skipFutureTopK = True
        else:
            inTopKResults['BruteForce'] = 'Too long!'

        if not skipFutureNotTopK:
            try:
                notInTopKResults['BruteForce'] = executor.submit(experiments.bruteForceNotInTopK, tuples, functions[borderlineFunc], k, t, d, unWrapFunction).result(timeout=3600)
            except concurrent.futures.TimeoutError:
                notInTopKResults['BruteForce'] = 'Too long!'
                skipFutureNotTopK = True
        else:
            notInTopKResults['BruteForce'] = 'Too long!'

        if not skipFutureWhyThisTopK:
            try:
                whyThisTopKResults['BruteForce'] = executor.submit(experiments.bruteForceWhyThisTopK, reverseTuples, reverseFunctions[t], k, d, unWrapFunction).result(timeout=3600)
            except concurrent.futures.TimeoutError:
                whyThisTopKResults['BruteForce'] = 'Too long!'
                skipFutureWhyThisTopK = True
        else:
            whyThisTopKResults['BruteForce'] = 'Too long!'


        if not skipFutureWhyTheseTopKs:
            try:
                whyInTheseTopKResults['BruteForce'] = executor.submit(experiments.bruteForceWhyInTheseTopK, tuples, functions, k, t, d, unWrapFunction).result(timeout=3600)
            except concurrent.futures.TimeoutError:
                whyInTheseTopKResults['BruteForce'] = 'Too long!'
                skipFutureWhyTheseTopKs = True
        else:
            whyInTheseTopKResults['BruteForce'] = 'Too long!'

        if not apprxSkipFutureTopK:
            try:
                inTopKResults['Approximate'] = executor.submit(experiments.approximateInTopK, tuples, functions[topkFunc], 100, k, t, d, inTopKResults['BruteForce']['ShapleyValues'] if type(inTopKResults['BruteForce']) is dict else [0.0 for x in range(d)], unWrapFunction).result(timeout=3600)
            except concurrent.futures.TimeoutError:
                inTopKResults['Approximate'] = 'Too long!'
                apprxSkipFutureTopK = True
        else:
            inTopKResults['Approximate'] = 'Too long!'

        if not shapSkipFutureTopK:
            try:
                inTopKResults['SHAP'] = executor.submit(experiments.shapInTopK, inTopKModel, d, 100*d, inTopKResults['BruteForce']['ShapleyValues']).result(timeout=3600)
            except concurrent.futures.TimeoutError:
                inTopKResults['SHAP'] = 'Too long!'
                apprxSkipFutureTopK = True
        else:
            inTopKResults['SHAP'] = 'Too long!'

        if not apprxSkipFutureNotTopK:
            try:
                notInTopKResults['Approximate'] = executor.submit(experiments.approximateInTopK, tuples, functions[borderlineFunc], 100, k, t, d, notInTopKResults['BruteForce']['ShapleyValues'] if type(notInTopKResults['BruteForce']) is dict else [0.0 for x in range(d)], unWrapFunction).result(timeout=3600)
            except concurrent.futures.TimeoutError:
                notInTopKResults['Approximate'] = 'Too long!'
                apprxSkipFutureNotTopK = True
        else:
            notInTopKResults['Approximate'] = 'Too long!'

        if not shapSkipFutureNotTopK:
            try:
                notInTopKResults['SHAP'] = executor.submit(experiments.shapNotInTopK, notInTopKModel, d, 100*d, notInTopKResults['BruteForce']['ShapleyValues']).result(timeout=3600)
            except concurrent.futures.TimeoutError:
                notInTopKResults['SHAP'] = 'Too long!'
                apprxSkipFutureTopK = True
        else:
            notInTopKResults['SHAP'] = 'Too long!'

        if not apprxSkipFutureWhyThisTopK:
            try:
                whyThisTopKResults['Approximate'] = executor.submit(experiments.approximateWhyThisTopK, reverseTuples, reverseFunctions[t], 100, k, d, whyThisTopKResults['BruteForce']['ShapleyValues'] if type(whyThisTopKResults['BruteForce']) is dict else [0.0 for x in range(d)], unWrapFunction).result(timeout=3600)
            except concurrent.futures.TimeoutError:
                whyThisTopKResults['Approximate'] = 'Too long!'
                apprxSkipFutureWhyThisTopK = True
        else:
            whyThisTopKResults['Approximate'] = 'Too long!'

        if not shapSkipFutureWhyThisTopK:
            try:
                whyThisTopKResults['SHAP'] = executor.submit(experiments.shapWhyThisTopK, whyThisTopKModel, d, 100*d, whyThisTopKResults['BruteForce']['ShapleyValues']).result(timeout=3600)
            except concurrent.futures.TimeoutError:
                whyThisTopKResults['SHAP'] = 'Too long!'
                apprxSkipFutureTopK = True
        else:
            whyThisTopKResults['SHAP'] = 'Too long!'

        if not apprxSkipFutureWhyTheseTopKs:
            try:
                whyInTheseTopKResults['Approximate'] = executor.submit(experiments.approximateWhyInTheseTopK, tuples, functions, 100, k, t, d, whyInTheseTopKResults['BruteForce']['ShapleyValues'] if type(whyInTheseTopKResults['BruteForce']) is dict else [0.0 for x in range(d)], unWrapFunction).result(timeout=3600)
            except concurrent.futures.TimeoutError:
                whyInTheseTopKResults['Approximate'] = 'Too long!'
                apprxSkipFutureWhyTheseTopKs = True
        else:
            whyInTheseTopKResults['Approximate'] = 'Too long!'

        if not shapSkipFutureWhyTheseTopKs:
            try:
                whyInTheseTopKResults['SHAP'] = executor.submit(experiments.shapWhyTheseTopKs, whyInTheseTopKModel, d, 100*d, whyInTheseTopKResults['BruteForce']['ShapleyValues']).result(timeout=3600)
            except concurrent.futures.TimeoutError:
                whyInTheseTopKResults['SHAP'] = 'Too long!'
                apprxSkipFutureTopK = True
        else:
            whyInTheseTopKResults['SHAP'] = 'Too long!'

        results['InTopK'] = inTopKResults
        results['NotInTopK'] = notInTopKResults
        results['WhyThisTopK'] = whyThisTopKResults
        results['WhyInTheseTopKs'] = whyInTheseTopKResults
        resultsFinal[index] = results

    return resultsFinal


def runExperiments(settings):
    if 'AZL' in settings:
        dill.dump(varyingDExperiment(dill.load(open('data/a_z_l_varying_d.dill', 'rb')), None, 3, 6, 5), open('SyntheticDAZL-Clean.dill', 'wb'))
    if 'CZL' in settings:
        dill.dump(varyingDExperiment(dill.load(open('data/c_z_l_varying_d.dill', 'rb')), None, 3, 6, 5), open('SyntheticDCZL-Clean.dill', 'wb'))
    if 'IZL' in settings:
        dill.dump(varyingDExperiment(dill.load(open('data/i_z_l_varying_d.dill', 'rb')), None, 3, 6, 5), open('SyntheticDIZL-Clean.dill', 'wb'))
    if 'AZNL' in settings:
        dill.dump(varyingDExperiment(dill.load(open('data/a_z_nl_varying_d.dill', 'rb')), None, 3, 6, 5), open('SyntheticDAZNL-Clean.dill', 'wb'))
    if 'CZNL' in settings:
        dill.dump(varyingDExperiment(dill.load(open('data/c_z_nl_varying_d.dill', 'rb')), None, 3, 6, 5), open('SyntheticDCZNL-Clean.dill', 'wb'))
    if 'IZNL' in settings:
        dill.dump(varyingDExperiment(dill.load(open('data/i_z_nl_varying_d.dill', 'rb')), None, 3, 6, 5), open('SyntheticDIZNL-Clean.dill', 'wb'))


if __name__ == '__main__':
    runExperiments(sys.argv[1:])