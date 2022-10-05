# Varying samples for SHAP
import dill
from varying_m_experiment import findQueryPointOld
from model import ModelGenerator
import time
from shap_bipartite import shap_bipartite

def varying_samples_SHAP_InTopK(database, evaluation_function, k, target, samples):
    model = ModelGenerator()
    model.database(database).eval_func(evaluation_function).k(k).target(target).setup_top_k()

    D = len(database[0])
    N = len(database)

    start_time = time.time()
    shap_model = shap_bipartite(model.in_top_k, D, N, samples)
    shap_values = shap_model.solve()
    end_time = time.time()
    elapsed_time = end_time - start_time

    return {'time': elapsed_time, 'SHAP': shap_values}

if __name__ == "__main":
    k=5
    d=9
    samples=50
    datasets = dill.load(open('data/a_z_l_varying_d.dill', 'rb'))
    # AZL = varyingMExperiment(datasets[9][0], datasets[9][1], datasets[9][2], datasets[9][3], 9, None)
    t, topkFunc, borderlineFunc = findQueryPointOld(datasets[d][0], k, datasets[d][1], d, None)
    print((t, topkFunc, borderlineFunc))
    varying_samples_SHAP_InTopK(datasets[d][0], datasets[d][1][topkFunc], k, t, samples)