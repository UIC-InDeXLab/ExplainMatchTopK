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
    shap_model = shap_bipartite(model.top_k_look_like_this, D, N, samples)
    shap_values = shap_model.solve_lin_alg()
    end_time = time.time()
    elapsed_time = end_time - start_time

    return {'time': elapsed_time, 'SHAP': shap_values}

if __name__ == "__main__":
    k=5
    d=9
    samples_set =[25,50,75,100,125,150,175,200,225,250]
    datasets = dill.load(open('data/a_z_l_2_varying_d.dill', 'rb'))
    # AZL = varyingMExperiment(datasets[9][0], datasets[9][1], datasets[9][2], datasets[9][3], 9, None)
    results = []
    t, topkFunc, borderlineFunc = findQueryPointOld(datasets[d][0], k, datasets[d][1], d, None)
    print((t, topkFunc, borderlineFunc))
    for samples in samples_set:
        results.append(varying_samples_SHAP_InTopK(datasets[d][2], datasets[d][3][t], k, t, samples))
    dill.dump(results, open('SHAP-AZL.dill', 'wb'))
    #print(results)
