import shap
from model import ModelGenerator
import dill
from varying_m_experiment import findQueryPointOld
import numpy as np

datasets = dill.load(open('data/a_z_l_2_varying_d.dill', 'rb'))
d = 9
k = 5

t, topkFunc, borderlineFunc = findQueryPointOld(datasets[d][0], k, datasets[d][1], d, None)
print(t, topkFunc, borderlineFunc)

model = ModelGenerator()

model.database(datasets[9][0]).eval_func(datasets[9][1][topkFunc]).k(k).target(t)

# database = [[5, 3, 1], [2, 4, 4], [3, 1, 2], [4, 1, 3], [1, 2, 5]]
# weights = [5, 4, 1]
# evaluation_function = lambda e: (sum([(weights[x] * e[x]) if e[x] is not None else 0 for x in range(len(weights))]))
# model.database(database).eval_func(evaluation_function).k(2).target(0)

reference = np.zeros(9)
# reference = np.zeros(3)
explainer = shap.KernelExplainer(model.in_top_k, np.reshape(reference, (1, len(reference))))
shap_values = explainer.shap_values(np.ones(9), nsamples=250)
# shap_values = explainer.shap_values(np.ones(3), nsamples = 4)
print("shap_values =", shap_values)
print("expected shap value is ", [0.5, 0.5, 0.0])
print("base value =", explainer.expected_value)