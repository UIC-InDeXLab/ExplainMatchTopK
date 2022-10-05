import shap
from model import ModelGenerator
import dill
from varying_m_experiment import findQueryPointOld
import numpy as np

datasets = dill.load(open('data/a_z_l_2_varying_d.dill', 'rb'))
d = 9
k = 5

t, topkFunc, borderlineFunc = findQueryPointOld(datasets[d][0], k, datasets[d][1], d, None)

model = ModelGenerator()
model.database(datasets[9][0]).eval_func(datasets[9][1][topkFunc]).k(k).target(t)


explainer = shap.KernelExplainer(model.in_top_k, np.zeros(9))
shap_values = explainer.shap_values(np.ones(9), nsamples=250)
print("shap_values =", shap_values)
print("base value =", explainer.expected_value)