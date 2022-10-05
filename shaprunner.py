import shap
from model import ModelGenerator
import dill
from varying_m_experiment import findQueryPointOld

datasets = dill.load(open('data/a_z_l_2_varying_d.dill', 'rb'))
d = 9
k = 5

t, topkFunc, borderlineFunc = findQueryPointOld(datasets[d][0], k, datasets[d][1], d, None)

model = ModelGenerator()
model.database(datasets[9][0]).eval_func(datasets[9][1][topkFunc]).k(k).target(t)

explainer = shap.Explainer(model.in_top_k, [0]*9)

print(explainer([1]*9))