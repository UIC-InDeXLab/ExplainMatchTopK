import pdb
import bruteforce
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn
from model import ModelGenerator
import pickle
import shutil
import os
from sklearn.preprocessing import MinMaxScaler

file_name = "explain_US/Admission.csv"
csv_file = pd.read_csv(file_name)
csv_file = csv_file.drop(['Serial No.'], axis=1)
cols = list(csv_file.columns)
"""
train = csv_file[:400]
test = csv_file[400:]

train_X = train[cols[:-1]]
train_y = train[cols[-1]]
test_X = test[cols[:-1]]
test_y = test[cols[-1]]
"""
np_array = np.array(csv_file)
X_array = np_array[:, range(np_array.shape[1]-1)]
import autosklearn.regression

filename = "automl_admissions.pickle"
filepath = os.path.join('.', filename)
if os.path.exists(filepath):
    with open(filename, "rb") as input_file:
        automl_dict = pickle.load(input_file)
        automl = automl_dict["automl"]
        minmax_scaler = automl_dict["minmax_scaler"]
else:
    dirpath = os.path.join('.', 'automl_admissions.pickle')
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    minmax_scaler = MinMaxScaler()

    df_scaled = minmax_scaler.fit_transform(csv_file.to_numpy()[:, :-1])
    # Chance to predict needs to be scaled back as it is a regression value
    df_scaled  = np.c_[df_scaled, csv_file.to_numpy()[:, -1]]
    csv_file = pd.DataFrame(df_scaled, columns=list(csv_file.columns))
    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=300,
        per_run_time_limit=20,
        tmp_folder="temp",
    )
    train, test = train_test_split(csv_file, test_size=0.2)

    train_X = train[cols[:-1]]
    train_y = train[cols[-1]]
    test_X = test[cols[:-1]]
    test_y = test[cols[-1]]

    train_X_np = np.array(train_X)
    train_y_np = np.array(train_y)
    automl.fit(train_X_np, train_y_np, dataset_name="Admissions")
    train_predictions = automl.predict(train_X_np)
    print("Train R2 score:", sklearn.metrics.r2_score(train_y_np, train_predictions))
    test_predictions = automl.predict(np.array(test_X))
    print("Test R2 score:", sklearn.metrics.r2_score(np.array(test_y), test_predictions))
    from pprint import pprint
    pprint(automl.show_models(), indent=4)
    with open(filename, "wb") as output_file:
        pickle.dump({"automl": automl, "minmax_scaler": minmax_scaler}, output_file)
from pprint import pprint
pprint(automl.show_models(), indent=4)
#ComputeShapleyNotInTopK(vectors, evaluationFunction, k, j, d, unWrapFunction, algorithm='GeneralPurpose'):
#print(bruteforce.ComputeShapleyNotInTopK(X_array.tolist(), automl.predict, k, j, d, None))
k = 10
j = 59 # Item to check the Shapley value for
d = np_array.shape[1]-1 # dimensions
model = ModelGenerator()
def predict_func(x):
    pdb.set_trace()
    x_transform = []
    for i in x:
        if i is not None:
            x_transform.append(i)
        else:
            x_transform(0)
    x_transform = np.array(x_transform).reshape(1 , -1)
    x_scaled = minmax_scaler.transform(x_transform)
    score = automl.predict(x_scaled)
    return score
#predict_func = lambda x : automl.predict([i if i is not None else 0 for i in x])
pdb.set_trace()
model.database(X_array.tolist()).eval_func(predict_func).k(k).target(j).setup_top_k()




def test():
    database = [[5, 3, 1], [2, 4, 4], [3, 1, 2], [4, 1, 3], [1, 2, 5]]
    weights = [5, 4, 1]
    evaluation_function = lambda e: (sum([(weights[x] * e[x]) if e[x] is not None else 0 for x in range(len(weights))]))

    model = ModelGenerator()
    model.database(database).eval_func(evaluation_function).k(2).target(0).setup_top_k()

    masks = [[0,0,1],[0,1,0],[1,0,0],[0,1,1],[1,0,1],[1,1,0]]

    results_in_top_k = model.in_top_k(masks)

    assert(np.all(results_in_top_k == np.array([0, 1, 1, 1, 1, 1])))

    results_not_in_top_k = model.not_in_top_k(masks)

    assert(np.all(results_not_in_top_k == np.array([1, 0, 0, 0, 0, 0])))

    top_k_look_like_this = model.top_k_look_like_this(masks)

    assert(np.all(top_k_look_like_this == [1/3, 1/1, 1/3, 1/1, 1/3, 1/1]))

    model = ModelGenerator()

    weights1 = [80, 90, 4]
    weights2 = [90, 10, 5]
    weights3 = [50, 60, 10]

    evalFuncs = [lambda e:(sum([(weights1[x]*e[x]) if e[x] is not None else 0 for x in range(len(weights))])),
                 lambda e:(sum([(weights2[x]*e[x]) if e[x] is not None else 0 for x in range(len(weights))])),
                 lambda e:(sum([(weights3[x]*e[x]) if e[x] is not None else 0 for x in range(len(weights))]))]

    model.database(database).eval_funcs(evalFuncs).k(2).target(0).setup_top_ks()

    in_these_top_ks = model.in_these_top_ks(masks)

    assert(np.all(in_these_top_ks == np.array([0/1, 1/1, 1/1, 2/3, 1/1, 1/1])))
    print('All tests passed!')

