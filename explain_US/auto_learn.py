import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn

file_name = "Admission.csv"
csv_file = pd.read_csv(file_name)
csv_file = csv_file.drop(['Serial No.'], axis=1)
train = csv_file[:400]
test = csv_file[400:]
cols = list(train.columns)

train_X = train[cols[:-1]]
train_y = train[cols[-1]]
test_X = test[cols[:-1]]
test_y = test[cols[-1]]

import autosklearn.regression

automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=300,
    per_run_time_limit=20,
    tmp_folder="temp",
)
automl.fit(train_X, train_y, dataset_name="Admissions")
train_predictions = automl.predict(train_X)
print("Train R2 score:", sklearn.metrics.r2_score(train_y, train_predictions))
test_predictions = automl.predict(test_X)
print("Test R2 score:", sklearn.metrics.r2_score(test_y, test_predictions))
from pprint import pprint
pprint(automl.show_models(), indent=4)
import pdb
pdb.set_trace()

