import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

file_name = "Admission.csv"
csv_file = pd.read_csv(file_name)
csv_file = csv_file.drop(['Serial No.'], axis=1)
csv_file = csv_file.sample(frac=1)
train = csv_file[:400]
test = csv_file[400:]
cols = list(train.columns)

train_X = train[cols[:-1]]
train_y = train[cols[-1]]
test_X = test[cols[:-1]]
test_y = test[cols[-1]]

reg = HistGradientBoostingRegressor(monotonic_cst=[1]*7).fit(train_X, train_y)
print("Train score : {}".format(reg.score(train_X, train_y)))
#pred_y = reg.predict(test_X)
#print(1-np.linalg.norm(pred_y-test_y, ord=2))
print("Test score : {}".format(reg.score(test_X, test_y)))
