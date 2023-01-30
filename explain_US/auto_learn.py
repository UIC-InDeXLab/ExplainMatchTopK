import pandas as pd
import lime
import lime.lime_tabular
import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn
import pdb
import os
from sklearn.preprocessing import MinMaxScaler
import autosklearn.regression
from sklearn.model_selection import train_test_split
import shutil

total_time = 90
dataset = None
minmax_scaler = None
def get_model(csv_file, cols):
  train, test = train_test_split(csv_file, test_size=0.2)
  
  train_X = train[cols[:-1]]
  train_y = train[cols[-1]]
  test_X = test[cols[:-1]]
  test_y = test[cols[-1]]
  
  dirpath = "temp"
  if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)
  automl = autosklearn.regression.AutoSklearnRegressor(
      time_left_for_this_task=30,
      per_run_time_limit=15,
      tmp_folder=dirpath,
  )
  automl.fit(train_X, train_y, dataset_name="Admissions")
  train_predictions = automl.predict(train_X)
  print("Train R2 score:", sklearn.metrics.r2_score(train_y, train_predictions))
  test_predictions = automl.predict(test_X)
  print("Test R2 score:", sklearn.metrics.r2_score(test_y, test_predictions))
  from pprint import pprint
  pprint(automl.show_models(), indent=4)
  return automl

def get_classifier(k):
  file_name = "Admission.csv"
  csv_file = pd.read_csv(file_name)
  csv_file = csv_file.drop(['Serial No.'], axis=1)
  cols = list(csv_file.columns)
  csv_X = csv_file[cols[:-1]]
  csv_y = csv_file[cols[-1]]
  global minmax_scaler
  minmax_scaler = MinMaxScaler()
  
  df_scaled = minmax_scaler.fit_transform(csv_file.to_numpy())
  # Chamce to predict needs to be scaled back as it is a regression value
  #df_scaled [:, -1] = df_scaled[:, -1]*0.97
  csv_file = pd.DataFrame(df_scaled, columns=cols)#list(csv_file.columns))
  automl = get_model(csv_file, cols)
  vals = automl.predict(csv_file[cols[:-1]])
  s = sorted(list(vals), reverse=True)
  k_val = s[k]
  global dataset
  dataset = csv_file
  return lambda x: np.array(automl.predict(x) >= k_val).astype(int)

k=10
classifier = get_classifier(k)
new_pred = np.array(classifier(dataset[dataset.columns[:-1]])).astype(int)
X_dataset = dataset[dataset.columns[:-1]]
pdb.set_trace()
explainer = lime.lime_tabular.LimeTabularExplainer(dataset, feature_names=dataset.columns[:-1], class_names=[dataset.columns[-1]], verbose=True, mode='regression')
print(explainer.explain(X_dataset[0], classifier, num_features=len(dataset.columns[:-1])))
