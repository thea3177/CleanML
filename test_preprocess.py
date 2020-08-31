import pandas as pd
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
from preprocess import preprocess
from preprocess import preprocess_data

dataset = "Titanic"

dirty_X_train = pd.read_csv("./data-robustml-mv/Titanic/missing_values/split_144/dirty_X_train.csv")
dirty_y_train = pd.read_csv("./data-robustml-mv/Titanic/missing_values/split_144/dirty_y_train.csv")
dirty_X_test = pd.read_csv("./data-robustml-mv/Titanic/missing_values/split_144/dirty_X_test.csv")
dirty_y_test = pd.read_csv("./data-robustml-mv/Titanic/missing_values/split_144/dirty_y_test.csv")

# remove missing values
mv_rows = dirty_X_train.isnull().values.any(axis=1)
dirty_X_train = dirty_X_train[mv_rows]
dirty_y_train = dirty_y_train[mv_rows]

mv_rows = dirty_X_test.isnull().values.any(axis=1)
dirty_X_test = dirty_X_test[mv_rows]
dirty_y_test = dirty_y_test[mv_rows]

X_train, y_train, X_test_list, y_test_list = preprocess_data(dataset, dirty_X_train, dirty_y_train, [dirty_X_test], [dirty_y_test])


