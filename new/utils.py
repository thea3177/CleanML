from schema.autosklearn_model.AutoSklearnModel import AutoSklearnModel
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
import numpy as np

def get_X_y(data, target_label, drop_labels=[]):
    data_y = data[target_label]
    data_X = data.drop(target_label, 1)
    for drop_label in drop_labels:
        data_X = data.drop(drop_label, 1)
    return data_y, data_X

def eval(data, target_label, fold_ids, drop_labels=[], feat_type=None):
    data_y, data_X = get_X_y(data, target_label, drop_labels)

    if type(feat_type) != type(None):
        for ci in range(len(feat_type)):
            if feat_type[ci] == 'Categorical':
                data_X[data_X.columns[ci]] = data_X[data_X.columns[ci]].astype('category')



    scores = []
    for train_index, test_index in fold_ids:
        model= AutoSklearnModel()
        model.fit(X=data_X.iloc[train_index, :], y=data_y.values[train_index], feat_type=feat_type)
        y_pred = model.predict(data_X.iloc[test_index])
        scores.append(balanced_accuracy_score(data_y[test_index], y_pred))
        print(scores)
    return scores

def get_fold_ids(data, target_label, drop_labels=[]):
    data_y, data_X = get_X_y(data, target_label, drop_labels)

    skf = StratifiedKFold(n_splits=5)
    fold_ids = list(skf.split(data_X, data_y))
    return fold_ids

def get_feat_type(data, target_label, drop_labels=[]):
    data_y, data_X = get_X_y(data, target_label, drop_labels)
    feat_type = [
        'Categorical' if str(x) == 'object' else 'Numerical'
        for x in data_X.dtypes
    ]
    return feat_type

def run(clean_path, dirty_path, target_label, drop_labels=[]):
    holoclean_train = pd.read_csv(clean_path)
    dirty_train = pd.read_csv(dirty_path)
    assert len(holoclean_train) == len(dirty_train)

    fold_ids = get_fold_ids(holoclean_train, target_label, drop_labels)
    feat_type = get_feat_type(holoclean_train, target_label, drop_labels)
    clean_scores = eval(holoclean_train, target_label, fold_ids, drop_labels, feat_type)
    dirty_scores = eval(dirty_train, target_label, fold_ids, drop_labels, feat_type)

    print('number of errors: ' + str(np.sum(holoclean_train != dirty_train)))
    print('dirty scores: ' + str(dirty_scores))
    print('clean scores: ' + str(clean_scores))
