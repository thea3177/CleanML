from schema.autosklearn_model.AutoSklearnModel import AutoSklearnModel
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
import numpy as np
from new.utils import get_fold_ids
from new.utils import eval
from new.utils import run

if __name__ == "__main__":
    target_label = 'Eye'
    dirty_path = '/home/neutatz/Software/CleanML/data/EEG/raw/raw.csv'
    clean_path = '/home/neutatz/Software/CleanML/data/EEG/raw/Holoclean_outlier_clean.csv'

    run(clean_path, dirty_path, target_label)


