from schema.autosklearn_model.AutoSklearnModel import AutoSklearnModel
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
import numpy as np
from new.utils import get_fold_ids
from new.utils import eval
from new.utils import run

from pathlib import Path
import os


if __name__ == "__main__":
    target_label = "genres"
    #drop_labels = 
    dirty_path = "/Users/yejingwen/Documents/priNextloud/Documents/pri/github/CleanML/data/Movie/raw/raw.csv"
    clean_path = "/Users/yejingwen/Documents/priNextloud/Documents/pri/github/CleanML/data/Movie/raw/inconsistency_clean_raw.csv"

    run(clean_path, dirty_path, target_label)
    

