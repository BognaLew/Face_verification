import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve

from const import AGEDB_RESULTS_PATH, LFW_RESULTS_PATH, PREDICTION_CSV

def evaluate(prediction_df: pd.DataFrame):
    y_true = prediction_df["label"]
    y_pred = prediction_df["prediction"]
    score = prediction_df["similarity"]
    
    impostors = y_true == 0
    genuines = y_true == 1

    far = ((y_pred == 1) & impostors).sum() / impostors.sum()
    frr  = ((y_pred == 0) & genuines).sum() / genuines.sum()

    auc = roc_auc_score(y_true, score)

    f_score = f1_score(y_true, y_pred)
    
    metrics = {
        "FAR": far,
        "FRR": frr,
        "f1-score": f_score,
        "auc": auc
    }
    
    return metrics


if __name__=="__main__":
    predictions = pd.read_csv(os.path.join(AGEDB_RESULTS_PATH, PREDICTION_CSV))
    metrics = evaluate(predictions)
    print(metrics)