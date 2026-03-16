import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve

from const import AGEDB_RESULTS_PATH, PREDICTION_CSV

def evaluate(prediction_df: pd.DataFrame):
    y_true = prediction_df["label"]
    y_pred = prediction_df["prediction"]
    dist = -prediction_df["distance"]

    tn, fp, fn, tp = confusion_matrix(y_pred, y_true).ravel()

    far = fp / (fp + tn)
    frr = fn / (fn + tp)
    auc = roc_auc_score(y_true, dist)

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