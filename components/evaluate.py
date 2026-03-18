import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


def evaluate(prediction_df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluates system by calculating FAR, FRR, f1-score and ROC AUC.

    Args:
        prediction_df (pd.DataFrame): DataFrame with prediction data. Should 
            contain:
            - similarity (float): calculated cosine similarity
            - prediction (int): decision of a system
            - label (int): expected decision
    Returns:
        Dict[str, Any]: Dictionary with:
            - FAR (np.float): calculated FAR metric
            - FRR (np.float): calculated FRR metric
            - f1-score (float): calculated f1-score
            - auc (float): calculated ROC AUC
    """
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
        "auc": auc,
    }
    
    return metrics


if __name__=="__main__":
    from constants import AGEDB_RESULTS_PATH, LFW_RESULTS_PATH, PREDICTION_CSV

    predictions = pd.read_csv(os.path.join(AGEDB_RESULTS_PATH, PREDICTION_CSV))
    metrics = evaluate(prediction_df=predictions)
    print(metrics)
