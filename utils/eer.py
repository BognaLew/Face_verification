import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

def calculate_eer(prediction_df: pd.DataFrame, fig_path: str):
    y_true = prediction_df["label"]
    score = prediction_df["similarity"]

    fpr, tpr, thresholds = roc_curve(y_true, score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2
    thr = thresholds[idx]

    plt.figure(figsize=(8,5))
    plt.plot(thresholds, fpr, label='FAR')
    plt.plot(thresholds, fnr, label='FRR')
    plt.axvline(thr, color='red', linestyle='--', label=f'EER Threshold={thr:.2f}')

    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_path)
    plt.clf()

    return eer

if __name__=="__main__":
    from constants import AGEDB_RESULTS_PATH, LFW_RESULTS_PATH, PREDICTION_CSV

    predictions = pd.read_csv(os.path.join(AGEDB_RESULTS_PATH, PREDICTION_CSV))
    eer = calculate_eer(
        prediction_df=predictions, 
        fig_path=os.path.join(os.path.join(AGEDB_RESULTS_PATH, "eer.png"))
    )
    print(eer)
    