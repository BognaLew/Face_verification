import ast
import itertools
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from const import AGEDB_PAIR_LIST, AGEDB_RESULTS_PATH, PREDICTION_CSV, FEATURES_CSV, THRESHOLD

def predict(pair_list_path: str, features_csv: pd.DataFrame, distance_csv: str):
    if os.path.exists(distance_csv):
        os.remove(distance_csv)

    pair_list_content = np.loadtxt(pair_list_path, dtype=int)

    results_df = pd.DataFrame()

    header = True
    for pair in pair_list_content:
        img1_data = features_csv.loc[features_csv["img_id"] == pair[0]]
        img2_data = features_csv.loc[features_csv["img_id"] == pair[1]]
        
        pred = 0
        for det1, det2 in itertools.product(img1_data["detections"].iloc[0], img2_data["detections"].iloc[0]):
            dist = cosine(det1['features'], det2['features'])
            pred = 1 if dist < THRESHOLD else 0
            if pred == 1:
              break

        result = pd.DataFrame({
            "img_1_id": pair[0],
            "img_2_id": pair[1],
            "distance": dist,
            "prediction": pred,
            "label": pair[2]
        }, index=[0])

        results_df = pd.concat([results_df, result], ignore_index=True)

        result.to_csv(distance_csv, mode='a', header=header, index=False)
        if header:
          header = False
    return results_df


if __name__=="__main__":
    features = pd.read_csv(os.path.join(AGEDB_RESULTS_PATH, FEATURES_CSV), 
                           converters={"detections": ast.literal_eval})

    _ = predict(AGEDB_PAIR_LIST,
                features,
                os.path.join(AGEDB_RESULTS_PATH, PREDICTION_CSV))