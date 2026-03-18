import ast
import itertools
import os

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from constants import PREDICTION_CSV, THRESHOLD


def predict(pair_list_path: str, features_csv_path: str, results_path: str) -> str:
    """Calculates cosine similarity for given pairs of pictures and predicts if 
       they present the same person.

    Args:
        pair_list_path (str): Path to a file with defined pairs of pictures and
            expected prediction
        features_csv_path (str): Path to a csv file with calculated embeddings
        results_path (str): Path where results are going to be stored.
    Returns:
        str: Path to a csv file where calculated similarity and
            predictions are going to be stored
    """
    prediction_csv_file = os.path.join(results_path, PREDICTION_CSV)
    if os.path.exists(prediction_csv_file):
        print(f'Removing: {prediction_csv_file}')
        os.remove(prediction_csv_file)
    features = pd.read_csv(features_csv_path, 
                           converters={"detections": ast.literal_eval})


    pair_list_content = np.loadtxt(pair_list_path, dtype=int)

    header = True
    for pair in pair_list_content:
        img1_data = features.loc[features["img_id"] == pair[0]]
        img2_data = features.loc[features["img_id"] == pair[1]]

        pred = 0
        for det1, det2 in itertools.product(img1_data["detections"].iloc[0], 
           img2_data["detections"].iloc[0]):
            img1_f = np.array(det1['features'])
            img2_f = np.array(det2['features'])
            
            img1_f = img1_f/np.linalg.norm(img1_f)
            img2_f = img2_f/np.linalg.norm(img2_f)

            score = cosine_similarity([img1_f], [img2_f])
            pred = 1 if score > THRESHOLD else 0
            if pred == 1:
              break

        result = pd.DataFrame({
            "img_1_id": pair[0],
            "img_2_id": pair[1],
            "similarity": score[0],
            "prediction": pred,
            "label": pair[2],
        }, index=[0])

        result.to_csv(prediction_csv_file, mode='a', header=header, index=False)
        if header:
          header = False
    return prediction_csv_file


if __name__=="__main__":
    from constants import AGEDB_PAIR_LIST, AGEDB_RESULTS_PATH, LFW_PAIR_LIST, \
        LFW_RESULTS_PATH, FEATURES_CSV
    
    predict(pair_list_path=LFW_PAIR_LIST,
            features_csv_path=os.path.join(LFW_RESULTS_PATH, FEATURES_CSV),
            results_path=LFW_RESULTS_PATH,
    )
    