import ast
import itertools
import os

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from constants import THRESHOLD


def predict(pair_list_path: str, features: pd.DataFrame, prediction_csv: str) \
    -> pd.DataFrame:
    """Calculates cosine similarity for given pairs of pictures and predicts if 
       they present the same person.

    Args:
        pair_list_path (str): Path to a file with defined pairs of pictures and
            expected prediction
        features (pd.DataFrame): DataFrame with calculated embeddings for each
            picture. Should contain:
            - img_id (int): id of an image
            - detections(List[Dict[str, Any]]): list of dictionaries with 
                calculated embeddings for detected face. Each dictionary 
                contains:
                    - face_image_path (str): Path to an image with detected face
                    - features (List[float]): Calculated embedding 
        prediction_csv (str): Path to a csv file where calculated similarity and
            predictions are going to be stored
    Returns:
        pd.DataFrame: DataFrame with columns:
            - img_1_id (int): id of the first image
            - img_2_id (int): id of the second image
            - similarity (float): calculated cosine similarity
            - prediction (int): decision of the system
            - label (int): expected decision
    """
    if os.path.exists(prediction_csv):
        print(f'Removing: {prediction_csv}')
        os.remove(prediction_csv)

    pair_list_content = np.loadtxt(pair_list_path, dtype=int)

    results_df = pd.DataFrame()

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

        results_df = pd.concat([results_df, result], ignore_index=True)

        result.to_csv(prediction_csv, mode='a', header=header, index=False)
        if header:
          header = False
    return results_df


if __name__=="__main__":
    from constants import AGEDB_PAIR_LIST, AGEDB_RESULTS_PATH, LFW_PAIR_LIST, \
        LFW_RESULTS_PATH, PREDICTION_CSV, FEATURES_CSV
    
    features = pd.read_csv(os.path.join(LFW_RESULTS_PATH, FEATURES_CSV), 
                           converters={"detections": ast.literal_eval})

    predict(pair_list_path=LFW_PAIR_LIST,
            features=features,
            prediction_csv=os.path.join(LFW_RESULTS_PATH, PREDICTION_CSV),
    )
    