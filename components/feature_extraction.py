import ast
import os

from deepface import DeepFace
import pandas as pd

from constants import FEATURE_EXTRACTION_MODEL


def extract_features(detections_csv: str, features_csv: str):
    if os.path.exists(features_csv):
        os.remove(features_csv)
    features = pd.DataFrame()
    detections = pd.read_csv(detections_csv,
                             converters={"detections": ast.literal_eval})

    for i, img_data in detections.iterrows():
        results = []
        for det_data in img_data["detections"]:
            print(f'Processing: {det_data["face_image_path"]}')

            result = DeepFace.represent(
                img_path=det_data["face_image_path"],
                model_name=FEATURE_EXTRACTION_MODEL,
                detector_backend="skip",
                enforce_detection=False
            )
            results.append({
                "face_image_path": det_data["face_image_path"],
                "features": result[0]["embedding"]
            })

        f = pd.DataFrame([{
            "img_id": img_data["img_id"],
            "detections": results
        }])

        features = pd.concat([features, f], ignore_index=True)

        header = True if i==0 else False
        f.to_csv(features_csv, mode='a', header=header, index=False)
    return features


if __name__=="__main__":
    from constants import AGEDB_RESULTS_PATH, DETECTIONS_CSV, FEATURES_CSV, \
        LFW_RESULTS_PATH


    extract_features(
        detections_csv=os.path.join(LFW_RESULTS_PATH, DETECTIONS_CSV), 
        features_csv=os.path.join(LFW_RESULTS_PATH, FEATURES_CSV)
    )
    