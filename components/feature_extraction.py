import ast
import os

from deepface import DeepFace
import pandas as pd

from constants import FEATURE_EXTRACTION_MODEL, FEATURES_CSV


def extract_features(detections_csv_path: str, results_path: str) -> str:
    """Calculates embeddings for each detected face.
    
    Args:
        detection_csv_path (str): Path to a csv file with detections data.
        results_path (str): Path where results are going to be stored.
    Returns:
        str: Path to a csv file where calculated embeddings
            are going to be stored.
    """
    features_csv_path = os.path.join(results_path, FEATURES_CSV)
    if os.path.exists(features_csv_path):
        print(f'Removing: {features_csv_path}')
        os.remove(features_csv_path)
    detections = pd.read_csv(detections_csv_path,
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
                "features": result[0]["embedding"],
            })

        f = pd.DataFrame([{
            "img_id": img_data["img_id"],
            "detections": results,
        }])

        header = True if i==0 else False
        f.to_csv(features_csv_path, mode='a', header=header, index=False)
    return features_csv_path


if __name__=="__main__":
    from constants import AGEDB_RESULTS_PATH, DETECTIONS_CSV, LFW_RESULTS_PATH


    extract_features(
        detections_csv_path=os.path.join(LFW_RESULTS_PATH, DETECTIONS_CSV), 
        features_csv_path=LFW_RESULTS_PATH,
    )
    