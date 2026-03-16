import os

from const import AGEDB_IMAGE_LIST, AGEDB_PAIR_LIST, AGEDB_RESULTS_PATH, \
    FEATURES_CSV, PREDICTION_CSV
from evaluate import evaluate
from face_detection import detect_faces
from feature_extraction import extract_features
from prediction import predict

def pipeline(list_path: str, pair_path: str, results_path: str):
    detections_csv = detect_faces(list_path, results_path)
    features = extract_features(detections_csv, 
                                os.path.join(results_path, FEATURES_CSV))
    predictions = predict(pair_path,
                features,
                os.path.join(results_path, PREDICTION_CSV))
    metrics = evaluate(predictions)
    return metrics


if __name__=="__main__":
    metrics = pipeline(AGEDB_IMAGE_LIST, AGEDB_PAIR_LIST, AGEDB_RESULTS_PATH)
    print(metrics)