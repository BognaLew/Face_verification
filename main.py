import os

from constants import AGEDB_IMAGE_LIST, AGEDB_PAIR_LIST, AGEDB_RESULTS_PATH
from components.evaluate import evaluate
from components.face_detection import detect_faces
from components.feature_extraction import extract_features
from components.prediction import predict


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
    from constants import AGEDB_IMAGE_LIST, AGEDB_PAIR_LIST, AGEDB_RESULTS_PATH

    metrics = pipeline(AGEDB_IMAGE_LIST, AGEDB_PAIR_LIST, AGEDB_RESULTS_PATH)
    print(metrics)
    