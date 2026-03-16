import os

from const import FEATURES_CSV, PREDICTION_CSV
from face_detection import detect_faces
from feature_extraction import extract_features
from prediction import predict

def pipeline(list_path: str, pair_path: str, results_path: str):
    detections_csv = detect_faces(list_path, results_path)
    features = extract_features(detections_csv, 
                                os.path.join(results_path, FEATURES_CSV))
    distances = predict(pair_path,
                features,
                os.path.join(results_path, PREDICTION_CSV))