import os
from typing import Any, Dict

from constants import FEATURES_CSV, PREDICTION_CSV
from components.evaluate import evaluate
from components.face_detection import detect_faces
from components.feature_extraction import extract_features
from components.prediction import predict


def pipeline(list_path: str, pair_path: str, results_path: str) -> Dict[str, Any]:
    """Runs face verification pipeline.
    
    Args:
        list_path (str): Path to a file with a list of images.
        pair_path (str): Path to a file with defined pairs of images and expected
            predictions.
        results_path (str): Path to a directory where results are going to be 
            stored.
    Returns:
        Dict[str, Any]: Dictionary with evaluation metrics.
    """
    detections_csv = detect_faces(
        img_list_path=list_path, 
        results_path=results_path,
    )
    features = extract_features(
        detections_csv_path=detections_csv, 
        features_csv_path=os.path.join(results_path, FEATURES_CSV),
    )
    predictions = predict(
        pair_list_path=pair_path,
        features=features,
        prediction_csv=os.path.join(results_path, PREDICTION_CSV),
    )
    metrics = evaluate(prediction_df=predictions)
    return metrics


if __name__=="__main__":
    from constants import AGEDB_IMAGE_LIST, AGEDB_PAIR_LIST, AGEDB_RESULTS_PATH

    metrics = pipeline(
        list_path=AGEDB_IMAGE_LIST, 
        pair_path=AGEDB_PAIR_LIST, 
        results_path=AGEDB_RESULTS_PATH,
    )
    print(metrics)
    