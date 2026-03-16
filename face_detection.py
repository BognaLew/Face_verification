import os
from typing import Any, Dict, List

import cv2
from deepface import DeepFace
import numpy as np
import pandas as pd

from const import AGEDB_IMAGE_LIST, AGEDB_RESULTS_PATH, \
    DETECTIONS_CSV, DETECTOR_BACKEND, FACES_DIR


def process_detections(detections: List[Dict[str, Any]], image_id: int,
                       images_path: str) -> List[Dict[str, Any]]:
    """
    Saves image fragments with detected faces and processes detection 
    informations.

    Args:
        detections (List[Dict[str, Any]]): Output of DeepFace.extract_faces() 
            function.
        image_id (int): Index of the processed image.
        images_path (str): Path to the catalog where fragments with detected 
            faces are going to
            be stored.
    Returns:
        (List[Dict[str, Any]]): List of processed data for each detected face.
    """
    results = []
    for i, detection in enumerate(detections):
        face = detection["face"]

        # Values of pixels returned by DeepFace.extract_faces(...) are in range 
        # from 0 to 1, therefore they need to be cast to range from 0 to 255, 
        # otherwise the result will be black rectangle.
        if face.max() <= 1.0:
            face = (face * 255).astype(np.uint8)

        file_path = os.path.join(images_path, f"{image_id}_{i}.jpg")
        results.append({
            "face_image_path": file_path,
            "facial_area": detection["facial_area"],
            "confidence": detection["confidence"]
        })

        cv2.imwrite(
            file_path,
            cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        )
    return results


def detect_faces_from_image(image_path: str, image_id: int, images_path: str, 
                            detections_csv: str, align: bool=True):
    """
    Runs DeepFace.extract_faces() function to get face detections and detection 
        results from an image.

    Args:
        image_path (str): Path of the processed image.
        image_id (int): Index of the processed image.
        images_path (str): Path to the catalog where fragments with detected 
            faces are going to be stored.
        detections_csv (str): Path to csv file where detection information is 
            going to be stored.
        align (bool): Flag to enable face alignment (default is True).
    """
    try:
        detections = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=align
        )

        results = process_detections(detections, image_id, images_path)
        header = True if image_id==0 else False
        pd.DataFrame([{
            "img_id": image_id,
            "detections": results
        }]).to_csv(detections_csv, mode='a', header=header, index=False)
    except:
        pass

def detect_faces(image_list_path: str, results_path: str, align: bool=True) -> str:
    """
    Runs face detection for each image from the list.

    Args:
        image_list_path (str): Path to the file with a list of image paths.
        results_path (str): Path where detection results are going to be stored.
        align (bool): Flag to enable face alignment (default is True).
    Returns:
        (str): Path to a csv file with detection data.
    """
    imgs_path = os.path.join(results_path, FACES_DIR)
    os.makedirs(imgs_path, exist_ok=True)

    detections_csv = os.path.join(results_path, DETECTIONS_CSV)

    if os.path.exists(detections_csv):
        os.remove(detections_csv)

    with open(image_list_path) as list_file:
        for i, img_path in enumerate(list_file):
            img_path = img_path.strip()
            print(f'Processing: {img_path}')
            detect_faces_from_image(img_path, i, imgs_path, detections_csv, align)
    return detections_csv


if __name__=="__main__":
    _ = detect_faces(AGEDB_IMAGE_LIST, AGEDB_RESULTS_PATH)