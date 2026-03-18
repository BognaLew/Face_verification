# Face_verification
The aim of this project was to implement a simple face verification pipeline using [DeepFace](https://github.com/serengil/deepface). The system processes input images, extracts facial features, and determines whether two images belong to the same identity.

## Pipeline
The pipeline consists of the following stages:

**1. Face Detection**

Faces are detected and aligned from input images using `DeepFace.extract_faces()` with *RetinaFace* model. Detected faces as stored as images and detection data is written to a csv file.

**2. Feature Extraction**

From the detected faces are calculated embeddings using `DeepFace.represent()` with *ArcFace* model. Results are stored in a csv file.

**3. Prediction**

Decision is made based on cosine similarity calculated for defined image pairs. The threshold was chosen using EER metric. Results are stored in a csv file.

**4. Evaluation**

The system is evaluated using:

- FAR (False Acceptance Rate)
- FRR (False Rejection Rate)
- f1-score
- ROC AUC