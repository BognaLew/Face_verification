# Datasets
AGEDB_MAIN_PATH = "./data/agedb"
AGEDB_IMAGE_LIST = AGEDB_MAIN_PATH + "/img.list"
AGEDB_PAIR_LIST = AGEDB_MAIN_PATH + "/pair.list"

LFW_MAIN_PATH = "./data/lfw"
LFW_IMAGE_LIST = LFW_MAIN_PATH + "/img.list"
LFW_PAIR_LIST = LFW_MAIN_PATH + "/pair.list"

# Results
AGEDB_RESULTS_PATH = "./results/agedb"
LFW_RESULTS_PATH = "./results/lfw"

# Detection config
DETECTOR_BACKEND = "retinaface"
DETECTIONS_CSV = "detections.csv"
FACES_DIR = "faces"

# Feature extraction config
FEATURE_EXTRACTION_MODEL = "ArcFace"
FEATURES_CSV = "features.csv"

# Distance config
PREDICTION_CSV = "prediction.csv"
THRESHOLD = .23