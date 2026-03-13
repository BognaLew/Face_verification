import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATASETS
AGEDB_MAIN_PATH = "./data/agedb"
AGEDB_IMAGE_LIST = AGEDB_MAIN_PATH + "/img.list"
AGEDB_PAIR_LIST = AGEDB_MAIN_PATH + "/pair.list"

LFW_MAIN_PATH = "./data/lfw"
LFW_IMAGE_LIST = LFW_MAIN_PATH + "/img.list"
LFW_PAIR_LIST = LFW_MAIN_PATH + "/pair.list"

# RESULTS
AGEDB_RESULTS_PATH = "./results/agedb"
LFW_RESULTS_PATH = "./results/lfw"

# DETECTION CONFIG
DETECTOR_BACKEND = "retinaface"
ALIGNED_DETECTIONS_CSV = "aligned_detections.csv"
DETECTIONS_CSV = "detections.csv"
FACES_CATALOG = "faces"
ALIGNED_FACES_CATALOG = "aligned_faces"