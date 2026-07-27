from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
DATA_DESCRIPTION_PATH = DATA_DIR / "data_description.txt"

TARGET_COLUMN = "SalePrice"
ID_COLUMN = "Id"

RANDOM_STATE = 42
N_SPLITS = 5
SCORING = "neg_root_mean_squared_error"