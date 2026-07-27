import pandas as pd

from configs.config import SAMPLE_SUBMISSION_PATH, TEST_PATH, TRAIN_PATH


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load House Prices train, test and sample submission datasets."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    return train, test, sample_submission