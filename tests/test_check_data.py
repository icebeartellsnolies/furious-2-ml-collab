import pandas as pd

from src.check_data import check_data

SAMPLE = "tests/data/sample.csv"


def test_sample_passes():
    assert check_data(pd.read_csv(SAMPLE)) == []


def test_bad_data_is_caught():
    df = pd.read_csv(SAMPLE)
    df.loc[0, "global_subjectivity"] = 5  # out of range
    df.loc[1, "num_imgs"] = None  # null
    df = df.drop(columns=["shares"])  # schema
    errors = " ".join(check_data(df))
    assert "global_subjectivity" in errors
    assert "null counts" in errors
    assert "missing columns" in errors
