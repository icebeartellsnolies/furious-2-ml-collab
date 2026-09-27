import pandas as pd

from src.data_loader import generate_synthetic_data, load_data, split_data


def test_generate_synthetic_data():
    df = generate_synthetic_data(num_samples=50, random_state=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50
    assert "shares" in df.columns


def test_load_data_strips_whitespace(tmp_path):
    # Create sample CSV with leading whitespace in headers
    dummy_csv = tmp_path / "dummy.csv"
    data = """ url, timedelta, n_tokens_title, shares
    http://example.com/1, 700, 10, 1500
    http://example.com/2, 600, 12, 2300
    """
    dummy_csv.write_text(data.strip())

    df = load_data(str(dummy_csv))
    # 'url' and 'timedelta' should be dropped
    assert "url" not in df.columns
    assert "timedelta" not in df.columns
    # Whitespace in column names should be stripped
    assert "n_tokens_title" in df.columns
    assert "shares" in df.columns
    assert len(df) == 2


def test_split_data():
    df = generate_synthetic_data(num_samples=100, random_state=42)
    X_train, X_test, y_train, y_test = split_data(
        df, target_col="shares", test_size=0.25, random_state=42
    )

    assert len(X_train) == 75
    assert len(X_test) == 25
    assert len(y_train) == 75
    assert len(y_test) == 25
    assert "shares" not in X_train.columns
    assert "shares" not in X_test.columns
