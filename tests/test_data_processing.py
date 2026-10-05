import pytest
import pandas as pd
from src.data_processing import (
    remove_urls,
    remove_special_chars,
    normalize_whitespace,
    clean_text,
    LABEL2ID,
    ID2LABEL,
    process_dataset,
    create_hf_datasets,
    VIETNAMESE_STOPWORDS,
)


def test_remove_urls():
    text = "Xem tại https://example.com rất hay"
    result = remove_urls(text)
    assert "https" not in result
    assert "example.com" not in result
    assert "Xem" in result


def test_remove_special_chars():
    text = "Xin chào! (Hello) @test #tag"
    cleaned = remove_special_chars(text)
    assert "!" not in cleaned
    assert "(" not in cleaned
    assert ")" not in cleaned
    assert "@" not in cleaned
    # Vietnamese characters should be preserved
    assert "chào" in cleaned or "chào" in cleaned.lower()


def test_normalize_whitespace():
    text = "   Tôi   cảm   thấy   rất   vui   vẻ   "
    expected = "Tôi cảm thấy rất vui vẻ"
    assert normalize_whitespace(text) == expected


def test_clean_text():
    text = "  Xem tại https://example.com rất hay! (Yes)  "
    cleaned = clean_text(text)
    assert "https" not in cleaned
    assert "!" not in cleaned
    assert cleaned == cleaned.lower()  # Should be lowercased
    assert "  " not in cleaned  # No double spaces


def test_clean_text_handles_non_string():
    result = clean_text(123)
    assert isinstance(result, str)


def test_label_mapping():
    assert len(LABEL2ID) == 6
    assert len(ID2LABEL) == 6
    for label, id_ in LABEL2ID.items():
        assert ID2LABEL[id_] == label
    # Check all labels exist
    expected_labels = {'sadness', 'joy', 'love', 'anger', 'fear', 'surprise'}
    assert set(LABEL2ID.keys()) == expected_labels


def test_load_data():
    """Test that actual data files load correctly."""
    from pathlib import Path
    data_dir = Path(__file__).resolve().parent.parent / "data"
    
    for filename in ["train.xlsx", "test.xlsx", "val.xlsx"]:
        filepath = data_dir / filename
        if filepath.exists():
            df = pd.read_excel(filepath)
            assert "text" in df.columns
            assert "label" in df.columns
            assert len(df) > 0
            assert df["text"].isnull().sum() == 0


def test_process_dataset(sample_df):
    processed = process_dataset(sample_df)
    assert "text" in processed.columns
    assert "label" in processed.columns
    # Labels should be mapped to integers
    assert processed["label"].dtype in [int, "int64", "int32"]
    assert all(0 <= label <= 5 for label in processed["label"])


@pytest.mark.slow
def test_tokenization(tokenizer):
    text = "Tôi cảm thấy rất vui vẻ hôm nay"
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    assert "input_ids" in tokens
    assert "attention_mask" in tokens
    assert tokens["input_ids"].shape[1] > 0


@pytest.mark.slow
def test_create_hf_datasets(sample_df, tokenizer):
    processed = process_dataset(sample_df)
    dataset = create_hf_datasets(processed, processed, processed, tokenizer)
    assert "train" in dataset
    assert "validation" in dataset
    assert "test" in dataset
    assert len(dataset["train"]) == len(processed)
    assert "input_ids" in dataset["train"].features


def test_vietnamese_stopwords():
    assert isinstance(VIETNAMESE_STOPWORDS, set)
    assert len(VIETNAMESE_STOPWORDS) > 0
    assert "các" in VIETNAMESE_STOPWORDS
    assert "và" in VIETNAMESE_STOPWORDS
