import pytest
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained("vinai/phobert-base-v2")


@pytest.fixture(scope="session")
def model():
    return AutoModelForSequenceClassification.from_pretrained(
        "vinai/phobert-base-v2", num_labels=6
    )


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "text": [
            "Tôi cảm thấy rất vui vẻ hôm nay",
            "Thật buồn khi phải chia tay",
            "Tôi tức giận vì bị lừa dối",
            "Xem tại https://example.com rất hay",
        ],
        "label": [
            "joy",
            "sadness",
            "anger",
            "joy",
        ],
    })
