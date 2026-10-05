import pytest
import torch
import numpy as np
from src.train import compute_metrics


@pytest.mark.slow
def test_model_loads(model):
    assert model is not None
    assert model.num_labels == 6


@pytest.mark.slow
def test_model_output_shape(model, tokenizer):
    texts = ["Tôi cảm thấy rất vui vẻ hôm nay", "Tôi đang rất tức giận"]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    assert logits.shape == (2, 6)


@pytest.mark.slow
def test_predict_single(model, tokenizer):
    text = "Thật buồn khi phải chia tay"
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=-1).item()
    assert 0 <= prediction < 6


@pytest.mark.slow
def test_predict_batch(model, tokenizer):
    texts = [
        "Tôi cảm thấy rất vui vẻ hôm nay",
        "Thật buồn khi phải chia tay",
        "Tôi tức giận vì bị lừa dối",
    ]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=-1).tolist()
    assert len(predictions) == 3
    for p in predictions:
        assert 0 <= p < 6


def test_compute_metrics():
    """Test compute_metrics with a mock eval prediction tuple."""
    # compute_metrics receives (logits, labels) as a tuple
    logits = np.array([
        [0.1, 0.8, 0.05, 0.05, 0.0, 0.0],   # argmax=1 (joy)
        [0.05, 0.05, 0.7, 0.1, 0.05, 0.05],  # argmax=2 (love)
        [0.8, 0.05, 0.05, 0.05, 0.05, 0.0],  # argmax=0 (sadness)
    ])
    labels = np.array([1, 2, 0])  # All correct

    eval_pred = (logits, labels)
    metrics = compute_metrics(eval_pred)

    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "f1_weighted" in metrics
    assert "precision_macro" in metrics
    assert "recall_macro" in metrics
    assert metrics["accuracy"] == 1.0  # All correct


def test_compute_metrics_with_errors():
    """Test compute_metrics with some wrong predictions."""
    logits = np.array([
        [0.1, 0.8, 0.05, 0.05, 0.0, 0.0],   # argmax=1 (joy)
        [0.05, 0.8, 0.05, 0.05, 0.05, 0.0],  # argmax=1 (joy) - WRONG, should be 2
    ])
    labels = np.array([1, 2])

    eval_pred = (logits, labels)
    metrics = compute_metrics(eval_pred)

    assert metrics["accuracy"] == 0.5  # 1/2 correct
    assert 0 <= metrics["f1_macro"] <= 1.0
