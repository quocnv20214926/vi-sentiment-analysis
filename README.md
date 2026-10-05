# 🇻🇳 Vietnamese Sentiment Analysis with PhoBERT

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Phân tích cảm xúc văn bản tiếng Việt sử dụng PhoBERT v2**

*Fine-tuned [PhoBERT-base-v2](https://huggingface.co/vinai/phobert-base-v2) for 6-class emotion classification on Vietnamese text.*

[Getting Started](#-getting-started) •
[Usage](#-usage) •
[Training](#-training) •
[Results](#-results) •
[Architecture](#-architecture) •
[Testing](#-testing)

</div>

---

## 📋 Overview

This project performs **multi-class emotion classification** on Vietnamese text, detecting 6 emotion categories:

| Emotion | Label | Emoji | Description |
|---------|-------|-------|-------------|
| Sadness | `sadness` | 😢 | Buồn bã, thất vọng |
| Joy | `joy` | 😊 | Vui vẻ, hạnh phúc |
| Love | `love` | ❤️ | Yêu thương, trìu mến |
| Anger | `anger` | 😠 | Tức giận, bực bội |
| Fear | `fear` | 😨 | Sợ hãi, lo lắng |
| Surprise | `surprise` | 😲 | Ngạc nhiên, bất ngờ |

### Key Features

- 🚀 **PhoBERT v2** – State-of-the-art Vietnamese language model from VinAI Research
- ⚖️ **Weighted Loss** – Handles class imbalance with computed class weights
- 📊 **Comprehensive Evaluation** – Classification reports, confusion matrices, per-class F1 charts
- 🎯 **Interactive CLI** – Predict sentiment in real-time from the command line
- 🧪 **Full Test Suite** – pytest-based tests with data and model validation
- 📦 **Batch Prediction** – Process files with thousands of texts efficiently

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────┐
│                Input Text                    │
│    "Tôi cảm thấy rất vui hôm nay"          │
└──────────────────┬──────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│          Text Preprocessing                   │
│  • URL removal                                │
│  • Special character removal                  │
│  • Whitespace normalization                   │
│  • Lowercasing                                │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│       Vietnamese Word Segmentation            │
│       (underthesea word_tokenize)             │
│    "tôi cảm_thấy rất vui hôm_nay"           │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│         PhoBERT v2 Tokenization               │
│    AutoTokenizer (max_length=256)             │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│           PhoBERT v2 Encoder                  │
│    (vinai/phobert-base-v2)                    │
│    12 layers, 768 hidden, 12 heads            │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│        Classification Head                    │
│    Linear(768 → 6) + Softmax                  │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│              Output                           │
│    😊 joy (confidence: 92.3%)                │
└──────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
NLP20241PRJ-1/
├── config.py                   # Centralized configuration
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project metadata & pytest config
├── README.md                   # This file
│
├── data/                       # Dataset (Excel format)
│   ├── train.xlsx              # Training set (16,000 samples)
│   ├── val.xlsx                # Validation set (2,000 samples)
│   └── test.xlsx               # Test set (2,000 samples)
│
├── originaldata/               # Original English data
│   ├── train.txt
│   ├── test.txt
│   └── val.txt
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── data_processing.py      # Data loading, cleaning, tokenization
│   ├── train.py                # Model training with HuggingFace Trainer
│   ├── evaluate.py             # Model evaluation & visualization
│   ├── predict.py              # Prediction CLI
│   └── utils.py                # Utility functions
│
├── tests/                      # Test suite
│   ├── conftest.py             # Shared pytest fixtures
│   ├── test_data_processing.py # Data processing tests
│   └── test_model.py           # Model & metrics tests
│
└── outputs/                    # Generated outputs
    ├── best_model/             # Saved best model
    ├── checkpoints/            # Training checkpoints
    └── results/                # Evaluation results & plots
```

---

## 🚀 Getting Started


# Install dependencies
pip install -r requirements.txt


## 💡 Usage

### Quick Prediction

```bash
# Single text prediction
python -m src.predict "Tôi cảm thấy rất vui hôm nay"

# Interactive mode
python -m src.predict --interactive

# Batch prediction from file
python -m src.predict --file input.txt --output predictions.csv

# JSON output
python -m src.predict "Tôi rất buồn" --json
```

**Example output:**
```
😊 Sentiment: JOY
   Confidence: 94.2%

   All probabilities:
   😊 joy        94.2% ████████████████████████████
   ❤️ love        3.1% █
   😲 surprise    1.2%
   😢 sadness     0.8%
   😠 anger       0.4%
   😨 fear        0.3%
```

### Interactive Mode

```
🇻🇳 Vietnamese Sentiment Analysis - Interactive Mode
============================================================
Type a Vietnamese sentence to predict its sentiment.
Commands: 'quit' to exit, 'json' to toggle JSON output.

📝 > Tôi yêu gia đình mình rất nhiều
   ❤️ LOVE (confidence: 89.7%)

📝 > Thật tức giận khi bị lừa đảo
   😠 ANGER (confidence: 91.3%)
```

---

## 🏋️ Training

### Train the Model

```bash
# Train with default hyperparameters
python -m src.train

# Custom hyperparameters
python -m src.train --epochs 15 --batch_size 32 --lr 3e-5 --max_length 128
```

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 10 | Number of training epochs |
| `--batch_size` | 16 | Training batch size |
| `--lr` | 2e-5 | Learning rate |
| `--max_length` | 256 | Max token sequence length |

### Key Training Features

- **Model**: `vinai/phobert-base-v2` (135M parameters)
- **Optimizer**: AdamW with weight decay (0.01)
- **Scheduler**: Linear warmup (10% of steps) + linear decay
- **Class Imbalance**: Weighted CrossEntropyLoss with `sklearn.compute_class_weight`
- **Mixed Precision**: FP16 training enabled on CUDA
- **Best Model Selection**: Based on validation `f1_macro`

---

## 📊 Results

### Evaluate the Trained Model

```bash
# Run evaluation
python -m src.evaluate

# With interactive mode
python -m src.evaluate --interactive
```

This generates:
- `outputs/results/classification_report.txt` – Detailed per-class metrics
- `outputs/results/confusion_matrix.png` – Confusion matrix heatmap
- `outputs/results/f1_scores.png` – Per-class F1 score bar chart

### Dataset Statistics

| Split | Samples | Joy | Sadness | Anger | Fear | Love | Surprise |
|-------|---------|-----|---------|-------|------|------|----------|
| Train | 16,000 | 5,362 | 4,666 | 2,159 | 1,937 | 1,304 | 572 |
| Val | 2,000 | 704 | 550 | 275 | 212 | 178 | 81 |
| Test | 2,000 | 695 | 581 | 275 | 224 | 159 | 66 |

> [!IMPORTANT]
> The dataset is imbalanced — **joy** has 9.4x more samples than **surprise**. The weighted loss function compensates for this.

---

## 🧪 Testing

```bash
# Run all fast tests
python -m pytest tests/ -m "not slow" -v

# Run all tests (including model loading – requires internet)
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

### Test Categories

| Test File | Tests | Description |
|-----------|-------|-------------|
| `test_data_processing.py` | 10 | Text cleaning, tokenization, label mapping, data loading |
| `test_model.py` | 6 | Model loading, output shape, predictions, metrics computation |

---

## 🔧 Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language Model | [PhoBERT v2](https://github.com/VinAIResearch/PhoBERT) | base |
| Framework | [PyTorch](https://pytorch.org/) | ≥ 2.0 |
| Training API | [HuggingFace Transformers](https://huggingface.co/transformers) | ≥ 4.40 |
| Word Segmentation | [underthesea](https://github.com/undertheseanlp/underthesea) | ≥ 6.8 |
| Data Processing | [pandas](https://pandas.pydata.org/) | ≥ 2.0 |
| Visualization | [matplotlib](https://matplotlib.org/) + [seaborn](https://seaborn.pydata.org/) | latest |
| Testing | [pytest](https://pytest.org/) | ≥ 8.0 |

### Improvements over Previous Version

| Aspect | Old Version | New Version |
|--------|------------|-------------|
| **Model** | BiLSTM + custom vocab | PhoBERT v2 (pre-trained transformer) |
| **Tokenization** | Manual word2id mapping | PhoBERT AutoTokenizer |
| **Training** | Custom training loop | HuggingFace Trainer API |
| **Class Imbalance** | Not handled | Weighted CrossEntropyLoss |
| **Evaluation** | Basic accuracy | Full classification report + F1 + confusion matrix |
| **Paths** | Hardcoded absolute paths | Relative paths with pathlib |
| **Code Quality** | Single notebook | Modular Python package with type hints |
| **Testing** | None | Comprehensive pytest suite |
| **Prediction** | Notebook only | CLI with interactive mode & batch processing |
| **Reproducibility** | No seed management | Full seed control across all RNGs |

---

## 📖 References

- **PhoBERT**: Dat Quoc Nguyen & Anh Tuan Nguyen. (2020). *PhoBERT: Pre-trained language models for Vietnamese*. Findings of EMNLP.
- **Dataset**: Based on the [Emotion Dataset](https://huggingface.co/datasets/dair-ai/emotion) translated to Vietnamese.
- **underthesea**: Vietnamese NLP Toolkit – https://github.com/undertheseanlp/underthesea

---

## 📄 License

This project is for educational purposes as part of the NLP course (2024-1).

---

<div align="center">

**Made with ❤️ for Vietnamese NLP**

</div>
