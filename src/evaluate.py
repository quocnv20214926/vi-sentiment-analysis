import sys
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Set up project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "outputs" / "best_model"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
DATA_DIR = PROJECT_ROOT / "data"

LABEL_NAMES = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
NUM_LABELS = len(LABEL_NAMES)

# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class SentimentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: AutoTokenizer, max_len: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_test_data(file_path: Path) -> Tuple[List[str], List[int]]:
    """Loads test data from an Excel or CSV file."""
    if file_path.suffix == '.xlsx':
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError(f"Data at {file_path} must contain 'text' and 'label' columns.")
    
    # Map string labels to integers if needed
    label2id = {name: i for i, name in enumerate(LABEL_NAMES)}
    labels = df['label'].tolist()
    if isinstance(labels[0], str):
        labels = [label2id[l] for l in labels]
    
    return df['text'].tolist(), labels

def load_model_and_tokenizer(model_path: Path, device: torch.device):
    """Loads the trained model and tokenizer."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found at {model_path}. Please train the model first.")
        
    print(f"Loading model and tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path), num_labels=NUM_LABELS)
    model.to(device)
    model.eval()
    return model, tokenizer

def predict_single(text: str, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, device: torch.device) -> str:
    """Predicts sentiment for a single text."""
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        
    return LABEL_NAMES[preds[0]]

def predict_batch(texts: List[str], model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, device: torch.device, batch_size: int = 32) -> List[str]:
    """Predicts sentiment for a batch of texts."""
    model.eval()
    predictions = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting batches"):
        batch_texts = texts[i:i + batch_size]
        
        encoding = tokenizer.batch_encode_plus(
            batch_texts,
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend([LABEL_NAMES[p] for p in preds])
            
    return predictions

def plot_confusion_matrix(y_true: List[int], y_pred: List[int], save_path: Path):
    """Generates and saves a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def plot_f1_scores(y_true: List[int], y_pred: List[int], save_path: Path):
    """Generates and saves a bar chart of per-class F1 scores."""
    f1_scores = f1_score(y_true, y_pred, average=None)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=LABEL_NAMES, y=f1_scores, palette='viridis')
    plt.title('Per-Class F1 Scores')
    plt.ylabel('F1 Score')
    plt.xlabel('Emotion')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"F1 scores plot saved to {save_path}")

def plot_training_history(history_file: Path, save_path: Path):
    """Plots training history if available."""
    if not history_file.exists():
        print(f"Training history not found at {history_file}. Skipping history plot.")
        return
        
    with open(history_file, 'r', encoding='utf-8') as f:
        history = json.load(f)
        
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    if 'val_loss' in history:
        plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy/F1 if available
    plt.subplot(1, 2, 2)
    if 'train_acc' in history:
        plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    if 'val_acc' in history:
        plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Metric')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training history plot saved to {save_path}")

def evaluate_model():
    """Main evaluation routine."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model, tokenizer = load_model_and_tokenizer(MODEL_DIR, device)
    
    # Attempt to load test data
    test_file_csv = DATA_DIR / "test.csv"
    test_file_xlsx = DATA_DIR / "test.xlsx"
    
    test_file = None
    if test_file_csv.exists():
        test_file = test_file_csv
    elif test_file_xlsx.exists():
        test_file = test_file_xlsx
        
    if test_file:
        print(f"Loading test data from {test_file}...")
        texts, labels = load_test_data(test_file)
        
        print("Evaluating on test set...")
        dataset = SentimentDataset(texts, labels, tokenizer)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                b_labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(b_labels.cpu().numpy())
                
        # Generate metrics
        print("\n--- Classification Report ---")
        report = classification_report(all_labels, all_preds, target_names=LABEL_NAMES, digits=4)
        print(report)
        
        report_path = RESULTS_DIR / "classification_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Classification Report\n")
            f.write("="*50 + "\n")
            f.write(report)
        print(f"Saved classification report to {report_path}")
        
        plot_confusion_matrix(all_labels, all_preds, RESULTS_DIR / "confusion_matrix.png")
        plot_f1_scores(all_labels, all_preds, RESULTS_DIR / "f1_scores.png")
        
        history_path = PROJECT_ROOT / "outputs" / "training_history.json"
        plot_training_history(history_path, RESULTS_DIR / "training_history.png")
    else:
        print(f"No test data found at {DATA_DIR}. Skipping dataset evaluation.")

    return model, tokenizer, device

def interactive_mode(model, tokenizer, device):
    """Runs interactive sentiment prediction."""
    print("\n" + "="*50)
    print("Interactive Mode: Type a Vietnamese sentence to predict its sentiment.")
    print("Type 'quit' or 'exit' to stop.")
    print("="*50)
    
    while True:
        try:
            text = input("\nEnter text: ").strip()
            if text.lower() in ['quit', 'exit']:
                break
            if not text:
                continue
                
            pred = predict_single(text, model, tokenizer, device)
            print(f"Predicted Sentiment: {pred}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error during prediction: {e}")

if __name__ == "__main__":
    model, tokenizer, device = evaluate_model()
    
    # Optional interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_mode(model, tokenizer, device)
    else:
        ans = input("\nDo you want to enter interactive mode? (y/n): ")
        if ans.lower().startswith('y'):
            interactive_mode(model, tokenizer, device)
