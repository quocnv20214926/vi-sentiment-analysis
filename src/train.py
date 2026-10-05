import argparse
import logging
import os
import torch
from torch import nn
import numpy as np
from typing import Dict, Any, Tuple
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

from src.data_processing import load_and_process_all, create_hf_datasets

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class WeightedTrainer(Trainer):
    """
    Custom Trainer that computes a weighted CrossEntropyLoss to handle class imbalance.
    """
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Compute custom weighted loss
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Computes accuracy, f1_macro, f1_weighted, precision_macro, recall_macro.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
    precision_macro = precision_score(labels, predictions, average='macro', zero_division=0)
    recall_macro = recall_score(labels, predictions, average='macro', zero_division=0)
    
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro
    }

def train(args: argparse.Namespace):
    # Set seed for reproducibility
    set_seed(42)
    
    model_name = "vinai/phobert-base-v2"
    num_labels = 6
    id2label = {
        0: "sadness", 
        1: "joy", 
        2: "love", 
        3: "anger", 
        4: "fear", 
        5: "surprise"
    }
    label2id = {v: k for k, v in id2label.items()}
    
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    logger.info("Loading and processing data...")
    train_df, val_df, test_df = load_and_process_all()
    
    logger.info("Creating HuggingFace datasets...")
    hf_datasets = create_hf_datasets(
        train_df, 
        val_df, 
        test_df, 
        tokenizer, 
        max_length=args.max_length
    )
    
    # Compute class weights
    logger.info("Computing class weights for handling imbalance...")
    train_labels = hf_datasets["train"]["label"]
    classes = np.unique(train_labels)
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=train_labels
    )
    class_weights = torch.tensor(class_weights_array, dtype=torch.float)
    logger.info(f"Class weights: {class_weights}")
    
    logger.info("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    
    output_dir = 'outputs/checkpoints'
    best_model_dir = 'outputs/best_model'
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    
    logger.info("Configuring TrainingArguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=32,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model='f1_macro',
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        seed=42,
        report_to='none',
    )
    
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=hf_datasets["train"],
        eval_dataset=hf_datasets["validation"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training completed. Saving best model...")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(hf_datasets["test"])
    
    logger.info("Final Evaluation Results on Test Set:")
    for key, value in test_results.items():
        logger.info(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PhoBERT for Vietnamese Sentiment Analysis")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    train(args)
