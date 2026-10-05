import re
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any
from underthesea import word_tokenize
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

LABEL2ID: Dict[str, int] = {
    'sadness': 0, 
    'joy': 1, 
    'love': 2, 
    'anger': 3, 
    'fear': 4, 
    'surprise': 5
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}

VIETNAMESE_STOPWORDS = {
    'tôi', 'của', 'và', 'là', 'có', 'một', 'được', 'cho', 'này', 'đó', 
    'với', 'các', 'từ', 'như', 'khi', 'thì', 'cũng', 'trong', 'để', 'nhưng', 
    'đã', 'sẽ', 'rất', 'không', 'ở', 'theo', 'về', 'đến', 'bởi', 'hơn', 'nữa', 
    'mà', 'lại', 'hay', 'chỉ', 'vì', 'do', 'ra', 'lên', 'xuống', 'vào'
}

def remove_urls(text: str) -> str:
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', str(text))

def remove_special_chars(text: str) -> str:
    """Remove special characters but keep Vietnamese characters and spaces."""
    # Keeping words and whitespace characters. 
    # Python's \w naturally matches unicode characters including Vietnamese.
    return re.sub(r'[^\w\s]', ' ', text)

def normalize_whitespace(text: str) -> str:
    """Normalize whitespace by replacing multiple spaces with a single space."""
    return re.sub(r'\s+', ' ', text).strip()

def clean_text(text: str) -> str:
    """Complete pipeline for text cleaning."""
    if not isinstance(text, str):
        text = str(text)
    text = remove_urls(text)
    text = remove_special_chars(text)
    text = normalize_whitespace(text)
    return text.lower()

def process_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cleaning and word segmentation to the 'text' column, 
    and map labels to integer IDs.
    """
    df_processed = df.copy()
    
    def process_row(text: str) -> str:
        cleaned = clean_text(text)
        # Word segmentation using underthesea for PhoBERT (format="text")
        # Example output: "hôm_nay trời đẹp"
        segmented = word_tokenize(cleaned, format="text")
        return segmented
        
    df_processed['text'] = df_processed['text'].apply(process_row)
    
    # Map labels to IDs
    if 'label' in df_processed.columns:
        df_processed['label'] = df_processed['label'].map(LABEL2ID)
        
        # Drop rows with unmapped labels or empty text
        df_processed = df_processed.dropna(subset=['text', 'label'])
        df_processed['label'] = df_processed['label'].astype(int)
        
    return df_processed

def load_and_process_all() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, val, test datasets from Excel, process them, and return dataframes."""
    train_df = pd.read_excel(DATA_DIR / "train.xlsx")
    val_df = pd.read_excel(DATA_DIR / "val.xlsx")
    test_df = pd.read_excel(DATA_DIR / "test.xlsx")
    
    train_df = process_dataset(train_df)
    val_df = process_dataset(val_df)
    test_df = process_dataset(test_df)
    
    return train_df, val_df, test_df

def create_hf_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: Any = None,
    max_length: int = 256,
) -> DatasetDict:
    """
    Create HuggingFace Datasets from dataframes with tokenization.

    Args:
        train_df: Training dataframe with 'text' and 'label' columns.
        val_df: Validation dataframe with 'text' and 'label' columns.
        test_df: Test dataframe with 'text' and 'label' columns.
        tokenizer: A HuggingFace tokenizer. If None, loads vinai/phobert-base-v2.
        max_length: Maximum sequence length for tokenization.

    Returns:
        A DatasetDict with tokenized train/validation/test splits.
    """
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']], preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df[['text', 'label']], preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']], preserve_index=False)

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })

    # Load PhoBERT tokenizer if not provided
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

    def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    # Map tokenization function to datasets
    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)

    return tokenized_datasets
