"""
Vietnamese Sentiment Analysis - Prediction CLI

A standalone prediction script that loads the trained model and predicts
sentiment for user-provided Vietnamese text.

Usage:
    python -m src.predict "Tôi cảm thấy rất vui hôm nay"
    python -m src.predict --interactive
    python -m src.predict --file input.txt --output predictions.csv
"""
import argparse
import sys
import json
import csv
import torch
from pathlib import Path
from typing import List, Dict

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from underthesea import word_tokenize

from src.data_processing import clean_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "outputs" / "best_model"

LABEL_NAMES = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
EMOJI_MAP = {
    'sadness': '😢',
    'joy': '😊',
    'love': '❤️',
    'anger': '😠',
    'fear': '😨',
    'surprise': '😲',
}


def load_model(model_path: Path = MODEL_DIR) -> tuple:
    """Load the trained model and tokenizer."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please train the model first with: python -m src.train"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path), num_labels=len(LABEL_NAMES)
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


def predict(
    text: str,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> Dict[str, float]:
    """
    Predict sentiment for a single Vietnamese text.

    Returns:
        Dict with 'label', 'confidence', and 'probabilities' for all classes.
    """
    # Preprocess
    cleaned = clean_text(text)
    segmented = word_tokenize(cleaned, format="text")

    # Tokenize
    encoding = tokenizer(
        segmented,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    predicted_idx = int(probs.argmax())
    return {
        "label": LABEL_NAMES[predicted_idx],
        "confidence": float(probs[predicted_idx]),
        "probabilities": {name: float(p) for name, p in zip(LABEL_NAMES, probs)},
    }


def predict_batch(
    texts: List[str],
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int = 32,
) -> List[Dict[str, float]]:
    """Predict sentiment for a batch of texts."""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        cleaned = [word_tokenize(clean_text(t), format="text") for t in batch]

        encoding = tokenizer(
            cleaned,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

        for p in probs:
            idx = int(p.argmax())
            results.append({
                "label": LABEL_NAMES[idx],
                "confidence": float(p[idx]),
                "probabilities": {name: float(v) for name, v in zip(LABEL_NAMES, p)},
            })
    return results


def interactive_mode(model, tokenizer, device):
    """Run interactive prediction loop."""
    print("\n" + "=" * 60)
    print("🇻🇳 Vietnamese Sentiment Analysis - Interactive Mode")
    print("=" * 60)
    print("Type a Vietnamese sentence to predict its sentiment.")
    print("Commands: 'quit' to exit, 'json' to toggle JSON output.\n")

    json_mode = False
    while True:
        try:
            text = input("📝 > ").strip()
            if text.lower() in ("quit", "exit", "q"):
                print("Goodbye! 👋")
                break
            if text.lower() == "json":
                json_mode = not json_mode
                print(f"JSON mode: {'ON' if json_mode else 'OFF'}")
                continue
            if not text:
                continue

            result = predict(text, model, tokenizer, device)
            if json_mode:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                emoji = EMOJI_MAP.get(result["label"], "")
                print(
                    f"   {emoji} {result['label'].upper()} "
                    f"(confidence: {result['confidence']:.1%})"
                )
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"   ❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Vietnamese Sentiment Analysis - Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.predict "Tôi rất vui"
  python -m src.predict --interactive
  python -m src.predict --file input.txt --output predictions.csv
        """,
    )
    parser.add_argument("text", nargs="?", help="Text to predict sentiment for")
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument("--file", "-f", type=str, help="File with texts (one per line)")
    parser.add_argument("--output", "-o", type=str, help="Output CSV file for predictions")
    parser.add_argument(
        "--model-dir", type=str, default=str(MODEL_DIR), help="Path to model directory"
    )
    parser.add_argument("--json", action="store_true", help="Output in JSON format")

    args = parser.parse_args()

    model, tokenizer, device = load_model(Path(args.model_dir))
    print(f"✅ Model loaded (device: {device})")

    if args.interactive:
        interactive_mode(model, tokenizer, device)
    elif args.file:
        # Batch prediction from file
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

        results = predict_batch(texts, model, tokenizer, device)

        if args.output:
            with open(args.output, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["text", "label", "confidence"])
                writer.writeheader()
                for text, result in zip(texts, results):
                    writer.writerow({
                        "text": text,
                        "label": result["label"],
                        "confidence": f"{result['confidence']:.4f}",
                    })
            print(f"✅ Predictions saved to {args.output}")
        else:
            for text, result in zip(texts, results):
                emoji = EMOJI_MAP.get(result["label"], "")
                print(f"{emoji} [{result['label']}] ({result['confidence']:.1%}) {text}")
    elif args.text:
        result = predict(args.text, model, tokenizer, device)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            emoji = EMOJI_MAP.get(result["label"], "")
            print(f"\n{emoji} Sentiment: {result['label'].upper()}")
            print(f"   Confidence: {result['confidence']:.1%}")
            print(f"\n   All probabilities:")
            for name, prob in sorted(
                result["probabilities"].items(), key=lambda x: x[1], reverse=True
            ):
                bar = "█" * int(prob * 30)
                e = EMOJI_MAP.get(name, "")
                print(f"   {e} {name:10s} {prob:.1%} {bar}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
