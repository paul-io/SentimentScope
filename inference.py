import os
import torch
from transformers import AutoTokenizer

from model import SentimentTransformer

# Config — must match train.py exactly
MAX_LENGTH = 128

CONFIG = {
    "num_classes": 2,
    "d_embed": 128,
    "context_size": MAX_LENGTH,
    "layers_num": 4,
    "heads_num": 4,
    "head_size": 32,
    "dropout_rate": 0.1,
    "use_bias": True,
}

MODEL_DIR = "model"

# Load model
def load_model(model_dir: str, device: torch.device) -> tuple:
    """Load saved weights and tokenizer from model_dir."""
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))

    CONFIG["vocabulary_size"] = tokenizer.vocab_size
    model = SentimentTransformer(CONFIG).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, "weights.pt"), map_location=device))
    model.eval()

    return model, tokenizer


# Predict
def predict(text: str, model: SentimentTransformer, tokenizer, device: torch.device) -> dict:
    """
    Run inference on a single string.

    Returns a dict with:
        label      — 'Positive' or 'Negative'
        confidence — probability of the predicted class (0–100)
    """
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)

    with torch.no_grad():
        probs = torch.softmax(model(input_ids), dim=-1).squeeze()

    predicted_class = probs.argmax().item()
    confidence      = probs[predicted_class].item() * 100

    return {
        "label":      "Positive" if predicted_class == 1 else "Negative",
        "confidence": round(confidence, 1),
    }


# Main — simple CLI for quick testing
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading model...")

    model, tokenizer = load_model(MODEL_DIR, device)
    print("Model loaded. Type 'quit' to exit.\n")

    while True:
        text = input("Enter text: ").strip()
        if text.lower() == "quit":
            break
        if not text:
            continue

        result = predict(text, model, tokenizer, device)
        print(f"  → {result['label']} ({result['confidence']}% confidence)\n")


if __name__ == "__main__":
    main()