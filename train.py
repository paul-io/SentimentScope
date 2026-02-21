import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from model import SentimentTransformer

# Config
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 6

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

DATA_DIR  = "aclImdb"
MODEL_DIR = "model"

# Data loading
def load_reviews(folder: str) -> list[str]:
    """Read all .txt files in a folder and return their contents as a list."""
    texts = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts


def build_dataframes(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the IMDB dataset into train and test DataFrames."""
    train_pos = load_reviews(os.path.join(data_dir, "train", "pos"))
    train_neg = load_reviews(os.path.join(data_dir, "train", "neg"))
    test_pos  = load_reviews(os.path.join(data_dir, "test",  "pos"))
    test_neg  = load_reviews(os.path.join(data_dir, "test",  "neg"))

    train_df = pd.DataFrame({
        "review": train_pos + train_neg,
        "label":  [1] * len(train_pos) + [0] * len(train_neg),
    })
    test_df = pd.DataFrame({
        "review": test_pos + test_neg,
        "label":  [1] * len(test_pos) + [0] * len(test_neg),
    })

    return train_df, test_df


# Dataset
class IMDBDataset(Dataset):
    """Tokenizes IMDB reviews and returns (input_ids, label) pairs."""

    def __init__(self, data: pd.DataFrame, tokenizer, max_length: int = MAX_LENGTH):
        self.data       = data
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        text  = self.data.iloc[idx]["review"]
        label = int(self.data.iloc[idx]["label"])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return encoding["input_ids"].squeeze(), label


# Training utilities
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Return accuracy (%) on the given data loader."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            preds = model(input_ids).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return (correct / total) * 100


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device) -> dict:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    history = {"epoch_loss": [], "val_accuracy": []}

    for epoch in range(EPOCHS):
        model.train()
        running_loss  = 0.0
        epoch_loss    = 0.0
        epoch_steps   = 0

        for step, (input_ids, labels) in enumerate(train_loader):
            input_ids, labels = input_ids.to(device), labels.to(device)

            logits = model(input_ids)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_loss   += loss.item()
            epoch_steps  += 1

            if (step + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Step [{step+1}/{len(train_loader)}] Loss: {running_loss/100:.4f}")
                running_loss = 0.0

        val_acc = evaluate(model, val_loader, device)
        avg_loss = epoch_loss / epoch_steps

        history["epoch_loss"].append(round(avg_loss, 4))
        history["val_accuracy"].append(round(val_acc, 2))

        print(f"Epoch {epoch+1} — Val Accuracy: {val_acc:.2f}%")

    return history


# Main
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    print("Loading dataset...")
    train_df, test_df = build_dataframes(DATA_DIR)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    split    = int(0.9 * len(train_df))
    train_data, val_data = train_df.iloc[:split], train_df.iloc[split:]

    train_loader = DataLoader(IMDBDataset(train_data, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(IMDBDataset(val_data,   tokenizer), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(IMDBDataset(test_df,    tokenizer), batch_size=BATCH_SIZE, shuffle=False)

    CONFIG["vocabulary_size"] = tokenizer.vocab_size
    model = SentimentTransformer(CONFIG).to(device)

    print("Training...")
    history = train(model, train_loader, val_loader, device)

    test_acc = evaluate(model, test_loader, device)
    print(f"\nTest Accuracy: {test_acc:.2f}%")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "weights.pt"))
    tokenizer.save_pretrained(os.path.join(MODEL_DIR, "tokenizer"))

    history["test_accuracy"] = round(test_acc, 2)
    with open(os.path.join(MODEL_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"Model and training history saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()