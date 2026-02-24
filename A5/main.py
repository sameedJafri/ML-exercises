import os
import random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt


class TextPreprocessor:
    def __init__(self, model_name="bert-base-uncased", max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.vocab_size = self.tokenizer.vocab_size

    def preprocess(self, text):
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return encoding["input_ids"]


class TextDataset(Dataset):
    def __init__(self, texts, labels, preprocessor):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        input_ids = self.preprocessor.preprocess(text)
        return input_ids.squeeze(), torch.tensor(label)


def load_data(max_length=512, batch_size=32, val_split=0.1, seed=42):
    print("Loading IMDB from Hugging Face...")
    dataset = load_dataset("stanfordnlp/imdb")

    preprocessor = TextPreprocessor(max_length=max_length)

    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    full_train_dataset = TextDataset(train_texts, train_labels, preprocessor)
    test_dataset = TextDataset(test_texts, test_labels, preprocessor)

    total_train = len(full_train_dataset)
    val_size = int(total_train * val_split)
    train_size = total_train - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, preprocessor.vocab_size

class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        num_filters=100,
        kernel_sizes=(3, 4, 5),
        output_dim=1,
        dropout=0.5,
        padding_idx=0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        self.convs = nn.ModuleList(
            [nn.Conv1d(embedding_dim, num_filters, k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids):
        x = self.embedding(input_ids)  # [B, L, E]
        x = x.transpose(1, 2)  # [B, E, L]
        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(x))  # [B, F, L']
            p = torch.max(c, dim=2).values  # [B, F]
            pooled.append(p)
        x = torch.cat(pooled, dim=1)  # [B, F * K]
        x = self.dropout(x)
        x = self.fc(x)  # [B, output_dim]
        return self.sigmoid(x)


def build_model(
    vocab_size,
    embedding_dim=128,
    num_filters=100,
    kernel_sizes=(3, 4, 5),
    output_dim=1,
    dropout=0.5,
    padding_idx=0,
):
    return TextCNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_filters=num_filters,
        kernel_sizes=kernel_sizes,
        output_dim=output_dim,
        dropout=dropout,
        padding_idx=padding_idx,
    )


@dataclass
class Config:
    max_length: int = 512
    batch_size: int = 32
    val_split: float = 0.1
    seed: int = 42
    embedding_dim: int = 128
    num_filters: int = 100
    kernel_sizes: tuple = (3, 4, 5)
    dropout: float = 0.5
    lr: float = 1e-3
    epochs: int = 5
    output_dir: str = "outputs"
    device: str = "cpu"


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch_labels(labels, device):
    labels = labels.float().to(device)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    return labels


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = _batch_labels(labels, device)

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = (outputs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = _batch_labels(labels, device)
        outputs = model(input_ids)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        preds = (outputs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        outputs = model(input_ids).squeeze(1)
        preds = (outputs >= 0.5).long().cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.long().tolist())
    return all_labels, all_preds

class EarlyStopping:
    """Stops training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=3, min_delta=0.0, save_path="best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss, model):
        # If this is the first epoch or the loss improved significantly
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.save_path)


class Visualizer:
    @staticmethod
    def plot_learning_curves(history, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        epochs = list(range(1, len(history["train_loss"]) + 1))

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(epochs, history["train_loss"], label="Train")
        ax[0].plot(epochs, history["val_loss"], label="Val")
        ax[0].set_title("Loss")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].legend()

        ax[1].plot(epochs, history["train_acc"], label="Train")
        ax[1].plot(epochs, history["val_acc"], label="Val")
        ax[1].set_title("Accuracy")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Accuracy")
        ax[1].legend()

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "learning_curves.png"))
        plt.close(fig)

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "confusion_matrix.png"))
        plt.close(fig)





if __name__ == "__main__":
    cfg = Config()
    set_seed(cfg.seed)

    train_dl, val_dl, test_dl, vocab_size = load_data(
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        val_split=cfg.val_split,
        seed=cfg.seed,
    )

    model = build_model(
        vocab_size=vocab_size,
        embedding_dim=cfg.embedding_dim,
        num_filters=cfg.num_filters,
        kernel_sizes=cfg.kernel_sizes,
        dropout=cfg.dropout,
        padding_idx=0,
    ).to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.BCELoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    early_stopping = EarlyStopping(patience=3, save_path=os.path.join(cfg.output_dir, "best_textcnn.pth"))
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"\nStarting training on {cfg.device}...")

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_dl, optimizer, criterion, cfg.device
        )
        val_loss, val_acc = evaluate(model, val_dl, criterion, cfg.device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}/{cfg.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered! Halting training.")
            break # Exit the training loop early
    
    print("\nLoading the best model weights for final evaluation...")
    # This ensures we test on the best version of the model, not the overfitted final epoch
    model.load_state_dict(torch.load(early_stopping.save_path))

    print("\nEvaluating on Test Set...")
    y_true, y_pred = predict(model, test_dl, cfg.device)
    print("\nClassification Report:")
    print(
        classification_report(
            y_true, y_pred, target_names=["Negative (0)", "Positive (1)"]
        )
    )

    print(f"\nSaving visualizations to ./{cfg.output_dir}/ ...")
    Visualizer.plot_learning_curves(history, cfg.output_dir)
    Visualizer.plot_confusion_matrix(y_true, y_pred, cfg.output_dir)
