import logging
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import wandb.sdk
from torch.utils.data import DataLoader, Dataset, Subset

from src.jepa import DNAEncoder, get_torch_seqs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MASK_TOKEN_ID = 4
VOCAB_SIZE = 5
IGNORE_INDEX = -100


class DNALabelsDataset(Dataset):
    def __init__(
        self, sequences_t: torch.LongTensor, labels_t: torch.LongTensor
    ) -> None:
        """
        sequences_t: LongTensor of shape (N_windows, window_size), values {0..3}.
        labels_t:    LongTensor of shape (N_windows,)
        """
        assert sequences_t.dtype == torch.long and sequences_t.dim() == 2
        self.sequences = sequences_t
        self.labels = labels_t

    def __len__(self) -> int:
        return self.sequences.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.LongTensor, torch.LongTensor]:
        return self.sequences[idx], self.labels[idx]  # type: ignore


class SequenceClassificationHead(nn.Module):
    def __init__(self, hidden_dim: int, input_dim: int, num_classes: int = 2) -> None:
        """
        A two-layer MLP: hidden_dim → hidden_dim → num_classes,
        with a ReLU in between.
        Used for both frozen-encoder and raw embedding pipelines.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, hidden_dim) → logits: (B, num_classes)
        return self.net(x)


class RawSequenceClassifier(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, num_classes: int = 2) -> None:
        """
        Instead of a learnable Embedding, we one-hot encode each position (0..vocab_size-1).
        Then we average-pool over the sequence length to get a (B, vocab_size) tensor.
        Finally, we feed that (B, vocab_size) into a two-layer MLP head.
        """
        super().__init__()
        self.vocab_size = vocab_size
        # The head now expects `hidden_dim = vocab_size`
        self.head = SequenceClassificationHead(
            hidden_dim=hidden_dim, input_dim=vocab_size, num_classes=num_classes
        )

    def forward(self, seqs: torch.LongTensor) -> torch.Tensor:
        """
        seqs: (B, window_size), values ∈ {0,1,2,3}
        1) one_hot: (B, window_size, vocab_size)
        2) pooled:  (B, vocab_size)
        3) logits:  (B, num_classes)
        """
        # (B, L, V), float
        one_hot = F.one_hot(seqs, num_classes=self.vocab_size).float()
        # average‐pool over L → (B, V)
        pooled = one_hot.mean(dim=1)
        return self.head(pooled)


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_classifier(
    classification_df: pd.DataFrame,
    encoder_ckpt: Path,
    hidden_dim: int,
    dim_feedforward: int,
    nhead: int,
    num_layers: int,
    head_hidden_dim: int,
    clf_lr: float,
    clf_epochs: int,
    clf_batch_size: int,
    val_steps: int,
    out_dir: Path,
    seed: int,
    wb_logger: None | wandb.sdk.wandb_run.Run,
    use_raw: bool = False,
) -> None:
    """
    Two modes:
      • Frozen-encoder mode (use_raw=False): load a JEPA encoder, freeze it, and train the two-layer MLP head on pooled embeddings.
      • Raw-sequence mode (use_raw=True): train Embedding+same two-layer MLP head directly on integer-encoded windows.
    Runs validation every `val_steps` training steps, and also at the end of each epoch.
    Saves:
      - out_dir/"classifier.pth"       (frozen-encoder mode)
      - out_dir/"raw_classifier.pth"   (use_raw=True)
    """
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    # 1) Prepare full dataset of (sequences → ints, labels → ints)
    # classification_df = classification_df.head(10)
    all_seqs_t, all_labels_t = get_torch_seqs(classification_df)
    dataset = DNALabelsDataset(sequences_t=all_seqs_t, labels_t=all_labels_t)

    # 2) Split train/val by index, using same seed for reproducibility
    num_samples = len(dataset)
    indices = list(range(num_samples))
    random.Random(seed).shuffle(indices)
    # TODO
    split = int(0.8 * num_samples)
    train_idxs, val_idxs = indices[:split], indices[split:]

    train_subset = Subset(dataset, train_idxs)
    val_subset = Subset(dataset, val_idxs)

    train_loader = DataLoader(
        train_subset, batch_size=clf_batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_subset, batch_size=clf_batch_size, shuffle=False, num_workers=2
    )

    mode_str = "raw-sequence" if use_raw else "frozen-encoder"
    logger.info(
        f"Starting classification training ({mode_str} mode): "
        f"{len(train_subset)} samples train, {len(val_subset)} samples val."
    )

    if use_raw:
        classifier_model = RawSequenceClassifier(
            vocab_size=VOCAB_SIZE, hidden_dim=head_hidden_dim, num_classes=2
        ).to(device)
        model_name = "raw_classifier.pth"
        logger.info("Will be using raw classifier.")
    else:
        # 1) Load pretrained JEPA encoder from encoder_ckpt
        encoder = DNAEncoder(
            hidden_dim=hidden_dim,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            num_layers=num_layers,
        ).to(device)
        encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False

        # 2) Use the same two-layer MLP head
        classifier_model = SequenceClassificationHead(
            hidden_dim=head_hidden_dim, input_dim=hidden_dim, num_classes=2
        ).to(device)
        model_name = "classifier.pth"

        logger.info("Will be using pretrained encoder.")

    logger.info(f"Number of trainable params = {count_trainable(classifier_model)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier_model.parameters(), lr=clf_lr)

    global_step = 0
    for epoch in range(clf_epochs):
        classifier_model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            seqs, labels = batch  # seqs: (B, window_size), labels: (B,)
            seqs = seqs.to(device)
            labels = labels.to(device)

            if use_raw:
                # Directly feed raw sequences
                logits = classifier_model(seqs)  # (B, 2)
                loss = criterion(logits, labels)
            else:
                # Frozen-encoder: get JEPA embeddings first
                with torch.no_grad():
                    emb = encoder(seqs)  # (B, window_size, hidden_dim)
                    pooled = emb.mean(dim=1)  # → (B, hidden_dim)
                logits = classifier_model(pooled)  # (B, 2)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size_actual = seqs.size(0)
            epoch_loss += loss.item() * batch_size_actual
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size_actual
            global_step += 1

            # Periodic validation every `val_steps` global steps
            if global_step % val_steps == 0:
                classifier_model.eval()
                val_correct = 0
                val_total = 0
                val_loss = 0.0
                with torch.no_grad():
                    for v_seqs, v_labels in val_loader:
                        v_seqs = v_seqs.to(device)
                        v_labels = v_labels.to(device)

                        if use_raw:
                            v_logits = classifier_model(v_seqs)
                            v_loss_batch = criterion(v_logits, v_labels)
                        else:
                            v_emb = encoder(v_seqs)
                            v_pooled = v_emb.mean(dim=1)
                            v_logits = classifier_model(v_pooled)
                            v_loss_batch = criterion(v_logits, v_labels)

                        val_loss += v_loss_batch.item() * v_seqs.size(0)
                        v_preds = torch.argmax(v_logits, dim=1)
                        val_correct += (v_preds == v_labels).sum().item()
                        val_total += v_seqs.size(0)

                avg_val_loss_mid = val_loss / len(val_subset)
                val_acc_mid = val_correct / val_total
                logger.info(
                    f"[Step {global_step}] Mid-Epoch Validation → Loss={avg_val_loss_mid:.4f}, Acc={val_acc_mid:.4f}"
                )

                if wb_logger:
                    key_loss = "clf_raw_mid_val_loss" if use_raw else "clf_mid_val_loss"
                    key_acc = "clf_raw_mid_val_acc" if use_raw else "clf_mid_val_acc"
                    wb_logger.log(
                        {key_loss: avg_val_loss_mid, key_acc: val_acc_mid},
                        step=global_step,
                    )
                classifier_model.train()  # back to train mode

            if wb_logger:
                key_batch = "clf_raw_batch_loss" if use_raw else "clf_batch_loss"
                wb_logger.log({key_batch: loss.item()}, step=global_step)

        avg_loss = epoch_loss / len(train_subset)
        train_acc = correct / total

        # End-of-epoch Validation
        classifier_model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs = seqs.to(device)
                labels = labels.to(device)

                if use_raw:
                    logits = classifier_model(seqs)
                    loss_batch = criterion(logits, labels)
                else:
                    emb = encoder(seqs)
                    pooled = emb.mean(dim=1)
                    logits = classifier_model(pooled)
                    loss_batch = criterion(logits, labels)

                val_loss += loss_batch.item() * seqs.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += seqs.size(0)

        avg_val_loss = val_loss / len(val_subset)
        val_acc = val_correct / val_total
        logger.info(
            f"Epoch {epoch + 1}/{clf_epochs} [Train] Loss={avg_loss:.4f}, Acc={train_acc:.4f} [Val] Loss={avg_val_loss:.4f}, Acc={val_acc:.4f}"
        )

        if wb_logger:
            if use_raw:
                wb_logger.log(
                    {
                        "clf_raw_epoch_train_loss": avg_loss,
                        "clf_raw_epoch_train_acc": train_acc,
                        "clf_raw_epoch_val_loss": avg_val_loss,
                        "clf_raw_epoch_val_acc": val_acc,
                    },
                    step=global_step,
                )
            else:
                wb_logger.log(
                    {
                        "clf_epoch_train_loss": avg_loss,
                        "clf_epoch_train_acc": train_acc,
                        "clf_epoch_val_loss": avg_val_loss,
                        "clf_epoch_val_acc": val_acc,
                    },
                    step=global_step,
                )

    # 7) Save classification head
    torch.save(classifier_model.state_dict(), out_dir / model_name)
    logger.info(f"Saved classifier model to {out_dir}/{model_name}")
