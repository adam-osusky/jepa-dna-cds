import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import wandb.sdk
from numpy.typing import NDArray
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MASK_TOKEN_ID = 4
VOCAB_SIZE = 5
IGNORE_INDEX = -100

char2int = (
    np.zeros(256, dtype=np.int64) + 10
)  # so we would get index error if some unexpected char in seq
char2int[ord("A")] = 0
char2int[ord("C")] = 1
char2int[ord("G")] = 2
char2int[ord("T")] = 3


def seq_string_to_np_array(seq: str) -> NDArray[np.int64]:
    """
    seq: a window_size-character DNA string (e.g. "ACGT…"), assumed to contain only A/C/G/T.
    Returns a 1D np.ndarray of shape (window_size,), dtype=int64, where
      A→0, C→1, G→2, T→3.
    """
    seq = seq.upper()
    arr_bytes = np.frombuffer(
        seq.encode("ascii"), dtype=np.uint8
    )  # shape (window_size,)
    return char2int[arr_bytes]


def get_numpy_seqs(df: DataFrame) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Given a DataFrame with columns "sequence" (window_size-bp strings) and "label" (ints),
    returns:
      - all_windows_np: np.ndarray of shape (N_windows, window_size), dtype=int64, values in {0,1,2,3}
      - all_labels_np:  np.ndarray of shape (N_windows,), dtype=int64, copies df["label"]
    """

    sequences = df["sequence"].tolist()
    labels = df["label"].to_numpy(dtype=np.int64)  # shape (N_windows,)

    all_arrays = [seq_string_to_np_array(s) for s in sequences]

    all_windows_np = np.vstack(all_arrays)  # dtype=int64

    return all_windows_np, labels


def get_torch_seqs(df: DataFrame) -> tuple[torch.LongTensor, torch.LongTensor]:
    """
    Given a DataFrame with columns "sequence" and "label", returns:
      - all_windows_torch: LongTensor of shape (N_windows, window_size), values in {0..3}
      - all_labels_torch:  LongTensor of shape (N_windows,), values from df["label"]
    """
    all_windows_np, labels_np = get_numpy_seqs(df)

    all_windows_torch = torch.from_numpy(all_windows_np).long()  # shape (N,window_size)
    all_labels_torch = torch.from_numpy(labels_np).long()  # shape (N,)

    return all_windows_torch, all_labels_torch  # type: ignore


def mask_torch_sequence(
    seq_t: torch.Tensor,
    mask_prob: float = 0.15,
    mask_token_id: int = MASK_TOKEN_ID,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given:
      seq_t:         torch.LongTensor of shape (window_size,), values in {0,1,2,3}
      mask_prob:     fraction of positions to mask (default 0.15)
      mask_token_id: integer ID for <mask> (default 4)
      ignore_index:  label for “no loss” positions (default -100)

    Returns:
      input_ids_masked: LongTensor (window_size,), with ~mask_prob masked → mask_token_id.
      labels:           LongTensor (window_size,), = original ID where masked, else ignore_index.
    """
    # 1) Sample per‐position Bernoulli to decide masking
    #    torch.rand_like(seq_t, dtype=torch.float) → uniform [0,1) per position
    rand_tensor = torch.rand(seq_t.shape, device=seq_t.device)  # shape (window_size,)
    mask_bool = rand_tensor < mask_prob  # bool mask

    # 2) Build labels with ignore_index everywhere except masked positions
    labels_t = torch.full_like(seq_t, fill_value=ignore_index)  # (window_size,)
    labels_t[mask_bool] = seq_t[mask_bool]

    # 3) Replace masked positions in input_ids with mask_token_id
    input_ids_t = seq_t.clone()
    input_ids_t[mask_bool] = mask_token_id

    return input_ids_t, labels_t


class DNADataset(Dataset):
    def __init__(self, sequences_t: torch.LongTensor) -> None:
        """
        sequences_t: LongTensor of shape (N_windows, window_size), values {0,1,2,3}.
        """
        assert sequences_t.dtype == torch.long
        assert sequences_t.dim() == 2

        self.sequences = sequences_t

    def __len__(self) -> int:
        return self.sequences.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]  # shape: (window_size,)


class DNAEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        dim_feedforward: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        vocab_size: int = VOCAB_SIZE,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim)
        self.pos_emb = nn.Embedding(
            num_embeddings=max_seq_len, embedding_dim=hidden_dim
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=enc_layer, num_layers=num_layers
        )

        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.nhead = nhead
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        input_ids: (B, window_size), values in {0,1,2,3,4}.
        Returns:  (B, window_size, hidden_dim)
        """
        B, L = input_ids.shape
        device = input_ids.device
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # (B, L)

        tok_emb = self.tok_emb(input_ids)  # (B, L, hidden_dim)
        pos_emb = self.pos_emb(pos_ids)  # (B, L, hidden_dim)
        x = tok_emb + pos_emb  # (B, L, hidden_dim)

        x = x.permute(1, 0, 2)  # (L, B, hidden_dim)
        x = self.encoder(x)  # (L, B, hidden_dim)
        x = x.permute(1, 0, 2)  # (B, L, hidden_dim)
        return x


class Predictor(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, window_size, hidden_dim)
        B, L, D = hidden_states.shape
        # Flatten to (B*L, D), run through MLP, then reshape back
        flat = hidden_states.reshape(B * L, D)
        out = self.net(flat)  # (B*L, D)
        return out.view(B, L, D)  # (B, window_size, hidden_dim)


def clone_encoder_for_ema(encoder: DNAEncoder) -> DNAEncoder:
    """
    Make an exact, detached copy of the online encoder to serve as the EMA (target) encoder.
    """
    target = DNAEncoder(
        hidden_dim=encoder.hidden_dim,
        dim_feedforward=encoder.dim_feedforward,
        nhead=encoder.nhead,
        num_layers=encoder.num_layers,
        vocab_size=encoder.vocab_size,
    )
    # Copy weights over
    for p_src, p_tgt in zip(encoder.parameters(), target.parameters()):
        p_tgt.data.copy_(p_src.data)
        p_tgt.requires_grad = False  # Freeze EMA encoder
    return target


def update_ema(online: DNAEncoder, target: DNAEncoder, tau: float) -> None:
    """
    In-place EMA update of target's parameters:
      θ_target ← τ·θ_target + (1-τ)·θ_online
    Typically τ is very close to 1 (e.g. 0.999 or 0.99).
    """
    with torch.no_grad():
        for p_online, p_target in zip(online.parameters(), target.parameters()):
            p_target.data.mul_(tau).add_(p_online.data * (1.0 - tau))


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_jepa(
    classification_df: pd.DataFrame,
    mask_prob: float,
    batch_size: int,
    num_workers: int,
    hidden_dim: int,
    dim_feedforward: int,
    nhead: int,
    num_layers: int,
    lr: float,
    ema_tau: float,
    num_epochs: int,
    out_dir: Path,
    wb_logger: None | wandb.sdk.wandb_run.Run,
) -> None:
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    df = classification_df
    torch_seqs, torch_labels = get_torch_seqs(df)

    dataset = DNADataset(
        sequences_t=torch_seqs,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    encoder = DNAEncoder(
        hidden_dim=hidden_dim,
        vocab_size=VOCAB_SIZE,
        dim_feedforward=dim_feedforward,
        nhead=nhead,
        num_layers=num_layers,
    ).to(device)

    target_encoder = clone_encoder_for_ema(encoder).to(device)
    predictor = Predictor(hidden_dim=hidden_dim).to(device)

    enc_params = count_trainable(encoder)
    predictor_params = count_trainable(predictor)
    total_params = enc_params + predictor_params
    logger.info(f"Encoder trainable parameters: {enc_params:,}")
    logger.info(f"MLM head trainable parameters: {predictor_params:,}")
    logger.info(f"Total trainable parameters: {total_params:,}")

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()), lr=lr
    )
    criterion = nn.L1Loss()

    sqd_step = 0
    for epoch in range(num_epochs):
        encoder.train()
        predictor.train()
        total_loss = 0.0

        for full_seqs in loader:
            # full_seqs: LongTensor (B, window_size), values ∈ {0..3}
            full_seqs = full_seqs.to(device)

            # 1) Mask locally:
            input_ids_masked, _ = mask_torch_sequence(
                full_seqs,
                mask_prob=mask_prob,
                mask_token_id=MASK_TOKEN_ID,
                ignore_index=IGNORE_INDEX,
            )
            # input_ids_masked: (B, window_size), ∈ {0..4}

            # 2) Online encoder on masked input → emb_masked
            emb_masked = encoder(input_ids_masked)
            # emb_masked: (B, window_size, hidden_dim)

            # 3) Target encoder on the **unmasked** full sequences (no grad)
            with torch.no_grad():
                emb_target = target_encoder(full_seqs)
                # emb_target: (B, window_size, hidden_dim)

            # 4) Predictor tries to map emb_masked → emb_pred
            emb_pred = predictor(emb_masked)
            # emb_pred: (B, window_size, hidden_dim)

            # 5) Compute MSE loss (over entire sequence length)
            loss = criterion(emb_pred, emb_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 6) EMA update of target encoder
            update_ema(online=encoder, target=target_encoder, tau=ema_tau)

            batch_loss = loss.item()
            total_loss += batch_loss
            sqd_step += 1

            if wb_logger:
                wb_logger.log({"batch_l1_loss": batch_loss}, step=sqd_step)

            if sqd_step % 100 == 0:
                logger.info(f"Step [{sqd_step}] - —  JEPA L1 Loss = {batch_loss:.6f}")

        avg_loss = total_loss / len(loader)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}  —  JEPA L1 Loss = {avg_loss:.6f}")

    torch.save(encoder.state_dict(), out_dir / "dna_encoder_jepa.pth")
    torch.save(predictor.state_dict(), out_dir / "predictor_jepa.pth")
    logger.info(f"Saved models into {out_dir}")
