import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset

MASK_TOKEN_ID = 4
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


def get_torch_seqs(df: DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a DataFrame with columns "sequence" and "label", returns:
      - all_windows_torch: LongTensor of shape (N_windows, window_size), values in {0..3}
      - all_labels_torch:  LongTensor of shape (N_windows,), values from df["label"]
    """
    all_windows_np, labels_np = get_numpy_seqs(df)

    all_windows_torch = torch.from_numpy(all_windows_np).long()  # shape (N,window_size)
    all_labels_torch = torch.from_numpy(labels_np).long()  # shape (N,)

    return all_windows_torch, all_labels_torch


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


class TorchDNAMaskedDataset(Dataset):
    def __init__(
        self,
        sequences_t: torch.LongTensor,
        mask_prob: float = 0.15,
        mask_token_id: int = MASK_TOKEN_ID,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        """
        sequences_t: LongTensor of shape (N_windows, window_size), values in {0,1,2,3}.
        mask_prob: fraction of positions to mask per window.
        """
        assert sequences_t.dtype == torch.long, "sequences_t must be LongTensor"
        assert sequences_t.dim() == 2  # and sequences_t.size(1) == window_size

        self.sequences = sequences_t
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.ignore_index = ignore_index

    def __len__(self):
        return self.sequences.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq_t = self.sequences[idx]  # shape (window_size,), dtype=torch.long

        # Apply masking:
        input_ids_t, labels_t = mask_torch_sequence(
            seq_t,
            mask_prob=self.mask_prob,
            mask_token_id=self.mask_token_id,
            ignore_index=self.ignore_index,
        )

        return {
            "input_ids": input_ids_t,  # LongTensor (window_size,)
            "labels": labels_t,  # LongTensor (window_size,)
        }
