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
