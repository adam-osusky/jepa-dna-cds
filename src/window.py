import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def extract_cds_regions(record: SeqRecord) -> list[tuple[int, int]]:
    """
    Extract CDS regions from a SeqRecord.

    Returns a list of (start, end) tuples.
    """
    cds_regions: list[tuple[int, int]] = []
    for feature in record.features:
        if feature.type == "CDS":
            start: int = int(feature.location.start)  # type: ignore
            end: int = int(feature.location.end)  # type: ignore
            cds_regions.append((start, end))
    return cds_regions


def label_window(
    start: int,
    end: int,
    cds_regions: list[tuple[int, int]],
    min_overlap_fraction: float,
) -> int:
    """
    Determine if a window [start, end) should be labeled as coding.

    A window is labeled 1 if the fraction of its bases overlapping any CDS
    regions is >= min_overlap_fraction; otherwise 0.
    """
    overlap: int = 0
    window_length: int = end - start
    for cds_start, cds_end in cds_regions:
        overlap_start: int = max(start, cds_start)
        overlap_end: int = min(end, cds_end)
        if overlap_start < overlap_end:
            overlap += overlap_end - overlap_start
    return int((overlap / window_length) >= min_overlap_fraction)


def create_classification_dataset(
    gb_file: str,
    window_size: int,
    step_size: int,
    min_overlap_fraction: float,
) -> pd.DataFrame:
    """
    Read a GenBank file, slide a window of size window_size with step step_size
    over the sequence, and assign labels based on CDS overlap.

    Returns two lists: windows (as strings) and labels (0/1).
    """
    record: SeqRecord = SeqIO.read(gb_file, "genbank")
    sequence: str = str(record.seq).upper()
    cds_regions: list[tuple[int, int]] = extract_cds_regions(record)

    windows: list[str] = []
    labels: list[int] = []

    seq_len: int = len(sequence)
    for start in range(0, seq_len - window_size + 1, step_size):
        end: int = start + window_size
        window_seq: str = sequence[start:end]

        # # Skip windows containing 'N'
        # if "N" in window_seq:
        #     continue

        label: int = label_window(start, end, cds_regions, min_overlap_fraction)
        windows.append(window_seq)
        labels.append(label)

    df: pd.DataFrame = pd.DataFrame({"sequence": windows, "label": labels})

    return df
