"""
src/preprocessing.py
--------------------
Text preprocessing pipeline for service desk ticket classification.
Handles cleaning, tokenization, vocabulary building, and vectorization.
"""

import re
import string
import logging
import pickle
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Vocabulary ──────────────────────────────────────────────────────────────

class Vocabulary:
    """
    Bidirectional token ↔ index mapping with special tokens.

    Special tokens
    --------------
    <PAD>  index 0  – used for padding sequences to equal length
    <UNK>  index 1  – used for out-of-vocabulary tokens at inference time
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_IDX   = 0
    UNK_IDX   = 1

    def __init__(self):
        self.token2idx: Dict[str, int] = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
        }
        self.idx2token: Dict[int, str] = {v: k for k, v in self.token2idx.items()}
        self._frozen = False

    # ── building ──────────────────────────────────────────────────────────────

    def build_from_texts(
        self,
        texts: List[List[str]],
        max_vocab_size: int = 10_000,
        min_freq: int = 1,
    ) -> None:
        """
        Populate vocabulary from a list of pre-tokenised texts.

        Parameters
        ----------
        texts         : list of token lists
        max_vocab_size: hard cap on vocabulary size (includes special tokens)
        min_freq      : tokens appearing fewer times than this are discarded
        """
        counter: Counter = Counter()
        for tokens in texts:
            counter.update(tokens)

        # Keep only tokens above minimum frequency, sorted by frequency
        valid = [
            (tok, freq)
            for tok, freq in counter.most_common()
            if freq >= min_freq
        ]

        # Respect the maximum vocabulary size (subtract 2 for specials)
        valid = valid[: max_vocab_size - 2]

        for tok, _ in valid:
            idx = len(self.token2idx)
            self.token2idx[tok] = idx
            self.idx2token[idx] = tok

        logger.info(
            "Vocabulary built: %d tokens (min_freq=%d, cap=%d)",
            len(self.token2idx), min_freq, max_vocab_size,
        )

    # ── encoding / decoding ───────────────────────────────────────────────────

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token2idx.get(t, self.UNK_IDX) for t in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        return [self.idx2token.get(i, self.UNK_TOKEN) for i in indices]

    def __len__(self) -> int:
        return len(self.token2idx)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Vocabulary saved -> %s", path)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("Vocabulary loaded <- %s  (%d tokens)", path, len(obj))
        return obj


# ─── Label Encoder ───────────────────────────────────────────────────────────

class LabelEncoder:
    """Simple string ↔ integer label mapping."""

    def __init__(self):
        self.label2idx: Dict[str, int] = {}
        self.idx2label: Dict[int, str] = {}

    def fit(self, labels: List[str]) -> "LabelEncoder":
        unique = sorted(set(labels))
        self.label2idx = {lbl: i for i, lbl in enumerate(unique)}
        self.idx2label = {i: lbl for lbl, i in self.label2idx.items()}
        logger.info("Labels encoded: %s", list(self.label2idx.keys()))
        return self

    def transform(self, labels: List[str]) -> List[int]:
        return [self.label2idx[lbl] for lbl in labels]

    def inverse_transform(self, indices: List[int]) -> List[str]:
        return [self.idx2label[i] for i in indices]

    @property
    def classes(self) -> List[str]:
        return [self.idx2label[i] for i in range(len(self.idx2label))]

    def __len__(self) -> int:
        return len(self.label2idx)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("LabelEncoder saved -> %s", path)

    @classmethod
    def load(cls, path: str) -> "LabelEncoder":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj


# ─── Text Cleaner ─────────────────────────────────────────────────────────────

class TextPreprocessor:
    """
    Cleans and tokenises raw ticket text.

    Steps
    -----
    1. Lowercase (optional)
    2. Remove URLs and email addresses
    3. Remove special characters / excess whitespace
    4. Whitespace tokenisation
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = False,
        remove_stopwords: bool = False,
        stopwords: Optional[List[str]] = None,
    ):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.stopwords = set(stopwords or [])

    # ── single text ───────────────────────────────────────────────────────────

    def clean(self, text: str) -> str:
        """Return a cleaned version of *text*."""
        if not isinstance(text, str):
            text = str(text)

        if self.lowercase:
            text = text.lower()

        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        # Remove email addresses
        text = re.sub(r"\S+@\S+\.\S+", " ", text)
        # Remove Windows-style error codes like 0x80070005
        text = re.sub(r"\b0x[0-9a-fA-F]+\b", "errorcode", text)
        # Normalise numbers to a generic token
        text = re.sub(r"\b\d+\b", "NUM", text)

        if self.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenise(self, text: str) -> List[str]:
        """Clean then split *text* into tokens."""
        cleaned = self.clean(text)
        tokens = cleaned.split()
        if self.remove_stopwords and self.stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        return tokens

    # ── batch ─────────────────────────────────────────────────────────────────

    def tokenise_batch(self, texts: List[str]) -> List[List[str]]:
        return [self.tokenise(t) for t in texts]


# ─── Sequence Utilities ───────────────────────────────────────────────────────

def pad_sequence(
    sequence: List[int],
    max_length: int,
    pad_value: int = Vocabulary.PAD_IDX,
) -> List[int]:
    """Truncate or pad *sequence* to *max_length*."""
    if len(sequence) >= max_length:
        return sequence[:max_length]
    return sequence + [pad_value] * (max_length - len(sequence))


def encode_and_pad(
    tokenised_texts: List[List[str]],
    vocab: Vocabulary,
    max_length: int,
) -> np.ndarray:
    """
    Encode token lists to integer indices and pad to *max_length*.

    Returns
    -------
    np.ndarray of shape (N, max_length)
    """
    encoded = []
    for tokens in tokenised_texts:
        indices = vocab.encode(tokens)
        padded  = pad_sequence(indices, max_length)
        encoded.append(padded)
    return np.array(encoded, dtype=np.int64)


# ─── Data Loader ─────────────────────────────────────────────────────────────

def load_raw_data(path: str, text_col: str = "text", label_col: str = "category") -> pd.DataFrame:
    """Load CSV and return a DataFrame with only the relevant columns."""
    df = pd.read_csv(path)
    required = {text_col, label_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    df = df[[text_col, label_col]].dropna()
    logger.info("Loaded %d records from %s", len(df), path)
    return df


def compute_class_weights(labels: List[int], num_classes: int) -> np.ndarray:
    """
    Compute inverse-frequency class weights for handling class imbalance.

    Formula:  weight_c = N / (K * count_c)
    where N = total samples, K = number of classes.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1, counts)          # avoid division by zero
    weights = len(labels) / (num_classes * counts)
    logger.info("Class weights: %s", dict(enumerate(weights.round(4))))
    return weights
