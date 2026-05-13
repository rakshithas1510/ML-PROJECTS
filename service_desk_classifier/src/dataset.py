"""
src/dataset.py
--------------
PyTorch Dataset and DataLoader factories for ticket classification.
"""

import logging
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from src.preprocessing import (
    TextPreprocessor,
    Vocabulary,
    LabelEncoder,
    load_raw_data,
    encode_and_pad,
    compute_class_weights,
)

logger = logging.getLogger(__name__)


# ─── PyTorch Dataset ─────────────────────────────────────────────────────────

class TicketDataset(Dataset):
    """
    Maps integer-encoded ticket sequences to label indices.

    Parameters
    ----------
    sequences : np.ndarray  shape (N, max_length)
    labels    : np.ndarray  shape (N,)
    """

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels    = torch.tensor(labels,    dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


# ─── Data Pipeline ────────────────────────────────────────────────────────────

class DataPipeline:
    """
  Orchestrates the full data pipeline:
    load -> clean -> tokenise -> build vocab -> encode -> split -> wrap in DataLoaders
    """

    def __init__(self, config: dict):
        self.cfg          = config
        self.preprocessor = TextPreprocessor(
            lowercase         = config["preprocessing"]["lowercase"],
            remove_punctuation= config["preprocessing"]["remove_punctuation"],
            remove_stopwords  = config["preprocessing"]["remove_stopwords"],
        )
        self.vocab         = Vocabulary()
        self.label_encoder = LabelEncoder()

    # ── public API ────────────────────────────────────────────────────────────

    def prepare(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Run the full pipeline and return (train_loader, val_loader, test_loader).
        """
        cfg_data = self.cfg["data"]
        cfg_pre  = self.cfg["preprocessing"]
        cfg_tr   = self.cfg["training"]

        # 1. Load raw CSV
        df = load_raw_data(
            cfg_data["raw_data_path"],
            text_col  = cfg_data["text_column"],
            label_col = cfg_data["label_column"],
        )

        texts  = df[cfg_data["text_column"]].tolist()
        labels = df[cfg_data["label_column"]].tolist()

        # 2. Tokenise
        tokenised = self.preprocessor.tokenise_batch(texts)

        # 3. Build vocabulary (on ALL data – we mask test during evaluation only)
        self.vocab.build_from_texts(
            tokenised,
            max_vocab_size = cfg_pre["vocab_size"],
            min_freq       = cfg_pre["min_token_freq"],
        )

        # 4. Encode labels
        self.label_encoder.fit(labels)
        encoded_labels = np.array(self.label_encoder.transform(labels), dtype=np.int64)

        # 5. Encode & pad sequences
        sequences = encode_and_pad(tokenised, self.vocab, cfg_pre["max_length"])

        # 6. Train / val / test split  (stratified)
        seed      = cfg_data["random_seed"]
        test_sz   = cfg_data["test_size"]
        val_sz    = cfg_data["val_size"]

        X_tmp,  X_test,  y_tmp,  y_test  = train_test_split(
            sequences, encoded_labels,
            test_size    = test_sz,
            stratify     = encoded_labels,
            random_state = seed,
        )
        # val_size is expressed relative to the *original* dataset
        val_ratio = val_sz / (1.0 - test_sz)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp,
            test_size    = val_ratio,
            stratify     = y_tmp,
            random_state = seed,
        )

        logger.info(
            "Split sizes -> train: %d  val: %d  test: %d",
            len(X_train), len(X_val), len(X_test),
        )

        # 7. Class weights for handling imbalance
        num_classes  = len(self.label_encoder)
        class_weights = compute_class_weights(y_train.tolist(), num_classes)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float)

        # 8. Wrap in Dataset objects
        train_ds = TicketDataset(X_train, y_train)
        val_ds   = TicketDataset(X_val,   y_val)
        test_ds  = TicketDataset(X_test,  y_test)

        # 9. WeightedRandomSampler for training to oversample minority classes
        sample_weights = torch.tensor(
            [class_weights[lbl] for lbl in y_train], dtype=torch.float
        )
        sampler = WeightedRandomSampler(
            weights     = sample_weights,
            num_samples = len(sample_weights),
            replacement = True,
        )

        batch = cfg_tr["batch_size"]

        train_loader = DataLoader(
            train_ds,
            batch_size = batch,
            sampler    = sampler,   # replaces shuffle=True
            num_workers= 0,
            pin_memory = torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size  = batch,
            shuffle     = False,
            num_workers = 0,
            pin_memory  = torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size  = batch,
            shuffle     = False,
            num_workers = 0,
            pin_memory  = torch.cuda.is_available(),
        )

        return train_loader, val_loader, test_loader
