"""
tests/test_pipeline.py
----------------------
Unit tests for preprocessing, dataset, and model components.

Run with:
    pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import (
    TextPreprocessor,
    Vocabulary,
    LabelEncoder,
    pad_sequence,
    encode_and_pad,
    compute_class_weights,
)
from src.model import BiLSTMClassifier, TextCNNClassifier, build_model


# ─── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_TEXTS = [
    "my laptop screen is black and won't turn on",
    "cannot log into email after password reset",
    "printer offline in room 204",
    "need to install photoshop on my workstation",
    "vpn keeps dropping every ten minutes",
]

SAMPLE_LABELS = ["Hardware", "Account Access", "Hardware", "Software Installation", "Network"]

MINIMAL_CONFIG = {
    "preprocessing": {
        "max_length": 20,
        "vocab_size": 500,
        "min_token_freq": 1,
        "lowercase": True,
        "remove_punctuation": False,
        "remove_stopwords": False,
    },
    "model": {
        "embedding_dim": 32,
        "hidden_dim": 64,
        "num_layers": 1,
        "dropout": 0.1,
        "bidirectional": True,
        "architecture": "bilstm",
    },
}


# ─── TextPreprocessor ─────────────────────────────────────────────────────────

class TestTextPreprocessor:
    def setup_method(self):
        self.pp = TextPreprocessor(lowercase=True)

    def test_lowercase(self):
        assert self.pp.clean("HELLO World") == "hello world"

    def test_url_removal(self):
        cleaned = self.pp.clean("visit https://example.com for details")
        assert "example.com" not in cleaned

    def test_email_removal(self):
        cleaned = self.pp.clean("send to admin@company.org please")
        assert "@" not in cleaned

    def test_tokenise_returns_list(self):
        tokens = self.pp.tokenise("my laptop is broken")
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_tokenise_batch(self):
        result = self.pp.tokenise_batch(SAMPLE_TEXTS)
        assert len(result) == len(SAMPLE_TEXTS)
        assert all(isinstance(r, list) for r in result)


# ─── Vocabulary ───────────────────────────────────────────────────────────────

class TestVocabulary:
    def setup_method(self):
        self.vocab = Vocabulary()
        pp = TextPreprocessor()
        tokenised = pp.tokenise_batch(SAMPLE_TEXTS)
        self.vocab.build_from_texts(tokenised, max_vocab_size=500, min_freq=1)

    def test_special_tokens(self):
        assert self.vocab.token2idx[Vocabulary.PAD_TOKEN] == 0
        assert self.vocab.token2idx[Vocabulary.UNK_TOKEN] == 1

    def test_vocab_size_positive(self):
        assert len(self.vocab) > 2

    def test_encode_known_token(self):
        token = "laptop"
        if token in self.vocab.token2idx:
            idx = self.vocab.encode([token])[0]
            assert idx > 1  # not PAD or UNK

    def test_encode_unknown_token(self):
        idx = self.vocab.encode(["zzzunknownzzz"])[0]
        assert idx == Vocabulary.UNK_IDX

    def test_roundtrip(self):
        tokens = ["laptop", "screen", "broken"]
        indices = self.vocab.encode(tokens)
        decoded = self.vocab.decode(indices)
        # Known tokens should round-trip exactly
        for orig, dec in zip(tokens, decoded):
            if orig in self.vocab.token2idx:
                assert orig == dec


# ─── LabelEncoder ─────────────────────────────────────────────────────────────

class TestLabelEncoder:
    def setup_method(self):
        self.le = LabelEncoder()
        self.le.fit(SAMPLE_LABELS)

    def test_num_classes(self):
        assert len(self.le) == len(set(SAMPLE_LABELS))

    def test_transform_inverse(self):
        encoded  = self.le.transform(SAMPLE_LABELS)
        decoded  = self.le.inverse_transform(encoded)
        assert decoded == SAMPLE_LABELS

    def test_classes_sorted(self):
        assert self.le.classes == sorted(set(SAMPLE_LABELS))


# ─── Sequence Utilities ───────────────────────────────────────────────────────

class TestSequenceUtils:
    def test_pad_short_sequence(self):
        result = pad_sequence([1, 2, 3], max_length=6)
        assert result == [1, 2, 3, 0, 0, 0]
        assert len(result) == 6

    def test_truncate_long_sequence(self):
        result = pad_sequence(list(range(10)), max_length=5)
        assert result == [0, 1, 2, 3, 4]
        assert len(result) == 5

    def test_encode_and_pad_shape(self):
        vocab = Vocabulary()
        pp = TextPreprocessor()
        tokenised = pp.tokenise_batch(SAMPLE_TEXTS)
        vocab.build_from_texts(tokenised)
        arr = encode_and_pad(tokenised, vocab, max_length=15)
        assert arr.shape == (len(SAMPLE_TEXTS), 15)
        assert arr.dtype == np.int64


# ─── Class Weights ────────────────────────────────────────────────────────────

class TestClassWeights:
    def test_shape(self):
        labels = [0, 0, 1, 2, 2, 2]
        weights = compute_class_weights(labels, num_classes=3)
        assert weights.shape == (3,)

    def test_minority_class_higher_weight(self):
        # Class 1 appears once, class 0 appears 5 times → class 1 weight > class 0 weight
        labels = [0, 0, 0, 0, 0, 1]
        weights = compute_class_weights(labels, num_classes=2)
        assert weights[1] > weights[0]


# ─── Models ───────────────────────────────────────────────────────────────────

VOCAB_SIZE   = 200
NUM_CLASSES  = 5
BATCH_SIZE   = 4
SEQ_LEN      = 20


class TestBiLSTM:
    def setup_method(self):
        self.model = BiLSTMClassifier(
            vocab_size    = VOCAB_SIZE,
            embedding_dim = 32,
            hidden_dim    = 64,
            num_classes   = NUM_CLASSES,
            num_layers    = 1,
            dropout       = 0.0,
        )

    def test_output_shape(self):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        out = self.model(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_output_finite(self):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        out = self.model(x)
        assert torch.isfinite(out).all()

    def test_pad_sequence_handled(self):
        # All-padding sequence should not crash
        x = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
        out = self.model(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)


class TestTextCNN:
    def setup_method(self):
        self.model = TextCNNClassifier(
            vocab_size    = VOCAB_SIZE,
            embedding_dim = 32,
            num_classes   = NUM_CLASSES,
            num_filters   = 32,
            filter_sizes  = (2, 3),
            dropout       = 0.0,
        )

    def test_output_shape(self):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        out = self.model(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)


class TestBuildModel:
    def test_bilstm_factory(self):
        model = build_model("bilstm", VOCAB_SIZE, NUM_CLASSES, MINIMAL_CONFIG)
        assert isinstance(model, BiLSTMClassifier)

    def test_textcnn_factory(self):
        model = build_model("textcnn", VOCAB_SIZE, NUM_CLASSES, MINIMAL_CONFIG)
        assert isinstance(model, TextCNNClassifier)

    def test_unknown_arch_raises(self):
        with pytest.raises(ValueError):
            build_model("unknown_arch", VOCAB_SIZE, NUM_CLASSES, MINIMAL_CONFIG)
