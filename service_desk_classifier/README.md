# 🎫 Service Desk Ticket Classifier

A production-ready **Deep Learning** system for automatically classifying IT service desk tickets into categories using **PyTorch**.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Architectures** | Bidirectional LSTM + Attention, TextCNN (Kim 2014) |
| **Class Imbalance** | Weighted CrossEntropyLoss + WeightedRandomSampler |
| **Preprocessing** | Custom tokeniser, vocabulary builder, sequence padding |
| **Evaluation** | Accuracy, Weighted F1, Macro F1, Confusion Matrix, Classification Report |
| **Training** | Early stopping, LR scheduling, gradient clipping, checkpointing |
| **Inference** | Single ticket, batch file, or CSV prediction |
| **VS Code** | Ready-to-use launch configurations |

---

## 📁 Project Structure

```
service_desk_classifier/
├── data/
│   └── tickets.csv              # 120-sample labelled dataset (5 categories)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py         # Vocabulary, LabelEncoder, TextPreprocessor
│   ├── dataset.py               # PyTorch Dataset + DataPipeline
│   ├── model.py                 # BiLSTMClassifier, TextCNNClassifier
│   ├── trainer.py               # Training loop, EarlyStopping, Checkpointing
│   ├── evaluate.py              # Metrics, plots, classification report
│   └── utils.py                 # Config, logging, seeding, device
├── configs/
│   └── config.yaml              # All hyper-parameters in one place
├── models/                      # Saved after training: best_model.pt, vocab.pkl, label_encoder.pkl
├── results/                     # confusion_matrix.png, training_curves.png, metrics.json
├── notebooks/
│   └── exploration.ipynb        # Interactive EDA + training notebook
├── scripts/
│   └── setup_and_run.py         # One-shot install + train script
├── tests/
│   └── test_pipeline.py         # pytest unit tests
├── .vscode/
│   ├── launch.json              # Debug/run configurations
│   └── settings.json
├── train.py                     # Main training entry point
├── predict.py                   # Inference / prediction script
└── requirements.txt
```

---

## 🚀 Quick Start

### Option A – One command (recommended)
```bash
python scripts/setup_and_run.py
```

### Option B – Manual steps

**1. Create a virtual environment**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the model**
```bash
python train.py
```

**4. Predict on a new ticket**
```bash
python predict.py --text "My laptop screen is black and won't turn on"
```

---

## 🏷️ Ticket Categories

The model classifies tickets into **5 categories**:

| Category | Examples |
|---|---|
| `Hardware` | Screen issues, keyboard failures, printer problems |
| `Account Access` | Password resets, locked accounts, new user access |
| `Network` | WiFi issues, VPN drops, connectivity problems |
| `Software Installation` | Installing apps, setting up dev environments |
| `Software Bug` | App crashes, error messages, unexpected behaviour |

---

## ⚙️ Configuration

All hyper-parameters live in `configs/config.yaml`. Key settings:

```yaml
preprocessing:
  max_length: 128        # Sequence padding/truncation length
  vocab_size: 10000      # Maximum vocabulary size

model:
  architecture: bilstm   # bilstm | textcnn
  embedding_dim: 128
  hidden_dim: 256
  num_layers: 2
  dropout: 0.3
  bidirectional: true

training:
  batch_size: 32
  epochs: 30
  learning_rate: 0.001
  patience: 7            # Early stopping patience
```

---

## 🔧 CLI Reference

### `train.py`
```bash
python train.py [OPTIONS]

Options:
  --config  Path to YAML config (default: configs/config.yaml)
  --arch    Model architecture: bilstm | textcnn
  --epochs  Number of training epochs
  --lr      Learning rate
  --batch   Batch size
```

### `predict.py`
```bash
# Single ticket
python predict.py --text "Cannot connect to WiFi"

# Batch from text file (one ticket per line)
python predict.py --file my_tickets.txt

# From CSV
python predict.py --csv new_tickets.csv --text_col description
```

---

## 📊 Evaluation Output

After training, `results/` contains:

- **`confusion_matrix_test.png`** – colour-coded confusion matrix
- **`training_curves.png`** – loss and accuracy over epochs
- **`classification_report_test.txt`** – per-class precision, recall, F1
- **`metrics.json`** – summary metrics (accuracy, weighted F1, macro F1)

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🏗️ Architecture Details

### BiLSTM + Attention (default)
```
Input tokens
     │
 Embedding (128-d)
     │
 BiLSTM × 2 layers (256-d hidden, bidirectional)
     │
 Attention Pooling  ← learns which tokens matter most
     │
 Dropout
     │
 Linear → num_classes
```

### TextCNN (Kim 2014)
```
Input tokens
     │
 Embedding (128-d)
     │
 Conv1D × 4 kernels (sizes 2,3,4,5) → ReLU
     │
 Max-over-time Pooling
     │
 Dropout
     │
 Linear → num_classes
```

---

## 📦 Adding Your Own Data

1. Prepare a CSV with at minimum a `text` column and a `category` column.
2. Update `data.raw_data_path` in `configs/config.yaml`.
3. Update `data.text_column` and `data.label_column` if needed.
4. Run `python train.py` – the vocabulary and label encoder are rebuilt automatically.

---

## 🖥️ VS Code Integration

Open the project folder in VS Code. Four launch configurations are pre-configured in `.vscode/launch.json`:

- **Train Model** – train with default config
- **Train TextCNN** – train with TextCNN architecture
- **Predict (single ticket)** – run inference on a sample ticket
- **Predict from CSV** – batch inference on the sample dataset

Press **F5** or use the *Run and Debug* panel.

---

## 📋 Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0 (CPU or CUDA)
- scikit-learn ≥ 1.3
- pandas, numpy, matplotlib, pyyaml, tqdm

See `requirements.txt` for pinned versions.

---

## 📄 License

MIT License – free for personal and commercial use.
