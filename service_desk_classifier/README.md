# Service Desk Ticket Classifier using Deep Learning

## Project Overview
This project is a Deep Learning based Service Desk Ticket Classification System developed using Python and PyTorch.

The aim of this project is to automatically classify IT support/service desk tickets into appropriate categories based on the issue description. This helps reduce manual effort in ticket triaging and improves response efficiency.

The system takes ticket text as input, preprocesses it, and predicts the most relevant category using trained deep learning models.

---

## Problem Statement
In many organizations, service desk teams receive a large number of support tickets daily.

Manually reading and categorizing each ticket:
- takes time
- increases workload
- can lead to incorrect classification

This project automates that process using Natural Language Processing and Deep Learning.

---

## Features
- Automatic service desk ticket classification
- Deep learning based text classification
- Multiple model architecture support
- BiLSTM + Attention model
- TextCNN model
- Custom preprocessing pipeline
- Training and prediction support
- Single ticket prediction
- Batch prediction using text files or CSV
- Performance evaluation with metrics
- Confusion matrix generation
- Configurable training parameters

---

## Tech Stack
- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- YAML
- tqdm

---

## Project Structure

```text
service_desk_classifier/
│
├── configs/          # configuration files
├── data/             # dataset
├── notebooks/        # experimentation notebooks
├── scripts/          # helper scripts
├── src/              # source code
├── tests/            # testing
│
├── train.py          # training script
├── predict.py        # prediction script
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Model Architectures Used

### 1. BiLSTM with Attention
This model captures contextual relationships in ticket descriptions and focuses on important keywords using an attention mechanism.

Best for:
- sequential text understanding
- contextual issue classification

---

### 2. TextCNN
CNN-based text classifier that extracts important text patterns using convolution filters.

Best for:
- fast text classification
- keyword pattern detection

---

## Ticket Categories
Example categories handled by the system:
- Hardware Issues
- Network Issues
- Software Installation
- Software Bugs
- Account Access Issues

---

## Installation

### Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/service_desk_classifier.git
cd service_desk_classifier
```

### Create Virtual Environment
Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:
```bash
python -m venv venv
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Training the Model
Run:

```bash
python train.py
```

Custom training:

```bash
python train.py --arch textcnn --epochs 20 --lr 0.0005
```

---

## Prediction

Single ticket prediction:

```bash
python predict.py --text "VPN keeps disconnecting frequently"
```

Batch prediction from file:

```bash
python predict.py --file tickets.txt
```

Batch prediction from CSV:

```bash
python predict.py --csv tickets.csv --text_col description
```

---

## Evaluation Metrics
The project evaluates model performance using:

- Accuracy
- Precision
- Recall
- F1 Score
- Macro F1 Score
- Weighted F1 Score
- Confusion Matrix
- Classification Report

---

## Future Improvements
Possible enhancements:
- Transformer models (BERT)
- FastAPI deployment
- Docker containerization
- Web dashboard
- Real-time ticket monitoring
- Larger enterprise dataset support

---

## Learning Outcomes
Through this project, I learned:
- NLP preprocessing
- Deep learning for text classification
- PyTorch model development
- handling class imbalance
- model evaluation
- training pipeline design
- batch inference workflows

---

## Author
Rakshitha S  
Artificial Intelligence & Machine Learning Engineering Student
