"""
predict.py
----------
Classify new service desk tickets using a trained model.

Usage examples
--------------
# Single ticket (interactive)
python predict.py --text "My laptop screen is black and won't turn on"

# Batch from a text file (one ticket per line)
python predict.py --file my_tickets.txt

# Batch CSV (specify the text column)
python predict.py --csv new_tickets.csv --text_col description

# Use a different model checkpoint
python predict.py --text "VPN keeps disconnecting" --model models/best_model.pt
"""

import argparse
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

# ── allow running from project root ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import TextPreprocessor, Vocabulary, LabelEncoder
from src.preprocessing import encode_and_pad
from src.model import build_model
from src.utils import load_config, get_device


# ─── Predictor ───────────────────────────────────────────────────────────────

class TicketPredictor:
    """
    Load a trained checkpoint and classify arbitrary ticket text.

    Parameters
    ----------
    model_dir : directory containing best_model.pt, vocab.pkl, label_encoder.pkl
    config    : loaded config dict
    device    : torch device
    """

    def __init__(self, model_dir: str, config: dict, device: torch.device):
        self.config = config
        self.device = device
        self.max_length = config["preprocessing"]["max_length"]

        model_dir = Path(model_dir)

        # ── Load artefacts ────────────────────────────────────────────────────
        self.vocab = Vocabulary.load(str(model_dir / "vocab.pkl"))
        self.label_encoder = LabelEncoder.load(str(model_dir / "label_encoder.pkl"))
        self.preprocessor  = TextPreprocessor(
            lowercase          = config["preprocessing"]["lowercase"],
            remove_punctuation = config["preprocessing"]["remove_punctuation"],
            remove_stopwords   = config["preprocessing"]["remove_stopwords"],
        )

        # ── Rebuild model & load weights ──────────────────────────────────────
        checkpoint = torch.load(
            model_dir / config["paths"]["best_model_name"],
            map_location=device,
        )
        self.model = build_model(
            architecture = config["model"]["architecture"],
            vocab_size   = len(self.vocab),
            num_classes  = len(self.label_encoder),
            config       = config,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, texts: List[str]) -> List[dict]:
        """
        Classify a list of ticket texts.

        Returns
        -------
        List of dicts with keys: text, predicted_class, confidence, all_scores
        """
        tokenised = [self.preprocessor.tokenise(t) for t in texts]
        sequences = encode_and_pad(tokenised, self.vocab, self.max_length)

        tensor = torch.tensor(sequences, dtype=torch.long).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)                        # (N, C)
            probs  = F.softmax(logits, dim=-1).cpu().numpy()  # (N, C)

        results = []
        for text, prob_row in zip(texts, probs):
            pred_idx = prob_row.argmax()
            all_scores = {
                cls: round(float(prob_row[i]), 4)
                for i, cls in enumerate(self.label_encoder.classes)
            }
            results.append({
                "text":            text,
                "predicted_class": self.label_encoder.idx2label[int(pred_idx)],
                "confidence":      round(float(prob_row[pred_idx]), 4),
                "all_scores":      dict(
                    sorted(all_scores.items(), key=lambda x: -x[1])
                ),
            })
        return results

    def predict_one(self, text: str) -> dict:
        return self.predict([text])[0]


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict service desk ticket categories")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", type=str,         help="Single ticket text")
    g.add_argument("--file", type=str,         help="Plain-text file (one ticket per line)")
    g.add_argument("--csv",  type=str,         help="CSV file path")
    p.add_argument("--text_col", default="text", help="Column name when using --csv")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--model",  default=None,
                   help="Model directory (overrides config paths.model_save_dir)")
    return p.parse_args()


def print_result(result: dict, index: int = None) -> None:
    prefix = f"[{index}] " if index is not None else ""
    print(f"\n{prefix}Ticket   : {result['text'][:120]}")
    print(f"  Category   : {result['predicted_class']}")
    print(f"  Confidence : {result['confidence']:.1%}")
    print("  All scores :")
    for cls, score in result["all_scores"].items():
        bar = "#" * int(score * 30)
        print(f"    {cls:<25} {score:.3f}  {bar}")


def main() -> None:
    args   = parse_args()
    config = load_config(args.config)
    device = get_device()

    model_dir = args.model or config["paths"]["model_save_dir"]

    print(f"\nLoading model from '{model_dir}' …")
    predictor = TicketPredictor(model_dir, config, device)
    print("Model ready.\n")

    # ── Collect texts ─────────────────────────────────────────────────────────
    if args.text:
        texts = [args.text]

    elif args.file:
        path = Path(args.file)
        if not path.exists():
            sys.exit(f"File not found: {path}")
        texts = [line.strip() for line in path.read_text().splitlines() if line.strip()]

    elif args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv)
        if args.text_col not in df.columns:
            sys.exit(f"Column '{args.text_col}' not found. Available: {list(df.columns)}")
        texts = df[args.text_col].dropna().tolist()

    # ── Predict ───────────────────────────────────────────────────────────────
    results = predictor.predict(texts)

    if len(results) == 1:
        print_result(results[0])
    else:
        for i, r in enumerate(results, 1):
            print_result(r, index=i)

    print("\nDone.")


if __name__ == "__main__":
    main()
