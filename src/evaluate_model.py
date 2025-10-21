"""
evaluate_model.py

Evaluation script that loads a fine-tuned HuggingFace model and evaluates on a CSV dataset.
Produces precision, recall, f1, accuracy, confusion matrix, and per-class metrics.
Also saves predictions to a CSV for inspection.
"""

import argparse
import csv
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, cohen_kappa_score
from utils import read_csv_pairs, qa_to_text, index_to_label

def evaluate(model_name_or_path: str, data_csv: str, output_csv: str, device: str='cpu'):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path).to(device)
    pairs = read_csv_pairs(data_csv)

    texts = [qa_to_text(p['qa1'], p['qa2']) for p in pairs]
    labels = [p['label'] for p in pairs]
    true_idxs = [ {'Support':0,'Conflict':1,'Neutral':2}[l] for l in labels ]

    batch_size = 16
    preds = []
    probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            enc = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            outputs = model(**enc)
            logits = outputs.logits.cpu().numpy()
            batch_probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
            batch_preds = np.argmax(logits, axis=1).tolist()
            preds.extend(batch_preds)
            probs.extend(batch_probs.tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(true_idxs, preds, average='weighted', zero_division=0)
    acc = accuracy_score(true_idxs, preds)
    kappa = cohen_kappa_score(true_idxs, preds)
    cm = confusion_matrix(true_idxs, preds)

    # Per-class metrics
    per_class = precision_recall_fscore_support(true_idxs, preds, average=None, labels=[0,1,2])

    # Save CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['qa1', 'qa2', 'true_label', 'pred_label', 'pred_prob_support', 'pred_prob_conflict', 'pred_prob_neutral'])
        for p, t, pr, pb in zip(pairs, labels, preds, probs):
            writer.writerow([p['qa1'], p['qa2'], t, index_to_label(pr), pb[0], pb[1], pb[2]])

    results = {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(acc),
        'kappa': float(kappa),
        'confusion_matrix': cm.tolist(),
        'per_class': {
            'precision': per_class[0].tolist(),
            'recall': per_class[1].tolist(),
            'f1': per_class[2].tolist()
        }
    }
    print("Evaluation results:")
    print(results)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned SciBERT model")
    parser.add_argument("--model", required=True, help="Path or HF name of model")
    parser.add_argument("--data", required=True, help="CSV with qa pairs (qa1,qa2,label)")
    parser.add_argument("--out", required=True, help="Output CSV to save predictions")
    parser.add_argument("--device", default='cpu', help="cpu or cuda")
    args = parser.parse_args()
    evaluate(args.model, args.data, args.out, device=args.device)
