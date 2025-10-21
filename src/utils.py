"""
utils.py

Utility functions for dataset loading, preprocessing, tokenization helpers, and simple IO.
Assumes input CSV with columns: qa1, qa2, label (label in {Support, Conflict, Neutral})
"""

import csv
import json
from typing import List, Dict
import random

def read_csv_pairs(path: str) -> List[Dict[str,str]]:
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Accept either combined qa_pair column or qa1, qa2 columns
            if 'qa_pair' in r and r['qa_pair'].strip():
                qa = r['qa_pair'].strip()
                if '[SEP]' in qa:
                    qa1, qa2 = [s.strip() for s in qa.split('[SEP]')]
                else:
                    qa1 = qa
                    qa2 = ''
            else:
                qa1 = r.get('qa1','').strip()
                qa2 = r.get('qa2','').strip()
            label = r.get('label', '').strip()
            pairs.append({'qa1': qa1, 'qa2': qa2, 'label': label})
    return pairs

def write_json(obj, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)

def sample_pairs(pairs: List[Dict[str,str]], n: int=10):
    return random.sample(pairs, min(n, len(pairs)))

def qa_to_text(qa1: str, qa2: str) -> str:
    # Format as SciBERT input: "QA1 [SEP] QA2"
    return f"{qa1} [SEP] {qa2}"

def label_to_index(label: str) -> int:
    mapping = {'Support': 0, 'Conflict': 1, 'Neutral': 2}
    return mapping.get(label, -1)

def index_to_label(idx: int) -> str:
    mapping = {0: 'Support', 1: 'Conflict', 2: 'Neutral'}
    return mapping.get(idx, 'Unknown')
