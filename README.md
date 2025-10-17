# QARM-LM: Explainable Modeling of Quality Attribute Relationships with Fine-Tuned Language Models

This repository contains code and data for the paper:  
*"Explainable Modeling of Quality Attribute Relationships with Fine-Tuned Language Models"* (REFSQ Submission).

## Overview
We fine-tune SciBERT to classify relationships between Quality Attributes (QAs) extracted from software requirement texts. The model predicts **Support**, **Conflict**, or **Neutral** relationships, forming the basis for adaptive Quality Attribute Relationship Matrices (QARMs).

## Repository Contents
- `data/` - Sample anonymized QA pairs for testing.
- `src/` - Scripts for preprocessing, training, evaluation, and feature attribution.
- `notebooks/` - Interactive demo notebook for inference.
- `models/` - Saved model checkpoints.
- `requirements.txt` - Python dependencies.

## Installation
```bash
git clone https://github.com/yourusername/QARM-BERT.git
cd QARM-BERT

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage
Preprocessing:
python src/preprocess.py --input data/sample_QA_pairs.csv --output data/processed.csv

Train SciBERT:
python src/train_model.py --data data/processed.csv --epochs 5 --batch_size 16

Evaluate Model:
python src/evaluate_model.py --model models/scibert_checkpoint --data data/processed.csv

Feature Importance:
python src/integrated_gradients.py --model models/scibert_checkpoint --data data/processed.csv

Demo Notebook:
Open notebooks/demo_inference.ipynb in Jupyter or Google Colab to run inference on sample QA pairs.

Notes
- Sample data is synthetic/anonymized to avoid copyright issues.
- To use full IEEE, Springer, ACM abstracts, follow instructions in data/README.md for data acquisition.

License
MIT License

