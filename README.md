# QARM-BERT: Operationalizing NFR Relationships Using SciBERT

This repository contains code and data for the paper:  
*"Explainable Modeling of Quality Attribute Relationships with Fine-Tuned Language Models"* (REFSQ Submission).

## Overview
We fine-tune SciBERT to classify relationships between Quality Attributes (QAs) extracted from software requirement texts.  
The model predicts **Support**, **Conflict**, or **Neutral** relationships, forming the basis for adaptive Quality Attribute Relationship Matrices (QARMs).

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
