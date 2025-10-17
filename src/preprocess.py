```python
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

if __name__ == "__main__":
    df = pd.read_csv("data/sample_QA_pairs.csv")
    df["QA1_clean"] = df["QA1"].apply(preprocess_text)
    df["QA2_clean"] = df["QA2"].apply(preprocess_text)
    df.to_csv("data/processed.csv", index=False)

