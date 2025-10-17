from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import torch

# Load dataset
df = pd.read_csv("data/processed.csv")
labels = {"Support": 0, "Conflict": 1, "Neutral": 2}
df["label"] = df["Relationship"].map(labels)

# Tokenization
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
encodings = tokenizer(list(df["QA1_clean"] + " [SEP] " + df["QA2_clean"]),
                      truncation=True, padding=True)
class QADataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
dataset = QADataset(encodings, list(df["label"]))

model = AutoModelForSequenceClassification.from_pretrained("allenai/scibert_scivocab_uncased", num_labels=3)
training_args = TrainingArguments(
    output_dir="models/scibert_checkpoint",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
evaluate_model.py
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import pandas as pd

# Load predictions and labels from model inference
y_true = [...]
y_pred = [...]

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
accuracy = (sum([yt==yp for yt, yp in zip(y_true, y_pred)])) / len(y_true)

cm = confusion_matrix(y_true, y_pred)

print(f"Precision: {precision}, Recall: {recall}, F1: {f1}, Accuracy: {accuracy}")
print("Confusion Matrix:")
print(cm)

