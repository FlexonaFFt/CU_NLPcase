import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import f1_score, classification_report

df = pd.read_csv("train-labeled-2.csv")
print(df["label"].value_counts())

labels = df["label"].unique().tolist()
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}
df["label_id"] = df["label"].map(label2id)

MODEL_NAME = "ai-forever/ruBert-base" 

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
dataset = Dataset.from_pandas(df[["text", "label_id"]])

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label_id", "labels")
dataset = dataset.train_test_split(test_size=0.1, seed=42)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

def compute_metrics(pred):
    labels_true = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "f1_weighted": f1_score(labels_true, preds, average="weighted"),
        "f1_macro": f1_score(labels_true, preds, average="macro")
    }

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='./logs',
    load_best_model_at_end=True,
    fp16=True,
    metric_for_best_model="f1_weighted"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# отчёт по тесту
preds = trainer.predict(dataset["test"])
y_true = preds.label_ids
y_pred = np.argmax(preds.predictions, axis=1)

print(classification_report(y_true, y_pred, target_names=labels))
test = pd.read_csv("test.csv")
test_ds = Dataset.from_pandas(test)
test_ds = test_ds.map(tokenize, batched=True)

preds = trainer.predict(test_ds)
test["pred_label"] = [id2label[i] for i in np.argmax(preds.predictions, axis=1)]
test[["id", "pred_label"]].to_csv("submission.csv", index=False)