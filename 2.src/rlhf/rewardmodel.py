# === Step 1: Import libraries ===
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import random
import torch

# === Step 2: Load and prepare dataset ===
df = pd.read_csv(r'C:\Users\sneha\Downloads\projectdsmm\chatter_files\Project\CodesToUpload\rlhf\feedback_collection.csv')

# Convert feedback to numeric labels
df['label'] = df['feedback'].map({'good': 1, 'bad': 0})

# Combine query and answer
df['input_text'] = df['query'] + " [SEP] " + df['answer']

# Save for training
df[['input_text', 'label']].to_csv(r'C:\Users\sneha\Downloads\projectdsmm\chatter_files\Project\CodesToUpload\rlhf\reward_dataset.csv', index=False)

print("Prepared reward_dataset.csv")

# === Step 3: Load dataset ===
dataset = load_dataset('csv', data_files=r'C:\Users\sneha\Downloads\projectdsmm\chatter_files\Project\CodesToUpload\rlhf\reward_dataset.csv')

# Split dataset into train and eval
split_dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
dataset = split_dataset

# === Step 4: Tokenizer and preprocessing ===
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(examples):
    return tokenizer(examples['input_text'], truncation=True, padding='max_length', max_length=256)

tokenized = dataset.map(preprocess, batched=True)

# === Step 5: Define model ===
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# === Step 6: Metrics ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='binary')  # Use 'binary' for two classes
    }


# === Step 7: Training arguments ===
output_dir = r"C:\Users\sneha\Downloads\projectdsmm\chatter_files\Project\CodesToUpload\rlhf\reward_model"

args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    num_train_epochs=5,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    save_total_limit=1,
    load_best_model_at_end=True,
    logging_dir=output_dir,
    logging_steps=10,
    save_strategy="epoch"
)


# === Step 8: Trainer ===
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['test'],  # Now using proper evaluation set
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]

)


# === Step 9: Train and save ===
if __name__ == "__main__":
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Reward model trained and saved at: {output_dir}")


