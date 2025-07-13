# train.py
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __getitem__(self, idx):
        encoded = self.tokenizer(self.texts[idx], padding="max_length", truncation=True, max_length=32, return_tensors="pt")
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.texts)

def train_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = SentimentDataset("sentiment_data.csv")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    args = TrainingArguments(
        output_dir="./model",
        num_train_epochs=5,
        per_device_train_batch_size=4,
        logging_steps=5,
        save_total_limit=1,
        save_strategy="epoch",
        overwrite_output_dir=True
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset,tokenizer=tokenizer)
    trainer.train()
    tokenizer.save_pretrained("./model") 
    trainer.save_model("./model")


if __name__ == "__main__":
    print("Starting training from CSV...")
    train_model()
    print("Training complete. Model saved to ./model")