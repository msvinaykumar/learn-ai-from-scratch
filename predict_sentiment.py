from transformers import BertTokenizer, BertForSequenceClassification
import torch

MODEL_PATH = "./model"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  # Set to eval mode

def predict_sentiment(text: str):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=32)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=1).item()
    return {"label": "Positive" if pred == 1 else "Negative", "score": float(probs[0][pred]), "predicted_class": pred}


if __name__ == "__main__":
    test_text = "I hate this product."
    result = predict_sentiment(test_text)
    print(f"Input: {test_text}")
    print(f"Prediction: {result}")