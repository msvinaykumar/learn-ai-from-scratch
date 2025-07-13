from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd
import urllib.parse
from predict_sentiment import predict_sentiment
import train
import os

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput, request: Request):
    result = predict_sentiment(input.text)
    encoded_text = urllib.parse.quote(input.text)

    base_url = str(request.base_url).rstrip("/")
    right_url = f"{base_url}/feedback/right?text={encoded_text}"
    wrong_url = f"{base_url}/feedback/wrong?text={encoded_text}&predicted_class={result['predicted_class']}"

    return {
        "prediction": result["label"],
        "confidence": result["score"],
        "feedback_links": {
            "right": right_url,
            "wrong": wrong_url
        }
    }

@app.get("/feedback/right")
def feedback_right(text: str):
    return {"message": "Thanks! We'll keep improving."}

@app.get("/feedback/wrong")
def feedback_wrong(text: str, predicted_class: int):
    # Flip the class for correction
    correct_label = 1 - predicted_class
    new_entry = pd.DataFrame([[text, correct_label]], columns=["text", "label"])
    if os.path.exists("sentiment_data.csv"):
        new_entry.to_csv("sentiment_data.csv", mode="a", header=False, index=False)
    else:
        new_entry.to_csv("sentiment_data.csv", index=False)

    # Retrain the model (note: full retrain)
    train.train_model()

    return {
        "message": f"Thanks! We corrected the label to {correct_label} and retrained the model."
    }
