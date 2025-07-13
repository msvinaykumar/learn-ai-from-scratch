# learn-ai-from-scratch
An easy-to-follow AI learning repo made for students and beginners who want to get into machine learning.
Welcome to the AI/ML project workspace! This repository contains sub-projects that demonstrate building real-world machine learning systems from scratch.  
Here we highlight one of the core projects: **Sentiment Classification with BERT**.

---

## 🧠 Project: Sentiment Classifier (BERT + FastAPI)

### 🔍 Overview

The **SentimentPrediction** project is a beginner-friendly, production-style implementation of a sentiment classification system.  
It walks you through building a model that can classify text (like review comments etc.) as **Positive** or **Negative**, using the powerful **BERT** language model.

This is **not just a pre-trained model use case** — we actually fine-tune the BERT model from scratch on a small CSV dataset.

---

### ⚙️ What You’ll Learn

- How to train a BERT model using Hugging Face Transformers and PyTorch
- How to prepare training data and tokenize inputs
- How to serve the model via FastAPI for real-time inference
- How to add user feedback to retrain and improve the model
- How to automate re-training on incorrect predictions

---

### 🧪 Model Training Details

| Parameter                  | Value               |
|---------------------------|---------------------|
| Model                     | `bert-base-uncased` |
| Training Epochs           | 3                   |
| Batch Size (per device)   | 8                   |
| Save Strategy             | every epoch         |
| Tokenizer Used            | BERT Tokenizer      |
| Loss Function             | CrossEntropyLoss     |
| Optimizer & LR            | AdamW, 5e-5 (default) |
| Output Directory          | `./model/`          |
| Feedback-based Retraining | ✅ Yes               |

---

### 📂 Folder Structure

```
MainProject/
├── SentimentPrediction/         # BERT Sentiment Classifier project
│   ├── app.py                    # FastAPI server
│   ├── train.py                  # Model training logic
│   ├── predict_sentiment.py      # Predict function
│   ├── sentiment_data.csv        # Training data
│   ├── requirements.txt
│   └── README.md                 # Hands-on setup
├── README.md                     # ← You're here!
```

---

## 🚀 Get Started with Sentiment Project

👉 **Hands-on guide here:** [SentimentPrediction/README.md](./SentimentPrediction/README.md)

It will walk you through:
- Installing dependencies
- Training the model
- Running FastAPI server
- Testing prediction and feedback API

---

## 📬 Contribution & Next Steps

Feel free to extend this project by:
- Adding more classes (e.g., neutral sentiment)
- Improving model with better preprocessing
- Adding frontend with React/Streamlit

We welcome contributions! Let’s make learning AI/ML easier for everyone.

---
