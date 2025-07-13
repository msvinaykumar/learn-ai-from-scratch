# 🧠 Sentiment Classifier API (BERT + FastAPI)

A minimal machine learning application that:
- Trains a BERT model from labeled CSV (`sentiment_data.csv`)
- Exposes a FastAPI endpoint for real-time sentiment prediction
- Allows users to give feedback (Right/Wrong)
- Retrains the model when feedback indicates an incorrect prediction

---

## 📦 Tech Stack

- 🤗 Transformers (BERT base)
- 🧠 PyTorch
- 🚀 FastAPI
- 🐼 Pandas

---

## 📁 Project Structure

```
SentimentPrediction/
├── app.py               # FastAPI app for prediction & feedback
├── train.py             # Script to train and save the model
├── predict_sentiment.py # Utility to load model & predict
├── sentiment_data.csv   # CSV file with labeled training data
├── requirements.txt     # Python dependencies
└── README.md            # You're here!
```

---

## 🚀 Quick Start

### 1. Clone the project and install dependencies

```bash
git clone https://github.com/msvinaykumar/learn-ai-from-scratch
cd SentimentPrediction
pip install -r requirements.txt
```

### 2. Train the model (using provided CSV)

```bash
python train.py
```

This:
- Fine-tunes `bert-base-uncased` on `sentiment_data.csv`
- Saves model and tokenizer to `./model/`

### 3. Start the FastAPI server

```bash
uvicorn app:app --reload
```

API will run at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🎯 API Usage

### 🔎 POST `/predict`

**Input:**
```json
{
  "text": "I love this product"
}
```

**Response:**
```json
{
  "prediction": "Positive",
  "confidence": 0.92,
  "feedback_links": {
    "right": "http://127.0.0.1:8000/feedback/right?text=I%20love%20this%20product",
    "wrong": "http://127.0.0.1:8000/feedback/wrong?text=I%20love%20this%20product&predicted_class=1"
  }
}
```

### 🧠 GET `/feedback/wrong`

- When user clicks this, the server:
  - Infers correct label
  - Appends feedback to `sentiment_data.csv`
  - Retrains the model

### 🧠 GET `/feedback/right`

- Just logs positive feedback (no training)

---

## 🧪 Example CSV Format (`sentiment_data.csv`)

```csv
text,label
I love this movie,1
I hate this product,0
This is amazing,1
Worst experience ever,0
```

- Label `1` = Positive  
- Label `0` = Negative

---

## 🛠️ TODO / Enhancements

- [ ] Async retraining in background
- [ ] Add neutral class (multi-class classification)
- [ ] Deploy via Docker
- [ ] Streamlit or React UI for frontend
- [ ] Batch feedback queue with scheduled training

---

## ✨ Credits

Built with ❤️ by using:
- Hugging Face 🤗
- PyTorch 🔥
- FastAPI ⚡

---

## 📜 License

MIT
=======
