from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from transformers import pipeline

app = FastAPI(title="Newslens")

# What user sends (text only)
class SentimentRequest(BaseModel):
    text: str

# What API returns (label + score)
class SentimentResponse(BaseModel):
    sentiment: str 
    confidence: float
    probabilities: Dict[str, float] # Ex. {"positive": 0.9, "negative": 0.1, "neutral": 0.0}


sentiment_clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@app.get("/health")
def health():
    # Returns JSON to confirm that the server is running
    return {"ok": True}


@app.post("/predict", response_model=SentimentResponse)
def predict(userReq: SentimentRequest):
    text = userReq.text.strip()

    if not text:
        raise HTTPException(400,'text is empty')

    out = sentiment_clf(text, truncation=True)[0]  # {'label': 'POSITIVE', 'score': 0.99}
    label = out["label"].lower()
    score = float(out["score"])

    probs = {
        "positive": score if label == "positive" else (1.0 - score),
        "negative": score if label == "negative" else (1.0 - score),

    }
    
    probs["neutral"] = max(0.0, 1.0 - (probs["positive"] + probs["negative"]))

    sentiment = max(probs,key=probs.get)
    confidence = float(probs[sentiment])

    return SentimentResponse(sentiment=sentiment, confidence=confidence, probabilities=probs)