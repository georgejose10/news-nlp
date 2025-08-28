from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import torch


app = FastAPI(title="Newslens")

# What user sends (text only)
class SentimentRequest(BaseModel):
    text: str

# What API returns (label + score)
class SentimentResponse(BaseModel):
    sentiment: str 
    confidence: float
    probabilities: Dict[str, float] # Ex. {"positive": 0.9, "negative": 0.1, "neutral": 0.0}


class ChargedWordsResponse(BaseModel):
    charged_words_found: list[str]
    count: int




# MODEL Setup

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"

id2label  = {0: "negative", 1: "neutral", 2: "positive"}

label2id = {}

for key,value in id2label .items():
    label2id[value] = key

# Use fast switchs from away from a python based tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL, id2label=id2label,label2id=label2id)

# 0 = first CUDA GPU if available, else -1 = CPU
device= 0 if torch.cuda.is_available() else -1

# pipeline object for text classification
sentiment_clf = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device,)

# Run the pipeline and return a dictionary of probs (assumes one input string)
def predict_probs(text:str) -> Dict[str,float]:

    outputs = sentiment_clf(text, truncation=True,max_length=512,top_k=None)

    probs = {}

    for i in outputs:
        label = i["label"] # Ex. "POSITIVE"
        label = label.lower()

        score = i["score"]
        score = float(score)

        probs[label] = score

    # Ensure all three keys exist
    for k in ("negative", "neutral","positive"):
        probs.setdefault(k, 0.0)

    return probs


# Endpoints
@app.get("/health")
def health():
    # Returns JSON to confirm that the server is running
    return {"ok": True, "device": "cuda" if device == 0 else "cpu", "model": MODEL}


@app.post("/predict", response_model=SentimentResponse)
def predict(userReq: SentimentRequest):
    text = userReq.text.strip()

    if not text:
        raise HTTPException(400,'text is empty')

    probs = predict_probs(text)

    # Choose the winning label and get its confidence 
    sentiment = max(probs,key=probs.get)
    confidence = float(probs[sentiment])

    return SentimentResponse(sentiment=sentiment, confidence=confidence, probabilities=probs)

# New Endpoint for detecting charged words
@app.post("/chargedWords", response_model=ChargedWordsResponse)
def find_charged_words(userReq: SentimentRequest):
    charged_words = {"crisis","outrage","disaster","shocking","radical","extremist"}
    words = userReq.text.lower().split()

    found = []

    for word in words:
        if word in charged_words:
            found.append(word)

    return ChargedWordsResponse(charged_words_found=found, count=len(found))