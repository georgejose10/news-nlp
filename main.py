from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import torch

import re
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


class ChargedWordsResponse(BaseModel):
    charged_words_found: list[str]
    count: int



class BiasResponse(BaseModel):
    bias: str
    probabilities: Dict[str,float]


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

# pipeline object for text classification (sentiment)
sentiment_clf = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device,)

# For Bias detection
bias_clf = pipeline(task="zero-shot-classification", model="facebook/bart-large-mnli", device=device)
biases = ["left", "right", "neutral"]

# Sentiment 

# Run the pipeline and return a dictionary of probs (assumes one input string)
def predict_probs(text:str) -> Dict[str,float]:

    output_senti = sentiment_clf(text, truncation=True,max_length=512,top_k=None)

    probs = {}

    for i in output_senti:
        label = i["label"] 
        label = label.lower()

        score = i["score"]
        score = float(score)

        probs[label] = round(score,3)

    # Ensure all three keys exist
    for k in ("negative", "neutral","positive"):
        probs.setdefault(k, 0.0)

    return probs

# Bias Detection

def bias_probs(text: str) -> Dict[str,float]:

    # Model will return "this text has blank leaning" and will pick the leaning one for said text
    output_bias = bias_clf(text,candidate_labels=biases, hypothesis_template="This text has a {} political leaning.", multi_label=False)
    
    # To built a dict of label score pair
    scores = {}

    for i in range(len(output_bias["labels"])):
        label = output_bias["labels"][i]
        score = float(output_bias['scores'][i])
        scores[label] = round(score,3)

    # Sort the labels in Biases order

    final_scores = {}
    for bias in biases:
        if bias in scores:
            final_scores[bias] = scores[bias]
        
        else:
            final_scores[bias] = 0.0

    return final_scores


# Endpoints
@app.get("/health")
def health():
    # Returns JSON to confirm that the server is running
    return {"ok": True}


@app.post("/sentiment", response_model=SentimentResponse)
def predict(userReq: SentimentRequest):

    # Sentiment Analysis

    text = userReq.text.strip()

    if not text:
        raise HTTPException(400,'text is empty')

    probs = predict_probs(text)

    # Choose the winning label and get its confidence 
    sentiment = max(probs,key=probs.get)
    confidence = float(probs[sentiment])
    

    return SentimentResponse(sentiment=sentiment, confidence=confidence, probabilities=probs)

# New Endpoint for detecting charged words
@app.post("/emotive language", response_model=ChargedWordsResponse)
def find_charged_words(userReq: SentimentRequest):
    charged_words = {"crisis","outrage","disaster","shocking","radical","extremist"}
    words = userReq.text.lower().split()

    found = []

    for word in words:
        if word in charged_words:
            found.append(word)

    return ChargedWordsResponse(charged_words_found=found, count=len(found))

@app.post("/bias",response_model=BiasResponse)
def bias_endpoint(userReq: SentimentRequest):
    text = userReq.text.strip()

    probs = bias_probs(text)
    # Finds the key with the largest value
    label = max(probs, key=probs.get)

    return BiasResponse(bias=label, probabilities=probs)