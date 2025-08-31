from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import torch

import re
from transformers import pipeline # For the zero-shot NLI pipeline(facebook/bart-large-mnli)

import requests
from bs4 import BeautifulSoup

from fastapi import Body

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
    charged_words_found: List[str]
    count: int

class BiasResponse(BaseModel):
    bias: str
    probabilities: Dict[str,float]


class SentenceResult(BaseModel):
    text: str
    sentiment: str
    sentiment_probs: Dict[str, float]
    bias: str
    bias_probabilities: Dict[str, float]
    charged_words_found: List[str]
    charged_count: int


class AnalyzeRequest(BaseModel):
    text: str

class AnalyzeResponse(BaseModel):
    sentiment_overall: Dict[str,float]
    bias_overall: Dict[str, float]
    sentiment: str
    bias: str
    charged_total: int
    charged_unique: list[str]
    sentences: List[SentenceResult] 

    top_positive: List[dict]
    top_negative: List[dict]
    top_biased: List[dict]

class AnalyzeUrlRequest(BaseModel):
    url: str




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

biases = [
    "The sentence expresses a liberal political view",
    "The sentence expresses a conservative political view",
    "The sentence is politically neutral (The sentence presents information without a political stance)"
]

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
bias_label_map = {
    "The sentence expresses a liberal political view": "liberal",
    "The sentence expresses a conservative political view": "conservative",
    "The sentence is politically neutral (The sentence presents information without a political stance)": "neutral",
}


def bias_probs(text: str) -> Dict[str,float]:

    # Model will return "this text has blank leaning" and will pick the leaning one for said text
    output_bias = bias_clf(text,candidate_labels=biases, hypothesis_template="This text has a {} political leaning.", multi_label=False)
    
    # To built a dict of label score pair
    scores = {}

    # for i in range(len(output_bias["labels"])):
    #     label = output_bias["labels"][i]
    #     score = float(output_bias['scores'][i])
    #     scores[label] = round(score,3)

    for long_label, score in zip(output_bias["labels"], output_bias["scores"]):
        # Tries to get the value of the long label, if not it falls back to the long label from the loop
        short_label = bias_label_map.get(long_label,long_label )
        scores[short_label] = round(score,3)

    return scores

# Charged Words Finder

def search_for_charged(text: str) -> tuple[list[str], int]:

    charged_words = {"crisis","outrage","disaster","shocking","radical","extremist"}

    # Convert text to lowercase and split into words
    words = text.lower().split()
    
    # Remove punctuation 
    words_without_pun = []

    for w in words:
        cleaned = w.strip(".,!?:;\"'()[]{}")
        words_without_pun.append(cleaned)

    found = []

    # Add the found words to the list and return them/count
    for w in words_without_pun:
        if w in charged_words:
            found.append(w)

    return found, len(found)




#  For Article Analysis

def sentence_spliter(text:str) -> list:

    # The split function removes any empty spaces and then regex finds a pattern of any .!? followed by a space
    sentences = re.split(r'(?<=[.!?])\s+',text.strip())

    filtered_sentences = []

    # Skip empty strings
    for s in sentences:
        if s:
            filtered_sentences.append(s)

    return filtered_sentences


# Sentence Analysis
def analyze_sentence(sentence: str) -> SentenceResult:

    sen_probs = predict_probs(sentence)
    sen_label = max(sen_probs, key=sen_probs.get)

    bias_Probs = bias_probs(sentence)
    bias_label = max(bias_Probs, key=bias_Probs.get)

    found_words, count = search_for_charged(sentence)

    return SentenceResult(
        text=sentence, 
        sentiment=sen_label, sentiment_probs=sen_probs, 
        bias=bias_label, bias_probabilities=bias_Probs, 
        charged_words_found=found_words, charged_count=count)

def analyze_text(userReq:str) -> AnalyzeResponse:

    # text = userReq.text.strip()

    text = userReq.strip()
    # Breaks paragraphs into sentences
    sents = sentence_spliter(text)

    # rows should be a list of sentence result objects
    rows: List[SentenceResult] = []

    # Goes through each sentence in sents and analyzes it (see func above) 
    for s in sents:
        result = analyze_sentence(s)
        rows.append(result)


    # Grab the top 3 neg/pos and biased sentences
    pos_list = []
    neg_list = []
    biased_list = []

    # Takes per sentence results of rows and combines them into an overall sentiment/bias score for the paragraph
    if rows:
        total_len = 0
        for r in rows:
            total_len += len(r.text)
        if total_len == 0:
            total_len = 1 # avoiding division by 0

        # Grabbing first row so we know what keys exist in each dictionary
        first_row = rows[0]

        sentiment_keys = list(first_row.sentiment_probs.keys())
        bias_keys = list(first_row.bias_probabilities.keys())

        # Creating two dictionaries for each key
        sentiment_acc ={}
        for key in sentiment_keys:
            sentiment_acc[key] = 0.0

        bias_acc ={}
        for key in bias_keys:
            bias_acc[key] = 0.0


        charged_total = 0
        unique_terms = set()

        for r in rows:
            weight = len(r.text) / total_len

            for key in sentiment_keys:
                value = r.sentiment_probs.get(key,0.0)
                sentiment_acc[key] += value * weight

            for key in bias_keys:
                value = r.bias_probabilities.get(key, 0.0)
                bias_acc[key] += value * weight

            charged_total += r.charged_count

            for word in r.charged_words_found:
                unique_terms.add(word)


        top_sentiment = ""
        # Makes sure the first real value always gets stored
        top_sentiment_score = -1.0

        for key in sentiment_acc:
            if sentiment_acc[key] > top_sentiment_score:
                top_sentiment_score = sentiment_acc[key]
                top_sentiment = key

        top_bias = ""
        top_bias_score = -1.0

        for key in bias_acc:
            if bias_acc[key] > top_bias_score:
                top_bias_score = bias_acc[key]
                top_bias = key

        # Grabs pos/neg sentiment probablitity from dict
        for r in rows:
            pos_score = float(r.sentiment_probs.get("positive", 0.0))
            neg_score = float(r.sentiment_probs.get("negative", 0.0))
            pos_list.append({"text": r.text, "score": round(pos_score,3)})
            neg_list.append({"text": r.text, "score": round(neg_score,3)})


            lib_score = float(r.bias_probabilities.get("liberal", 0.0))
            cons_score = float(r.bias_probabilities.get("conservative", 0.0))

            if lib_score > cons_score:
               biased_list.append({"text": r.text, "side": "liberal", "score": round(lib_score, 3)})
            else:
                 biased_list.append({"text": r.text, "side": "conservative", "score": round(cons_score, 3)})



        sentiment_overal = sentiment_acc
        bias_overal = bias_acc
        sentiment_label = top_sentiment
        bias_label = top_bias

        sentiment_overall_rounded = {}
        bias_overall_rounded = {}

        for k,v in sentiment_overal.items():
            sentiment_overall_rounded[k] = round(v,3)
        
        for k,v in bias_overal.items():
            bias_overall_rounded[k] = round(v,3)

    else:

        sentiment_overal = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
        bias_overal = {}
        sentiment_label = ""
        bias_label = ""
        charged_total = 0
        unique_terms = set()

    pos_list.sort(key=get_score, reverse=True)
    neg_list.sort(key=get_score, reverse=True)
    biased_list.sort(key=get_score, reverse=True)

    top_positive = pos_list[:3]
    top_negative = neg_list[:3]
    top_biased = biased_list[:3]

    return AnalyzeResponse(
        sentiment_overall= sentiment_overall_rounded,
        bias_overall=bias_overall_rounded,
        sentiment=sentiment_label,
        bias=bias_label,
        charged_total=charged_total,
        charged_unique=sorted(unique_terms),
        sentences=rows,

        top_positive=top_positive,
        top_negative= top_negative,
        top_biased=top_biased,

    )

def get_score(item):
    return item["score"]

def extract_text_from_url(url:str) -> str:

    headers = {"User-Agent": "Newslens/1.0 (+https://example.com/)"}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    html = r.text

    try:
        from readability import Document
        doc = Document(html)
        content_html = doc.summary()
        soup = BeautifulSoup(content_html, "html.parser")

        text_parts = []
        for p in soup.find_all(["p", "li"]):
            text_parts.append(p.get_text(separator=" ", strip=True))

        text = " ".join(text_parts)

    except Exception:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript","footer", "header","nav", "form","aside"]):
            tag.decompose()

        text_parts = []
        for p in soup.findAll("p"):
            text_parts.append(p.get_text(separator=" ", strip=True))

        text= " ".join(text_parts)

    text = re.sub(r"\s+", " ", text).strip()
    return text

# print(extract_text_from_url("https://www.cbc.ca/news/politics/finland-sweden-canada-defence-lessons-1.7621502"))

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
@app.post("/emotive-language", response_model=ChargedWordsResponse)
def find_charged_words(userReq: SentimentRequest):
    
    found, count = search_for_charged(userReq.text)

    return ChargedWordsResponse(charged_words_found=found, count=count)

@app.post("/bias",response_model=BiasResponse)
def bias_endpoint(userReq: SentimentRequest):
    text = userReq.text.strip()

    probs = bias_probs(text)
    # Finds the key with the largest value
    label = max(probs, key=probs.get)

    return BiasResponse(bias=label, probabilities=probs)


@app.post("/analyze-raw-text", response_model=AnalyzeResponse)
def analyze_raw(body: str = Body(..., media_type="text/plain")):
    return analyze_text(body)

@app.post("/anaylze_url", response_model=AnalyzeResponse)
def analyze_url(req: AnalyzeUrlRequest):
    try:
        article_text = extract_text_from_url(req.url)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch/extract: {e}")
    
    if not article_text:
        raise HTTPException(status_code=400, detail=f"No article text extracted")
    
    return analyze_text(article_text)