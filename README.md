<!-- README TOP -->
<!-- ReadME Title -->
# NewsLens - Read Bias & Tone at a Glance

> In an age of short attention spans, many people tend to skim articles and may overlook subtle tactics used by authors to introduce bias or shape a particular narrative. NewsLens helps uncover these hidden elements by identifying biased or emotionally charged language in news articles, text that might be missed during a quick read.
---

## About The Project

**NewsLens** uses NLP (Natural Language Processing) to analyze news articles or pasted text. 
It highlights: 
- Mood (positive, negative, neutral) 
- Potential political bias (liberal, conservative, neutral) 
- Charged words (meant to sway the reader emotionally).


---

## Features

- **Sentiment Detection** : see if text is positive, negative, or neutral  
- **Charged Word Highlighting** : spot emotionally loaded language instantly  
- **Bias Classification** : AI model suggests political leanings of text  
- **Flexible Input** : analyze by pasting text *or* dropping in a news URL  
- **Clear Results** : outputs an overall score (Sentiment, Bias, Charged Words), with the top 3 positive/negative sentences displayed, along with the top 3 potentially biased setences.


---

## Tech Stack

- **Frontend**: React, TypeScript, Bootstrap
- **Backend**: FastAPI, Python 3.12
- **NLP Models**: HuggingFace Transformers (`cardiffnlp/twitter-roberta-base-sentiment-latest, facebook/bart-large-mnli`)

---

## Getting Started (Run it on your own computer)

### Prerequisites

- Node.js & npm
- Python 3.12+

### Installation

```bash
git clone https://github.com/georgejose10/news-nlp.git
cd news-nlp
```

#### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

#### Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## Usage

- Open your browser at `localhost:5173`
- Paste a news URL or enter your own text
- Click "Analyze"
- See sentiment breakdown, bias hints, and charged language


---

## Roadmap

- [ ] Created Basic FastAPI app with predict endpoints for sentiment analysis
- [ ] Added /chargedWords endpoint to display charged words present
- [ ] Added /bias endpoint for bias analysis
- [ ] Added /analyze-text and /analyze-url endpoints that returns sentiment, bias and charged words
- [ ] Added top 3 positve/negative and biased sentences 
- [ ] Added a simple frontend (React & Bootstrap)

---

## Acknowledgments

- https://huggingface.co/blog/sentiment-analysis-python
- HuggingFace Transformers
- Cardiff NLP models
- Bootstrap UI

---


