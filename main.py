"""
main.py — Essay Score Prediction API (local FastAPI endpoint)
IT9204 | Student 202507410 Faiza Malik
Run: python main.py
Test: http://127.0.0.1:8000/docs
"""
import re, json, joblib
import numpy as np
import nltk, textstat
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# Load model files from models/ folder
model         = joblib.load("models/model.pkl")
feature_names = json.load(open("models/feature_names.json"))
score_meta    = json.load(open("models/score_meta.json"))
print(f"Model loaded: {score_meta['model_name']}")
print(f"Features    : {feature_names}")

def extract_features(text):
    text = re.sub(r"\s+", " ", str(text).strip())
    sentences  = sent_tokenize(text)
    words      = [w for w in word_tokenize(text.lower()) if w.isalpha()]
    wc         = len(words)
    sl         = [len(word_tokenize(s)) for s in sentences]
    try:
        fe = textstat.flesch_reading_ease(text)
        fk = textstat.flesch_kincaid_grade(text)
        gf = textstat.gunning_fog(text)
    except:
        fe, fk, gf = 50.0, 8.0, 10.0
    return {
        "word_count"       : wc,
        "char_count"       : len(text),
        "sentence_count"   : max(len(sentences), 1),
        "unique_word_ratio": len(set(words)) / max(wc, 1),
        "avg_word_length"  : float(np.mean([len(w) for w in words]))
                             if words else 0.0,
        "avg_sentence_len" : float(np.mean(sl)) if sl else 0.0,
        "long_sent_ratio"  : sum(1 for l in sl if l > 20)
                             / max(len(sl), 1),
        "flesch_ease"      : fe,
        "flesch_grade"     : fk,
        "gunning_fog"      : gf,
    }

app = FastAPI(
    title       = "Essay Score Prediction API",
    description = "IT9204 Project — deterministic ML scoring for essay grading agent",
    version     = "1.0",
)

class EssayRequest(BaseModel):
    essay_text: str

    class Config:
        json_schema_extra = {
            "example": {
                "essay_text": "The water cycle is a natural process. Water evaporates from the ocean when heated by the sun. It rises into the atmosphere and forms clouds. Then it rains and the water returns to the ground."
            }
        }

@app.get("/")
def root():
    return {
        "status"       : "running",
        "model"        : score_meta["model_name"],
        "score_range"  : "0.0 to 10.0 (normalised from 1-6)",
        "feature_count": len(feature_names),
    }

@app.get("/features")
def get_features():
    return {"features": feature_names}

@app.post("/predict")
def predict(req: EssayRequest):
    """
    Predict a numeric score for a student essay.
    Score is deterministic — same essay always returns same score.
    The LLM agent receives this score to explain, never to estimate.
    """
    essay = req.essay_text.strip()
    if not essay:
        return {"error": "essay_text cannot be empty"}
    if len(essay.split()) < 5:
        return {"error": "Essay too short — minimum 5 words"}

    feats = extract_features(essay)
    arr   = np.array([feats[c] for c in feature_names]).reshape(1, -1)
    score = round(float(np.clip(
        model.predict(arr)[0], 0.0, 10.0
    )), 2)

    raw_approx = round(score / 10 * 5 + 1, 1)

    return {
        "predicted_score"    : score,
        "predicted_score_raw": raw_approx,
        "score_min"          : 0.0,
        "score_max"          : 10.0,
        "model_used"         : score_meta["model_name"],
        "rmse"               : score_meta.get("rmse", "N/A"),
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)