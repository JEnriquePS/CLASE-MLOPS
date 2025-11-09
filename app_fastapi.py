import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # quieter TF logs; set before importing TF

import re
import pickle
from typing import List, Literal
import sys
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import model_from_json
import tensorflow.keras.preprocessing.text as tf_text
#import tensorflow.keras.preprocessing.sequence as tf_seq
import uvicorn

# ---------- Config ----------
sys.modules.setdefault('keras.preprocessing.text', tf_text)
MODEL_DIR = os.getenv("MODEL_DIR", "models")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pickle")
MODEL_JSON_PATH = os.path.join(MODEL_DIR, "model.json")
MODEL_WEIGHTS_PATH = os.path.join(MODEL_DIR, ".model.weights.h5")  # keep your path; change if needed
MAXLEN = 1000
LABELS: List[Literal["Detractor", "Pasivo", "Promotor"]] = ["Detractor", "Pasivo", "Promotor"]

app = FastAPI(title="NPS LSTM API", version="1.0.0")

# Will be populated at startup
app.state.tokenizer = None
app.state.model = None

# ---------- Esquema ----------
class PredictIn(BaseModel):
    comentario: str

class PredictOut(BaseModel):
    pred: Literal["Detractor", "Pasivo", "Promotor"]
    probs: List[float]  # model softmax outputs [p0, p1, p2]

# Prometheus


# --- NEW: imports for metrics ---
from prometheus_client import Counter, Histogram, make_asgi_app

# ---------- Metrics (NEW) ----------
REQUESTS = Counter(
    "http_requests_total", "Total HTTP requests", ["path", "method", "status"]
)
PREDICTIONS = Counter(
    "nps_predictions_total", "Total predictions by label", ["label"]
)
INFERENCE_LATENCY = Histogram(
    "model_inference_seconds", "Time spent in model.predict (seconds)",
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5)
)

# Expose /metrics without writing our own endpoint
app.mount("/metrics", make_asgi_app())

# ---------- Request metrics middleware (NEW) ----------
@app.middleware("http")
async def _metrics_middleware(request, call_next):
    # avoid self-scrape noise
    if request.url.path == "/metrics":
        return await call_next(request)
    response = await call_next(request)
    try:
        REQUESTS.labels(path=request.url.path, method=request.method, status=str(response.status_code)).inc()
    except Exception:
        # be safe—never break the request because of metrics
        pass
    return response




# ---------- Helpers ----------
_clean_re = re.compile(r"[^A-Za-z0-9\s]")

def _preprocess(text: str) -> List[List[int]]:
    t = text.lower()
    t = _clean_re.sub("", t)
    seq = app.state.tokenizer.texts_to_sequences([t])
    return pad_sequences(seq, maxlen=MAXLEN, padding="pre")

def _predict_label(probs: np.ndarray) -> str:
    idx = int(np.argmax(probs))
    return LABELS[idx]

# ---------- Lifecycle ----------
@app.on_event("startup")
def load_artifacts():
    # Load tokenizer
    with open(TOKENIZER_PATH, "rb") as f:
        # ensure pickle can find modules referenced by the saved Tokenizer (aliases to tf.keras)

        #sys.modules.setdefault('keras.preprocessing.sequence', tf_seq)
        #sys.modules.setdefault('keras.preprocessing', __import__('tensorflow.keras.preprocessing', fromlist=['']))
        app.state.tokenizer = pickle.load(f)

    # Load model from JSON + weights
    with open(MODEL_JSON_PATH, "r") as f:
        model_json = f.read()
    model = model_from_json(model_json)
    model.load_weights(MODEL_WEIGHTS_PATH)

    # Optionally do a warm-up (tiny dummy input) so first request is fast
    _ = model.predict(pad_sequences([[0]], maxlen=MAXLEN, padding="pre"))

    app.state.model = model

# ---------- Endpoints ----------
@app.get("/health")
def health():
    ok = (app.state.model is not None) and (app.state.tokenizer is not None)
    return {
        "status": "ok" if ok else "not_ready",
        "model_loaded": app.state.model is not None,
        "tokenizer_loaded": app.state.tokenizer is not None,
        "maxlen": MAXLEN,
        "labels": LABELS,
    }

@app.post("/predict", response_model=PredictOut, summary="Predecir NPS desde un comentario")
def predict(inp: PredictIn):
    if not inp.comentario or not inp.comentario.strip():
        # Count 400s too (middleware will count but we can early-return if you like)
        raise HTTPException(status_code=400, detail="Comentario vacío.")

    X = _preprocess(inp.comentario)

    # --- NEW: time only the model inference ---
    with INFERENCE_LATENCY.time():
        probs = app.state.model.predict(X)[0].tolist()

    pred = _predict_label(np.array(probs))
    try:
        PREDICTIONS.labels(label=pred).inc()
    except Exception:
        pass

    return PredictOut(pred=pred, probs=[float(p) for p in probs])

# uvicorn app_fastapi:app --host 0.0.0.0 --port 8000
