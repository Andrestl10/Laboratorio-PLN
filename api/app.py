import mlflow
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel

# ─── Configuración MLflow ────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = "http://ec2-3-95-197-233.compute-1.amazonaws.com:5000/"
RUN_ID              = "58d6a8bbcf6042ff81c443e3b5ad2491"
MODEL_URI           = f"runs:/{RUN_ID}/model"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ─── Cargar modelo desde MLflow ──────────────────────────────────────────────
print(f"Cargando modelo desde MLflow: {MODEL_URI}")
model = mlflow.sklearn.load_model(MODEL_URI)
print("Modelo cargado exitosamente.")

# ─── App FastAPI ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="API Análisis de Sentimientos IMDB",
    description="Predice el sentimiento (positivo/negativo) de reseñas de películas.",
    version="1.0.0"
)

class ReviewRequest(BaseModel):
    review: str

class ReviewResponse(BaseModel):
    review: str
    sentiment: str
    confidence: float

@app.get("/")
def root():
    return {"message": "API de Análisis de Sentimientos - Modelo: TF-IDF + Logistic Regression"}

@app.get("/health")
def health():
    return {"status": "ok", "model": "baseline_tfidf_lr"}

@app.post("/predict", response_model=ReviewResponse)
def predict(request: ReviewRequest):
    review = request.review
    prediction = model.predict([review])[0]
    probability = model.predict_proba([review])[0]

    sentiment = "positivo" if prediction == 1 else "negativo"
    confidence = float(max(probability))

    return ReviewResponse(
        review=review,
        sentiment=sentiment,
        confidence=round(confidence, 4)
    )

@app.post("/predict/batch")
def predict_batch(reviews: list[str]):
    predictions = model.predict(reviews)
    probabilities = model.predict_proba(reviews)

    results = []
    for review, pred, prob in zip(reviews, predictions, probabilities):
        results.append({
            "review": review,
            "sentiment": "positivo" if pred == 1 else "negativo",
            "confidence": round(float(max(prob)), 4)
        })
    return results
