from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from altcredit.config import MODEL_LABELS
from altcredit.comparison import evaluate_uploaded_testing_data
from altcredit.scoring import load_model_bundle, score_features, score_trace
from altcredit.templates import score_json_template, testing_json_template


app = FastAPI(title="Alternative Credit MVP", version="0.1.0")


class ScoreRequest(BaseModel):
    model_type: str = Field(..., examples=["retail_active"])
    features: dict[str, Any]


class TestingDataRequest(BaseModel):
    model_type: str = Field(..., examples=["retail_active"])
    records: list[dict[str, Any]]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def models() -> dict[str, str]:
    return MODEL_LABELS


@app.get("/templates/score/{model_type}")
def score_template(model_type: str) -> dict[str, Any]:
    bundle = _bundle_or_404(model_type)
    return score_json_template(bundle)


@app.get("/templates/testing-data/{model_type}")
def testing_data_template(model_type: str) -> dict[str, Any]:
    bundle = _bundle_or_404(model_type)
    return testing_json_template(bundle)


@app.post("/score")
def score(request: ScoreRequest) -> dict[str, Any]:
    bundle = _bundle_or_404(request.model_type)
    return score_features(bundle, request.features)


@app.post("/score/trace")
def trace(request: ScoreRequest) -> dict[str, Any]:
    bundle = _bundle_or_404(request.model_type)
    return score_trace(bundle, request.features)


@app.post("/evaluate/testing-data")
def evaluate_testing_data(request: TestingDataRequest) -> dict[str, Any]:
    bundle = _bundle_or_404(request.model_type)
    try:
        return evaluate_uploaded_testing_data(bundle, request.records)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _bundle_or_404(model_type: str) -> dict[str, Any]:
    if model_type not in MODEL_LABELS:
        raise HTTPException(status_code=400, detail=f"Unknown model_type: {model_type}")
    try:
        return load_model_bundle(model_type)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
