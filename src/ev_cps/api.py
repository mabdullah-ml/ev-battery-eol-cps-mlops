"""FastAPI inference & fleet health monitoring service for EV battery EOL prediction."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from ev_cps.model import classify_eol_alert, train_and_evaluate_fleet
from ev_cps.telemetry import BatteryCycleTelemetry

app = FastAPI(
    title="EV Battery EOL & CPS Telemetry MLOps API",
    version="1.0.0",
)

_MODEL, _METRICS = train_and_evaluate_fleet()


class TelemetryPredictionRequest(BaseModel):
    pack_id: str = Field(default="EV-PACK-LIVE-01")
    cycle_index: int = Field(..., ge=0, le=5000)
    soh_ratio: float = Field(..., ge=0.5, le=1.05)
    internal_resistance_mohm: float = Field(..., ge=5.0, le=200.0)
    peak_temp_c: float = Field(..., ge=-20.0, le=90.0)
    charge_c_rate: float = Field(default=1.0, ge=0.1, le=6.0)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ev-battery-eol-cps-mlops"}


@app.get("/api/v1/model-metrics")
def model_metrics() -> dict[str, Any]:
    return asdict(_METRICS)


@app.post("/api/v1/predict-eol")
def predict_eol(req: TelemetryPredictionRequest) -> dict[str, Any]:
    rec = BatteryCycleTelemetry(
        pack_id=req.pack_id,
        cycle_index=req.cycle_index,
        soh_ratio=req.soh_ratio,
        internal_resistance_mohm=req.internal_resistance_mohm,
        peak_temp_c=req.peak_temp_c,
        charge_c_rate=req.charge_c_rate,
        true_rul_cycles=0.0,
    )
    rul = _MODEL.predict_rul(rec)
    status = classify_eol_alert(req.soh_ratio, rul)
    return {
        "pack_id": req.pack_id,
        "predicted_rul_cycles": rul,
        "soh_ratio": req.soh_ratio,
        "eol_status": status,
    }
