"""EV Battery Remaining Useful Life (RUL / EOL) predictor with PyTorch & lightweight edge backends."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ev_cps.telemetry import EOL_SOH_THRESHOLD, BatteryCycleTelemetry, generate_fleet_dataset


@dataclass(frozen=True)
class ModelEvalMetrics:
    mae_cycles: float
    rmse_cycles: float
    r2_score: float
    train_samples: int
    val_samples: int


def _extract_features(rec: BatteryCycleTelemetry) -> list[float]:
    soh_margin = max(0.0, rec.soh_ratio - EOL_SOH_THRESHOLD) / 0.20
    fade_per_cycle = (1.0 - rec.soh_ratio) / max(1.0, float(rec.cycle_index))
    physics_rul_est = min(1.5, (soh_margin * 0.20) / max(1e-4, fade_per_cycle) / 600.0)
    return [
        rec.cycle_index / 600.0,
        soh_margin,
        rec.internal_resistance_mohm / 60.0,
        rec.peak_temp_c / 50.0,
        physics_rul_est,
    ]


class BatteryRULPredictor:
    """Standardized linear/ridge RUL regressor runnable on edge gateways without heavy GPU deps."""

    def __init__(self) -> None:
        self.weights: list[float] = [0.0] * 5
        self.bias: float = 0.0
        self.is_trained: bool = False

    def fit(
        self,
        records: list[BatteryCycleTelemetry],
        epochs: int = 800,
        lr: float = 0.08,
        l2_reg: float = 1e-4,
    ) -> None:
        xs = [_extract_features(r) for r in records]
        ys = [r.true_rul_cycles / 600.0 for r in records]
        n = max(1, len(xs))
        self.weights = [0.0] * 5
        self.bias = 0.0

        for _ in range(epochs):
            grad_w = [0.0] * 5
            grad_b = 0.0
            for x, y in zip(xs, ys, strict=False):
                pred = sum(w * xi for w, xi in zip(self.weights, x, strict=False)) + self.bias
                err = pred - y
                for j in range(5):
                    grad_w[j] += (2.0 / n) * err * x[j]
                grad_b += (2.0 / n) * err
            for j in range(5):
                self.weights[j] -= lr * (grad_w[j] + l2_reg * self.weights[j])
            self.bias -= lr * grad_b

        self.is_trained = True

    def predict_rul(self, rec: BatteryCycleTelemetry) -> float:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling predict_rul().")
        x = _extract_features(rec)
        raw = sum(w * xi for w, xi in zip(self.weights, x, strict=False)) + self.bias
        return round(max(0.0, raw * 600.0), 1)

    def evaluate(self, records: list[BatteryCycleTelemetry]) -> tuple[float, float, float]:
        preds = [self.predict_rul(r) for r in records]
        trues = [r.true_rul_cycles for r in records]
        n = max(1, len(trues))
        mae = sum(abs(p - y) for p, y in zip(preds, trues, strict=False)) / n
        rmse = math.sqrt(sum((p - y) ** 2 for p, y in zip(preds, trues, strict=False)) / n)
        mean_y = sum(trues) / n
        ss_tot = sum((y - mean_y) ** 2 for y in trues)
        ss_res = sum((y - p) ** 2 for p, y in zip(preds, trues, strict=False))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
        return round(mae, 2), round(rmse, 2), round(r2, 4)


def build_pytorch_rul_network(input_dim: int = 5):
    """Build PyTorch MLP RUL predictor when PyTorch (`pip install -e .[ml]`) is installed."""
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(input_dim, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )


def train_and_evaluate_fleet() -> tuple[BatteryRULPredictor, ModelEvalMetrics]:
    dataset = generate_fleet_dataset(num_packs=8)
    train_set = [r for r in dataset if r.pack_id not in ("EV-PACK-003", "EV-PACK-006")]
    val_set = [r for r in dataset if r.pack_id in ("EV-PACK-003", "EV-PACK-006")]

    model = BatteryRULPredictor()
    model.fit(train_set, epochs=1200, lr=0.12)
    mae, rmse, r2 = model.evaluate(val_set)
    metrics = ModelEvalMetrics(
        mae_cycles=mae,
        rmse_cycles=rmse,
        r2_score=r2,
        train_samples=len(train_set),
        val_samples=len(val_set),
    )
    return model, metrics


def classify_eol_alert(soh_ratio: float, predicted_rul_cycles: float) -> str:
    if soh_ratio <= EOL_SOH_THRESHOLD or predicted_rul_cycles <= 30.0:
        return "CRITICAL_EOL_REPLACEMENT"
    if soh_ratio <= 0.85 or predicted_rul_cycles <= 100.0:
        return "MAINTENANCE_WARNING"
    return "HEALTHY"


def main() -> None:
    _, metrics = train_and_evaluate_fleet()
    print(
        f"EV Battery EOL Model Evaluation | Train={metrics.train_samples} Val={metrics.val_samples} "
        f"| MAE={metrics.mae_cycles} cycles | RMSE={metrics.rmse_cycles} | R2={metrics.r2_score}"
    )


if __name__ == "__main__":
    main()
