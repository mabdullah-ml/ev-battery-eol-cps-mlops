"""Unit & evaluation tests for EV battery degradation telemetry, RUL predictor, and FastAPI."""

from __future__ import annotations

import unittest

from ev_cps.api import TelemetryPredictionRequest, health, model_metrics, predict_eol
from ev_cps.model import classify_eol_alert, train_and_evaluate_fleet
from ev_cps.telemetry import simulate_battery_pack


class TestEVBatteryCPS(unittest.TestCase):
    def test_simulated_pack_exhibits_capacity_fade_and_resistance_growth(self) -> None:
        pack = simulate_battery_pack("EV-TEST-01", total_eol_cycles=450, seed=11)
        self.assertGreater(pack[0].soh_ratio, pack[-1].soh_ratio)
        self.assertLess(pack[0].internal_resistance_mohm, pack[-1].internal_resistance_mohm)

    def test_trained_rul_model_achieves_high_r2_on_unseen_packs(self) -> None:
        _, metrics = train_and_evaluate_fleet()
        self.assertGreaterEqual(metrics.r2_score, 0.80)
        self.assertLessEqual(metrics.mae_cycles, 60.0)

    def test_eol_alert_classification_and_inference_api(self) -> None:
        self.assertEqual(health()["status"], "ok")
        self.assertIn("r2_score", model_metrics())
        self.assertEqual(classify_eol_alert(0.79, 15.0), "CRITICAL_EOL_REPLACEMENT")
        self.assertEqual(classify_eol_alert(0.95, 380.0), "HEALTHY")

        res = predict_eol(
            TelemetryPredictionRequest(
                pack_id="EV-PACK-999",
                cycle_index=460,
                soh_ratio=0.79,
                internal_resistance_mohm=52.0,
                peak_temp_c=39.5,
                charge_c_rate=1.2,
            )
        )
        self.assertEqual(res["eol_status"], "CRITICAL_EOL_REPLACEMENT")


if __name__ == "__main__":
    unittest.main()
