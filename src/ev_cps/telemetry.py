"""Physics-informed Li-ion EV battery degradation telemetry simulator (NASA/Oxford-style curves)."""

from __future__ import annotations

import random
from dataclasses import dataclass

EOL_SOH_THRESHOLD = 0.80


@dataclass(frozen=True)
class BatteryCycleTelemetry:
    pack_id: str
    cycle_index: int
    soh_ratio: float
    internal_resistance_mohm: float
    peak_temp_c: float
    charge_c_rate: float
    true_rul_cycles: float


def simulate_battery_pack(
    pack_id: str,
    total_eol_cycles: int = 500,
    seed: int = 42,
) -> list[BatteryCycleTelemetry]:
    """Generate realistic capacity fade (SoH 1.0 -> 0.80) and resistance growth telemetry."""
    rng = random.Random(seed)
    records: list[BatteryCycleTelemetry] = []
    base_res = rng.uniform(28.0, 32.0)

    for cycle in range(1, total_eol_cycles + 1, 5):
        progress = cycle / float(total_eol_cycles)
        # Non-linear knee acceleration near EOL
        soh = max(0.76, 1.0 - (0.16 * progress + 0.04 * (progress**2)) + rng.uniform(-0.002, 0.002))
        resistance = base_res + (18.0 * progress) + (6.0 * (progress**2)) + rng.uniform(-0.3, 0.3)
        temp_c = 30.0 + (8.5 * progress) + rng.uniform(-0.8, 0.8)
        c_rate = round(rng.uniform(0.8, 1.6), 2)
        rul = max(0.0, float(total_eol_cycles - cycle))

        records.append(
            BatteryCycleTelemetry(
                pack_id=pack_id,
                cycle_index=cycle,
                soh_ratio=round(soh, 4),
                internal_resistance_mohm=round(resistance, 2),
                peak_temp_c=round(temp_c, 2),
                charge_c_rate=c_rate,
                true_rul_cycles=rul,
            )
        )
    return records


def generate_fleet_dataset(num_packs: int = 8) -> list[BatteryCycleTelemetry]:
    all_records: list[BatteryCycleTelemetry] = []
    for idx in range(num_packs):
        eol_cycles = 440 + (idx * 25)
        all_records.extend(
            simulate_battery_pack(
                pack_id=f"EV-PACK-{idx + 1:03d}",
                total_eol_cycles=eol_cycles,
                seed=100 + idx,
            )
        )
    return all_records
