-- TimescaleDB Hypertable Schema for Connected EV Battery Cyber-Physical System (CPS)

CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS ev_battery_telemetry (
    recorded_at TIMESTAMPTZ NOT NULL,
    vehicle_id TEXT NOT NULL,
    pack_id TEXT NOT NULL,
    cycle_index INTEGER NOT NULL,
    soh_ratio DOUBLE PRECISION NOT NULL,
    internal_resistance_mohm DOUBLE PRECISION NOT NULL,
    peak_temp_c DOUBLE PRECISION NOT NULL,
    charge_c_rate DOUBLE PRECISION NOT NULL,
    predicted_rul_cycles DOUBLE PRECISION
);

SELECT create_hypertable('ev_battery_telemetry', 'recorded_at', if_not_exists => TRUE);

CREATE MATERIALIZED VIEW IF NOT EXISTS vw_battery_soh_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', recorded_at) AS bucket_day,
    pack_id,
    AVG(soh_ratio) AS avg_soh,
    MAX(internal_resistance_mohm) AS max_resistance_mohm,
    MAX(peak_temp_c) AS max_temp_c,
    MIN(predicted_rul_cycles) AS min_predicted_rul
FROM ev_battery_telemetry
GROUP BY bucket_day, pack_id;
