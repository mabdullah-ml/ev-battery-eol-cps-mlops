# EV Battery End-of-Life (EOL / RUL) CPS & MLOps Platform

[![CI](https://github.com/mabdullah-ml/ev-battery-eol-cps-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/mabdullah-ml/ev-battery-eol-cps-mlops/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776ab?style=flat-square)](#)
[![PyTorch & FastAPI](https://img.shields.io/badge/MLOps-PyTorch%20%7C%20MLflow%20%7C%20TimescaleDB-ee4c2c?style=flat-square)](#)

> 🌐 **[日本語ドキュメントはこちら (README.ja.md)](./README.ja.md)**

A **Cyber-Physical System (CPS)** IoT telemetry and **EV Li-ion Battery Remaining Useful Life (RUL) / End-of-Life (EOL)** prediction platform combining physics-informed degradation modeling, PyTorch & lightweight edge RUL inference, TimescaleDB hypertable analytics, FastAPI serving, and automated CI evaluation.

---

## Architecture

```mermaid
flowchart LR
    EV["Connected EV BMS<br/>Cycle · SoH · Resistance · Temp"] --> TS[("TimescaleDB Hypertable<br/>Daily Continuous Aggregates")]
    TS --> ML["RUL / EOL Predictor<br/>PyTorch MLP / Edge Regressor"]
    ML --> API["FastAPI Inference Service<br/>/api/v1/predict-eol"]
    API --> ALERT["Fleet Maintenance Alerts<br/>HEALTHY · WARNING · CRITICAL_EOL"]
```

## Quick Start

```bash
pip install -e ".[dev]"
ruff check src tests
pytest -v
python -m ev_cps.model

# Start TimescaleDB + FastAPI inference stack
docker compose up --build
```

## Author

**Abdullah Mamun, Ph.D.** — Senior Generative AI / LLM Engineer · Data Scientist · PMP  
[GitHub](https://github.com/mabdullah-ml) · [LinkedIn](https://www.linkedin.com/in/mamunabdullah01)
