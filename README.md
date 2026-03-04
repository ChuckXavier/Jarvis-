# JARVIS V2 — Autonomous ETF Alpha Engine

A quantitative trading system that generates alpha from 25 ETFs using
four validated signals: Cross-Sectional Momentum, Time-Series Trend,
Mean Reversion, and Volatility Regime Detection.

## Quick Start

1. Set environment variables (see `.env.example`)
2. Run `python main.py` to initialize database and download 10 years of data
3. Data quality checks run automatically

## Architecture

```
DATA → ALPHA → PORTFOLIO → RISK → EXECUTION
```

Built on the proven principles of Renaissance Technologies, D.E. Shaw,
Two Sigma, AQR, and Bridgewater Associates.

## Status: Phase 1 — Data Foundation
