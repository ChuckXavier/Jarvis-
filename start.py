"""
JARVIS V3 - Railway Entry Point
================================
ARCHITECTURAL NOTE (what changed vs V2):
Startup now (1) validates env vars with a clear failure message, (2) ensures
ALL persistence tables exist — including the new regime_machine and
pipeline_runs tables — before anything runs, (3) warms the stock-universe
cache (non-fatal if Wikipedia is unreachable; the DB cache or seed list takes
over), and (4) exposes a /health endpoint that reports the last pipeline run
and the CURRENT PERSISTED REGIME, so a stuck machine is visible from a
browser instead of being discovered ten weeks later. The dashboard app from
api_server.py is reused if importable; otherwise a minimal FastAPI app serves
/health alone. Set RUN_PIPELINE_ON_START=true to fire one pipeline run
immediately after deploy (default: wait for the 10:00 ET cron).
"""

import os
import sys
import threading

from loguru import logger

os.makedirs("logs", exist_ok=True)
logger.add("logs/jarvis.log", rotation="10 MB", retention="30 days", level="INFO")


def _validate_env() -> bool:
    from config.settings import validate_config
    missing = validate_config()
    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        logger.error("Set them in Railway -> Variables, then redeploy.")
        return False
    return True


def _init_database() -> bool:
    try:
        from data.db import create_all_tables
        create_all_tables()
        logger.info("Core tables ready")
    except Exception as e:
        logger.error(f"Database init failed: {e}")
        return False
    # New V3 tables — each ensure is independent and non-fatal in isolation.
    try:
        from risk.regime import _ensure_tables as ensure_regime
        ensure_regime()
        logger.info("Regime tables ready")
    except Exception as e:
        logger.warning(f"Regime table init: {e}")
    try:
        from scheduler import _ensure_runs_table
        _ensure_runs_table()
    except Exception as e:
        logger.warning(f"pipeline_runs init: {e}")
    try:
        from config.universe import _ensure_table as ensure_universe
        ensure_universe()
    except Exception as e:
        logger.warning(f"universe table init: {e}")
    return True


def _warm_universe():
    """Try to populate the constituents cache. Never fatal."""
    try:
        from config.universe import refresh_universe
        ref = refresh_universe()
        logger.info(f"Universe: {ref['count']} stocks "
                    f"({'refreshed' if ref['refreshed'] else 'cached'})")
    except Exception as e:
        logger.warning(f"Universe warm-up failed (cache/seed will serve): {e}")


def _build_app():
    """Reuse the dashboard app if present; otherwise a minimal health app."""
    app = None
    try:
        from api_server import app as dashboard_app
        app = dashboard_app
        logger.info("Using api_server dashboard app")
    except Exception as e:
        logger.warning(f"api_server unavailable ({e}) — minimal health app")
        from fastapi import FastAPI
        app = FastAPI(title="JARVIS V3")

    @app.get("/health")
    def health():
        from scheduler import get_last_run
        from risk.regime import get_current_regime
        return {
            "status": "alive",
            "version": "v3",
            "regime": get_current_regime(),
            "last_pipeline_run": get_last_run(),
        }
    return app


def main():
    logger.info("=" * 60)
    logger.info("JARVIS V3 — starting")
    logger.info("=" * 60)

    if not _validate_env():
        sys.exit(1)
    if not _init_database():
        sys.exit(1)
    _warm_universe()

    app = _build_app()

    def serve():
        import uvicorn
        port = int(os.getenv("PORT", "8000"))
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")

    threading.Thread(target=serve, daemon=True, name="api").start()
    logger.info("API thread started (/health live)")

    if os.getenv("RUN_PIPELINE_ON_START", "false").lower() == "true":
        logger.info("RUN_PIPELINE_ON_START=true — running pipeline now")
        try:
            from scheduler import run_daily_pipeline
            run_daily_pipeline()
        except Exception as e:
            logger.exception(f"startup pipeline failed: {e}")

    from scheduler import run_scheduler
    run_scheduler()   # blocks: keeps the container alive


if __name__ == "__main__":
    main()
