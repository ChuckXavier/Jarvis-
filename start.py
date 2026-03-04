"""
JARVIS V2 - Combined Startup
===============================
Runs BOTH the daily scheduler AND the API server simultaneously.
- Scheduler: Runs the trading pipeline at 10:00 AM ET daily
- API Server: Serves live data to the React dashboard on port 8000

This is the main entry point for Railway deployment.
"""

import os
import sys
import threading
import time
from loguru import logger

os.makedirs("logs", exist_ok=True)
logger.add("logs/jarvis.log", rotation="10 MB", retention="30 days", level="INFO")


def run_api_server():
    """Run the FastAPI server in a background thread."""
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting API server on port {port}...")
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, log_level="warning")


def run_scheduler():
    """Run the daily trading scheduler."""
    logger.info("Starting scheduler...")
    from scheduler import run_scheduler as start_scheduler
    start_scheduler()


def main():
    print()
    print("=" * 60)
    print("  J A R V I S   V 2")
    print("  Autonomous ETF Alpha Engine")
    print("  Starting API Server + Daily Scheduler")
    print("=" * 60)
    print()

    # Validate config
    from config.settings import validate_config
    missing = validate_config()
    if missing:
        logger.error(f"Missing environment variables: {missing}")
        print(f"  ❌ Missing: {missing}")
        print("  Set these in Railway dashboard → Variables tab")
        sys.exit(1)

    logger.info("Configuration validated ✓")

    # Start API server in a background thread
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()
    logger.info("API server thread started ✓")

    # Give API server a moment to start
    time.sleep(2)

    port = int(os.environ.get("PORT", 8000))
    logger.info(f"API server running on port {port}")
    logger.info("Scheduler starting — pipeline runs at 10:00 AM ET daily")

    # Run scheduler in the main thread (blocking)
    run_scheduler()


if __name__ == "__main__":
    main()
