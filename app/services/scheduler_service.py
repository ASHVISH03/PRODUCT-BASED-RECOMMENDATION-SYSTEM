import logging
import os
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler

from app.services.training_service import run_training_pipeline

logger = logging.getLogger(__name__)

_scheduler: Optional[BackgroundScheduler] = None


def start_scheduler() -> Optional[BackgroundScheduler]:
    """
    Start the background retrain scheduler, unless disabled.

    Controlled by env vars:
      ENABLE_RETRAIN_SCHEDULER  — "true"/"false" (default: true)
      RETRAIN_INTERVAL_HOURS    — float, hours between runs (default: 24)

    The interval trigger's first run fires one full interval after startup,
    not immediately — a fresh deployment doesn't retrain before it's served
    any traffic.
    """
    global _scheduler
    if _scheduler is not None:
        return _scheduler

    if os.environ.get("ENABLE_RETRAIN_SCHEDULER", "true").strip().lower() not in ("1", "true", "yes"):
        logger.info("Retrain scheduler disabled (ENABLE_RETRAIN_SCHEDULER).")
        return None

    interval_hours = float(os.environ.get("RETRAIN_INTERVAL_HOURS", "24"))

    scheduler = BackgroundScheduler(daemon=True)
    scheduler.add_job(
        run_training_pipeline,
        trigger="interval",
        hours=interval_hours,
        id="scheduled_retrain",
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    _scheduler = scheduler
    logger.info(f"Retrain scheduler started: every {interval_hours}h")
    return scheduler


def shutdown_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
