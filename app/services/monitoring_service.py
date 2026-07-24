import json
import logging
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import psutil
from sqlalchemy.orm import Session

from app.middleware.latency_middleware import get_latency_samples

logger = logging.getLogger(__name__)

DRIFT_REPORT_PATH = Path("artifacts/drift_report.json")


def get_system_metrics() -> Dict[str, Any]:
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(".")
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory": {
            "total_mb": round(memory.total / (1024 ** 2), 1),
            "used_mb": round(memory.used / (1024 ** 2), 1),
            "percent": memory.percent,
        },
        "disk": {
            "total_gb": round(disk.total / (1024 ** 3), 1),
            "used_gb": round(disk.used / (1024 ** 3), 1),
            "percent": disk.percent,
        },
    }


def get_latency_metrics() -> Dict[str, Any]:
    samples = get_latency_samples()
    if not samples:
        return {"count": 0}

    durations = sorted(s["duration_ms"] for s in samples)

    def _percentile(p: float) -> float:
        idx = min(len(durations) - 1, int(len(durations) * p))
        return round(durations[idx], 2)

    return {
        "count": len(durations),
        "avg_ms": round(mean(durations), 2),
        "p50_ms": _percentile(0.50),
        "p95_ms": _percentile(0.95),
        "p99_ms": _percentile(0.99),
    }


def get_drift_report() -> Dict[str, Any]:
    if not DRIFT_REPORT_PATH.exists():
        return {"status": "not_available"}
    try:
        return json.loads(DRIFT_REPORT_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Failed to read drift report: {e}")
        return {"status": "error", "detail": str(e)}


def get_training_history(db: Session, limit: int = 20) -> List[Dict[str, Any]]:
    from app.models import TrainingHistory

    rows = (
        db.query(TrainingHistory)
        .order_by(TrainingHistory.training_date.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "run_id": r.run_id,
            "training_date": r.training_date.isoformat() if r.training_date else None,
            "duration_seconds": r.duration_seconds,
            "vocab_size": r.vocab_size,
            "dataset_rows": r.dataset_rows,
            "precision_at_5": r.metrics_precision_at_5,
            "coverage": r.metrics_coverage,
            "model_version": r.model_version,
        }
        for r in rows
    ]
