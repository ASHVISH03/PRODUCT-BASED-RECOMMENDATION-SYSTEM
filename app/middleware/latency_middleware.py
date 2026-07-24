import time
from collections import deque
from typing import Deque, Dict, List

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

_MAX_SAMPLES = 500
_latency_samples: Deque[Dict] = deque(maxlen=_MAX_SAMPLES)


class LatencyTrackingMiddleware(BaseHTTPMiddleware):
    """Records a rolling window of recent request latencies for /monitoring/latency."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        _latency_samples.append(
            {
                "path": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            }
        )
        return response


def get_latency_samples() -> List[Dict]:
    return list(_latency_samples)
