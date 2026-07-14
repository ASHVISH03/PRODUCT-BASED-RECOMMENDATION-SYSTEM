"""
Common Utilities
=================
Shared utility functions used across all modules.

Includes:
- File I/O helpers (YAML, JSON, pickle)
- Directory management
- Timing decorators
- Data formatting
- Hash computation
- Project root resolution
"""

import hashlib
import json
import logging
import os
import pickle
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import yaml

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------
# Project Root
# ---------------------------------------------------------------

def get_project_root() -> Path:
    """
    Return the absolute path to the project root directory.

    The root is determined by walking up from this file until
    a directory containing 'requirements.txt' is found.
    Falls back to the current working directory if not found.
    """
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "requirements.txt").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT = get_project_root()


def resolve_path(relative_path: str) -> Path:
    """
    Resolve a path relative to the project root.

    Args:
        relative_path: Path string relative to project root.

    Returns:
        Absolute Path object.
    """
    return PROJECT_ROOT / relative_path


# ---------------------------------------------------------------
# Directory Management
# ---------------------------------------------------------------

def create_directories(paths: List[Union[str, Path]]) -> None:
    """
    Create multiple directories if they do not exist.

    Args:
        paths: List of directory paths (str or Path) to create.
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a single directory exists, creating it if needed.

    Args:
        path: Directory path.

    Returns:
        The Path object for the created/existing directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------

def read_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read and parse a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    if not isinstance(content, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(content)}")
    return content


def write_yaml(path: Union[str, Path], content: Dict[str, Any]) -> None:
    """
    Write a dictionary to a YAML file.

    Args:
        path: Output file path.
        content: Dictionary to serialize.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(content, f, default_flow_style=False, allow_unicode=True, indent=2)


# ---------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------

def load_json(path: Union[str, Path]) -> Any:
    """
    Load a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed Python object (dict, list, etc.).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(
    path: Union[str, Path],
    content: Any,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """
    Save an object to a JSON file.

    Args:
        path: Output file path.
        content: JSON-serializable Python object.
        indent: JSON indentation level.
        ensure_ascii: If True, escape non-ASCII characters.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=indent, ensure_ascii=ensure_ascii, default=str)


# ---------------------------------------------------------------
# Pickle I/O
# ---------------------------------------------------------------

def save_pickle(path: Union[str, Path], obj: Any) -> None:
    """
    Serialize an object to a pickle file.

    Args:
        path: Output file path (.pkl).
        obj: Python object to pickle (must be picklable).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Union[str, Path]) -> Any:
    """
    Deserialize an object from a pickle file.

    Args:
        path: Path to the .pkl file.

    Returns:
        Deserialized Python object.

    Raises:
        FileNotFoundError: If the pickle file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------
# File Information
# ---------------------------------------------------------------

def get_file_size_mb(path: Union[str, Path]) -> float:
    """
    Return the size of a file in megabytes.

    Args:
        path: Path to the file.

    Returns:
        File size in MB, rounded to 2 decimal places.
    """
    p = Path(path)
    if not p.exists():
        return 0.0
    return round(p.stat().st_size / (1024 * 1024), 2)


def get_file_hash(path: Union[str, Path], algorithm: str = "md5") -> str:
    """
    Compute the hash of a file for data versioning.

    Args:
        path: Path to the file.
        algorithm: Hash algorithm ('md5', 'sha256').

    Returns:
        Hex digest string.
    """
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------

def get_timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """
    Return the current datetime as a formatted string.

    Args:
        fmt: strftime format string.

    Returns:
        Formatted timestamp string.
    """
    return datetime.now().strftime(fmt)


def get_iso_timestamp() -> str:
    """Return the current datetime in ISO 8601 format."""
    return datetime.now().isoformat()


# ---------------------------------------------------------------
# Timing Decorator
# ---------------------------------------------------------------

def timer(logger: Optional[logging.Logger] = None) -> Callable[[F], F]:
    """
    Decorator that measures and optionally logs function execution time.

    Args:
        logger: Logger instance. If provided, logs timing at INFO level.

    Returns:
        Decorator that wraps the target function.

    Usage:
        @timer(logger=my_logger)
        def expensive_function():
            ...
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            message = f"{func.__qualname__} completed in {elapsed:.3f}s"
            if logger:
                logger.info(message)
            return result
        return wrapper  # type: ignore[return-value]
    return decorator


def measure_time(func: F) -> F:
    """
    Simple decorator that adds `_duration_seconds` attribute to the return value.
    If the return value is a dict, the key 'duration_seconds' is inserted.

    Use @timer() for logging; use this when you need the timing data in code.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = round(time.perf_counter() - start, 4)
        if isinstance(result, dict):
            result["duration_seconds"] = elapsed
        return result
    return wrapper  # type: ignore[return-value]


# ---------------------------------------------------------------
# Data Formatting
# ---------------------------------------------------------------

def format_size(size_bytes: int) -> str:
    """
    Format a byte count into a human-readable string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Formatted string (e.g., '10.5 MB').
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate a string to a maximum length, appending a suffix.

    Args:
        text: Input string.
        max_length: Maximum character count including suffix.
        suffix: String appended when truncated.

    Returns:
        Original string if within limit, otherwise truncated version.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float, returning default on failure.

    Args:
        value: Value to convert.
        default: Fallback if conversion fails.

    Returns:
        Float value or default.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert a value to int, returning default on failure.

    Args:
        value: Value to convert.
        default: Fallback if conversion fails.

    Returns:
        Integer value or default.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------
# List / Collection Utilities
# ---------------------------------------------------------------

def deduplicate(items: List[Any], key: Optional[Callable] = None) -> List[Any]:
    """
    Remove duplicates from a list while preserving order.

    Args:
        items: Input list.
        key: Optional callable to extract the comparison key.

    Returns:
        Deduplicated list in original order.
    """
    seen = set()
    result = []
    for item in items:
        k = key(item) if key else item
        if k not in seen:
            seen.add(k)
            result.append(item)
    return result


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of a given size.

    Args:
        lst: Input list.
        chunk_size: Maximum items per chunk.

    Returns:
        List of chunks.
    """
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


# ---------------------------------------------------------------
# Environment Helpers
# ---------------------------------------------------------------

def is_development() -> bool:
    """Return True if APP_ENV is 'development' (default)."""
    return os.environ.get("APP_ENV", "development").lower() == "development"


def is_production() -> bool:
    """Return True if APP_ENV is 'production'."""
    return os.environ.get("APP_ENV", "development").lower() == "production"


def get_env(key: str, default: Any = None, required: bool = False) -> Any:
    """
    Read an environment variable with optional enforcement.

    Args:
        key: Environment variable name.
        default: Value returned if the variable is not set.
        required: If True, raises ValueError when variable is missing.

    Returns:
        Environment variable value or default.

    Raises:
        ValueError: If required=True and the variable is not set.
    """
    value = os.environ.get(key)
    if value is None:
        if required:
            raise ValueError(
                f"Required environment variable '{key}' is not set. "
                f"Add it to your .env file."
            )
        return default
    return value


# ---------------------------------------------------------------
# File Metadata Helpers
# ---------------------------------------------------------------

def get_file_hash(path: Union[str, Path], algorithm: str = "md5") -> str:
    """
    Compute a hex digest hash of a file's contents.

    Useful for dataset versioning and change detection.

    Args:
        path: Path to the file.
        algorithm: Hash algorithm name (default 'md5').
                   Supported: 'md5', 'sha1', 'sha256'.

    Returns:
        Hex digest string (e.g. 'a3f8c9d1b2e4').

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot hash non-existent file: {path}")

    h = hashlib.new(algorithm)
    # Read in 64KB blocks to avoid loading large files into memory
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def get_file_size_mb(path: Union[str, Path]) -> float:
    """
    Return the size of a file in megabytes.

    Args:
        path: Path to the file.

    Returns:
        File size in MB (float), or 0.0 if file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def get_iso_timestamp() -> str:
    """
    Return the current UTC time as an ISO 8601 string.

    Returns:
        e.g. '2024-01-15T10:30:45Z'
    """
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
