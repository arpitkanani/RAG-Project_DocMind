import logging
import os
import shutil
import sys
from datetime import datetime

LOG_DIRECTORY = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIRECTORY, exist_ok=True)

# Standard format: date_month_year+hour_minute_second.log (e.g. 08_11_2026+15_15_15.log)
LOG_FILE_NAME = f"{datetime.now().strftime('%d_%m_%Y+%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIRECTORY, LOG_FILE_NAME)

file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

# Terminal console handler: restricted to WARNING and ERROR so application internal logs
# do not pollute the terminal, while standard uvicorn request lines remain visible.
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(
    logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
)

# Configure root logger with force=True (Python 3.8+) so uvicorn doesn't override handlers
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler],
    force=True,
)

# Explicitly ensure root logger and uvicorn loggers output to this active file
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
if file_handler not in root_logger.handlers:
    root_logger.addHandler(file_handler)

for _uv_name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
    _uv_logger = logging.getLogger(_uv_name)
    if file_handler not in _uv_logger.handlers:
        _uv_logger.addHandler(file_handler)

# Keep logs/latest.log updated with the current session file path for easy inspection
try:
    latest_log_path = os.path.join(LOG_DIRECTORY, "latest.log")
    with open(latest_log_path, "w", encoding="utf-8") as _f_latest:
        _f_latest.write(f"Active log file: {LOG_FILE_PATH}\nSession started: {datetime.now().isoformat()}\n\n")
except Exception:
    pass

from logging.handlers import RotatingFileHandler

# Dedicated latency log file for all flows (Custom Bots, RAG, Architect, Prompts, etc.)
LATENCY_LOG_PATH = os.path.join(LOG_DIRECTORY, "latency.log")
latency_handler = RotatingFileHandler(
    LATENCY_LOG_PATH,
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
latency_handler.setLevel(logging.INFO)
latency_formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
latency_handler.setFormatter(latency_formatter)

latency_logger = logging.getLogger("latency_telemetry")
latency_logger.setLevel(logging.INFO)
if not latency_logger.handlers:
    latency_logger.addHandler(latency_handler)
    # Also output to the active session's dated log file
    latency_logger.addHandler(file_handler)
latency_logger.propagate = False


def log_latency(flow: str, message: str, **kwargs) -> None:
    """
    Unified latency logging utility across all application flows.
    Writes structured latency records directly to logs/latency.log
    as well as the active session dated log file (logs/<date_month_year>+<time>.log).
    """
    prefix = f"[{flow.upper()}]" if flow else "[LATENCY]"
    formatted_msg = f"{prefix} {message}"
    if kwargs:
        extra_parts = [f"{k}={v}" for k, v in kwargs.items()]
        formatted_msg = f"{formatted_msg} | " + " | ".join(extra_parts)
    latency_logger.info(formatted_msg)


