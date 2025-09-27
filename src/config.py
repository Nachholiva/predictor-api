import os
from typing import Optional


class Config:

    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8002"))
    API_DEBUG: bool = os.getenv("API_DEBUG", "false").lower() == "true"

    BINANCE_API_URL: str = "https://api.binance.com/api/v3"
    BINANCE_DATA_URL: str = "https://data-api.binance.vision/api/v3"
    BINANCE_USE_DATA_API: bool = os.getenv("BINANCE_USE_DATA_API", "true").lower() == "true"
    BINANCE_TIMEOUT: int = int(os.getenv("BINANCE_TIMEOUT", "10"))
    USER_AGENT: str = os.getenv("USER_AGENT", "CryptoPredictor/1.0 (+https://localhost) Python-requests")

    DEFAULT_SYMBOL: str = "BTCUSDT"
    DEFAULT_INTERVAL: str = "5m"
    DEFAULT_LOOKBACK_HOURS: int = 24

    SUPPORTED_INTERVALS: list = ['5m', '15m', '30m', '1h', '6h', '12h', '1d', '1M']

    LOOKBACK_MIN_HOURS: dict = {
        '5m': 1,
        '15m': 1,
        '30m': 1,
        '1h': 2,
        '6h': 12,
        '12h': 24,
        '1d': 120,
        '1M': 2880,
    }
    LOOKBACK_MAX_HOURS: dict = {
        '5m': 168,
        '15m': 720,
        '30m': 720,
        '1h': 2160,
        '6h': 4320,
        '12h': 8760,
        '1d': 17520,
        '1M': 43800,
    }

    MAX_SYMBOLS_COMPARE: int = int(os.getenv("MAX_SYMBOLS_COMPARE", "10"))

    MAX_REQUESTS_PER_MINUTE: int = 60
    RATE_LIMIT_BACKOFF_FACTOR: float = 2.0
    MAX_RETRIES: int = 5
    SOFT_WEIGHT_LIMIT_1M: int = int(os.getenv("SOFT_WEIGHT_LIMIT_1M", "1100"))
    ENABLE_JITTER_BACKOFF: bool = os.getenv("ENABLE_JITTER_BACKOFF", "true").lower() == "true"

    ENABLE_TIME_SYNC: bool = os.getenv("ENABLE_TIME_SYNC", "false").lower() == "true"
    TIME_SYNC_TTL_SECONDS: int = int(os.getenv("TIME_SYNC_TTL_SECONDS", "300"))
    DEFAULT_RECV_WINDOW_MS: int = int(os.getenv("DEFAULT_RECV_WINDOW_MS", "5000"))

    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "false").lower() == "true"
    CACHE_TTL_SECONDS: int = 300

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


config = Config()
