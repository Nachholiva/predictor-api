from typing import List, Dict, Optional, Union, Annotated
from pydantic import BaseModel, Field, field_validator, StringConstraints
from datetime import datetime
from config import config


class PredictorConfig(BaseModel):
    symbol: str = "BTCUSDT"
    interval: str = "5m"
    lookback_hours: int = 24


class SupportResistance(BaseModel):
    price: float
    percent_from_current: float


class PredictionProbabilities(BaseModel):
    buy: float
    sell: float


class PredictionResult(BaseModel):
    prediction: str
    confidence: float
    probabilities: PredictionProbabilities
    current_price: float
    support: SupportResistance
    resistance: SupportResistance
    volatility_24h: float
    analysis_time: str
    symbol: str
    interval: str
    lookback_hours: int


class TimeframeAnalysis(BaseModel):
    prediction: str
    confidence: float
    probabilities: PredictionProbabilities
    current_price: float
    support: SupportResistance
    resistance: SupportResistance
    volatility_24h: float
    analysis_time: str
    symbol: str
    interval: str
    lookback_hours: int


class MultiTimeframeSummary(BaseModel):
    total_timeframes: int
    buy_signals: int
    sell_signals: int
    overall_bias: str


class MultiTimeframeResult(BaseModel):
    results: Dict[str, TimeframeAnalysis]
    summary: MultiTimeframeSummary


class TradingPairAnalysis(BaseModel):
    symbol: str
    prediction: str
    confidence: float
    support: SupportResistance
    resistance: SupportResistance


class ComparisonResult(BaseModel):
    pairs: List[TradingPairAnalysis]
    interval: str
    analysis_time: str


class MarketDataPoint(BaseModel):
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketDataResponse(BaseModel):
    symbol: str
    interval: str
    data: List[MarketDataPoint]
    count: int
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None


class ErrorResponse(BaseModel):
    error: str
    message: Optional[str] = None
    timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str = "1.0.0"


SymbolStr = Annotated[str, StringConstraints(pattern=r'^[A-Z0-9]{3,}$')]
LookbackHours = Annotated[int, Field(gt=0, le=43800)]


class PredictorRequest(BaseModel):
    symbol: Optional[SymbolStr] = "BTCUSDT"
    interval: Optional[str] = "5m"
    lookback_hours: Optional[LookbackHours] = 24
    analysis_type: str = "single"

    @field_validator('symbol', mode='before')
    @classmethod
    def ensure_upper_symbol(cls, v):
        if isinstance(v, str):
            return v.upper()
        return v


class ComparisonRequest(BaseModel):
    symbols: List[SymbolStr]
    interval: str = "5m"
    lookback_hours: Optional[LookbackHours] = 24

    @field_validator('symbols', mode='before')
    @classmethod
    def normalize_symbols(cls, v):
        if isinstance(v, list):
            seen = set()
            norm = []
            for s in v:
                s_up = s.upper() if isinstance(s, str) else s
                if s_up and s_up not in seen:
                    seen.add(s_up)
                    norm.append(s_up)
            return norm
        return v

    @field_validator('symbols')
    @classmethod
    def validate_symbols_list(cls, v):
        if not v or len(v) == 0:
            raise ValueError("symbols list must not be empty")
        if len(v) > config.MAX_SYMBOLS_COMPARE:
            raise ValueError(f"symbols list exceeds maximum of {config.MAX_SYMBOLS_COMPARE}")
        return v


class MarketDataRequest(BaseModel):
    symbol: SymbolStr = "BTCUSDT"
    interval: str = "5m"
    lookback_hours: LookbackHours = 24

    @field_validator('symbol', mode='before')
    @classmethod
    def ensure_upper_symbol(cls, v):
        if isinstance(v, str):
            return v.upper()
        return v
