from fastapi import FastAPI, HTTPException, Depends, Request, Query
from fastapi.responses import JSONResponse
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import time
from datetime import datetime
import socket
import logging
from collections import deque

from predictor_core import create_predictor, BinanceShortTermPredictor
from models import (
    PredictorRequest, PredictionResult,
    ComparisonRequest, ComparisonResult, MarketDataResponse,
    MarketDataRequest, HealthResponse, ErrorResponse, PredictorConfig, MarketDataPoint
)
from config import config


predictor_cache = {}
validator = BinanceShortTermPredictor(
    symbol=config.DEFAULT_SYMBOL,
    interval=config.DEFAULT_INTERVAL,
    lookback_hours=config.DEFAULT_LOOKBACK_HOURS,
)

_rate_buckets = {}

async def rate_limiter(request: Request):
    try:
        limit = int(config.MAX_REQUESTS_PER_MINUTE)
    except Exception:
        limit = 60
    if limit <= 0:
        return 

    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window_start = now - 60.0

    bucket = _rate_buckets.get(client_ip)
    if bucket is None:
        bucket = deque()
        _rate_buckets[client_ip] = bucket

    while bucket and bucket[0] < window_start:
        bucket.popleft()

    if len(bucket) >= limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please slow down and retry later.")

    bucket.append(now)


def get_cached_predictor(symbol: str, interval: str, lookback_hours: int) -> BinanceShortTermPredictor:
    cache_key = f"{symbol}_{interval}_{lookback_hours}"

    if cache_key not in predictor_cache:
        predictor_cache[cache_key] = create_predictor(symbol, interval, lookback_hours)

    return predictor_cache[cache_key]


def validate_symbol(symbol: str):
    if not symbol or len(symbol) < 3:
        raise HTTPException(status_code=400, detail="Trading pair symbol must be at least 3 characters long")
    if not symbol.isupper():
        raise HTTPException(status_code=400, detail="Trading pair symbol must be uppercase")
    try:
        if not validator.is_valid_symbol(symbol):
            raise HTTPException(status_code=400, detail=f"Symbol '{symbol}' is not a TRADING pair on Binance")
    except Exception as e:
        logging.debug(f"Exchange info validation failed: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid or unverifiable symbol '{symbol}'. Please use a valid Binance TRADING pair like BTCUSDT.")


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


def validate_interval(interval: str, symbol: str = None):
    if interval not in config.SUPPORTED_INTERVALS:
        raise HTTPException(status_code=400, detail=f"Unsupported interval '{interval}'. Supported intervals: {config.SUPPORTED_INTERVALS}")
    try:
        if symbol:
            validator.validate_symbol_and_interval(symbol, interval)
        else:
            validator.validate_symbol_and_interval(config.DEFAULT_SYMBOL, interval)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logging.debug(f"Interval validation via exchange info failed: {e}")


def validate_lookback(interval: str, lookback_hours: int, multi_timeframe: bool = False):
    if lookback_hours is None:
        raise HTTPException(status_code=400, detail="lookback_hours is required")
    try:
        lookback_hours = int(lookback_hours)
    except Exception:
        raise HTTPException(status_code=400, detail="lookback_hours must be an integer")
    if lookback_hours <= 0:
        raise HTTPException(status_code=400, detail="lookback_hours must be greater than 0")

    if lookback_hours is None:
        raise HTTPException(status_code=400, detail="lookback_hours is required")
    try:
        lookback_hours = int(lookback_hours)
    except Exception:
        raise HTTPException(status_code=400, detail="lookback_hours must be an integer")
    if lookback_hours <= 0:
        raise HTTPException(status_code=400, detail="lookback_hours must be greater than 0")

    if multi_timeframe:
        max_caps = [config.LOOKBACK_MAX_HOURS.get(iv, 43800) for iv in config.SUPPORTED_INTERVALS]
        max_of_caps = max(max_caps) if max_caps else 43800
        if lookback_hours > max_of_caps:
            raise HTTPException(status_code=400, detail=f"lookback_hours too large; maximum is {max_of_caps} hours for multi-timeframe")
        return

    min_required = config.LOOKBACK_MIN_HOURS.get(interval, 1)
    max_allowed = config.LOOKBACK_MAX_HOURS.get(interval, 43800)
    if lookback_hours < min_required:
        raise HTTPException(status_code=400, detail=f"lookback_hours too small for {interval}; minimum is {min_required} hours")
    if lookback_hours > max_allowed:
        raise HTTPException(status_code=400, detail=f"lookback_hours too large for {interval}; maximum is {max_allowed} hours")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
        format=config.LOG_FORMAT,
        handlers=[logging.StreamHandler()]
    )
    
    ip = get_local_ip()
    print(f"🚀 Starting Crypto Predictor API v1.0.0")
    print(f"📊 Default symbol: {config.DEFAULT_SYMBOL}")
    print(f"⏰ Default interval: {config.DEFAULT_INTERVAL}")
    print(f"📈 Default lookback: {config.DEFAULT_LOOKBACK_HOURS} hours")
    print(f"🌐 Network access: http://{ip}:{config.API_PORT}")

    yield

    print("🛑 Shutting down Crypto Predictor API")
    predictor_cache.clear()


app = FastAPI(
    title="Crypto Predictor API",
    description="REST API for cryptocurrency trading analysis using machine learning",
    version="1.0.0",
    lifespan=lifespan,
    dependencies=[Depends(rate_limiter)],
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


@app.get("/config", response_model=PredictorConfig)
async def get_config():
    return PredictorConfig(
        symbol=config.DEFAULT_SYMBOL,
        interval=config.DEFAULT_INTERVAL,
        lookback_hours=config.DEFAULT_LOOKBACK_HOURS
    )


@app.post("/predict", response_model=PredictionResult)
async def predict(request: PredictorRequest):
    try:
        # Input validation
        if not request.symbol:
            raise ValueError("Symbol cannot be empty")
        validate_symbol(request.symbol)
        validate_interval(request.interval, request.symbol)
        validate_lookback(request.interval, request.lookback_hours)

        # Get predictor and make prediction
        predictor = get_cached_predictor(
            request.symbol,
            request.interval,
            request.lookback_hours
        )
        result = predictor.predict_next_interval()
        
        # Validate prediction result
        if not result or not isinstance(result, dict):
            raise ValueError("Invalid prediction result format")
            
        return PredictionResult(**result)
        
    except HTTPException as e:
        raise e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/analysis/summary")
async def get_multi_timeframe_analysis(
    symbol: str = Query(..., description="Trading pair symbol (e.g., BTCUSDT). Must be a valid symbol listed on Binance."),
    lookback_hours: int = Query(24, ge=1, le=720, description="Hours of historical data to analyze (1-720, default: 24)"),
    intervals: List[str] = Query(
        ["5m", "15m", "30m", "1h", "6h", "12h", "1d"],
        description="List of time intervals to analyze. Supported intervals: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M"
    )
):
    try:
        if not symbol:
            raise ValueError("Symbol cannot be empty")
            
        validate_symbol(symbol)
        
        MAX_LOOKBACK_HOURS = 24 * 30
        if not (0 < lookback_hours <= MAX_LOOKBACK_HOURS):
            raise ValueError(f"Lookback hours must be between 1 and {MAX_LOOKBACK_HOURS}")
        
        for interval in intervals:
            try:
                validate_interval(interval, symbol)
            except ValueError as e:
                raise ValueError(f"Invalid interval '{interval}': {str(e)}")
        
        results = {}
        
        for interval in intervals:
            try:
                predictor = get_cached_predictor(
                    symbol,
                    interval,
                    lookback_hours
                )
                
                result = predictor.predict_next_interval()
                
                if result and isinstance(result, dict):
                    results[interval] = {
                        "prediction": result.get("prediction", "UNKNOWN"),
                        "confidence": float(result.get("confidence", 0)),
                        "support": float(result.get("support", 0)),
                        "resistance": float(result.get("resistance", 0))
                    }
                else:
                    results[interval] = {"error": "Invalid prediction result format"}
                    
            except Exception as e:
                logging.error(f"Error analyzing {symbol} {interval}: {str(e)}")
                results[interval] = {"error": str(e)}
        
        response = {
            "symbol": symbol.upper(),
            "analysis": results,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "metadata": {
                "lookback_hours": lookback_hours,
                "intervals_analyzed": len([v for v in results.values() if "error" not in v]),
                "intervals_failed": len([v for v in results.values() if "error" in v])
            }
        }
        
        valid_results = [v for v in results.values() if "error" not in v]
        if len(valid_results) > 1:
            response["overall_bias"] = _calculate_overall_bias(results)
            response["average_confidence"] = _calculate_average_confidence(results)
            
        return response
        
    except HTTPException as e:
        raise e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Error in analysis/summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/compare", response_model=ComparisonResult)
async def compare_trading_pairs(request: ComparisonRequest):
    MAX_SYMBOLS = 10
    
    try:
        if not request.symbols or not isinstance(request.symbols, list):
            raise ValueError("Symbols must be a non-empty list")
            
        if len(request.symbols) > MAX_SYMBOLS:
            raise ValueError(f"Cannot compare more than {MAX_SYMBOLS} symbols at once")
            
        if not all(isinstance(s, str) for s in request.symbols):
            raise ValueError("All symbols must be strings")
            
        unique_symbols = []
        seen = set()
        for symbol in request.symbols:
            if symbol not in seen:
                seen.add(symbol)
                unique_symbols.append(symbol)
        
        for symbol in unique_symbols:
            validate_symbol(symbol)
            
        validate_interval(request.interval)
        validate_lookback(request.interval, request.lookback_hours)

        pairs_analysis = []
        analysis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for symbol in request.symbols:
            try:
                predictor = get_cached_predictor(
                    symbol,
                    request.interval,
                    request.lookback_hours
                )

                result = predictor.predict_next_interval()

                pair_result = {
                    "symbol": symbol,
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                    "support": result["support"],
                    "resistance": result["resistance"]
                }

                pairs_analysis.append(pair_result)

            except Exception as e:
                print(f"Warning: Failed to analyze {symbol}: {e}")
                continue

        if not pairs_analysis:
            raise HTTPException(
                status_code=500,
                detail="Failed to analyze any of the requested trading pairs"
            )

        return ComparisonResult(
            pairs=pairs_analysis,
            interval=request.interval,
            analysis_time=analysis_time
        )

    except HTTPException as e:
        raise e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/market-data", response_model=MarketDataResponse)
async def get_market_data(request: MarketDataRequest):
    try:
        validate_symbol(request.symbol)
        validate_interval(request.interval)
        validate_lookback(request.interval, request.lookback_hours)

        predictor = get_cached_predictor(
            request.symbol,
            request.interval,
            request.lookback_hours
        )

        df = predictor.fetch_historical_data()

        data_points = []
        for _, row in df.iterrows():
            data_points.append(MarketDataPoint(
                date=row['date'],
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume'])
            ))

        return MarketDataResponse(
            symbol=request.symbol,
            interval=request.interval,
            data=data_points,
            count=len(data_points),
            period_start=df['date'].min(),
            period_end=df['date'].max()
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/supported-intervals")
async def get_supported_intervals():
    return {
        "intervals": config.SUPPORTED_INTERVALS,
        "description": "Supported time intervals for analysis"
    }


@app.get("/predictor-info")
async def get_predictor_info():
    return {
        "model": "RandomForestClassifier",
        "features": [
            "open", "high", "low", "close", "volume",
            "sma_7", "sma_25", "rsi", "macd", "signal_line",
            "roc", "bb_width"
        ],
        "indicators": {
            "SMA": "Simple Moving Average (7 and 25 periods)",
            "RSI": "Relative Strength Index",
            "MACD": "Moving Average Convergence Divergence",
            "ROC": "Rate of Change",
            "Bollinger_Bands": "Bollinger Bands with 2 standard deviations"
        },
        "target": "Binary classification: price increase (BUY) or decrease (SELL) in next interval"
    }


@app.get("/")
async def root():
    return {
        "name": "Crypto Predictor API",
        "version": "1.0.0",
        "description": "REST API for cryptocurrency trading analysis using machine learning",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "multi-timeframe": "POST /predict/multi-timeframe",
            "compare": "POST /compare",
            "market-data": "POST /market-data",
            "supported-intervals": "GET /supported-intervals",
            "predictor-info": "GET /predictor-info"
        },
        "documentation": "/docs"
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logging.warning(f"HTTPException: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": f"HTTP {exc.status_code}",
            "message": exc.detail,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logging.exception("Unhandled exception in API", exc_info=exc)
    message = str(exc) if config.API_DEBUG else "An unexpected error occurred"
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": message,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.API_DEBUG,
        log_level=config.LOG_LEVEL.lower()
    )

@app.get("/ticker/price", response_model=dict)
async def get_ticker_price(symbol: str):
    validate_symbol(symbol)
    resp = validator._request("/ticker/price", params={"symbol": symbol})
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Failed to fetch ticker price")
    return resp.json()

@app.get("/exchangeInfo", response_model=dict)
async def get_exchange_info_endpoint():
    info = validator.get_exchange_info()
    return info

@app.get("/klines", response_model=list)
async def get_klines(symbol: str, interval: str = config.DEFAULT_INTERVAL, limit: int = 1000):
    validate_symbol(symbol)
    validate_interval(interval, symbol)
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - limit * 60 * 1000
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": now_ms,
        "limit": limit,
    }
    resp = validator._request("/klines", params=params)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Failed to fetch klines")
    return resp.json()
