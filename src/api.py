from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict, Optional
from pydantic import BaseModel
import warnings
import requests

from .short_term_pred import BinanceShortTermPredictor

warnings.filterwarnings('ignore')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    lookback_hours: int = 24

class PredictionResponse(BaseModel):
    symbol: str
    interval: str
    prediction: str
    confidence: float
    current_price: float
    support: float
    resistance: float
    timestamp: datetime

predictor = BinanceShortTermPredictor()

@app.get("/health")
async def health_check():
    try:
        url = 'https://api.binance.com/api/v3/ping'
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return {"status": "healthy", "binance_api": "connected"}
        return {"status": "warning", "binance_api": f"unexpected status {response.status_code}"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_next_interval(request: PredictionRequest):
    try:
        predictor.symbol = request.symbol.upper()
        predictor.interval = request.interval.lower()
        predictor.lookback_hours = request.lookback_hours
        predictor.data = None
        
        predictor.fetch_historical_data()
        prediction, confidence, support, resistance = predictor.predict_next_interval(show_output=False)
        
        current_price = predictor.data.iloc[-1]['close']
        
        return {
            "symbol": predictor.symbol,
            "interval": predictor.interval,
            "prediction": "UP" if prediction == 1 else "DOWN",
            "confidence": float(confidence * 100),
            "current_price": float(current_price),
            "support": float(support),
            "resistance": float(resistance),
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/analysis/summary")
async def get_multi_timeframe_analysis(
    symbol: str = Query(..., description="Trading pair (e.g., BTCUSDT)"),
    lookback_hours: int = Query(24, description="Hours of historical data to use"),
    intervals: List[str] = Query(["5m", "15m", "30m", "1h", "6h", "12h", "1d"], 
                               description="List of intervals to analyze")
):
    results = {}
    original_settings = {
        'symbol': predictor.symbol,
        'interval': predictor.interval,
        'lookback_hours': predictor.lookback_hours,
        'data': predictor.data
    }
    
    try:
        predictor.symbol = symbol.upper()
        predictor.lookback_hours = lookback_hours
        
        for interval in intervals:
            try:
                predictor.interval = interval.lower()
                predictor.data = None
                
                predictor.fetch_historical_data()
                prediction, confidence, support, resistance = predictor.predict_next_interval(show_output=False)
                
                results[interval] = {
                    "prediction": "UP" if prediction == 1 else "DOWN",
                    "confidence": float(confidence * 100),
                    "support": float(support),
                    "resistance": float(resistance),
                    "current_price": float(predictor.data.iloc[-1]['close'])
                }
            except Exception as e:
                results[interval] = {"error": str(e)}
                
        return {
            "symbol": predictor.symbol,
            "analysis": results,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        for key, value in original_settings.items():
            setattr(predictor, key, value)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
