# Crypto Predictor API

A high-performance REST API for cryptocurrency trading analysis powered by machine learning. Built with FastAPI, this service delivers real-time predictions, technical analysis, and market insights for various trading pairs.

## ✨ Features

- 🔮 **Price Predictions**: ML-based predictions for short-term price movements
- 📊 **Multi-Timeframe Analysis**: Analyze opportunities across different time intervals
- 🔄 **Trading Pair Comparison**: Compare multiple trading pairs simultaneously
- 📈 **Technical Indicators**: RSI, MACD, SMA, Bollinger Bands, and more
- 🎯 **Support/Resistance Levels**: Automatic calculation of key price levels
- ⚡ **Optimized Performance**: Efficient data handling with Binance Data API
- 🛡️ **Robust Error Handling**: Comprehensive rate limiting and retry mechanisms
- 📝 **Detailed Logging**: Comprehensive logging for monitoring and debugging

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Nachholiva/predictor-api.git
   cd predictor-api
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the API:**
   ```bash
   python src/main.py
   ```

6. **Access the API:**
   - Interactive API Docs: http://localhost:8002/docs
   - Alternative Documentation: http://localhost:8002/redoc
   - Health Check: http://localhost:8002/health


### Supported Intervals
```http
GET /supported-intervals
```
Get list of supported time intervals.

- `5m` - 5 minutes
- `15m` - 15 minutes
- `30m` - 30 minutes
- `1h` - 1 hour
- `6h` - 6 hours
- `12h` - 12 hours
- `1d` - 1 day
- `1M` - 1 month

## Technical Indicators

The prediction model uses the following technical indicators:

- **SMA (Simple Moving Average)**: 7 and 25 period averages
- **RSI (Relative Strength Index)**: Momentum oscillator
- **MACD (Moving Average Convergence Divergence)**: Trend-following indicator
- **ROC (Rate of Change)**: Price momentum indicator
- **Bollinger Bands**: Volatility bands with 2 standard deviations

## Machine Learning Model

- **Algorithm**: Random Forest Classifier
- **Features**: 12 technical indicators
- **Target**: Binary classification (BUY/SELL)
- **Training**: Automatic retraining on each prediction request

## Development

### Project Structure

```
crypto-predictor/
├── main.py              # FastAPI application
├── predictor_core.py    # Core prediction logic
├── models.py            # Pydantic data models
├── config.py            # Configuration settings
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

### Running in Development Mode

```bash
# Install dependencies
pip install -r requirements.txt

# Run with auto-reload
python main.py

# Or with uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8002
```

### Testing the API

Use the interactive documentation at http://localhost:8002/docs to test all endpoints directly in your browser.


## Disclaimer

This API is for educational and research purposes only. Cryptocurrency trading involves significant risk, and you should not use this tool as the sole basis for making trading decisions. Always do your own research and consider consulting with financial professionals.

---

**API Version**: 1.0.0
**Last Updated**: September 2025
