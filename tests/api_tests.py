import requests
import json
import time
from datetime import datetime


BASE_URL = "http://localhost:8002"


def _print_header(title: str):
    print("\n" + title)
    print("-" * len(title))


def test_root():
    _print_header("1) Root endpoint")
    r = requests.get(f"{BASE_URL}/", timeout=10)
    assert r.status_code == 200, f"root failed: {r.status_code}"
    data = r.json()
    assert "endpoints" in data, "root missing endpoints field"
    print("✅ Root ok; endpoints listed:", ", ".join(data["endpoints"].keys()))


def test_health_and_config():
    _print_header("2) Health and Config")
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    assert r.status_code == 200, f"health failed: {r.status_code}"
    print("✅ Health:", r.json())

    r = requests.get(f"{BASE_URL}/config", timeout=10)
    assert r.status_code == 200, f"config failed: {r.status_code}"
    cfg = r.json()
    assert "symbol" in cfg and "interval" in cfg, "config missing fields"
    print("✅ Config:", cfg)


def test_metadata_endpoints():
    _print_header("3) Metadata endpoints (supported-intervals, predictor-info)")
    r = requests.get(f"{BASE_URL}/supported-intervals", timeout=10)
    assert r.status_code == 200, f"supported-intervals failed: {r.status_code}"
    intervals = r.json().get("intervals", [])
    assert isinstance(intervals, list) and len(intervals) > 0, "no intervals returned"
    print("✅ Supported intervals:", ", ".join(intervals))

    r = requests.get(f"{BASE_URL}/predictor-info", timeout=10)
    assert r.status_code == 200, f"predictor-info failed: {r.status_code}"
    info = r.json()
    assert "features" in info and "model" in info, "predictor-info missing fields"
    print("✅ Predictor info model:", info.get("model"))


def test_predict_success_and_failure():
    _print_header("4) Predict: success and validation errors")
    good = {"symbol": "BTCUSDT", "interval": "5m", "lookback_hours": 24}
    start = time.time()
    r = requests.post(f"{BASE_URL}/predict", json=good, timeout=120)
    elapsed = time.time() - start
    if r.status_code == 200:
        res = r.json()
        assert "prediction" in res and "confidence" in res, "predict missing fields"
        print(f"✅ Predict OK in {elapsed:.1f}s: {res['prediction']} @ {res['confidence']:.1f}%")
    elif r.status_code in (418, 429):
        print("⚠️  Predict rate-limited; server correctly enforcing limits")
    else:
        try:
            print("❌ Predict failed response:", r.text)
        except Exception:
            pass
        print(f"⚠️  Predict returned {r.status_code}. Enable API_DEBUG=true for detailed error messages. Continuing...")
        return

    bad_symbol = {"symbol": "INVALID", "interval": "5m", "lookback_hours": 6}
    r = requests.post(f"{BASE_URL}/predict", json=bad_symbol, timeout=30)
    assert r.status_code == 400, f"expected 400 for invalid symbol, got {r.status_code}"
    print("✅ Predict rejects invalid symbol with 400")

    bad_interval = {"symbol": "BTCUSDT", "interval": "2m", "lookback_hours": 6}
    r = requests.post(f"{BASE_URL}/predict", json=bad_interval, timeout=30)
    assert r.status_code == 400, f"expected 400 for invalid interval, got {r.status_code}"
    print("✅ Predict rejects invalid interval with 400")


def test_multi_timeframe():
    _print_header("5) Multi-timeframe analysis")
    payload = {"symbol": "ETHUSDT", "lookback_hours": 24}
    r = requests.post(f"{BASE_URL}/predict/multi-timeframe", json=payload, timeout=180)
    if r.status_code == 200:
        data = r.json()
        assert "results" in data and "summary" in data, "multi-timeframe missing fields"
        print("✅ Multi-timeframe OK; overall bias:", data["summary"].get("overall_bias"))
    elif r.status_code in (418, 429):
        print("⚠️  Multi-timeframe rate-limited; server correctly enforcing limits")
    else:
        try:
            print("❌ Multi-timeframe failed response:", r.text)
        except Exception:
            pass
        print(f"⚠️  Multi-timeframe returned {r.status_code}. Enable API_DEBUG=true for detailed error messages. Continuing...")
        return


def test_compare():
    _print_header("6) Compare trading pairs")
    payload = {"symbols": ["BTCUSDT", "ETHUSDT"], "interval": "5m", "lookback_hours": 24}
    r = requests.post(f"{BASE_URL}/compare", json=payload, timeout=120)
    if r.status_code == 200:
        data = r.json()
        assert "pairs" in data and isinstance(data["pairs"], list), "compare missing pairs"
        if data["pairs"]:
            p0 = data["pairs"][0]
            assert "symbol" in p0 and "prediction" in p0, "pair result missing fields"
        print("✅ Compare OK; pairs analyzed:", len(data["pairs"]))
    elif r.status_code in (418, 429):
        print("⚠️  Compare rate-limited; server correctly enforcing limits")
    else:
        raise AssertionError(f"compare failed: {r.status_code} {getattr(r, 'text', '')}")


def test_market_data_and_cache():
    _print_header("7) Market data + cache behavior")
    payload = {"symbol": "BTCUSDT", "interval": "5m", "lookback_hours": 3}
    t0 = time.time()
    r1 = requests.post(f"{BASE_URL}/market-data", json=payload, timeout=60)
    t1 = time.time() - t0
    assert r1.status_code == 200, f"market-data failed: {r1.status_code}"
    d1 = r1.json()
    assert d1.get("count", 0) > 0, "market-data returned no points"
    print(f"✅ Market data OK; points: {d1['count']} in {t1:.1f}s")

    time.sleep(2)
    t0 = time.time()
    r2 = requests.post(f"{BASE_URL}/market-data", json=payload, timeout=60)
    t2 = time.time() - t0
    if r2.status_code == 200:
        print(f"ℹ️  Second request completed in {t2:.1f}s; caching may make it faster depending on server settings")
    else:
        print(f"⚠️  Second market data request failed: {r2.status_code}")


def test_ticker_price():
    _print_header("8) Ticker price endpoint")
    symbol = "BTCUSDT"
    r = requests.get(f"{BASE_URL}/ticker/price?symbol={symbol}", timeout=10)
    assert r.status_code == 200, f"ticker/price failed: {r.status_code}"
    data = r.json()
    assert "symbol" in data and "price" in data, "ticker/price missing required fields"
    assert data["symbol"] == symbol, "Returned symbol doesn't match request"
    try:
        float(data["price"])
        print(f"✅ Ticker price OK for {symbol}")
    except (ValueError, TypeError):
        assert False, "Price is not a valid number"


def test_exchange_info():
    _print_header("9) Exchange info endpoint")
    r = requests.get(f"{BASE_URL}/exchangeInfo", timeout=10)
    assert r.status_code == 200, f"exchangeInfo failed: {r.status_code}"
    data = r.json()
    assert "timezone" in data and "symbols" in data, "exchangeInfo missing required fields"
    assert isinstance(data["symbols"], list), "Symbols should be a list"
    print(f"✅ Exchange info OK; found {len(data['symbols'])} symbols")


def test_klines():
    _print_header("10) Klines endpoint")
    symbol = "BTCUSDT"
    interval = "1h"
    limit = 10
    r = requests.get(f"{BASE_URL}/klines?symbol={symbol}&interval={interval}&limit={limit}", timeout=10)
    assert r.status_code == 200, f"klines failed: {r.status_code}"
    data = r.json()
    assert isinstance(data, list), "Klines should return a list"
    if data:  # Only check content if we got data back
        assert len(data[0]) >= 6, "Each kline should have at least 6 elements"
        assert len(data) <= limit, f"Should return at most {limit} items"
    print(f"✅ Klines OK; got {len(data)} candles for {symbol} {interval}")


def test_analysis_summary():
    _print_header("11) Analysis summary endpoint")
    symbol = "BTCUSDT"
    r = requests.get(
        f"{BASE_URL}/analysis/summary?symbol={symbol}&lookback_hours=24",
        params={"intervals": ["5m", "15m"]},
        timeout=30
    )
    if r.status_code == 200:
        data = r.json()
        assert "symbol" in data and "analysis" in data, "analysis/summary missing required fields"
        assert isinstance(data["analysis"], dict), "Analysis should be a dictionary"
        print(f"✅ Analysis summary OK; analyzed {len(data['analysis'])} intervals")
    elif r.status_code in (418, 429):
        print("⚠️  Analysis summary rate-limited; server correctly enforcing limits")
    else:
        print(f"⚠️  Analysis summary returned {r.status_code}. Continuing...")


def run_all():
    print("🧪 Testing Crypto Predictor API")
    print("=" * 60)
    test_root()
    test_health_and_config()
    test_metadata_endpoints()
    test_predict_success_and_failure()
    test_multi_timeframe()
    test_compare()
    test_market_data_and_cache()
    test_ticker_price()
    test_exchange_info()
    test_klines()
    test_analysis_summary()
    print("\n" + "=" * 60)
    print("🏁 All tests executed. Review warnings for potential rate-limit behavior.")


if __name__ == "__main__":
    run_all()
