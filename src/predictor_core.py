import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
import time
import logging
from typing import Optional, Dict, Any, Tuple
from config import config

warnings.filterwarnings('ignore')


class BinanceShortTermPredictor:
    def __init__(self, symbol='BTCUSDT', interval='5m', lookback_hours=24):
        self.symbol = symbol
        self.interval = interval
        self.lookback_hours = lookback_hours
        self.timeout = config.BINANCE_TIMEOUT
        self.data = None
        self.model = None
        self.scaler = MinMaxScaler()
        self.session = requests.Session()
        default_headers = {
            "User-Agent": config.USER_AGENT,
            "Accept": "application/json",
        }
        self.session.headers.update(default_headers)
        self._exchange_info_cache: Optional[Tuple[dict, float]] = None
        self._time_sync_cache: Optional[Tuple[int, float]] = None
        self._klines_cache: Dict[Tuple[str, str, int, int], Tuple[list, float]] = {}

    def _maybe_jitter(self, base: float) -> float:
        if not config.ENABLE_JITTER_BACKOFF:
            return base
        try:
            import random
            return base * (0.9 + 0.2 * random.random())
        except Exception:
            return base

    def _soft_weight_throttle(self, resp: Optional[requests.Response]):
        if not resp:
            return
        used = resp.headers.get("X-MBX-USED-WEIGHT-1M") or resp.headers.get("X-MBX-USED-WEIGHT")
        if used:
            try:
                used_val = int(used)
                if used_val >= config.SOFT_WEIGHT_LIMIT_1M:
                    sleep_s = 1.0
                    logging.info(f"Approaching weight limit (used={used_val}). Sleeping {sleep_s}s to throttle.")
                    time.sleep(sleep_s)
            except ValueError:
                pass

    def _get_server_time_ms(self) -> Optional[int]:
        if not config.ENABLE_TIME_SYNC:
            return None
        now = time.time()
        if self._time_sync_cache and (now - self._time_sync_cache[1]) < config.TIME_SYNC_TTL_SECONDS:
            return self._time_sync_cache[0]
        try:
            resp = self.session.get(self.api_base.replace("/api/v3", "") + "/api/v3/time", timeout=self.timeout)
            if resp.status_code == 200:
                server_time_ms = int(resp.json().get("serverTime"))
                self._time_sync_cache = (server_time_ms, now)
                return server_time_ms
        except Exception as e:
            logging.debug(f"Time sync failed: {e}")
        return None

    def get_exchange_info(self) -> dict:
        now = time.time()
        ttl = 600
        if self._exchange_info_cache and (now - self._exchange_info_cache[1]) < ttl:
            return self._exchange_info_cache[0]
        resp = self._request("/exchangeInfo", use_main_api=True)
        if resp.status_code != 200:
            raise Exception(f"Failed to fetch exchangeInfo: {resp.status_code} {resp.text}")
        data = resp.json()
        self._exchange_info_cache = (data, now)
        return data

    def is_valid_symbol(self, symbol: str) -> bool:
        info = self.get_exchange_info()
        for s in info.get("symbols", []):
            if s.get("symbol") == symbol and s.get("status") == "TRADING":
                return True
        return False

    def validate_symbol_and_interval(self, symbol: str, interval: str):
        valid_intervals = {
            '1m','3m','5m','15m','30m','1h','2h','4h','6h','8h','12h','1d','3d','1w','1M'
        }
        if interval not in valid_intervals:
            raise ValueError(f"Unsupported Binance interval '{interval}'. Must be one of {sorted(valid_intervals)}")
        if not self.is_valid_symbol(symbol):
            raise ValueError(f"Symbol '{symbol}' is not a TRADING pair on Binance.")

    def _request(self, path: str, params: Optional[Dict[str, Any]] = None, use_main_api: bool = False) -> requests.Response:
        base = self.api_base if use_main_api else self.data_base
        url = f"{base}{path}"
        retries = 0
        backoff = 1.0
        tried_fallback = False
        while True:
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code in (418, 429):
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after is not None:
                        try:
                            sleep_s = float(retry_after)
                        except ValueError:
                            sleep_s = backoff
                    else:
                        sleep_s = backoff
                    sleep_s = self._maybe_jitter(sleep_s)
                    logging.warning(f"Rate limited ({resp.status_code}). Retrying in {sleep_s:.1f}s...")
                    time.sleep(sleep_s)
                elif 500 <= resp.status_code < 600:
                    wait = self._maybe_jitter(backoff)
                    logging.warning(f"Server error {resp.status_code}. Retrying in {wait:.1f}s...")
                    time.sleep(wait)
                else:
                    self._soft_weight_throttle(resp)
                    return resp
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                wait = self._maybe_jitter(backoff)
                logging.warning(f"Network issue: {e}. Retrying in {wait:.1f}s...")
                time.sleep(wait)

            retries += 1
            if retries >= config.MAX_RETRIES:
                if not use_main_api and not tried_fallback:
                    logging.warning("Falling back to main Binance API for this request...")
                    base = self.api_base
                    url = f"{base}{path}"
                    retries = 0
                    backoff = 1.0
                    tried_fallback = True
                    continue
                raise Exception(f"Failed to fetch from Binance after {config.MAX_RETRIES} retries (path={path})")
            backoff *= max(1.1, config.RATE_LIMIT_BACKOFF_FACTOR)

    def fetch_historical_data(self):
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=self.lookback_hours)

        if self.interval == '1M' and self.lookback_hours < 2880:
            raise ValueError(f"Monthly interval typically requires at least 2880 hours of lookback. "
                           f"Current lookback is only {self.lookback_hours} hours.")

        elif self.interval == '1d' and self.lookback_hours < 120:
            raise ValueError(f"Daily interval typically requires at least 120 hours of lookback. "
                           f"Current lookback is only {self.lookback_hours} hours.")

        start_time_ms = int(start_time.timestamp() * 1000)
        end_time_ms = int(end_time.timestamp() * 1000)

        all_data = []

        while True:
            params = {
                "symbol": self.symbol,
                "interval": self.interval,
                "startTime": start_time_ms,
                "endTime": end_time_ms,
                "limit": 1000,
            }
            cache_key = (self.symbol, self.interval, start_time_ms, end_time_ms)
            use_cache = config.CACHE_ENABLED
            cache_hit = False
            if use_cache and cache_key in self._klines_cache:
                cached_data, cached_at = self._klines_cache[cache_key]
                if (time.time() - cached_at) < config.CACHE_TTL_SECONDS:
                    data = cached_data
                    cache_hit = True
                else:
                    self._klines_cache.pop(cache_key, None)
            if not cache_hit:
                response = self._request("/klines", params=params, use_main_api=False)
                used_weight = response.headers.get("X-MBX-USED-WEIGHT-1M") or response.headers.get("X-MBX-USED-WEIGHT")
                if used_weight:
                    logging.debug(f"Binance weight used (1m): {used_weight}")

                if response.status_code != 200:
                    raise Exception(f"Failed to fetch data ({response.status_code}): {getattr(response, 'text', '')}")

                data = response.json()
                if use_cache:
                    self._klines_cache[cache_key] = (data, time.time())
            if not isinstance(data, list):
                raise Exception(f"API response is not a list. Response: {data}")

            if not data:
                break

            all_data.extend(data)

            last_time = data[-1][0]
            start_time_ms = last_time + 1

            if start_time_ms > end_time_ms:
                break

        columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]

        df = pd.DataFrame(all_data, columns=columns)

        if df.empty:
            raise ValueError(f"No data returned from Binance API for {self.symbol} with {self.interval} interval. "
                         f"Try a different symbol, interval, or increase lookback_hours.")

        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        df['date'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

        min_required = 30
        if self.interval in ['6h', '12h']:
            min_required = 20
        elif self.interval == '1d':
            min_required = 15
        elif self.interval == '1M':
            min_required = 10

        if len(df) < min_required:
            raise ValueError(f"Insufficient data points ({len(df)}). Need at least {min_required} for {self.interval} interval. "
                         f"Try increasing lookback_hours or using a different trading pair.")

        self.data = df
        return df

    def preprocess_data(self, for_training=True):
        if self.data is None:
            self.fetch_historical_data()

        if self.data.empty:
            raise ValueError("No data available for preprocessing. Please check your parameters.")

        df = self.data.copy()

        sma_short_window = 7
        sma_long_window = 25
        rsi_window = 7
        macd_fast = 6
        macd_slow = 13
        macd_signal = 4

        if self.interval in ['6h', '12h']:
            sma_short_window = 5
            sma_long_window = 20
            rsi_window = 9
            macd_fast = 8
            macd_slow = 17
            macd_signal = 5
        elif self.interval == '1d':
            sma_short_window = 5
            sma_long_window = 20
            rsi_window = 14
            macd_fast = 12
            macd_slow = 26
            macd_signal = 9
        elif self.interval == '1M':
            sma_short_window = 3
            sma_long_window = 12
            rsi_window = 14
            macd_fast = 12
            macd_slow = 26
            macd_signal = 9

        df['sma_7'] = df['close'].rolling(window=sma_short_window).mean()
        df['sma_25'] = df['close'].rolling(window=sma_long_window).mean()

        df['rsi'] = self.calculate_rsi(df['close'], window=rsi_window)

        df['macd'], df['signal_line'] = self.calculate_macd(df['close'],
                                                          fast=macd_fast,
                                                          slow=macd_slow,
                                                          signal=macd_signal)

        df['roc'] = (df['close'] / df['close'].shift(1) - 1) * 100

        df['bb_middle'] = df['close'].rolling(window=14).mean()
        df['bb_std'] = df['close'].rolling(window=14).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

        if for_training:
            df = df.dropna()
        else:
            _feat_cols = ['open', 'high', 'low', 'close', 'volume', 'sma_7', 'sma_25',
                          'rsi', 'macd', 'signal_line', 'roc', 'bb_width']
            df[_feat_cols] = df[_feat_cols].ffill().bfill()

        features = ['open', 'high', 'low', 'close', 'volume', 'sma_7', 'sma_25',
                   'rsi', 'macd', 'signal_line', 'roc', 'bb_width']

        if df.empty:
            raise ValueError("After preprocessing and dropping NaN values, no data remains. "
                         "Try increasing lookback_hours or using a different interval.")

        if len(df[features]) == 0:
            raise ValueError("No features data available for scaling. Check your feature calculations.")

        if for_training:
            if len(df) < 5:
                raise ValueError(f"Only {len(df)} data points remain after filtering. "
                             f"Need more data for reliable analysis.")

            self.scaler.fit(df[features])
            df[features] = self.scaler.transform(df[features])
        else:
            if hasattr(self.scaler, 'n_features_in_'):
                df[features] = self.scaler.transform(df[features])
            else:
                raise ValueError("Scaler has not been fitted. Run train_model() first.")

        return df

    def calculate_rsi(self, series, window=7):
        delta = series.diff(1)
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def calculate_macd(self, series, fast=6, slow=13, signal=4):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def train_model(self):
        df = self.preprocess_data()

        features = ['open', 'high', 'low', 'close', 'volume', 'sma_7', 'sma_25',
                   'rsi', 'macd', 'signal_line', 'roc', 'bb_width']

        if df.empty:
            raise ValueError("No data available after preprocessing. Try increasing lookback period.")

        missing_features = [f for f in features if f not in df.columns]
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features in dataset: {missing_features}")

        X = df[features].values
        y = df['target'].values

        if len(X) == 0 or len(y) == 0:
            raise ValueError("Empty feature or target arrays. Check your data preprocessing.")

        min_data_points = 30
        if self.interval in ['6h', '12h']:
            min_data_points = 20
        elif self.interval == '1d':
            min_data_points = 15
        elif self.interval == '1M':
            min_data_points = 10

        if len(X) < min_data_points:
            raise ValueError(f"Not enough data points ({len(X)}) for reliable training with {self.interval} interval. "
                          f"Try increasing lookback_hours (min required: {min_data_points})")

        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            majority = int(unique_classes[0])
            logging.warning("Only one target class present after preprocessing; using trivial model.")
            self.scaler.fit(df[features])
            class _TrivialModel:
                def __init__(self, label: int):
                    self.label = int(label)
                def predict(self, X):
                    import numpy as _np
                    return _np.full((len(X),), self.label, dtype=int)
                def predict_proba(self, X):
                    import numpy as _np
                    if self.label == 1:
                        return _np.tile(_np.array([0.0, 1.0]), (len(X), 1))
                    else:
                        return _np.tile(_np.array([1.0, 0.0]), (len(X), 1))
            self.model = _TrivialModel(majority)
            return self.model, 1.0

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42
            )

        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        self.model = model
        return model, accuracy

    def predict_next_interval(self):
        if self.model is None:
            self.fetch_historical_data()
            self.train_model()

        df = self.preprocess_data(for_training=False)

        features = ['open', 'high', 'low', 'close', 'volume', 'sma_7', 'sma_25',
                   'rsi', 'macd', 'signal_line', 'roc', 'bb_width']
        latest_data = df.iloc[-1][features].values.reshape(1, -1)

        prediction = self.model.predict(latest_data)[0]
        proba = self.model.predict_proba(latest_data)[0]

        def _safe_float(x, default: float = 0.0) -> float:
            try:
                fx = float(x)
                if not np.isfinite(fx):
                    return default
                return fx
            except Exception:
                return default

        current_price = _safe_float(self.data.iloc[-1]['close'])
        support, resistance = self.calculate_support_resistance()
        support = _safe_float(support)
        resistance = _safe_float(resistance)

        volatility = _safe_float(df['close'].pct_change().std() * 100)

        if current_price == 0.0:
            support_pct = 0.0
            resistance_pct = 0.0
        else:
            support_pct = _safe_float(((support - current_price) / current_price) * 100)
            resistance_pct = _safe_float(((resistance - current_price) / current_price) * 100)

        current_time = self.data.iloc[-1]['date']

        return {
            'prediction': 'BUY' if prediction == 1 else 'SELL',
            'confidence': _safe_float(proba[int(prediction)] * 100),
            'probabilities': {
                'buy': _safe_float(proba[1] * 100),
                'sell': _safe_float(proba[0] * 100)
            },
            'current_price': _safe_float(current_price),
            'support': {
                'price': _safe_float(support),
                'percent_from_current': _safe_float(support_pct)
            },
            'resistance': {
                'price': _safe_float(resistance),
                'percent_from_current': _safe_float(resistance_pct)
            },
            'volatility_24h': _safe_float(volatility),
            'analysis_time': current_time.strftime("%Y-%m-%d %H:%M:%S"),
            'symbol': self.symbol,
            'interval': self.interval,
            'lookback_hours': self.lookback_hours
        }

    def calculate_support_resistance(self, window=None):
        df = self.data.copy()

        if window is None:
            if self.interval == '5m':
                window = 10
            elif self.interval == '15m':
                window = 10
            elif self.interval == '30m':
                window = 8
            elif self.interval == '1h':
                window = 8
            elif self.interval == '6h':
                window = 6
            elif self.interval == '12h':
                window = 5
            elif self.interval == '1d':
                window = 7
            elif self.interval == '1M':
                window = 3
            else:
                window = 10

        recent_lows = df['low'].rolling(window=window).min().dropna()
        recent_highs = df['high'].rolling(window=window).max().dropna()

        if len(recent_lows) == 0 or len(recent_highs) == 0:
            support = float(df['low'].tail(window or 10).min())
            resistance = float(df['high'].tail(window or 10).max())
        else:
            support = float(recent_lows.iloc[-1])
            resistance = float(recent_highs.iloc[-1])

        return support, resistance

    def generate_trading_summary(self):
        original_interval = self.interval
        intervals = ['5m', '15m', '30m', '1h', '6h', '12h', '1d', '1M']
        results = {}

        for interval in intervals:
            self.interval = interval
            self.data = None
            self.model = None

            try:
                self.fetch_historical_data()
                if len(self.data) < 30:
                    continue

                self.train_model()
                prediction_data = self.predict_next_interval()

                results[interval] = prediction_data

            except Exception:
                continue

        self.interval = original_interval
        self.data = None

        buy_count = sum(1 for data in results.values() if data['prediction'] == 'BUY')
        sell_count = sum(1 for data in results.values() if data['prediction'] == 'SELL')

        overall_bias = "NEUTRAL"
        if buy_count > sell_count:
            overall_bias = "BULLISH"
        elif sell_count > buy_count:
            overall_bias = "BEARISH"

        return {
            'results': results,
            'summary': {
                'total_timeframes': len(results),
                'buy_signals': buy_count,
                'sell_signals': sell_count,
                'overall_bias': overall_bias
            }
        }


def create_predictor(symbol='BTCUSDT', interval='5m', lookback_hours=24):
    return BinanceShortTermPredictor(symbol, interval, lookback_hours)
