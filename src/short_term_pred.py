import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

class BinanceShortTermPredictor:
    def __init__(self, symbol='BTCUSDT', interval='5m', lookback_hours=24):
        self.symbol = symbol
        self.interval = interval
        self.lookback_hours = lookback_hours
        self.base_url = 'https://api.binance.com/api/v3'
        self.data = None
        self.model = None
        self.scaler = MinMaxScaler()
        
    def fetch_historical_data(self):
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=self.lookback_hours)
        
        if self.interval == '1M' and self.lookback_hours < 2880:
            print(f"\n\033[33mWarning: Monthly interval typically requires at least 2880 hours of lookback.\033[0m")
            print(f"Current lookback is only {self.lookback_hours} hours. Results may be unreliable.")
        
        elif self.interval == '1d' and self.lookback_hours < 120:
            print(f"\n\033[33mWarning: Daily interval typically requires at least 120 hours of lookback.\033[0m")
            print(f"Current lookback is only {self.lookback_hours} hours. Results may be unreliable.")
        
        start_time_ms = int(start_time.timestamp() * 1000)
        end_time_ms = int(end_time.timestamp() * 1000)
        
        all_data = []
        
        while True:
            url = f"{self.base_url}/klines?symbol={self.symbol}&interval={self.interval}&startTime={start_time_ms}&endTime={end_time_ms}&limit=1000"
            response = requests.get(url)
            
            if response.status_code != 200:
                raise Exception(f"Failed to fetch data: {response.text}")
                
            data = response.json()
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
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
        
        df['date'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        
        earliest_date = df['date'].min().strftime("%Y-%m-%d %H:%M")
        latest_date = df['date'].max().strftime("%Y-%m-%d %H:%M")
        data_points = len(df)
        
        print(f"\nFetched {data_points} data points for {self.symbol} ({self.interval})")
        print(f"Period: {earliest_date} to {latest_date}")
        
        min_required = 30
        if self.interval in ['6h', '12h']:
            min_required = 20
        elif self.interval == '1d':
            min_required = 15
        elif self.interval == '1M':
            min_required = 10
            
        if data_points < min_required:
            raise ValueError(f"Insufficient data points ({data_points}). Need at least {min_required} for {self.interval} interval. "
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
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained with accuracy: {accuracy*100:.2f}%")
        print(f"Data points used: {len(X)}")
        
        self.model = model
        return model, accuracy
    
    def predict_next_interval(self, show_output=True):
        if self.model is None:
            self.train_model()
            
        df = self.preprocess_data(for_training=False)
        
        features = ['open', 'high', 'low', 'close', 'volume', 'sma_7', 'sma_25', 
                   'rsi', 'macd', 'signal_line', 'roc', 'bb_width']
        latest_data = df.iloc[-1][features].values.reshape(1, -1)
        
        prediction = self.model.predict(latest_data)[0]
        proba = self.model.predict_proba(latest_data)[0]
        
        current_price = self.data.iloc[-1]['close']
        support, resistance = self.calculate_support_resistance()
        
        volatility = df['close'].pct_change().std() * 100
        
        support_pct = ((support - current_price) / current_price) * 100
        resistance_pct = ((resistance - current_price) / current_price) * 100
        
        current_time = self.data.iloc[-1]['date']
        time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        
        if show_output:
            print("\n" + "="*60)
            print(f"Short-Term Analysis for {self.symbol} ({self.interval} interval)")
            print(f"Analysis Time: {time_str}")
            print(f"Current Price: {current_price:.6f}")
            print(f"24hr Volatility: {volatility:.2f}%")
            print(f"Support Level: {support:.6f} ({support_pct:.2f}% from current)")
            print(f"Resistance Level: {resistance:.6f} ({resistance_pct:.2f}% from current)")
            print("="*60)
        
        if show_output:
            if prediction == 1:
                print("\nPREDICTION: BUY (Price likely to rise in next interval)")
                print(f"Confidence: {proba[1]*100:.1f}%")
                print(f"\nSafe Buy Zone: Below {resistance * 0.995:.6f} ({(resistance * 0.995 - current_price) / current_price * 100:.2f}% from current)")
                print(f"Ideal Buy Point: Near {support:.6f} ({support_pct:.2f}% from current)")
            else:
                print("\nPREDICTION: SELL (Price likely to fall in next interval)")
                print(f"Confidence: {proba[0]*100:.1f}%")
                print(f"\nSafe Sell Zone: Above {support * 1.005:.6f} ({(support * 1.005 - current_price) / current_price * 100:.2f}% from current)")
                print(f"Ideal Sell Point: Near {resistance:.6f} ({resistance_pct:.2f}% from current)")
                
            print("\n" + "="*60)
            print(f"Short-Term Risk Management ({self.interval} timeframe):")
            print(f"- Consider taking profits at {resistance:.6f} if holding long positions")
            print(f"- Consider stop-loss below {support * 0.99:.6f} (1% below support) for short-term longs")
            print(f"- Consider stop-loss above {resistance * 1.01:.6f} (1% above resistance) for short-term shorts")
            print(f"- Due to {self.interval} timeframe volatility of {volatility:.2f}%, use tight stop-losses")
            print("="*60 + "\n")
        
        if show_output:
            self.plot_price_levels(support, resistance)
        
        return prediction, proba, support, resistance
    
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
        
        support = recent_lows.iloc[-1]
        resistance = recent_highs.iloc[-1]
        
        return support, resistance
    
    def plot_price_levels(self, support, resistance):
        plt.figure(figsize=(12, 6))
        
        plt.plot(self.data['date'], self.data['close'], label='Close Price', color='blue', linewidth=1.5)
        
        if len(self.data) > 100:
            plt.fill_between(
                self.data['date'][-100:],
                self.data['low'][-100:],
                self.data['high'][-100:],
                alpha=0.2,
                color='gray',
                label='Price Range (High-Low)'
            )
        else:
            plt.fill_between(
                self.data['date'],
                self.data['low'],
                self.data['high'],
                alpha=0.2,
                color='gray',
                label='Price Range (High-Low)'
            )
        
        plt.axhline(y=support, color='green', linestyle='--', label=f'Support: {support:.6f}')
        plt.axhline(y=resistance, color='red', linestyle='--', label=f'Resistance: {resistance:.6f}')
        
        plt.title(f'{self.symbol} {self.interval} Price with Support & Resistance (24hr window)')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def generate_trading_summary(self):
        original_interval = self.interval
        intervals = ['5m', '15m', '30m', '1h', '6h', '12h', '1d', '1M']
        results = {}
        
        print("\n" + "="*70)
        print(f"MULTI-TIMEFRAME ANALYSIS FOR {self.symbol}")
        print("="*70)
        
        for interval in intervals:
            self.interval = interval
            self.data = None
            self.model = None
            
            try:
                self.fetch_historical_data()
                if len(self.data) < 30:
                    print(f"Not enough data for {interval} analysis. Skipping.")
                    continue
                    
                self.train_model()
                prediction, proba, support, resistance = self.predict_next_interval()
                
                results[interval] = {
                    'prediction': 'BUY' if prediction == 1 else 'SELL',
                    'confidence': proba[prediction] * 100,
                    'support': support,
                    'resistance': resistance
                }
                
            except Exception as e:
                print(f"Error analyzing {interval} timeframe: {e}")
        
        self.interval = original_interval
        self.data = None
        
        print("\n" + "="*70)
        print("TIMEFRAME SUMMARY:")
        print("-"*70)
        print(f"{'Timeframe':<10} | {'Signal':<6} | {'Confidence':<12} | {'Support':<12} | {'Resistance':<12}")
        print("-"*70)
        
        for interval, data in results.items():
            print(f"{interval:<10} | {data['prediction']:<6} | {data['confidence']:.1f}%{' ':>7} | {data['support']:.6f} | {data['resistance']:.6f}")
        
        print("="*70)
        print("Trading Strategy Recommendation:")
        
        buy_count = sum(1 for data in results.values() if data['prediction'] == 'BUY')
        sell_count = sum(1 for data in results.values() if data['prediction'] == 'SELL')
        
        if buy_count > sell_count:
            print("Overall Bias: BULLISH (More timeframes showing buy signals)")
        elif sell_count > buy_count:
            print("Overall Bias: BEARISH (More timeframes showing sell signals)")
        else:
            print("Overall Bias: NEUTRAL (Equal buy and sell signals across timeframes)")
            
        print("="*70 + "\n")
        
        return results


def get_valid_input(prompt, valid_options=None, validation_func=None, error_message=None):
    while True:
        user_input = input(prompt).strip()
        
        if not user_input:
            print("⚠️  Input cannot be empty. Please try again.")
            continue
            
        if valid_options is not None:
            if user_input.lower() in [opt.lower() for opt in valid_options]:
                return user_input
                
        if validation_func is not None:
            if validation_func(user_input):
                return user_input
        
        if error_message:
            print(f"⚠️  {error_message}")
        else:
            if valid_options:
                print(f"⚠️  Invalid input. Valid options: {', '.join(valid_options)}")
            else:
                print("⚠️  Invalid input. Please try again.")

def save_chart_to_file(predictor, chart_filename=None):
    import matplotlib.pyplot as plt
    if not chart_filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_filename = f"{predictor.symbol}_{predictor.interval}_{timestamp}.png"
    
    support, resistance = predictor.calculate_support_resistance()
    
    plt.figure(figsize=(12, 6))
    plt.plot(predictor.data['date'], predictor.data['close'], label='Close Price', color='blue', linewidth=1.5)
    
    if len(predictor.data) > 100:
        plt.fill_between(
            predictor.data['date'][-100:],
            predictor.data['low'][-100:],
            predictor.data['high'][-100:],
            alpha=0.2,
            color='gray',
            label='Price Range (High-Low)'
        )
    else:
        plt.fill_between(
            predictor.data['date'],
            predictor.data['low'],
            predictor.data['high'],
            alpha=0.2,
            color='gray',
            label='Price Range (High-Low)'
        )
    
    plt.axhline(y=support, color='green', linestyle='--', label=f'Support: {support:.6f}')
    plt.axhline(y=resistance, color='red', linestyle='--', label=f'Resistance: {resistance:.6f}')
    
    plt.title(f'{predictor.symbol} {predictor.interval} Price with Support & Resistance (24hr window)')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(chart_filename)
    plt.close()
    
    print(f"📊 Chart saved as '{chart_filename}'")
    return chart_filename


def run_single_timeframe_analysis(predictor):
    clear_screen()
    display_banner()
    print(f"\n🔍 Running analysis for {predictor.symbol} on {predictor.interval} timeframe...")
    print(f"Looking back {predictor.lookback_hours} hours")
    
    try:
        display_loading_animation("Fetching data from Binance API")
        predictor.fetch_historical_data()  
        
        display_loading_animation("Training prediction model")
        prediction, proba, support, resistance = predictor.predict_next_interval()
        
        print("\n")
        save_option = get_valid_input("Would you like to save the chart to file? (y/n): ", valid_options=["y", "n", "yes", "no"])
        if save_option.lower() in ["y", "yes"]:
            save_chart_to_file(predictor)
        
        input("\nPress Enter to return to the main menu...")
    except Exception as e:
        print(f"\n❌ Error in analysis: {e}")
        print(f"\n\033[33mTroubleshooting tips:\033[0m")
        print("1. Try increasing the lookback hours for this interval")
        print("2. Verify the trading pair exists on Binance")
        print("3. For longer intervals (1d, 1M), ensure sufficient lookback time")
        print("4. Check your internet connection")
        input("\nPress Enter to return to the main menu...")

def run_multi_timeframe_analysis(predictor):
    clear_screen()
    display_banner()
    print(f"\n🔍 Running multi-timeframe analysis for {predictor.symbol}...")
    display_loading_animation("Analyzing multiple timeframes", duration=2.0)
    
    try:
        results = predictor.generate_trading_summary()
        
        print("\n")
        save_option = get_valid_input("Would you like to save charts for any timeframe? (y/n): ", valid_options=["y", "n", "yes", "no"])
        if save_option.lower() in ["y", "yes"]:
            available_timeframes = list(results.keys())
            if available_timeframes:
                timeframe_str = ", ".join(available_timeframes)
                tf_choice = get_valid_input(f"Which timeframe chart to save ({timeframe_str}, all)? ", 
                                          valid_options=available_timeframes + ["all"])
                
                original_interval = predictor.interval
                original_data = predictor.data
                
                if tf_choice.lower() == "all":
                    for tf in available_timeframes:
                        predictor.interval = tf
                        predictor.fetch_historical_data()
                        save_chart_to_file(predictor)
                else:
                    predictor.interval = tf_choice
                    predictor.fetch_historical_data()
                    save_chart_to_file(predictor)
                    
                predictor.interval = original_interval
                predictor.data = original_data
        
        input("\nPress Enter to return to the main menu...")
    except Exception as e:
        print(f"\n❌ Error in multi-timeframe analysis: {e}")
        input("\nPress Enter to return to the main menu...")

def compare_trading_pairs(predictor):
    clear_screen()
    display_banner()
    print("\n🔍 COMPARE TRADING PAIRS")
    print("=" * 50)
    
    pairs_input = get_valid_input("Enter trading pairs separated by commas (e.g., BTCUSDT,ETHUSDT,SOLUSDT): ",
                                validation_func=lambda s: all(len(p.strip()) >= 3 for p in s.split(",")))
    pairs = [p.strip().upper() for p in pairs_input.split(",")]
    
    interval = get_valid_input("Select interval for comparison (5m, 15m, 30m, 1h, 6h, 12h, 1d, 1M): ", 
                             valid_options=["5m", "15m", "30m", "1h", "6h", "12h", "1d", "1M"])
    
    print("\n📊 COMPARISON RESULTS:")
    print("-" * 70)
    print(f"{'Trading Pair':<12} | {'Signal':<6} | {'Confidence':<12} | {'Support':<12} | {'Resistance':<12}")
    print("-" * 70)
    
    saved_charts = []
    original_symbol = predictor.symbol
    original_interval = predictor.interval
    
    try:
        for pair in pairs:
            display_loading_animation(f"Analyzing {pair}", duration=1.0)
            predictor.symbol = pair
            predictor.interval = interval
            predictor.data = None
            
            try:
                predictor.fetch_historical_data()
                predictor.train_model()
                prediction, proba, support, resistance = predictor.predict_next_interval(show_output=False)
                
                signal = "BUY" if prediction == 1 else "SELL"
                confidence = proba[prediction] * 100
                
                print(f"{pair:<12} | {signal:<6} | {confidence:.1f}%{' ':>7} | {support:.6f} | {resistance:.6f}")
                
                chart_file = save_chart_to_file(predictor, f"{pair}_{interval}_comparison.png")
                saved_charts.append(chart_file)
                
            except Exception as e:
                print(f"{pair:<12} | ERROR: {str(e)}")
    except Exception as e:
        print(f"\n❌ Error in comparison: {e}")
    
    predictor.symbol = original_symbol
    predictor.interval = original_interval
    predictor.data = None
    
    if saved_charts:
        print(f"\n✅ Saved {len(saved_charts)} comparison charts")
    
    input("\nPress Enter to return to the main menu...")

def check_binance_connection():
    try:
        url = 'https://api.binance.com/api/v3/ping'
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return True, "Connected to Binance API successfully"
        elif response.status_code == 429:
            return False, "Rate limit exceeded. Please wait before trying again."
        elif response.status_code >= 500:
            return False, "Binance API is experiencing issues. Please try again later."
        else:
            return False, f"Binance API returned status code: {response.status_code}"
    except requests.exceptions.Timeout:
        return False, "Connection to Binance API timed out. Please check your internet connection."
    except requests.exceptions.RequestException as e:
        return False, f"Failed to connect to Binance API: {str(e)}"

def prompt_for_lookback_hours(interval):
    if interval in ['5m', '15m', '30m', '1h']:
        default_hours = 24
    elif interval in ['6h', '12h']:
        default_hours = 72
    elif interval == '1d':
        default_hours = 120
    elif interval == '1M':
        default_hours = 2880
    else:
        default_hours = 24
        
    if interval == '1M':
        min_hours = 720
        max_hours = 8760
        recommendation = "monthly interval (minimum 30 days)"
    elif interval == '1d':
        min_hours = 72
        max_hours = 4380
        recommendation = "daily interval (minimum 3 days)"
    elif interval in ['6h', '12h']:
        min_hours = 48
        max_hours = 1440
        recommendation = "longer hourly intervals (minimum 2 days)"
    else:
        min_hours = 6
        max_hours = 720
        recommendation = "short intervals (minimum 6 hours)"
    
    print(f"\n\033[1mLookback Period for {interval} Interval\033[0m")
    print(f"Default: {default_hours} hours")
    print(f"Recommended for {recommendation}: {min_hours}-{max_hours} hours")
    
    custom_lookback = get_valid_input(f"Would you like to customize the lookback period? (y/n): ",
                                    valid_options=["y", "n", "yes", "no"])
    
    if custom_lookback.lower() in ["y", "yes"]:
        hours = int(get_valid_input(
            f"Enter lookback hours (minimum {min_hours} recommended): ",
            validation_func=lambda h: h.isdigit() and int(h) >= min_hours,
            error_message=f"Please enter a valid number >= {min_hours}"))
        return hours
    else:
        return default_hours

def main():
    clear_screen()
    display_banner()
    
    connected, message = check_binance_connection()
    if not connected:
        print(f"\n❌ {message}")
        print("\nThe application requires connection to Binance API.")
        input("Press Enter to exit...")
        return
    
    symbol = get_valid_input("Enter trading pair (e.g., BTCUSDT): ", 
                           validation_func=lambda s: len(s) >= 3,
                           error_message="Trading pair should be at least 3 characters").upper()
    interval = get_valid_input("Enter interval (5m, 15m, 30m, 1h, 6h, 12h, 1d, 1M): ", 
                             valid_options=["5m", "15m", "30m", "1h", "6h", "12h", "1d", "1M"])
    
    lookback_hours = prompt_for_lookback_hours(interval)
    
    try:
        predictor = BinanceShortTermPredictor(symbol=symbol, interval=interval, lookback_hours=lookback_hours)
        
        while True:
            clear_screen()
            display_banner()
            choice = display_menu()
            
            if choice == "1":
                run_single_timeframe_analysis(predictor)
            elif choice == "2":
                run_multi_timeframe_analysis(predictor)
            elif choice == "3":
                compare_trading_pairs(predictor)
            elif choice == "4":
                predictor = settings_menu(predictor)
            elif choice == "5":
                clear_screen()
                print("\nThank you for using the Binance Short-Term Crypto Trading Analyzer!")
                print("Exiting program...\n")
                break
                
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nCritical Error: {e}")
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
