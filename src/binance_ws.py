import json
import logging
import threading
import time
from typing import Callable, Optional

from websocket import WebSocketApp

from config import config


class BinanceKlineStream:

    def __init__(self, symbol: str, interval: str, on_message: Callable[[dict], None], reconnect: bool = True):
        self.symbol = symbol.lower()
        self.interval = interval
        self.on_message = on_message
        self.reconnect = reconnect
        self.ws: Optional[WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def _url(self) -> str:
        return f"wss://stream.binance.com:9443/ws/{self.symbol}@kline_{self.interval}"

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        backoff = 1.0
        while not self._stop.is_set():
            try:
                self.ws = WebSocketApp(
                    self._url(),
                    on_message=self._handle_message,
                    on_error=self._handle_error,
                    on_close=self._handle_close,
                )
                logging.info("Connecting to Binance WebSocket stream...")
                self.ws.run_forever(ping_interval=20, ping_timeout=10)
                if not self.reconnect:
                    break
            except Exception as e:
                logging.warning(f"WebSocket error: {e}")
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)

    def _handle_message(self, ws, message: str):
        try:
            data = json.loads(message)
            if callable(self.on_message):
                self.on_message(data)
        except Exception as e:
            logging.debug(f"Failed to parse WebSocket message: {e}")

    def _handle_error(self, ws, error):
        logging.warning(f"WebSocket error: {error}")

    def _handle_close(self, ws, status_code, msg):
        logging.info(f"WebSocket closed: {status_code} {msg}")
