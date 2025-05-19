import os
import json
import time
import logging
import datetime
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from google import genai
from web3 import Web3
import ta  # Technical Analysis library
import pickle

# Configure logging for more verbose output to troubleshoot webhook issues
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_mcp.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CryptoMCP")

print("Starting Enhanced CryptoMCP System with CryptoQuant Integration...")

# Default webhook URLs - directly taken from your config
DEFAULT_WEBHOOK_URLS = {
    'BTC': 'http://localhost:5000/webhook/btc',
    'SOL': 'http://localhost:5000/webhook/sol'
}
DEFAULT_WEBHOOK_URL = 'http://localhost:5000/webhook/default'

# Default ticker map
DEFAULT_TICKER_MAP = {
    'BTC': 'BTCUSDT',
    'SOL': 'SOLUSDT'
}

# Updated Configuration
@dataclass
class Config:
    gemini_api_key: str
    webhook_urls: Dict[str, str]  # Mapping of ticker -> webhook URL
    cryptocurrencies: List[str]
    cryptopanic_api_key: str  # CryptoPanic API key for news
    cryptoquant_api_key: str  # Added CryptoQuant API key
    cryptoquant_supported: List[str] = field(default_factory=lambda: ["BTC", "ETH"])  # List of cryptos supported by CryptoQuant
    web3_providers: Dict[str, str] = field(default_factory=dict)  # Network -> provider URL
    data_fetch_interval: int = 3600  # Default 1 hour
    model_name: str = "gemini-pro"  # Gemini model to use
    webhook_enabled: bool = True  # Default to enabled for testing
    default_webhook_url: Optional[str] = None  # Default URL if ticker not found
    track_whale_wallets: bool = True  # Track large wallet movements
    technical_indicators: List[str] = field(default_factory=lambda: ["rsi", "macd", "bollinger"])
    lookback_days: int = 30  # Days of historical data to analyze
    backtest_enabled: bool = True  # Enable backtesting
    whale_threshold: float = 1000000  # $1M USD for whale transactions
    performance_evaluation_interval: int = 86400  # 24 hours
    check_tp_sl_interval: int = 10  # Check take profit/stop loss every 10 seconds for faster response
    position_check_interval: int = 10  # Check positions every 10 seconds for immediate detection
    
    @classmethod
    def from_file(cls, filename: str) -> 'Config':
        print(f"Loading config from {filename}...")
        try:
            with open(filename, 'r') as f:
                config_data = json.load(f)
            
            # Handle webhook_urls in different formats
            webhook_urls = {}
            
            # Check if we have the new webhook_urls format
            if 'webhook_urls' in config_data:
                webhook_urls = config_data['webhook_urls']
            # Check if we have the old webhook_url format (use as default)
            elif 'webhook_url' in config_data:
                webhook_urls = {}
                config_data['default_webhook_url'] = config_data['webhook_url']
            
            # Add web3 providers with defaults if not present
            if 'web3_providers' not in config_data:
                config_data['web3_providers'] = {
                    "ethereum": "https://eth-mainnet.g.alchemy.com/v2/demo",
                    "bsc": "https://bsc-dataseed.binance.org/",
                    "polygon": "https://polygon-rpc.com"
                }
                
            # Add CryptoQuant supported coins if not present
            if 'cryptoquant_supported' not in config_data:
                config_data['cryptoquant_supported'] = ["BTC", "ETH"]  # Only these are fully supported

            # Create a new dict with the values we need
            cleaned_config = {
                'gemini_api_key': config_data.get('gemini_api_key', ''),
                'webhook_urls': webhook_urls,
                'cryptocurrencies': config_data.get('cryptocurrencies', []),
                'cryptopanic_api_key': config_data.get('cryptopanic_api_key', ''),
                'cryptoquant_api_key': config_data.get('cryptoquant_api_key', ''),
                'cryptoquant_supported': config_data.get('cryptoquant_supported', ["BTC", "ETH"]),
                'web3_providers': config_data.get('web3_providers', {}),
                'data_fetch_interval': config_data.get('data_fetch_interval', 3600),
                'model_name': config_data.get('model_name', 'gemini-2.0-flash'),
                'webhook_enabled': config_data.get('webhook_enabled', True),
                'default_webhook_url': config_data.get('default_webhook_url', DEFAULT_WEBHOOK_URL),
                'track_whale_wallets': config_data.get('track_whale_wallets', True),
                'technical_indicators': config_data.get('technical_indicators', ["rsi", "macd", "bollinger"]),
                'lookback_days': config_data.get('lookback_days', 30),
                'backtest_enabled': config_data.get('backtest_enabled', True),
                'whale_threshold': config_data.get('whale_threshold', 1000000),
                'performance_evaluation_interval': config_data.get('performance_evaluation_interval', 86400),
                'check_tp_sl_interval': config_data.get('check_tp_sl_interval', 10),
                'position_check_interval': config_data.get('position_check_interval', 10)
            }
            
            # Ensure cryptocurrencies is a list
            if not isinstance(cleaned_config['cryptocurrencies'], list):
                cleaned_config['cryptocurrencies'] = [cleaned_config['cryptocurrencies']]
            
            print(f"Loaded config with webhook URLs: {cleaned_config['webhook_urls']}")
            print(f"Default webhook URL: {cleaned_config['default_webhook_url']}")
            print(f"Webhook enabled: {cleaned_config['webhook_enabled']}")
            print(f"CryptoQuant supported currencies: {cleaned_config['cryptoquant_supported']}")
            print(f"CryptoQuant API key: {cleaned_config['cryptoquant_api_key'][:5]}...{cleaned_config['cryptoquant_api_key'][-4:]}")
            
            return cls(**cleaned_config)
        except Exception as e:
            print(f"Error loading config: {e}")
            logger.error(f"Error loading config: {e}")
            raise

# Enhanced data structures
@dataclass
class MarketData:
    ticker: str
    price: float
    open: float
    high: float
    low: float
    volume: float
    timestamp: datetime.datetime
    market_cap: Optional[float] = None
    price_change_24h: Optional[float] = None
    price_change_percentage_24h: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_lower: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None

@dataclass
class NewsItem:
    title: str
    summary: str
    source: str
    url: str
    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None
    timestamp: Optional[datetime.datetime] = None
    tickers_mentioned: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

@dataclass
class OnChainData:
    ticker: str
    network: str
    timestamp: datetime.datetime
    # Flag to indicate if this is real or simulated data
    is_simulated: bool = False
    # Enhanced fields with real on-chain data from CryptoQuant
    exchange_reserve: Optional[float] = None
    exchange_netflow: Optional[float] = None 
    exchange_inflow: Optional[float] = None
    exchange_outflow: Optional[float] = None
    transaction_count: Optional[int] = None
    address_count: Optional[int] = None
    in_house_flow: Optional[float] = None
    mpi: Optional[float] = None  # Miner Position Index
    exchange_shutdown_index: Optional[float] = None
    exchange_whale_ratio: Optional[float] = None
    fund_flow_ratio: Optional[float] = None
    stablecoins_ratio: Optional[float] = None
    inflow_age_distribution: Dict[str, float] = field(default_factory=dict)
    inflow_supply_distribution: Dict[str, float] = field(default_factory=dict)
    sopr: Optional[float] = None  # Spent Output Profit Ratio
    # Keep original fields
    large_transactions: List[Dict[str, Any]] = field(default_factory=list)
    active_addresses_24h: Optional[int] = None
    transaction_volume_24h: Optional[float] = None
    avg_transaction_value: Optional[float] = None
    whale_wallet_changes: Dict[str, float] = field(default_factory=dict)
    dex_volume: Optional[float] = None
    exchange_inflows: Optional[float] = None
    exchange_outflows: Optional[float] = None

@dataclass
class TradeSignal:
    ticker: str
    action: str  # buy, sell, exit_buy, exit_sell, hold
    price: float
    time: datetime.datetime
    confidence_score: float = 0.0
    size: Optional[float] = None
    sl: Optional[float] = None   # Stop loss
    tp: Optional[float] = None   # Take profit
    rationale: Optional[str] = None
    expected_holding_period: Optional[str] = None  # short, medium, long
    risk_assessment: Optional[str] = None  # low, medium, high
    source_signals: Dict[str, Any] = field(default_factory=dict)  # market, news, onchain, etc.
    per: Optional[float] = None  # Fixed value: 100 for TP hit, 0 for SL hit
    reason: Optional[str] = None  # Reason for exit
    
    def get_action_prefix(self) -> str:
        """Get the uppercase action prefix for the webhook message"""
        if self.action == "buy":
            return "BUY"
        elif self.action == "sell":
            return "SELL"
        elif self.action == "exit_buy":
            return "EXIT BUY"
        elif self.action == "exit_sell":
            return "EXIT SELL"
        else:
            return self.action.upper()

@dataclass
class Position:
    ticker: str
    action: str  # buy or sell
    entry_price: float
    entry_time: datetime.datetime
    size: float
    sl: float
    tp: float
    exit_price: Optional[float] = None
    exit_time: Optional[datetime.datetime] = None
    status: str = "open"  # open, closed_tp, closed_sl, closed_manual
    
    def is_take_profit_hit(self, current_price: float) -> bool:
        """Check if take profit has been hit"""
        if current_price is None or self.tp is None:
            return False
            
        if self.action == "buy":
            return current_price >= self.tp
        else:  # sell
            return current_price <= self.tp
    
    def is_stop_loss_hit(self, current_price: float) -> bool:
        """Check if stop loss has been hit"""
        if current_price is None or self.sl is None:
            return False
            
        if self.action == "buy":
            return current_price <= self.sl
        else:  # sell
            return current_price >= self.sl
    
    def calculate_profit_percentage(self) -> float:
        """Calculate profit percentage"""
        if not self.exit_price:
            return 0.0
        
        if self.action == "buy":
            profit_pct = (self.exit_price - self.entry_price) / self.entry_price * 100
        else:  # sell
            profit_pct = (self.entry_price - self.exit_price) / self.entry_price * 100
            
        return profit_pct
    
    def to_exit_signal(self, current_price: float, reason: str) -> 'TradeSignal':
        """Convert position to exit signal with fixed per values"""
        exit_action = f"exit_{self.action}"
        exit_time = datetime.datetime.now()
        
        # Set fixed per values based on reason
        if reason == "tp_hit":
            per_value = 100  # Always 100 for take profit
        elif reason == "sl_hit":
            per_value = 0    # Always 0 for stop loss
        else:
            # For any other reason, use actual calculation
            if self.action == "buy":
                actual_per = (current_price - self.entry_price) / self.entry_price * 100
            else:  # sell
                actual_per = (self.entry_price - current_price) / self.entry_price * 100
            # Use 0 if negative, actual percentage if positive
            per_value = max(0, actual_per)
        
        return TradeSignal(
            ticker=self.ticker,
            action=exit_action,
            price=current_price,
            time=exit_time,
            size=self.size,
            sl=self.sl,
            tp=self.tp,
            rationale=f"Position closed: {reason}",
            per=per_value,  # Fixed values: 100 for TP, 0 for SL
            reason=reason
        )

@dataclass
class PerformanceMetrics:
    ticker: str
    start_date: datetime.datetime
    end_date: datetime.datetime
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    profit_loss: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: Optional[float] = None
    win_rate: Optional[float] = None
    avg_profit_per_trade: Optional[float] = None
    avg_loss_per_trade: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    
    def calculate_metrics(self, completed_trades: List[Dict[str, Any]]):
        """Calculate performance metrics from completed trades"""
        if not completed_trades:
            return
        
        self.total_trades = len(completed_trades)
        profits = [t['profit'] for t in completed_trades if t['profit'] > 0]
        losses = [t['profit'] for t in completed_trades if t['profit'] < 0]
        
        self.winning_trades = len(profits)
        self.losing_trades = len(losses)
        
        self.profit_loss = sum(profits) + sum(losses)
        
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
        
        if profits:
            self.avg_profit_per_trade = sum(profits) / len(profits)
        
        if losses:
            self.avg_loss_per_trade = sum(losses) / len(losses)
        
        if self.avg_loss_per_trade and self.avg_profit_per_trade and self.avg_loss_per_trade != 0:
            self.risk_reward_ratio = abs(self.avg_profit_per_trade / self.avg_loss_per_trade)
        
        # Calculate drawdown
        equity_curve = []
        running_total = 0
        for trade in completed_trades:
            running_total += trade['profit']
            equity_curve.append(running_total)
        
        if equity_curve:
            peak = 0
            max_dd = 0
            
            for i, equity in enumerate(equity_curve):
                peak = max(peak, equity)
                drawdown = peak - equity
                max_dd = max(max_dd, drawdown)
            
            self.max_drawdown = max_dd

# Enhanced CoinGecko Provider with Technical Indicators
class CoinGeckoProvider:
    """Enhanced provider for CoinGecko with technical analysis indicators"""
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        # Simple ticker mapping
        self.ticker_map = {
            "BTC": "bitcoin",
            "SOL": "solana"
        }
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 6  # Reduced from 12 to 6 seconds for faster updates
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a request to the CoinGecko API with rate limiting"""
        # Apply rate limiting
        time_since_last_request = time.time() - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        if params is None:
            params = {}
        
        # Make the request
        url = f"{self.base_url}/{endpoint}"
        logger.debug(f"Making request to {url} with params {params}")
        
        try:
            response = requests.get(url, params=params, timeout=30)
            self.last_request_time = time.time()
            
            # Check for rate limiting
            if response.status_code == 429:
                wait_time = int(response.headers.get('retry-after', 60))
                logger.warning(f"Rate limit hit. Waiting for {wait_time} seconds before retrying.")
                time.sleep(wait_time)
                return self._make_request(endpoint, params)
            
            # Handle other errors
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API request error: {e}")
            # Return empty dict instead of raising to avoid crashing
            return {}
    
    def get_coin_id(self, ticker: str) -> str:
        """Get the CoinGecko ID for a cryptocurrency ticker"""
        ticker = ticker.upper()
        
        # Check the predefined mappings first
        if ticker in self.ticker_map:
            return self.ticker_map[ticker]
        
        # Return the ticker in lowercase as fallback (might work for some coins)
        logger.warning(f"No mapping found for {ticker}, using lowercase ticker as ID")
        return ticker.lower()
    
    def get_market_data(self, ticker: str) -> MarketData:
        """Fetch current market data from CoinGecko"""
        try:
            coin_id = self.get_coin_id(ticker)
            
            # Use the simpler markets endpoint
            market_data = self._make_request('coins/markets', {
                'vs_currency': 'usd',
                'ids': coin_id,
                'price_change_percentage': '24h'
            })
            
            if not market_data or len(market_data) == 0:
                logger.error(f"No market data returned for {ticker}")
                raise ValueError(f"No market data returned for {ticker}")
            
            data = market_data[0]
            logger.debug(f"Received data for {ticker}: {data}")
            
            # Extract data from the response
            current_price = data.get('current_price', 0)
            price_change_24h = data.get('price_change_24h', 0)
            
            # Create base market data
            market_data = MarketData(
                ticker=ticker,
                price=current_price,
                open=current_price - price_change_24h if price_change_24h else current_price,
                high=data.get('high_24h', current_price),
                low=data.get('low_24h', current_price),
                volume=data.get('total_volume', 0),
                timestamp=datetime.datetime.now(),
                market_cap=data.get('market_cap', 0),
                price_change_24h=price_change_24h,
                price_change_percentage_24h=data.get('price_change_percentage_24h', 0)
            )
            
            # Add technical indicators from historical data
            self._add_technical_indicators(market_data, ticker)
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data for {ticker}: {e}")
            # Return placeholder data as fallback
            return MarketData(
                ticker=ticker,
                price=0,
                open=0,
                high=0,
                low=0,
                volume=0,
                timestamp=datetime.datetime.now()
            )
    
    def _add_technical_indicators(self, market_data: MarketData, ticker: str):
        """Calculate and add technical indicators to market data"""
        try:
            # Get 30 days of historical data for calculating indicators
            historical = self.get_historical_data(ticker, days=30)
            
            if not historical:
                logger.warning(f"Could not calculate technical indicators for {ticker}: no historical data")
                return
            
            # Convert to DataFrame for technical analysis
            df = pd.DataFrame([{
                'timestamp': h.timestamp,
                'close': h.price,
                'open': h.open,
                'high': h.high,
                'low': h.low,
                'volume': h.volume
            } for h in historical])
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Calculate RSI
            rsi = ta.momentum.RSIIndicator(df['close'], window=14)
            market_data.rsi = rsi.rsi().iloc[-1]
            
            # Calculate MACD
            macd = ta.trend.MACD(df['close'])
            market_data.macd = macd.macd().iloc[-1]
            market_data.macd_signal = macd.macd_signal().iloc[-1]
            market_data.macd_histogram = macd.macd_diff().iloc[-1]
            
            # Calculate Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['close'])
            market_data.bollinger_upper = bollinger.bollinger_hband().iloc[-1]
            market_data.bollinger_middle = bollinger.bollinger_mavg().iloc[-1]
            market_data.bollinger_lower = bollinger.bollinger_lband().iloc[-1]
            
            # Calculate Moving Averages
            market_data.sma_20 = ta.trend.sma_indicator(df['close'], window=20).iloc[-1]
            market_data.sma_50 = ta.trend.sma_indicator(df['close'], window=50).iloc[-1]
            market_data.sma_200 = ta.trend.sma_indicator(df['close'], window=200).iloc[-1]
            market_data.ema_12 = ta.trend.ema_indicator(df['close'], window=12).iloc[-1]
            market_data.ema_26 = ta.trend.ema_indicator(df['close'], window=26).iloc[-1]
            
            logger.debug(f"Added technical indicators for {ticker}")
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators for {ticker}: {e}")
    
    def get_historical_data(self, ticker: str, days: int = 30) -> List[MarketData]:
        """Fetch historical data from CoinGecko with technical indicators"""
        try:
            coin_id = self.get_coin_id(ticker)
            
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'daily'
            }
            
            data = self._make_request(f'coins/{coin_id}/market_chart', params)
            
            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])
            
            # Combine the data
            result = []
            for i, (price_data, volume_data) in enumerate(zip(prices, volumes)):
                timestamp = datetime.datetime.fromtimestamp(price_data[0] / 1000)
                price = price_data[1]
                
                result.append(MarketData(
                    ticker=ticker,
                    price=price,
                    open=price,  # Simplification
                    high=price,  # Simplification
                    low=price,   # Simplification
                    volume=volume_data[1],
                    timestamp=timestamp
                ))
            
            return result
        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker}: {e}")
            return []

# Enhanced CryptoPanic Provider with Keyword Extraction
class CryptoPanicProvider:
    """Enhanced provider for crypto news from CryptoPanic with sentiment analysis"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://cryptopanic.com/api/v1/posts/"
        # Keywords to track
        self.important_keywords = [
            "regulation", "sec", "lawsuit", "hack", "security", "breach", 
            "partnership", "adoption", "launch", "update", "upgrade", 
            "hardfork", "fork", "listing", "delisting", "bankruptcy",
            "whale", "pump", "dump", "scam", "fraud", "investigation"
        ]
    
    def get_news(self, ticker: str, limit: int = 10) -> List[NewsItem]:
        """Fetch crypto news from CryptoPanic API with enhanced analysis"""
        try:
            params = {
                'auth_token': self.api_key,
                'currencies': ticker,
                'public': 'true',
                'limit': limit
            }
            
            logger.debug(f"Fetching news for {ticker}")
            response = requests.get(self.base_url, params=params)
            
            if response.status_code != 200:
                logger.error(f"CryptoPanic API error: {response.status_code}")
                return []
            
            data = response.json()
            
            news_items = []
            for item in data.get('results', []):
                # Simple sentiment from votes (positive, negative or neutral)
                votes = item.get('votes', {})
                positive = votes.get('positive', 0)
                negative = votes.get('negative', 0)
                
                sentiment = "neutral"
                sentiment_score = 0.0
                
                if positive > negative:
                    sentiment = "positive"
                    sentiment_score = min(0.5 + (positive / (positive + negative + 1)) * 0.5, 1.0)
                elif negative > positive:
                    sentiment = "negative"
                    sentiment_score = max(-0.5 - (negative / (positive + negative + 1)) * 0.5, -1.0)
                
                # Parse timestamp if available
                timestamp = None
                if item.get('published_at'):
                    try:
                        timestamp = datetime.datetime.fromisoformat(item['published_at'].replace('Z', '+00:00'))
                    except:
                        timestamp = datetime.datetime.now()
                
                # Extract title and text
                title = item.get('title', '')
                
                # Extract mentioned tickers
                tickers_mentioned = []
                currencies = item.get('currencies', [])
                for currency in currencies:
                    currency_code = currency.get('code', '')
                    if currency_code:
                        tickers_mentioned.append(currency_code)
                
                # Extract important keywords
                keywords = []
                text = title.lower()
                for keyword in self.important_keywords:
                    if keyword in text:
                        keywords.append(keyword)
                
                # Create news item
                news_items.append(NewsItem(
                    title=title,
                    summary=title,  # Use title as summary for simplicity
                    source=item.get('source', {}).get('title', 'CryptoPanic'),
                    url=item.get('url', ''),
                    sentiment=sentiment,
                    sentiment_score=sentiment_score,
                    timestamp=timestamp,
                    tickers_mentioned=tickers_mentioned,
                    keywords=keywords
                ))
            
            return news_items
            
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")
            return []

# Fixed CryptoQuant Provider with proper response handling
class CryptoQuantProvider:
    """Enhanced provider for on-chain data using CryptoQuant API with improved error handling"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.cryptoquant.com/v1"
        self.headers = {'Authorization': f'Bearer {self.api_key}'}
        self.exchanges = {
            "BTC": ["binance", "coinbase", "ftx", "huobi", "okex", "kraken"],
            "ETH": ["binance", "coinbase", "ftx", "huobi", "okex"]
        }
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1  # 1 second between requests
        # Supported cryptocurrencies by CryptoQuant (only BTC is fully supported)
        self.supported_tickers = ["BTC", "ETH"]
        
        # Test the API key at init for better error detection
        self._test_api_key()
    
    def _test_api_key(self):
        """Test the API key to make sure it's valid"""
        try:
            test_endpoint = "btc/exchange-flows/reserve"
            test_params = {
                "exchange": "binance",
                "window": "day",
                "from": self._get_date_str(7),
                "limit": 1
            }
            
            response = requests.get(
                f"{self.base_url}/{test_endpoint}", 
                headers=self.headers, 
                params=test_params,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("CryptoQuant API key is valid")
            else:
                logger.warning(f"CryptoQuant API key may not be valid: {response.status_code} - {response.text}")
        except Exception as e:
            logger.warning(f"Could not verify CryptoQuant API key: {e}")
    
    def is_supported_ticker(self, ticker: str) -> bool:
        """Check if a ticker is supported by CryptoQuant"""
        return ticker.upper() in self.supported_tickers
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Any:
        """Make a request to the CryptoQuant API with improved error handling"""
        # Apply rate limiting
        time_since_last_request = time.time() - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            logger.debug(f"CryptoQuant rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        # Make the request
        url = f"{self.base_url}/{endpoint}"
        logger.debug(f"Making CryptoQuant request to {url}")
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            self.last_request_time = time.time()
            
            # Handle status code errors
            if response.status_code != 200:
                logger.error(f"CryptoQuant API error: {response.status_code} - {response.text}")
                return None
            
            # Log truncated response for debugging
            logger.debug(f"CryptoQuant response: {response.text[:300]}...")
            
            try:
                data = response.json()
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON response from CryptoQuant: {response.text[:100]}")
                return None
            
            # Check for error status in the nested structure
            status_code = data.get('status', {}).get('code', 0) 
            if status_code != 200:
                status_message = data.get('status', {}).get('message', 'Unknown error')
                logger.error(f"CryptoQuant API error: {status_message}")
                return None
                
            # Extract data from the nested structure
            # The actual data structure is: {"status": {...}, "result": {"window": "day", "data": [...]}}
            if 'result' in data and 'data' in data['result']:
                return data['result']['data']
            
            # Check for alternative data structures
            if 'data' in data:
                return data['data']
            elif 'items' in data:
                return data['items']
                
            logger.warning("CryptoQuant API returned unexpected data structure")
            return None
                
        except Exception as e:
            logger.error(f"CryptoQuant API request error: {e}")
            return None
    
    def get_exchange_reserve(self, ticker: str, exchange: str = "binance") -> Optional[float]:
        """Get exchange reserve data with improved error handling"""
        # Only supported coins (primarily BTC)
        if not self.is_supported_ticker(ticker):
            logger.warning(f"{ticker} not directly supported by CryptoQuant")
            return None
            
        ticker = ticker.lower()  # API expects lowercase
        
        endpoint = f"{ticker}/exchange-flows/reserve"
        params = {
            "exchange": exchange,
            "window": "day",
            "from": self._get_date_str(30),
            "limit": 2
        }
        
        try:
            data = self._make_request(endpoint, params)
            if not data or len(data) == 0:
                return None
            
            # Get the most recent result
            result = data[0]
            if "reserve" in result:
                return float(result["reserve"])
            elif "value" in result:
                return float(result["value"])
                
            return None
        except Exception as e:
            logger.error(f"Error getting exchange reserve for {ticker}: {e}")
            return None
    
    def get_exchange_netflow(self, ticker: str, exchange: str = "binance") -> Optional[float]:
        """Get exchange netflow data with improved error handling"""
        if not self.is_supported_ticker(ticker):
            return None
            
        ticker = ticker.lower()
        
        endpoint = f"{ticker}/exchange-flows/netflow"
        params = {
            "exchange": exchange,
            "window": "day",
            "from": self._get_date_str(30),
            "limit": 2
        }
        
        try:
            data = self._make_request(endpoint, params)
            if not data or len(data) == 0:
                return None
                
            # Get the most recent result
            result = data[0]
            if "netflow_total" in result:
                return float(result["netflow_total"])
            elif "value" in result:
                return float(result["value"])
                
            return None
        except Exception as e:
            logger.error(f"Error getting exchange netflow for {ticker}: {e}")
            return None
    
    def get_exchange_inflow(self, ticker: str, exchange: str = "binance") -> Optional[float]:
        """Get exchange inflow data with improved error handling"""
        if not self.is_supported_ticker(ticker):
            return None
            
        ticker = ticker.lower()
        
        endpoint = f"{ticker}/exchange-flows/inflow"
        params = {
            "exchange": exchange,
            "window": "day",
            "from": self._get_date_str(30),
            "limit": 2
        }
        
        try:
            data = self._make_request(endpoint, params)
            if not data or len(data) == 0:
                return None
                
            # Get the most recent result
            result = data[0]
            if "inflow_total" in result:
                return float(result["inflow_total"])
            elif "value" in result:
                return float(result["value"])
                
            return None
        except Exception as e:
            logger.error(f"Error getting exchange inflow for {ticker}: {e}")
            return None
    
    def get_exchange_outflow(self, ticker: str, exchange: str = "binance") -> Optional[float]:
        """Get exchange outflow data with improved error handling"""
        if not self.is_supported_ticker(ticker):
            return None
            
        ticker = ticker.lower()
        
        endpoint = f"{ticker}/exchange-flows/outflow"
        params = {
            "exchange": exchange,
            "window": "day",
            "from": self._get_date_str(30),
            "limit": 2
        }
        
        try:
            data = self._make_request(endpoint, params)
            if not data or len(data) == 0:
                return None
                
            # Get the most recent result
            result = data[0]
            if "outflow_total" in result:
                return float(result["outflow_total"])
            elif "value" in result:
                return float(result["value"])
                
            return None
        except Exception as e:
            logger.error(f"Error getting exchange outflow for {ticker}: {e}")
            return None
    
    def get_transactions_count(self, ticker: str, exchange: str = "binance") -> Optional[int]:
        """Get transactions count data with improved error handling"""
        if not self.is_supported_ticker(ticker):
            return None
            
        ticker = ticker.lower()
        
        endpoint = f"{ticker}/exchange-flows/transactions-count"
        params = {
            "exchange": exchange,
            "window": "day",
            "from": self._get_date_str(30),
            "limit": 2
        }
        
        try:
            data = self._make_request(endpoint, params)
            if not data or len(data) == 0:
                return None
                
            # Get the most recent result
            result = data[0]
            if "transactions_count_inflow" in result:
                return int(result["transactions_count_inflow"])
            elif "value" in result:
                return int(result["value"])
                
            return None
        except Exception as e:
            logger.error(f"Error getting transactions count for {ticker}: {e}")
            return None
    
    def get_addresses_count(self, ticker: str, exchange: str = "binance") -> Optional[int]:
        """Get addresses count data with improved error handling"""
        if not self.is_supported_ticker(ticker):
            return None
            
        ticker = ticker.lower()
        
        endpoint = f"{ticker}/exchange-flows/addresses-count"
        params = {
            "exchange": exchange,
            "window": "day",
            "from": self._get_date_str(30),
            "limit": 2
        }
        
        try:
            data = self._make_request(endpoint, params)
            if not data or len(data) == 0:
                return None
                
            # Get the most recent result
            result = data[0]
            if "addresses_count_inflow" in result:
                return int(result["addresses_count_inflow"])
            elif "value" in result:
                return int(result["value"])
                
            return None
        except Exception as e:
            logger.error(f"Error getting addresses count for {ticker}: {e}")
            return None
    
    def get_in_house_flow(self, ticker: str, exchange: str = "binance") -> Optional[float]:
        """Get in-house flow data with improved error handling"""
        if not self.is_supported_ticker(ticker):
            return None
            
        ticker = ticker.lower()
        
        endpoint = f"{ticker}/exchange-flows/in-house-flow"
        params = {
            "exchange": exchange,
            "window": "day",
            "from": self._get_date_str(30),
            "limit": 2
        }
        
        try:
            data = self._make_request(endpoint, params)
            if not data or len(data) == 0:
                return None
                
            # Get the most recent result
            result = data[0]
            if "value" in result:
                return float(result["value"])
                
            return None
        except Exception as e:
            logger.error(f"Error getting in-house flow for {ticker}: {e}")
            return None
    
    def get_mpi(self, ticker: str) -> Optional[float]:
        """Get Miner Position Index (MPI) data with improved error handling"""
        # MPI is BTC specific only
        if ticker.upper() != "BTC":
            logger.info(f"MPI is BTC-specific, not available for {ticker}")
            return None
            
        endpoint = "btc/flow-indicator/mpi"
        params = {
            "window": "day",
            "from": self._get_date_str(30),
            "limit": 2
        }
        
        try:
            data = self._make_request(endpoint, params)
            if not data or len(data) == 0:
                return None
                
            # Get the most recent result
            result = data[0]
            if "mpi" in result:
                return float(result["mpi"])
            elif "value" in result:
                return float(result["value"])
                
            return None
        except Exception as e:
            logger.error(f"Error getting MPI for {ticker}: {e}")
            return None
    
    def get_exchange_shutdown_index(self, ticker: str, exchange: str = "binance") -> Optional[float]:
        """Get exchange shutdown index data with improved error handling"""
        if not self.is_supported_ticker(ticker):
            return None
            
        ticker = ticker.lower()
        
        endpoint = f"{ticker}/flow-indicator/exchange-shutdown-index"
        params = {
            "exchange": exchange,
            "window": "day",
            "from": self._get_date_str(30),
            "limit": 2
        }
        
        try:
            data = self._make_request(endpoint, params)
            if not data or len(data) == 0:
                return None
                
            # Get the most recent result
            result = data[0]
            if "is_shutdown" in result:
                return float(result["is_shutdown"])
            elif "value" in result:
                return float(result["value"])
                
            return None
        except Exception as e:
            logger.error(f"Error getting exchange shutdown index for {ticker}: {e}")
            return None
    
    def get_exchange_whale_ratio(self, ticker: str, exchange: str = "binance") -> Optional[float]:
        """Get exchange whale ratio data with improved error handling"""
        if not self.is_supported_ticker(ticker):
            return None
            
        ticker = ticker.lower()
        
        endpoint = f"{ticker}/flow-indicator/exchange-whale-ratio"
        params = {
            "exchange": exchange,
            "window": "day",
            "from": self._get_date_str(30),
            "limit": 2
        }
        
        try:
            data = self._make_request(endpoint, params)
            if not data or len(data) == 0:
                return None
                
            # Get the most recent result
            result = data[0]
            if "exchange_whale_ratio" in result:
                return float(result["exchange_whale_ratio"])
            elif "value" in result:
                return float(result["value"])
                
            return None
        except Exception as e:
            logger.error(f"Error getting exchange whale ratio for {ticker}: {e}")
            return None
    
    def get_fund_flow_ratio(self, ticker: str, exchange: str = "binance") -> Optional[float]:
        """Get fund flow ratio data with improved error handling"""
        if not self.is_supported_ticker(ticker):
            return None
            
        ticker = ticker.lower()
        
        endpoint = f"{ticker}/flow-indicator/fund-flow-ratio"
        params = {
            "exchange": exchange,
            "window": "day",
            "from": self._get_date_str(30),
            "limit": 2
        }
        
        try:
            data = self._make_request(endpoint, params)
            if not data or len(data) == 0:
                return None
                
            # Get the most recent result
            result = data[0]
            if "fund_flow_ratio" in result:
                return float(result["fund_flow_ratio"])
            elif "value" in result:
                return float(result["value"])
                
            return None
        except Exception as e:
            logger.error(f"Error getting fund flow ratio for {ticker}: {e}")
            return None
    
    def get_stablecoins_ratio(self, ticker: str, exchange: str = "binance") -> Optional[float]:
        """Get stablecoins ratio data with improved error handling"""
        if not self.is_supported_ticker(ticker):
            return None
            
        ticker = ticker.lower()
        
        endpoint = f"{ticker}/flow-indicator/stablecoins-ratio"
        params = {
            "exchange": exchange,
            "window": "day",
            "from": self._get_date_str(30),
            "limit": 2
        }
        
        try:
            data = self._make_request(endpoint, params)
            if not data or len(data) == 0:
                return None
                
            # Get the most recent result
            result = data[0]
            if "stablecoins_ratio" in result:
                return float(result["stablecoins_ratio"])
            elif "value" in result:
                return float(result["value"])
                
            return None
        except Exception as e:
            logger.error(f"Error getting stablecoins ratio for {ticker}: {e}")
            return None
    
    def get_inflow_age_distribution(self, ticker: str, exchange: str = "binance") -> Dict[str, float]:
        """Get inflow age distribution data with improved error handling"""
        if not self.is_supported_ticker(ticker):
            return {}
            
        ticker = ticker.lower()
        
        endpoint = f"{ticker}/flow-indicator/exchange-inflow-age-distribution"
        params = {
            "exchange": exchange,
            "window": "day",
            "from": self._get_date_str(30),
            "limit": 2
        }
        
        try:
            data = self._make_request(endpoint, params)
            if not data or len(data) == 0:
                return {}
                
            # Get the most recent result
            result = data[0]
            if not result:
                return {}
                
            # Parse the age distribution, excluding timestamp fields
            age_distribution = {}
            for key, value in result.items():
                if key not in ["timestamp", "date"]:
                    try:
                        age_distribution[key] = float(value)
                    except (ValueError, TypeError):
                        pass
            
            return age_distribution
        except Exception as e:
            logger.error(f"Error getting inflow age distribution for {ticker}: {e}")
            return {}
    
    def get_inflow_supply_distribution(self, ticker: str, exchange: str = "binance") -> Dict[str, float]:
        """Get inflow supply distribution data with improved error handling"""
        if not self.is_supported_ticker(ticker):
            return {}
            
        ticker = ticker.lower()
        
        endpoint = f"{ticker}/flow-indicator/exchange-inflow-supply-distribution"
        params = {
            "exchange": exchange,
            "window": "day",
            "from": self._get_date_str(30),
            "limit": 2
        }
        
        try:
            data = self._make_request(endpoint, params)
            if not data or len(data) == 0:
                return {}
                
            # Get the most recent result
            result = data[0]
            if not result:
                return {}
                
            # Parse the supply distribution, excluding timestamp fields
            supply_distribution = {}
            for key, value in result.items():
                if key not in ["timestamp", "date"]:
                    try:
                        supply_distribution[key] = float(value)
                    except (ValueError, TypeError):
                        pass
            
            return supply_distribution
        except Exception as e:
            logger.error(f"Error getting inflow supply distribution for {ticker}: {e}")
            return {}
    
    def get_sopr(self, ticker: str) -> Optional[float]:
        """Get Spent Output Profit Ratio (SOPR) data with improved error handling"""
        if not self.is_supported_ticker(ticker):
            return None
            
        ticker = ticker.lower()
        
        endpoint = f"{ticker}/market-indicator/sopr"
        params = {
            "window": "day",
            "from": self._get_date_str(30),
            "limit": 2
        }
        
        try:
            data = self._make_request(endpoint, params)
            if not data or len(data) == 0:
                return None
                
            # Get the most recent result
            result = data[0]
            if "sopr" in result:
                return float(result["sopr"])
            elif "value" in result:
                return float(result["value"])
                
            return None
        except Exception as e:
            logger.error(f"Error getting SOPR for {ticker}: {e}")
            return None
    
    def _get_date_str(self, days_ago: int = 30) -> str:
        """Get date string in YYYYMMDD format for X days ago"""
        date = datetime.datetime.now() - datetime.timedelta(days=days_ago)
        return date.strftime("%Y%m%d")


# Enhanced OnChain Data Provider with improved CryptoQuant integration
class OnChainDataProvider:
    """Enhanced provider for on-chain data using CryptoQuant API with improved error handling"""
    
    def __init__(self, config: Config):
        self.config = config
        self.web3_connections = {}
        self.cryptoquant = CryptoQuantProvider(config.cryptoquant_api_key)
        
        # Cache for simulated data to maintain consistency
        self.simulation_cache = {}
        
        # Whale wallets for monitoring (public addresses of major holders)
        self.whale_wallets = {
            "BTC": [
                "1P5ZEDWTKTFGxQjZphgWPQUpe554WKDfHQ",  # Binance
                "3Kzh9qAqVWQhEsfQz7zEQL1EuSx5tyNLNS",  # Bitfinex
                "bc1qgdjqv0av3q56jvd82tkdjpy7gdp9ut8tlqmgrpmv24sq90ecnvqqjwvw97"  # Largest BTC wallet
            ],
            "ETH": [
                "0xDA9dfA130Df4dE4673b89022EE50ff26f6EA73Cf",  # Binance
                "0x28C6c06298d514Db089934071355E5743bf21d60",  # Binance 2
                "0xA929022c9107643515F5c777cE9a910F0D1e490C"  # Major wallet
            ],
            "SOL": [
                "3LKy8xNEWAzNtX9YcPj3pTxce5Wxh6Lq8mzuMRz7zxVM",  # Major wallet
                "8rUvvjhJHMJrfMwGC4QX9aDL1T8maYvJ5Dq4qxcXBjK6",  # Major wallet
                "7vYe1Dzuod7BgwVCFVFMGHWtpMvXCi5wAtUwR3422qZ4"  # Exchange wallet
            ]
        }
        
        # Initialize Web3 connections for additional on-chain data
        for network, provider_url in self.config.web3_providers.items():
            try:
                self.web3_connections[network] = Web3(Web3.HTTPProvider(provider_url))
                logger.info(f"Connected to {network} blockchain")
            except Exception as e:
                logger.error(f"Failed to connect to {network} blockchain: {e}")
    
    def _get_network_for_ticker(self, ticker: str) -> Optional[str]:
        """Map ticker to appropriate blockchain network"""
        ticker = ticker.upper()
        if ticker == "BTC":
            return "bitcoin"
        elif ticker in ["ETH", "LINK", "UNI", "AAVE", "MKR"]:
            return "ethereum"
        elif ticker in ["BNB", "CAKE"]:
            return "bsc"
        elif ticker in ["SOL"]:
            return "solana"
        elif ticker in ["MATIC"]:
            return "polygon"
        else:
            return None
    
    def get_onchain_data(self, ticker: str) -> OnChainData:
        """Get enhanced on-chain data with improved error handling and metric tracking"""
        network = self._get_network_for_ticker(ticker)
        original_ticker = ticker.upper()  # Save original ticker
        
        # Create base OnChainData object
        onchain_data = OnChainData(
            ticker=original_ticker,
            network=network if network else "unknown",
            timestamp=datetime.datetime.now()
        )
        
        # Check if ticker is supported by CryptoQuant
        is_supported = original_ticker in self.config.cryptoquant_supported
        
        # For SOL and other unsupported coins, use dedicated simulation directly
        if not is_supported:
            logger.info(f"{original_ticker} is not directly supported by CryptoQuant, using specialized simulated data")
            return self._get_specialized_simulated_data(original_ticker, network)
        
        # If we reach here, we're dealing with BTC or another supported cryptocurrency
        try:
            # For BTC, attempt to get real data from CryptoQuant
            logger.info(f"Fetching on-chain data for {original_ticker} from CryptoQuant")
            
            # Use lowercase ticker for API calls
            ticker_lower = original_ticker.lower()
            exchange = "binance"  # Default exchange
            
            # Track successful metrics to determine if we have enough real data
            successful_metrics = 0
            
            # Get each metric with extensive error handling and tracking
            
            # Exchange Reserve - critical metric
            retry_count = 0
            max_retries = 2
            
            while retry_count < max_retries:
                onchain_data.exchange_reserve = self.cryptoquant.get_exchange_reserve(original_ticker, exchange)
                if onchain_data.exchange_reserve is not None:
                    successful_metrics += 1
                    break
                retry_count += 1
                logger.warning(f"Retry {retry_count}/{max_retries} for exchange_reserve")
                time.sleep(1)  # Brief pause between retries
            
            # Exchange Netflow
            onchain_data.exchange_netflow = self.cryptoquant.get_exchange_netflow(original_ticker, exchange)
            if onchain_data.exchange_netflow is not None:
                successful_metrics += 1
            
            # Exchange Inflow
            onchain_data.exchange_inflow = self.cryptoquant.get_exchange_inflow(original_ticker, exchange)
            if onchain_data.exchange_inflow is not None:
                successful_metrics += 1
            
            # Exchange Outflow
            onchain_data.exchange_outflow = self.cryptoquant.get_exchange_outflow(original_ticker, exchange)
            if onchain_data.exchange_outflow is not None:
                successful_metrics += 1
            
            # Store exchange_inflows and exchange_outflows for backward compatibility
            onchain_data.exchange_inflows = onchain_data.exchange_inflow
            onchain_data.exchange_outflows = onchain_data.exchange_outflow
            
            # Transaction Count
            onchain_data.transaction_count = self.cryptoquant.get_transactions_count(original_ticker, exchange)
            if onchain_data.transaction_count is not None:
                successful_metrics += 1
            
            # Address Count
            onchain_data.address_count = self.cryptoquant.get_addresses_count(original_ticker, exchange)
            if onchain_data.address_count is not None:
                successful_metrics += 1
            
            # In-house Flow
            onchain_data.in_house_flow = self.cryptoquant.get_in_house_flow(original_ticker, exchange)
            if onchain_data.in_house_flow is not None:
                successful_metrics += 1
            
            # MPI (BTC only)
            if original_ticker == "BTC":
                onchain_data.mpi = self.cryptoquant.get_mpi(original_ticker)
                if onchain_data.mpi is not None:
                    successful_metrics += 1
            
            # Exchange Shutdown Index
            onchain_data.exchange_shutdown_index = self.cryptoquant.get_exchange_shutdown_index(original_ticker, exchange)
            if onchain_data.exchange_shutdown_index is not None:
                successful_metrics += 1
            
            # Exchange Whale Ratio
            onchain_data.exchange_whale_ratio = self.cryptoquant.get_exchange_whale_ratio(original_ticker, exchange)
            if onchain_data.exchange_whale_ratio is not None:
                successful_metrics += 1
            
            # Fund Flow Ratio
            onchain_data.fund_flow_ratio = self.cryptoquant.get_fund_flow_ratio(original_ticker, exchange)
            if onchain_data.fund_flow_ratio is not None:
                successful_metrics += 1
            
            # Stablecoins Ratio
            onchain_data.stablecoins_ratio = self.cryptoquant.get_stablecoins_ratio(original_ticker, exchange)
            if onchain_data.stablecoins_ratio is not None:
                successful_metrics += 1
            
            # Inflow Age Distribution
            onchain_data.inflow_age_distribution = self.cryptoquant.get_inflow_age_distribution(original_ticker, exchange)
            if onchain_data.inflow_age_distribution:
                successful_metrics += 1
            
            # Inflow Supply Distribution
            onchain_data.inflow_supply_distribution = self.cryptoquant.get_inflow_supply_distribution(original_ticker, exchange)
            if onchain_data.inflow_supply_distribution:
                successful_metrics += 1
            
            # SOPR
            onchain_data.sopr = self.cryptoquant.get_sopr(original_ticker)
            if onchain_data.sopr is not None:
                successful_metrics += 1
            
            # Check if we got enough real data - we need at least 2 successful metrics
            if successful_metrics >= 2:
                # We have real data - calculate remaining derived metrics
                
                # Mark as real data
                onchain_data.is_simulated = False
                
                # Estimate active addresses if not directly available
                if onchain_data.active_addresses_24h is None and onchain_data.address_count is not None:
                    onchain_data.active_addresses_24h = int(onchain_data.address_count * 1.5)  # Conservative estimate
                
                # Calculate transaction volume if not directly available
                if onchain_data.transaction_volume_24h is None and onchain_data.exchange_inflow is not None and onchain_data.exchange_outflow is not None:
                    onchain_data.transaction_volume_24h = onchain_data.exchange_inflow + onchain_data.exchange_outflow
                
                # Calculate average transaction value if possible
                if onchain_data.transaction_volume_24h is not None and onchain_data.transaction_count is not None and onchain_data.transaction_count > 0:
                    onchain_data.avg_transaction_value = onchain_data.transaction_volume_24h / onchain_data.transaction_count
                
                # Add whale wallet simulation - in a production system, this would use real wallet tracking
                if original_ticker in self.whale_wallets:
                    for wallet in self.whale_wallets[original_ticker]:
                        # Use exchange netflow direction to guide the simulation
                        netflow_signal = 1 if onchain_data.exchange_netflow and onchain_data.exchange_netflow > 0 else -1
                        change = np.random.normal(0, 50000) * netflow_signal  
                        if abs(change) > 25000:  # Only track significant changes
                            onchain_data.whale_wallet_changes[wallet] = change
                
                logger.info(f"Successfully fetched {successful_metrics} on-chain metrics for {original_ticker} from CryptoQuant")
                return onchain_data
            else:
                # Not enough real data - fall back to simulation
                logger.error(f"Failed to get minimum required on-chain data from CryptoQuant for {original_ticker} (only got {successful_metrics} metrics)")
                return self._get_specialized_simulated_data(original_ticker, network)
            
        except Exception as e:
            logger.error(f"Error fetching on-chain data from CryptoQuant for {original_ticker}: {e}")
            # Fall back to specialized simulation
            return self._get_specialized_simulated_data(original_ticker, network)
    
    def _get_specialized_simulated_data(self, ticker: str, network: str) -> OnChainData:
        """Generate realistic simulated on-chain data customized for each cryptocurrency"""
        logger.warning(f"Using specialized simulated on-chain data for {ticker}")
        
        # Create base OnChainData object with simulation flag set to True
        onchain_data = OnChainData(
            ticker=ticker,
            network=network if network else "unknown",
            timestamp=datetime.datetime.now(),
            is_simulated=True  # Mark as simulated data
        )
        
        # Check if we have cached simulation data to maintain consistency
        if ticker in self.simulation_cache:
            cache_time = self.simulation_cache[ticker]['timestamp']
            current_time = datetime.datetime.now()
            
            # Use cached data if it's less than 1 hour old
            if (current_time - cache_time).total_seconds() < 3600:
                logger.info(f"Using cached simulation data for {ticker}")
                return self.simulation_cache[ticker]['data']
        
        # Customize simulation based on the ticker to make it realistic
        ticker = ticker.upper()
        
        if ticker == "BTC":
            # BTC-specific realistic simulation based on typical values
            onchain_data.active_addresses_24h = np.random.randint(900000, 1200000)
            onchain_data.transaction_volume_24h = np.random.uniform(8000000000, 15000000000)
            onchain_data.exchange_reserve = np.random.uniform(500000, 650000)
            onchain_data.exchange_netflow = np.random.normal(0, 2000)
            onchain_data.exchange_inflow = np.random.uniform(20000, 50000)
            onchain_data.exchange_outflow = np.random.uniform(20000, 50000)
            onchain_data.transaction_count = np.random.randint(300000, 500000)
            onchain_data.address_count = np.random.randint(700000, 900000)
            onchain_data.in_house_flow = np.random.uniform(10000, 30000)
            onchain_data.mpi = np.random.uniform(0, 3)
            onchain_data.exchange_shutdown_index = np.random.uniform(0, 0.1)
            onchain_data.exchange_whale_ratio = np.random.uniform(0.5, 0.9)
            onchain_data.fund_flow_ratio = np.random.uniform(0.7, 1.2)
            onchain_data.stablecoins_ratio = np.random.uniform(0.8, 1.5)
            onchain_data.sopr = np.random.uniform(0.9, 1.1)
            
            # Create realistic inflow age distribution
            onchain_data.inflow_age_distribution = {
                "range_0d_1d": np.random.uniform(10000, 20000),
                "range_1d_1w": np.random.uniform(30000, 60000),
                "range_1w_1m": np.random.uniform(50000, 100000),
                "range_1m_3m": np.random.uniform(150000, 300000),
                "range_3m_6m": np.random.uniform(200000, 400000),
                "range_6m_12m": np.random.uniform(300000, 600000)
            }
            
            # Create realistic inflow supply distribution
            onchain_data.inflow_supply_distribution = {
                "range_0_001": np.random.uniform(5000, 10000),
                "range_001_01": np.random.uniform(20000, 40000),
                "range_01_1": np.random.uniform(50000, 100000),
                "range_1_10": np.random.uniform(100000, 200000),
                "range_10_100": np.random.uniform(150000, 300000),
                "range_100_1k": np.random.uniform(1000, 5000)
            }
            
        elif ticker == "SOL":
            # SOL-specific realistic simulation based on metrics that would be typical for Solana
            # Using value ranges that reflect SOL's ecosystem characteristics
            onchain_data.active_addresses_24h = np.random.randint(350000, 650000)
            onchain_data.transaction_volume_24h = np.random.uniform(600000000, 2000000000)
            onchain_data.exchange_reserve = np.random.uniform(80000000, 120000000)
            onchain_data.exchange_netflow = np.random.normal(0, 200000)
            onchain_data.exchange_inflow = np.random.uniform(2000000, 6000000)
            onchain_data.exchange_outflow = np.random.uniform(2000000, 6000000)
            onchain_data.transaction_count = np.random.randint(25000000, 40000000)  # SOL has very high TPS
            onchain_data.address_count = np.random.randint(250000, 400000)
            onchain_data.in_house_flow = np.random.uniform(1000000, 3000000)
            # No MPI for SOL
            onchain_data.exchange_shutdown_index = np.random.uniform(0, 0.15)
            onchain_data.exchange_whale_ratio = np.random.uniform(0.4, 0.8)
            onchain_data.fund_flow_ratio = np.random.uniform(0.6, 1.1)
            onchain_data.stablecoins_ratio = np.random.uniform(0.7, 1.4)
            onchain_data.sopr = np.random.uniform(0.95, 1.05)
            
            # Create SOL-specific inflow age distribution
            onchain_data.inflow_age_distribution = {
                "range_0d_1d": np.random.uniform(500000, 1000000),
                "range_1d_1w": np.random.uniform(1500000, 3000000),
                "range_1w_1m": np.random.uniform(3000000, 6000000),
                "range_1m_3m": np.random.uniform(6000000, 10000000),
                "range_3m_6m": np.random.uniform(10000000, 15000000),
                "range_6m_12m": np.random.uniform(15000000, 25000000)
            }
            
            # Create SOL-specific inflow supply distribution
            onchain_data.inflow_supply_distribution = {
                "range_0_001": np.random.uniform(300000, 600000),
                "range_001_01": np.random.uniform(800000, 1500000),
                "range_01_1": np.random.uniform(2000000, 4000000),
                "range_1_10": np.random.uniform(6000000, 10000000),
                "range_10_100": np.random.uniform(12000000, 20000000),
                "range_100_1k": np.random.uniform(100000, 500000)
            }
        else:
            # Generic simulation for other cryptocurrencies
            onchain_data.active_addresses_24h = np.random.randint(50000, 500000)
            onchain_data.transaction_volume_24h = np.random.uniform(10000000, 500000000)
            onchain_data.exchange_reserve = np.random.uniform(1000000, 10000000)
            onchain_data.exchange_netflow = np.random.normal(0, 50000)
            onchain_data.exchange_inflow = np.random.uniform(500000, 2000000)
            onchain_data.exchange_outflow = np.random.uniform(500000, 2000000)
            onchain_data.transaction_count = np.random.randint(50000, 200000)
            onchain_data.address_count = np.random.randint(100000, 300000)
            onchain_data.in_house_flow = np.random.uniform(200000, 500000)
            # No MPI for others
            onchain_data.exchange_shutdown_index = np.random.uniform(0, 0.2)
            onchain_data.exchange_whale_ratio = np.random.uniform(0.3, 0.7)
            onchain_data.fund_flow_ratio = np.random.uniform(0.5, 1.0)
            onchain_data.stablecoins_ratio = np.random.uniform(0.6, 1.2)
            onchain_data.sopr = np.random.uniform(0.9, 1.1)
        
        # Calculate avg_transaction_value consistently
        if onchain_data.transaction_count and onchain_data.transaction_count > 0:
            onchain_data.avg_transaction_value = onchain_data.transaction_volume_24h / onchain_data.transaction_count
        
        # Ensure exchange flows are consistent with netflow
        if onchain_data.exchange_netflow is not None:
            netflow = onchain_data.exchange_netflow
            if onchain_data.exchange_inflow is None:
                onchain_data.exchange_inflow = max(0, netflow) + np.random.uniform(100000, 500000)
            if onchain_data.exchange_outflow is None:
                onchain_data.exchange_outflow = max(0, -netflow) + np.random.uniform(100000, 500000)
        
        # Set exchange_inflows and exchange_outflows for backward compatibility
        onchain_data.exchange_inflows = onchain_data.exchange_inflow
        onchain_data.exchange_outflows = onchain_data.exchange_outflow
        
        # Generate realistic whale wallet changes
        if ticker in self.whale_wallets:
            for wallet in self.whale_wallets[ticker]:
                # Calculate change based on netflow direction
                netflow_signal = 1 if onchain_data.exchange_netflow and onchain_data.exchange_netflow > 0 else -1
                
                # Scale change amount based on the ticker
                if ticker == "BTC":
                    change_scale = 100000
                elif ticker == "SOL":
                    change_scale = 1000000
                else:
                    change_scale = 50000
                    
                change = np.random.normal(0, change_scale) * netflow_signal  
                if abs(change) > change_scale / 2:  # Only track significant changes
                    onchain_data.whale_wallet_changes[wallet] = change
        
        # Simulate a few large transactions
        transaction_count = np.random.randint(0, 4)  # 0-3 large transactions
        for _ in range(transaction_count):
            # Scale transaction value based on ticker
            if ticker == "BTC":
                tx_value = np.random.uniform(500000, 5000000)  # $500K to $5M for BTC
            elif ticker == "SOL":
                tx_value = np.random.uniform(200000, 2000000)  # $200K to $2M for SOL
            else:
                tx_value = np.random.uniform(100000, 1000000)  # $100K to $1M for others
                
            tx_type = np.random.choice(["deposit", "withdrawal", "transfer"])
            onchain_data.large_transactions.append({
                "value": tx_value,
                "type": tx_type,
                "timestamp": datetime.datetime.now() - datetime.timedelta(hours=np.random.randint(0, 24))
            })
        
        # Cache the simulation for consistency
        self.simulation_cache[ticker] = {
            'timestamp': datetime.datetime.now(),
            'data': onchain_data
        }
        
        return onchain_data

# Enhanced Market Analyst with Technical Analysis
class MarketAnalyst:
    """Enhanced Market Analyst with technical indicators"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = genai.Client(api_key=config.gemini_api_key)
    
    def analyze(self, ticker: str, market_data: MarketData, historical_data: List[MarketData]) -> str:
        """Analyze market data with technical indicators and provide insights"""
        if not market_data or market_data.price <= 0:
            return "Insufficient data for analysis"
        
        # Create an enhanced prompt with technical indicators
        
        prompt = f"""
        You are an expert cryptocurrency market analyst with deep knowledge of technical analysis. Analyze this comprehensive market data for {ticker}:

        Current price: ${market_data.price:.2f}
        24h High: ${market_data.high:.2f}
        24h Low: ${market_data.low:.2f}
        24h Change: {market_data.price_change_percentage_24h:.2f}%
        24h Volume: ${market_data.volume:.2f}
        Market Cap: ${market_data.market_cap:.2f}

        Technical Indicators:
        RSI (14): {f"{market_data.rsi:.2f}" if market_data.rsi is not None and not pd.isna(market_data.rsi) else "N/A"}
        MACD: {f"{market_data.macd:.4f}" if market_data.macd is not None and not pd.isna(market_data.macd) else "N/A"}
        MACD Signal: {f"{market_data.macd_signal:.4f}" if market_data.macd_signal is not None and not pd.isna(market_data.macd_signal) else "N/A"}
        MACD Histogram: {f"{market_data.macd_histogram:.4f}" if market_data.macd_histogram is not None and not pd.isna(market_data.macd_histogram) else "N/A"}

        Bollinger Bands:
          - Upper: ${f"{market_data.bollinger_upper:.2f}" if market_data.bollinger_upper is not None and not pd.isna(market_data.bollinger_upper) else "N/A"}
          - Middle: ${f"{market_data.bollinger_middle:.2f}" if market_data.bollinger_middle is not None and not pd.isna(market_data.bollinger_middle) else "N/A"}
          - Lower: ${f"{market_data.bollinger_lower:.2f}" if market_data.bollinger_lower is not None and not pd.isna(market_data.bollinger_lower) else "N/A"}
        Moving Averages:
          - SMA 20: ${f"{market_data.sma_20:.2f}" if market_data.sma_20 is not None and not pd.isna(market_data.sma_20) else "N/A"}
          - SMA 50: ${f"{market_data.sma_50:.2f}" if market_data.sma_50 is not None and not pd.isna(market_data.sma_50) else "N/A"}
          - SMA 200: ${f"{market_data.sma_200:.2f}" if market_data.sma_200 is not None and not pd.isna(market_data.sma_200) else "N/A"}
          - EMA 12: ${f"{market_data.ema_12:.2f}" if market_data.ema_12 is not None and not pd.isna(market_data.ema_12) else "N/A"}
          - EMA 26: ${f"{market_data.ema_26:.2f}" if market_data.ema_26 is not None and not pd.isna(market_data.ema_26) else "N/A"}
        
        Historical price data (last {len(historical_data)} days):
        {self._format_historical_data(historical_data)}
        
        Provide a comprehensive market analysis focusing on:
        1. Overall trend direction (bullish, bearish, or neutral)
        2. Support/resistance levels (identify key price levels)
        3. Technical indicator analysis (RSI, MACD, Bollinger Bands, MAs)
        4. Chart patterns or formations (if any)
        5. Volume analysis
        6. Short-term price prediction (1-7 days)
        7. Medium-term outlook (1-4 weeks)
        
        Keep your analysis structured but comprehensive. Focus on data-driven insights, not general market sentiment.
        """
        
        try:
            # Create a chat and send the prompt
            chat = self.client.chats.create(model=self.config.model_name)
            response = chat.send_message(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating market analysis: {e}")
            return f"Error generating analysis: {str(e)}"
    
    def _format_historical_data(self, data: List[MarketData]) -> str:
        """Format historical data for the prompt"""
        result = []
        
        # Only use the last 7 data points to keep the prompt manageable
        sample_data = data[-7:] if len(data) > 7 else data
        
        for entry in sample_data:
            result.append(f"{entry.timestamp.strftime('%Y-%m-%d')}: Price: ${entry.price:.2f}, Volume: ${entry.volume:.2f}")
        
        return "\n".join(result)

# Enhanced News Analyst with Topic Extraction
class NewsAnalyst:
    """Enhanced News Analyst with topic extraction"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = genai.Client(api_key=config.gemini_api_key)
    
    def analyze(self, ticker: str, news_items: List[NewsItem]) -> str:
        """Analyze news with topic extraction and provide insights"""
        if not news_items:
            return "No news items available for analysis"
        
        # Extract keywords and sentiment from news items
        keywords = set()
        sentiment_scores = []
        for item in news_items:
            keywords.update(item.keywords)
            if item.sentiment_score is not None:
                sentiment_scores.append(item.sentiment_score)
        
        # Calculate average sentiment
        avg_sentiment = 0
        if sentiment_scores:
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        # Create enhanced prompt for news analysis
        prompt = f"""
        You are an expert cryptocurrency news analyst. Analyze these news items for {ticker} with particular attention to keywords and sentiment:
        
        {self._format_news_items(news_items)}
        
        Important keywords detected: {', '.join(keywords) if keywords else 'None'}
        Average sentiment score: {avg_sentiment:.2f} (-1.0 is very negative, +1.0 is very positive)
        
        Based on these news items:
        1. What is the overall news sentiment (bullish, bearish, or neutral)?
        2. Identify any major events or developments and explain their significance for {ticker}
        3. Analyze recurring themes or topics in the news
        4. Are there any regulatory or legal concerns mentioned?
        5. How might institutional or whale investors react to this news?
        6. How might this news affect market behavior in the short term (1-7 days)?
        7. Are there any contradicting news items that could create market confusion?
        
        Provide a detailed news analysis that synthesizes all this information.
        """
        
        try:
            # Create a chat and send the prompt
            chat = self.client.chats.create(model=self.config.model_name)
            response = chat.send_message(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating news analysis: {e}")
            return f"Error generating analysis: {str(e)}"
    
    def _format_news_items(self, news_items: List[NewsItem]) -> str:
        """Format news items for the prompt"""
        result = []
        
        for i, item in enumerate(news_items[:8]):  # Include up to 8 news items
            timestamp_str = item.timestamp.strftime('%Y-%m-%d') if item.timestamp else 'N/A'
            keywords_str = ', '.join(item.keywords) if item.keywords else 'None'
            tickers_str = ', '.join(item.tickers_mentioned) if item.tickers_mentioned else 'None'
            
            # Format sentiment score properly
            if item.sentiment_score is not None:
                sentiment_score_str = f"{item.sentiment_score:.2f}"
            else:
                sentiment_score_str = "0.0"
            
            result.append(f"""
            {i+1}. Title: {item.title}
            Source: {item.source}
            Date: {timestamp_str}
            Sentiment: {item.sentiment} (score: {sentiment_score_str})
            Keywords: {keywords_str}
            Tickers mentioned: {tickers_str}
            """)
        
        return "\n".join(result)

# Enhanced OnChain Analyst with improved data handling
class OnChainAnalyst:
    """Enhanced analyst for on-chain data with clear indication of real vs. simulated data"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = genai.Client(api_key=config.gemini_api_key)
    
    def _safe_format(self, value, format_spec=",.2f", default="N/A"):
        """Safely format a value that might be None"""
        if value is None:
            return default
        elif isinstance(value, (int, float)):
            return f"${value:{format_spec}}" if abs(value) >= 1 else f"{value:{format_spec}}"
        else:
            return str(value)
    
    def analyze(self, ticker: str, onchain_data: OnChainData) -> str:
        """Analyze enhanced on-chain data with clear indication of data source (real vs. simulated)"""
        if not onchain_data:
            return "Insufficient on-chain data for analysis"
        
        # Format CryptoQuant specific metrics
        exchange_reserve = self._safe_format(onchain_data.exchange_reserve)
        exchange_netflow = self._safe_format(onchain_data.exchange_netflow)
        exchange_inflow = self._safe_format(onchain_data.exchange_inflow)
        exchange_outflow = self._safe_format(onchain_data.exchange_outflow)
        transaction_count = f"{onchain_data.transaction_count:,}" if onchain_data.transaction_count else "N/A"
        address_count = f"{onchain_data.address_count:,}" if onchain_data.address_count else "N/A"
        in_house_flow = self._safe_format(onchain_data.in_house_flow)
        
        # Format flow indicators
        mpi_str = self._safe_format(onchain_data.mpi, format_spec=".2f", default="N/A")
        exchange_shutdown_index = self._safe_format(onchain_data.exchange_shutdown_index, format_spec=".4f")
        exchange_whale_ratio = self._safe_format(onchain_data.exchange_whale_ratio, format_spec=".4f")
        fund_flow_ratio = self._safe_format(onchain_data.fund_flow_ratio, format_spec=".4f")
        stablecoins_ratio = self._safe_format(onchain_data.stablecoins_ratio, format_spec=".4f")
        sopr = self._safe_format(onchain_data.sopr, format_spec=".4f")
        
        # Format distribution data
        inflow_age_distribution = "N/A"
        if onchain_data.inflow_age_distribution:
            age_items = []
            for age, value in onchain_data.inflow_age_distribution.items():
                age_items.append(f"  - {age}: {self._safe_format(value)}")
            inflow_age_distribution = "\n".join(age_items) if age_items else "N/A"
        
        inflow_supply_distribution = "N/A"
        if onchain_data.inflow_supply_distribution:
            supply_items = []
            for supply_range, value in onchain_data.inflow_supply_distribution.items():
                supply_items.append(f"  - {supply_range}: {self._safe_format(value)}")
            inflow_supply_distribution = "\n".join(supply_items) if supply_items else "N/A"
        
        # Format large transactions
        large_txs = []
        for tx in onchain_data.large_transactions:
            value = tx.get('value', 0)
            tx_type = tx.get('type', 'unknown')
            timestamp = tx.get('timestamp', datetime.datetime.now())
            large_txs.append(f"- ${value:.2f} {tx_type} at {timestamp.strftime('%Y-%m-%d %H:%M')}")
        
        large_txs_str = "\n".join(large_txs) if large_txs else "None detected"
        
        # Format whale wallet changes
        whale_changes = []
        for wallet, change in onchain_data.whale_wallet_changes.items():
            if wallet and change is not None:
                direction = "accumulated" if change > 0 else "distributed"
                # Make sure wallet is long enough to slice
                if len(wallet) > 14:
                    wallet_display = f"{wallet[:8]}...{wallet[-6:]}"
                else:
                    wallet_display = wallet
                whale_changes.append(f"- Wallet {wallet_display}: {direction} ${abs(change):.2f}")
        
        whale_changes_str = "\n".join(whale_changes) if whale_changes else "No significant movements"
        
        # Safely format other values
        active_addresses = f"{onchain_data.active_addresses_24h:,}" if onchain_data.active_addresses_24h is not None else "N/A"
        tx_volume = self._safe_format(onchain_data.transaction_volume_24h)
        avg_tx_value = self._safe_format(onchain_data.avg_transaction_value)
        
        # Determine if this is real or simulated data
        data_source = "SIMULATED" if onchain_data.is_simulated else "REAL"
        
        # Add a disclaimer about data reliability based on the source
        data_reliability_note = ""
        if onchain_data.is_simulated:
            data_reliability_note = f"""
            IMPORTANT NOTE: The on-chain data for {ticker} is simulated since CryptoQuant doesn't directly support this cryptocurrency.
            The values are based on realistic simulations but should be considered approximations rather than actual blockchain data.
            When analyzing, consider this limitation and give more weight to market and news analysis for {ticker}.
            """
        else:
            data_reliability_note = f"""
            The on-chain data for {ticker} is based on real blockchain data from CryptoQuant.
            This provides high-confidence insights into actual blockchain activity.
            """
        
        # Create enhanced prompt for on-chain analysis with clear data source indication
        prompt = f"""
        You are an expert cryptocurrency on-chain analyst. Analyze this {data_source} on-chain data for {ticker} on the {onchain_data.network} network:
        
        {data_reliability_note}
        
        Exchange Metrics:
        - Exchange Reserve: {exchange_reserve}
        - Exchange Net Flow: {exchange_netflow}
        - Exchange Inflow: {exchange_inflow}
        - Exchange Outflow: {exchange_outflow}
        - Transaction Count: {transaction_count}
        - Address Count: {address_count}
        - In-House Flow: {in_house_flow}
        
        Flow Indicators:
        - Miner Position Index (MPI): {mpi_str} {"(BTC only)" if ticker != "BTC" else ""}
        - Exchange Shutdown Index: {exchange_shutdown_index}
        - Exchange Whale Ratio: {exchange_whale_ratio}
        - Fund Flow Ratio: {fund_flow_ratio}
        - Stablecoins Ratio: {stablecoins_ratio}
        - SOPR (Spent Output Profit Ratio): {sopr}
        
        Inflow Age Distribution:
        {inflow_age_distribution}
        
        Inflow Supply Distribution:
        {inflow_supply_distribution}
        
        Transaction Metrics:
        - Active addresses (24h): {active_addresses}
        - Transaction volume (24h): {tx_volume}
        - Average transaction value: {avg_tx_value}
        
        Large Transactions:
        {large_txs_str}
        
        Whale Wallet Activity:
        {whale_changes_str}
        
        Based on this {'simulated' if onchain_data.is_simulated else 'real'} on-chain data:
        1. What does the overall on-chain activity suggest about {ticker}?
        2. Analyze the exchange flows and their implications for price
        3. Interpret key indicators (Exchange Whale Ratio, SOPR, etc.)
        4. What conclusions can be drawn from whale wallet activity?
        5. Identify any divergences between on-chain metrics and market price
        6. Are there any warning signs or bullish signals in the data?
        7. What might this on-chain data suggest for short and medium-term price action?
        
        Provide a detailed on-chain analysis with actionable trading insights. Be specific about what these metrics indicate for investor sentiment and potential price movements.
        {'Since this is simulated data, focus on the relationships between different metrics rather than absolute values.' if onchain_data.is_simulated else ''}
        """
        
        try:
            # Create a chat and send the prompt
            chat = self.client.chats.create(model=self.config.model_name)
            response = chat.send_message(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating on-chain analysis: {e}")
            return f"Error generating on-chain analysis: {str(e)}"

# Enhanced Position Manager with improved tracking
class PositionManager:
    """Manages open positions and handles take profit/stop loss conditions"""
    
    def __init__(self, config: Config):
        self.config = config
        self.open_positions: Dict[str, List[Position]] = {}
        self.closed_positions: List[Position] = []
        self.last_check_time = time.time()
        self.max_positions_per_ticker = 1  # IMPORTANT: Limit to 1 position per ticker
        
        # Create position data directory
        os.makedirs("data/positions", exist_ok=True)
        
        # Load existing positions
        self._load_positions()
    
    def _load_positions(self):
        """Load positions from disk"""
        try:
            if os.path.exists("data/positions/open_positions.pkl"):
                with open("data/positions/open_positions.pkl", "rb") as f:
                    self.open_positions = pickle.load(f)
                    logger.info(f"Loaded open positions for {len(self.open_positions)} tickers")
            
            if os.path.exists("data/positions/closed_positions.pkl"):
                with open("data/positions/closed_positions.pkl", "rb") as f:
                    self.closed_positions = pickle.load(f)
                    logger.info(f"Loaded {len(self.closed_positions)} closed positions")
        except Exception as e:
            logger.error(f"Error loading positions: {e}")
            self.open_positions = {}
            self.closed_positions = []
    
    def _save_positions(self):
        """Save positions to disk"""
        try:
            with open("data/positions/open_positions.pkl", "wb") as f:
                pickle.dump(self.open_positions, f)
            
            with open("data/positions/closed_positions.pkl", "wb") as f:
                pickle.dump(self.closed_positions, f)
            
            logger.info("Saved positions")
        except Exception as e:
            logger.error(f"Error saving positions: {e}")
    
    def get_position_count(self, ticker: str) -> int:
        """Get number of open positions for a ticker"""
        return len(self.open_positions.get(ticker, []))
    
    def add_position(self, signal: TradeSignal) -> Optional[Position]:
        """Add a new position from a trade signal"""
        # Only add buy or sell signals, not exit or hold
        if signal.action not in ["buy", "sell"]:
            logger.warning(f"Cannot add position for action {signal.action}")
            return None
        
        # Check if we already have the maximum positions for this ticker
        if self.get_position_count(signal.ticker) >= self.max_positions_per_ticker:
            logger.warning(f"Maximum positions ({self.max_positions_per_ticker}) already open for {signal.ticker}, skipping")
            return None
        
        # Calculate appropriate TP/SL based on asset price and action
        # Use different ranges for different price levels
        tp_distance = self._calculate_tp_distance(signal.ticker, signal.price)
        sl_distance = self._calculate_sl_distance(signal.ticker, signal.price)
        
        # Set TP/SL based on action (buy or sell)
        if signal.action == "buy":
            sl = signal.price - sl_distance
            tp = signal.price + tp_distance
        else:  # sell
            sl = signal.price + sl_distance
            tp = signal.price - tp_distance
        
        position = Position(
            ticker=signal.ticker,
            action=signal.action,
            entry_price=signal.price,
            entry_time=signal.time,
            size=signal.size or 10.0,  # Default to 10% if not specified
            sl=sl,
            tp=tp
        )
        
        # Add to open positions
        if signal.ticker not in self.open_positions:
            self.open_positions[signal.ticker] = []
        
        self.open_positions[signal.ticker].append(position)
        logger.info(f"Added new {signal.action} position for {signal.ticker} at {signal.price} with TP: {position.tp}, SL: {position.sl}")
        
        # Save positions
        self._save_positions()
        
        return position
    
    def _calculate_tp_distance(self, ticker: str, price: float) -> float:
        """Calculate take profit distance based on ticker and price"""
        # Use different distances based on ticker and price levels
        ticker = ticker.upper()
        
        if ticker == "BTC":
            return 200  # $200 for BTC
        elif ticker == "ETH":
            return 100   # $100 for ETH
        elif ticker == "SOL":
            return 2     # $3 for SOL
        else:
            # For other assets, use percentage-based approach
            if price > 10000:
                return price * 0.02  # 2% for high-priced assets
            elif price > 1000:
                return price * 0.03  # 3% for medium-priced assets
            elif price > 100:
                return price * 0.05  # 5% for lower-priced assets
            elif price > 10:
                return price * 0.07  # 7% for low-priced assets
            elif price > 1:
                return price * 0.10  # 10% for very low-priced assets
            else:
                return price * 0.15  # 15% for micro-priced assets
    
    def _calculate_sl_distance(self, ticker: str, price: float) -> float:
        """Calculate stop loss distance based on ticker and price"""
        # Use the same distances as TP for now
        # This could be adjusted based on ticker volatility
        return self._calculate_tp_distance(ticker, price)
    
    def close_position(self, position: Position, current_price: float, reason: str) -> Position:
        """Close an open position"""
        # Update position
        position.exit_price = current_price
        position.exit_time = datetime.datetime.now()
        position.status = reason
        
        # Remove from open positions
        if position.ticker in self.open_positions:
            self.open_positions[position.ticker] = [p for p in self.open_positions[position.ticker] if p != position]
            
            # Remove ticker key if no more positions
            if not self.open_positions[position.ticker]:
                del self.open_positions[position.ticker]
        
        # Add to closed positions
        self.closed_positions.append(position)
        
        logger.info(f"Closed {position.action} position for {position.ticker}: {reason}")
        
        # Save positions
        self._save_positions()
        
        return position
    
    def check_positions(self, ticker: str, current_price: float) -> List[TradeSignal]:
        """Check open positions for take profit/stop loss conditions"""
        if ticker not in self.open_positions:
            return []
        
        exit_signals = []
        
        # Check each position for this ticker
        for position in self.open_positions.get(ticker, [])[:]:  # Create a copy to avoid modifying during iteration
            if position.is_take_profit_hit(current_price):
                # Close position at take profit
                self.close_position(position, current_price, "closed_tp")
                exit_signal = position.to_exit_signal(current_price, "tp_hit")
                exit_signals.append(exit_signal)
                logger.info(f"Take profit hit for {ticker} at {current_price}")
                
            elif position.is_stop_loss_hit(current_price):
                # Close position at stop loss
                self.close_position(position, current_price, "closed_sl")
                exit_signal = position.to_exit_signal(current_price, "sl_hit")
                exit_signals.append(exit_signal)
                logger.info(f"Stop loss hit for {ticker} at {current_price}")
        
        return exit_signals
    
    def get_positions_for_ticker(self, ticker: str) -> List[Position]:
        """Get all open positions for a ticker"""
        return self.open_positions.get(ticker, [])
    
    def has_open_positions(self, ticker: str) -> bool:
        """Check if there are open positions for a ticker"""
        return ticker in self.open_positions and len(self.open_positions[ticker]) > 0
    
    def should_check_positions(self) -> bool:
        """Check if it's time to check positions"""
        current_time = time.time()
        elapsed = current_time - self.last_check_time
        
        if elapsed >= self.config.position_check_interval:
            self.last_check_time = current_time
            return True
        
        return False
    
    def clear_positions(self):
        """Clear all positions - useful for testing"""
        self.open_positions = {}
        self.closed_positions = []
        self._save_positions()
        logger.info("All positions cleared for testing")

# Reflection Agent
class ReflectionAgent:
    """Agent for evaluating past performance and learning"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = genai.Client(api_key=config.gemini_api_key)
        self.completed_trades = []
        self.signals_history = {}
        self.last_reflection_time = time.time()
        
        # Create directories for storing reflection data
        os.makedirs("data/performance", exist_ok=True)
        
        # Load any existing trade history
        self._load_history()
    
    def _load_history(self):
        """Load trade history from disk"""
        try:
            if os.path.exists("data/performance/completed_trades.pkl"):
                with open("data/performance/completed_trades.pkl", "rb") as f:
                    self.completed_trades = pickle.load(f)
                    logger.info(f"Loaded {len(self.completed_trades)} completed trades")
            
            if os.path.exists("data/performance/signals_history.pkl"):
                with open("data/performance/signals_history.pkl", "rb") as f:
                    self.signals_history = pickle.load(f)
                    logger.info(f"Loaded signals history for {len(self.signals_history)} tickers")
        except Exception as e:
            logger.error(f"Error loading trade history: {e}")
    
    def _save_history(self):
        """Save trade history to disk"""
        try:
            with open("data/performance/completed_trades.pkl", "wb") as f:
                pickle.dump(self.completed_trades, f)
            
            with open("data/performance/signals_history.pkl", "wb") as f:
                pickle.dump(self.signals_history, f)
            
            logger.info("Saved trade history")
        except Exception as e:
            logger.error(f"Error saving trade history: {e}")
    
    def add_signal(self, signal: TradeSignal):
        """Add a new signal to history"""
        if signal.ticker not in self.signals_history:
            self.signals_history[signal.ticker] = []
        
        # Add the signal
        self.signals_history[signal.ticker].append({
            "signal": signal,
            "timestamp": datetime.datetime.now(),
            "evaluated": False,
            "outcome": None
        })
        
        # Save history
        self._save_history()
    
    def add_trade_outcome(self, ticker: str, entry_time: datetime.datetime, 
                          exit_time: datetime.datetime, entry_price: float,
                          exit_price: float, action: str, size: float):
        """Add a completed trade outcome"""
        # Calculate profit/loss
        if action == "buy":
            profit = (exit_price - entry_price) / entry_price * size
        else:  # sell
            profit = (entry_price - exit_price) / entry_price * size
        
        # Create trade record
        trade = {
            "ticker": ticker,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "action": action,
            "size": size,
            "profit": profit,
            "profit_percentage": profit * 100 / size
        }
        
        # Add to completed trades
        self.completed_trades.append(trade)
        
        # Update signal history
        if ticker in self.signals_history:
            # Find the matching entry signal
            for signal_data in self.signals_history[ticker]:
                if not signal_data["evaluated"] and signal_data["signal"].action == action:
                    # Calculate time difference to make sure it's the right signal
                    time_diff = abs((signal_data["timestamp"] - entry_time).total_seconds())
                    if time_diff < 3600:  # Within an hour
                        signal_data["evaluated"] = True
                        signal_data["outcome"] = trade
                        break
        
        # Save history
        self._save_history()
        
        return trade
    
    def simulate_outcomes(self, ticker: str, current_price: float):
        """Simulate outcomes for signals that haven't been evaluated yet"""
        if ticker not in self.signals_history:
            return
        
        # Get current time
        now = datetime.datetime.now()
        
        # Check all signals for this ticker
        for signal_data in self.signals_history[ticker]:
            # Skip already evaluated signals
            if signal_data["evaluated"]:
                continue
            
            signal = signal_data["signal"]
            
            # Skip signals without tp/sl values
            if signal.tp is None or signal.sl is None:
                continue
                
            # Check if signal is at least 24 hours old
            age = (now - signal_data["timestamp"]).total_seconds() / 3600
            if age >= 24:
                # Simulate outcome
                if signal.action == "buy":
                    # For buy signals, check if price hit take profit or stop loss
                    if current_price >= signal.tp:
                        # Take profit hit
                        self.add_trade_outcome(
                            ticker=ticker,
                            entry_time=signal_data["timestamp"],
                            exit_time=now,
                            entry_price=signal.price,
                            exit_price=signal.tp,
                            action=signal.action,
                            size=signal.size if signal.size is not None else 10.0
                        )
                    elif current_price <= signal.sl:
                        # Stop loss hit
                        self.add_trade_outcome(
                            ticker=ticker,
                            entry_time=signal_data["timestamp"],
                            exit_time=now,
                            entry_price=signal.price,
                            exit_price=signal.sl,
                            action=signal.action,
                            size=signal.size if signal.size is not None else 10.0
                        )
                elif signal.action == "sell":
                    # For sell signals, check if price hit take profit or stop loss
                    if current_price <= signal.tp:
                        # Take profit hit
                        self.add_trade_outcome(
                            ticker=ticker,
                            entry_time=signal_data["timestamp"],
                            exit_time=now,
                            entry_price=signal.price,
                            exit_price=signal.tp,
                            action=signal.action,
                            size=signal.size if signal.size is not None else 10.0
                        )
                    elif current_price >= signal.sl:
                        # Stop loss hit
                        self.add_trade_outcome(
                            ticker=ticker,
                            entry_time=signal_data["timestamp"],
                            exit_time=now,
                            entry_price=signal.price,
                            exit_price=signal.sl,
                            action=signal.action,
                            size=signal.size if signal.size is not None else 10.0
                        )
    
    def analyze_performance(self, ticker: str = None) -> str:
        """Analyze trading performance overall or for a specific ticker"""
        # Filter trades if ticker is specified
        trades = self.completed_trades
        if ticker:
            trades = [t for t in trades if t["ticker"] == ticker]
        
        if not trades:
            return "Insufficient trade data for analysis"
        
        # Calculate performance metrics
        start_date = min(t["entry_time"] for t in trades)
        end_date = max(t["exit_time"] for t in trades)
        
        metrics = PerformanceMetrics(
            ticker=ticker if ticker else "ALL",
            start_date=start_date,
            end_date=end_date
        )
        
        metrics.calculate_metrics(trades)
        
        # Create an analysis prompt
        prompt = f"""
        You are a cryptocurrency trading performance analyst. Analyze these trading performance metrics:

        Time Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}
        Asset: {"All cryptocurrencies" if not ticker else ticker}

        Performance Metrics:
        - Total Trades: {metrics.total_trades}
        - Winning Trades: {metrics.winning_trades} ({f"{metrics.win_rate*100:.1f}%" if metrics.win_rate else "0%"})
        - Losing Trades: {metrics.losing_trades} ({f"{(1-metrics.win_rate)*100:.1f}%" if metrics.win_rate is not None else "0%"})
        - Total Profit/Loss: {metrics.profit_loss:.2f}%
        - Maximum Drawdown: {metrics.max_drawdown:.2f}%
        - Average Profit per Winning Trade: {metrics.avg_profit_per_trade:.2f}% (if available)
        - Average Loss per Losing Trade: {metrics.avg_loss_per_trade:.2f}% (if available)
        - Risk-Reward Ratio: {metrics.risk_reward_ratio:.2f if metrics.risk_reward_ratio else "N/A"}
        
        Trade Breakdown:
        {self._format_trades(trades[:5])}
        
        Based on this performance data:
        1. Evaluate the overall trading strategy effectiveness
        2. Identify strengths and weaknesses in the trading approach
        3. Analyze patterns in winning vs losing trades
        4. Suggest specific improvements to increase win rate and profit
        5. Identify any risk management issues
        6. Provide actionable recommendations for future trading
        
        Provide a comprehensive performance analysis with actionable insights.
        """
        
        try:
            # Create a chat and send the prompt
            chat = self.client.chats.create(model=self.config.model_name)
            response = chat.send_message(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating performance analysis: {e}")
            return f"Error generating performance analysis: {str(e)}"
    
    def _format_trades(self, trades: List[Dict[str, Any]]) -> str:
        """Format trades for the prompt"""
        result = []
        
        for i, trade in enumerate(trades):
            result.append(f"""
            {i+1}. {trade['ticker']} {trade['action'].upper()}
               Entry: ${trade['entry_price']:.2f} at {trade['entry_time'].strftime('%Y-%m-%d %H:%M')}
               Exit: ${trade['exit_price']:.2f} at {trade['exit_time'].strftime('%Y-%m-%d %H:%M')}
               Size: {trade['size']:.1f}%
               P/L: {trade['profit_percentage']:.2f}%
            """)
        
        return "\n".join(result)
    
    def should_run_reflection(self) -> bool:
        """Check if it's time to run reflection"""
        current_time = time.time()
        elapsed = current_time - self.last_reflection_time
        
        if elapsed >= self.config.performance_evaluation_interval:
            self.last_reflection_time = current_time
            return True
        
        return False
    
    def run_reflection(self):
        """Run the reflection analysis"""
        if self.should_run_reflection():
            logger.info("Running performance reflection...")
            print("Running performance reflection...")
            
            # Analyze overall performance
            overall_analysis = self.analyze_performance()
            logger.info(f"Performance reflection: {overall_analysis[:100]}...")
            print(f"Performance Analysis: {overall_analysis[:100]}...")

# Enhanced Trading Advisor with on-chain data integration
class TradingAdvisor:
    """Enhanced Trading Advisor using Gemini and on-chain data"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = genai.Client(api_key=config.gemini_api_key)
    
    def get_signal(self, ticker: str, price: float, market_analysis: str, 
                   news_analysis: str, onchain_analysis: str, has_open_position: bool) -> Optional[TradeSignal]:
        """Generate an advanced trading signal based on multiple analyses including enhanced on-chain data"""
        
        # If there's already an open position, return a hold signal
        if has_open_position:
            logger.info(f"Already have an open position for {ticker}, generating HOLD signal")
            return TradeSignal(
                ticker=ticker,
                action="hold",
                price=price,
                time=datetime.datetime.now(),
                rationale="Already have an open position for this ticker"
            )
        
        # Create a comprehensive prompt for trading decisions with enhanced on-chain data
        prompt = f"""
        You are an expert cryptocurrency trading advisor with deep knowledge of on-chain analytics. Based on the following comprehensive information for {ticker},
        decide whether to BUY, SELL, or HOLD.
        
        Current price: ${price:.2f}
        
        MARKET ANALYSIS:
        {market_analysis}
        
        NEWS ANALYSIS:
        {news_analysis}
        
        ON-CHAIN ANALYSIS (Enhanced with CryptoQuant data):
        {onchain_analysis}
        
        Considering all these factors - technical indicators, news sentiment, and comprehensive on-chain activity from CryptoQuant - provide your most informed trading decision.
        Pay special attention to the on-chain data, as it often provides leading indicators ahead of price movements.
        
        Your response must be in this exact format:
        
        DECISION: [BUY/SELL/HOLD]
        CONFIDENCE: [0.0 to 1.0]
        TIME_HORIZON: [SHORT/MEDIUM/LONG]
        RISK_LEVEL: [LOW/MEDIUM/HIGH]
        REASON: [Detailed explanation with key factors]
        STOP_LOSS: [Price level for stop loss]
        TAKE_PROFIT: [Price level for take profit]
        SIZE: [Position size as percentage, 1-10%]
        MARKET_SIGNAL: [Bullish/Bearish/Neutral]
        NEWS_SIGNAL: [Bullish/Bearish/Neutral]
        ONCHAIN_SIGNAL: [Bullish/Bearish/Neutral]
        
        Make your decision now.
        """
        
        try:
            # Create a chat and send the prompt
            chat = self.client.chats.create(model=self.config.model_name)
            response = chat.send_message(prompt)
            
            # Parse the response
            return self._parse_signal(response.text, ticker, price)
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return None
    
    def _parse_signal(self, response_text: str, ticker: str, current_price: float) -> Optional[TradeSignal]:
        """Parse the AI response into a trading signal with enhanced attributes"""
        lines = response_text.strip().split('\n')
        data = {}
        
        # Extract data from response
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                data[key.strip().upper()] = value.strip()
        
        # Get the decision
        decision = data.get('DECISION', 'HOLD').upper()
        
        # Convert decision to action
        action = decision.lower() if decision in ['BUY', 'SELL'] else 'hold'
        
        # Extract other parameters with default fallbacks
        try:
            confidence = float(data.get('CONFIDENCE', '0.8'))
        except:
            confidence = 0.8
            
        reason = data.get('REASON', 'No rationale provided')
        time_horizon = data.get('TIME_HORIZON', 'MEDIUM').lower()
        risk_level = data.get('RISK_LEVEL', 'MEDIUM').lower()
        
        # Source signals
        source_signals = {
            'market': data.get('MARKET_SIGNAL', 'Neutral'),
            'news': data.get('NEWS_SIGNAL', 'Neutral'),
            'onchain': data.get('ONCHAIN_SIGNAL', 'Neutral')
        }
        
        # Parse size
        try:
            size_str = data.get('SIZE', '10').replace('%', '')
            size = float(size_str)
            # Cap at 20%
            size = min(size, 20.0)
        except:
            size = 10.0
        
        # Parse TP/SL - these will be calculated by position manager if None
        tp = None
        sl = None
        
        # Try to extract TP/SL from the response (optional)
        try:
            tp_str = data.get('TAKE_PROFIT', '').replace('%', '').strip()
            if tp_str and tp_str != 'None':
                tp = float(tp_str)
        except:
            tp = None  # Will be calculated by position manager
            
        try:
            sl_str = data.get('STOP_LOSS', '').replace('%', '').strip()
        
            if sl_str and sl_str != 'None':
                sl = float(sl_str)
        except:
            sl = None  # Will be calculated by position manager
        
        # Create and return signal
        return TradeSignal(
            ticker=ticker,
            action=action,
            price=current_price,
            time=datetime.datetime.now(),
            confidence_score=confidence,
            size=size,
            sl=sl,  # May be None, will be calculated by position manager
            tp=tp,  # May be None, will be calculated by position manager
            rationale=reason,
            expected_holding_period=time_horizon,
            risk_assessment=risk_level,
            source_signals=source_signals
        )
        
        
        
class SignalDispatcher:
    """
    Enhanced system for reliable signal delivery with position tracking
    """
    def __init__(self):
        self.webhook_urls = DEFAULT_WEBHOOK_URLS
        self.ticker_map = DEFAULT_TICKER_MAP
        self.max_retries = 3
        self.backoff_factor = 2
        self.successful_formats = {}
        self.active_trades = {}  # To track active trades for each symbol
    
    def get_webhook_url(self, symbol):
        """Get the correct webhook URL for the given symbol"""
        if symbol in self.webhook_urls:
            return self.webhook_urls[symbol]
        return DEFAULT_WEBHOOK_URL
    
    def get_api_ticker(self, symbol):
        """Convert the trading symbol to the API-compatible ticker format"""
        # First check if we have a mapping
        if symbol in self.ticker_map:
            return self.ticker_map[symbol]
        
        # Check if we've successfully used a format before
        if symbol in self.successful_formats:
            return self.successful_formats[symbol]
        
        # Default to just the base currency
        return symbol
    
    def send_webhook(self, symbol, action, price, **kwargs):
        """Send a trading signal to the appropriate webhook with enhanced reliability"""
        # Skip if symbol/price is invalid
        if symbol is None or price is None or price <= 0:
            logger.error(f"Invalid symbol or price for webhook: {symbol}, {price}")
            return False
            
        # Check if there's already an active trade for this symbol
        if symbol in self.active_trades and action == "buy":
            logger.warning(f"Trade already active for {symbol}, skipping BUY signal")
            return False
            
        # Get the API-compatible ticker format
        api_ticker = self.get_api_ticker(symbol)
        
        # Get the correct webhook URL for this symbol
        webhook_url = self.get_webhook_url(symbol)
        
        if not webhook_url:
            logger.error(f"No webhook URL configured for {symbol}")
            return False
        
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format the payload based on action type
        if action == "buy":
            payload = {
                "ticker": api_ticker,
                "action": "buy",
                "price": str(price),
                "time": current_time,
                "duration": "11.2h",
                "confidence": str(kwargs.get("confidence", "0.7")),
                "regime": "Ranging"
            }
            message = "BUY"
        elif action == "sell":
            payload = {
                "ticker": api_ticker,
                "action": "sell",
                "price": str(price),
                "time": current_time,
                "duration": "11.2h",
                "confidence": str(kwargs.get("confidence", "0.7")),
                "regime": "Ranging"
            }
            message = "SELL"
        elif action == "exit_buy":
            # Use fixed values: 100 for TP, 0 for SL
            per_value = kwargs.get("per", 0)
            
            payload = {
                "ticker": api_ticker,
                "action": "exit_buy",
                "price": str(price),
                "time": current_time,
                "size": str(kwargs.get("size", "1")),
                "per": f"{per_value}%",  # Use the fixed value directly
                "sl": str(kwargs.get("sl", "")),
                "tp": str(kwargs.get("tp", ""))
            }
            
            # Add reason if available
            if "reason" in kwargs:
                payload["reason"] = kwargs["reason"]
                
            message = "EXIT BUY"
        elif action == "exit_sell":
            # Use fixed values: 100 for TP, 0 for SL
            per_value = kwargs.get("per", 0)
            
            payload = {
                "ticker": api_ticker,
                "action": "exit_sell",
                "price": str(price),
                "time": current_time,
                "size": str(kwargs.get("size", "1")),
                "per": f"{per_value}%",  # Use the fixed value directly
                "sl": str(kwargs.get("sl", "")),
                "tp": str(kwargs.get("tp", ""))
            }
            
            # Add reason if available
            if "reason" in kwargs:
                payload["reason"] = kwargs["reason"]
                
            message = "EXIT SELL"
        else:
            # Invalid action
            logger.error(f"Invalid action: {action}")
            return False
        
        logger.info(f"Webhook payload for {symbol}: {json.dumps(payload, indent=2)}")
        
        success = False
        attempted_formats = [api_ticker]
        
        # Try sending with exponential backoff
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Attempt {attempt+1} sending webhook for {symbol} ({api_ticker}): {message}")
                logger.info(f"Using webhook URL: {webhook_url}")
                logger.info(f"Payload: {json.dumps(payload, indent=2)}")
                
                # IMPORTANT: Use json=payload instead of data=webhook_message
                response = requests.post(
                    webhook_url,
                    json=payload,  # Send as JSON, not plain text
                    timeout=30
                )
                
                # Log the full response
                logger.info(f"Response status code: {response.status_code}")
                try:
                    logger.info(f"Response text: {response.text}")
                except:
                    logger.info("Could not log response text")
                
                if response.status_code == 200:
                    logger.info(f"Webhook sent successfully: {response.text}")
                    
                    # Save the successful format
                    if symbol not in self.successful_formats:
                        self.successful_formats[symbol] = api_ticker
                    
                    # Update active trade tracking
                    if action == "buy" or action == "sell":
                        self.register_trade(symbol, None, action, price, 
                                           kwargs.get("sl"), kwargs.get("tp"))
                    elif action == "exit_buy" or action == "exit_sell":
                        self.remove_trade(symbol)
                    
                    success = True
                    break
                else:
                    logger.error(f"Failed to send webhook. Status code: {response.status_code}, Response: {response.text}")
                    
                    # If we get a "no matching pair" error, try alternative formats
                    if "No matching bot pair found" in response.text:
                        # Try alternative formats based on ticker
                        if symbol == 'BTC':
                            alternative_formats = ['BTCUSDT', 'BTC-USD', 'XBTUSD', 'BTC']
                        elif symbol == 'SOL':
                            alternative_formats = ['SOLUSDT', 'SOL-USD', 'SOL']
                        else:
                            base = symbol.split('/')[0] if '/' in symbol else symbol
                            alternative_formats = [f"{base}USDT", f"{base}-USD", base]
                        
                        for alt_ticker in alternative_formats:
                            if alt_ticker in attempted_formats:
                                continue
                                
                            attempted_formats.append(alt_ticker)
                            logger.info(f"Trying alternative ticker format: {alt_ticker}")
                            
                            # Update the payload with the new ticker
                            payload["ticker"] = alt_ticker
                            
                            try:
                                alt_response = requests.post(
                                    webhook_url,
                                    json=payload,
                                    timeout=30
                                )
                                
                                if alt_response.status_code == 200:
                                    logger.info(f"Webhook sent successfully with alternative ticker {alt_ticker}")
                                    self.ticker_map[symbol] = alt_ticker
                                    self.successful_formats[symbol] = alt_ticker
                                    success = True
                                    break
                            except Exception as alt_e:
                                logger.error(f"Error trying alternative ticker {alt_ticker}: {alt_e}")
                        
                        if success:
                            break
            
            except Exception as e:
                logger.error(f"Error sending webhook (attempt {attempt+1}): {str(e)}")
            
            # Wait before retrying with exponential backoff
            if attempt < self.max_retries - 1:
                sleep_time = self.backoff_factor ** attempt
                logger.info(f"Waiting {sleep_time} seconds before retrying...")
                time.sleep(sleep_time)
        
        return success
    
    def register_trade(self, symbol, trade_id, action, entry_price, sl, tp, duration=None):
        """Register an active trade"""
        self.active_trades[symbol] = {
            'trade_id': trade_id,
            'action': action,
            'entry_price': entry_price,
            'sl': sl,
            'tp': tp,
            'entry_time': datetime.datetime.now(),
            'expected_duration': duration
        }
        logger.info(f"Registered active trade for {symbol}: {action} at {entry_price}")
    
    def get_active_trade(self, symbol):
        """Get information about an active trade"""
        return self.active_trades.get(symbol)
    
    def remove_trade(self, symbol):
        """Remove a trade after it's closed"""
        if symbol in self.active_trades:
            logger.info(f"Removed active trade for {symbol}")
            del self.active_trades[symbol]
    
    def has_active_trade(self, symbol):
        """Check if there's an active trade for the symbol"""
        return symbol in self.active_trades

# Enhanced CryptoMCP Class
class CryptoMCP:
    """Enhanced Multi-agent Crypto Trading System"""
    
    def __init__(self, config_file: str):
        print(f"Initializing Enhanced CryptoMCP with config file: {config_file}")
        # Load configuration
        self.config = Config.from_file(config_file)
        print(f"Loaded configuration for cryptocurrencies: {self.config.cryptocurrencies}")
        
        # Initialize providers
        self.market_provider = CoinGeckoProvider()
        self.news_provider = CryptoPanicProvider(self.config.cryptopanic_api_key)
        self.onchain_provider = OnChainDataProvider(self.config)
        
        # Initialize analysts and agents
        self.market_analyst = MarketAnalyst(self.config)
        self.news_analyst = NewsAnalyst(self.config)
        self.onchain_analyst = OnChainAnalyst(self.config)
        self.trading_advisor = TradingAdvisor(self.config)
        self.position_manager = PositionManager(self.config)
        self.reflection_agent = ReflectionAgent(self.config)
        
        # Initialize signal dispatcher
        self.dispatcher = SignalDispatcher()
        
        # Create necessary directories
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/market", exist_ok=True)
        os.makedirs("data/news", exist_ok=True)
        os.makedirs("data/signals", exist_ok=True)
        
        print("Initialization complete")
        
        # Debug information - log webhook configuration
        logger.info(f"Webhooks enabled with URLs: {DEFAULT_WEBHOOK_URLS}")
        logger.info(f"Default webhook URL: {DEFAULT_WEBHOOK_URL}")

        # Clear existing positions to start fresh - comment this out if you want to keep existing positions
        self.position_manager.clear_positions()
        print("All existing positions cleared for testing")
    
    def process_ticker(self, ticker: str):
        """Process a single cryptocurrency ticker with enhanced analysis"""
        logger.info(f"Processing {ticker}")
        print(f"Processing {ticker}...")
        
        try:
            # 1. Get market data with technical indicators
            print(f"Fetching market data for {ticker}...")
            market_data = self.market_provider.get_market_data(ticker)
            
            if market_data.price <= 0:
                logger.warning(f"Invalid price data for {ticker}, skipping")
                print(f"Invalid price data for {ticker}, skipping")
                return
            
            print(f"Current price for {ticker}: ${market_data.price:.2f}")
            
            # 2. Check if we already have an open position for this ticker
            has_open_position = self.position_manager.has_open_positions(ticker)
            
            # Also check if there's an active trade in the dispatcher
            has_active_trade = self.dispatcher.has_active_trade(ticker)
            
            if has_open_position:
                print(f"We already have an open position for {ticker}")
            
            if has_active_trade:
                print(f"We already have an active trade for {ticker}")
            
            # Combine both checks
            has_position = has_open_position or has_active_trade
            
            # 3. Check open positions for take profit / stop loss
            exit_signals = self.position_manager.check_positions(ticker, market_data.price)
            
            # Send exit signals to webhook
            for signal in exit_signals:
                print(f"Position closed: {signal.action.upper()} {ticker} at ${signal.price:.2f}")
                logger.info(f"Position closed: {signal.action.upper()} {ticker} at ${signal.price:.2f}")
                
                # Use the new webhook sending method
                self.dispatcher.send_webhook(
                    symbol=ticker,
                    action=signal.action,
                    price=signal.price,
                    size=signal.size,
                    per=signal.per,
                    sl=signal.sl,
                    tp=signal.tp,
                    reason=signal.reason
                )
            
            # 4. Update reflection agent with current price for simulating outcomes
            self.reflection_agent.simulate_outcomes(ticker, market_data.price)
            
            # 5. Get historical data
            historical_data = self.market_provider.get_historical_data(ticker, days=self.config.lookback_days)
            
            # 6. Get news
            print(f"Fetching news for {ticker}...")
            news_items = self.news_provider.get_news(ticker, limit=10)
            
            # 7. Get on-chain data
            print(f"Fetching on-chain data for {ticker}...")
            onchain_data = self.onchain_provider.get_onchain_data(ticker)
            
            # 8. Analyze market data
            print(f"Analyzing market data for {ticker}...")
            market_analysis = self.market_analyst.analyze(ticker, market_data, historical_data)
            print(f"Market Analysis: {market_analysis[:100]}...")
            
            # 9. Analyze news
            print(f"Analyzing news for {ticker}...")
            news_analysis = self.news_analyst.analyze(ticker, news_items)
            print(f"News Analysis: {news_analysis[:100]}...")
            
            # 10. Analyze on-chain data
            print(f"Analyzing on-chain data for {ticker}...")
            try:
                onchain_analysis = self.onchain_analyst.analyze(ticker, onchain_data)
                print(f"On-chain Analysis: {onchain_analysis[:100]}...")
            except Exception as e:
                logger.error(f"Error analyzing on-chain data for {ticker}: {e}")
                print(f"Error analyzing on-chain data for {ticker}: {e}")
                # Set a default value to continue processing
                onchain_analysis = f"Error analyzing on-chain data for {ticker}: {e}"
            
            # 11. Get trading signal - now passing whether we have an open position
            print(f"Generating trading signal for {ticker}...")
            signal = self.trading_advisor.get_signal(
                ticker=ticker,
                price=market_data.price,
                market_analysis=market_analysis,
                news_analysis=news_analysis,
                onchain_analysis=onchain_analysis,
                has_open_position=has_position  # Pass if we already have a position
            )
            
            # 12. Process signal
            if signal:
                if signal.action in ["buy", "sell"]:
                    print(f"Generated signal: {signal.action.upper()} {ticker} at ${signal.price:.2f}")
                    logger.info(f"Generated signal: {signal.action.upper()} {ticker} at ${signal.price:.2f}")
                    
                    # Add position
                    position = self.position_manager.add_position(signal)
                    if position:
                        logger.info(f"Added {signal.action} position for {ticker} with TP: {position.tp}, SL: {position.sl}")
                    
                    # Add signal to reflection agent
                    self.reflection_agent.add_signal(signal)
                    
                    # Send webhook 
                    print(f"Sending webhook for {ticker} {signal.action.upper()}...")
                    success = self.dispatcher.send_webhook(
                        symbol=ticker,
                        action=signal.action,
                        price=signal.price,
                        confidence=f"{signal.confidence_score:.2f}"  # Format confidence score
                    )
                    if success:
                        print(f"Successfully sent webhook for {ticker} {signal.action.upper()}!")
                    else:
                        print(f"Failed to send webhook for {ticker} {signal.action.upper()}!")
                else:
                    print(f"No actionable trading signal for {ticker} (HOLD recommendation)")
                    logger.info(f"HOLD recommendation for {ticker}")
                    
            else:
                print(f"No trading signal for {ticker}")
                logger.info(f"No trading signal for {ticker}")
        
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            print(f"Error processing {ticker}: {e}")
    
    def check_all_positions(self):
        """Check positions for all cryptocurrencies"""
        for ticker in self.config.cryptocurrencies:
            try:
                # Only fetch market data if there are positions to check
                if self.position_manager.has_open_positions(ticker):
                    market_data = self.market_provider.get_market_data(ticker)
                    
                    if market_data.price <= 0:
                        logger.warning(f"Invalid price data for {ticker}, skipping position check")
                        continue
                    
                    # Check positions
                    exit_signals = self.position_manager.check_positions(ticker, market_data.price)
                    
                    # Send exit signals to webhook immediately
                    for signal in exit_signals:
                        print(f"Position closed: {signal.action.upper()} {ticker} at ${signal.price:.2f}")
                        logger.info(f"Position closed: {signal.action.upper()} {ticker} at ${signal.price:.2f}")
                        
                        # Direct webhook sending
                        self.dispatcher.send_webhook(
                            symbol=ticker,
                            action=signal.action,
                            price=signal.price,
                            size=signal.size,
                            per=signal.per,
                            sl=signal.sl,
                            tp=signal.tp,
                            reason=signal.reason
                        )
            except Exception as e:
                logger.error(f"Error checking positions for {ticker}: {e}")
    
    def run_once(self):
        """Run one cycle of the enhanced system"""
        logger.info("Running Enhanced CryptoMCP cycle")
        print("Running Enhanced CryptoMCP cycle...")
        
        for ticker in self.config.cryptocurrencies:
            self.process_ticker(ticker)
        
        # Run reflection if needed
        self.reflection_agent.run_reflection()
        
        logger.info("Cycle completed")
        print("Cycle completed")
    
    def run(self):
        """Run the system continuously"""
        logger.info("Starting Enhanced CryptoMCP system")
        print("Starting Enhanced CryptoMCP system")
        
        while True:
            try:
                self.run_once()
                
                interval = self.config.data_fetch_interval
                logger.info(f"Sleeping for {interval} seconds")
                print(f"Sleeping for {interval} seconds")
                
                # Sleep in smaller increments so we can check positions frequently
                last_position_check = time.time()
                last_normal_cycle = time.time()
                
                while (time.time() - last_normal_cycle) < interval:
                    # Check if it's time to check positions
                    if (time.time() - last_position_check) >= self.config.position_check_interval:
                        print("Checking positions during sleep cycle...")
                        self.check_all_positions()
                        last_position_check = time.time()
                    
                    # Sleep a short time
                    time.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("Shutting down Enhanced CryptoMCP system")
                print("Shutting down Enhanced CryptoMCP system")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                print(f"Unexpected error: {e}")
                # Sleep for a shorter time before retrying
                time.sleep(60)

# Main entry point
if __name__ == "__main__":
    # Run the system
    try:
        print("Starting Enhanced CryptoMCP system...")
        # Use the default config file path
        config_file = "C:\\path\\alex\\ai-agent\\mcp\\config_template.json"
        
        # Test if the config file exists
        if not os.path.exists(config_file):
            print(f"Config file {config_file} not found. Creating a default config.")
            # Create a default config file
            with open(config_file, "w") as f:
                json.dump({
                    "gemini_api_key": "AIzaSyDK0JbwANnhWfqK2HTkHNvRvjD3mBVY6ew",
                    "webhook_urls": DEFAULT_WEBHOOK_URLS,
                    "default_webhook_url": DEFAULT_WEBHOOK_URL,
                    "cryptocurrencies": ["BTC", "SOL"],
                    "cryptopanic_api_key": "9b84be3aa755877273c581025badf243eef7f19f",
                    "cryptoquant_api_key": "av44iQ476js5H9wrjRPuorzIDZd25FyIPO9Imdgo",
                    "web3_providers": {
                        "ethereum": "https://eth-mainnet.g.alchemy.com/v2/demo",
                        "bsc": "https://bsc-dataseed.binance.org/",
                        "polygon": "https://polygon-rpc.com"
                    },
                    "data_fetch_interval": 3600,
                    "model_name": "gemini-2.0-flash",
                    "webhook_enabled": True,
                    "track_whale_wallets": True,
                    "technical_indicators": ["rsi", "macd", "bollinger", "sma", "ema"],
                    "lookback_days": 30,
                    "backtest_enabled": True,
                    "whale_threshold": 1000000,
                    "performance_evaluation_interval": 86400,
                    "check_tp_sl_interval": 10,
                    "position_check_interval": 10
                }, f)
            print(f"Created default config file at {config_file}")
            
        # Create and run the system
        mcp = CryptoMCP(config_file)
        mcp.run()
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()