"""
Data Manager for fetching and processing financial data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import pickle

class DataManager:
    """
    Manages financial data downloading, caching, and preprocessing
    """
    
    def __init__(self, cache_dir="data/cache"):
        """
        Initialize DataManager
        
        Args:
            cache_dir (str): Directory for caching data
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_stock_data(self, symbol, period="5y", force_refresh=False):
        """
        Get stock data from Yahoo Finance with caching
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            period (str): Time period ('5y', '2y', '1y', etc.)
            force_refresh (bool): Force refresh cached data
            
        Returns:
            pd.DataFrame: Stock data with OHLCV columns
        """
        cache_file = os.path.join(self.cache_dir, f"{symbol}_{period}.pkl")
        
        # Check if cached data exists and is recent (less than 1 day old)
        if not force_refresh and os.path.exists(cache_file):
            try:
                modified_time = os.path.getmtime(cache_file)
                if (datetime.now().timestamp() - modified_time) < 86400:  # 24 hours
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
            except:
                pass
        
        try:
            # Download data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Cache the data
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            return data
            
        except Exception as e:
            raise Exception(f"Error fetching data for {symbol}: {e}")
    
    def calculate_returns(self, data, return_type='log'):
        """
        Calculate returns from price data
        
        Args:
            data (pd.DataFrame): Price data
            return_type (str): 'log' or 'simple'
            
        Returns:
            pd.Series: Returns
        """
        prices = data['Close']
        
        if return_type == 'log':
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()
        
        return returns.dropna()
    
    def calculate_volatility(self, returns, window=252):
        """
        Calculate historical volatility
        
        Args:
            returns (pd.Series): Return series
            window (int): Rolling window size
            
        Returns:
            float: Annualized volatility
        """
        return returns.std() * np.sqrt(window)
    
    def get_risk_free_rate(self):
        """
        Get risk-free rate (using 10-year Treasury as proxy)
        
        Returns:
            float: Risk-free rate
        """
        try:
            treasury = yf.Ticker("^TNX")
            data = treasury.history(period="1mo")
            rate = data['Close'].iloc[-1] / 100  # Convert percentage to decimal
            return rate
        except:
            return 0.02  # Default 2% if unable to fetch
    
    def get_market_data_summary(self, symbol):
        """
        Get comprehensive market data summary
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Market data summary
        """
        data = self.get_stock_data(symbol)
        returns = self.calculate_returns(data)
        
        current_price = data['Close'].iloc[-1]
        volatility = self.calculate_volatility(returns)
        risk_free_rate = self.get_risk_free_rate()
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'volatility': volatility,
            'risk_free_rate': risk_free_rate,
            'data': data,
            'returns': returns,
            'start_date': data.index[0].strftime('%Y-%m-%d'),
            'end_date': data.index[-1].strftime('%Y-%m-%d'),
            'num_observations': len(data)
        }
    
    def get_option_chain(self, symbol):
        """
        Get option chain data (if available)
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Option chain data
        """
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            
            if not expirations:
                return None
            
            # Get first expiration date
            option_chain = ticker.option_chain(expirations[0])
            
            return {
                'calls': option_chain.calls,
                'puts': option_chain.puts,
                'expiration': expirations[0]
            }
        except:
            return None