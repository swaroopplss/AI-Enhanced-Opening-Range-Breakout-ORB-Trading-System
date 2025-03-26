import os
import pandas as pd
import numpy as np
from polygon import RESTClient
from datetime import datetime, timedelta
import pytz
import logging

logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self, api_key):
        self.client = RESTClient(api_key)
        self.eastern = pytz.timezone('US/Eastern')
    
    def fetch_historical_data(self, symbol, start_date, end_date):
        """Fetch 1-minute data for a symbol over a date range"""
        logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
        
        aggs = []
        for a in self.client.list_aggs(
            symbol,
            1,
            "minute",
            start_date,
            end_date,
            limit=50000,
        ):
            aggs.append(a)
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'timestamp': agg.timestamp,
            'datetime': pd.to_datetime(agg.timestamp, unit='ms'),
            'open': agg.open,
            'high': agg.high,
            'low': agg.low,
            'close': agg.close,
            'volume': agg.volume,
            'vwap': agg.vwap,
            'transactions': getattr(agg, 'transactions', None)
        } for agg in aggs])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Add date and time columns
        df['date'] = df['datetime'].dt.date
        df['time'] = df['datetime'].dt.time
        
        return df
    
    def identify_trading_days(self, df):
        """Group data by trading days"""
        # Filter to regular trading hours
        market_hours = df[
            (df['datetime'].dt.hour >= 9) & 
            ((df['datetime'].dt.hour < 16) | 
             ((df['datetime'].dt.hour == 16) & (df['datetime'].dt.minute == 0)))
        ]
        
        # Get list of unique trading days
        trading_days = market_hours['date'].unique()
        
        return trading_days, market_hours
    
    def identify_opening_ranges(self, df):
        """Identify 5-minute opening ranges for each trading day"""
        # Group by date
        grouped = df.groupby('date')
        
        # For each day, calculate opening range
        opening_ranges = []
        
        for date, day_df in grouped:
            # Get market open time (9:30 AM)
            market_open = day_df[
                (day_df['datetime'].dt.hour == 9) & 
                (day_df['datetime'].dt.minute >= 30)
            ]
            
            if len(market_open) == 0:
                continue
                
            # Get first 5 minutes
            first_5min = market_open.iloc[:5]
            
            if len(first_5min) < 5:
                continue
                
            # Calculate OR high and low
            or_high = first_5min['high'].max()
            or_low = first_5min['low'].min()
            
            # Mark data after opening range
            day_df = day_df.copy()
            day_df['or_high'] = or_high
            day_df['or_low'] = or_low
            day_df['or_range'] = or_high - or_low
            
            # Calculate minutes from market open
            first_candle = market_open.iloc[0]['datetime']
            day_df['minutes_from_open'] = (day_df['datetime'] - first_candle).dt.total_seconds() / 60
            
            # Mark if this is within opening range
            day_df['is_opening_range'] = day_df['minutes_from_open'] < 5
            
            # Calculate targets
            day_df['or_upper_target'] = or_high + (or_high - or_low) * 0.5  # 50% extension
            day_df['or_lower_target'] = or_low - (or_high - or_low) * 0.5  # 50% extension
            
            opening_ranges.append(day_df)
        
        if opening_ranges:
            return pd.concat(opening_ranges)
        else:
            return df

    def add_enhanced_features(self, df):
        """Add enhanced features for AI analysis"""
        import ta
        
        # Basic features
        df['candle_size'] = (df['high'] - df['low']) / df['low'] * 100
        df['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
        df['is_bullish'] = df['close'] > df['open']
        
        # Distance from opening range
        if 'or_high' in df.columns and 'or_low' in df.columns:
            df['dist_from_or_high'] = (df['close'] - df['or_high']) / df['or_high'] * 100
            df['dist_from_or_low'] = (df['close'] - df['or_low']) / df['or_low'] * 100
        
        # Volume indicators
        df['volume_ma10'] = df['volume'].rolling(window=10).mean()
        df['rel_volume'] = df['volume'] / df['volume_ma10'].shift(1)
        
        # Price indicators
        df['ema9'] = ta.trend.ema_indicator(df['close'], window=9)
        df['ema20'] = ta.trend.ema_indicator(df['close'], window=20)
        df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
        
        # Momentum indicators
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['macd'] = ta.trend.macd(df['close'])
        df['macd_signal'] = ta.trend.macd_signal(df['close'])
        df['macd_diff'] = ta.trend.macd_diff(df['close'])
        
        # Volatility indicator
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        
        # Trend indicators
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
        
        # Time-based features
        df['minutes_in_session'] = (df['datetime'] - df['datetime'].dt.normalize()).dt.total_seconds() / 60 - 9.5 * 60
        
        return df