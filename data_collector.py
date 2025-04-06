import os
import pandas as pd
import numpy as np
from polygon import RESTClient
from datetime import datetime, timedelta
import pytz
import logging
import time

logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self, api_key):
        self.client = RESTClient(api_key)
        self.eastern = pytz.timezone('US/Eastern')
    
    def fetch_historical_data(self, symbol, start_date, end_date):
        """Fetch 1-minute data for a symbol over a date range with rate limiting"""
        logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
        
        # Convert dates to datetime objects if they're strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).date()
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).date()
        
        # Check if cached data exists
        cache_file = f"{symbol}_data_cache_{start_date}_{end_date}.csv"
        if os.path.exists(cache_file):
            logger.info(f"Loading cached data from {cache_file}")
            df = pd.read_csv(cache_file, parse_dates=['datetime'])
            return df
        
        # Break the request into smaller chunks (30-day periods)
        chunk_size = timedelta(days=30)
        current_start = start_date
        all_aggs = []
        
        while current_start <= end_date:
            # Calculate end of chunk (either 30 days later or end_date, whichever comes first)
            current_end = min(current_start + chunk_size, end_date)
            
            logger.info(f"Fetching chunk: {current_start} to {current_end}")
            
            try:
                # Fetch this chunk
                chunk_aggs = []
                for a in self.client.list_aggs(
                    symbol,
                    1,
                    "minute",
                    current_start,
                    current_end,
                    limit=50000,
                ):
                    chunk_aggs.append(a)
                
                all_aggs.extend(chunk_aggs)
                logger.info(f"Successfully fetched {len(chunk_aggs)} candles")
                
                # Sleep between chunks to avoid rate limiting
                time.sleep(12)  # Wait 12 seconds between chunks
                
            except Exception as e:
                logger.error(f"Error fetching chunk {current_start} to {current_end}: {str(e)}")
                # If we hit a rate limit, wait longer and retry
                if "429" in str(e):
                    retry_wait = 60  # Wait 60 seconds before retrying
                    logger.info(f"Rate limit hit. Waiting {retry_wait} seconds before retrying...")
                    time.sleep(retry_wait)
                    continue
                else:
                    # For other errors, try a shorter date range
                    logger.info("Trying with a smaller chunk size...")
                    half_chunk = (current_end - current_start) / 2
                    if half_chunk.days < 1:
                        logger.error("Chunk size too small, skipping this period")
                        current_start = current_end + timedelta(days=1)
                        continue
                    
                    new_end = current_start + half_chunk
                    logger.info(f"Retrying with: {current_start} to {new_end}")
                    try:
                        # Fetch smaller chunk
                        smaller_chunk_aggs = []
                        for a in self.client.list_aggs(
                            symbol,
                            1,
                            "minute",
                            current_start,
                            new_end,
                            limit=50000,
                        ):
                            smaller_chunk_aggs.append(a)
                        
                        all_aggs.extend(smaller_chunk_aggs)
                        logger.info(f"Successfully fetched {len(smaller_chunk_aggs)} candles from smaller chunk")
                        
                        # Update start for next iteration
                        current_start = new_end + timedelta(days=1)
                        time.sleep(12)  # Rate limit avoidance
                        continue
                    except Exception as e2:
                        logger.error(f"Error with smaller chunk: {str(e2)}")
                        # Skip this chunk and move on
                        current_start = current_end + timedelta(days=1)
                        continue
            
            # Move to next chunk
            current_start = current_end + timedelta(days=1)
        
        # If we didn't get any data, raise an error
        if not all_aggs:
            logger.error("Failed to fetch any data")
            raise ValueError("No data received from API")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'timestamp': agg.timestamp,
            'datetime': pd.to_datetime(agg.timestamp, unit='ms'),
            'open': agg.open,
            'high': agg.high,
            'low': agg.low,
            'close': agg.close,
            'volume': agg.volume,
            'vwap': getattr(agg, 'vwap', None),
            'transactions': getattr(agg, 'transactions', None),
            'symbol': symbol  # Add symbol for reference
        } for agg in all_aggs])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Add date and time columns
        df['date'] = df['datetime'].dt.date
        df['time'] = df['datetime'].dt.time
        
        # Cache the data
        df.to_csv(cache_file, index=False)
        logger.info(f"Saved data to cache: {cache_file}")
        
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
    
    def fetch_premarket_data(self, symbol, date):
        """Fetch premarket data for a specific trading day"""
        if isinstance(date, str):
            date = pd.to_datetime(date).date()
        
        # Define premarket as 4:00 AM to 9:30 AM on the trading day
        # We use 4:00 AM as start to include most relevant premarket activity
        try:
            pm_start = datetime.combine(date, datetime.strptime("04:00", "%H:%M").time())
            pm_end = datetime.combine(date, datetime.strptime("09:30", "%H:%M").time())
            
            pm_start = self.eastern.localize(pm_start)
            pm_end = self.eastern.localize(pm_end)
            
            cache_file = f"{symbol}_premarket_{date}.csv"
            if os.path.exists(cache_file):
                logger.info(f"Loading cached premarket data from {cache_file}")
                pm_df = pd.read_csv(cache_file, parse_dates=['datetime'])
                
                if not pm_df.empty:
                    pm_high = pm_df['high'].max()
                    pm_low = pm_df['low'].min()
                    return pm_high, pm_low, pm_df
            
            aggs = []
            for a in self.client.list_aggs(
                symbol,
                1,
                "minute",
                pm_start,
                pm_end,
                limit=50000,
            ):
                aggs.append(a)
            
            # Convert to DataFrame
            pm_df = pd.DataFrame([{
                'timestamp': agg.timestamp,
                'datetime': pd.to_datetime(agg.timestamp, unit='ms'),
                'open': agg.open,
                'high': agg.high,
                'low': agg.low,
                'close': agg.close,
                'volume': agg.volume,
                'vwap': getattr(agg, 'vwap', None)
            } for agg in aggs])
            
            # Cache the data
            if not pm_df.empty:
                pm_df.to_csv(cache_file, index=False)
            
            # Calculate premarket high and low
            if not pm_df.empty:
                pm_high = pm_df['high'].max()
                pm_low = pm_df['low'].min()
                return pm_high, pm_low, pm_df
            else:
                return None, None, pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching premarket data for {date}: {str(e)}")
            return None, None, pd.DataFrame()
    
    def identify_opening_ranges(self, df):
        """Identify 5-minute opening ranges and premarket levels for each trading day"""
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
            first_5min = market_open.iloc[:5] if len(market_open) >= 5 else market_open
            
            if len(first_5min) < 1:  # Ensure we have at least one candle
                continue
                
            # Calculate OR high and low
            or_high = first_5min['high'].max()
            or_low = first_5min['low'].min()
            or_mid = (or_high + or_low) / 2
            
            # Get premarket high and low
            try:
                symbol = day_df['symbol'].iloc[0]
                pm_high, pm_low, _ = self.fetch_premarket_data(symbol, date)
            except:
                pm_high, pm_low = None, None
            
            # Get previous day high and low
            prev_date = date - timedelta(days=1)
            prev_day_df = df[df['date'] == prev_date]
            
            if len(prev_day_df) > 0:
                pd_high = prev_day_df['high'].max()
                pd_low = prev_day_df['low'].min()
            else:
                pd_high, pd_low = None, None
            
            # Mark data after opening range
            day_df = day_df.copy()
            day_df['or_high'] = or_high
            day_df['or_low'] = or_low
            day_df['or_mid'] = or_mid
            day_df['or_range'] = or_high - or_low
            day_df['or_range_pct'] = (or_high - or_low) / or_low * 100  # Percentage size of OR
            
            if pm_high is not None:
                day_df['pm_high'] = pm_high
            if pm_low is not None:
                day_df['pm_low'] = pm_low
                
            if pd_high is not None:
                day_df['pd_high'] = pd_high
            if pd_low is not None:
                day_df['pd_low'] = pd_low
            
            # Calculate minutes from market open
            first_candle = market_open.iloc[0]['datetime']
            day_df['minutes_from_open'] = (day_df['datetime'] - first_candle).dt.total_seconds() / 60
            
            # Mark if this is within opening range
            day_df['is_opening_range'] = day_df['minutes_from_open'] < 5
            
            # Calculate targets
            day_df['or_upper_target'] = or_high + (or_high - or_low) * 0.5  # 50% extension
            day_df['or_lower_target'] = or_low - (or_high - or_low) * 0.5  # 50% extension
            day_df['or_mid_upper_target'] = or_mid + (or_high - or_mid) * 0.5  # 50% extension from mid to high
            day_df['or_mid_lower_target'] = or_mid - (or_mid - or_low) * 0.5  # 50% extension from mid to low
            
            opening_ranges.append(day_df)
        
        if opening_ranges:
            result_df = pd.concat(opening_ranges)
            return result_df
        else:
            return df

    def identify_candlestick_patterns(self, df):
        """Identify candlestick patterns in the data"""
        # Initialize pattern columns
        df['hammer'] = False
        df['inverted_hammer'] = False
        df['doji'] = False
        df['bullish_engulfing'] = False
        df['bearish_engulfing'] = False
        
        # Calculate necessary pattern components
        range_candle = df['high'] - df['low']
        body_candle = abs(df['close'] - df['open'])
        body_pct = body_candle / range_candle
        
        upper_wick = df['high'] - df.apply(lambda x: max(x['open'], x['close']), axis=1)
        lower_wick = df.apply(lambda x: min(x['open'], x['close']), axis=1) - df['low']
        
        upper_wick_pct = upper_wick / range_candle
        lower_wick_pct = lower_wick / range_candle
        
        # Store these for later analysis
        df['body_size_pct'] = body_pct
        df['upper_wick_pct'] = upper_wick_pct
        df['lower_wick_pct'] = lower_wick_pct
        
        # Identify hammers (bullish)
        hammer_conditions = (
            (body_pct < 0.3) &  # Small body
            (lower_wick_pct > 0.6) &  # Long lower wick
            (upper_wick_pct < 0.1)  # Very small or no upper wick
        )
        df.loc[hammer_conditions, 'hammer'] = True
        
        # Identify inverted hammers / shooting stars
        inv_hammer_conditions = (
            (body_pct < 0.3) &  # Small body
            (upper_wick_pct > 0.6) &  # Long upper wick
            (lower_wick_pct < 0.1)  # Very small or no lower wick
        )
        df.loc[inv_hammer_conditions, 'inverted_hammer'] = True
        
        # Identify doji
        doji_conditions = (
            (body_pct < 0.1)  # Extremely small body
        )
        df.loc[doji_conditions, 'doji'] = True
        
        # Bullish engulfing (requires previous candle)
        for i in range(1, len(df)):
            # Ensure we're comparing candles from same day
            if df.iloc[i]['date'] == df.iloc[i-1]['date']:
                if (df.iloc[i]['open'] < df.iloc[i-1]['close'] and 
                    df.iloc[i]['close'] > df.iloc[i-1]['open'] and
                    df.iloc[i]['close'] > df.iloc[i]['open']):
                    df.iloc[i, df.columns.get_loc('bullish_engulfing')] = True
        
        # Bearish engulfing (requires previous candle)
        for i in range(1, len(df)):
            # Ensure we're comparing candles from same day
            if df.iloc[i]['date'] == df.iloc[i-1]['date']:
                if (df.iloc[i]['open'] > df.iloc[i-1]['close'] and 
                    df.iloc[i]['close'] < df.iloc[i-1]['open'] and
                    df.iloc[i]['close'] < df.iloc[i]['open']):
                    df.iloc[i, df.columns.get_loc('bearish_engulfing')] = True
        
        return df
        
    def detect_orb_breakouts_and_retests(self, df):
        """
        Detect breakouts of ORB levels and subsequent retests
        Adds columns marking breakouts and retests
        """
        # Check if opening range data exists
        if 'or_high' not in df.columns or 'or_low' not in df.columns:
            logger.warning("No opening range data found, can't detect breakouts")
            return df
        
        # Initialize breakout and retest columns
        df['orb_high_breakout'] = False  # True when price breaks above OR high
        df['orb_low_breakout'] = False   # True when price breaks below OR low
        df['orb_high_retest'] = False    # True when price retests OR high from above
        df['orb_low_retest'] = False     # True when price retests OR low from below
        df['orb_mid_breakout_up'] = False # True when price breaks above OR mid
        df['orb_mid_breakout_down'] = False # True when price breaks below OR mid
        df['orb_mid_retest_above'] = False  # True when price retests OR mid from above
        df['orb_mid_retest_below'] = False  # True when price retests OR mid from below
        
        # Breakout and retest detection happens per trading day
        for date, day_df in df.groupby('date'):
            # Skip if not enough data
            if len(day_df) < 6:  # Need at least OR + 1 candle
                continue
                
            # Get only the post-opening range data
            post_or = day_df[day_df['minutes_from_open'] >= 5].copy()
            
            if len(post_or) == 0:
                continue
                
            # Get OR levels
            or_high = day_df['or_high'].iloc[0]
            or_low = day_df['or_low'].iloc[0]
            or_mid = day_df['or_mid'].iloc[0]
            
            # Track breakout status for this day
            high_breakout_occurred = False
            low_breakout_occurred = False
            mid_up_breakout_occurred = False
            mid_down_breakout_occurred = False
            
            high_breakout_idx = None  # Store index of high breakout
            low_breakout_idx = None   # Store index of low breakout
            mid_up_breakout_idx = None
            mid_down_breakout_idx = None
            
            # Detect OR high breakout
            for i in range(len(post_or)):
                idx = post_or.index[i]
                current = post_or.iloc[i]
                
                # Check for OR high breakout
                if not high_breakout_occurred and current['high'] > or_high:
                    high_breakout_occurred = True
                    high_breakout_idx = idx
                    df.loc[idx, 'orb_high_breakout'] = True
                
                # Check for OR low breakout
                if not low_breakout_occurred and current['low'] < or_low:
                    low_breakout_occurred = True
                    low_breakout_idx = idx
                    df.loc[idx, 'orb_low_breakout'] = True
                
                # Check for OR mid breakouts
                if not mid_up_breakout_occurred and current['close'] > or_mid:
                    mid_up_breakout_occurred = True
                    mid_up_breakout_idx = idx
                    df.loc[idx, 'orb_mid_breakout_up'] = True
                
                if not mid_down_breakout_occurred and current['close'] < or_mid:
                    mid_down_breakout_occurred = True
                    mid_down_breakout_idx = idx
                    df.loc[idx, 'orb_mid_breakout_down'] = True
            
            # Detect retests after breakouts
            if high_breakout_occurred:
                # Get data after high breakout
                after_high_breakout = post_or.loc[post_or.index > high_breakout_idx].copy()
                
                for i in range(len(after_high_breakout)):
                    idx = after_high_breakout.index[i]
                    current = after_high_breakout.iloc[i]
                    
                    # Retest detection: price comes back close to OR high from above
                    if current['low'] <= or_high * 1.002:  # Within 0.2% of OR high
                        df.loc[idx, 'orb_high_retest'] = True
            
            if low_breakout_occurred:
                # Get data after low breakout
                after_low_breakout = post_or.loc[post_or.index > low_breakout_idx].copy()
                
                for i in range(len(after_low_breakout)):
                    idx = after_low_breakout.index[i]
                    current = after_low_breakout.iloc[i]
                    
                    # Retest detection: price comes back close to OR low from below
                    if current['high'] >= or_low * 0.998:  # Within 0.2% of OR low
                        df.loc[idx, 'orb_low_retest'] = True
            
            # Detect OR mid retests
            if mid_up_breakout_occurred:
                after_mid_up = post_or.loc[post_or.index > mid_up_breakout_idx].copy()
                
                for i in range(len(after_mid_up)):
                    idx = after_mid_up.index[i]
                    current = after_mid_up.iloc[i]
                    
                    if current['low'] <= or_mid * 1.002:
                        df.loc[idx, 'orb_mid_retest_above'] = True
            
            if mid_down_breakout_occurred:
                after_mid_down = post_or.loc[post_or.index > mid_down_breakout_idx].copy()
                
                for i in range(len(after_mid_down)):
                    idx = after_mid_down.index[i]
                    current = after_mid_down.iloc[i]
                    
                    if current['high'] >= or_mid * 0.998:
                        df.loc[idx, 'orb_mid_retest_below'] = True
        
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
            
            # Distance from OR mid
            if 'or_mid' in df.columns:
                df['dist_from_or_mid'] = (df['close'] - df['or_mid']) / df['or_mid'] * 100
        
        # Distance from premarket levels
        if 'pm_high' in df.columns:
            df['dist_from_pm_high'] = (df['close'] - df['pm_high']) / df['pm_high'] * 100
        if 'pm_low' in df.columns:
            df['dist_from_pm_low'] = (df['close'] - df['pm_low']) / df['pm_low'] * 100
            
        # Distance from previous day levels
        if 'pd_high' in df.columns:
            df['dist_from_pd_high'] = (df['close'] - df['pd_high']) / df['pd_high'] * 100
        if 'pd_low' in df.columns:
            df['dist_from_pd_low'] = (df['close'] - df['pd_low']) / df['pd_low'] * 100
        
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
    
    def process_data_for_orb_analysis(self, symbol, start_date, end_date, force_refresh=False):
        """Complete data processing pipeline for ORB analysis"""
        # Check for processed data cache
        processed_file = f"{symbol}_processed_{start_date}_{end_date}.csv"
        
        if os.path.exists(processed_file) and not force_refresh:
            logger.info(f"Loading processed data from {processed_file}")
            return pd.read_csv(processed_file, parse_dates=['datetime', 'date'])
        
        # Fetch raw data
        df = self.fetch_historical_data(symbol, start_date, end_date)
        
        # Process data
        logger.info("Identifying opening ranges...")
        df = self.identify_opening_ranges(df)
        
        logger.info("Identifying candlestick patterns...")
        df = self.identify_candlestick_patterns(df)
        
        logger.info("Detecting breakouts and retests...")
        df = self.detect_orb_breakouts_and_retests(df)
        
        logger.info("Adding enhanced features...")
        df = self.add_enhanced_features(df)
        
        # Save processed data
        df.to_csv(processed_file, index=False)
        logger.info(f"Saved processed data to {processed_file}")
        
        return df
    
    def fetch_daily_data(self, symbol, start_date, end_date):
        """Fetch daily data for a symbol over a date range (uses less API calls)"""
        logger.info(f"Fetching daily data for {symbol} from {start_date} to {end_date}")
        
        # Check if cached data exists
        cache_file = f"{symbol}_daily_cache_{start_date}_{end_date}.csv"
        if os.path.exists(cache_file):
            logger.info(f"Loading cached daily data from {cache_file}")
            df = pd.read_csv(cache_file, parse_dates=['datetime'])
            return df
        
        try:
            # Fetch daily data (much fewer API calls than minute data)
            aggs = []
            for a in self.client.list_aggs(
                symbol,
                1,
                "day",
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
                'vwap': getattr(agg, 'vwap', None)
            } for agg in aggs])
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            df['date'] = df['datetime'].dt.date
            
            # Cache the data
            df.to_csv(cache_file, index=False)
            logger.info(f"Saved daily data to cache: {cache_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching daily data: {str(e)}")
            raise
