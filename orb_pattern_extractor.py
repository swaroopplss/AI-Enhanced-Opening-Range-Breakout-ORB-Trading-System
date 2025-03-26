import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import ta

logger = logging.getLogger(__name__)

class ORBPatternExtractor:
    def __init__(self, profit_target_pct=0.5, stop_loss_pct=0.25):
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
    
    def add_technical_indicators(self, df):
        """Add technical indicators to the dataframe"""
        # Volume-based indicators
        df['volume_ma10'] = df['volume'].rolling(window=10).mean()
        df['rel_volume'] = df['volume'] / df['volume_ma10']
        
        # Price-based indicators
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
        
        # Candle pattern recognition
        df['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
        df['upper_wick'] = (df['high'] - np.maximum(df['open'], df['close'])) / (df['high'] - df['low'])
        df['lower_wick'] = (np.minimum(df['open'], df['close']) - df['low']) / (df['high'] - df['low'])
        
        # Identify hammer and shooting star patterns
        df['hammer'] = (df['lower_wick'] > 0.6) & (df['upper_wick'] < 0.2) & (df['body_size'] < 0.4)
        df['shooting_star'] = (df['upper_wick'] > 0.6) & (df['lower_wick'] < 0.2) & (df['body_size'] < 0.4)
        
        return df
    
    def extract_patterns(self, df, min_volume_threshold=1.2):
        """Extract ORB patterns from dataframe"""
        # Add technical indicators
        df = self.add_technical_indicators(df)
        
        # Group by date
        grouped = df.groupby('date')
        
        # For each day, identify ORB patterns
        patterns = []
        
        for date, day_df in grouped:
            # Skip if no opening range data
            if 'or_high' not in day_df.columns:
                continue
                
            # Get data after opening range
            post_or = day_df[day_df['minutes_from_open'] >= 5].reset_index(drop=True)
            
            if len(post_or) < 30:  # Need at least 30 minutes of data after opening
                continue
            
            # Get OR high and low
            or_high = day_df['or_high'].iloc[0]
            or_low = day_df['or_low'].iloc[0]
            or_range = or_high - or_low
            
            # Look for breakouts and retests
            self._process_upside_patterns(date, post_or, or_high, or_low, or_range, min_volume_threshold, patterns)
            self._process_downside_patterns(date, post_or, or_high, or_low, or_range, min_volume_threshold, patterns)
        
        return patterns
    
    def _process_upside_patterns(self, date, post_or, or_high, or_low, or_range, min_volume_threshold, patterns):
        """Process upside (long) patterns"""
        # Find first breakout above OR high
        breakout_up = post_or[post_or['high'] > or_high]
        if breakout_up.empty:
            return
            
        breakout_up_idx = breakout_up.index[0]
        breakout_candle = post_or.loc[breakout_up_idx]
        
        # Look for retest after breakout
        if breakout_up_idx < len(post_or) - 5:
            # Get data after breakout
            after_breakout = post_or.loc[breakout_up_idx+1:]
            
            # Look for pullback to OR high level
            for i in range(len(after_breakout)):
                idx = after_breakout.index[i]
                row = after_breakout.iloc[i]
                
                # Check if low is close to OR high (within 0.2%)
                if (abs(row['low'] - or_high) / or_high) < 0.002:
                    # Check for bullish confirmation
                    if ((row['close'] > row['open'] and row['rel_volume'] > min_volume_threshold) or 
                        (row['hammer'] and row['close'] > row['open'] * 1.0005)):
                        
                        # Extract pattern and determine outcome
                        pattern = self._create_pattern(
                            date=date,
                            post_or=post_or,
                            retest_idx=idx,
                            breakout_idx=breakout_up_idx,
                            direction='LONG',
                            or_high=or_high,
                            or_low=or_low
                        )
                        
                        patterns.append(pattern)
                        # Only record the first valid retest
                        break
    
    def _process_downside_patterns(self, date, post_or, or_high, or_low, or_range, min_volume_threshold, patterns):
        """Process downside (short) patterns"""
        # Find first breakout below OR low
        breakout_down = post_or[post_or['low'] < or_low]
        if breakout_down.empty:
            return
            
        breakout_down_idx = breakout_down.index[0]
        breakout_candle = post_or.loc[breakout_down_idx]
        
        # Look for retest after breakout
        if breakout_down_idx < len(post_or) - 5:
            # Get data after breakout
            after_breakout = post_or.loc[breakout_down_idx+1:]
            
            # Look for pullback to OR low level
            for i in range(len(after_breakout)):
                idx = after_breakout.index[i]
                row = after_breakout.iloc[i]
                
                # Check if high is close to OR low (within 0.2%)
                if (abs(row['high'] - or_low) / or_low) < 0.002:
                    # Check for bearish confirmation
                    if ((row['close'] < row['open'] and row['rel_volume'] > min_volume_threshold) or 
                        (row['shooting_star'] and row['close'] < row['open'] * 0.9995)):
                        
                        # Extract pattern and determine outcome
                        pattern = self._create_pattern(
                            date=date,
                            post_or=post_or,
                            retest_idx=idx,
                            breakout_idx=breakout_down_idx,
                            direction='SHORT',
                            or_high=or_high,
                            or_low=or_low
                        )
                        
                        patterns.append(pattern)
                        # Only record the first valid retest
                        break
    
    def _create_pattern(self, date, post_or, retest_idx, breakout_idx, direction, or_high, or_low):
        """Create pattern record with outcome determination"""
        retest_candle = post_or.loc[retest_idx]
        breakout_candle = post_or.loc[breakout_idx]
        
        # Extract features for AI analysis
        pre_retest = post_or.loc[max(0, retest_idx-5):retest_idx-1]
        features = self._extract_features(pre_retest, retest_candle)
        
        # Calculate pattern outcome
        after_retest = post_or.loc[retest_idx:min(retest_idx+30, len(post_or)-1)]
        
        if direction == 'LONG':
            entry_price = retest_candle['close']
            target_price = entry_price * (1 + self.profit_target_pct/100)
            stop_price = entry_price * (1 - self.stop_loss_pct/100)
        else:  # SHORT
            entry_price = retest_candle['close']
            target_price = entry_price * (1 - self.profit_target_pct/100)
            stop_price = entry_price * (1 + self.stop_loss_pct/100)
        
        # Check if target or stop was hit first
        target_hit = False
        stop_hit = False
        exit_time = None
        
        for i, row in after_retest.iterrows():
            if direction == 'LONG':
                if row['high'] >= target_price:
                    target_hit = True
                    exit_time = row['datetime']
                    break
                if row['low'] <= stop_price:
                    stop_hit = True
                    exit_time = row['datetime']
                    break
            else:  # SHORT
                if row['low'] <= target_price:
                    target_hit = True
                    exit_time = row['datetime']
                    break
                if row['high'] >= stop_price:
                    stop_hit = True
                    exit_time = row['datetime']
                    break
        
        # Determine outcome
        if target_hit:
            outcome = 1  # Win
        elif stop_hit:
            outcome = -1  # Loss
        else:
            # Calculate P&L at end of day
            final_close = after_retest.iloc[-1]['close']
            if direction == 'LONG':
                pnl = (final_close - entry_price) / entry_price * 100
            else:  # SHORT
                pnl = (entry_price - final_close) / entry_price * 100
            
            outcome = 1 if pnl > 0 else -1
        
        # Create pattern record
        pattern = {
            'date': date,
            'entry_time': retest_candle['datetime'],
            'exit_time': exit_time,
            'direction': direction,
            'outcome': outcome,
            'or_high': or_high,
            'or_low': or_low,
            'breakout_time': breakout_candle['datetime'],
            'breakout_price': breakout_candle['high'] if direction == 'LONG' else breakout_candle['low'],
            'retest_price': retest_candle['close'],
            'entry_price': entry_price,
            'target_price': target_price,
            'stop_price': stop_price,
            'rsi': retest_candle['rsi'],
            'rel_volume': retest_candle['rel_volume'],
            'price_to_vwap': retest_candle['close'] / retest_candle['vwap'] - 1 if 'vwap' in retest_candle and retest_candle['vwap'] > 0 else 0,
            'price_to_ema9': retest_candle['close'] / retest_candle['ema9'] - 1 if not np.isnan(retest_candle['ema9']) and retest_candle['ema9'] > 0 else 0,
            'or_range_pct': (or_high - or_low) / or_low * 100,
            'macd': retest_candle['macd'],
            'macd_signal': retest_candle['macd_signal'],
            'adx': retest_candle['adx'],
            'hammer': bool(retest_candle['hammer']) if 'hammer' in retest_candle else False,
            'shooting_star': bool(retest_candle['shooting_star']) if 'shooting_star' in retest_candle else False,
            'features': features
        }
        
        return pattern
    
    def _extract_features(self, pre_retest, retest_candle):
        """Extract feature vector from price action before retest with robust NaN handling"""
        import numpy as np
        features = []
        
        # Add price changes for last 5 candles
        for i in range(len(pre_retest)):
            try:
                candle = pre_retest.iloc[i]
                # Ensure we don't divide by zero
                if candle['open'] > 0 and candle['low'] > 0:
                    features.append(candle['close'] / candle['open'] - 1)  # Candle body
                    features.append((candle['high'] - candle['low']) / candle['low'] * 100)  # Candle range
                else:
                    features.append(0)  # Default to zero if we have division by zero
                    features.append(0)
            except Exception as e:
                # Log warning but continue with zeros
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Error processing candle data: {str(e)}")
                features.append(0)
                features.append(0)
        
        # Pad with zeros if we have fewer than 5 candles
        features.extend([0] * (10 - len(features)))
        
        # Add technical indicators with safe extraction
        features.append(self._safe_get_value(retest_candle, 'rsi', 100, 50))
        features.append(min(self._safe_get_value(retest_candle, 'rel_volume', 3, 1), 1))
        features.append(self._safe_get_value(retest_candle, 'macd', None, 0))
        features.append(self._safe_get_value(retest_candle, 'macd_signal', None, 0))
        features.append(self._safe_get_value(retest_candle, 'adx', 100, 50))
        
        # Add candle pattern information
        features.append(1 if retest_candle.get('hammer', False) else 0)
        features.append(1 if retest_candle.get('shooting_star', False) else 0)
        
        # Final safety check - replace any NaN with zeros
        features = [0 if np.isnan(x) else x for x in features]
        
        return features

    def _safe_get_value(self, candle, field, divisor=None, default=0):
        """Safely get a value from the candle, handling missing values and NaNs"""
        import numpy as np
        import pandas as pd
        
        try:
            # Check if field exists
            if field not in candle:
                return default
                
            # Get the value and check if it's NaN
            value = candle[field]
            if pd.isna(value) or np.isnan(value):
                return default
                
            # If we have a divisor, divide the value
            if divisor is not None:
                return value / divisor
                
            return value
        except:
            return default