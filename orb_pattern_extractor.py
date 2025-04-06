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
        
        # Identify doji patterns
        df['doji'] = (df['body_size'] < 0.1)
        
        # Identify engulfing patterns
        for i in range(1, len(df)):
            # Ensure we're comparing candles from same day
            if 'date' in df.columns and df.iloc[i]['date'] == df.iloc[i-1]['date']:
                # Bullish engulfing
                if (df.iloc[i]['open'] < df.iloc[i-1]['close'] and 
                    df.iloc[i]['close'] > df.iloc[i-1]['open'] and
                    df.iloc[i]['close'] > df.iloc[i]['open']):
                    df.iloc[i, df.columns.get_loc('bullish_engulfing')] = True
                
                # Bearish engulfing
                if (df.iloc[i]['open'] > df.iloc[i-1]['close'] and 
                    df.iloc[i]['close'] < df.iloc[i-1]['open'] and
                    df.iloc[i]['close'] < df.iloc[i]['open']):
                    df.iloc[i, df.columns.get_loc('bearish_engulfing')] = True
        
        return df
    
    def extract_patterns(self, df, min_volume_threshold=1.2):
        """Extract ORB patterns from dataframe"""
        # Add technical indicators if not already present
        if 'rsi' not in df.columns:
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
            or_mid = day_df['or_mid'].iloc[0] if 'or_mid' in day_df.columns else (or_high + or_low) / 2
            or_range = or_high - or_low
            
            # Get premarket levels if available
            pm_high = day_df['pm_high'].iloc[0] if 'pm_high' in day_df.columns else None
            pm_low = day_df['pm_low'].iloc[0] if 'pm_low' in day_df.columns else None
            
            # Get previous day levels if available
            pd_high = day_df['pd_high'].iloc[0] if 'pd_high' in day_df.columns else None
            pd_low = day_df['pd_low'].iloc[0] if 'pd_low' in day_df.columns else None
            
            # Look for breakouts and retests
            self._process_upside_patterns(date, post_or, or_high, or_low, or_mid, or_range, min_volume_threshold, patterns, pm_high, pm_low, pd_high, pd_low)
            self._process_downside_patterns(date, post_or, or_high, or_low, or_mid, or_range, min_volume_threshold, patterns, pm_high, pm_low, pd_high, pd_low)
            
            # Also process patterns for ORB mid level
            self._process_orb_mid_patterns(date, post_or, or_high, or_low, or_mid, or_range, min_volume_threshold, patterns)
        
        return patterns
    
    def _process_upside_patterns(self, date, post_or, or_high, or_low, or_mid, or_range, min_volume_threshold, patterns, pm_high=None, pm_low=None, pd_high=None, pd_low=None):
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
                    bullish_confirmation = False
                    
                    # Strong bullish candle with volume
                    if row['close'] > row['open'] and row['rel_volume'] > min_volume_threshold:
                        bullish_confirmation = True
                    
                    # Hammer pattern at support
                    elif row.get('hammer', False) and row['close'] > row['open'] * 1.0005:
                        bullish_confirmation = True
                    
                    # Bullish engulfing pattern
                    elif row.get('bullish_engulfing', False):
                        bullish_confirmation = True
                    
                    # Check if this also coincides with a key level (pm_high or pd_high)
                    level_confluence = False
                    confluence_level = None
                    
                    if pm_high is not None and (abs(row['low'] - pm_high) / pm_high) < 0.002:
                        level_confluence = True
                        confluence_level = 'pm_high'
                    
                    if pd_high is not None and (abs(row['low'] - pd_high) / pd_high) < 0.002:
                        level_confluence = True
                        confluence_level = 'pd_high'
                    
                    if bullish_confirmation:
                        # Extract pattern and determine outcome
                        pattern = self._create_pattern(
                            date=date,
                            post_or=post_or,
                            retest_idx=idx,
                            breakout_idx=breakout_up_idx,
                            direction='LONG',
                            or_high=or_high,
                            or_low=or_low,
                            or_mid=or_mid,
                            level_confluence=level_confluence,
                            confluence_level=confluence_level,
                            pm_high=pm_high,
                            pm_low=pm_low,
                            pd_high=pd_high,
                            pd_low=pd_low
                        )
                        
                        patterns.append(pattern)
                        # Only record the first valid retest
                        break
    
    def _process_downside_patterns(self, date, post_or, or_high, or_low, or_mid, or_range, min_volume_threshold, patterns, pm_high=None, pm_low=None, pd_high=None, pd_low=None):
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
                    bearish_confirmation = False
                    
                    # Strong bearish candle with volume
                    if row['close'] < row['open'] and row['rel_volume'] > min_volume_threshold:
                        bearish_confirmation = True
                    
                    # Shooting star pattern at resistance
                    elif row.get('shooting_star', False) and row['close'] < row['open'] * 0.9995:
                        bearish_confirmation = True
                    
                    # Bearish engulfing pattern
                    elif row.get('bearish_engulfing', False):
                        bearish_confirmation = True
                    
                    # Check if this also coincides with a key level (pm_low or pd_low)
                    level_confluence = False
                    confluence_level = None
                    
                    if pm_low is not None and (abs(row['high'] - pm_low) / pm_low) < 0.002:
                        level_confluence = True
                        confluence_level = 'pm_low'
                    
                    if pd_low is not None and (abs(row['high'] - pd_low) / pd_low) < 0.002:
                        level_confluence = True
                        confluence_level = 'pd_low'
                    
                    if bearish_confirmation:
                        # Extract pattern and determine outcome
                        pattern = self._create_pattern(
                            date=date,
                            post_or=post_or,
                            retest_idx=idx,
                            breakout_idx=breakout_down_idx,
                            direction='SHORT',
                            or_high=or_high,
                            or_low=or_low,
                            or_mid=or_mid,
                            level_confluence=level_confluence,
                            confluence_level=confluence_level,
                            pm_high=pm_high,
                            pm_low=pm_low,
                            pd_high=pd_high,
                            pd_low=pd_low
                        )
                        
                        patterns.append(pattern)
                        # Only record the first valid retest
                        break
    
    def _process_orb_mid_patterns(self, date, post_or, or_high, or_low, or_mid, or_range, min_volume_threshold, patterns):
        """Process patterns around the ORB mid level"""
        # Skip if we don't have many candles
        if len(post_or) < 15:
            return
            
        # First, check for a break above ORB mid
        breakout_above_mid = post_or[post_or['close'] > or_mid]
        if not breakout_above_mid.empty:
            breakout_idx = breakout_above_mid.index[0]
            
            # Look for retest from above
            after_breakout = post_or.loc[breakout_idx+1:] if breakout_idx < len(post_or) - 1 else pd.DataFrame()
            
            for i in range(len(after_breakout)):
                idx = after_breakout.index[i]
                row = after_breakout.iloc[i]
                
                # Check if low is close to OR mid (within 0.2%)
                if (abs(row['low'] - or_mid) / or_mid) < 0.002:
                    # Check for bullish confirmation
                    if ((row['close'] > row['open'] and row['rel_volume'] > min_volume_threshold) or 
                        row.get('hammer', False) or row.get('bullish_engulfing', False)):
                        
                        # Create pattern
                        pattern = self._create_pattern(
                            date=date,
                            post_or=post_or,
                            retest_idx=idx,
                            breakout_idx=breakout_idx,
                            direction='LONG_MID',  # Special type for mid-level longs
                            or_high=or_high,
                            or_low=or_low,
                            or_mid=or_mid
                        )
                        
                        patterns.append(pattern)
                        break
        
        # Then, check for a break below ORB mid
        breakout_below_mid = post_or[post_or['close'] < or_mid]
        if not breakout_below_mid.empty:
            breakout_idx = breakout_below_mid.index[0]
            
            # Look for retest from below
            after_breakout = post_or.loc[breakout_idx+1:] if breakout_idx < len(post_or) - 1 else pd.DataFrame()
            
            for i in range(len(after_breakout)):
                idx = after_breakout.index[i]
                row = after_breakout.iloc[i]
                
                # Check if high is close to OR mid (within 0.2%)
                if (abs(row['high'] - or_mid) / or_mid) < 0.002:
                    # Check for bearish confirmation
                    if ((row['close'] < row['open'] and row['rel_volume'] > min_volume_threshold) or 
                        row.get('shooting_star', False) or row.get('bearish_engulfing', False)):
                        
                        # Create pattern
                        pattern = self._create_pattern(
                            date=date,
                            post_or=post_or,
                            retest_idx=idx,
                            breakout_idx=breakout_idx,
                            direction='SHORT_MID',  # Special type for mid-level shorts
                            or_high=or_high,
                            or_low=or_low,
                            or_mid=or_mid
                        )
                        
                        patterns.append(pattern)
                        break
    
    def _create_pattern(self, date, post_or, retest_idx, breakout_idx, direction, or_high, or_low, or_mid=None,
                       level_confluence=False, confluence_level=None, pm_high=None, pm_low=None, pd_high=None, pd_low=None):
        """Create pattern record with outcome determination"""
        retest_candle = post_or.loc[retest_idx]
        breakout_candle = post_or.loc[breakout_idx]
        
        # Extract features for AI analysis
        pre_retest = post_or.loc[max(0, retest_idx-5):retest_idx-1]
        features = self._extract_features(pre_retest, retest_candle)
        
        # Calculate pattern outcome
        after_retest = post_or.loc[retest_idx:min(retest_idx+30, len(post_or)-1)]
        
        # Determine entry, target and stop
        if direction in ('LONG', 'LONG_MID'):
            entry_price = retest_candle['close']
            target_price = entry_price * (1 + self.profit_target_pct/100)
            stop_price = entry_price * (1 - self.stop_loss_pct/100)
        else:  # SHORT or SHORT_MID
            entry_price = retest_candle['close']
            target_price = entry_price * (1 - self.profit_target_pct/100)
            stop_price = entry_price * (1 + self.stop_loss_pct/100)
        
        # Check if target or stop was hit first
        target_hit = False
        stop_hit = False
        exit_time = None
        
        for i, row in after_retest.iterrows():
            if direction in ('LONG', 'LONG_MID'):
                if row['high'] >= target_price:
                    target_hit = True
                    exit_time = row['datetime']
                    break
                if row['low'] <= stop_price:
                    stop_hit = True
                    exit_time = row['datetime']
                    break
            else:  # SHORT or SHORT_MID
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
            outcome = 0  # Loss
        else:
            # Calculate P&L at end of day
            final_close = after_retest.iloc[-1]['close']
            if direction in ('LONG', 'LONG_MID'):
                pnl = (final_close - entry_price) / entry_price * 100
            else:  # SHORT or SHORT_MID
                pnl = (entry_price - final_close) / entry_price * 100
            
            outcome = 1 if pnl > 0 else 0
        
        # Determine pattern quality based on confluence
        pattern_quality = 'standard'
        if level_confluence:
            pattern_quality = 'high'  # Higher quality due to level confluence
        
        # Extract candle pattern types
        candle_pattern = 'none'
        if retest_candle.get('hammer', False):
            candle_pattern = 'hammer'
        elif retest_candle.get('shooting_star', False):
            candle_pattern = 'shooting_star'
        elif retest_candle.get('doji', False):
            candle_pattern = 'doji'
        elif retest_candle.get('bullish_engulfing', False):
            candle_pattern = 'bullish_engulfing'
        elif retest_candle.get('bearish_engulfing', False):
            candle_pattern = 'bearish_engulfing'
        
        # Create pattern record
        pattern = {
            'date': date,
            'entry_time': retest_candle['datetime'],
            'exit_time': exit_time,
            'direction': direction,
            'outcome': outcome,
            'or_high': or_high,
            'or_low': or_low,
            'or_mid': or_mid,
            'breakout_time': breakout_candle['datetime'],
            'breakout_price': breakout_candle['high'] if direction in ('LONG', 'LONG_MID') else breakout_candle['low'],
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
            'candle_pattern': candle_pattern,
            'pattern_quality': pattern_quality,
            'level_confluence': level_confluence,
            'confluence_level': confluence_level,
            'features': features
        }
        
        # Add distance from key levels
        if pm_high is not None:
            pattern['dist_from_pm_high'] = (entry_price - pm_high) / pm_high * 100
        
        if pm_low is not None:
            pattern['dist_from_pm_low'] = (entry_price - pm_low) / pm_low * 100
        
        if pd_high is not None:
            pattern['dist_from_pd_high'] = (entry_price - pd_high) / pd_high * 100
        
        if pd_low is not None:
            pattern['dist_from_pd_low'] = (entry_price - pd_low) / pd_low * 100
        
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
        features.append(min(self._safe_get_value(retest_candle, 'rel_volume', 3, 1), 3))
        features.append(self._safe_get_value(retest_candle, 'macd', None, 0))
        features.append(self._safe_get_value(retest_candle, 'macd_signal', None, 0))
        features.append(self._safe_get_value(retest_candle, 'adx', 100, 20))
        
        # Add candle pattern information
        features.append(1 if retest_candle.get('hammer', False) else 0)
        features.append(1 if retest_candle.get('shooting_star', False) else 0)
        features.append(1 if retest_candle.get('doji', False) else 0)
        features.append(1 if retest_candle.get('bullish_engulfing', False) else 0)
        features.append(1 if retest_candle.get('bearish_engulfing', False) else 0)
        
        # Add body and wick proportions
        features.append(self._safe_get_value(retest_candle, 'body_size', 1, 0))
        features.append(self._safe_get_value(retest_candle, 'upper_wick', 1, 0))
        features.append(self._safe_get_value(retest_candle, 'lower_wick', 1, 0))
        
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
    
    def analyze_pattern_success_by_type(self, patterns):
        """Analyze success rates of different pattern types"""
        if not patterns:
            return {}
        
        analysis = {
            'overall': {'count': 0, 'wins': 0, 'win_rate': 0},
            'by_direction': {},
            'by_candle_pattern': {},
            'by_quality': {},
            'by_confluence': {}
        }
        
        # Overall analysis
        total_patterns = len(patterns)
        winning_patterns = sum(1 for p in patterns if p['outcome'] == 1)
        win_rate = winning_patterns / total_patterns * 100 if total_patterns > 0 else 0
        
        analysis['overall'] = {
            'count': total_patterns,
            'wins': winning_patterns,
            'win_rate': win_rate
        }
        
        # Analysis by direction
        directions = set(p['direction'] for p in patterns)
        for direction in directions:
            direction_patterns = [p for p in patterns if p['direction'] == direction]
            direction_wins = sum(1 for p in direction_patterns if p['outcome'] == 1)
            direction_win_rate = direction_wins / len(direction_patterns) * 100 if direction_patterns else 0
            
            analysis['by_direction'][direction] = {
                'count': len(direction_patterns),
                'wins': direction_wins,
                'win_rate': direction_win_rate
            }
        
        # Analysis by candle pattern
        candle_patterns = set(p.get('candle_pattern', 'none') for p in patterns)
        for pattern_type in candle_patterns:
            pattern_matches = [p for p in patterns if p.get('candle_pattern', 'none') == pattern_type]
            pattern_wins = sum(1 for p in pattern_matches if p['outcome'] == 1)
            pattern_win_rate = pattern_wins / len(pattern_matches) * 100 if pattern_matches else 0
            
            analysis['by_candle_pattern'][pattern_type] = {
                'count': len(pattern_matches),
                'wins': pattern_wins,
                'win_rate': pattern_win_rate
            }
        
        # Analysis by pattern quality
        qualities = set(p.get('pattern_quality', 'standard') for p in patterns)
        for quality in qualities:
            quality_patterns = [p for p in patterns if p.get('pattern_quality', 'standard') == quality]
            quality_wins = sum(1 for p in quality_patterns if p['outcome'] == 1)
            quality_win_rate = quality_wins / len(quality_patterns) * 100 if quality_patterns else 0
            
            analysis['by_quality'][quality] = {
                'count': len(quality_patterns),
                'wins': quality_wins,
                'win_rate': quality_win_rate
            }
        
        # Analysis by level confluence
        confluence_patterns = [p for p in patterns if p.get('level_confluence', False)]
        if confluence_patterns:
            confluence_wins = sum(1 for p in confluence_patterns if p['outcome'] == 1)
            confluence_win_rate = confluence_wins / len(confluence_patterns) * 100
            
            analysis['by_confluence']['with_confluence'] = {
                'count': len(confluence_patterns),
                'wins': confluence_wins,
                'win_rate': confluence_win_rate
            }
            
            # Further breakdown by confluence level
            confluence_levels = set(p.get('confluence_level', 'unknown') for p in confluence_patterns)
            for level in confluence_levels:
                if level:
                    level_patterns = [p for p in confluence_patterns if p.get('confluence_level') == level]
                    level_wins = sum(1 for p in level_patterns if p['outcome'] == 1)
                    level_win_rate = level_wins / len(level_patterns) * 100 if level_patterns else 0
                    
                    analysis['by_confluence'][level] = {
                        'count': len(level_patterns),
                        'wins': level_wins,
                        'win_rate': level_win_rate
                    }
        
        return analysis
