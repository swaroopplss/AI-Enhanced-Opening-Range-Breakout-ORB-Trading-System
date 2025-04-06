import pandas as pd
import numpy as np
from polygon import RESTClient
from datetime import datetime, timedelta
import pytz
import time
import logging
import os
from pattern_model import PatternModel
from orb_pattern_extractor import ORBPatternExtractor
from ai_reasoning_engine import LocalAIReasoningEngine, HybridReasoningEngine, RuleBasedReasoningEngine
from data_collector import DataCollector

logger = logging.getLogger(__name__)

class ORBTradingAssistant:
    """
    Trading assistant for implementing the Opening Range Breakout (ORB) strategy
    with AI-enhanced decision making.
    """
    
    def __init__(self, api_key, model_path=None, use_ai=True, ai_model_path=None, data_dir="market_data"):
        """
        Initialize the trading assistant with AI integration
        
        Args:
            api_key: Polygon.io API key
            model_path: Path to trained pattern model file
            use_ai: Whether to use AI reasoning (True) or rule-based only (False)
            ai_model_path: Path to AI language model (for LocalAIReasoningEngine)
            data_dir: Directory to store market data and analysis
        """
        self.client = RESTClient(api_key)
        self.api_key = api_key
        self.eastern = pytz.timezone('US/Eastern')
        self.extractor = ORBPatternExtractor()
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Create data collector
        self.data_collector = DataCollector(api_key)
        
        # Load pattern model if path provided
        self.pattern_model = None
        if model_path and os.path.exists(model_path):
            try:
                self.pattern_model = PatternModel.load(model_path)
                logger.info(f"Loaded pattern model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading pattern model: {str(e)}")
                logger.info("Creating new pattern model")
                self.pattern_model = PatternModel()
        
        # Initialize the hybrid reasoning engine
        self.reasoning_engine = HybridReasoningEngine(
            model_path=ai_model_path,
            use_ai=use_ai,
            data_dir=data_dir
        )
        
        # Trade tracking for continuous learning
        self.active_trades = {}
        self.trade_history = []
        
    def get_current_data(self, symbol, lookback_minutes=60):
        """
        Get current market data for a symbol including ORB, premarket, and previous day levels
        
        Args:
            symbol: Trading symbol to fetch data for
            lookback_minutes: How many minutes of data to fetch
            
        Returns:
            DataFrame with processed market data or None if data unavailable
        """
        now = datetime.now(self.eastern)
        today = now.date()
        yesterday = today - timedelta(days=1)
        
        # Attempt to get data from cache first
        cache_file = f"{symbol}_today_data.csv"
        if os.path.exists(cache_file):
            try:
                cached_df = pd.read_csv(cache_file, parse_dates=['datetime', 'date'])
                last_update = cached_df['datetime'].max()
                
                # If cache is recent (within 5 minutes), use it
                if (now - last_update).total_seconds() < 300:
                    logger.info(f"Using cached data (last update: {last_update})")
                    return cached_df
            except Exception as e:
                logger.warning(f"Error reading cache: {str(e)}")
        
        # Fetch data for today and yesterday
        try:
            # Format dates for API
            end_date = today.strftime('%Y-%m-%d')
            start_date = (today - timedelta(days=2)).strftime('%Y-%m-%d')
            
            # Use data collector to fetch and process data
            df = self.data_collector.fetch_historical_data(symbol, start_date, end_date)
            
            # Skip if no data
            if df.empty:
                logger.warning("No data returned from API")
                return None
            
            # Add date and time columns if not present
            if 'date' not in df.columns:
                df['date'] = df['datetime'].dt.date
            if 'time' not in df.columns:
                df['time'] = df['datetime'].dt.time
            
            # Process data with enhanced features
            df = self.data_collector.identify_opening_ranges(df)
            df = self.data_collector.identify_candlestick_patterns(df)
            df = self.data_collector.detect_orb_breakouts_and_retests(df)
            df = self.data_collector.add_enhanced_features(df)
            
            # Get today's data
            today_data = df[df['date'] == today].copy()
            
            if today_data.empty:
                logger.warning("No data available for today")
                return None
            
            # Add previous day high/low if not already present
            if 'pd_high' not in today_data.columns or 'pd_low' not in today_data.columns:
                prev_day_data = df[df['date'] == yesterday]
                if not prev_day_data.empty:
                    pd_high = prev_day_data['high'].max()
                    pd_low = prev_day_data['low'].min()
                    today_data['pd_high'] = pd_high
                    today_data['pd_low'] = pd_low
            
            # Calculate premarket high/low if not already present
            if 'pm_high' not in today_data.columns or 'pm_low' not in today_data.columns:
                premarket = today_data[
                    (today_data['datetime'].dt.hour < 9) | 
                    ((today_data['datetime'].dt.hour == 9) & (today_data['datetime'].dt.minute < 30))
                ]
                
                if not premarket.empty:
                    pm_high = premarket['high'].max()
                    pm_low = premarket['low'].min()
                    today_data['pm_high'] = pm_high
                    today_data['pm_low'] = pm_low
            
            # Calculate opening range
            market_open = today_data[
                (today_data['datetime'].dt.hour == 9) & 
                (today_data['datetime'].dt.minute >= 30)
            ]
            
            if len(market_open) < 5:
                logger.warning(f"Not enough data for opening range. Have {len(market_open)} minutes after market open.")
                return None
                
            # Calculate OR high, low, and mid if not already present
            if 'or_high' not in today_data.columns or 'or_low' not in today_data.columns:
                first_5min = market_open.iloc[:5]
                or_high = first_5min['high'].max()
                or_low = first_5min['low'].min()
                or_mid = (or_high + or_low) / 2
                
                # Add to dataframe
                today_data['or_high'] = or_high
                today_data['or_low'] = or_low
                today_data['or_mid'] = or_mid
                today_data['or_range'] = or_high - or_low
                today_data['or_range_pct'] = (or_high - or_low) / or_low * 100
                
                # Calculate minutes from market open
                first_candle = market_open.iloc[0]['datetime']
                today_data['minutes_from_open'] = (today_data['datetime'] - first_candle).dt.total_seconds() / 60
                
                # Mark if this is within opening range
                today_data['is_opening_range'] = today_data['minutes_from_open'] < 5
            
            # Log key levels
            logger.info(f"Opening Range - High: ${or_high:.2f}, Low: ${or_low:.2f}, Mid: ${or_mid:.2f}, Range: ${or_high - or_low:.2f}")
            
            if 'pm_high' in today_data.columns and 'pm_low' in today_data.columns:
                pm_high = today_data['pm_high'].iloc[0]
                pm_low = today_data['pm_low'].iloc[0]
                logger.info(f"Premarket - High: ${pm_high:.2f}, Low: ${pm_low:.2f}")
            
            if 'pd_high' in today_data.columns and 'pd_low' in today_data.columns:
                pd_high = today_data['pd_high'].iloc[0]
                pd_low = today_data['pd_low'].iloc[0]
                logger.info(f"Previous Day - High: ${pd_high:.2f}, Low: ${pd_low:.2f}")
            
            # Cache the data
            today_data.to_csv(cache_file, index=False)
            
            return today_data
            
        except Exception as e:
            logger.error(f"Error fetching current data: {str(e)}", exc_info=True)
            return None
    
    def check_for_setup(self, data):
        """
        Check if current market condition presents an ORB setup using hybrid reasoning
        
        Args:
            data: DataFrame with current market data
            
        Returns:
            Analysis dictionary or None if no setup found
        """
        # Only consider data after opening range
        post_or = data[data['minutes_from_open'] >= 5]
        
        if post_or.empty:
            return None
            
        # Get the latest data and make a proper copy
        latest = post_or.iloc[-1].copy()
        
        # Extract features for AI analysis with proper error handling
        try:
            pre_retest = post_or.iloc[-6:-1].copy() if len(post_or) > 5 else post_or.iloc[:-1].copy()
            features = self.extractor._extract_features(pre_retest, latest)
            
            # Check for NaN values in features
            if np.isnan(np.array(features).astype(float)).any():
                logger.warning("NaN values found in features - they will be handled safely")
                
            latest['features'] = features
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return None
        
        # Use hybrid reasoning engine for detailed analysis
        try:
            analysis = self.reasoning_engine.analyze_setup(latest, post_or, self.pattern_model)
        except Exception as e:
            logger.error(f"Error in reasoning engine: {str(e)}")
            return None
        
        if analysis['decision'] != 'NO_TRADE':
            return analysis
        else:
            return None
    
    def monitor_symbol(self, symbol, interval_seconds=60):
        """
        Monitor a symbol for ORB setups in real-time
        
        Args:
            symbol: Trading symbol to monitor
            interval_seconds: How often to check for new setups
        """
        logger.info(f"Starting to monitor {symbol} for ORB setups...")
        
        while True:
            try:
                # Check current market hours
                now = datetime.now(self.eastern)
                
                # Only run during market hours
                if not self._is_market_open(now):
                    sleep_time = min(self._time_to_next_market_session(now), 300)  # Max 5 minutes
                    logger.info(f"Market closed. Sleeping for {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    continue
                
                # Get current data
                data = self.get_current_data(symbol)
                
                if data is None:
                    logger.warning("No data available, waiting for next interval")
                    time.sleep(interval_seconds)
                    continue
                
                # Check active trades for exit conditions
                self._check_active_trades(data)
                
                # Check for new setup
                analysis = self.check_for_setup(data)
                
                if analysis:
                    logger.info("\n" + "=" * 80)
                    logger.info(f"ORB SETUP DETECTED: {analysis['decision']}")
                    logger.info("=" * 80)
                    
                    # Print detailed reasoning
                    logger.info("\nAI REASONING PROCESS:")
                    for line in analysis['reasoning']:
                        logger.info(line)
                    
                    # Print trade details
                    if analysis['decision'] in ('BUY', 'SELL'):
                        logger.info("\nTRADE DETAILS:")
                        logger.info(f"Entry: ${analysis['entry_price']:.2f}")
                        logger.info(f"Stop Loss: ${analysis['stop_loss']:.2f}")
                        logger.info(f"Target: ${analysis['target']:.2f}")
                        logger.info(f"Risk/Reward: {analysis['risk_reward']:.2f}")
                        logger.info(f"Confidence: {analysis['confidence']*100:.1f}%")
                        
                        # Add to active trades
                        self._add_active_trade(symbol, analysis, data.iloc[-1])
                    
                    logger.info("=" * 80 + "\n")
                
                time.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                logger.info("\nMonitoring stopped by user")
                # Save trade history before exiting
                self.save_trade_history()
                break
            
            except Exception as e:
                logger.error(f"Error: {str(e)}", exc_info=True)
                time.sleep(interval_seconds)
    
    def _add_active_trade(self, symbol, analysis, current_candle):
        """
        Add a new active trade to monitor
        
        Args:
            symbol: Trading symbol
            analysis: Analysis dictionary from reasoning engine
            current_candle: Current market data candle
            
        Returns:
            trade_id: ID of the new trade
        """
        trade_id = f"{symbol}-{current_candle['datetime'].strftime('%Y%m%d-%H%M%S')}"
        
        trade = {
            'id': trade_id,
            'symbol': symbol,
            'entry_time': current_candle['datetime'],
            'entry_price': analysis['entry_price'],
            'direction': analysis['decision'],
            'stop_loss': analysis['stop_loss'],
            'target': analysis['target'],
            'risk_reward': analysis['risk_reward'],
            'confidence': analysis['confidence'],
            'reasoning': analysis['reasoning'],
            'status': 'ACTIVE',
            'exit_time': None,
            'exit_price': None,
            'profit_loss': None,
            'outcome': None,
            'level_type': analysis.get('level_type', None),  # Store which level triggered the trade
            'candle_pattern': self._identify_candle_pattern(current_candle) # Store the entry candle pattern
        }
        
        self.active_trades[trade_id] = trade
        logger.info(f"New trade added: {trade_id}")
        
        return trade_id
    
    def _identify_candle_pattern(self, candle):
        """Identify the main candle pattern in a candle"""
        if candle.get('hammer', False):
            return 'hammer'
        elif candle.get('inverted_hammer', False) or candle.get('shooting_star', False):
            return 'inverted_hammer'
        elif candle.get('doji', False):
            return 'doji'
        elif candle.get('bullish_engulfing', False):
            return 'bullish_engulfing'
        elif candle.get('bearish_engulfing', False):
            return 'bearish_engulfing'
        
        # If no specific pattern, check general candle properties
        if 'open' in candle and 'close' in candle:
            if candle['close'] > candle['open']:
                return 'bullish_candle'
            elif candle['close'] < candle['open']:
                return 'bearish_candle'
        
        return 'none'
    
    def _check_active_trades(self, current_data):
        """
        Check if any active trades have hit their targets or stops
        
        Args:
            current_data: DataFrame with current market data
        """
        if not self.active_trades:
            return
            
        latest_candle = current_data.iloc[-1]
        current_price = latest_candle['close']
        current_time = latest_candle['datetime']
        
        trades_to_remove = []
        
        for trade_id, trade in self.active_trades.items():
            # Check if target or stop loss hit
            if trade['direction'] == 'BUY':
                if latest_candle['high'] >= trade['target']:
                    # Target hit
                    self._close_trade(trade_id, trade['target'], current_time, 'TARGET_HIT', 'WIN')
                    trades_to_remove.append(trade_id)
                elif latest_candle['low'] <= trade['stop_loss']:
                    # Stop loss hit
                    self._close_trade(trade_id, trade['stop_loss'], current_time, 'STOP_LOSS_HIT', 'LOSS')
                    trades_to_remove.append(trade_id)
            elif trade['direction'] == 'SELL':
                if latest_candle['low'] <= trade['target']:
                    # Target hit
                    self._close_trade(trade_id, trade['target'], current_time, 'TARGET_HIT', 'WIN')
                    trades_to_remove.append(trade_id)
                elif latest_candle['high'] >= trade['stop_loss']:
                    # Stop loss hit
                    self._close_trade(trade_id, trade['stop_loss'], current_time, 'STOP_LOSS_HIT', 'LOSS')
                    trades_to_remove.append(trade_id)
        
        # Remove closed trades from active list
        for trade_id in trades_to_remove:
            del self.active_trades[trade_id]
    
    def _close_trade(self, trade_id, exit_price, exit_time, exit_reason, outcome):
        """
        Close a trade and record results
        
        Args:
            trade_id: ID of the trade to close
            exit_price: Price at which trade was closed
            exit_time: Time when trade was closed
            exit_reason: Reason for closing the trade
            outcome: WIN or LOSS
        """
        trade = self.active_trades[trade_id]
        
        # Update trade details
        trade['exit_price'] = exit_price
        trade['exit_time'] = exit_time
        trade['status'] = 'CLOSED'
        trade['exit_reason'] = exit_reason
        trade['outcome'] = outcome
        
        # Calculate profit/loss
        if trade['direction'] == 'BUY':
            trade['profit_loss'] = (exit_price - trade['entry_price']) / trade['entry_price'] * 100
        else:  # SELL
            trade['profit_loss'] = (trade['entry_price'] - exit_price) / trade['entry_price'] * 100
        
        # Add to trade history
        self.trade_history.append(trade)
        
        # Update AI reasoning engine memory if using AI
        if hasattr(self.reasoning_engine, 'ai_engine') and self.reasoning_engine.ai_engine:
            self.reasoning_engine.ai_engine.update_trade_outcome(
                trade['entry_time'], 
                'success' if outcome == 'WIN' else 'failure'
            )
            
        # Update level statistics in the hybrid reasoning engine's level analyzer
        if hasattr(self.reasoning_engine, 'level_analyzer') and trade.get('level_type'):
            level_type = trade['level_type']
            self.reasoning_engine.update_retest_outcome(
                level_type, 
                trade['entry_time'].date(), 
                trade['entry_time'].strftime('%H:%M:%S'),
                'success' if outcome == 'WIN' else 'failure'
            )
        
        logger.info(f"Trade {trade_id} closed: {outcome}, P/L: {trade['profit_loss']:.2f}%")
    
    def save_trade_history(self, filepath=None):
        """
        Save trade history to CSV file
        
        Args:
            filepath: Path to save file, defaults to "{symbol}_trade_history.csv"
        """
        if not self.trade_history:
            logger.warning("No trades to save")
            return
        
        # Generate default filepath if not provided
        if filepath is None:
            symbols = set(trade['symbol'] for trade in self.trade_history)
            symbol_str = '_'.join(symbols)
            filepath = f"{symbol_str}_trade_history.csv"
            
        # Add data_dir to filepath
        filepath = os.path.join(self.data_dir, filepath)
            
        # Convert to DataFrame
        df = pd.DataFrame(self.trade_history)
        
        # Handle reasoning column (convert lists to strings)
        if 'reasoning' in df.columns:
            df['reasoning'] = df['reasoning'].apply(lambda x: str(x))
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        logger.info(f"Trade history saved to {filepath}")
        
        # Also save trade statistics
        self._save_trade_statistics(filepath.replace('.csv', '_stats.csv'))
    
    def _save_trade_statistics(self, filepath):
        """
        Calculate and save trade statistics
        
        Args:
            filepath: Path to save statistics file
        """
        if not self.trade_history:
            return
            
        # Calculate statistics
        stats = {}
        
        # Overall statistics
        wins = [t for t in self.trade_history if t['outcome'] == 'WIN']
        losses = [t for t in self.trade_history if t['outcome'] == 'LOSS']
        
        stats['total_trades'] = len(self.trade_history)
        stats['winning_trades'] = len(wins)
        stats['losing_trades'] = len(losses)
        stats['win_rate'] = len(wins) / len(self.trade_history) if self.trade_history else 0
        
        stats['total_profit'] = sum(t['profit_loss'] for t in wins) if wins else 0
        stats['total_loss'] = abs(sum(t['profit_loss'] for t in losses)) if losses else 0
        stats['net_profit'] = stats['total_profit'] - stats['total_loss']
        stats['profit_factor'] = stats['total_profit'] / stats['total_loss'] if stats['total_loss'] > 0 else 0
        
        stats['avg_win'] = sum(t['profit_loss'] for t in wins) / len(wins) if wins else 0
        stats['avg_loss'] = sum(t['profit_loss'] for t in losses) / len(losses) if losses else 0
        
        # Statistics by level type
        level_stats = {}
        level_types = set(t.get('level_type', 'unknown') for t in self.trade_history)
        
        for level_type in level_types:
            if level_type:
                level_trades = [t for t in self.trade_history if t.get('level_type') == level_type]
                level_wins = [t for t in level_trades if t['outcome'] == 'WIN']
                
                level_stats[level_type] = {
                    'total_trades': len(level_trades),
                    'winning_trades': len(level_wins),
                    'win_rate': len(level_wins) / len(level_trades) if level_trades else 0,
                    'avg_profit': sum(t['profit_loss'] for t in level_trades) / len(level_trades) if level_trades else 0
                }
        
        # Statistics by candle pattern
        pattern_stats = {}
        patterns = set(t.get('candle_pattern', 'none') for t in self.trade_history)
        
        for pattern in patterns:
            if pattern and pattern != 'none':
                pattern_trades = [t for t in self.trade_history if t.get('candle_pattern') == pattern]
                pattern_wins = [t for t in pattern_trades if t['outcome'] == 'WIN']
                
                pattern_stats[pattern] = {
                    'total_trades': len(pattern_trades),
                    'winning_trades': len(pattern_wins),
                    'win_rate': len(pattern_wins) / len(pattern_trades) if pattern_trades else 0,
                    'avg_profit': sum(t['profit_loss'] for t in pattern_trades) / len(pattern_trades) if pattern_trades else 0
                }
        
        # Save statistics
        with open(filepath, 'w') as f:
            f.write("=== OVERALL STATISTICS ===\n")
            for key, value in stats.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.2f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            
            f.write("\n=== STATISTICS BY LEVEL TYPE ===\n")
            for level_type, level_data in level_stats.items():
                f.write(f"\n{level_type.upper()}:\n")
                for key, value in level_data.items():
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.2f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
            
            f.write("\n=== STATISTICS BY CANDLE PATTERN ===\n")
            for pattern, pattern_data in pattern_stats.items():
                f.write(f"\n{pattern.upper()}:\n")
                for key, value in pattern_data.items():
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.2f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
        
        logger.info(f"Trade statistics saved to {filepath}")
    
    def _is_market_open(self, dt):
        """
        Check if the market is open at the given datetime
        
        Args:
            dt: Datetime to check
            
        Returns:
            bool: True if market is open, False otherwise
        """
        # Check if it's a weekday
        if dt.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            return False
        
        # Check if it's between 9:30 AM and 4:00 PM Eastern
        market_open = dt.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = dt.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= dt < market_close
    
    def _time_to_next_market_session(self, dt):
        """
        Calculate seconds until next market session
        
        Args:
            dt: Current datetime
            
        Returns:
            int: Seconds until next market session
        """
        # If it's before market open today
        today_open = dt.replace(hour=9, minute=30, second=0, microsecond=0)
        if dt.weekday() < 5 and dt < today_open:
            return (today_open - dt).total_seconds()
        
        # Calculate next weekday
        days_ahead = 1
        if dt.weekday() >= 4:  # Friday or Saturday
            days_ahead = 7 - dt.weekday()  # Next Monday
        
        next_day = dt + timedelta(days=days_ahead)
        next_open = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
        
        return (next_open - dt).total_seconds()

    def backtest(self, symbol, start_date, end_date, use_ai=False):
        """
        Run backtest on historical data
        
        Args:
            symbol: Trading symbol
            start_date: Start date for backtest
            end_date: End date for backtest
            use_ai: Whether to use AI for analysis (slower but more accurate)
            
        Returns:
            dict: Backtest results
        """
        logger.info(f"Running backtest for {symbol} from {start_date} to {end_date}")
        
        # Get historical data using the enhanced data collector
        try:
            df = self.data_collector.process_data_for_orb_analysis(symbol, start_date, end_date)
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return None
        
        # Get trading days
        trading_days = df['date'].unique()
        
        # Store trade results
        trades = []
        
        # Create a rule-based engine for faster backtesting
        rule_engine = RuleBasedReasoningEngine(data_dir=self.data_dir)
        
        # Store original reasoning engine
        original_engine = self.reasoning_engine
        
        # Use rule-based engine for backtesting unless AI is specifically requested
        if not use_ai:
            self.reasoning_engine = rule_engine
            logger.info("Using rule-based reasoning for faster backtesting")
        
        for day in trading_days:
            logger.info(f"Analyzing trading day: {day}")
            
            # Get data for this day
            day_data = df[df['date'] == day].copy()
            
            # Skip if we don't have opening range
            if 'or_high' not in day_data.columns:
                continue
                
            # Simulate the trading day
            for i in range(5, len(day_data)):
                # Create market snapshot up to current minute
                current_market = day_data.iloc[:i+1].copy()
                current_candle = current_market.iloc[-1].copy()  # Explicit copy
                
                # Skip if already in a trade for this day
                if any(t['date'] == day and t['status'] == 'ACTIVE' for t in trades):
                    # Check if we need to close any active trades
                    for trade in [t for t in trades if t['date'] == day and t['status'] == 'ACTIVE']:
                        if trade['direction'] == 'BUY':
                            if current_candle['high'] >= trade['target']:
                                trade['status'] = 'CLOSED'
                                trade['exit_price'] = trade['target']
                                trade['exit_time'] = current_candle['datetime']
                                trade['exit_reason'] = 'TARGET_HIT'
                                trade['outcome'] = 'WIN'
                                trade['profit_loss'] = (trade['target'] - trade['entry_price']) / trade['entry_price'] * 100
                            elif current_candle['low'] <= trade['stop_loss']:
                                trade['status'] = 'CLOSED'
                                trade['exit_price'] = trade['stop_loss']
                                trade['exit_time'] = current_candle['datetime']
                                trade['exit_reason'] = 'STOP_LOSS_HIT'
                                trade['outcome'] = 'LOSS'
                                trade['profit_loss'] = (trade['stop_loss'] - trade['entry_price']) / trade['entry_price'] * 100
                        elif trade['direction'] == 'SELL':
                            if current_candle['low'] <= trade['target']:
                                trade['status'] = 'CLOSED'
                                trade['exit_price'] = trade['target']
                                trade['exit_time'] = current_candle['datetime']
                                trade['exit_reason'] = 'TARGET_HIT'
                                trade['outcome'] = 'WIN'
                                trade['profit_loss'] = (trade['entry_price'] - trade['target']) / trade['entry_price'] * 100
                            elif current_candle['high'] >= trade['stop_loss']:
                                trade['status'] = 'CLOSED'
                                trade['exit_price'] = trade['stop_loss']
                                trade['exit_time'] = current_candle['datetime']
                                trade['exit_reason'] = 'STOP_LOSS_HIT'
                                trade['outcome'] = 'LOSS'
                                trade['profit_loss'] = (trade['entry_price'] - trade['stop_loss']) / trade['entry_price'] * 100
