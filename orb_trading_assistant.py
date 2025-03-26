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

logger = logging.getLogger(__name__)

class ORBTradingAssistant:
# Replace the __init__ method in orb_trading_assistant.py with this version:

    def __init__(self, api_key, model_path=None, use_ai=True, ai_model_path=None):
        """Initialize the trading assistant with AI integration"""
        self.client = RESTClient(api_key)
        self.eastern = pytz.timezone('US/Eastern')
        self.extractor = ORBPatternExtractor()
        
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
            use_ai=use_ai
        )
        
        # Trade tracking for continuous learning
        self.active_trades = {}
        self.trade_history = []
        
    def get_current_data(self, symbol, lookback_minutes=30):
        """Get current market data for a symbol"""
        now = datetime.now(self.eastern)
        end = now.strftime('%Y-%m-%d')
        start = (now - timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Fetch recent data
        aggs = []
        for a in self.client.list_aggs(
            symbol,
            1,
            "minute",
            start,
            end,
            limit=500,
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
            'vwap': getattr(agg, 'vwap', None),
            'transactions': getattr(agg, 'transactions', None)
        } for agg in aggs])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Add date and time columns
        df['date'] = df['datetime'].dt.date
        df['time'] = df['datetime'].dt.time
        
        # Get today's data
        today = now.date()
        today_data = df[df['date'] == today]
        
        if len(today_data) == 0:
            logger.warning("No data available for today")
            return None
        
        # Calculate opening range
        market_open = today_data[
            (today_data['datetime'].dt.hour == 9) & 
            (today_data['datetime'].dt.minute >= 30)
        ]
        
        if len(market_open) < 5:
            logger.warning(f"Not enough data for opening range. Have {len(market_open)} minutes after market open.")
            return None
            
        # Calculate OR high and low
        first_5min = market_open.iloc[:5]
        or_high = first_5min['high'].max()
        or_low = first_5min['low'].min()
        
        # Add to dataframe
        today_data['or_high'] = or_high
        today_data['or_low'] = or_low
        today_data['or_range'] = or_high - or_low
        
        # Calculate minutes from market open
        first_candle = market_open.iloc[0]['datetime']
        today_data['minutes_from_open'] = (today_data['datetime'] - first_candle).dt.total_seconds() / 60
        
        # Mark if this is within opening range
        today_data['is_opening_range'] = today_data['minutes_from_open'] < 5
        
        # Add technical indicators and enhanced features
        today_data = self.extractor.add_technical_indicators(today_data)
        
        logger.info(f"Opening Range - High: {or_high:.2f}, Low: {or_low:.2f}, Range: {or_high - or_low:.2f}")
        
        return today_data
    
    def check_for_setup(self, data):
        """Check if current market condition presents an ORB setup using hybrid reasoning"""
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
            analysis = self.reasoning_engine.analyze_setup(latest, data, self.pattern_model)
        except Exception as e:
            logger.error(f"Error in reasoning engine: {str(e)}")
            return None
        
        if analysis['decision'] != 'NO_TRADE':
            return analysis
        else:
            return None
    
    def monitor_symbol(self, symbol, interval_seconds=60):
        """Monitor a symbol for ORB setups"""
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
                break
            
            except Exception as e:
                logger.error(f"Error: {str(e)}", exc_info=True)
                time.sleep(interval_seconds)
    
    def _add_active_trade(self, symbol, analysis, current_candle):
        """Add a new active trade to monitor"""
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
            'outcome': None
        }
        
        self.active_trades[trade_id] = trade
        logger.info(f"New trade added: {trade_id}")
        
        return trade_id
    
    def _check_active_trades(self, current_data):
        """Check if any active trades have hit their targets or stops"""
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
        """Close a trade and record results"""
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
                'WIN' if outcome == 'WIN' else 'LOSS'
            )
        
        logger.info(f"Trade {trade_id} closed: {outcome}, P/L: {trade['profit_loss']:.2f}%")
    
    def save_trade_history(self, filepath="trade_history.csv"):
        """Save trade history to CSV file"""
        if not self.trade_history:
            logger.warning("No trades to save")
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(self.trade_history)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        logger.info(f"Trade history saved to {filepath}")
    
    def _is_market_open(self, dt):
        """Check if the market is open at the given datetime"""
        # Check if it's a weekday
        if dt.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            return False
        
        # Check if it's between 9:30 AM and 4:00 PM Eastern
        market_open = dt.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = dt.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= dt < market_close
    
    def _time_to_next_market_session(self, dt):
        """Calculate seconds until next market session"""
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

    def backtest(self, symbol, start_date, end_date, api_key=None):
        """Run backtest on historical data"""
        logger.info(f"Running backtest for {symbol} from {start_date} to {end_date}")
        
        # Use instance api_key if none provided
        if api_key is None:
            # Fix for RESTClient - get the API key directly
            try:
                # Try to get API key from client object
                api_key = self.client._api_key  # Some versions use this
            except AttributeError:
                try:
                    api_key = self.client.api_key  # Some versions use this
                except AttributeError:
                    # If all else fails, we need the user to provide it
                    logger.error("Could not retrieve API key from client, please provide it explicitly")
                    return None
        
        # Create a collector for fetching data
        from data_collector import DataCollector
        collector = DataCollector(api_key)
        
        # Get historical data
        df = collector.fetch_historical_data(symbol, start_date, end_date)
        df = collector.identify_opening_ranges(df)
        
        # Add technical indicators
        df = self.extractor.add_technical_indicators(df)
        
        # Get trading days
        trading_days = df['date'].unique()
        
        # Store trade results
        trades = []
        
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
                current_candle = current_market.iloc[-1].copy()  # Explicit copy to avoid SettingWithCopyWarning
                
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
                    
                    # Skip to next candle
                    continue
                
                # Extract features with proper error handling
                try:
                    # Use proper copy to avoid pandas warnings
                    pre_retest = current_market.iloc[-6:-1].copy() if len(current_market) > 5 else current_market.iloc[:-1].copy()
                    
                    # Extract features with NaN handling
                    features = self.extractor._extract_features(pre_retest, current_candle)
                    
                    # Check for NaN values in features
                    import numpy as np
                    if np.isnan(np.array(features).astype(float)).any():
                        logger.warning("NaN values found in features - these will be handled by pattern model")
                    
                    current_candle['features'] = features
                except Exception as e:
                    logger.error(f"Error extracting features: {str(e)}")
                    continue
                
                # Check for setup - use only rule-based engine for backtesting to speed things up
                temp_reasoning_engine = self.reasoning_engine
                rule_engine = RuleBasedReasoningEngine()
                self.reasoning_engine = rule_engine

                try:
                    # Make sure we have the pattern_model but handle errors gracefully
                    if self.pattern_model:
                        analysis = self.reasoning_engine.analyze_setup(current_candle, current_market, self.pattern_model)
                    else:
                        analysis = self.reasoning_engine.analyze_setup(current_candle, current_market, None)
                except Exception as e:
                    logger.error(f"Error in setup analysis: {str(e)}")
                    self.reasoning_engine = temp_reasoning_engine  # Restore original engine
                    continue

                self.reasoning_engine = temp_reasoning_engine  # Restore original engine
                
                if analysis['decision'] != 'NO_TRADE' and analysis['confidence'] > 0.5:
                    # Create trade record
                    trade = {
                        'date': day,
                        'entry_time': current_candle['datetime'],
                        'entry_price': analysis['entry_price'],
                        'direction': analysis['decision'],
                        'stop_loss': analysis['stop_loss'],
                        'target': analysis['target'],
                        'risk_reward': analysis['risk_reward'],
                        'confidence': analysis['confidence'],
                        'reasoning': analysis['reasoning'] if isinstance(analysis['reasoning'], list) else [analysis['reasoning']],
                        'status': 'ACTIVE',
                        'exit_time': None,
                        'exit_price': None,
                        'exit_reason': None,
                        'outcome': None,
                        'profit_loss': None
                    }
                    
                    trades.append(trade)
                    logger.info(f"Backtest trade triggered: {day} {current_candle['datetime']} {trade['direction']}")
        
        # Close any trades that didn't hit target or stop (simulate close at end of day)
        for trade in trades:
            if trade['status'] == 'ACTIVE':
                day_data = df[df['date'] == trade['date']]
                last_price = day_data.iloc[-1]['close']
                
                trade['status'] = 'CLOSED'
                trade['exit_price'] = last_price
                trade['exit_time'] = day_data.iloc[-1]['datetime']
                trade['exit_reason'] = 'EOD_CLOSE'
                
                if trade['direction'] == 'BUY':
                    profit_pct = (last_price - trade['entry_price']) / trade['entry_price'] * 100
                    trade['profit_loss'] = profit_pct
                    trade['outcome'] = 'WIN' if profit_pct > 0 else 'LOSS'
                else:  # SELL
                    profit_pct = (trade['entry_price'] - last_price) / trade['entry_price'] * 100
                    trade['profit_loss'] = profit_pct
                    trade['outcome'] = 'WIN' if profit_pct > 0 else 'LOSS'
        
        # Calculate backtest results
        if not trades:
            logger.warning("No trades found in backtest")
            return {
                'trades': [],
                'win_rate': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'total_profit': 0
            }
            
        # Calculate statistics
        wins = [t for t in trades if t['outcome'] == 'WIN']
        losses = [t for t in trades if t['outcome'] == 'LOSS']
        
        win_rate = len(wins) / len(trades) if trades else 0
        
        total_profit = sum(t['profit_loss'] for t in wins) if wins else 0
        total_loss = abs(sum(t['profit_loss'] for t in losses)) if losses else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        avg_win = sum(t['profit_loss'] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t['profit_loss'] for t in losses) / len(losses) if losses else 0
        
        # Save backtest results to CSV
        backtest_df = pd.DataFrame(trades)
        backtest_df.to_csv(f"{symbol}_backtest_results.csv", index=False)
        
        # Log backtest results
        logger.info(f"Backtest completed with {len(trades)} trades")
        logger.info(f"Win rate: {win_rate*100:.2f}%")
        logger.info(f"Profit factor: {profit_factor:.2f}")
        logger.info(f"Average win: {avg_win:.2f}%")
        logger.info(f"Average loss: {avg_loss:.2f}%")
        logger.info(f"Total net profit: {total_profit-total_loss:.2f}%")
        
        # Return results
        return {
            'trades': trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_profit': total_profit - total_loss
        }