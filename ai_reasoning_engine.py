import os
import numpy as np
import pandas as pd
import logging
from pattern_model import PatternModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re
from datetime import datetime, timedelta
import pickle
from collections import defaultdict

logger = logging.getLogger(__name__)

class MarketLevelAnalyzer:
    """
    Analyzes and stores information about price action at key market levels:
    - Premarket high/low
    - Opening Range (5-min) high/low
    - Previous day high/low
    """
    
    def __init__(self, data_dir="market_levels_data"):
        """Initialize the market level analyzer"""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Create storage for market level data
        self.level_data = {
            'premarket_high': defaultdict(list),
            'premarket_low': defaultdict(list),
            'orb_high': defaultdict(list),
            'orb_low': defaultdict(list),
            'prev_day_high': defaultdict(list),
            'prev_day_low': defaultdict(list),
        }
        
        # Statistics storage
        self.stats = {
            'premarket_high': {'success': 0, 'failure': 0, 'patterns': {}},
            'premarket_low': {'success': 0, 'failure': 0, 'patterns': {}},
            'orb_high': {'success': 0, 'failure': 0, 'patterns': {}},
            'orb_low': {'success': 0, 'failure': 0, 'patterns': {}},
            'prev_day_high': {'success': 0, 'failure': 0, 'patterns': {}},
            'prev_day_low': {'success': 0, 'failure': 0, 'patterns': {}},
        }
        
        # Load existing data if available
        self._load_data()
    
    def _load_data(self):
        """Load existing market level data if available"""
        stats_file = os.path.join(self.data_dir, "level_stats.pkl")
        level_data_file = os.path.join(self.data_dir, "level_data.pkl")
        
        if os.path.exists(stats_file):
            try:
                with open(stats_file, 'rb') as f:
                    self.stats = pickle.load(f)
                logger.info("Successfully loaded market level statistics")
            except Exception as e:
                logger.error(f"Failed to load market level statistics: {str(e)}")
        
        if os.path.exists(level_data_file):
            try:
                with open(level_data_file, 'rb') as f:
                    self.level_data = pickle.load(f)
                logger.info("Successfully loaded market level data")
            except Exception as e:
                logger.error(f"Failed to load market level data: {str(e)}")
    
    def save_data(self):
        """Save market level data to disk"""
        stats_file = os.path.join(self.data_dir, "level_stats.pkl")
        level_data_file = os.path.join(self.data_dir, "level_data.pkl")
        
        try:
            with open(stats_file, 'wb') as f:
                pickle.dump(self.stats, f)
            
            with open(level_data_file, 'wb') as f:
                pickle.dump(self.level_data, f)
            
            logger.info("Successfully saved market level data and statistics")
        except Exception as e:
            logger.error(f"Failed to save market level data: {str(e)}")
    
    def record_retest(self, level_type, date, candle_data, outcome):
        """
        Record a retest event of a specific market level
        
        Args:
            level_type: Type of level ('premarket_high', 'orb_low', etc.)
            date: Date of the event
            candle_data: Dictionary with candle data at retest (OHLCV, patterns)
            outcome: 'success' or 'failure' based on what happened after the retest
        """
        if level_type not in self.level_data:
            logger.warning(f"Unknown level type: {level_type}")
            return
        
        # Format date as string if it's a datetime object
        date_str = date.strftime('%Y-%m-%d') if isinstance(date, datetime) else date
        
        # Add data to the appropriate level
        self.level_data[level_type][date_str].append({
            'time': candle_data.get('time', ''),
            'open': candle_data.get('open', 0),
            'high': candle_data.get('high', 0),
            'low': candle_data.get('low', 0),
            'close': candle_data.get('close', 0),
            'volume': candle_data.get('volume', 0),
            'is_hammer': candle_data.get('hammer', False),
            'is_inverted_hammer': candle_data.get('inverted_hammer', False),
            'is_doji': candle_data.get('doji', False),
            'is_bullish_engulfing': candle_data.get('bullish_engulfing', False),
            'is_bearish_engulfing': candle_data.get('bearish_engulfing', False),
            'body_size_pct': candle_data.get('body_size_pct', 0),
            'upper_wick_pct': candle_data.get('upper_wick_pct', 0),
            'lower_wick_pct': candle_data.get('lower_wick_pct', 0),
            'outcome': outcome
        })
        
        # Update statistics
        self.stats[level_type][outcome] += 1
        
        # Update pattern statistics
        for pattern in ['hammer', 'inverted_hammer', 'doji', 'bullish_engulfing', 'bearish_engulfing']:
            if candle_data.get(pattern, False):
                if pattern not in self.stats[level_type]['patterns']:
                    self.stats[level_type]['patterns'][pattern] = {'success': 0, 'failure': 0}
                self.stats[level_type]['patterns'][pattern][outcome] += 1
        
        # Save data after each update
        self.save_data()
    
    def get_level_stats(self, level_type=None):
        """
        Get statistics for a specific level type or all levels
        
        Args:
            level_type: Optional level type to filter stats
            
        Returns:
            Dictionary with level statistics
        """
        if level_type:
            if level_type in self.stats:
                return self._calculate_level_stats(level_type)
            else:
                logger.warning(f"Unknown level type: {level_type}")
                return {}
        else:
            # Return stats for all levels
            all_stats = {}
            for level in self.stats:
                all_stats[level] = self._calculate_level_stats(level)
            return all_stats
    
    def _calculate_level_stats(self, level_type):
        """Calculate detailed statistics for a specific level type"""
        stats = self.stats[level_type]
        total = stats['success'] + stats['failure']
        
        if total == 0:
            win_rate = 0
        else:
            win_rate = stats['success'] / total * 100
        
        # Calculate pattern win rates
        pattern_stats = {}
        for pattern, outcomes in stats['patterns'].items():
            pattern_total = outcomes['success'] + outcomes['failure']
            if pattern_total > 0:
                pattern_win_rate = outcomes['success'] / pattern_total * 100
                pattern_stats[pattern] = {
                    'total': pattern_total,
                    'success': outcomes['success'],
                    'failure': outcomes['failure'],
                    'win_rate': pattern_win_rate
                }
        
        # Get recent examples
        recent_examples = []
        for date in sorted(self.level_data[level_type].keys(), reverse=True)[:10]:  # Get 10 most recent days
            for entry in self.level_data[level_type][date]:
                recent_examples.append({
                    'date': date,
                    'time': entry.get('time', ''),
                    'outcome': entry.get('outcome', ''),
                    'pattern': self._identify_main_pattern(entry)
                })
        
        return {
            'total_retests': total,
            'success_count': stats['success'],
            'failure_count': stats['failure'],
            'win_rate': win_rate,
            'pattern_stats': pattern_stats,
            'recent_examples': recent_examples[:5]  # Limit to 5 examples
        }
    
    def _identify_main_pattern(self, candle_data):
        """Identify the main pattern in a candle"""
        if candle_data.get('is_hammer', False):
            return 'hammer'
        elif candle_data.get('is_inverted_hammer', False):
            return 'inverted_hammer'
        elif candle_data.get('is_bullish_engulfing', False):
            return 'bullish_engulfing'
        elif candle_data.get('is_bearish_engulfing', False):
            return 'bearish_engulfing'
        elif candle_data.get('is_doji', False):
            return 'doji'
        else:
            return 'none'
    
    def find_similar_retests(self, level_type, candle_data, max_results=5):
        """
        Find similar historical retests based on candle patterns and properties
        
        Args:
            level_type: Type of level ('premarket_high', 'orb_low', etc.)
            candle_data: Dictionary with current candle data
            max_results: Maximum number of similar retests to return
            
        Returns:
            List of similar historical retests
        """
        if level_type not in self.level_data:
            logger.warning(f"Unknown level type: {level_type}")
            return []
        
        similar_retests = []
        
        # Get the main pattern in current candle
        current_pattern = self._identify_main_pattern(candle_data)
        
        # Convert candle properties to numeric values for comparison
        current_props = {
            'body_size_pct': candle_data.get('body_size_pct', 0),
            'upper_wick_pct': candle_data.get('upper_wick_pct', 0),
            'lower_wick_pct': candle_data.get('lower_wick_pct', 0)
        }
        
        # Search through historical data
        for date, entries in self.level_data[level_type].items():
            for entry in entries:
                # First, check if the pattern matches
                entry_pattern = self._identify_main_pattern(entry)
                if entry_pattern == current_pattern or current_pattern == 'none':
                    # Calculate similarity score based on candle properties
                    sim_score = self._calculate_similarity(current_props, {
                        'body_size_pct': entry.get('body_size_pct', 0),
                        'upper_wick_pct': entry.get('upper_wick_pct', 0),
                        'lower_wick_pct': entry.get('lower_wick_pct', 0)
                    })
                    
                    similar_retests.append({
                        'date': date,
                        'time': entry.get('time', ''),
                        'pattern': entry_pattern,
                        'outcome': entry.get('outcome', ''),
                        'similarity_score': sim_score,
                        'candle_data': entry
                    })
        
        # Sort by similarity score
        similar_retests.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return similar_retests[:max_results]
    
    def _calculate_similarity(self, props1, props2):
        """Calculate similarity between two sets of candle properties"""
        # Simple Euclidean distance-based similarity
        squared_diff = 0
        for key in props1:
            if key in props2:
                squared_diff += (props1[key] - props2[key]) ** 2
        
        # Convert to similarity score (0-1, higher is more similar)
        distance = np.sqrt(squared_diff)
        similarity = 1 / (1 + distance)
        
        return similarity


class LocalAIReasoningEngine:
    """
    A reasoning engine that uses locally downloaded DeepSeek model for enhanced trading decisions.
    """
    
    def __init__(self, model_path="deepseek-ai/deepseek-coder-6.7b-instruct", data_dir="market_data"):
        """Initialize the reasoning engine with DeepSeek model"""
        self.model_path = model_path
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        logger.info(f"Initializing local AI reasoning engine with model: {model_path}")
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        self.initialized = False
        
        # Setup memory system for continuous learning
        self.memory = []
        self.max_memories = 100
        
        # Initialize market level analyzer
        self.level_analyzer = MarketLevelAnalyzer(os.path.join(data_dir, "level_analysis"))
        
    def load_model(self):
        """Load the model (lazy initialization to save memory)"""
        if self.initialized:
            return
            
        logger.info("Loading DeepSeek model...")
        try:
            # Set trust_remote_code for DeepSeek models
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Load with conservative settings
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=False
            )
            self.initialized = True
            logger.info("DeepSeek model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading DeepSeek model: {str(e)}")
            raise
    
    def analyze_setup(self, current_data, market_data=None, pattern_results=None):
        """Use AI to analyze trading setup in detail"""
        # Ensure model is loaded
        self.load_model()
        
        # Prepare market data in a format the model can understand
        market_summary = self._prepare_market_data(current_data, market_data)
        pattern_summary = self._prepare_pattern_analysis(pattern_results)
        memory_summary = self._get_relevant_memories(current_data)
        
        # Get level statistics and similar retests
        level_stats_summary = self._get_level_statistics(current_data)
        
        # Construct the prompt
        prompt = self._construct_prompt(market_summary, pattern_summary, memory_summary, level_stats_summary)
        
        # Get model response
        response = self._get_model_inference(prompt)
        
        # Parse the response
        result = self._parse_analysis(response)
        
        # Store this analysis in memory if it's a valid setup
        if result['decision'] != 'NO_TRADE' and result['confidence'] > 0.5:
            self._store_in_memory(current_data, result)
        
        return result
    
    def _prepare_market_data(self, current_data, market_data):
        """Format market data for the model prompt"""
        summary = []
        
        # Add basic market information
        summary.append(f"Current time: {current_data['datetime']}")
        summary.append(f"Current price: ${current_data['close']:.2f}")
        
        # Add opening range information
        if 'or_high' in current_data and 'or_low' in current_data:
            summary.append(f"Opening Range High: ${current_data['or_high']:.2f}")
            summary.append(f"Opening Range Low: ${current_data['or_low']:.2f}")
            summary.append(f"Opening Range Size: ${(current_data['or_high'] - current_data['or_low']):.2f} ({(current_data['or_high'] - current_data['or_low']) / current_data['or_low'] * 100:.2f}%)")
        
        # Add premarket levels if available
        if 'pm_high' in current_data and 'pm_low' in current_data:
            summary.append(f"Premarket High: ${current_data['pm_high']:.2f}")
            summary.append(f"Premarket Low: ${current_data['pm_low']:.2f}")

        # Add previous day levels if available
        if 'pd_high' in current_data and 'pd_low' in current_data:
            summary.append(f"Previous Day High: ${current_data['pd_high']:.2f}")
            summary.append(f"Previous Day Low: ${current_data['pd_low']:.2f}")
        
        # Add technical indicators
        if 'rsi' in current_data:
            summary.append(f"RSI: {current_data['rsi']:.1f}")
        if 'rel_volume' in current_data:
            summary.append(f"Relative Volume: {current_data['rel_volume']:.1f}x average")
        if 'macd' in current_data:
            summary.append(f"MACD: {current_data['macd']:.3f}")
            if 'macd_signal' in current_data:
                summary.append(f"MACD Signal: {current_data['macd_signal']:.3f}")
                summary.append(f"MACD Histogram: {current_data['macd'] - current_data['macd_signal']:.3f}")
        if 'adx' in current_data:
            summary.append(f"ADX: {current_data['adx']:.1f}")
            
        # Add price relative to EMAs
        if 'ema9' in current_data and not pd.isna(current_data['ema9']):
            summary.append(f"Price to EMA9: {(current_data['close'] / current_data['ema9'] - 1) * 100:.2f}%")
        if 'ema20' in current_data and not pd.isna(current_data['ema20']):
            summary.append(f"Price to EMA20: {(current_data['close'] / current_data['ema20'] - 1) * 100:.2f}%")
        if 'ema50' in current_data and not pd.isna(current_data['ema50']):
            summary.append(f"Price to EMA50: {(current_data['close'] / current_data['ema50'] - 1) * 100:.2f}%")
        
        # Add candle information with detailed pattern analysis
        summary.append(f"Current candle: Open=${current_data['open']:.2f}, High=${current_data['high']:.2f}, Low=${current_data['low']:.2f}, Close=${current_data['close']:.2f}")
        
        # Calculate and add candle properties
        if 'high' in current_data and 'low' in current_data and 'open' in current_data and 'close' in current_data:
            body_size = abs(current_data['close'] - current_data['open'])
            total_range = current_data['high'] - current_data['low']
            
            if total_range > 0:
                body_size_pct = body_size / total_range
                upper_wick = current_data['high'] - max(current_data['open'], current_data['close'])
                lower_wick = min(current_data['open'], current_data['close']) - current_data['low']
                
                upper_wick_pct = upper_wick / total_range if total_range > 0 else 0
                lower_wick_pct = lower_wick / total_range if total_range > 0 else 0
                
                summary.append(f"Candle body size: {body_size_pct:.2f} of range")
                summary.append(f"Upper wick: {upper_wick_pct:.2f} of range")
                summary.append(f"Lower wick: {lower_wick_pct:.2f} of range")
                
                # Store these calculations for later use
                current_data['body_size_pct'] = body_size_pct
                current_data['upper_wick_pct'] = upper_wick_pct
                current_data['lower_wick_pct'] = lower_wick_pct
            
        # Add pattern identification
        if 'hammer' in current_data and current_data['hammer']:
            summary.append("Candle pattern: Hammer (bullish)")
        if 'inverted_hammer' in current_data and current_data['inverted_hammer']:
            summary.append("Candle pattern: Inverted Hammer/Shooting Star")
        if 'doji' in current_data and current_data['doji']:
            summary.append("Candle pattern: Doji (indecision)")
        if 'bullish_engulfing' in current_data and current_data['bullish_engulfing']:
            summary.append("Candle pattern: Bullish Engulfing")
        if 'bearish_engulfing' in current_data and current_data['bearish_engulfing']:
            summary.append("Candle pattern: Bearish Engulfing")
            
        # Add distance from key levels
        if 'or_high' in current_data:
            dist_from_or_high = ((current_data['close'] / current_data['or_high']) - 1) * 100
            summary.append(f"Distance from OR high: {dist_from_or_high:.2f}%")
            current_data['dist_from_or_high'] = dist_from_or_high
            
        if 'or_low' in current_data:
            dist_from_or_low = ((current_data['close'] / current_data['or_low']) - 1) * 100
            summary.append(f"Distance from OR low: {dist_from_or_low:.2f}%")
            current_data['dist_from_or_low'] = dist_from_or_low
            
        if 'pm_high' in current_data:
            dist_from_pm_high = ((current_data['close'] / current_data['pm_high']) - 1) * 100
            summary.append(f"Distance from Premarket high: {dist_from_pm_high:.2f}%")
            
        if 'pm_low' in current_data:
            dist_from_pm_low = ((current_data['close'] / current_data['pm_low']) - 1) * 100
            summary.append(f"Distance from Premarket low: {dist_from_pm_low:.2f}%")
        
        # Add market context from historical data if available
        if market_data is not None and not market_data.empty:
            # Detect breakout status
            post_or = market_data[market_data['minutes_from_open'] >= 5].copy() if 'minutes_from_open' in market_data.columns else pd.DataFrame()
            
            if not post_or.empty and 'or_high' in current_data and 'or_low' in current_data:
                or_high = current_data['or_high'] 
                or_low = current_data['or_low']
                
                # Check for OR high breakout
                if post_or['high'].max() > or_high:
                    breakout_rows = post_or[post_or['high'] > or_high]
                    if not breakout_rows.empty:
                        time_of_breakout = breakout_rows.iloc[0]['datetime']
                        summary.append(f"Price broke above Opening Range High at {time_of_breakout}")
                        
                        # Check if we've retested the breakout level
                        after_breakout = post_or.loc[post_or['datetime'] > time_of_breakout]
                        if not after_breakout.empty:
                            # Look for retest (price coming back close to the OR high)
                            retest_threshold = 0.2  # % distance for retest
                            min_dist_pct = min(abs((row['low'] / or_high - 1) * 100) for _, row in after_breakout.iterrows())
                            
                            if min_dist_pct < retest_threshold:
                                summary.append("Price has retested OR high (potential long setup)")
                                
                                # Mark current candle as a retest if it's close to OR high
                                current_dist_pct = abs((current_data['low'] / or_high - 1) * 100)
                                if current_dist_pct < retest_threshold:
                                    summary.append("CURRENT CANDLE: Retesting OR high")
                                    
                                    # Check market hours
                                    is_market_hours = True  # Set based on your market hours logic
                                    
                                    # Record this retest for learning
                                    if is_market_hours:
                                        self._record_retest(current_data, 'orb_high')
                
                # Check for OR low breakout
                if post_or['low'].min() < or_low:
                    breakout_rows = post_or[post_or['low'] < or_low]
                    if not breakout_rows.empty:
                        time_of_breakout = breakout_rows.iloc[0]['datetime']
                        summary.append(f"Price broke below Opening Range Low at {time_of_breakout}")
                        
                        # Check if we've retested the breakout level
                        after_breakout = post_or.loc[post_or['datetime'] > time_of_breakout]
                        if not after_breakout.empty:
                            # Look for retest (price coming back close to the OR low)
                            retest_threshold = 0.2  # % distance for retest
                            min_dist_pct = min(abs((row['high'] / or_low - 1) * 100) for _, row in after_breakout.iterrows())
                            
                            if min_dist_pct < retest_threshold:
                                summary.append("Price has retested OR low (potential short setup)")
                                
                                # Mark current candle as a retest if it's close to OR low
                                current_dist_pct = abs((current_data['high'] / or_low - 1) * 100)
                                if current_dist_pct < retest_threshold:
                                    summary.append("CURRENT CANDLE: Retesting OR low")
                                    
                                    # Check market hours
                                    is_market_hours = True  # Set based on your market hours logic
                                    
                                    # Record this retest for learning
                                    if is_market_hours:
                                        self._record_retest(current_data, 'orb_low')
            
            # Check for premarket high/low retests
            if 'pm_high' in current_data and 'pm_low' in current_data:
                pm_high = current_data['pm_high']
                pm_low = current_data['pm_low']
                
                # Check if current candle is retesting premarket high
                if abs((current_data['low'] / pm_high - 1) * 100) < 0.2:
                    summary.append("CURRENT CANDLE: Retesting Premarket high")
                    self._record_retest(current_data, 'premarket_high')
                
                # Check if current candle is retesting premarket low
                if abs((current_data['high'] / pm_low - 1) * 100) < 0.2:
                    summary.append("CURRENT CANDLE: Retesting Premarket low")
                    self._record_retest(current_data, 'premarket_low')
            
            # Check for previous day high/low retests
            if 'pd_high' in current_data and 'pd_low' in current_data:
                pd_high = current_data['pd_high']
                pd_low = current_data['pd_low']
                
                # Check if current candle is retesting previous day high
                if abs((current_data['low'] / pd_high - 1) * 100) < 0.2:
                    summary.append("CURRENT CANDLE: Retesting Previous Day high")
                    self._record_retest(current_data, 'prev_day_high')
                
                # Check if current candle is retesting previous day low
                if abs((current_data['high'] / pd_low - 1) * 100) < 0.2:
                    summary.append("CURRENT CANDLE: Retesting Previous Day low")
                    self._record_retest(current_data, 'prev_day_low')
        
        return "\n".join(summary)
    
    def _record_retest(self, current_data, level_type):
        """Record a retest event for learning"""
        # We don't know the outcome yet, but we'll record it for now
        # The outcome will be updated later when we know what happens after the retest
        
        # Prepare candle data
        candle_data = {
            'time': current_data.get('datetime', datetime.now()).strftime('%H:%M:%S') if isinstance(current_data.get('datetime'), datetime) else current_data.get('datetime', ''),
            'open': current_data.get('open', 0),
            'high': current_data.get('high', 0),
            'low': current_data.get('low', 0),
            'close': current_data.get('close', 0),
            'volume': current_data.get('volume', 0),
            'hammer': current_data.get('hammer', False),
            'inverted_hammer': current_data.get('inverted_hammer', False),
            'doji': current_data.get('doji', False),
            'bullish_engulfing': current_data.get('bullish_engulfing', False),
            'bearish_engulfing': current_data.get('bearish_engulfing', False),
            'body_size_pct': current_data.get('body_size_pct', 0),
            'upper_wick_pct': current_data.get('upper_wick_pct', 0),
            'lower_wick_pct': current_data.get('lower_wick_pct', 0)
        }
        
        # Get date
        date = current_data.get('datetime', datetime.now())
        if not isinstance(date, datetime):
            try:
                date = pd.to_datetime(date)
            except:
                date = datetime.now()
        
        # For now, set outcome to 'pending'
        # We'll update this later based on what happens after the retest
        self.level_analyzer.record_retest(level_type, date, candle_data, 'pending')

    def update_trade_outcome(self, entry_time, outcome):
        """Update memory with actual trade outcome"""
        # Find the memory item for this trade
        for mem in self.memory:
            if mem['timestamp'].strftime('%Y-%m-%d %H:%M') == entry_time.strftime('%Y-%m-%d %H:%M'):
                mem['actual_outcome'] = outcome
                logger.info(f"Updated memory with outcome: {outcome}")
                return True
        
        return False

class RuleBasedReasoningEngine:
    """
    A reasoning engine that applies rule-based decision-making to ORB pattern trading.
    This engine checks all criteria for valid ORB setups and provides a explanation
    of trading decisions based on predefined rules.
    """
    
    def __init__(self, data_dir="market_data"):
        """Initialize the rule-based reasoning engine"""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize market level analyzer - shared with AI engine if needed
        self.level_analyzer = MarketLevelAnalyzer(os.path.join(data_dir, "level_analysis"))
    
    def analyze_setup(self, current_data, market_data=None, pattern_results=None):
        """
        Analyze a potential ORB setup using rule-based logic
        """
        # Initialize analysis structure
        analysis = {
            'decision': 'NO_TRADE',
            'confidence': 0.0,
            'reasoning': [],
            'criteria_met': {},
            'entry_price': None,
            'stop_loss': None,
            'target': None,
            'risk_reward': None
        }
        
        # Check if we have opening range data
        if 'or_high' not in current_data or 'or_low' not in current_data:
            analysis['reasoning'].append("❌ No opening range data available")
            return analysis
            
        # Extract key values
        or_high = current_data['or_high']
        or_low = current_data['or_low']
        current_price = current_data['close']
        
        # Step 1: Check if market has opened and 5-min OR is established
        if market_data is not None:
            opening_range = market_data[market_data['is_opening_range'] == True]
            if len(opening_range) < 5:
                analysis['reasoning'].append("❌ Complete 5-minute opening range not established yet")
                analysis['criteria_met']['complete_or'] = False
                return analysis
            
            analysis['criteria_met']['complete_or'] = True
            analysis['reasoning'].append("✅ 5-minute opening range established")
            
            # Add OR details to reasoning
            analysis['reasoning'].append(f"Opening Range High: ${or_high:.2f}")
            analysis['reasoning'].append(f"Opening Range Low: ${or_low:.2f}")
            analysis['reasoning'].append(f"Opening Range Size: ${(or_high - or_low):.2f} ({(or_high - or_low) / or_low * 100:.2f}%)")
        
        # Step 2: Look for breakout above OR high or below OR low
        breakout_up = False
        breakout_down = False
        post_or = None
        
        if market_data is not None:
            post_or = market_data[market_data['minutes_from_open'] >= 5]
            if not post_or.empty:
                breakout_up = post_or['high'].max() > or_high
                breakout_down = post_or['low'].min() < or_low
                
                if breakout_up:
                    analysis['criteria_met']['breakout'] = True
                    analysis['reasoning'].append(f"✅ Price broke above OR high (${or_high:.2f})")
                    
                    # Find the breakout candle
                    breakout_candle = post_or[post_or['high'] > or_high].iloc[0]
                    analysis['reasoning'].append(f"Breakout occurred at {breakout_candle['datetime'].strftime('%H:%M:%S')}")
                elif breakout_down:
                    analysis['criteria_met']['breakout'] = True
                    analysis['reasoning'].append(f"✅ Price broke below OR low (${or_low:.2f})")
                    
                    # Find the breakout candle
                    breakout_candle = post_or[post_or['low'] < or_low].iloc[0]
                    analysis['reasoning'].append(f"Breakout occurred at {breakout_candle['datetime'].strftime('%H:%M:%S')}")
                else:
                    analysis['criteria_met']['breakout'] = False
                    analysis['reasoning'].append("❌ No breakout of OR high/low detected yet")
                    return analysis
            else:
                analysis['criteria_met']['breakout'] = False
                analysis['reasoning'].append("❌ No post-opening range data available")
                return analysis
        
        # Step 3: Check for retest of the breakout level
        retest_detected = False
        retest_quality = 0
        
        if breakout_up:
            # Calculate how close current price is to OR high
            distance_to_or_high = abs(current_price - or_high) / or_high * 100
            
            if distance_to_or_high <= 0.2:
                retest_detected = True
                retest_quality = 1.0 - (distance_to_or_high / 0.2)  # Higher score for closer retests
                analysis['criteria_met']['retest'] = True
                analysis['reasoning'].append(f"✅ Price is retesting OR high (within {distance_to_or_high:.2f}%)")
                analysis['level_type'] = 'orb_high'  # Store level type for reference
            else:
                analysis['criteria_met']['retest'] = False
                analysis['reasoning'].append(f"❌ Price not retesting OR high (distance: {distance_to_or_high:.2f}%)")
                return analysis
        
        elif breakout_down:
            # Calculate how close current price is to OR low
            distance_to_or_low = abs(current_price - or_low) / or_low * 100
            
            if distance_to_or_low <= 0.2:
                retest_detected = True
                retest_quality = 1.0 - (distance_to_or_low / 0.2)  # Higher score for closer retests
                analysis['criteria_met']['retest'] = True
                analysis['reasoning'].append(f"✅ Price is retesting OR low (within {distance_to_or_low:.2f}%)")
                analysis['level_type'] = 'orb_low'  # Store level type for reference
            else:
                analysis['criteria_met']['retest'] = False
                analysis['reasoning'].append(f"❌ Price not retesting OR low (distance: {distance_to_or_low:.2f}%)")
                return analysis
            
        # Step 4: Check confirming candle patterns
        candle_confirmation = False
        candle_quality = 0

        if breakout_up:
            # For long setups, we want bullish candles
            if current_data['close'] > current_data['open']:
                body_size = (current_data['close'] - current_data['open']) / (current_data['high'] - current_data['low'])
                
                if body_size > 0.5:  # Strong bullish candle
                    candle_confirmation = True
                    candle_quality = min(1.0, body_size)
                    analysis['criteria_met']['candle_pattern'] = True
                    analysis['reasoning'].append(f"✅ Strong bullish candle confirming upside momentum")
                elif current_data.get('hammer', False):
                    candle_confirmation = True
                    candle_quality = 0.8
                    analysis['criteria_met']['candle_pattern'] = True
                    analysis['reasoning'].append(f"✅ Hammer candle at support (bullish)")
                elif current_data.get('bullish_engulfing', False):
                    candle_confirmation = True
                    candle_quality = 0.9
                    analysis['criteria_met']['candle_pattern'] = True
                    analysis['reasoning'].append(f"✅ Bullish engulfing pattern at support")
                else:
                    analysis['criteria_met']['candle_pattern'] = False
                    analysis['reasoning'].append(f"❌ Candle not showing strong bullish momentum")
            else:
                analysis['criteria_met']['candle_pattern'] = False
                analysis['reasoning'].append(f"❌ Bearish candle at retest (expecting bullish)")

        elif breakout_down:
            # For short setups, we want bearish candles
            if current_data['close'] < current_data['open']:
                body_size = (current_data['open'] - current_data['close']) / (current_data['high'] - current_data['low'])
                
                if body_size > 0.5:  # Strong bearish candle
                    candle_confirmation = True
                    candle_quality = min(1.0, body_size)
                    analysis['criteria_met']['candle_pattern'] = True
                    analysis['reasoning'].append(f"✅ Strong bearish candle confirming downside momentum")
                elif current_data.get('inverted_hammer', False) or current_data.get('shooting_star', False):
                    candle_confirmation = True
                    candle_quality = 0.8
                    analysis['criteria_met']['candle_pattern'] = True
                    analysis['reasoning'].append(f"✅ Shooting star candle at resistance (bearish)")
                elif current_data.get('bearish_engulfing', False):
                    candle_confirmation = True
                    candle_quality = 0.9
                    analysis['criteria_met']['candle_pattern'] = True
                    analysis['reasoning'].append(f"✅ Bearish engulfing pattern at resistance")
                else:
                    analysis['criteria_met']['candle_pattern'] = False
                    analysis['reasoning'].append(f"❌ Candle not showing strong bearish momentum")
            else:
                analysis['criteria_met']['candle_pattern'] = False
                analysis['reasoning'].append(f"❌ Bullish candle at retest (expecting bearish)")
        
        # Step 5: Check volume
        volume_confirmation = False
        volume_quality = 0
        
        # Check if rel_volume is available
        if 'rel_volume' in current_data:
            rel_volume = current_data['rel_volume']
            
            if rel_volume > 1.2:
                volume_confirmation = True
                volume_quality = min(1.0, (rel_volume - 1.2) / 2 + 0.5)  # Scale from 0.5 to 1.0
                analysis['criteria_met']['volume'] = True
                analysis['reasoning'].append(f"✅ Increased volume: {rel_volume:.1f}x average")
            else:
                analysis['criteria_met']['volume'] = False
                analysis['reasoning'].append(f"❌ Volume not above threshold: {rel_volume:.1f}x average")
        else:
            analysis['criteria_met']['volume'] = None
            analysis['reasoning'].append("⚠️ Volume data not available")
            # If volume data isn't available, we'll rely more on other criteria
            volume_confirmation = True
            volume_quality = 0.5
        
        # Step 6: Check RSI for momentum confirmation
        momentum_confirmation = False
        momentum_quality = 0
        
        if 'rsi' in current_data:
            rsi = current_data['rsi']
            
            if breakout_up and rsi > 50:
                momentum_confirmation = True
                momentum_quality = min(1.0, (rsi - 50) / 20)  # Scale from 0 to 1.0
                analysis['criteria_met']['momentum'] = True
                analysis['reasoning'].append(f"✅ RSI showing bullish momentum: {rsi:.1f}")
            elif breakout_down and rsi < 50:
                momentum_confirmation = True
                momentum_quality = min(1.0, (50 - rsi) / 20)  # Scale from 0 to 1.0
                analysis['criteria_met']['momentum'] = True
                analysis['reasoning'].append(f"✅ RSI showing bearish momentum: {rsi:.1f}")
            else:
                analysis['criteria_met']['momentum'] = False
                analysis['reasoning'].append(f"❌ RSI not confirming momentum: {rsi:.1f}")
        else:
            analysis['criteria_met']['momentum'] = None
            analysis['reasoning'].append("⚠️ RSI data not available")
            # If RSI data isn't available, we'll rely more on other criteria
            momentum_confirmation = True
            momentum_quality = 0.5
        
        # Step 7: Consider pattern recognition results if available
        pattern_confidence = 0.5  # Default neutral
        
        if pattern_results is not None:
            try:
                # Check if pattern_results is a dictionary or an object
                if isinstance(pattern_results, dict):
                    pattern_confidence = pattern_results.get('final_confidence', 0.5)
                elif hasattr(pattern_results, 'final_confidence'):
                    pattern_confidence = pattern_results.final_confidence
                else:
                    logger.warning("pattern_results doesn't contain final_confidence")
                
                if pattern_confidence > 0.6:
                    analysis['reasoning'].append(f"✅ Pattern recognition model confidence: {pattern_confidence*100:.1f}%")
                else:
                    analysis['reasoning'].append(f"⚠️ Pattern recognition model confidence: {pattern_confidence*100:.1f}%")
            except Exception as e:
                logger.error(f"Error using pattern results: {str(e)}")
                analysis['reasoning'].append("⚠️ Error using pattern recognition model")
        
        # Step 8: Make final decision based on all criteria
        base_criteria_met = retest_detected and candle_confirmation
        supporting_criteria_met = volume_confirmation or momentum_confirmation
        
        # Calculate final confidence score
        # Weight factors - adjust these based on what's most important for the strategy
        criteria_weights = {
            'retest': 0.3,
            'candle': 0.3,
            'volume': 0.15,
            'momentum': 0.15,
            'patterns': 0.1
        }
        
        # Calculate confidence
        confidence = (
            retest_quality * criteria_weights['retest'] +
            candle_quality * criteria_weights['candle'] +
            volume_quality * criteria_weights['volume'] +
            momentum_quality * criteria_weights['momentum'] +
            pattern_confidence * criteria_weights['patterns']
        )
        
        # Make decision
        if base_criteria_met and supporting_criteria_met:  # Must meet the core criteria
            if breakout_up:
                analysis['decision'] = 'BUY'
                analysis['entry_price'] = current_price
                analysis['stop_loss'] = max(current_price * 0.995, or_low)  # Either 0.5% below entry or OR low
                analysis['target'] = or_high + (or_high - or_low) * 0.5  # 50% extension
            elif breakout_down:
                analysis['decision'] = 'SELL'
                analysis['entry_price'] = current_price
                analysis['stop_loss'] = min(current_price * 1.005, or_high)  # Either 0.5% above entry or OR high
                analysis['target'] = or_low - (or_high - or_low) * 0.5  # 50% extension
            
            analysis['confidence'] = confidence
            
            # Calculate risk/reward
            if analysis['decision'] in ('BUY', 'SELL'):
                risk = abs(analysis['entry_price'] - analysis['stop_loss'])
                reward = abs(analysis['entry_price'] - analysis['target'])
                analysis['risk_reward'] = reward / risk if risk > 0 else 0
                
                # Add decision reasoning
                analysis['reasoning'].append("\n🔍 DECISION ANALYSIS:")
                analysis['reasoning'].append(f"Strategy indicates {analysis['decision']} at ${analysis['entry_price']:.2f}")
                analysis['reasoning'].append(f"Stop Loss: ${analysis['stop_loss']:.2f}")
                analysis['reasoning'].append(f"Target: ${analysis['target']:.2f}")
                analysis['reasoning'].append(f"Risk/Reward: {analysis['risk_reward']:.2f}")
                analysis['reasoning'].append(f"Confidence: {confidence*100:.1f}%")
                
                # Check if risk/reward is acceptable
                if analysis['risk_reward'] < 1.5:
                    analysis['reasoning'].append("⚠️ Risk/reward ratio below 1.5, consider skipping this trade")
                    analysis['confidence'] *= 0.8  # Reduce confidence
        else:
            analysis['reasoning'].append("\n🔍 DECISION ANALYSIS:")
            analysis['reasoning'].append("Not all criteria are met for a valid ORB setup")
            
            # List missing criteria
            if not retest_detected:
                analysis['reasoning'].append("❌ Missing: Price retest of breakout level")
            if not candle_confirmation:
                analysis['reasoning'].append("❌ Missing: Confirming candle pattern")
            if not volume_confirmation:
                analysis['reasoning'].append("❌ Missing: Increased volume")
            if not momentum_confirmation:
                analysis['reasoning'].append("❌ Missing: Momentum confirmation")
        
        return analysis


class HybridReasoningEngine:
    """
    A hybrid reasoning engine that combines rule-based reasoning with AI-based reasoning
    for more robust trading decisions on ORB 5-minute strategy.
    """
    
    def __init__(self, model_path=None, use_ai=True, data_dir="market_data"):
        """Initialize the hybrid reasoning engine"""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Create both engines that share the same data directory
        self.rule_engine = RuleBasedReasoningEngine(data_dir=data_dir)
        self.use_ai = use_ai
        
        if use_ai:
            self.ai_engine = LocalAIReasoningEngine(model_path=model_path, data_dir=data_dir)
        else:
            self.ai_engine = None
            
        # Initialize shared market level analyzer
        self.level_analyzer = self.rule_engine.level_analyzer
        
    def update_retest_outcome(self, level_type, date, time, outcome):
        """Pass through method to update retest outcomes"""
        if self.use_ai and self.ai_engine:
            return self.ai_engine.update_retest_outcome(level_type, date, time, outcome)
        else:
            # Implement a basic version for rule-based engine
            logger.warning("AI engine not available, using direct level analyzer update")
            # Format date as string if it's a datetime object
            date_str = date.strftime('%Y-%m-%d') if isinstance(date, datetime) else date
            
            if level_type in self.level_analyzer.level_data and date_str in self.level_analyzer.level_data[level_type]:
                for i, retest in enumerate(self.level_analyzer.level_data[level_type][date_str]):
                    if retest.get('time', '') == time:
                        self.level_analyzer.level_data[level_type][date_str][i]['outcome'] = outcome
                        self.level_analyzer.save_data()
                        return True
            return False
    
    def analyze_setup(self, current_data, market_data=None, pattern_model=None):
        """
        Analyze a potential ORB setup using both rule-based and AI reasoning
        
        Args:
            current_data: Current market state (latest candle)
            market_data: Historical data for context (today's trading)
            pattern_model: PatternModel instance for finding similar patterns
            
        Returns:
            Analysis dict with decision and combined reasoning
        """
        # Find similar patterns if pattern model is provided
        pattern_results = None
        if pattern_model is not None and 'features' in current_data:
            try:
                # Safely call the analyze_pattern method
                if hasattr(pattern_model, 'analyze_pattern') and callable(pattern_model.analyze_pattern):
                    pattern_results = pattern_model.analyze_pattern(current_data['features'])
                else:
                    logger.warning("Pattern model doesn't have analyze_pattern method")
            except Exception as e:
                logger.error(f"Error analyzing pattern: {str(e)}")
        
        # First, get rule-based analysis
        rule_analysis = self.rule_engine.analyze_setup(current_data, market_data, pattern_results)
        
        # If the rule engine doesn't detect a valid setup, we can stop here
        if rule_analysis['decision'] == 'NO_TRADE' and rule_analysis['confidence'] < 0.4:
            rule_analysis['reasoning_source'] = 'rule-based'
            return rule_analysis
        
        # If AI is enabled, get AI analysis
        ai_analysis = None
        if self.use_ai and self.ai_engine:
            try:
                ai_analysis = self.ai_engine.analyze_setup(current_data, market_data, pattern_results)
            except Exception as e:
                logger.error(f"Error with AI analysis: {str(e)}")
                # Continue with just rule-based analysis if AI fails
        
        # If we don't have AI analysis, return rule-based analysis
        if not ai_analysis:
            rule_analysis['reasoning_source'] = 'rule-based'
            return rule_analysis
        
        # Now combine the analyses
        return self._combine_analyses(rule_analysis, ai_analysis)
    
    def _combine_analyses(self, rule_analysis, ai_analysis):
        """Combine rule-based and AI analyses into a final decision"""
        # Start with a fresh analysis structure
        combined = {
            'decision': 'NO_TRADE',
            'confidence': 0.0,
            'reasoning': [],
            'criteria_met': rule_analysis.get('criteria_met', {}),
            'entry_price': None,
            'stop_loss': None,
            'target': None,
            'risk_reward': None,
            'rule_decision': rule_analysis['decision'],
            'ai_decision': ai_analysis['decision'],
            'reasoning_source': 'hybrid',
            'level_type': rule_analysis.get('level_type')  # Preserve level type from rule analysis
        }
        
        # Add headers for different reasoning sources
        combined['reasoning'].append("📊 RULE-BASED ANALYSIS:")
        combined['reasoning'].extend(rule_analysis.get('reasoning', []))
        combined['reasoning'].append("\n🧠 AI ANALYSIS:")
        combined['reasoning'].append(ai_analysis.get('reasoning', ""))
        
        # Decision logic
        # If both agree, use that decision with high confidence
        if rule_analysis['decision'] == ai_analysis['decision'] and rule_analysis['decision'] != 'NO_TRADE':
            combined['decision'] = rule_analysis['decision']
            # Average the confidences, but weight more toward the rule-based analysis
            combined['confidence'] = 0.6 * rule_analysis['confidence'] + 0.4 * ai_analysis['confidence']
            
            # Use the rule-based parameters since they're more precisely calculated
            combined['entry_price'] = rule_analysis['entry_price']
            combined['stop_loss'] = rule_analysis['stop_loss']
            combined['target'] = rule_analysis['target']
            combined['risk_reward'] = rule_analysis['risk_reward']
            
            combined['reasoning'].append("\n🔍 COMBINED DECISION ANALYSIS:")
            combined['reasoning'].append(f"Both rule-based and AI analyses agree on {combined['decision']}")
            combined['reasoning'].append(f"Combined confidence: {combined['confidence']*100:.1f}%")
        
        # If they disagree, but one has high confidence, use that one
        elif rule_analysis['confidence'] > 0.7 and rule_analysis['decision'] != 'NO_TRADE':
            combined['decision'] = rule_analysis['decision']
            combined['confidence'] = rule_analysis['confidence'] * 0.9  # Reduce confidence slightly due to disagreement
            
            combined['entry_price'] = rule_analysis['entry_price']
            combined['stop_loss'] = rule_analysis['stop_loss']
            combined['target'] = rule_analysis['target']
            combined['risk_reward'] = rule_analysis['risk_reward']
            
            combined['reasoning'].append("\n🔍 COMBINED DECISION ANALYSIS:")
            combined['reasoning'].append(f"Rule-based analysis strongly suggests {combined['decision']}")
            combined['reasoning'].append(f"AI analysis suggests {ai_analysis['decision']}, but with lower confidence")
            combined['reasoning'].append(f"Using rule-based decision with adjusted confidence: {combined['confidence']*100:.1f}%")
        
        elif ai_analysis['confidence'] > 0.7 and ai_analysis['decision'] != 'NO_TRADE':
            combined['decision'] = ai_analysis['decision']
            combined['confidence'] = ai_analysis['confidence'] * 0.9  # Reduce confidence slightly due to disagreement
            
            combined['entry_price'] = ai_analysis['entry_price']
            combined['stop_loss'] = ai_analysis['stop_loss']
            combined['target'] = ai_analysis['target']
            combined['risk_reward'] = ai_analysis.get('risk_reward', None)
            
            combined['reasoning'].append("\n🔍 COMBINED DECISION ANALYSIS:")
            combined['reasoning'].append(f"AI analysis strongly suggests {combined['decision']}")
            combined['reasoning'].append(f"Rule-based analysis suggests {rule_analysis['decision']}, but with lower confidence")
            combined['reasoning'].append(f"Using AI decision with adjusted confidence: {combined['confidence']*100:.1f}%")
        
        # If both have moderate confidence but disagree, take the more conservative approach
        else:
            combined['decision'] = 'NO_TRADE'
            combined['confidence'] = max(rule_analysis['confidence'], ai_analysis['confidence']) * 0.5
            
            combined['reasoning'].append("\n🔍 COMBINED DECISION ANALYSIS:")
            combined['reasoning'].append(f"Rule-based analysis suggests {rule_analysis['decision']} with {rule_analysis['confidence']*100:.1f}% confidence")
            combined['reasoning'].append(f"AI analysis suggests {ai_analysis['decision']} with {ai_analysis['confidence']*100:.1f}% confidence")
            combined['reasoning'].append("Due to disagreement and moderate confidence levels, taking NO TRADE position")
        
        return combined
        
