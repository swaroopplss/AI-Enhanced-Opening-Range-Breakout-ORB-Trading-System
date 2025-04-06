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
