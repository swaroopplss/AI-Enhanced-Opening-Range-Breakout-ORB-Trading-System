import os
import numpy as np
import pandas as pd
import logging
from pattern_model import PatternModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re

logger = logging.getLogger(__name__)

class LocalAIReasoningEngine:
    """
    A reasoning engine that uses locally downloaded DeepSeek model for enhanced trading decisions.
    """
    
    def __init__(self, model_path="deepseek-ai/deepseek-coder-6.7b-instruct"):
        """Initialize the reasoning engine with DeepSeek model"""
        self.model_path = model_path
        logger.info(f"Initializing local AI reasoning engine with model: {model_path}")
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        self.initialized = False
        
        # Setup memory system for continuous learning
        self.memory = []
        self.max_memories = 100
        
    def load_model(self):
        """Load the model (lazy initialization to save memory)"""
        if self.initialized:
            return
            
        logger.info("Loading DeepSeek model...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,  # Use half precision to save memory
                device_map="auto"  # Auto-distribute across available hardware
            )
            self.initialized = True
            logger.info("DeepSeek model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading DeepSeek model: {str(e)}")
            raise
    
    def analyze_setup(self, current_data, market_data=None, pattern_model_results=None):
        """Use AI to analyze trading setup in detail"""
        # Ensure model is loaded
        self.load_model()
        
        # Prepare market data in a format the model can understand
        market_summary = self._prepare_market_data(current_data, market_data)
        pattern_summary = self._prepare_pattern_analysis(pattern_model_results)
        memory_summary = self._get_relevant_memories(current_data)
        
        # Construct the prompt
        prompt = self._construct_prompt(market_summary, pattern_summary, memory_summary)
        
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
        
        # Add candle information
        summary.append(f"Current candle: Open=${current_data['open']:.2f}, High=${current_data['high']:.2f}, Low=${current_data['low']:.2f}, Close=${current_data['close']:.2f}")
        if 'body_size' in current_data:
            summary.append(f"Candle body size: {current_data['body_size']:.2f} of range")
            
        if 'hammer' in current_data and current_data['hammer']:
            summary.append("Candle pattern: Hammer (bullish)")
        if 'shooting_star' in current_data and current_data['shooting_star']:
            summary.append("Candle pattern: Shooting star (bearish)")
            
        # Add distance from OR levels
        if 'dist_from_or_high' in current_data:
            summary.append(f"Distance from OR high: {current_data['dist_from_or_high']:.2f}%")
        if 'dist_from_or_low' in current_data:
            summary.append(f"Distance from OR low: {current_data['dist_from_or_low']:.2f}%")
        
        # Add market context from historical data if available
        if market_data is not None and not market_data.empty:
            # Detect breakout status
            post_or = market_data[market_data['minutes_from_open'] >= 5]
            if not post_or.empty and 'or_high' in current_data and 'or_low' in current_data:
                or_high = current_data['or_high'] 
                or_low = current_data['or_low']
                
                if post_or['high'].max() > or_high:
                    time_of_breakout = post_or[post_or['high'] > or_high].iloc[0]['datetime']
                    summary.append(f"Price broke above Opening Range High at {time_of_breakout}")
                    
                    # Check if we've retested the breakout level
                    after_breakout = post_or.loc[post_or['datetime'] > time_of_breakout]
                    if not after_breakout.empty:
                        if (after_breakout['low'].min() - or_high) / or_high < 0.002:
                            summary.append("Price has retested OR high (potential long setup)")
                
                if post_or['low'].min() < or_low:
                    time_of_breakout = post_or[post_or['low'] < or_low].iloc[0]['datetime']
                    summary.append(f"Price broke below Opening Range Low at {time_of_breakout}")
                    
                    # Check if we've retested the breakout level
                    after_breakout = post_or.loc[post_or['datetime'] > time_of_breakout]
                    if not after_breakout.empty:
                        if (or_low - after_breakout['high'].max()) / or_low < 0.002:
                            summary.append("Price has retested OR low (potential short setup)")
        
        return "\n".join(summary)
    
    def _prepare_pattern_analysis(self, pattern_results):
        """Format pattern recognition results for the model prompt"""
        if pattern_results is None:
            return "No pattern recognition results available."
            
        summary = []
        summary.append(f"Pattern Recognition Results:")
        summary.append(f"Recommended action: {pattern_results['recommended_action']}")
        summary.append(f"Confidence: {pattern_results['final_confidence'] * 100:.1f}%")
        
        # Add info about similar patterns
        if 'similar_patterns' in pattern_results and pattern_results['similar_patterns']:
            similar_patterns = pattern_results['similar_patterns']
            win_count = sum(1 for p in similar_patterns if p['outcome'] == 1)
            summary.append(f"Similar patterns found: {len(similar_patterns)}")
            summary.append(f"Win rate from similar patterns: {win_count / len(similar_patterns) * 100:.1f}%")
            
            # Add details from most similar pattern
            most_similar = similar_patterns[0]
            outcome = "Win" if most_similar['outcome'] == 1 else "Loss"
            summary.append(f"Most similar pattern ({outcome}):")
            summary.append(f"  Date: {most_similar['date']}")
            summary.append(f"  Direction: {most_similar['direction']}")
            summary.append(f"  RSI: {most_similar['rsi']:.1f}")
            summary.append(f"  Relative Volume: {most_similar['rel_volume']:.1f}x")
        
        return "\n".join(summary)
    
    def _get_relevant_memories(self, current_data):
        """Retrieve relevant memories from previous analyses"""
        if not self.memory:
            return "No previous trade analyses available."
            
        # Find memories with similar OR patterns
        if 'or_high' in current_data and 'or_low' in current_data:
            or_range_pct = (current_data['or_high'] - current_data['or_low']) / current_data['or_low'] * 100
            
            # Filter memories with similar OR ranges
            similar_memories = []
            for mem in self.memory:
                if 'or_range_pct' in mem['data']:
                    mem_range = mem['data']['or_range_pct']
                    if abs(mem_range - or_range_pct) < 0.2 * or_range_pct:  # Within 20% similarity
                        similar_memories.append(mem)
            
            if similar_memories:
                # Sort by most recent first
                similar_memories.sort(key=lambda x: x['timestamp'], reverse=True)
                
                # Return summary of the most relevant memories
                summary = []
                summary.append(f"Previous similar trades ({len(similar_memories)} found):")
                
                for i, mem in enumerate(similar_memories[:3]):  # Get top 3 memories
                    decision = mem['result']['decision']
                    outcome = mem.get('actual_outcome', 'Unknown')
                    confidence = mem['result']['confidence'] * 100
                    
                    summary.append(f"{i+1}. {decision} with {confidence:.1f}% confidence, outcome: {outcome}")
                    summary.append(f"   OR Range: {mem['data'].get('or_range_pct', 0):.2f}%")
                    summary.append(f"   Key factors: {mem['result'].get('key_factors', 'Not recorded')}")
                
                return "\n".join(summary)
        
        return "No relevant previous trades found."
    
    def _construct_prompt(self, market_summary, pattern_summary, memory_summary):
        """Create a detailed prompt for the model"""
        prompt = f"""You are an expert day trader with extensive experience in the Opening Range Breakout (ORB) strategy. Your task is to analyze the current market situation and provide a detailed trading decision.

### CURRENT MARKET DATA
{market_summary}

### PATTERN RECOGNITION ANALYSIS
{pattern_summary}

### HISTORICAL MEMORY
{memory_summary}

### OPENING RANGE BREAKOUT (ORB) STRATEGY RULES:
1. The ORB strategy is based on the price range established in the first 5 minutes after market open (9:30-9:35 AM).
2. A valid LONG setup requires:
   - A breakout above the OR high
   - A pullback to retest the OR high from above
   - Bullish candle confirmation at the retest
   - Increased volume at the retest
   - Technical indicators showing bullish momentum

3. A valid SHORT setup requires:
   - A breakout below the OR low
   - A pullback to retest the OR low from below
   - Bearish candle confirmation at the retest
   - Increased volume at the retest
   - Technical indicators showing bearish momentum

Based on all the information above, analyze whether there is a valid ORB setup right now. Consider market context, pattern recognition results, and your trading experience. Think step-by-step.

After your analysis, provide your final recommendation in the following structured format:

DECISION: [BUY/SELL/NO_TRADE]
CONFIDENCE: [0-100]
ENTRY_PRICE: [price]
STOP_LOSS: [price]
TARGET: [price]
RISK_REWARD: [ratio]
KEY_FACTORS: [Brief list of 3-5 key decision factors]
REASONING: [Your detailed step-by-step thought process]
"""
        return prompt
    
    def _get_model_inference(self, prompt):
        """Get response from the model"""
        logger.info("Getting inference from DeepSeek model")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=1024,
                temperature=0.1,
                top_p=0.95,
                do_sample=False
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the response
        if prompt in response:
            response = response.replace(prompt, "")
        
        return response
    
    def _parse_analysis(self, response_text):
        """Parse structured data from model response"""
        # Default result structure
        result = {
            'decision': 'NO_TRADE',
            'confidence': 0.0,
            'entry_price': None,
            'stop_loss': None,
            'target': None,
            'risk_reward': None,
            'key_factors': [],
            'reasoning': response_text.strip()
        }
        
        # Extract structured data using regex patterns
        decision_match = re.search(r'DECISION:\s*(\w+)', response_text)
        if decision_match:
            result['decision'] = decision_match.group(1).strip()
        
        confidence_match = re.search(r'CONFIDENCE:\s*(\d+)', response_text)
        if confidence_match:
            result['confidence'] = float(confidence_match.group(1)) / 100
        
        entry_match = re.search(r'ENTRY_PRICE:\s*\$?(\d+\.?\d*)', response_text)
        if entry_match:
            result['entry_price'] = float(entry_match.group(1))
        
        stop_match = re.search(r'STOP_LOSS:\s*\$?(\d+\.?\d*)', response_text)
        if stop_match:
            result['stop_loss'] = float(stop_match.group(1))
        
        target_match = re.search(r'TARGET:\s*\$?(\d+\.?\d*)', response_text)
        if target_match:
            result['target'] = float(target_match.group(1))
        
        rr_match = re.search(r'RISK_REWARD:\s*(\d+\.?\d*)', response_text)
        if rr_match:
            result['risk_reward'] = float(rr_match.group(1))
        
        key_factors_match = re.search(r'KEY_FACTORS:\s*(.+?)(?=\nREASONING|\Z)', response_text, re.DOTALL)
        if key_factors_match:
            key_factors_text = key_factors_match.group(1).strip()
            result['key_factors'] = [factor.strip() for factor in key_factors_text.split(',')]
        
        # Calculate risk/reward if necessary values are available but it wasn't calculated
        if (result['entry_price'] and result['stop_loss'] and result['target'] and 
            (result['risk_reward'] is None or result['risk_reward'] == 0)):
            if result['decision'] == 'BUY':
                risk = result['entry_price'] - result['stop_loss']
                reward = result['target'] - result['entry_price']
            else:  # SELL
                risk = result['stop_loss'] - result['entry_price']
                reward = result['entry_price'] - result['target']
                
            if risk > 0:
                result['risk_reward'] = reward / risk
        
        return result
    
    def _store_in_memory(self, data, result):
        """Store the analysis in memory for continuous learning"""
        memory_item = {
            'timestamp': pd.Timestamp.now(),
            'data': {k: v for k, v in data.items() if not isinstance(v, (list, np.ndarray))},
            'result': result,
            'actual_outcome': None  # To be updated later with actual trade outcome
        }
        
        self.memory.append(memory_item)
        
        # Trim memory if it exceeds the maximum size
        if len(self.memory) > self.max_memories:
            self.memory = self.memory[-self.max_memories:]
    
    def update_trade_outcome(self, entry_time, outcome):
        """Update memory with actual trade outcome"""
        # Find the memory item for this trade
        for mem in self.memory:
            if mem['timestamp'].strftime('%Y-%m-%d %H:%M') == entry_time.strftime('%Y-%m-%d %H:%M'):
                mem['actual_outcome'] = outcome
                logger.info(f"Updated memory with outcome: {outcome}")
                return True
        
        return False


class HybridReasoningEngine:
    """
    A hybrid reasoning engine that combines rule-based reasoning with AI-based reasoning
    for more robust trading decisions on ORB 5-minute strategy.
    """
    
    def __init__(self, model_path=None, use_ai=True):
        """Initialize the hybrid reasoning engine"""
        self.rule_engine = RuleBasedReasoningEngine()
        self.use_ai = use_ai
        
        if use_ai:
            self.ai_engine = LocalAIReasoningEngine(model_path=model_path)
        else:
            self.ai_engine = None
    
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
            pattern_results = pattern_model.analyze_pattern(current_data['features'])
        
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
            'reasoning_source': 'hybrid'
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


class RuleBasedReasoningEngine:
    """
    A reasoning engine that applies rule-based decision-making to ORB pattern trading.
    This engine checks all criteria for valid ORB setups and provides a explanation
    of trading decisions based on predefined rules.
    """
    
    def __init__(self):
        """Initialize the rule-based reasoning engine"""
        pass
    
    def analyze_setup(self, current_data, market_data=None, pattern_results=None):
        """
        Analyze a potential ORB setup using rule-based logic
        
        Args:
            current_data: Current market state (latest candle)
            market_data: Historical data for context (today's trading)
            pattern_results: Results from pattern recognition
            
        Returns:
            Analysis dict with decision and reasoning
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
                elif current_data.get('shooting_star', False):
                    candle_confirmation = True
                    candle_quality = 0.8
                    analysis['criteria_met']['candle_pattern'] = True
                    analysis['reasoning'].append(f"✅ Shooting star candle at resistance (bearish)")
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
            pattern_confidence = pattern_results['final_confidence']
            
            if pattern_confidence > 0.6:
                analysis['reasoning'].append(f"✅ Pattern recognition model confidence: {pattern_confidence*100:.1f}%")
            else:
                analysis['reasoning'].append(f"⚠️ Pattern recognition model confidence: {pattern_confidence*100:.1f}%")
        
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