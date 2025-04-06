import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
import pandas as pd
from data_collector import DataCollector
from orb_pattern_extractor import ORBPatternExtractor
from pattern_model import PatternModel
from orb_trading_assistant import ORBTradingAssistant
from ai_reasoning_engine import LocalAIReasoningEngine, RuleBasedReasoningEngine, HybridReasoningEngine

# Set up logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("orb_system.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("ORBSystem")

logger = setup_logging()

def train_model(api_key, symbol="QQQ", start_date="2023-01-01", end_date="2023-12-31", force_refresh=False, data_dir="market_data"):
    """
    Train the ORB pattern model on historical data
    
    Args:
        api_key: Polygon.io API key
        symbol: Trading symbol to train on
        start_date: Start date for training data
        end_date: End date for training data
        force_refresh: Whether to force download new data
        data_dir: Directory to store data and models
        
    Returns:
        Path to trained model file
    """
    logger.info(f"Training model for {symbol} from {start_date} to {end_date}")
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    data_file = os.path.join(data_dir, f"{symbol}_historical_data.csv")
    
    # Check if we already have the data
    if os.path.isfile(data_file) and not force_refresh:
        logger.info(f"Loading existing data from {data_file}")
        df = pd.read_csv(data_file, parse_dates=['datetime'])
        
        # Filter to date range if requested
        df = df[(df['datetime'].dt.date >= pd.to_datetime(start_date).date()) & 
               (df['datetime'].dt.date <= pd.to_datetime(end_date).date())]
    else:
        # Collect data
        logger.info("Collecting historical data...")
        collector = DataCollector(api_key)
        
        # Process data with all enhancements
        try:
            logger.info("Processing data with enhanced features...")
            df = collector.process_data_for_orb_analysis(symbol, start_date, end_date, force_refresh=force_refresh)
            
            if not isinstance(df, pd.DataFrame) or df.empty:
                logger.error("Failed to process data")
                return None
                
            logger.info(f"Successfully processed {len(df)} candles")
            df.to_csv(data_file, index=False)
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}", exc_info=True)
            logger.info("Falling back to simple data collection...")
            
            try:
                df = collector.fetch_historical_data(symbol, start_date, end_date)
                df = collector.identify_opening_ranges(df)
                df = collector.identify_candlestick_patterns(df)
                df = collector.add_enhanced_features(df)
                df.to_csv(data_file, index=False)
            except Exception as e2:
                logger.error(f"Error fetching data: {str(e2)}", exc_info=True)
                logger.error("Failed to fetch data. Check API key and limits.")
                return None
    
    # Extract patterns
    logger.info("Extracting ORB patterns...")
    extractor = ORBPatternExtractor()
    patterns = extractor.extract_patterns(df)
    
    # Calculate statistics
    long_patterns = [p for p in patterns if p['direction'] == 'LONG']
    short_patterns = [p for p in patterns if p['direction'] == 'SHORT']
    
    # Check for mid-level patterns too
    long_mid_patterns = [p for p in patterns if p['direction'] == 'LONG_MID']
    short_mid_patterns = [p for p in patterns if p['direction'] == 'SHORT_MID']
    
    # Calculate win rates
    long_wins = sum(1 for p in long_patterns if p['outcome'] == 1)
    short_wins = sum(1 for p in short_patterns if p['outcome'] == 1)
    long_mid_wins = sum(1 for p in long_mid_patterns if p['outcome'] == 1)
    short_mid_wins = sum(1 for p in short_mid_patterns if p['outcome'] == 1)
    
    # Log statistics
    if long_patterns:
        logger.info(f"Found {len(long_patterns)} long patterns, win rate: {long_wins / max(1, len(long_patterns)) * 100:.1f}%")
    else:
        logger.warning("No long patterns found")
        
    if short_patterns:
        logger.info(f"Found {len(short_patterns)} short patterns, win rate: {short_wins / max(1, len(short_patterns)) * 100:.1f}%")
    else:
        logger.warning("No short patterns found")
        
    if long_mid_patterns:
        logger.info(f"Found {len(long_mid_patterns)} long mid-level patterns, win rate: {long_mid_wins / max(1, len(long_mid_patterns)) * 100:.1f}%")
        
    if short_mid_patterns:
        logger.info(f"Found {len(short_mid_patterns)} short mid-level patterns, win rate: {short_mid_wins / max(1, len(short_mid_patterns)) * 100:.1f}%")
    
    # Save patterns
    patterns_file = os.path.join(data_dir, f"{symbol}_orb_patterns.csv")
    pd.DataFrame(patterns).to_csv(patterns_file, index=False)
    
    # Train model
    if patterns:
        logger.info("Training pattern recognition model...")
        model = PatternModel(model_type="hybrid")
        model.fit(patterns)
        
        # Save model
        model_file = os.path.join(data_dir, f"{symbol}_orb_model.joblib")
        model.save(model_file)
        logger.info(f"Model saved to {model_file}")
        
        # Generate feature importance plot if possible
        try:
            importance_plot = model.plot_feature_importance(top_n=10)
            if importance_plot:
                plot_file = os.path.join(data_dir, f"{symbol}_feature_importance.png")
                importance_plot.savefig(plot_file)
                logger.info(f"Feature importance plot saved to {plot_file}")
        except Exception as e:
            logger.warning(f"Could not generate feature importance plot: {str(e)}")
        
        return model_file
    else:
        logger.error("No patterns found, cannot train model")
        return None

def run_backtest(api_key, model_file, symbol="QQQ", start_date=None, end_date=None, use_ai=False, data_dir="market_data"):
    """
    Run backtest on historical data
    
    Args:
        api_key: Polygon.io API key
        model_file: Path to trained model file
        symbol: Trading symbol
        start_date: Start date for backtest
        end_date: End date for backtest
        use_ai: Whether to use AI reasoning (slower but more accurate)
        data_dir: Directory for data and results
        
    Returns:
        Backtest results dictionary
    """
    logger.info(f"Running backtest for {symbol}")
    
    # Default dates if not provided
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Create the trading assistant
    assistant = ORBTradingAssistant(
        api_key=api_key, 
        model_path=model_file,
        use_ai=use_ai,  # By default, use rule-based only for faster backtests
        data_dir=data_dir
    )
    
    # Run backtest
    results = assistant.backtest(symbol, start_date, end_date, use_ai=use_ai)
    
    if results:
        logger.info(f"Backtest completed with {len(results['trades'])} trades")
        logger.info(f"Win rate: {results['win_rate']*100:.2f}%")
        logger.info(f"Profit factor: {results['profit_factor']:.2f}")
        logger.info(f"Total profit: {results['total_profit']:.2f}%")
        logger.info(f"Results saved to {results.get('backtest_file', 'unknown location')}")
    else:
        logger.error("Backtest failed or returned no results")
    
    return results

def optimize_strategy(api_key, model_file, symbol="QQQ", start_date=None, end_date=None, data_dir="market_data"):
    """
    Optimize strategy parameters
    
    Args:
        api_key: Polygon.io API key
        model_file: Path to trained model file
        symbol: Trading symbol
        start_date: Start date for optimization
        end_date: End date for optimization
        data_dir: Directory for data and results
        
    Returns:
        Optimization results dictionary
    """
    logger.info(f"Optimizing strategy for {symbol}")
    
    # Default dates if not provided
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Create the trading assistant
    assistant = ORBTradingAssistant(
        api_key=api_key, 
        model_path=model_file,
        use_ai=False,  # Use rule-based only for optimization (much faster)
        data_dir=data_dir
    )
    
    # Run optimization
    results = assistant.optimize_parameters(symbol, start_date, end_date)
    
    if results:
        logger.info("Optimization completed successfully")
        logger.info(f"Best parameters: {results['best_params']}")
        logger.info(f"Best profit factor: {results['best_result']['profit_factor']:.2f}")
        logger.info(f"Best win rate: {results['best_result']['win_rate']*100:.2f}%")
    else:
        logger.error("Optimization failed or found no profitable parameters")
    
    return results

def run_live_assistant(api_key, model_file, symbol="QQQ", use_ai=True, ai_model_path=None, data_dir="market_data"):
    """
    Run the live trading assistant
    
    Args:
        api_key: Polygon.io API key
        model_file: Path to trained model file
        symbol: Trading symbol
        use_ai: Whether to use AI reasoning
        ai_model_path: Path to AI model
        data_dir: Directory for data and results
    """
    logger.info(f"Starting live trading assistant for {symbol}")
    
    # Create the assistant with AI options
    assistant = ORBTradingAssistant(
        api_key=api_key, 
        model_path=model_file,
        use_ai=use_ai,
        ai_model_path=ai_model_path,
        data_dir=data_dir
    )
    
    try:
        assistant.monitor_symbol(symbol)
    except KeyboardInterrupt:
        logger.info("Live monitoring stopped by user")
        # Make sure to save trade history before exiting
        assistant.save_trade_history()
    except Exception as e:
        logger.error(f"Error in live monitoring: {str(e)}", exc_info=True)
        # Try to save trade history even if there was an error
        try:
            assistant.save_trade_history()
        except:
            pass

def initialize_ai_model(model_path=None, data_dir="market_data"):
    """
    Initialize the AI model to ensure it's loaded
    
    Args:
        model_path: Path to AI model
        data_dir: Directory for data and model
        
    Returns:
        bool: True if initialization successful, False otherwise
    """
    logger.info("Initializing AI model")
    
    if not model_path:
        model_path = "deepseek-ai/deepseek-coder-6.7b-instruct"
    
    try:
        # Check required dependencies
        try:
            import transformers
            import torch
        except ImportError as e:
            logger.error(f"Missing required dependency: {str(e)}")
            logger.error("Please install: pip install transformers torch")
            return False
            
        # Try loading the model with more conservative settings
        engine = LocalAIReasoningEngine(model_path=model_path, data_dir=data_dir)
        
        # Override the load_model method with more conservative settings
        from types import MethodType
        
        def safe_load_model(self):
            """Safe model loading with fallbacks"""
            if self.initialized:
                return True
                
            logger.info("Loading DeepSeek model with conservative settings...")
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                import torch
                
                # Try to load with minimal settings first
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
                
                # Check for CUDA/GPU
                if torch.cuda.is_available():
                    logger.info("CUDA available, using GPU acceleration")
                    device_map = "auto"
                    dtype = torch.float16
                else:
                    logger.info("CUDA not available, using CPU only")
                    device_map = None
                    dtype = torch.float32  # Use float32 for CPU
                
                # Load with conservative settings
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    device_map=device_map,
                    low_cpu_mem_usage=True
                )
                
                self.initialized = True
                logger.info("DeepSeek model loaded successfully")
                
                # Test the model with a simple generation
                test_prompt = "What is 2+2?"
                logger.info("Testing model with a simple prompt...")
                inputs = self.tokenizer(test_prompt, return_tensors="pt")
                if device_map == "auto":
                    inputs = inputs.to("cuda")
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        max_new_tokens=10,
                        do_sample=False
                    )
                
                test_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                logger.info(f"Test response: {test_response}")
                return True
                
            except Exception as e:
                logger.error(f"Error loading DeepSeek model: {str(e)}")
                logger.warning("Falling back to rule-based reasoning only")
                return False
        
        # Assign the new method
        engine.load_model = MethodType(safe_load_model, engine)
        
        # Try loading
        success = engine.load_model()
        return success
    
    except Exception as e:
        logger.error(f"Error initializing AI model: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='ORB Trading System with AI Enhancement')
    
    # Basic arguments
    parser.add_argument('--mode', type=str, 
                        choices=['train', 'backtest', 'optimize', 'live', 'init_ai'], 
                        default='live', 
                        help='Mode: train model, run backtest, optimize strategy, or run live assistant')
    parser.add_argument('--symbol', type=str, default='QQQ',
                        help='Trading symbol (default: QQQ)')
    parser.add_argument('--api-key', type=str, required=True,
                        help='Polygon API key')
    
    # Date range arguments
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date for training/backtest data (default: depends on mode)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date for training/backtest data (default: yesterday)')
    
    # AI-related arguments
    parser.add_argument('--use-ai', action='store_true',
                        help='Use AI for reasoning (default: False)')
    parser.add_argument('--ai-model-path', type=str, default="deepseek-ai/deepseek-coder-6.7b-instruct",
                        help='Path to AI model (default: deepseek-ai/deepseek-coder-6.7b-instruct)')
    parser.add_argument('--rule-based-only', action='store_true',
                        help='Use only rule-based reasoning (ignore --use-ai flag)')
    
    # Data management arguments
    parser.add_argument('--force-refresh', action='store_true',
                        help='Force refresh of data when training')
    parser.add_argument('--data-dir', type=str, default="market_data",
                        help='Directory to store data and models (default: market_data)')
    
    # Backtest-specific arguments
    parser.add_argument('--backtests', type=int, default=1,
                        help='Number of backtests to run (default: 1)')
    
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Calculate default dates if not provided
    if args.start_date is None:
        if args.mode == 'train':
            # For training, use 1 year of data by default
            args.start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        else:
            # For other modes, use 60 days by default
            args.start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    
    if args.end_date is None:
        args.end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Force rule-based only if requested
    if args.rule_based_only:
        args.use_ai = False
        logger.info("Using rule-based reasoning only (AI disabled)")
    
    # First, check if we need to initialize the AI model
    if args.mode == 'init_ai' or (args.use_ai and args.mode != 'train'):
        logger.info("Pre-loading AI model...")
        success = initialize_ai_model(args.ai_model_path, args.data_dir)
        
        if args.mode == 'init_ai':
            if success:
                logger.info("AI model initialization successful")
                return
            else:
                logger.error("AI model initialization failed")
                logger.warning("Proceeding with rule-based reasoning only")
                args.use_ai = False
    
    # Determine model file path
    model_file = os.path.join(args.data_dir, f"{args.symbol}_orb_model.joblib")
    
    # Execute the requested mode
    try:
        if args.mode == 'train':
            trained_model = train_model(
                args.api_key, 
                args.symbol, 
                args.start_date, 
                args.end_date, 
                args.force_refresh,
                args.data_dir
            )
            
            if not trained_model:
                logger.error("Training failed")
                return
                
            model_file = trained_model
            
        elif args.mode == 'backtest':
            # Check if model exists, train if not
            if not os.path.isfile(model_file):
                logger.warning(f"Model file {model_file} not found. Training new model...")
                model_file = train_model(
                    args.api_key, 
                    args.symbol, 
                    args.start_date, 
                    args.end_date,
                    False,  # No force refresh
                    args.data_dir
                )
                
                if not model_file:
                    logger.error("Training failed, cannot run backtest")
                    return
            
            # Run multiple backtests if requested
            for i in range(args.backtests):
                if args.backtests > 1:
                    logger.info(f"Running backtest {i+1} of {args.backtests}")
                
                run_backtest(
                    args.api_key, 
                    model_file, 
                    args.symbol, 
                    args.start_date, 
                    args.end_date,
                    args.use_ai,
                    args.data_dir
                )
            
        elif args.mode == 'optimize':
            # Check if model exists, train if not
            if not os.path.isfile(model_file):
                logger.warning(f"Model file {model_file} not found. Training new model...")
                model_file = train_model(
                    args.api_key, 
                    args.symbol, 
                    args.start_date, 
                    args.end_date,
                    False,  # No force refresh
                    args.data_dir
                )
                
                if not model_file:
                    logger.error("Training failed, cannot run optimization")
                    return
            
            optimize_strategy(
                args.api_key, 
                model_file, 
                args.symbol, 
                args.start_date, 
                args.end_date,
                args.data_dir
            )
            
        elif args.mode == 'live':
            # Check if model exists, train if not
            if not os.path.isfile(model_file):
                logger.warning(f"Model file {model_file} not found. Training new model...")
                model_file = train_model(
                    args.api_key, 
                    args.symbol, 
                    args.start_date, 
                    args.end_date,
                    False,  # No force refresh
                    args.data_dir
                )
                
                if not model_file:
                    logger.error("Training failed, cannot run live assistant")
                    return
            
            # Run live trading assistant
            run_live_assistant(
                args.api_key, 
                model_file, 
                args.symbol,
                use_ai=args.use_ai,
                ai_model_path=args.ai_model_path,
                data_dir=args.data_dir
            )
    
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
