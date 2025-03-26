import argparse
import logging
import os
from datetime import datetime, timedelta
from data_collector import DataCollector
from orb_pattern_extractor import ORBPatternExtractor
from pattern_model import PatternModel
from orb_trading_assistant import ORBTradingAssistant
from ai_reasoning_engine import LocalAIReasoningEngine

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

def train_model(api_key, symbol="QQQ", start_date="2023-01-01", end_date="2023-12-31", force_refresh=False):
    """Train the ORB pattern model"""
    logger.info(f"Training model for {symbol} from {start_date} to {end_date}")
    
    data_file = f"{symbol}_historical_data.csv"
    
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
        df = collector.fetch_historical_data(symbol, start_date, end_date)
        df = collector.identify_opening_ranges(df)
        df.to_csv(data_file, index=False)
    
    # Extract patterns
    logger.info("Extracting ORB patterns...")
    extractor = ORBPatternExtractor()
    patterns = extractor.extract_patterns(df)
    
    # Calculate statistics
    long_patterns = [p for p in patterns if p['direction'] == 'LONG']
    short_patterns = [p for p in patterns if p['direction'] == 'SHORT']
    
    long_wins = sum(1 for p in long_patterns if p['outcome'] == 1)
    short_wins = sum(1 for p in short_patterns if p['outcome'] == 1)
    
    logger.info(f"Found {len(long_patterns)} long patterns, win rate: {long_wins / max(1, len(long_patterns)) * 100:.1f}%")
    logger.info(f"Found {len(short_patterns)} short patterns, win rate: {short_wins / max(1, len(short_patterns)) * 100:.1f}%")
    
    # Save patterns
    patterns_file = f"{symbol}_orb_patterns.csv"
    import pandas as pd
    pd.DataFrame(patterns).to_csv(patterns_file, index=False)
    
    # Train model
    logger.info("Training pattern recognition model...")
    model = PatternModel(model_type="hybrid")
    model.fit(patterns)
    
    # Save model
    model_file = f"{symbol}_orb_model.joblib"
    model.save(model_file)
    logger.info(f"Model saved to {model_file}")
    
    return model_file

def run_backtest(api_key, model_file, symbol="QQQ", start_date=None, end_date=None):
    """Run backtest on historical data"""
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
        use_ai=True
    )
    
    # Run backtest
    results = assistant.backtest(symbol, start_date, end_date)
    
    return results

def optimize_strategy(api_key, model_file, symbol="QQQ", start_date=None, end_date=None):
    """Optimize strategy parameters"""
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
        use_ai=True
    )
    
    # Run optimization
    results = assistant.optimize_parameters(symbol, start_date, end_date)
    
    return results

def run_live_assistant(api_key, model_file, symbol="QQQ", use_ai=True, ai_model_path=None):
    """Run the live trading assistant"""
    logger.info(f"Starting live trading assistant for {symbol}")
    
    # Create the assistant with AI options
    assistant = ORBTradingAssistant(
        api_key=api_key, 
        model_path=model_file,
        use_ai=use_ai,
        ai_model_path=ai_model_path
    )
    
    assistant.monitor_symbol(symbol)

def initialize_ai_model(model_path=None):
    """Initialize the AI model to ensure it's loaded"""
    logger.info("Initializing AI model")
    
    if not model_path:
        model_path = "deepseek-ai/deepseek-coder-6.7b-instruct"
    
    try:
        engine = LocalAIReasoningEngine(model_path=model_path)
        engine.load_model()
        logger.info("AI model initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing AI model: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='ORB Trading System with Deep Learning')
    parser.add_argument('--mode', type=str, choices=['train', 'backtest', 'optimize', 'live', 'init_ai'], 
                        default='live', help='Mode: train model, run backtest, optimize strategy, or run live assistant')
    parser.add_argument('--symbol', type=str, default='QQQ',
                        help='Trading symbol (default: QQQ)')
    parser.add_argument('--api-key', type=str, required=True,
                        help='Polygon API key')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date for training/backtest data (default: 60 days ago)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date for training/backtest data (default: yesterday)')
    parser.add_argument('--use-ai', action='store_true',
                        help='Use AI for reasoning')
    parser.add_argument('--ai-model-path', type=str, default="deepseek-ai/deepseek-coder-6.7b-instruct",
                        help='Path to AI model (default: deepseek-ai/deepseek-coder-6.7b-instruct)')
    parser.add_argument('--force-refresh', action='store_true',
                        help='Force refresh of data when training')
    
    args = parser.parse_args()
    
    # Calculate default dates if not provided
    if args.start_date is None:
        args.start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    if args.end_date is None:
        args.end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # First, check if we need to initialize the AI model
    if args.mode == 'init_ai' or (args.use_ai and args.mode != 'train'):
        logger.info("Pre-loading AI model...")
        initialize_ai_model(args.ai_model_path)
    
    if args.mode == 'train':
        model_file = train_model(args.api_key, args.symbol, args.start_date, args.end_date, args.force_refresh)
    elif args.mode == 'backtest':
        # Check if model exists
        model_file = f"{args.symbol}_orb_model.joblib"
        if not os.path.isfile(model_file):
            logger.warning(f"Model file {model_file} not found. Training new model...")
            model_file = train_model(args.api_key, args.symbol, args.start_date, args.end_date)
        
        run_backtest(args.api_key, model_file, args.symbol, args.start_date, args.end_date)
    elif args.mode == 'optimize':
        # Check if model exists
        model_file = f"{args.symbol}_orb_model.joblib"
        if not os.path.isfile(model_file):
            logger.warning(f"Model file {model_file} not found. Training new model...")
            model_file = train_model(args.api_key, args.symbol, args.start_date, args.end_date)
        
        optimize_strategy(args.api_key, model_file, args.symbol, args.start_date, args.end_date)
    elif args.mode == 'live':
        # Check if model exists
        model_file = f"{args.symbol}_orb_model.joblib"
        if not os.path.isfile(model_file):
            logger.warning(f"Model file {model_file} not found. Training new model...")
            model_file = train_model(args.api_key, args.symbol, args.start_date, args.end_date)
        
        # Pass AI arguments to the assistant
        run_live_assistant(
            args.api_key, 
            model_file, 
            args.symbol,
            use_ai=args.use_ai,
            ai_model_path=args.ai_model_path
        )

if __name__ == "__main__":
    main()