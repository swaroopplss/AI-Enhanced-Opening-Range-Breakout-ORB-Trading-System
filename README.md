# AI-Enhanced Opening Range Breakout (ORB) Trading System

This system implements an advanced Opening Range Breakout (ORB) trading strategy enhanced with AI-based pattern recognition and decision making. It uses the DeepSeek-Coder model for local inference to provide sophisticated trading analysis without API costs.

## Inspiration and Learning Resources

### YouTube Tutorial Reference
**Video Source:** [Opening Range Breakout (ORB) Trading Strategy Explained by ScarfaceTrades](https://www.youtube.com/watch?v=7-Wg0dh0myM&t=1813s&ab_channel=ScarfaceTrades)

This project was inspired by and developed alongside the comprehensive YouTube tutorial by ScarfaceTrades. The video provides an in-depth explanation of the Opening Range Breakout (ORB) trading strategy, which served as the foundational framework for this AI-enhanced implementation. 

Key learnings from the video that influenced this project include:
- Detailed breakdown of the ORB trading methodology
- Insights into identifying and trading opening range breakouts
- Practical strategies for entry, stop loss, and target determination
- Understanding market dynamics during the first hour of trading

While the original video provides a manual approach, this project extends the concept by incorporating AI-driven analysis and decision-making processes.

## System Architecture

The system consists of six main components:

1. **Data Collector**: Fetches historical and live market data from Polygon.io
2. **Pattern Extractor**: Identifies ORB patterns in historical data
3. **Pattern Model**: Machine learning model for recognizing similar patterns
4. **AI Reasoning Engine**: Local DeepSeek model for advanced trade analysis
5. **Hybrid Reasoning Engine**: Combines rule-based and AI analysis
6. **Trading Assistant**: Monitors markets and executes the trading strategy

## Installation

### Prerequisites

- Python 3.8+
- Polygon.io API key
- 64GB+ RAM for optimal AI model performance

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd orb-trading
```

2. Create and activate a virtual environment:
```bash
conda create -n orb-trading python=3.10
conda activate orb-trading
```

3. Install dependencies:
```bash
pip install torch numpy pandas scikit-learn joblib matplotlib polygon-api-client transformers ta pytz
```

4. Download the DeepSeek model:
```bash
huggingface-cli download deepseek-ai/deepseek-coder-6.7b-instruct
```

## Usage

### 1. Initialize AI Model (first time only)

Pre-load the DeepSeek model to verify it works correctly:

```bash
python main.py --mode init_ai --api-key YOUR_POLYGON_API_KEY
```

### 2. Train the Pattern Recognition Model

Train the model on historical data:

```bash
python main.py --mode train --api-key YOUR_POLYGON_API_KEY --symbol QQQ --start-date 2023-01-01 --end-date 2023-12-31
```

For a fresh download of data, add `--force-refresh`.

### 3. Run Backtests

Test the strategy on historical data:

```bash
python main.py --mode backtest --api-key YOUR_POLYGON_API_KEY --symbol QQQ --start-date 2023-06-01 --end-date 2023-08-31 --use-ai
```

### 4. Optimize Strategy Parameters

Find the best parameters for the strategy:

```bash
python main.py --mode optimize --api-key YOUR_POLYGON_API_KEY --symbol QQQ --use-ai
```

### 5. Run Live Trading Assistant

Monitor the market in real-time:

```bash
python main.py --mode live --api-key YOUR_POLYGON_API_KEY --symbol QQQ --use-ai
```

## System Components

### Data Collector (`data_collector.py`)

Handles data retrieval from Polygon.io and identifies 5-minute opening ranges.

### Pattern Extractor (`orb_pattern_extractor.py`)

Extracts ORB patterns from historical data, identifying breakouts, retests, and outcomes.

### Pattern Model (`pattern_model.py`)

Uses machine learning to recognize patterns and predict outcome probabilities.

### AI Reasoning Engine (`ai_reasoning_engine.py`)

Implements three reasoning engines:
1. `LocalAIReasoningEngine`: Uses the DeepSeek model locally
2. `RuleBasedReasoningEngine`: Makes decisions based on predefined rules
3. `HybridReasoningEngine`: Combines the two approaches

### Trading Assistant (`orb_trading_assistant.py`)

Monitors live market data, detects setups, manages active trades, and runs backtests.

### Main Script (`main.py`)

Orchestrates all components and provides a command-line interface.

## ORB Strategy Details

The ORB strategy focuses on:

1. Identifying the price range established in the first 5 minutes after market open (9:30-9:35 AM)
2. Looking for breakouts above/below the opening range high/low
3. Waiting for prices to pull back and retest the breakout level
4. Entering trades at the retest with confirmation signals (candle patterns, volume, technical indicators)
5. Using AI to analyze patterns and make better trade decisions

## Trade Execution Rules

For LONG trades:
- Entry: On retest of OR high with bullish confirmation
- Stop Loss: Below entry or OR low
- Target: 50% extension of OR range

For SHORT trades:
- Entry: On retest of OR low with bearish confirmation
- Stop Loss: Above entry or OR high
- Target: 50% extension of OR range downward

## Continuous Learning

The system implements a memory mechanism to learn from previous trades:
1. Stores all trade analyses with outcomes
2. Refers to similar past trades when analyzing new setups
3. Updates AI reasoning based on actual trade results

## Future Enhancements

Possible improvements to consider:
1. Multi-timeframe analysis
2. Additional trade management rules (trailing stops, partial exits)
3. Integration with additional data sources
4. Support for more sophisticated trade entries and exits
5. Fine-tuning the AI model on trading-specific data

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Trading financial instruments carries significant risk. Past performance of any trading system is not necessarily indicative of future results.
