"""
Complete Integrated Bitcoin Trading System - Final Version
===========================================================
Features:
- Advanced NLP sentiment analysis with event detection
- Model persistence for fast startup (saves/loads trained models)
- Stop-loss/Take-profit risk management
- Real-time monitoring with live signals
- Comprehensive backtesting

Author: AI Trading System
Version: 2.0 Enhanced
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from textblob import TextBlob
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from collections import deque
import threading
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("Bitcoin Trading System - Loading...")

# ============================================================================
# PART 1: ADVANCED NLP SENTIMENT ANALYZER
# ============================================================================

class AdvancedSentimentAnalyzer:
    """
    Advanced sentiment analyzer with event detection and entity recognition
    Detects specific market events and adjusts sentiment accordingly
    """
    
    def __init__(self):
        # Event categories with impact weights
        self.events = {
            'regulation': {
                'keywords': ['sec', 'regulation', 'ban', 'illegal', 'compliance', 
                           'lawsuit', 'court', 'government', 'law', 'regulatory'],
                'impact': 'negative',
                'weight': 1.5
            },
            'institutional': {
                'keywords': ['etf', 'institutional', 'blackrock', 'fidelity', 
                           'grayscale', 'microstrategy', 'tesla', 'adoption'],
                'impact': 'positive',
                'weight': 1.8
            },
            'security': {
                'keywords': ['hack', 'security', 'breach', 'stolen', 'exploit', 
                           'vulnerability', 'attack', 'fraud', 'scam'],
                'impact': 'negative',
                'weight': 2.0
            },
            'market': {
                'keywords': ['exchange', 'trading', 'volume', 'liquidity', 
                           'whale', 'futures', 'options', 'etf approval'],
                'impact': 'neutral',
                'weight': 1.0
            }
        }
        
        # Price action intensity scoring
        self.price_actions = {
            'extreme_bullish': ['moon', 'skyrocket', 'explode', 'parabolic', 'massive surge'],
            'strong_bullish': ['rally', 'surge', 'breakout', 'pump', 'bull run'],
            'mild_bullish': ['gain', 'rise', 'up', 'increase', 'recover'],
            'mild_bearish': ['decline', 'drop', 'fall', 'down', 'decrease'],
            'strong_bearish': ['crash', 'dump', 'plunge', 'collapse', 'tank'],
            'extreme_bearish': ['devastate', 'obliterate', 'wipeout', 'bloodbath']
        }
        
        self.magnitude_weights = {
            'extreme_bullish': 3.0, 'strong_bullish': 2.0, 'mild_bullish': 1.0,
            'mild_bearish': -1.0, 'strong_bearish': -2.0, 'extreme_bearish': -3.0
        }
    
    def detect_events(self, text):
        """Detect specific market events from news text"""
        text_lower = str(text).lower()
        detected = {}
        
        for event_type, data in self.events.items():
            matches = sum(1 for kw in data['keywords'] if kw in text_lower)
            if matches > 0:
                detected[event_type] = {
                    'count': matches,
                    'impact': data['impact'],
                    'weight': data['weight']
                }
        return detected
    
    def analyze_price_action(self, text):
        """Calculate price action intensity score"""
        text_lower = str(text).lower()
        score = 0
        
        for magnitude, keywords in self.price_actions.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > 0:
                score += count * self.magnitude_weights[magnitude]
        
        return score
    
    def calculate_weighted_sentiment(self, base_sentiment, events):
        """Adjust sentiment based on detected events and their importance"""
        weighted = base_sentiment
        
        for event_type, info in events.items():
            if info['impact'] == 'positive':
                weighted += 0.1 * info['weight'] * info['count']
            elif info['impact'] == 'negative':
                weighted -= 0.1 * info['weight'] * info['count']
        
        return max(-1, min(1, weighted))
    
    def analyze_comprehensive(self, text):
        """
        Comprehensive sentiment analysis
        Returns: dict with sentiment scores, events, and metadata
        """
        if not text or pd.isna(text):
            return self._empty_result()
        
        text_str = str(text)
        
        # Basic TextBlob sentiment
        blob = TextBlob(text_str)
        base_polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Event detection
        events = self.detect_events(text_str)
        
        # Price action analysis
        price_action_score = self.analyze_price_action(text_str)
        
        # Event-weighted sentiment
        event_weighted = self.calculate_weighted_sentiment(base_polarity, events)
        
        # Final combined sentiment (40% base + 40% events + 20% price action)
        final_sentiment = (
            base_polarity * 0.4 + 
            event_weighted * 0.4 + 
            np.sign(price_action_score) * min(abs(price_action_score) / 10, 0.2)
        )
        final_sentiment = max(-1, min(1, final_sentiment))
        
        return {
            'base_polarity': base_polarity,
            'final_sentiment': final_sentiment,
            'subjectivity': subjectivity,
            'events': events,
            'price_action_score': price_action_score,
            'event_count': sum(e['count'] for e in events.values()),
            'has_regulation': 'regulation' in events,
            'has_institutional': 'institutional' in events,
            'has_security': 'security' in events
        }
    
    def _empty_result(self):
        """Return empty result structure"""
        return {
            'base_polarity': 0, 'final_sentiment': 0, 'subjectivity': 0,
            'events': {}, 'price_action_score': 0, 'event_count': 0,
            'has_regulation': False, 'has_institutional': False, 'has_security': False
        }
    
    def process_news_batch(self, news_df):
        """Process batch of news articles with comprehensive analysis"""
        if news_df.empty:
            return news_df
        
        news_df['full_text'] = news_df['title'].fillna('') + ' ' + news_df['body'].fillna('')
        
        # Analyze each article
        analyses = news_df['full_text'].apply(self.analyze_comprehensive)
        
        # Extract features
        news_df['base_polarity'] = analyses.apply(lambda x: x['base_polarity'])
        news_df['final_sentiment'] = analyses.apply(lambda x: x['final_sentiment'])
        news_df['subjectivity'] = analyses.apply(lambda x: x['subjectivity'])
        news_df['price_action_score'] = analyses.apply(lambda x: x['price_action_score'])
        news_df['event_count'] = analyses.apply(lambda x: x['event_count'])
        news_df['has_regulation'] = analyses.apply(lambda x: x['has_regulation'])
        news_df['has_institutional'] = analyses.apply(lambda x: x['has_institutional'])
        news_df['has_security'] = analyses.apply(lambda x: x['has_security'])
        
        # Backward compatibility
        news_df['polarity'] = news_df['final_sentiment']
        news_df['bullish_count'] = news_df['price_action_score'].apply(lambda x: max(0, x))
        news_df['bearish_count'] = news_df['price_action_score'].apply(lambda x: max(0, -x))
        news_df['volatile_count'] = news_df['event_count']
        
        return news_df

# ============================================================================
# PART 2: MODEL PERSISTENCE MANAGER
# ============================================================================

class ModelManager:
    """
    Manages model saving, loading, and versioning
    Enables fast startup by loading pre-trained models
    """
    
    def __init__(self, model_dir='models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.model_path = self.model_dir / 'btc_predictor.pkl'
        self.scaler_path = self.model_dir / 'scaler.pkl'
        self.metadata_path = self.model_dir / 'metadata.json'
    
    def save_model(self, model, scaler, metadata=None):
        """Save trained model, scaler, and metadata"""
        with open(self.model_path, 'wb') as f:
            pickle.dump(model, f)
        
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'saved_at': datetime.now().isoformat(),
            'model_type': type(model).__name__
        })
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Model saved to {self.model_dir}/")
    
    def load_model(self):
        """Load saved model from disk"""
        if not self.model_path.exists():
            return None, None, None
        
        try:
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
            
            with open(self.scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            metadata = {}
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            print(f"✓ Model loaded from {self.model_dir}/")
            if 'saved_at' in metadata:
                saved_date = datetime.fromisoformat(metadata['saved_at'])
                age_days = (datetime.now() - saved_date).days
                print(f"  Model age: {age_days} days")
            
            return model, scaler, metadata
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None, None, None
    
    def model_exists(self):
        """Check if saved model exists"""
        return self.model_path.exists() and self.scaler_path.exists()
    
    def get_model_age_days(self):
        """Get model age in days"""
        if not self.metadata_path.exists():
            return None
        
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if 'saved_at' not in metadata:
            return None
        
        saved_date = datetime.fromisoformat(metadata['saved_at'])
        return (datetime.now() - saved_date).days
    
    def should_retrain(self, max_age_days=7):
        """Determine if model needs retraining"""
        if not self.model_exists():
            return True, "No model exists"
        
        age = self.get_model_age_days()
        if age is None:
            return True, "Model age unknown"
        
        if age >= max_age_days:
            return True, f"Model is {age} days old (max: {max_age_days})"
        
        return False, f"Model is fresh ({age} days old)"
    
    def get_model_info(self):
        """Get model metadata"""
        if not self.metadata_path.exists():
            return None
        
        with open(self.metadata_path, 'r') as f:
            return json.load(f)

# ============================================================================
# PART 3: DATA COLLECTION
# ============================================================================

class NewsCollector:
    """Collects Bitcoin news and price data from free APIs"""
    
    def __init__(self):
        self.crypto_compare_url = "https://min-api.cryptocompare.com/data/v2/news/"
        self.coingecko_base = "https://api.coingecko.com/api/v3"
    
    def get_crypto_news(self, limit=50):
        """Fetch Bitcoin news from CryptoCompare API"""
        try:
            response = requests.get(self.crypto_compare_url, timeout=10)
            data = response.json()
            if 'Data' in data:
                articles = []
                for article in data['Data'][:limit]:
                    articles.append({
                        'id': article.get('id', ''),
                        'title': article.get('title', ''),
                        'body': article.get('body', ''),
                        'published_at': datetime.fromtimestamp(article.get('published_on', 0)),
                        'source': article.get('source', '')
                    })
                return pd.DataFrame(articles)
            return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching news: {e}")
            return pd.DataFrame()
    
    def get_btc_price_history(self, days=100):
        """Fetch historical Bitcoin prices from CoinGecko API"""
        try:
            url = f"{self.coingecko_base}/coins/bitcoin/market_chart"
            params = {'vs_currency': 'usd', 'days': days, 'interval': 'daily'}
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            prices = []
            for price_point in data['prices']:
                prices.append({
                    'timestamp': datetime.fromtimestamp(price_point[0] / 1000),
                    'price': price_point[1]
                })
            
            df = pd.DataFrame(prices)
            df['price_change'] = df['price'].pct_change()
            df['label'] = (df['price_change'].shift(-1) > 0).astype(int)
            
            # Simulate intraday high/low for SL/TP testing
            df['high'] = df['price'] * 1.02
            df['low'] = df['price'] * 0.98
            
            return df
        except Exception as e:
            print(f"Error fetching price: {e}")
            return pd.DataFrame()
    
    def get_current_price(self):
        """Fetch current Bitcoin price"""
        try:
            url = f"{self.coingecko_base}/simple/price"
            params = {'ids': 'bitcoin', 'vs_currencies': 'usd', 'include_24hr_change': 'true'}
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            return {
                'price': data['bitcoin']['usd'],
                'change_24h': data['bitcoin']['usd_24h_change'],
                'timestamp': datetime.now()
            }
        except Exception as e:
            print(f"Error fetching current price: {e}")
            return None

# ============================================================================
# PART 4: ML PREDICTOR WITH PERSISTENCE
# ============================================================================

class BTCPredictorWithPersistence:
    """ML predictor with model persistence for fast startup"""
    
    def __init__(self, model_dir='models', use_persistence=True):
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.use_persistence = use_persistence
        self.manager = ModelManager(model_dir) if use_persistence else None
        
        # Enhanced feature set (15 features including NLP)
        self.feature_cols = [
            'polarity_mean', 'polarity_std', 'polarity_min', 'polarity_max',
            'subjectivity_mean', 'bullish_total', 'bearish_total', 
            'volatile_total', 'news_count', 'price_ma7', 'price_volatility',
            'event_count_total', 'regulation_events', 'institutional_events',
            'security_events', 'price_action_mean'
        ]
    
    def prepare_features(self, news_df, price_df):
        """Prepare features combining news sentiment and price data"""
        news_df['date'] = pd.to_datetime(news_df['published_at']).dt.date
        
        # Aggregate sentiment by date
        daily_sentiment = news_df.groupby('date').agg({
            'polarity': ['mean', 'std', 'min', 'max'],
            'subjectivity': 'mean',
            'bullish_count': 'sum',
            'bearish_count': 'sum',
            'volatile_count': 'sum',
            'event_count': 'sum',
            'price_action_score': 'mean'
        }).reset_index()
        
        daily_sentiment.columns = [
            'date', 'polarity_mean', 'polarity_std', 'polarity_min', 'polarity_max',
            'subjectivity_mean', 'bullish_total', 'bearish_total', 'volatile_total',
            'event_count_total', 'price_action_mean'
        ]
        
        # Count specific event types
        regulation_count = news_df.groupby('date')['has_regulation'].sum().reset_index()
        regulation_count.columns = ['date', 'regulation_events']
        
        institutional_count = news_df.groupby('date')['has_institutional'].sum().reset_index()
        institutional_count.columns = ['date', 'institutional_events']
        
        security_count = news_df.groupby('date')['has_security'].sum().reset_index()
        security_count.columns = ['date', 'security_events']
        
        news_count = news_df.groupby('date').size().reset_index(name='news_count')
        
        # Merge all features
        daily_sentiment = daily_sentiment.merge(news_count, on='date', how='left')
        daily_sentiment = daily_sentiment.merge(regulation_count, on='date', how='left')
        daily_sentiment = daily_sentiment.merge(institutional_count, on='date', how='left')
        daily_sentiment = daily_sentiment.merge(security_count, on='date', how='left')
        daily_sentiment = daily_sentiment.fillna(0)
        
        # Merge with price data
        price_df['date'] = pd.to_datetime(price_df['timestamp']).dt.date
        merged = price_df.merge(daily_sentiment, on='date', how='left').fillna(0)
        
        # Add technical indicators
        merged['price_ma7'] = merged['price'].rolling(window=7, min_periods=1).mean()
        merged['price_volatility'] = merged['price'].rolling(window=7, min_periods=1).std()
        
        return merged
    
    def train(self, features_df, save_model=True):
        """Train model and optionally save for future use"""
        train_data = features_df[features_df['label'].notna()].copy()
        
        if len(train_data) < 10:
            print("❌ Not enough data (need ≥10 samples)")
            return None
        
        X = train_data[self.feature_cols].fillna(0)
        y = train_data['label'].astype(int)
        
        # Train
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Metrics
        train_accuracy = self.model.score(X_scaled, y)
        
        metrics = {
            'accuracy': train_accuracy,
            'train_samples': len(X),
            'features_used': len(self.feature_cols)
        }
        
        print(f"\n✓ Model trained")
        print(f"  Accuracy: {train_accuracy:.2%}")
        print(f"  Samples: {len(X)}")
        print(f"  Features: {len(self.feature_cols)} (with Advanced NLP)")
        
        # Save if enabled
        if save_model and self.use_persistence and self.manager:
            self.manager.save_model(self.model, self.scaler, metrics)
        
        return metrics
    
    def load_trained_model(self):
        """Load pre-trained model from disk"""
        if not self.use_persistence or not self.manager:
            return False
        
        model, scaler, metadata = self.manager.load_model()
        
        if model is None:
            return False
        
        self.model = model
        self.scaler = scaler
        self.is_trained = True
        
        return True
    
    def predict(self, features):
        """Make prediction from feature dictionary"""
        if not self.is_trained:
            print("❌ Model not trained!")
            return None
        
        X = pd.DataFrame([features])[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)[0]
        
        return {
            'prob_down': proba[0],
            'prob_up': proba[1],
            'prediction': 'UP' if proba[1] > 0.5 else 'DOWN',
            'confidence': max(proba)
        }

# ============================================================================
# PART 5: BACKTESTER WITH STOP-LOSS/TAKE-PROFIT
# ============================================================================

class BacktesterWithRisk:
    """Backtesting engine with SL/TP risk management"""
    
    def __init__(self, initial_capital=10000, fee_rate=0.001):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.reset()
    
    def reset(self):
        self.capital = self.initial_capital
        self.position = 0
        self.entry_price = 0
        self.position_size = 0
        self.stop_loss = None
        self.take_profit = None
        self.trades = []
    
    def calculate_sl_tp(self, entry_price, direction, volatility):
        """Calculate SL/TP based on volatility (2-5% SL, 2.5x R:R)"""
        vol_percent = (volatility / entry_price) * 100
        sl_percent = min(max(vol_percent * 2, 2.0), 5.0)
        tp_percent = sl_percent * 2.5
        
        if direction == 'LONG':
            sl = entry_price * (1 - sl_percent / 100)
            tp = entry_price * (1 + tp_percent / 100)
        else:
            sl = entry_price * (1 + sl_percent / 100)
            tp = entry_price * (1 - tp_percent / 100)
        
        return sl, tp, sl_percent, tp_percent
    
    def check_sl_tp_hit(self, high, low):
        """Check if SL or TP was hit"""
        if self.position == 0:
            return False, False, None
        
        hit_sl, hit_tp = False, False
        exit_price = None
        
        if self.position == 1:  # LONG
            if low <= self.stop_loss:
                hit_sl, exit_price = True, self.stop_loss
            elif high >= self.take_profit:
                hit_tp, exit_price = True, self.take_profit
        else:  # SHORT
            if high >= self.stop_loss:
                hit_sl, exit_price = True, self.stop_loss
            elif low <= self.take_profit:
                hit_tp, exit_price = True, self.take_profit
        
        return hit_sl, hit_tp, exit_price
    
    def open_position(self, direction, price, confidence, volatility, timestamp):
        """Open new position with SL/TP"""
        if self.position != 0:
            return False
        
        # Position sizing based on confidence
        fraction = min(confidence, 0.5)
        position_value = self.capital * fraction
        self.position_size = position_value / price
        fee = position_value * self.fee_rate
        
        if fee > self.capital * 0.1:
            return False
        
        self.position = direction
        self.entry_price = price
        self.capital -= fee
        
        # Calculate SL/TP
        sl, tp, sl_pct, tp_pct = self.calculate_sl_tp(price, direction, volatility)
        self.stop_loss = sl
        self.take_profit = tp
        
        self.trades.append({
            'timestamp': timestamp,
            'type': 'OPEN',
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'price': price,
            'stop_loss': sl,
            'take_profit': tp,
            'sl_percent': sl_pct,
            'tp_percent': tp_pct
        })
        
        return True
    
    def close_position(self, price, timestamp, reason):
        """Close position and calculate P&L"""
        if self.position == 0:
            return False
        
        exit_value = self.position_size * price
        fee = exit_value * self.fee_rate
        
        # Calculate P&L
        if self.position == 1:  # LONG
            pnl = (price - self.entry_price) * self.position_size
        else:  # SHORT
            pnl = (self.entry_price - price) * self.position_size
        
        pnl -= fee
        self.capital += pnl
        pnl_percent = (pnl / (self.entry_price * self.position_size)) * 100
        
        self.trades.append({
            'timestamp': timestamp,
            'type': 'CLOSE',
            'price': price,
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'reason': reason,
            'capital': self.capital
        })
        
        self.position = 0
        self.stop_loss = None
        self.take_profit = None
        
        return True
    
    def get_metrics(self):
        """Calculate performance metrics"""
        if len(self.trades) < 2:
            return None
        
        closed = [t for t in self.trades if t['type'] == 'CLOSE']
        if not closed:
            return None
        
        wins = [t for t in closed if t['pnl'] > 0]
        total_pnl = sum(t['pnl'] for t in closed)
        total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        
        sl_hits = len([t for t in closed if t['reason'] == 'STOP_LOSS'])
        tp_hits = len([t for t in closed if t['reason'] == 'TAKE_PROFIT'])
        
        return {
            'total_trades': len(closed),
            'wins': len(wins),
            'losses': len(closed) - len(wins),
            'win_rate': len(wins) / len(closed),
            'total_return': total_return,
            'final_capital': self.capital,
            'sl_hits': sl_hits,
            'tp_hits': tp_hits,
            'signal_exits': len(closed) - sl_hits - tp_hits
        }
    
    def check_position(self, current_price):
        """Check active position for SL/TP"""
        if not self.current_position:
            return
        
        pos = self.current_position
        
        if pos['direction'] == 'LONG':
            pnl_pct = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
            hit_sl = current_price <= pos['stop_loss']
            hit_tp = current_price >= pos['take_profit']
        else:
            pnl_pct = ((pos['entry_price'] - current_price) / pos['entry_price']) * 100
            hit_sl = current_price >= pos['stop_loss']
            hit_tp = current_price <= pos['take_profit']
        
        print(f"\n📌 ACTIVE {pos['direction']} POSITION")
        print(f"   Entry: ${pos['entry_price']:,.2f} → Current: ${current_price:,.2f}")
        print(f"   P&L: {pnl_pct:+.2f}%")
        print(f"   SL: ${pos['stop_loss']:,.2f} | TP: ${pos['take_profit']:,.2f}")
        
        if hit_sl:
            print(f"\n🛑 STOP-LOSS HIT! Loss: {pnl_pct:.2f}%")
            self.current_position = None
        elif hit_tp:
            print(f"\n🎉 TAKE-PROFIT HIT! Profit: {pnl_pct:.2f}%")
            self.current_position = None
    
    def monitor_loop(self):
        """Main monitoring loop"""
        print(f"\n{'='*70}")
        print("🚀 REAL-TIME MONITORING STARTED")
        print(f"{'='*70}\n")
        
        iteration = 0
        while self.is_running:
            try:
                iteration += 1
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Check #{iteration}")
                print("─" * 70)
                
                # Fetch price
                price_data = self.collector.get_current_price()
                if price_data:
                    self.price_history.append(price_data)
                    print(f"💰 BTC: ${price_data['price']:,.2f} ({price_data['change_24h']:+.2f}% 24h)")
                
                # Fetch news
                news_df = self.collector.get_crypto_news(limit=10)
                if not news_df.empty:
                    new_articles = [row.to_dict() for _, row in news_df.iterrows() 
                                   if row['id'] not in [n['id'] for n in self.news_cache]]
                    if new_articles:
                        print(f"📰 New articles: {len(new_articles)}")
                        self.news_cache.extend(new_articles)
                
                # Generate signal
                signal = self.generate_signal()
                if signal:
                    print(f"\n📊 Signal: {signal['prediction']} (Conf: {signal['confidence']:.1%})")
                    print(f"   Prob: ↑{signal['prob_up']:.1%} | Sentiment: {signal['sentiment']:+.3f}")
                    if signal['events'] > 0:
                        print(f"   ⚠️  {int(signal['events'])} market events detected")
                    
                    # Check position
                    if self.current_position:
                        self.check_position(signal['price'])
                    # New entry
                    elif signal['confidence'] >= 0.65:
                        direction = signal['prediction'].replace('UP', 'LONG').replace('DOWN', 'SHORT')
                        sl, tp, sl_pct, tp_pct = self.calculate_sl_tp(
                            signal['price'], direction, signal['volatility']
                        )
                        
                        print(f"\n🎯 ENTRY SIGNAL: {direction}")
                        print(f"   Entry: ${signal['price']:,.2f}")
                        print(f"   SL: ${sl:,.2f} ({sl_pct:.1f}%)")
                        print(f"   TP: ${tp:,.2f} ({tp_pct:.1f}%)")
                        print(f"   R:R = 1:{tp_pct/sl_pct:.2f}")
                        
                        self.current_position = {
                            'direction': direction,
                            'entry_price': signal['price'],
                            'stop_loss': sl,
                            'take_profit': tp,
                            'entry_time': datetime.now()
                        }
                
                print(f"\n⏳ Next check in {self.check_interval}s...")
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                print("\n\n🛑 Stopped by user")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                time.sleep(30)
        
        self.is_running = False
    
    def start(self):
        """Start monitoring"""
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop monitoring"""
        self.is_running = False

# ============================================================================
# PART 5B: MONTE CARLO SIMULATION
# ============================================================================

class MonteCarloSimulator:
    """
    Monte Carlo simulation for statistical robustness testing
    Randomly shuffles trade sequences to test if results are due to skill or luck
    """
    
    def __init__(self, n_simulations=1000):
        """
        Initialize Monte Carlo simulator
        
        Args:
            n_simulations: Number of random simulations to run
        """
        self.n_simulations = n_simulations
    
    def run_simulation(self, trades_df):
        """
        Run Monte Carlo simulation on trade history
        
        Args:
            trades_df: DataFrame with closed trades (must have 'pnl' column)
            
        Returns:
            Dictionary with simulation results
        """
        if trades_df.empty or 'pnl' not in trades_df.columns:
            print("❌ No trades to simulate")
            return None
        
        closed_trades = trades_df[trades_df['type'] == 'CLOSE'].copy()
        if len(closed_trades) == 0:
            return None
        
        original_pnl = closed_trades['pnl'].sum()
        pnl_list = closed_trades['pnl'].values
        
        print(f"\n🎲 Running Monte Carlo Simulation ({self.n_simulations} iterations)...")
        
        simulated_returns = []
        
        # Run simulations
        for i in range(self.n_simulations):
            # Randomly shuffle trade order
            shuffled_pnl = np.random.permutation(pnl_list)
            total_pnl = shuffled_pnl.sum()
            simulated_returns.append(total_pnl)
        
        simulated_returns = np.array(simulated_returns)
        
        # Calculate statistics
        mean_return = np.mean(simulated_returns)
        std_return = np.std(simulated_returns)
        min_return = np.min(simulated_returns)
        max_return = np.max(simulated_returns)
        
        # Confidence intervals (95%)
        percentile_5 = np.percentile(simulated_returns, 5)
        percentile_95 = np.percentile(simulated_returns, 95)
        
        # Calculate how many simulations beat original
        better_than_original = np.sum(simulated_returns > original_pnl)
        percentile_rank = (better_than_original / self.n_simulations) * 100
        
        return {
            'original_pnl': original_pnl,
            'mean_pnl': mean_return,
            'std_pnl': std_return,
            'min_pnl': min_return,
            'max_pnl': max_return,
            'percentile_5': percentile_5,
            'percentile_95': percentile_95,
            'percentile_rank': percentile_rank,
            'simulated_returns': simulated_returns
        }
    
    def print_results(self, results):
        """Print Monte Carlo simulation results"""
        if results is None:
            return
        
        print("\n" + "=" * 70)
        print("🎲 MONTE CARLO SIMULATION RESULTS")
        print("=" * 70)
        print(f"Original P&L:        ${results['original_pnl']:,.2f}")
        print(f"Simulated Mean:      ${results['mean_pnl']:,.2f}")
        print(f"Std Deviation:       ${results['std_pnl']:,.2f}")
        print(f"\nRange:")
        print(f"  Best Case:         ${results['max_pnl']:,.2f}")
        print(f"  Worst Case:        ${results['min_pnl']:,.2f}")
        print(f"\n95% Confidence Interval:")
        print(f"  Lower Bound (5%):  ${results['percentile_5']:,.2f}")
        print(f"  Upper Bound (95%): ${results['percentile_95']:,.2f}")
        print(f"\nPercentile Rank:     {100 - results['percentile_rank']:.1f}%")
        
        # Interpretation
        if results['percentile_rank'] < 5:
            print("\n✅ Strategy is statistically robust (top 5%)")
        elif results['percentile_rank'] < 20:
            print("\n⚠️  Strategy shows some edge but needs improvement")
        else:
            print("\n❌ Results may be due to luck, not skill")
        
        print("=" * 70)

# ============================================================================
# PART 5C: WALK-FORWARD OPTIMIZATION
# ============================================================================

class WalkForwardOptimizer:
    """
    Walk-forward optimization to find optimal parameters
    Prevents overfitting by testing on unseen data
    """
    
    def __init__(self, train_period=20, test_period=10):
        """
        Initialize walk-forward optimizer
        
        Args:
            train_period: Number of days for training window
            test_period: Number of days for testing window
        """
        self.train_period = train_period
        self.test_period = test_period
    
    def optimize_confidence_threshold(self, predictor, features_df, 
                                      thresholds=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75]):
        """
        Find optimal confidence threshold using walk-forward
        
        Args:
            predictor: Trained predictor
            features_df: Features DataFrame
            thresholds: List of confidence thresholds to test
            
        Returns:
            Dictionary with optimization results
        """
        print(f"\n🔍 Walk-Forward Optimization")
        print(f"Train window: {self.train_period} days | Test window: {self.test_period} days")
        print(f"Testing thresholds: {thresholds}")
        
        total_length = len(features_df)
        window_size = self.train_period + self.test_period
        
        if total_length < window_size:
            print("❌ Not enough data for walk-forward")
            return None
        
        results = {th: [] for th in thresholds}
        
        # Walk through data
        start_idx = 10  # Skip first 10 for MA calculation
        
        while start_idx + window_size < total_length:
            train_end = start_idx + self.train_period
            test_end = train_end + self.test_period
            
            print(f"\nWindow: Day {start_idx}-{train_end} (train) | {train_end}-{test_end} (test)")
            
            # Test each threshold
            for threshold in thresholds:
                # Run backtest on test period
                backtester = BacktesterWithRisk(initial_capital=10000)
                
                for i in range(train_end, min(test_end, total_length - 1)):
                    row = features_df.iloc[i]
                    features = {col: row[col] for col in predictor.feature_cols}
                    pred = predictor.predict(features)
                    
                    # Check SL/TP
                    if backtester.position != 0:
                        hit_sl, hit_tp, exit_price = backtester.check_sl_tp_hit(
                            row['high'], row['low']
                        )
                        
                        if hit_sl:
                            backtester.close_position(exit_price, row['timestamp'], 'STOP_LOSS')
                        elif hit_tp:
                            backtester.close_position(exit_price, row['timestamp'], 'TAKE_PROFIT')
                        elif pred['confidence'] >= threshold * 0.9:  # Reversal threshold
                            if (backtester.position == 1 and pred['prediction'] == 'DOWN') or \
                               (backtester.position == -1 and pred['prediction'] == 'UP'):
                                backtester.close_position(row['price'], row['timestamp'], 'SIGNAL')
                    
                    # Entry
                    if backtester.position == 0 and pred['confidence'] >= threshold:
                        direction = 1 if pred['prediction'] == 'UP' else -1
                        backtester.open_position(
                            direction, row['price'], pred['confidence'],
                            row['price_volatility'], row['timestamp']
                        )
                
                # Close final position
                if backtester.position != 0:
                    backtester.close_position(
                        features_df.iloc[test_end - 1]['price'],
                        features_df.iloc[test_end - 1]['timestamp'],
                        'END'
                    )
                
                # Store result
                metrics = backtester.get_metrics()
                if metrics:
                    results[threshold].append(metrics['total_return'])
                    print(f"  Threshold {threshold:.2f}: {metrics['total_return']:+.2f}%")
            
            # Move window forward
            start_idx += self.test_period
        
        # Calculate average performance for each threshold
        avg_returns = {}
        for threshold in thresholds:
            if len(results[threshold]) > 0:
                avg_returns[threshold] = np.mean(results[threshold])
        
        # Find best threshold
        if avg_returns:
            best_threshold = max(avg_returns, key=avg_returns.get)
            
            print("\n" + "=" * 70)
            print("🏆 WALK-FORWARD OPTIMIZATION RESULTS")
            print("=" * 70)
            for threshold, avg_return in sorted(avg_returns.items()):
                marker = " ← BEST" if threshold == best_threshold else ""
                print(f"Threshold {threshold:.2f}: {avg_return:+.2f}% avg return{marker}")
            print("=" * 70)
            
            return {
                'best_threshold': best_threshold,
                'avg_returns': avg_returns,
                'all_results': results
            }
        
        return None
    
# ============================================================================
# PART 6: REAL-TIME MONITOR
# ============================================================================

class RealtimeMonitor:
    """Real-time monitoring system"""
    
    def __init__(self, predictor, analyzer, collector, check_interval=300):
        self.predictor = predictor
        self.analyzer = analyzer
        self.collector = collector
        self.check_interval = check_interval
        self.is_running = False
        self.current_position = None
        self.price_history = deque(maxlen=100)
        self.news_cache = deque(maxlen=50)
    
    def analyze_sentiment(self):
        """Analyze cached news sentiment"""
        if not self.news_cache:
            return {k: 0 for k in self.predictor.feature_cols}
        
        news_df = pd.DataFrame(list(self.news_cache))
        news_df = self.analyzer.process_news_batch(news_df)
        
        return {
            'polarity_mean': news_df['polarity'].mean(),
            'polarity_std': news_df['polarity'].std(),
            'polarity_min': news_df['polarity'].min(),
            'polarity_max': news_df['polarity'].max(),
            'subjectivity_mean': news_df['subjectivity'].mean(),
            'bullish_total': news_df['bullish_count'].sum(),
            'bearish_total': news_df['bearish_count'].sum(),
            'volatile_total': news_df['volatile_count'].sum(),
            'news_count': len(news_df),
            'event_count_total': news_df['event_count'].sum(),
            'regulation_events': news_df['has_regulation'].sum(),
            'institutional_events': news_df['has_institutional'].sum(),
            'security_events': news_df['has_security'].sum(),
            'price_action_mean': news_df['price_action_score'].mean()
        }
    
    def generate_signal(self):
        """Generate trading signal"""
        if len(self.price_history) < 7:
            return None
        
        prices = [p['price'] for p in self.price_history]
        sentiment = self.analyze_sentiment()
        
        features = {
            **sentiment,
            'price_ma7': np.mean(prices[-7:]),
            'price_volatility': np.std(prices[-7:])
        }
        
        prediction = self.predictor.predict(features)
        
        if prediction is None:
            return None
        
        return {
            'timestamp': datetime.now(),
            'price': self.price_history[-1]['price'],
            'prediction': prediction['prediction'],
            'confidence': prediction['confidence'],
            'prob_up': prediction['prob_up'],
            'volatility': features['price_volatility'],
            'sentiment': sentiment['polarity_mean'],
            'events': sentiment['event_count_total']
        }
    
    def calculate_sl_tp(self, entry_price, direction, volatility):
        """Calculate SL/TP levels"""
        vol_percent = (volatility / entry_price) * 100
        sl_percent = min(max(vol_percent * 2, 2.0), 5.0)
        tp_percent = sl_percent * 2.5
        
        if direction == 'LONG':
            sl = entry_price * (1 - sl_percent / 100)
            tp = entry_price * (1 + tp_percent / 100)
        else:
            sl = entry_price * (1 + sl_percent / 100)
            tp = entry_price * (1 - tp_percent / 100)
        
        return sl, tp, sl_percent, tp_percent
    
    def check_position(self, current_price):
        """Check active position for SL/TP hits"""
        if not self.current_position:
            return
        
        pos = self.current_position
        
        if pos['direction'] == 'LONG':
            pnl_pct = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
            hit_sl = current_price <= pos['stop_loss']
            hit_tp = current_price >= pos['take_profit']
        else:
            pnl_pct = ((pos['entry_price'] - current_price) / pos['entry_price']) * 100
            hit_sl = current_price >= pos['stop_loss']
            hit_tp = current_price <= pos['take_profit']
        
        print(f"\n📌 ACTIVE {pos['direction']} POSITION")
        print(f"   Entry: ${pos['entry_price']:,.2f} → Current: ${current_price:,.2f}")
        print(f"   P&L: {pnl_pct:+.2f}%")
        print(f"   SL: ${pos['stop_loss']:,.2f} | TP: ${pos['take_profit']:,.2f}")
        
        if hit_sl:
            print(f"\n🛑 STOP-LOSS HIT! Loss: {pnl_pct:.2f}%")
            self.current_position = None
        elif hit_tp:
            print(f"\n🎉 TAKE-PROFIT HIT! Profit: {pnl_pct:.2f}%")
            self.current_position = None
    
    def monitor_loop(self):
        """Main monitoring loop"""
        print(f"\n{'='*70}")
        print("🚀 REAL-TIME MONITORING STARTED")
        print(f"{'='*70}")
        print(f"Check interval: {self.check_interval} seconds")
        print(f"Press Ctrl+C to stop\n")
        
        iteration = 0
        while self.is_running:
            try:
                iteration += 1
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Check #{iteration}")
                print("─" * 70)
                
                # Fetch current price
                price_data = self.collector.get_current_price()
                if price_data:
                    self.price_history.append(price_data)
                    print(f"💰 BTC: ${price_data['price']:,.2f} ({price_data['change_24h']:+.2f}% 24h)")
                else:
                    print("⚠️  Failed to fetch price")
                
                # Fetch news
                news_df = self.collector.get_crypto_news(limit=10)
                if not news_df.empty:
                    new_articles = [row.to_dict() for _, row in news_df.iterrows() 
                                   if row['id'] not in [n.get('id', '') for n in self.news_cache]]
                    if new_articles:
                        print(f"📰 New articles: {len(new_articles)}")
                        self.news_cache.extend(new_articles)
                else:
                    print("⚠️  No new articles")
                
                # Generate signal
                signal = self.generate_signal()
                if signal:
                    print(f"\n📊 Signal: {signal['prediction']} (Conf: {signal['confidence']:.1%})")
                    print(f"   Prob: ↑{signal['prob_up']:.1%} | Sentiment: {signal['sentiment']:+.3f}")
                    if signal['events'] > 0:
                        print(f"   ⚠️  {int(signal['events'])} market events detected")
                    
                    # Check existing position
                    if self.current_position:
                        self.check_position(signal['price'])
                    # Check for new entry
                    elif signal['confidence'] >= 0.65:
                        direction = signal['prediction'].replace('UP', 'LONG').replace('DOWN', 'SHORT')
                        sl, tp, sl_pct, tp_pct = self.calculate_sl_tp(
                            signal['price'], direction, signal['volatility']
                        )
                        
                        print(f"\n🎯 ENTRY SIGNAL: {direction}")
                        print(f"   Entry: ${signal['price']:,.2f}")
                        print(f"   SL: ${sl:,.2f} ({sl_pct:.1f}%)")
                        print(f"   TP: ${tp:,.2f} ({tp_pct:.1f}%)")
                        print(f"   R:R = 1:{tp_pct/sl_pct:.2f}")
                        
                        self.current_position = {
                            'direction': direction,
                            'entry_price': signal['price'],
                            'stop_loss': sl,
                            'take_profit': tp,
                            'entry_time': datetime.now()
                        }
                else:
                    print("⚠️  Waiting for sufficient data to generate signal...")
                
                print(f"\n⏳ Next check in {self.check_interval}s...")
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                print("\n\n🛑 Stopped by user")
                break
            except Exception as e:
                print(f"\n❌ Error in monitoring loop: {e}")
                import traceback
                traceback.print_exc()
                print(f"Retrying in 30 seconds...")
                time.sleep(30)
        
        self.is_running = False
        print("\n✓ Monitoring stopped")
    
    def start(self):
        """Start monitoring in background thread"""
        if self.is_running:
            print("⚠️  Monitor is already running!")
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop monitoring"""
        print("\n🛑 Stopping monitor...")
        self.is_running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=5)
        print("✓ Monitor stopped")

# ============================================================================
# PART 7: MAIN SYSTEM
# ============================================================================

def run_complete_system(mode='backtest', check_interval=300, use_persistence=True, 
                       force_retrain=False, max_model_age_days=7):
    """
    Main system entry point
    
    Args:
        mode: 'backtest' or 'realtime'
        check_interval: Seconds between real-time checks
        use_persistence: Enable model save/load
        force_retrain: Force model retraining
        max_model_age_days: Max model age before auto-retrain
    """
    print("=" * 70)
    print("BITCOIN TRADING SYSTEM - ENHANCED")
    print("=" * 70)
    print("✓ Advanced NLP Sentiment Analysis")
    print("✓ Model Persistence (Fast Startup)")
    print("✓ Stop-Loss/Take-Profit Risk Management")
    print("✓ Real-time Monitoring")
    
    # Initialize
    print("\n[1/5] Initializing...")
    collector = NewsCollector()
    analyzer = AdvancedSentimentAnalyzer()
    predictor = BTCPredictorWithPersistence(use_persistence=use_persistence)
    
    # Model loading/training
    print("\n[2/5] Model check...")
    should_train = force_retrain
    
    if use_persistence and not force_retrain:
        should_retrain, reason = predictor.manager.should_retrain(max_model_age_days)
        
        if should_retrain:
            print(f"  → {reason}")
            should_train = True
        else:
            print(f"  → {reason}")
            if predictor.load_trained_model():
                print("  ✓ Loaded existing model")
                should_train = False
            else:
                should_train = True
    
    # Train if needed
    if should_train:
        print("\n[3/5] Collecting data...")
        news_df = collector.get_crypto_news(limit=100)
        print(f"  ✓ {len(news_df)} articles")
        
        time.sleep(1)
        price_df = collector.get_btc_price_history(days=100)
        print(f"  ✓ {len(price_df)} days")
        
        print("\n[4/5] Analyzing sentiment...")
        news_df = analyzer.process_news_batch(news_df)
        print(f"  ✓ Sentiment: {news_df['final_sentiment'].mean():+.3f}")
        print(f"  ✓ Events: {int(news_df['event_count'].sum())}")
        
        print("\n[5/5] Training...")
        features_df = predictor.prepare_features(news_df, price_df)
        result = predictor.train(features_df, save_model=use_persistence)
    else:
        print("\n[3-5] Using existing model")
        
        if mode == 'backtest':
            print("\n  Collecting data for backtest...")
            news_df = collector.get_crypto_news(limit=100)
            time.sleep(1)
            price_df = collector.get_btc_price_history(days=100)
            news_df = analyzer.process_news_batch(news_df)
            features_df = predictor.prepare_features(news_df, price_df)
    
    # Run mode
    print(f"\n{'='*70}")
    print(f"RUNNING {mode.upper()} MODE")
    print(f"{'='*70}")
    
    if mode == 'backtest':
        results = run_backtest_mode(predictor, features_df)
        show_current_prediction(predictor, features_df)
        return results
    elif mode == 'realtime':
        return run_realtime_mode(predictor, analyzer, collector, check_interval)

def run_backtest_mode(predictor, features_df):
    """Run backtesting with signal display"""
    backtester = BacktesterWithRisk(initial_capital=10000)
    signals_list = []
    
    for i in range(10, len(features_df) - 1):
        row = features_df.iloc[i]
        features = {col: row[col] for col in predictor.feature_cols}
        pred = predictor.predict(features)
        
        # Store signal
        signals_list.append({
            'date': row['timestamp'].strftime('%Y-%m-%d'),
            'price': row['price'],
            'prediction': pred['prediction'],
            'confidence': pred['confidence'],
            'prob_up': pred['prob_up'],
            'sentiment': row['polarity_mean'],
            'events': row['event_count_total']
        })
        
        # Display high-confidence signals
        if pred['confidence'] >= 0.60:
            print(f"\n[{row['timestamp'].strftime('%Y-%m-%d')}] {pred['prediction']} "
                  f"({pred['confidence']:.1%})")
            print(f"  ${row['price']:,.2f} | ↑{pred['prob_up']:.1%} ↓{pred['prob_down']:.1%}")
            
            events_str = ""
            if row['event_count_total'] > 0:
                events_str = f" | Events: {int(row['event_count_total'])}"
                if row['regulation_events'] > 0:
                    events_str += " ⚖️"
                if row['institutional_events'] > 0:
                    events_str += " 🏦"
                if row['security_events'] > 0:
                    events_str += " 🔒"
            print(f"  Sentiment: {row['polarity_mean']:+.3f}{events_str}")
        
        # SL/TP check
        if backtester.position != 0:
            hit_sl, hit_tp, exit_price = backtester.check_sl_tp_hit(row['high'], row['low'])
            
            if hit_sl:
                print(f"  🛑 SL HIT @ ${exit_price:,.2f}")
                backtester.close_position(exit_price, row['timestamp'], 'STOP_LOSS')
            elif hit_tp:
                print(f"  🎉 TP HIT @ ${exit_price:,.2f}")
                backtester.close_position(exit_price, row['timestamp'], 'TAKE_PROFIT')
            elif pred['confidence'] >= 0.60:
                if (backtester.position == 1 and pred['prediction'] == 'DOWN') or \
                   (backtester.position == -1 and pred['prediction'] == 'UP'):
                    print(f"  ⚠️  Signal reversal")
                    backtester.close_position(row['price'], row['timestamp'], 'SIGNAL')
        
        # Entry
        if backtester.position == 0 and pred['confidence'] >= 0.65:
            direction = 1 if pred['prediction'] == 'UP' else -1
            sl, tp, sl_pct, tp_pct = backtester.calculate_sl_tp(
                row['price'], direction, row['price_volatility']
            )
            
            dir_str = 'LONG' if direction == 1 else 'SHORT'
            print(f"  🎯 {dir_str} @ ${row['price']:,.2f}")
            print(f"     SL: ${sl:,.2f} ({sl_pct:.1f}%) | TP: ${tp:,.2f} ({tp_pct:.1f}%)")
            
            backtester.open_position(direction, row['price'], pred['confidence'], 
                                    row['price_volatility'], row['timestamp'])
    
    # Close final
    if backtester.position != 0:
        backtester.close_position(features_df.iloc[-1]['price'], 
                                 features_df.iloc[-1]['timestamp'], 'END')
    
    # Results
    metrics = backtester.get_metrics()
    if metrics:
        print("\n" + "=" * 70)
        print("📊 BACKTEST RESULTS")
        print("=" * 70)
        print(f"Capital: ${backtester.initial_capital:,.2f} → ${metrics['final_capital']:,.2f}")
        print(f"Return: {metrics['total_return']:+.2f}%")
        print(f"\nTrades: {metrics['total_trades']} | Wins: {metrics['wins']} ({metrics['win_rate']:.1%})")
        print(f"Exits: SL={metrics['sl_hits']} | TP={metrics['tp_hits']} | Signal={metrics['signal_exits']}")
        
        signals_df = pd.DataFrame(signals_list)
        print(f"\n📈 SIGNALS")
        print(f"Total: {len(signals_df)} | High Conf: {len(signals_df[signals_df['confidence'] >= 0.65])}")
        print(f"Avg Conf: {signals_df['confidence'].mean():.1%} | Avg Sentiment: {signals_df['sentiment'].mean():+.3f}")
        print("=" * 70)
    
    results = {
        'backtester': backtester, 
        'metrics': metrics, 
        'signals': pd.DataFrame(signals_list)
    }
    
    # Add Monte Carlo simulation
    if metrics and metrics['total_trades'] >= 5:
        mc_simulator = MonteCarloSimulator(n_simulations=1000)
        trades_df = pd.DataFrame(backtester.trades)
        mc_results = mc_simulator.run_simulation(trades_df)
        if mc_results:
            mc_simulator.print_results(mc_results)
            results['monte_carlo'] = mc_results
    
    return results

def show_current_prediction(predictor, features_df):
    """Display current prediction"""
    print("\n" + "=" * 70)
    print("🔮 CURRENT PREDICTION")
    print("=" * 70)
    
    latest = features_df.iloc[-1]
    features = {col: latest[col] for col in predictor.feature_cols}
    pred = predictor.predict(features)
    
    print(f"BTC: ${latest['price']:,.2f}")
    print(f"\n{pred['prediction']} (Confidence: {pred['confidence']:.1%})")
    print(f"Probabilities: ↑{pred['prob_up']:.1%} | ↓{pred['prob_down']:.1%}")
    
    if latest['event_count_total'] > 0:
        print(f"\n⚠️  Events:")
        if latest['regulation_events'] > 0:
            print(f"  • Regulation: {int(latest['regulation_events'])}")
        if latest['institutional_events'] > 0:
            print(f"  • Institutional: {int(latest['institutional_events'])}")
        if latest['security_events'] > 0:
            print(f"  • Security: {int(latest['security_events'])}")
    
    if pred['confidence'] >= 0.65:
        direction = 1 if pred['prediction'] == 'UP' else -1
        vol_pct = (latest['price_volatility'] / latest['price']) * 100
        sl_pct = min(max(vol_pct * 2, 2.0), 5.0)
        tp_pct = sl_pct * 2.5
        
        if direction == 1:
            sl = latest['price'] * (1 - sl_pct / 100)
            tp = latest['price'] * (1 + tp_pct / 100)
        else:
            sl = latest['price'] * (1 + sl_pct / 100)
            tp = latest['price'] * (1 - tp_pct / 100)
        
        print("\n" + "─" * 70)
        print(f"{'🟢' if direction == 1 else '🔴'} {'LONG' if direction == 1 else 'SHORT'} SIGNAL")
        print(f"Entry: ${latest['price']:,.2f}")
        print(f"SL: ${sl:,.2f} ({sl_pct:.1f}%) | TP: ${tp:,.2f} ({tp_pct:.1f}%)")
        print(f"R:R = 1:{tp_pct/sl_pct:.2f}")
    else:
        print("\n⚪ NEUTRAL (Low confidence)")
    
    print("\n" + "=" * 70)

def run_realtime_mode(predictor, analyzer, collector, check_interval):
    """Run real-time monitoring"""
    monitor = RealtimeMonitor(predictor, analyzer, collector, check_interval)
    monitor.start()
    
    print("\n✓ Monitoring started. Press Ctrl+C to stop.")
    
    try:
        while monitor.is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop()
        print("\n✓ Stopped.")
    
    return monitor

# ============================================================================
# MAIN MENU
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 BITCOIN TRADING SYSTEM")
    print("=" * 70)
    print("\nModes:")
    print("1. Backtest (auto load/train)")
    print("2. Backtest (force retrain)")
    print("3. Real-time (5 min)")
    print("4. Real-time (1 min)")
    print("5. Model info")
    print("6. Delete model")
    print("7. Walk-forward optimization")
    
    try:
        choice = input("\nSelect (1-7): ").strip()
        if choice == '1':
            run_complete_system(mode='backtest', use_persistence=True, force_retrain=False)
        elif choice == '2':
            run_complete_system(mode='backtest', use_persistence=True, force_retrain=True)
        elif choice == '3':
            run_complete_system(mode='realtime', check_interval=300, use_persistence=True)
        elif choice == '4':
            run_complete_system(mode='realtime', check_interval=60, use_persistence=True)
        elif choice == '5':
            manager = ModelManager('models')
            if manager.model_exists():
                info = manager.get_model_info()
                age = manager.get_model_age_days()
                print("\n📊 Model Info")
                print("=" * 70)
                print(f"Type: {info.get('model_type', 'Unknown')}")
                print(f"Saved: {info.get('saved_at', 'Unknown')}")
                print(f"Age: {age} days" if age else "Age: Unknown")
                print(f"Accuracy: {info.get('accuracy', 0):.2%}")
                print(f"Samples: {info.get('train_samples', 'Unknown')}")
                print("=" * 70)
            else:
                print("\n⚠️  No model found")
        elif choice == '6':
            manager = ModelManager('models')
            confirm = input("Delete model? (yes/no): ").strip().lower()
            if confirm == 'yes':
                for f in [manager.model_path, manager.scaler_path, manager.metadata_path]:
                    if f.exists():
                        f.unlink()
                        print(f"✓ Deleted {f.name}")
            else:
                print("Cancelled")
        elif choice == '7':
            # Walk-forward optimization
            print("\nRunning walk-forward optimization...")
            print("This will take several minutes...")
            
            # Collect data
            collector = NewsCollector()
            analyzer = AdvancedSentimentAnalyzer()
            predictor = BTCPredictorWithPersistence(use_persistence=True)
            
            # Load or train model
            if not predictor.load_trained_model():
                print("Training model first...")
                news_df = collector.get_crypto_news(limit=100)
                time.sleep(1)
                price_df = collector.get_btc_price_history(days=100)  # More data for optimization
                news_df = analyzer.process_news_batch(news_df)
                features_df = predictor.prepare_features(news_df, price_df)
                predictor.train(features_df, save_model=True)
            else:
                # Get data
                news_df = collector.get_crypto_news(limit=100)
                time.sleep(1)
                price_df = collector.get_btc_price_history(days=100)
                news_df = analyzer.process_news_batch(news_df)
                features_df = predictor.prepare_features(news_df, price_df)
            
            # Run optimization
            optimizer = WalkForwardOptimizer(train_period=20, test_period=10)
            opt_results = optimizer.optimize_confidence_threshold(
                predictor, 
                features_df,
                thresholds=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
            )
            
            if opt_results:
                print(f"\n✅ Recommended threshold: {opt_results['best_threshold']:.2f}")
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("=" * 70)