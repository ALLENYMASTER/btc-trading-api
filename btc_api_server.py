"""
Bitcoin Trading System - FastAPI Backend for iOS App
====================================================
RESTful API server that iOS app can connect to
Lightweight version optimized for mobile consumption
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
import asyncio
from datetime import datetime
import json
from pathlib import Path
from aioapns import APNs, NotificationRequest

# Import your existing trading system components
import sys
sys.path.append('.')
from btc_trading_system import (
    NewsCollector, 
    AdvancedSentimentAnalyzer,
    BTCPredictorWithPersistence,
    ModelManager,
    BacktesterWithRisk
)

app = FastAPI(title="BTC Trading API", version="1.0")

# Enable CORS for iOS app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your iOS app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
collector = NewsCollector()
analyzer = AdvancedSentimentAnalyzer()
predictor = BTCPredictorWithPersistence(use_persistence=True)
model_manager = ModelManager('models')

model_ready = False
model_training = False

# Cache
price_cache = {"timestamp": None, "data": None}
signal_cache = {"timestamp": None, "data": None}
CACHE_DURATION = 60  # seconds

# ============================================================================
# Response Models
# ============================================================================

class PriceResponse(BaseModel):
    price: float
    change_24h: float
    timestamp: str
    
class SignalResponse(BaseModel):
    prediction: str
    confidence: float
    prob_up: float
    prob_down: float
    current_price: float
    sentiment: float
    events: int
    timestamp: str
    recommendation: str
    
class NewsItem(BaseModel):
    title: str
    sentiment: float
    published_at: str
    source: str
    events: List[str]
    
class ModelStatus(BaseModel):
    exists: bool
    age_days: Optional[int]
    accuracy: Optional[float]
    last_updated: Optional[str]
    
class BacktestRequest(BaseModel):
    days: int = 30
    initial_capital: float = 10000
    
class BacktestResponse(BaseModel):
    total_return: float
    win_rate: float
    total_trades: int
    final_capital: float
    sl_hits: int
    tp_hits: int

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "online",
        "service": "BTC Trading System API",
        "version": "1.0",
        "model_ready": model_ready,
        "model_training": model_training,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """詳細健康檢查"""
    return {
        "status": "healthy",
        "model_ready": model_ready,
        "model_training": model_training,
        "predictor_trained": predictor.is_trained
    }

@app.get("/price")
async def get_current_price():
    try:
        data = collector.get_current_price()
        if not data:
            raise HTTPException(status_code=503, detail="Price service unavailable")
        
        return {
            "price": data['price'],
            "change_24h": data['change_24h'],
            "timestamp": data['timestamp'].isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/signal")
async def get_trading_signal():
    global model_ready, model_training
    
    if not model_ready and not predictor.is_trained:
        if not model_training:
            return {
                "status": "model_not_ready",
                "message": "Model is initializing. Please wait 2-3 minutes and try again.",
                "prediction": "WAIT",
                "confidence": 0.0,
                "prob_up": 0.5,
                "prob_down": 0.5,
                "current_price": 0.0,
                "sentiment": 0.0,
                "events": 0,
                "timestamp": datetime.now().isoformat(),
                "recommendation": "WAIT - Model initializing"
            }
        else:
            return {
                "status": "model_training",
                "message": "Model is currently training. Please wait.",
                "prediction": "WAIT",
                "confidence": 0.0,
                "prob_up": 0.5,
                "prob_down": 0.5,
                "current_price": 0.0,
                "sentiment": 0.0,
                "events": 0,
                "timestamp": datetime.now().isoformat(),
                "recommendation": "WAIT - Training in progress"
            }
    
    try:
        price_data = collector.get_current_price()
        if not price_data:
            raise HTTPException(status_code=503, detail="Price unavailable")
        
        news_df = collector.get_crypto_news(limit=20)
        if news_df.empty:
            return {
                "prediction": "NEUTRAL",
                "confidence": 0.5,
                "prob_up": 0.5,
                "prob_down": 0.5,
                "current_price": price_data['price'],
                "sentiment": 0.0,
                "events": 0,
                "timestamp": datetime.now().isoformat(),
                "recommendation": "WAIT - No news data"
            }
        
        news_df = analyzer.process_news_batch(news_df)
        
        features = {
            'polarity_mean': news_df['polarity'].mean(),
            'polarity_std': news_df['polarity'].std(),
            'polarity_min': news_df['polarity'].min(),
            'polarity_max': news_df['polarity'].max(),
            'subjectivity_mean': news_df['subjectivity'].mean(),
            'bullish_total': news_df['bullish_count'].sum(),
            'bearish_total': news_df['bearish_count'].sum(),
            'volatile_total': news_df['volatile_count'].sum(),
            'news_count': len(news_df),
            'price_ma7': price_data['price'],
            'price_volatility': 0,
            'event_count_total': news_df['event_count'].sum(),
            'regulation_events': news_df['has_regulation'].sum(),
            'institutional_events': news_df['has_institutional'].sum(),
            'security_events': news_df['has_security'].sum(),
            'price_action_mean': news_df['price_action_score'].mean()
        }
        
        prediction = predictor.predict(features)
        if not prediction:
            raise HTTPException(status_code=500, detail="Prediction failed")
        
        conf = prediction['confidence']
        if conf >= 0.70:
            rec = f"STRONG {prediction['prediction']}"
        elif conf >= 0.60:
            rec = f"MODERATE {prediction['prediction']}"
        else:
            rec = "WAIT - Low confidence"
        
        return {
            "prediction": prediction['prediction'],
            "confidence": prediction['confidence'],
            "prob_up": prediction['prob_up'],
            "prob_down": prediction['prob_down'],
            "current_price": price_data['price'],
            "sentiment": features['polarity_mean'],
            "events": int(features['event_count_total']),
            "timestamp": datetime.now().isoformat(),
            "recommendation": rec
        }
        
    except Exception as e:
        print(f"Signal error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/news")
async def get_latest_news(limit: int = 10):
    try:
        news_df = collector.get_crypto_news(limit=limit)
        if news_df.empty:
            return []
        
        news_df = analyzer.process_news_batch(news_df)
        
        items = []
        for _, row in news_df.iterrows():
            events = []
            if row['has_regulation']:
                events.append("Regulation")
            if row['has_institutional']:
                events.append("Institutional")
            if row['has_security']:
                events.append("Security")
            
            items.append({
                "title": row['title'],
                "sentiment": float(row['final_sentiment']),
                "published_at": row['published_at'].isoformat(),
                "source": row['source'],
                "events": events
            })
        
        return items
        
    except Exception as e:
        print(f"News error: {e}")
        return []

@app.get("/model/status", response_model=ModelStatus)
async def get_model_status():
    """Get model information"""
    try:
        exists = model_manager.model_exists()
        
        if not exists:
            return ModelStatus(
                exists=False,
                age_days=None,
                accuracy=None,
                last_updated=None
            )
        
        info = model_manager.get_model_info()
        age = model_manager.get_model_age_days()
        
        return ModelStatus(
            exists=True,
            age_days=age,
            accuracy=info.get('accuracy'),
            last_updated=info.get('saved_at')
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/train")
async def train_model(background_tasks: BackgroundTasks):
    """Trigger model training (runs in background)"""
    background_tasks.add_task(train_model_background)
    return {"status": "training_started", "message": "Model training in progress"}

async def train_model_background():
    """Background task to train model"""
    try:
        print("Starting model training...")
        
        # Collect data
        news_df = collector.get_crypto_news(limit=100)
        await asyncio.sleep(1)  # Rate limiting
        price_df = collector.get_btc_price_history(days=60)
        
        # Process
        news_df = analyzer.process_news_batch(news_df)
        features_df = predictor.prepare_features(news_df, price_df)
        
        # Train
        predictor.train(features_df, save_model=True)
        
        print("Model training completed")
        
    except Exception as e:
        print(f"Training error: {e}")

@app.post("/backtest", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    """Run backtest simulation"""
    try:
        # Ensure model is trained
        if not predictor.is_trained:
            if not predictor.load_trained_model():
                raise HTTPException(status_code=400, detail="Model not trained")
        
        # Get data
        news_df = collector.get_crypto_news(limit=100)
        await asyncio.sleep(1)
        price_df = collector.get_btc_price_history(days=request.days)
        
        # Process
        news_df = analyzer.process_news_batch(news_df)
        features_df = predictor.prepare_features(news_df, price_df)
        
        # Run backtest
        backtester = BacktesterWithRisk(initial_capital=request.initial_capital)
        
        for i in range(10, len(features_df) - 1):
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
            
            # Entry
            if backtester.position == 0 and pred['confidence'] >= 0.65:
                direction = 1 if pred['prediction'] == 'UP' else -1
                backtester.open_position(
                    direction, row['price'], pred['confidence'],
                    row['price_volatility'], row['timestamp']
                )
        
        # Final position
        if backtester.position != 0:
            backtester.close_position(
                features_df.iloc[-1]['price'],
                features_df.iloc[-1]['timestamp'],
                'END'
            )
        
        # Get metrics
        metrics = backtester.get_metrics()
        if not metrics:
            raise HTTPException(status_code=400, detail="No trades executed")
        
        return BacktestResponse(
            total_return=metrics['total_return'],
            win_rate=metrics['win_rate'],
            total_trades=metrics['total_trades'],
            final_capital=metrics['final_capital'],
            sl_hits=metrics['sl_hits'],
            tp_hits=metrics['tp_hits']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def send_silent_push(device_token: str, signal_data: dict):
    apns = APNs(
        key='path/to/key.p8',  # Apple Push Key
        key_id='YOUR_KEY_ID',
        team_id='YOUR_TEAM_ID',
        topic='com.yourapp.btctrading',
        use_sandbox=False
    )
    
    request = NotificationRequest(
        device_token=device_token,
        message={
            "aps": {
                "content-available": 1,  # silent push
                "sound": "",
                "badge": 0
            },
            "signal": signal_data
        }
    )
    
    await apns.send_notification(request)
    
# ============================================================================
# Server Startup
# ============================================================================
async def train_model_on_startup():
    global model_ready, model_training
    
    try:
        print("=" * 70)
        print("🚀 Starting model initialization...")
        print("=" * 70)
        
        model_training = True
        
        if predictor.load_trained_model():
            print("✓ Model loaded from disk")
            model_ready = True
            model_training = False
            return
        
        print("📊 Training new model...")
        
        news_df = collector.get_crypto_news(limit=100)
        await asyncio.sleep(1)
        price_df = collector.get_btc_price_history(days=60)
        
        news_df = analyzer.process_news_batch(news_df)
        features_df = predictor.prepare_features(news_df, price_df)
        
        predictor.train(features_df, save_model=True)
        
        model_ready = True
        model_training = False
        
        print("✓ Model training completed")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        model_training = False
        model_ready = False

@app.on_event("startup")
async def startup_event():
    print("=" * 70)
    print("🚀 BTC Trading API Server Starting...")
    print("=" * 70)
    
    asyncio.create_task(train_model_on_startup())
    
    print("✓ Server ready (model initializing in background)")
    print("=" * 70)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "btc_api_server:app",
        host="0.0.0.0",
        port=port,
        reload=False  
    )
