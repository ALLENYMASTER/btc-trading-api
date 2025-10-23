# Bitcoin Trading System API

FastAPI backend for BTC trading signal system.

## Features
- Real-time Bitcoin price tracking
- ML-powered trading signals
- Advanced NLP sentiment analysis
- Stop-loss/Take-profit risk management
- Backtesting capabilities

## Deployment
Deployed on Railway: [Your URL]

## API Documentation
Visit `/docs` for interactive API documentation.

## Endpoints
- `GET /price` - Current BTC price
- `GET /signal` - Trading signal
- `GET /news` - Latest analyzed news
- `POST /backtest` - Run backtest simulation
- `POST /model/train` - Retrain model

## Tech Stack
- FastAPI
- scikit-learn
- pandas
- TextBlob
