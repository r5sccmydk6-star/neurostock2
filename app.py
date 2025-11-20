"""
===============================
  REQUIREMENTS (FULL & CLEAN)
===============================
Flask==3.0.0
Flask-Login==0.6.3
Flask-Caching==2.1.0
Werkzeug==3.0.1

gunicorn==21.2.0

numpy==1.26.4
pandas==2.2.0
pandas-ta==0.3.14b0
yfinance==0.2.40
feedparser==6.0.10
textblob==0.17.1

scikit-learn==1.3.2

tensorflow==2.15.0
joblib==1.3.2

plotly==5.20.0
matplotlib==3.8.2

SQLAlchemy==2.0.25
"""

import os
BASE_DIR = os.path.dirname(__file__)  # Relative paths for all files

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import joblib
import feedparser
import sqlite3
import traceback
from datetime import date
from textblob import TextBlob
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_caching import Cache
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# ------------------------
# Flask setup
# ------------------------
app = Flask(__name__)
app.secret_key = 'neurostock_secret_key_secure'
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 300})

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# ------------------------
# Paths
# ------------------------
DB_FILE = os.path.join(BASE_DIR, "stock_data.db")
MODEL_DIR = os.path.join(BASE_DIR, "models_v15_gold")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# ------------------------
# App constants
# ------------------------
FEATURES_LIST = ['Close', 'Volume', 'RSI', 'MACD', 'EMA', 'ATR', 'BB_UPPER', 'BB_LOWER', 'VWAP', 'Pct_Change',
                 'SMA_7', 'SMA_30', 'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3']
TARGET_COL = 'Pct_Change'
TICKERS_DATA = [
    {"symbol": "AAPL", "name": "Apple Inc."}, {"symbol": "MSFT", "name": "Microsoft"},
    {"symbol": "GOOG", "name": "Google"}, {"symbol": "AMZN", "name": "Amazon"},
    {"symbol": "TSLA", "name": "Tesla"}, {"symbol": "NVDA", "name": "NVIDIA"},
    {"symbol": "BTC-USD", "name": "Bitcoin"}, {"symbol": "ETH-USD", "name": "Ethereum"},
    {"symbol": "SPY", "name": "S&P 500"}, {"symbol": "QQQ", "name": "Nasdaq 100"}
]

# ------------------------
# User & DB
# ------------------------
class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    with sqlite3.connect(DB_FILE) as conn:
        u = conn.cursor().execute("SELECT id, username FROM users WHERE id = ?", (user_id,)).fetchone()
        if u: return User(id=u[0], username=u[1])
    return None

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT UNIQUE,
                      password TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS portfolio
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_id INTEGER,
                      ticker TEXT,
                      shares REAL,
                      avg_price REAL)''')
        conn.commit()

init_db()

# ------------------------
# Data functions
# ------------------------
def flatten_yfinance_data(df):
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    date_col = next((c for c in df.columns if str(c).lower() in ['date', 'datetime', 'timestamp']), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
    else:
        df.index = pd.date_range(end=pd.Timestamp.today(), periods=len(df), freq='B')
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def add_features(df):
    df = df.copy()
    if 'Close' not in df.columns: return df
    df.loc[df['High'] == df['Low'], 'High'] += 1e-6
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.ema(length=20, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    v = df['Volume'].values
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = df.assign(vw=(v * tp)).groupby(df.index.date)['vw'].cumsum() / df.assign(v=v).groupby(df.index.date)['v'].cumsum()
    rename_map = {}
    for c in df.columns:
        if c == 'RSI_14': rename_map[c] = 'RSI'
        if c.startswith('MACD_') and 'h' not in c and 's' not in c: rename_map[c] = 'MACD'
        if c.startswith('BBU_'): rename_map[c] = 'BB_UPPER'
        if c.startswith('BBL_'): rename_map[c] = 'BB_LOWER'
        if c == 'EMA_20': rename_map[c] = 'EMA'
        if c == 'ATRr_14': rename_map[c] = 'ATR'
    df.rename(columns=rename_map, inplace=True)
    df['Pct_Change'] = df['Close'].pct_change()
    df['SMA_7'] = df['Close'].rolling(7).mean()
    df['SMA_30'] = df['Close'].rolling(30).mean()
    for i in range(1, 4): df[f'Close_Lag_{i}'] = df['Close'].shift(i)
    for f in FEATURES_LIST:
        if f not in df.columns: df[f] = 0
    return df.dropna()

@cache.memoize(timeout=300)
def get_data(ticker):
    try:
        raw = yf.download(ticker, period='2y', auto_adjust=True, progress=False)
        if raw.empty: raise ValueError("Empty Data")
        return add_features(flatten_yfinance_data(raw))
    except:
        return pd.DataFrame()

def get_model(ticker, feature_data, seq_len, horizon):
    safe = ''.join(e for e in ticker if e.isalnum())
    s_path = os.path.join(MODEL_DIR, f"{safe}_s.joblib")
    m_path = os.path.join(MODEL_DIR, f"{safe}_m.keras")
    if os.path.exists(s_path) and os.path.exists(m_path):
        try:
            scaler = joblib.load(s_path)
            if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != len(FEATURES_LIST): raise ValueError
            m = load_model(m_path, compile=False)
            return scaler, m
        except:
            if os.path.exists(s_path): os.remove(s_path)
            if os.path.exists(m_path): os.remove(m_path)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_data)
    X, y = [], []
    t_idx = FEATURES_LIST.index(TARGET_COL)
    for i in range(seq_len, len(scaled) - horizon + 1):
        X.append(scaled[i - seq_len:i])
        y.append(scaled[i:i + horizon, t_idx])

    model = Sequential([
        Input(shape=(seq_len, len(FEATURES_LIST))),
        LSTM(64, return_sequences=True),
        Dropout(0.1),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(horizon)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(np.array(X), np.array(y), epochs=20, batch_size=16, verbose=0)

    joblib.dump(scaler, s_path)
    model.save(m_path)
    return scaler, model

def get_exchange_rate():
    try:
        return yf.Ticker("USDINR=X").fast_info.last_price or 84.0
    except:
        return 84.0

# ------------------------
# Flask routes remain the same as your previous code
# Make sure any DB or model paths use BASE_DIR as above
# ------------------------

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
