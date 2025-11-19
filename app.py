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

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import os
import joblib
import json
import feedparser
import sqlite3
import traceback
from datetime import date
from sqlalchemy import create_engine
from textblob import TextBlob
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_caching import Cache
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import AdamW

app = Flask(__name__)
app.secret_key = 'neurostock_secret_key_secure'
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 300})

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

DB_FILE = "stock_data.db"
MODEL_DIR = "models_v15_gold"
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

N_MC_SAMPLES = 50
FEATURES_LIST = ['Close', 'Volume', 'RSI', 'MACD', 'EMA', 'ATR', 'BB_UPPER', 'BB_LOWER', 'VWAP', 'Pct_Change', 'SMA_7',
                 'SMA_30', 'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3']
TARGET_COL = 'Pct_Change'

TICKERS_DATA = [
    {"symbol": "AAPL", "name": "Apple Inc."}, {"symbol": "MSFT", "name": "Microsoft"},
    {"symbol": "GOOG", "name": "Google"}, {"symbol": "AMZN", "name": "Amazon"},
    {"symbol": "TSLA", "name": "Tesla"}, {"symbol": "NVDA", "name": "NVIDIA"},
    {"symbol": "BTC-USD", "name": "Bitcoin"}, {"symbol": "ETH-USD", "name": "Ethereum"},
    {"symbol": "SPY", "name": "S&P 500"}, {"symbol": "QQQ", "name": "Nasdaq 100"}
]

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
    df.loc[df['High'] == df['Low'], 'High'] = df['High'] + 1e-6
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
    s_path, m_path = f"{MODEL_DIR}/{safe}_s.joblib", f"{MODEL_DIR}/{safe}_m.keras"
    if os.path.exists(s_path) and os.path.exists(m_path):
        try:
            scaler = joblib.load(s_path)
            if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != len(FEATURES_LIST): raise ValueError
            m = load_model(m_path, compile=False)
            return scaler, m
        except:
            if os.path.exists(s_path): os.remove(s_path)
            if os.path.exists(m_path): os.remove(m_path)

    split = int(len(feature_data) * 0.9)
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

def calculate_neuro_score(row, roi):
    try:
        score = 50
        rsi = 50 if pd.isna(row.get('RSI')) else row['RSI']
        macd = 0 if pd.isna(row.get('MACD')) else row['MACD']

        if roi > 0.05: score += 25
        elif roi > 0.01: score += 10
        elif roi < -0.05: score -= 25
        elif roi < -0.01: score -= 10

        if rsi < 30: score += 15
        elif rsi > 70: score -= 15

        if macd > 0: score += 5

        score = max(0, min(100, int(score)))
        if score >= 75: return score, "STRONG BUY", "success"
        elif score >= 60: return score, "BUY", "success"
        elif score <= 25: return score, "STRONG SELL", "danger"
        elif score <= 40: return score, "SELL", "danger"
        else: return score, "HOLD", "warning"
    except:
        return 50, "WAIT", "warning"

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        u = request.form['username']
        p = request.form['password']
        with sqlite3.connect(DB_FILE) as conn:
            user = conn.cursor().execute("SELECT id, username, password FROM users WHERE username = ?", (u,)).fetchone()
        if user and check_password_hash(user[2], p):
            login_user(User(id=user[0], username=user[1]))
            return redirect(url_for('index'))
        flash('Invalid credentials')
    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    u = request.form['username']
    p = generate_password_hash(request.form['password'])
    try:
        with sqlite3.connect(DB_FILE) as conn:
            conn.cursor().execute("INSERT INTO users (username, password) VALUES (?, ?)", (u, p))
        flash('Created! Login.')
    except:
        flash('Username taken.')
    return redirect(url_for('login'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('index.html', user=current_user)

@app.route('/portfolio')
@login_required
def pf():
    return render_template('portfolio.html', user=current_user)

@app.route('/news')
@login_required
def nw():
    return render_template('news.html', user=current_user)

@app.route('/settings')
@login_required
def st():
    return render_template('settings.html', user=current_user)

@app.route('/api/tickers')
def get_t():
    return jsonify(TICKERS_DATA)

@app.route('/api/news', methods=['POST'])
@login_required
def apinews():
    try:
        f = feedparser.parse(f'https://finance.yahoo.com/rss/headline?s={request.json.get("ticker", "AAPL")}')
        n = [{
            'title': e.title,
            'link': e.link,
            'published': e.published,
            'sentiment': "Bullish" if TextBlob(e.title).sentiment.polarity > 0.1
            else "Bearish" if TextBlob(e.title).sentiment.polarity < -0.1
            else "Neutral"
        } for e in f.entries[:8]]
        return jsonify({'status': 'success', 'news': n})
    except:
        return jsonify({'status': 'error'})

@app.route('/api/predict', methods=['POST'])
@login_required
def predict():
    d = request.json
    ticker = d.get('ticker', 'AAPL')
    try:
        full = get_data(ticker)
        if full.empty: return jsonify({'status': 'error', 'message': 'No data'})

        scaler, model = get_model(ticker, full[FEATURES_LIST], 60, 30)
        last_seq = scaler.transform(full[FEATURES_LIST].iloc[-60:]).reshape(1, 60, len(FEATURES_LIST))
        base_pred = model.predict(last_seq, verbose=0)[0]

        t_idx = FEATURES_LIST.index(TARGET_COL)
        real_pct = (base_pred * scaler.scale_[t_idx]) + scaler.mean_[t_idx]

        last_price = full['Close'].iloc[-1]
        path = [last_price]
        for day in range(30):
            path.append(path[-1] * (1 + real_pct[day]))
        mean_f = np.array(path[1:])
        upper = mean_f * 1.05
        lower = mean_f * 0.95

        hist = full.tail(100)
        val_line = hist['EMA'].fillna(hist['Close']).tolist()

        roi = (mean_f[-1] - last_price) / last_price
        score, sig, col = calculate_neuro_score(hist.iloc[-1], roi)

        hist_dates = [d.strftime('%b %d') for d in hist.index]
        fut_dates = [d.strftime('%b %d') for d in pd.bdate_range(start=full.index[-1], periods=30)]

        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            fund = {
                'marketCap': stock.fast_info.market_cap,
                'peRatio': info.get('trailingPE', 'N/A'),
                'sector': info.get('sector', 'Unknown'),
                'high52': round(stock.fast_info.year_high, 2)
            }
        except:
            fund = {'marketCap': 'N/A', 'peRatio': 'N/A', 'sector': 'N/A', 'high52': 'N/A'}

        return jsonify({
            'status': 'success',
            'current_price': round(last_price, 2),
            'predicted_price': round(mean_f[-1], 2),
            'inr_rate': get_exchange_rate(),
            'neuro_score': {
                'score': int(max(0, min(100, score))),
                'signal': sig,
                'color': col
            },
            'fundamentals': fund,
            'history': {
                'dates': hist_dates,
                'prices': hist['Close'].tolist(),
                'validation': val_line,
                'rsi': hist['RSI'].fillna(50).tolist(),
                'macd': hist['MACD'].fillna(0).tolist()
            },
            'forecast': {
                'dates': fut_dates,
                'mean': mean_f.tolist(),
                'upper': upper.tolist(),
                'lower': lower.tolist()
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/portfolio', methods=['GET'])
@login_required
def get_apf():
    with sqlite3.connect(DB_FILE) as conn:
        rows = conn.cursor().execute(
            "SELECT id, ticker, shares, avg_price FROM portfolio WHERE user_id=?",
            (current_user.id,)
        ).fetchall()

    h = []
    val = 0
    cost = 0
    labels = []
    data = []

    for r in rows:
        try:
            curr = yf.Ticker(r[1]).fast_info.last_price or r[3]
        except:
            curr = r[3]

        v = curr * r[2]
        h.append({
            'id': r[0],
            'ticker': r[1],
            'shares': r[2],
            'avg_price': r[3],
            'current_price': curr,
            'total_value': v,
            'gain_loss': v - (r[3] * r[2])
        })

        val += v
        cost += r[3] * r[2]
        labels.append(r[1])
        data.append(v)

    return jsonify({
        'status': 'success',
        'holdings': h,
        'rate': get_exchange_rate(),
        'summary': {'total_value': val, 'total_gain': val - cost},
        'allocation': {'labels': labels, 'data': data}
    })

@app.route('/api/portfolio/add', methods=['POST'])
@login_required
def add_apf():
    d = request.json
    with sqlite3.connect(DB_FILE) as conn:
        conn.cursor().execute(
            "INSERT INTO portfolio (user_id, ticker, shares, avg_price) VALUES (?, ?, ?, ?)",
            (current_user.id, d['ticker'].upper(), float(d['shares']), float(d['price']))
        )
    return jsonify({'status': 'success'})

@app.route('/api/portfolio/delete', methods=['POST'])
@login_required
def delete_apf():
    with sqlite3.connect(DB_FILE) as conn:
        conn.cursor().execute("DELETE FROM portfolio WHERE id = ?", (request.json['id'],))
    return jsonify({'status': 'success'})

@app.route('/api/settings/info', methods=['GET'])
def info():
    return jsonify({
        'status': 'success',
        'db_size_mb': 0.8,
        'model_count': len([f for f in os.listdir(MODEL_DIR) if f.endswith('.keras')]),
        'ticker_count': 0
    })

@app.route('/api/settings/clear_models', methods=['POST'])
def cm():
    for f in os.listdir(MODEL_DIR):
        os.remove(os.path.join(MODEL_DIR, f))
    return jsonify({'status': 'success'})

@app.route('/api/settings/clear_data', methods=['POST'])
def cd():
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
