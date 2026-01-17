import os
import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from flask import Flask, request
from flask_cors import CORS
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import logging
import sys

# ==========================================
# ⚡ CONFIGURATION
# ==========================================
MOCK_MODE = True
CONFIDENCE_THRESHOLD = 0.60
SELL_THRESHOLD = 0.40       
SEQUENCE_LENGTH = 30        
TARGET_GAIN = 1.25          
STOP_LOSS = 0.90            
DB_FOLDER = "axiom_db"

# ==========================================
# 🧠 HYBRID MODEL
# ==========================================
class GraalHybrid:
    def __init__(self):
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6, 
            scale_pos_weight=10.0, eval_metric='logloss', use_label_encoder=False
        )
        inp = Input(shape=(SEQUENCE_LENGTH, 4))
        x = Masking(mask_value=0.0)(inp) 
        x = LSTM(64, return_sequences=False)(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        out = Dense(1, activation='sigmoid')(x)
        self.rnn = Model(inp, out)
        self.rnn.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        
    def train(self, Xt, Xs, y):
        print("   ✂️  Splitting Data...")
        # Ensure we have at least 2 samples to split
        if len(y) < 2: return 

        Xt_train, Xt_val, Xs_train, Xs_val, y_train, y_val = train_test_split(Xt, Xs, y, test_size=0.2, shuffle=False)
        
        print("   🚂 Fitting XGBoost...")
        self.xgb_model.fit(Xt_train, y_train)
        
        print("   🌊 Fitting LSTM...")
        self.rnn.fit(Xs_train, y_train, epochs=5, batch_size=64, verbose=0)
        
        print("\n📊 MODEL PERFORMANCE")
        print("=" * 50)
        try:
            p_xgb = self.xgb_model.predict_proba(Xt_val)[:, 1]
            p_rnn = self.rnn.predict(Xs_val, verbose=0).flatten()
            final_probs = (p_xgb * 0.5) + (p_rnn * 0.5)
            preds = (final_probs > CONFIDENCE_THRESHOLD).astype(int)
            
            # SAFE REPORT PRINTING
            unique_labels = np.unique(y_val)
            if len(unique_labels) > 1:
                print(classification_report(y_val, preds, target_names=['IGNORE', 'BUY']))
            else:
                print(f"   ⚠️ Validation set has only one class: {unique_labels}")
                print(f"   Accuracy: {np.mean(preds == y_val):.2f}")
        except Exception as e:
            print(f"   ⚠️ Could not print metrics: {e}")
        print("=" * 50)
        
    def predict(self, Xt, Xs):
        if not isinstance(Xt, np.ndarray): Xt = np.array(Xt)
        if not isinstance(Xs, np.ndarray): Xs = np.array(Xs)
        Xt = Xt.astype(np.float32)
        Xs = Xs.astype(np.float32)
        p_xgb = self.xgb_model.predict_proba(Xt)[:, 1]
        p_rnn = self.rnn.predict(Xs, verbose=0).flatten()
        return (p_xgb * 0.5) + (p_rnn * 0.5)

# ==========================================
# 🛠️ DATA PROCESSING
# ==========================================
def get_target_label(current_price, future_chart):
    for b in future_chart:
        if b['low'] <= current_price * STOP_LOSS: return 0
        if b['high'] >= current_price * TARGET_GAIN: return 1
    return 0

def process_data_payload(json_data, is_live=False):
    # 1. TABULAR
    raw = json_data.get('raw_stats') or {}
    t20 = json_data.get('t20_stats') or {}
    static_vector = [
        float(raw.get('dev_holds_percent') or 0),
        float(t20.get('top10HoldersPercent') or 0),
        float(t20.get('bundlersHoldPercent') or 0),
        float(t20.get('snipersHoldPercent') or 0),
        float(raw.get('lp_burned') or 0),
        float(raw.get('initial_liquidity_sol') or 0)
    ]

    # 2. SEQUENCE
    history = json_data.get('history') or []
    chart = json_data.get('consolidated_chart') or []

    if not chart: 
        if is_live: return "WAITING_DATA"
        else: return None, None, None

    try:
        # UNWRAP DICT
        if isinstance(chart, dict):
            if 'data' in chart: chart = chart['data']
            elif 'bars' in chart: chart = chart['bars']
            elif 'candles' in chart: chart = chart['candles']
            else: chart = list(chart.values())

        df_chart = pd.DataFrame(chart)
        rename_map = {'t': 'time', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}
        df_chart.rename(columns=rename_map, inplace=True)

        if 'time' not in df_chart.columns: 
            return None if is_live else (None, None, None)
        
        df_chart['time'] = df_chart['time'].astype(np.int64)
        df_chart = df_chart.sort_values('time')
        last_price = float(df_chart.iloc[-1]['close'])

        # HISTORY
        valid_history = []
        for h in history:
            if 'stats' in h and h['stats']:
                valid_history.append({
                    'time': int(h['timestamp']), 
                    'holders': h['stats'].get('numHolders', 0),
                    'fees': h['stats'].get('totalPairFeesPaid', 0)
                })
        
        if valid_history:
            df_hist = pd.DataFrame(valid_history).sort_values('time')
            df_hist['time'] = df_hist['time'].astype(np.int64)
            df_merged = pd.merge_asof(df_chart, df_hist, on='time', direction='backward').fillna(0)
        else:
            df_merged = df_chart.copy()
            df_merged['holders'] = 0
            df_merged['fees'] = 0

    except: return None if is_live else (None, None, None)

    if df_merged.empty: return None if is_live else (None, None, None)

    df_merged['price_pct'] = df_merged['close'].pct_change().fillna(0)
    df_merged['vol_log'] = np.log1p(df_merged['volume'])
    df_merged['holder_delta'] = df_merged['holders'].diff().fillna(0)
    df_merged['fee_delta'] = df_merged['fees'].diff().fillna(0)
    
    raw_seq = df_merged[['price_pct', 'vol_log', 'holder_delta', 'fee_delta']].values.astype('float32')

    if is_live:
        padded_seq = pad_sequences([raw_seq], maxlen=SEQUENCE_LENGTH, dtype='float32', padding='pre', truncating='pre')
        return np.array([static_vector], dtype=np.float32), padded_seq, last_price
    else:
        # TRAINING EXTRACTION
        X_tab, X_seq, y = [], [], []
        if len(df_merged) < 5: return None, None, None
        
        for i in range(5, len(df_merged), 2): 
            window_raw = raw_seq[max(0, i-SEQUENCE_LENGTH) : i]
            padded_window = pad_sequences([window_raw], maxlen=SEQUENCE_LENGTH, dtype='float32', padding='pre', truncating='pre')[0]
            
            curr_price = df_merged.iloc[i-1]['close']
            curr_time = df_merged.iloc[i-1]['time']
            future = [c for c in chart if c['time'] > curr_time]
            
            if not future: continue
            label = get_target_label(curr_price, future)
            
            X_tab.append(static_vector)
            X_seq.append(padded_window)
            y.append(label)
            
        if not X_seq: return None, None, None
        return X_tab, X_seq, y

def train_on_boot():
    print("🚜 Loading Database...")
    files = [f for f in os.listdir(DB_FOLDER) if f.endswith('.json')]
    Xt, Xs, y = [], [], []
    
    count = 0
    for i, f in enumerate(files):
        try:
            with open(os.path.join(DB_FOLDER, f), 'r') as file:
                # We call is_live=False to get the training lists
                res = process_data_payload(json.load(file), is_live=False)
                if res and res[0] is not None:
                    res1, res2, res3 = res
                    if len(res1) > 0: 
                        Xt.extend(res1)
                        Xs.extend(res2)
                        y.extend(res3)
                        count += 1
        except: pass
        if i % 100 == 0: print(f"   Scanned {i} files...", end='\r')
    
    # 🛡️ DUMMY DATA GENERATOR (CRASH PREVENTION)
    if len(y) == 0:
        print("\n⚠️ No Training Data Loaded. Using DUMMY BRAIN to prevent crash.")
        # Create 10 dummy samples, alternating classes 0 and 1
        Xt = np.zeros((10, 6))
        Xs = np.zeros((10, SEQUENCE_LENGTH, 4))
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    else:
        print(f"\n✅ Training on {len(y)} samples from {count} files.")

    Xt = np.array(Xt, dtype=np.float32)
    Xs = np.array(Xs, dtype=np.float32)
    y = np.array(y, dtype=int)
    
    bot = GraalHybrid()
    bot.train(Xt, Xs, y)
    return bot

# ==========================================
# 🌐 LIVE SERVER
# ==========================================
app = Flask(__name__)
CORS(app)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

bot_brain = train_on_boot()
active_trades = {}

@app.route('/new_token', methods=['POST'])
def webhook():
    try:
        data = request.json
        contract = data.get('contract')
        ticker = data.get('raw_stats', {}).get('token_ticker', '???')

        res = process_data_payload(data, is_live=True)
        
        if isinstance(res, str) and res == "WAITING_DATA":
            print(f"\r⏳ {ticker} Waiting...   ", end="", flush=True)
            return res, 200
            
        if res is None:
            return "ERROR", 200

        if isinstance(res, tuple):
            Xt, Xs, current_price = res
            
            prob = bot_brain.predict(Xt, Xs)[0]
            
            print(f"\n📥 {ticker} | Score: {prob*100:.1f}% | Price: {current_price:.8f}")

            if contract in active_trades:
                trade = active_trades[contract]
                roi = current_price / trade['entry']
                if roi >= TARGET_GAIN:
                    print(f"   💰 TP: {contract} (+{(roi-1)*100:.1f}%)")
                    del active_trades[contract]
                elif roi <= STOP_LOSS:
                    print(f"   🛑 SL: {contract} ({(roi-1)*100:.1f}%)")
                    del active_trades[contract]
                elif prob < SELL_THRESHOLD:
                    print(f"   📉 DUMP: {contract} (Conf: {prob*100:.1f}%)")
                    del active_trades[contract]
            else:
                if prob >= CONFIDENCE_THRESHOLD:
                    print(f"   🚀 SIGNAL: {ticker} | Conf: {prob*100:.1f}%")
                    if MOCK_MODE:
                        active_trades[contract] = {'entry': current_price, 'time': datetime.now()}

        return "OK", 200

    except Exception as e:
        print(f"\n⚠️ SERVER ERROR: {e}")
        return "ERROR", 500

if __name__ == "__main__":
    print(f"\n🤖 GRAAL LIVE V7 (STABLE) RUNNING...")
    app.run(host='0.0.0.0', port=5001)