import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
DB_FOLDER = "axiom_db"
SEQUENCE_LENGTH = 60  # Input: Last 60 seconds of candle/stat data
TARGET_GAIN = 1.25    # Win Condition: +25%
STOP_LOSS = 0.90      # Loss Condition: -10%

# ==========================================
# 1. STRICT PARSER (No Lookahead)
# ==========================================
def get_target_label(current_price, future_chart):
    """
    Returns 1 if price hits +25% before -10%, else 0.
    Strictly uses FUTURE data only.
    """
    if not future_chart: return 0
    
    for b in future_chart:
        # Check Stop First (Conservative)
        if b['low'] <= current_price * STOP_LOSS:
            return 0
        # Check Win
        if b['high'] >= current_price * TARGET_GAIN:
            return 1
    return 0

def process_token_file(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except: return None, None, None

    # --- A. STATIC FEATURES (Tabular) ---
    # These are known at T=0
    raw = data.get('raw_stats', {})
    t20 = data.get('t20_stats', {})
    
    if 'dev_holds_percent' not in raw: return None, None, None

    # Features: [Dev%, Top10%, Bundler%, Sniper%, LP Burned, Liq SOL]
    static_vector = [
        raw.get('dev_holds_percent', 0),
        t20.get('top10HoldersPercent', 0),
        t20.get('bundlersHoldPercent', 0),
        t20.get('snipersHoldPercent', 0),
        raw.get('lp_burned', 0),
        raw.get('initial_liquidity_sol', 0)
    ]

    # --- B. DYNAMIC FEATURES (Sequence) ---
    history = data.get('history', [])
    chart = data.get('consolidated_chart', [])
    
    if not history or not chart: return None, None, None

    # Convert to DataFrames
    df_chart = pd.DataFrame(chart).sort_values('time')
    df_hist = pd.DataFrame([
        {
            'time': h['timestamp'], 
            'holders': h['stats']['numHolders'],
            'fees': h['stats']['totalPairFeesPaid']
        } 
        for h in history
    ]).sort_values('time')

    # Merge: Strictly backward direction (No future data leakage)
    df_merged = pd.merge_asof(
        df_chart, 
        df_hist, 
        on='time', 
        direction='backward'
    ).fillna(0)

    # Feature Engineering
    df_merged['price_pct'] = df_merged['close'].pct_change().fillna(0)
    df_merged['vol_log'] = np.log1p(df_merged['volume'])
    df_merged['holder_delta'] = df_merged['holders'].diff().fillna(0)
    df_merged['fee_delta'] = df_merged['fees'].diff().fillna(0)

    X_tab_list = []
    X_seq_list = []
    y_list = []

    # Step size 10 to reduce redundancy
    for i in range(SEQUENCE_LENGTH, len(df_merged), 10):
        # 1. INPUT WINDOW (Past)
        window = df_merged.iloc[i-SEQUENCE_LENGTH : i]
        
        # Normalize Price relative to window start
        base_price = window.iloc[0]['close']
        window_norm = window.copy()
        window_norm['price_norm'] = (window['close'] / base_price) - 1.0
        
        # Select Features for LSTM: [Norm Price, Log Vol, Holder Delta, Fee Delta]
        seq_features = window_norm[['price_norm', 'vol_log', 'holder_delta', 'fee_delta']].values
        
        # 2. TARGET (Future)
        current_price = window.iloc[-1]['close']
        current_time = window.iloc[-1]['time']
        
        # Strict lookahead in original chart data
        future_chart = [c for c in chart if c['time'] > current_time]
        label = get_target_label(current_price, future_chart)

        X_tab_list.append(static_vector)
        X_seq_list.append(seq_features)
        y_list.append(label)

    if not X_seq_list: return None, None, None

    return np.array(X_tab_list), np.array(X_seq_list), np.array(y_list)

# ==========================================
# 2. DATA LOADER
# ==========================================
def load_and_prep_all():
    print("🚜 Loading data from axiom_db...")
    files = [f for f in os.listdir(DB_FOLDER) if f.endswith('.json')]
    
    all_X_tab = []
    all_X_seq = []
    all_y = []
    
    for f in files:
        path = os.path.join(DB_FOLDER, f)
        Xt, Xs, y = process_token_file(path)
        
        if Xt is not None:
            all_X_tab.append(Xt)
            all_X_seq.append(Xs)
            all_y.append(y)
            
    if not all_X_tab: 
        print("❌ No valid data found!")
        return None, None, None

    X_tab_final = np.concatenate(all_X_tab, axis=0)
    X_seq_final = np.concatenate(all_X_seq, axis=0)
    y_final = np.concatenate(all_y, axis=0)
    
    print(f"✅ Data Ready.")
    print(f"   Samples: {len(y_final)}")
    print(f"   Win Rate: {np.mean(y_final)*100:.2f}%")
    
    return X_tab_final, X_seq_final, y_final

# ==========================================
# 3. HYBRID MODEL CLASS (XGBoost + LSTM)
# ==========================================
class GraalHybrid:
    def __init__(self):
        # XGBoost (Tabular Expert)
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=500, 
            learning_rate=0.05, 
            max_depth=6, 
            scale_pos_weight=3.0, # Boost winners importance
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        # LSTM (Sequence Expert)
        inp = Input(shape=(SEQUENCE_LENGTH, 4))
        x = LSTM(64, return_sequences=False)(inp)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        out = Dense(1, activation='sigmoid')(x)
        
        self.rnn = Model(inp, out)
        self.rnn.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        
    def train(self, Xt, Xs, y):
        print("🚂 Training XGBoost (Structure)...")
        self.xgb_model.fit(Xt, y)
        
        print("🌊 Training LSTM (Momentum)...")
        self.rnn.fit(Xs, y, epochs=10, batch_size=64, validation_split=0.2, verbose=1)
        
    def predict(self, Xt, Xs):
        # 1. Get XGBoost Prediction
        p_xgb = self.xgb_model.predict_proba(Xt)[:, 1]
        
        # 2. Get LSTM Prediction
        p_rnn = self.rnn.predict(Xs).flatten()
        
        # 3. Weighted Ensemble (50/50 split)
        return (p_xgb * 0.5) + (p_rnn * 0.5)

# ==========================================
# 4. RUNNER
# ==========================================
if __name__ == "__main__":
    Xt, Xs, y = load_and_prep_all()
    
    if Xt is not None:
        bot = GraalHybrid()
        bot.train(Xt, Xs, y)
        
        # Test Prediction on first 5 samples
        preds = bot.predict(Xt[:5], Xs[:5])
        print(f"\n🔮 Sample Predictions: {preds}")
        print("🎉 Hybrid Model Trained Successfully.")