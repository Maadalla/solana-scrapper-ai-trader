import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ==========================================
# ⚙️ CONFIGURATION (The Scalper's Rules)
# ==========================================
DB_FOLDER = "axiom_db"
MODEL_FILE = "graal_model.json"

# DEFINITION OF A WIN (The Target)
# We look 5 minutes into the future from the current snapshot.
LOOKAHEAD_SECONDS = 300 
TARGET_PROFIT = 1.25    # +25% Gain (Take Profit)
STOP_LOSS = 0.90        # -10% Loss (Stop Loss)

# ==========================================
# 1. FEATURE ENGINEERING (The Eyes)
# ==========================================
def calculate_slope(prices):
    """Calculates linear regression slope of prices (Price Momentum)"""
    if len(prices) < 2: return 0
    x = np.arange(len(prices))
    y = np.array(prices)
    # Simple linear regression slope
    slope = np.polyfit(x, y, 1)[0]
    # Normalize by first price to get % change per tick
    return (slope / prices[0]) * 100

def extract_features(history, consolidated_chart, index):
    """
    Creates the Input Vector (X) for a specific moment in time.
    STRICT RULE: No looking ahead of history[index]['timestamp'].
    """
    current_snap = history[index]
    curr_stats = current_snap['stats']
    
    # --- A. FUNDAMENTAL VELOCITY ---
    # We compare current snapshot vs previous snapshot (approx 10s ago)
    prev_snap = history[index - 1] if index > 0 else current_snap
    
    dt = (current_snap['time_offset'] - prev_snap['time_offset']) / 1000 # Delta time in seconds
    if dt == 0: dt = 1 # Avoid div by zero
    
    # 1. Holder Velocity (New Holders per Second)
    d_holders = curr_stats.get('numHolders', 0) - prev_snap['stats'].get('numHolders', 0)
    vel_holders = d_holders / dt
    
    # 2. Fee Velocity (SOL Paid per Second)
    d_fees = curr_stats.get('totalPairFeesPaid', 0) - prev_snap['stats'].get('totalPairFeesPaid', 0)
    vel_fees = d_fees / dt
    
    # 3. Sniper Decay (Are snipers leaving?)
    curr_snipers = curr_stats.get('snipersHoldPercent', 0)
    prev_snipers = prev_snap['stats'].get('snipersHoldPercent', 0)
    d_snipers = curr_snipers - prev_snipers # Negative is GOOD

    # --- B. TECHNICAL MOMENTUM (The Blindfolded Chart) ---
    cutoff_ts = current_snap.get('chart_cutoff', 0)
    
    # Filter candles that existed AT THIS EXACT MOMENT
    # We take the last 60 seconds (approx 60 candles)
    past_candles = [b for b in consolidated_chart if b['time'] <= cutoff_ts]
    recent_candles = past_candles[-60:] if len(past_candles) > 60 else past_candles
    
    if not recent_candles:
        return None # No chart data yet
    
    closes = [b['close'] for b in recent_candles]
    volumes = [b['volume'] for b in recent_candles]
    
    # 4. Price Slope (Trend)
    slope = calculate_slope(closes)
    
    # 5. Volatility (High - Low) / Open
    last_candle = recent_candles[-1]
    volatility = (last_candle['high'] - last_candle['low']) / last_candle['open']

    # 6. Buying Pressure (Green Vol / Total Vol)
    green_vol = sum([b['volume'] for b in recent_candles if b['close'] > b['open']])
    total_vol = sum(volumes) if sum(volumes) > 0 else 1
    buy_pressure = green_vol / total_vol

    current_price = last_candle['close']

    return {
        "holders": curr_stats.get('numHolders', 0),
        "fees_total": curr_stats.get('totalPairFeesPaid', 0),
        "vel_holders": vel_holders,
        "vel_fees": vel_fees,
        "sniper_delta": d_snipers,
        "price_slope": slope,
        "volatility": volatility,
        "buy_pressure": buy_pressure,
        "current_price": current_price # Used for labeling, deleted later
    }

# ==========================================
# 2. LABELING (The Judge)
# ==========================================
def get_label(current_price, cutoff_ts, consolidated_chart):
    """
    Looks into the future (Consolidated Chart) to see if we Won or Lost.
    """
    # Filter for FUTURE candles (Time > cutoff) within Lookahead window
    future_candles = [
        b for b in consolidated_chart 
        if b['time'] > cutoff_ts and b['time'] <= cutoff_ts + (LOOKAHEAD_SECONDS * 1000)
    ]
    
    if not future_candles:
        return 0 # No future data (end of file), assume no pump
    
    max_price = 0
    min_price = float('inf')
    
    for b in future_candles:
        max_price = max(max_price, b['high'])
        min_price = min(min_price, b['low'])
        
        # STOP LOSS HIT?
        if min_price < current_price * STOP_LOSS:
            return 0 
            
        # TAKE PROFIT HIT?
        if max_price > current_price * TARGET_PROFIT:
            return 1 # WIN
            
    return 0 # Stagnant

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def run_training():
    print("🧠 GRAAL HUNTER: Training Phase Initialized...")
    
    files = [f for f in os.listdir(DB_FOLDER) if f.endswith('.json')]
    print(f"📂 Found {len(files)} datasets.")
    
    X = []
    y = []
    
    total_snapshots = 0
    
    for file in files:
        try:
            with open(os.path.join(DB_FOLDER, file), 'r') as f:
                data = json.load(f)
                
            history = data.get('history', [])
            chart = data.get('consolidated_chart', [])
            
            if not history or not chart: continue
            
            # Loop through every 10s snapshot
            for i in range(len(history)):
                feat = extract_features(history, chart, i)
                if not feat: continue
                
                # Get Label
                cutoff = history[i]['chart_cutoff']
                price = feat['current_price']
                label = get_label(price, cutoff, chart)
                
                # Clean up feature dict (remove price)
                del feat['current_price']
                
                X.append(feat)
                y.append(label)
                total_snapshots += 1
                
        except Exception as e:
            pass # Skip corrupted files

    # Create DataFrame
    df = pd.DataFrame(X)
    labels = np.array(y)
    
    print(f"📊 Training Data: {len(df)} snapshots.")
    winners = sum(labels)
    print(f"🏆 Win Signals Found: {winners} ({winners/len(labels)*100:.2f}%)")
    
    if winners < 10:
        print("⚠️ Not enough winners to train. Lower TARGET_PROFIT or collect more data.")
        return

    # Train XGBoost
    print("🚀 Training XGBoost Classifier...")
    
    # Scale_pos_weight fixes the imbalance (Winners are rare)
    weight = (len(labels) - winners) / winners
    
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=weight,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluation
    preds = model.predict(X_test)
    print("\n📝 RESULTS REPORT:")
    print(classification_report(y_test, preds))
    
    # Feature Importance (The "Why")
    print("\n🔑 KEY INDICATORS (Top 5):")
    imps = model.feature_importances_
    sorted_idx = np.argsort(imps)[::-1]
    for i in range(5):
        print(f"   {df.columns[sorted_idx[i]]}: {imps[sorted_idx[i]]:.4f}")

    # Save
    model.save_model(MODEL_FILE)
    print(f"\n💾 BRAIN SAVED: {MODEL_FILE}")

if __name__ == "__main__":
    run_training()