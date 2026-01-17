import os
import json
import time
from flask import Flask, request
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

DB_FOLDER = "axiom_db"
if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

def merge_charts(existing_bars, new_bars):
    """
    Merges new candles into the existing list without duplicates.
    Assumes both lists are sorted by time.
    """
    if not existing_bars:
        return new_bars
    
    # Create a dict for O(1) lookup by timestamp
    # We use the existing bars as the base
    bar_map = {b['time']: b for b in existing_bars}
    
    # Update/Add new bars
    for b in new_bars:
        bar_map[b['time']] = b
        
    # Convert back to sorted list
    merged = sorted(bar_map.values(), key=lambda x: x['time'])
    return merged

@app.route('/new_token', methods=['POST'])
def receive_token():
    try:
        incoming_data = request.json
        contract = incoming_data.get('contract')
        
        if not contract:
            return {"status": "error", "message": "No contract"}, 400

        filename = os.path.join(DB_FOLDER, f"{contract}.json")
        
        # 1. LOAD EXISTING DATA (if any)
        existing_data = {}
        if os.path.exists(filename):
            try:
                with open(filename, "r") as f:
                    existing_data = json.load(f)
            except:
                pass # Corrupt file or empty

        # 2. INITIALIZE MASTER STRUCTURE
        # We prefer the existing data's structure to persist history
        master_record = existing_data if existing_data else incoming_data
        
        # Ensure keys exist
        if 'history' not in master_record: master_record['history'] = []
        if 'consolidated_chart' not in master_record: master_record['consolidated_chart'] = []

        # 3. PROCESS INCOMING HISTORY
        # The extension sends an array of "snapshots" (stats + chart)
        # We need to extract the charts, merge them, and strip them from the history entry
        
        incoming_history = incoming_data.get('history', [])
        
        for snapshot in incoming_history:
            # Check if this timestamp is already processed
            # We use time_offset as a unique ID for the snapshot
            offset = snapshot.get('time_offset')
            
            # Check if we already have this snapshot in our master history
            exists = any(h.get('time_offset') == offset for h in master_record['history'])
            
            if not exists:
                # A. Handle Chart
                snapshot_chart_bars = []
                if 'chart' in snapshot and snapshot['chart'] and 'bars' in snapshot['chart']:
                    snapshot_chart_bars = snapshot['chart']['bars']
                
                # Merge these bars into the Master Chart
                master_record['consolidated_chart'] = merge_charts(
                    master_record['consolidated_chart'], 
                    snapshot_chart_bars
                )
                
                # Find the "Cutoff" (The last candle available at this moment)
                last_candle_ts = 0
                if snapshot_chart_bars:
                    last_candle_ts = snapshot_chart_bars[-1]['time']

                # B. Create Lightweight History Entry
                clean_entry = {
                    "time_offset": offset,
                    "timestamp": snapshot.get('timestamp'),
                    "stats": snapshot.get('stats'),
                    "chart_cutoff": last_candle_ts  # <--- THE MAGIC REFERENCE
                }
                
                master_record['history'].append(clean_entry)

        # 4. UPDATE RAW STATS / LATEST INFO
        # We always want the latest 'raw_stats', 'pair', etc.
        master_record['raw_stats'] = incoming_data.get('raw_stats', master_record.get('raw_stats'))
        master_record['t20_stats'] = incoming_data.get('t20_stats', master_record.get('t20_stats'))
        
        # 5. SORT & SAVE
        # Sort history by time just in case
        master_record['history'].sort(key=lambda x: x['time_offset'])
        
        with open(filename, "w") as f:
            json.dump(master_record, f, indent=4)
            
        print(f"   💾 [UPDATE] {contract}: {len(master_record['history'])} snapshots, {len(master_record['consolidated_chart'])} bars.")

        return {"status": "saved"}, 200

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"status": "error"}, 500

if __name__ == '__main__':
    print("🤖 GRAAL SERVER (Smart Deduplication) LISTENING...")
    app.run(port=5000, threaded=True)