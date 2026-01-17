// background.js
console.log("⚡ GRAAL HUNTER: 8-Min Recorder with GRADUATED THRESHOLDS");

// ==================================================================
// 1. INITIALIZATION
// ==================================================================
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.contract && request.pair) {
        const contract = request.contract;
        const ticker = request.raw_stats.token_ticker;

        console.log(`⏱️ [START] New Token: ${ticker}. Initializing...`);

        const record = {
            ...request,
            history: [],
            status: 'active',
            start_time: Date.now()
        };

        chrome.storage.local.set({ [contract]: record }, () => {
            // Schedule Gatekeeper at T+20s
            createAlarm(contract, 'gatekeeper', 20 / 60);
        });
    }
});

// ==================================================================
// 2. ALARM DISPATCHER
// ==================================================================
chrome.alarms.onAlarm.addListener((alarm) => {
    const [contract, action] = alarm.name.split("::");
    if (!contract || !action) return;

    if (action === 'gatekeeper') {
        processGatekeeper(contract);
    } else if (action === 'poll_10s') {
        processPolling(contract);
    }
});

// ==================================================================
// 3. PHASE 1: THE GATEKEEPER (T+20s)
// Strict Filter: 20 Holders + 0.1 Fees
// ==================================================================
function processGatekeeper(contract) {
    chrome.storage.local.get([contract], (result) => {
        const payload = result[contract];
        if (!payload) return;

        const ticker = payload.raw_stats.token_ticker;
        console.log(`🔍 [T+20s] Gatekeeper Checking: ${ticker}`);

        fetchWithRetry(`https://api6.axiom.trade/token-info?pairAddress=${payload.pair}`)
            .then(data => {
                const holders = data.numHolders || 0;
                const fees = data.totalPairFeesPaid || 0;

                // FILTER: Holders >= 20 AND Fees >= 0.1
                if (holders >= 20 && fees >= 0.1) {
                    console.log(`✅ [T+20s] PASSED: ${ticker} (${holders} H, ${fees} F). Starting Loop...`);

                    // Save snapshot
                    payload.t20_stats = data;

                    // Notify Server
                    sendToPython(payload);

                    chrome.storage.local.set({ [contract]: payload });

                    // Start the 10s Loop
                    createAlarm(contract, 'poll_10s', 10 / 60);

                } else {
                    console.log(`💀 [T+20s] KILLED: ${ticker} (Low Stats: ${holders}H, ${fees}F).`);
                    chrome.storage.local.remove(contract);
                }
            })
            .catch(err => {
                console.log(`⚠️ Gatekeeper Error for ${ticker}: ${err.message}. Killing.`);
                chrome.storage.local.remove(contract);
            });
    });
}

// ==================================================================
// 4. PHASE 2: THE RECORDER (T+30s to T+8m)
// "Graduated Survival" Logic Implemented Here
// ==================================================================
function processPolling(contract) {
    chrome.storage.local.get([contract], (result) => {
        const payload = result[contract];
        if (!payload) return;

        const ticker = payload.raw_stats.token_ticker;
        const now = Date.now();
        const timeElapsed = now - payload.start_time; // ms
        const seconds = Math.floor(timeElapsed / 1000);

        console.log(`🎥 [T+${seconds}s] Recording: ${ticker}`);

        const jitter = Math.floor(Math.random() * 500) + 100;

        setTimeout(async () => {
            try {
                // A. FETCH STATS
                const stats = await fetchWithRetry(`https://api6.axiom.trade/token-info?pairAddress=${payload.pair}`);

                const holders = stats.numHolders || 0;
                const fees = stats.totalPairFeesPaid || 0;

                // ---------------------------------------------------------
                // 🛑 GRADUATED SURVIVAL CHECK (The Zombie Killer)
                // ---------------------------------------------------------
                let minFees = 0.1;
                let minHolders = 20;

                // The Ladder
                if (seconds >= 30) minFees = 0.2;
                if (seconds >= 40) minFees = 0.3;
                if (seconds >= 50) minFees = 0.4;
                if (seconds >= 60) {
                    minFees = 1;
                    minHolders = 30; // Ramp up holders requirement at 1 min
                }

                // Extra safety: At 3 mins, if fees are still low (under 1), kill it?
                // Optional but recommended based on your logs
                if (seconds >= 180) { minFees = 1.0; }

                if (holders < minHolders || fees < minFees) {
                    console.log(`💀 [KILL SWITCH] ${ticker} Failed at T+${seconds}s.`);
                    console.log(`   Expected: >${minHolders}H, >${minFees}F | Actual: ${holders}H, ${fees}F`);
                    chrome.storage.local.remove(contract);
                    return; // Stop execution
                }

                // ---------------------------------------------------------

                // B. FETCH CHART (1s Candles)
                const lookback = 8 * 60 * 1000;
                const startTime = now - lookback;

                let openTime = startTime;
                if (payload.raw_stats.open_trading) {
                    try { openTime = new Date(payload.raw_stats.open_trading).getTime(); } catch (e) { }
                }

                const chartUrl = `https://api.axiom.trade/pair-chart?pairAddress=${payload.pair}` +
                    `&from=${startTime}&to=${now}` +
                    `&currency=USD` +
                    `&interval=1s` +
                    `&openTrading=${openTime}` +
                    `&lastTransactionTime=${now - 1000}` +
                    `&countBars=600&showOutliers=false&isNew=false&v=2`;

                const chart = await fetchWithRetry(chartUrl);

                // C. SAVE DATA
                const snapshot = {
                    time_offset: timeElapsed,
                    timestamp: now,
                    stats: stats,
                    chart: chart
                };

                payload.history.push(snapshot);

                // PUSH TO SERVER
                sendToPython(payload);

                // Update Storage
                chrome.storage.local.set({ [contract]: payload });

                // D. CHECK DURATION (8 Minutes)
                if (timeElapsed < (8 * 60 * 1000)) {
                    createAlarm(contract, 'poll_10s', 10 / 60);
                } else {
                    console.log(`🏁 [COMPLETE] Finished 8m Recording for ${ticker}.`);
                    chrome.storage.local.remove(contract);
                }

            } catch (err) {
                console.log(`⚠️ Poll Error ${ticker}: ${err.message}. Skipping tick.`);
                if (timeElapsed < (8 * 60 * 1000)) {
                    createAlarm(contract, 'poll_10s', 10 / 60);
                }
            }
        }, jitter);
    });
}

// ==================================================================
// 5. HELPER: FETCH WITH RETRY
// ==================================================================
async function fetchWithRetry(url, retries = 3, backoff = 2000) {
    try {
        const response = await fetch(url);

        if (response.status === 429) {
            if (retries > 0) {
                console.warn(`🛑 429 Rate Limit. Retrying in ${backoff}ms...`);
                await new Promise(r => setTimeout(r, backoff));
                return fetchWithRetry(url, retries - 1, backoff * 1.5);
            } else {
                throw new Error("Max Retries (429)");
            }
        }

        if (!response.ok && response.status >= 500) {
            if (retries > 0) {
                await new Promise(r => setTimeout(r, 1000));
                return fetchWithRetry(url, retries - 1, backoff);
            }
        }

        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();

    } catch (err) {
        if (retries > 0 && err.message.includes("Failed to fetch")) {
            await new Promise(r => setTimeout(r, 1000));
            return fetchWithRetry(url, retries - 1, backoff);
        }
        throw err;
    }
}

// ==================================================================
// 6. UTILS
// ==================================================================
function createAlarm(contract, action, delayMins) {
    chrome.alarms.create(`${contract}::${action}`, { delayInMinutes: delayMins });
}

function sendToPython(payload) {
    fetch('http://localhost:5000/new_token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    }).catch(err => console.error("Python Push Error:", err));


}