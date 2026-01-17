// spy.js
(function () {
    console.log("🕵️ SPY LOADED: TRIGGER MODE");

    const processedCoins = new Set();
    const TARGET_URL = 'wss://cluster9.axiom.trade/';

    function connect() {
        const ws = new WebSocket(TARGET_URL);

        ws.onopen = function () {
            console.log("✅ CONNECTED. JOINING ROOM...");
            ws.send(JSON.stringify({ action: "join", room: "new_pairs" }));
        };

        ws.onmessage = function (event) {
            try {
                if (event.data === '2' || event.data.includes('block_hash')) return;
                const parsed = JSON.parse(event.data);

                if (parsed.room === 'new_pairs' && parsed.content) {
                    const coin = parsed.content;
                    if (processedCoins.has(coin.token_address)) return;

                    // 1. BASIC SAFETY FILTER (T=0)
                    const isSafe =
                        coin.dev_holds_percent < 20 &&
                        coin.initial_liquidity_sol >= 15;

                    if (isSafe) {
                        // Hand off to Background immediately
                        processedCoins.add(coin.token_address);
                        window.postMessage({
                            type: "AXIOM_GEM_FOUND",
                            data: {
                                contract: coin.token_address,
                                pair: coin.pair_address,
                                raw_stats: coin
                            }
                        }, "*");
                    }
                }
            } catch (e) { }
        };
        ws.onclose = function () { setTimeout(connect, 1000); };
    }
    connect();
})();