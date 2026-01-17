// bridge.js
window.addEventListener("message", function (event) {
    if (event.source != window) return;
    if (event.data.type && (event.data.type == "AXIOM_GEM_FOUND")) {
        // Pass the hot potato to Background.js
        chrome.runtime.sendMessage(event.data.data);
    }
});