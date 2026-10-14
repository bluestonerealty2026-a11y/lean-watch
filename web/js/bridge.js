/* Connection to the local Python agent (agent/leanwatch/server.py).
   The agent serves this page itself and opens it as
   http://127.0.0.1:8765/?token=SECRET, so the page knows where to connect and
   proves it was opened by you. Opened any other way, the page stays in
   practice-screen mode. */
(function () {
  "use strict";
  const LW = window.LW;
  const { Emitter } = LW.util;

  const LOCAL_HOSTS = new Set(["127.0.0.1", "localhost", "[::1]"]);

  function readToken() {
    let token = null;
    try {
      const params = new URLSearchParams(location.search);
      token = params.get("token");
      if (token) {
        sessionStorage.setItem("lw-token", token);
        params.delete("token");
        const rest = params.toString();
        history.replaceState(null, "", location.pathname + (rest ? `?${rest}` : "") + location.hash);
      } else {
        token = sessionStorage.getItem("lw-token");
      }
    } catch { /* storage or history blocked */ }
    return token;
  }

  class Bridge extends Emitter {
    constructor() {
      super();
      this.token = readToken();
      this.isLocalPage = /^https?:$/.test(location.protocol) && LOCAL_HOSTS.has(location.hostname);
      this.state = "offline";
      this.info = null;
      this.ws = null;
      this._retry = 0;
    }

    /** Why the bridge can't connect, in words a person can act on, or null if it can. */
    get blocker() {
      if (!this.isLocalPage) return "Start the local agent with: python -m leanwatch. It opens this page already connected.";
      if (!this.token) return "This page was opened without its session token. Close it and run python -m leanwatch again.";
      return null;
    }

    connect() {
      if (this.blocker || this.ws) return;
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${location.host}/ws?token=${encodeURIComponent(this.token)}`);
      this.ws = ws;
      this._set("connecting");
      ws.addEventListener("open", () => { this._retry = 0; this._set("online"); this.send({ type: "hello" }); });
      ws.addEventListener("message", (e) => {
        let msg;
        try { msg = JSON.parse(e.data); } catch { return; }
        if (msg.type === "status") this.info = msg;
        this.emit(msg.type, msg);
        this.emit("message", msg);
      });
      ws.addEventListener("close", () => {
        this.ws = null;
        this._set("offline");
        const delay = Math.min(15000, 1000 * 2 ** this._retry++);
        setTimeout(() => this.connect(), delay);
      });
      ws.addEventListener("error", () => { /* close follows */ });
    }

    send(msg) {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) { this.ws.send(JSON.stringify(msg)); return true; }
      return false;
    }

    _set(state) {
      if (this.state === state) return;
      this.state = state;
      this.emit("state", { state });
    }
  }

  LW.Bridge = Bridge;
})();
