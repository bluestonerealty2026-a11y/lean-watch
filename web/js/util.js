/* LeanWatch shared helpers. Every script attaches to window.LW so the page
   works when opened straight from disk (file://) without a build step. */
(function () {
  "use strict";
  const LW = (window.LW = window.LW || {});

  /** Create an element: h("button.btn", {onclick}, "Label", child) */
  function h(tag, attrs, ...children) {
    const [name, ...classes] = tag.split(".");
    const el = document.createElement(name || "div");
    if (classes.length) el.className = classes.join(" ");
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
      else if (k === "dataset") Object.assign(el.dataset, v);
      else if (k === "html") el.innerHTML = v;
      else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat()) {
      if (c == null || c === false) continue;
      el.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return el;
  }

  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const lerp = (a, b, t) => a + (b - a) * t;

  /** Find the element inside root whose data-target equals name (safe for any characters). */
  function byTarget(root, name) {
    for (const el of root.querySelectorAll("[data-target]")) {
      if (el.dataset.target === name) return el;
    }
    return null;
  }

  /** Small event emitter. */
  class Emitter {
    constructor() { this._h = new Map(); }
    on(type, fn) {
      if (!this._h.has(type)) this._h.set(type, new Set());
      this._h.get(type).add(fn);
      return () => this._h.get(type).delete(fn);
    }
    emit(type, payload) {
      for (const fn of this._h.get(type) || []) {
        try { fn(payload); } catch (err) { console.error(err); }
      }
    }
  }

  /** Thrown when the user presses Stop. */
  class AbortedError extends Error {
    constructor() { super("Stopped"); this.name = "AbortedError"; }
  }

  /**
   * Run control shared by the agent and the cursor: stop() aborts,
   * pause()/resume() hold every sleep() until resumed.
   */
  class RunControl {
    constructor() {
      this.stopped = false;
      this.paused = false;
      this._resumeWaiters = [];
      this.speed = 1;
    }
    stop() { this.stopped = true; this.resume(); }
    pause() { this.paused = true; }
    resume() {
      this.paused = false;
      const waiters = this._resumeWaiters.splice(0);
      waiters.forEach((fn) => fn());
    }
    check() { if (this.stopped) throw new AbortedError(); }
    async gate() {
      this.check();
      while (this.paused) {
        await new Promise((r) => this._resumeWaiters.push(r));
        this.check();
      }
    }
    /** Sleep scaled by speed, interruptible by stop and held by pause. */
    async sleep(ms) {
      await this.gate();
      const end = performance.now() + ms / this.speed;
      while (performance.now() < end) {
        await new Promise((r) => setTimeout(r, Math.min(40, end - performance.now())));
        await this.gate();
      }
    }
  }

  /** localStorage that never throws (private windows, sandboxed frames). */
  const store = {
    get(key, fallback) {
      try {
        const raw = localStorage.getItem(key);
        return raw == null ? fallback : JSON.parse(raw);
      } catch { return fallback; }
    },
    set(key, value) {
      try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ }
    },
    remove(key) {
      try { localStorage.removeItem(key); } catch { /* storage unavailable */ }
    },
  };

  const deepClone = (o) => JSON.parse(JSON.stringify(o));
  const prefersReducedMotion = () =>
    window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  LW.util = { h, clamp, lerp, byTarget, Emitter, AbortedError, RunControl, store, deepClone, prefersReducedMotion };
})();
