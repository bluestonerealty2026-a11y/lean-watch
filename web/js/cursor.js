/* The agent's on-screen pointer.
   Movement follows two classic human-motor-control models so it is easy to
   follow with your eyes:
   - Fitts's law sets the duration: T = a + b * log2(D / W + 1)
   - a minimum-jerk profile sets the speed along a gently curved Bezier path:
     s(t) = 10t^3 - 15t^4 + 6t^5  (slow start, fast middle, soft landing) */
(function () {
  "use strict";
  const LW = window.LW;
  const { h, clamp, lerp, prefersReducedMotion } = LW.util;

  const FITTS_A = 220;   // ms, reaction + start-up time
  const FITTS_B = 140;   // ms per bit of difficulty
  const TARGET_W = 24;   // px, typical button width the cursor aims for

  const minJerk = (t) => t * t * t * (10 - 15 * t + 6 * t * t);
  const bezier = (p0, p1, p2, p3, s) => {
    const u = 1 - s;
    return u * u * u * p0 + 3 * u * u * s * p1 + 3 * u * s * s * p2 + s * s * s * p3;
  };
  const nextFrame = () => new Promise((r) => requestAnimationFrame(() => r()));

  const ARROW = '<svg viewBox="0 0 28 28" class="lw-cursor__arrow"><path d="M5 2.5 L5 22 L10.2 17.2 L14 25.5 L17.6 24 L13.8 15.8 L20.5 15.8 Z"/></svg>';

  class VirtualCursor {
    constructor(os) {
      this.os = os;
      this.x = os.W * 0.55;
      this.y = os.H * 0.45;
      this.lastMoveMs = 0;
      this._listeners = new Set();
      this.el = h("div.lw-cursor", { "aria-hidden": "true", hidden: true }, h("span.lw-cursor__halo"), h("span", { html: ARROW }));
      this.bubble = h("div.lw-bubble", { "aria-hidden": "true", hidden: true });
      this.spot = h("div.lw-spot", { "aria-hidden": "true", hidden: true });
      os.root.append(this.spot, this.el, this.bubble);
      this._place();
    }

    onMove(fn) { this._listeners.add(fn); return () => this._listeners.delete(fn); }
    show() { this.el.hidden = false; }
    hide() { this.el.hidden = true; this.say(null); }

    _place() {
      this.el.style.transform = `translate(${this.x}px, ${this.y}px)`;
      if (!this.bubble.hidden) this._placeBubble();
      for (const fn of this._listeners) fn({ x: this.x, y: this.y, ms: this.lastMoveMs });
    }

    _placeBubble() {
      const bw = this.bubble.offsetWidth || 240;
      const bh = this.bubble.offsetHeight || 60;
      let bx = this.x + 26;
      let by = this.y + 22;
      if (bx + bw > this.os.W - 8) bx = this.x - bw - 14;
      if (by + bh > this.os.H - 52) by = this.y - bh - 14;
      this.bubble.style.transform = `translate(${clamp(bx, 6, this.os.W - bw - 6)}px, ${clamp(by, 6, this.os.H - bh - 6)}px)`;
    }

    /** Show a caption next to the pointer, or hide it with null. */
    say(text) {
      if (!text) { this.bubble.hidden = true; return; }
      this.bubble.textContent = text;
      this.bubble.hidden = false;
      this._placeBubble();
    }

    /** Fitts's-law movement time in ms for a move of `dist` pixels. */
    static movementTime(dist, width = TARGET_W) {
      return FITTS_A + FITTS_B * Math.log2(dist / width + 1);
    }

    /** Animate to (x, y). onStep(point) runs every frame, e.g. to send drag events. */
    async moveTo(x, y, ctrl, onStep) {
      x = clamp(x, 0, this.os.W - 1);
      y = clamp(y, 0, this.os.H - 1);
      const x0 = this.x, y0 = this.y;
      const dist = Math.hypot(x - x0, y - y0);
      this.show();
      if (dist < 1) return;
      let duration = VirtualCursor.movementTime(dist) / ctrl.speed;
      if (prefersReducedMotion()) duration = Math.min(duration, 180);
      this.lastMoveMs = Math.round(duration);

      // Control points pushed sideways from the straight line give a natural arc.
      const bend = clamp(dist * 0.18, 8, 70) * (Math.random() < 0.5 ? -1 : 1);
      const nx = -(y - y0) / dist, ny = (x - x0) / dist;
      const c1x = lerp(x0, x, 0.3) + nx * bend, c1y = lerp(y0, y, 0.3) + ny * bend;
      const c2x = lerp(x0, x, 0.75) + nx * bend * 0.4, c2y = lerp(y0, y, 0.75) + ny * bend * 0.4;

      let elapsed = 0;
      let last = performance.now();
      while (elapsed < duration) {
        await nextFrame();
        if (ctrl.paused || ctrl.stopped) { await ctrl.gate(); last = performance.now(); continue; }
        const now = performance.now();
        elapsed += now - last;
        last = now;
        const s = minJerk(Math.min(1, elapsed / duration));
        this.x = bezier(x0, c1x, c2x, x, s);
        this.y = bezier(y0, c1y, c2y, y, s);
        this._place();
        if (onStep) onStep({ x: this.x, y: this.y });
      }
      this.x = x;
      this.y = y;
      this._place();
    }

    /** Visual feedback for a press: the arrow dips and a ring expands. */
    pulse(kind = "click") {
      const ring = h("span.lw-ripple", { class: `lw-ripple lw-ripple--${kind}`, style: { transform: `translate(${this.x}px, ${this.y}px)` } });
      this.os.root.append(ring);
      this.el.classList.remove("is-pressing");
      void this.el.offsetWidth;
      this.el.classList.add("is-pressing");
      setTimeout(() => ring.remove(), 700);
    }

    /** Pulsing outline around an element, used by "Now you try" practice mode. */
    spotlight(el) {
      if (!el) { this.spot.hidden = true; return; }
      const r = this.os.rectOf(el);
      Object.assign(this.spot.style, { left: `${r.x - 6}px`, top: `${r.y - 6}px`, width: `${r.w + 12}px`, height: `${r.h + 12}px` });
      this.spot.hidden = false;
    }
  }

  LW.VirtualCursor = VirtualCursor;
  LW.motion = { minJerk, movementTime: VirtualCursor.movementTime };
})();
