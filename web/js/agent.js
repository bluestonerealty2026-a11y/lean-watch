/* The executor: runs a plan on the practice screen.
   It only "touches" the screen the way a person does: it moves the pointer to
   a point, presses, types and drags. Clicks land on whatever element is under
   the pointer, so a dialog covering a button blocks the click, just like on a
   real computer.

   Two modes:
   - watch:    the agent drives and explains each step.
   - practice: "Now you try". The screen is rewound, each target is outlined,
               and the agent waits for the person to do the step themselves. */
(function () {
  "use strict";
  const LW = window.LW;
  const { clamp, Emitter, RunControl, AbortedError, byTarget } = LW.util;

  const appName = (os, id) => (os.apps.get(id) ? os.apps.get(id).def.title.split(" — ")[0] : id);
  const sleepRaw = (ms) => new Promise((r) => setTimeout(r, ms));

  class Agent extends Emitter {
    constructor(os, cursor) {
      super();
      this.os = os;
      this.cursor = cursor;
      this.speed = 1;
      this.speakEnabled = false;
      this.ctrl = null;
      this.running = false;
      this.mode = null;
      this._confirm = null;
      this._lastTarget = null;
    }

    setSpeed(s) { this.speed = s; if (this.ctrl) this.ctrl.speed = s; }
    pause(reason) {
      if (!this.ctrl || this.ctrl.paused) return;
      this.ctrl.pause();
      this.emit("paused", { reason });
    }
    resume() {
      if (!this.ctrl || !this.ctrl.paused) return;
      this.ctrl.resume();
      this.emit("resumed", {});
    }
    stop() {
      if (!this.ctrl) return;
      this.ctrl.stop();
      if (this._confirm) this._confirm(false);
      if (window.speechSynthesis) window.speechSynthesis.cancel();
    }
    /** Answer a pending confirmation (Allow / Skip). */
    respond(allow) { if (this._confirm) this._confirm(!!allow); }

    /* ------------------------------------------------------------ */
    async run(plan, mode = "watch") {
      if (this.running) return;
      const flat = [];
      plan.items.forEach((item, itemIndex) => item.steps.forEach((step) => flat.push({ itemIndex, step })));
      const ctrl = (this.ctrl = new RunControl());
      ctrl.speed = this.speed;
      this.running = true;
      this.mode = mode;
      const before = mode === "watch" ? this.os.snapshot() : null;
      this.emit("start", { plan, flat, mode });
      if (mode === "watch") this.cursor.show(); else this.cursor.hide();

      let outcome = { ok: true };
      try {
        for (let i = 0; i < flat.length; i++) {
          await ctrl.gate();
          let { step } = flat[i];
          let say = step.say;

          if (step.do === "open") {
            const name = appName(this.os, step.app);
            if (this.os.isOpen(step.app) && this.os.isFocused(step.app)) {
              await this._skip(i, `${name} is already open, so I can carry on.`);
              continue;
            }
            if (this.os.isOpen(step.app)) say = `${name} is already open behind another window. I click its title bar to bring it to the front.`;
          }
          if (step.skipIf && step.skipIf(this.os)) {
            await this._skip(i, step.skipSay);
            continue;
          }

          this.emit("step", { index: i, status: "active" });
          if (step.do !== "verify") await this._narrate(say, i);
          const r = mode === "practice" ? await this._practice(step, say) : await this._exec(step);

          if (r.declined) {
            this.emit("step", { index: i, status: "declined" });
            for (let j = i + 1; j < flat.length; j++) this.emit("step", { index: j, status: "cancelled" });
            outcome = { ok: false, reason: "declined", message: "Okay, I stopped there. Nothing else was changed." };
            break;
          }
          if (!r.ok) {
            this.emit("step", { index: i, status: "failed", note: r.message });
            for (let j = i + 1; j < flat.length; j++) this.emit("step", { index: j, status: "cancelled" });
            outcome = { ok: false, reason: "failed", message: r.message || "That step did not work." };
            break;
          }
          this.emit("step", { index: i, status: "done", note: r.note });
          await ctrl.sleep(mode === "practice" ? 150 : 320);
        }
      } catch (err) {
        if (err instanceof AbortedError) outcome = { ok: false, reason: "stopped", message: "Stopped. You are in control." };
        else { console.error(err); outcome = { ok: false, reason: "error", message: `Something went wrong: ${err.message}` }; }
      } finally {
        this.running = false;
        this.ctrl = null;
        this._confirm = null;
        this.cursor.spotlight(null);
        setTimeout(() => { if (!this.running) this.cursor.say(null); }, 2600);
      }
      this.emit("done", { ...outcome, mode, before });
      return outcome;
    }

    async _skip(i, text) {
      this.emit("step", { index: i, status: "skipped", note: text });
      if (text) await this._narrate(text, i, 0.6);
    }

    async _narrate(text, index, weight = 1) {
      if (!text) return;
      this.cursor.say(text);
      this.emit("say", { text, index });
      const ctrl = this.ctrl;
      const spoken = this._speak(text);
      const readMs = clamp(500 + text.length * 22, 900, 3200) * weight;
      if (spoken) await Promise.race([spoken, (async () => { while (!ctrl.stopped) await sleepRaw(100); })()]);
      else if (this.mode === "watch") await ctrl.sleep(readMs);
      ctrl.check();
    }

    _speak(text) {
      if (!this.speakEnabled || !("speechSynthesis" in window)) return null;
      try {
        window.speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(text);
        u.rate = clamp(0.95 * this.speed, 0.7, 1.6);
        return new Promise((resolve) => {
          const t = setTimeout(resolve, text.length * 90 + 2000);
          u.onend = u.onerror = () => { clearTimeout(t); resolve(); };
          window.speechSynthesis.speak(u);
        });
      } catch { return null; }
    }

    /* ------------------------------------------------------------ */
    /* Watch mode: the agent acts                                    */
    /* ------------------------------------------------------------ */
    async _exec(step) {
      const os = this.os, ctrl = this.ctrl;
      switch (step.do) {
        case "open": {
          const el = os.isOpen(step.app) ? os.target(`win.${step.app}.title`) : os.target(`taskbar.${step.app}`);
          await this._moveTo(el);
          this._click(el, 1);
          await ctrl.sleep(350);
          return { ok: os.isOpen(step.app), message: `${appName(os, step.app)} did not open.` };
        }
        case "click": {
          const el = await this._find(step.target);
          if (!el) return { ok: false, message: `I couldn't find that on the screen (${step.target}).` };
          await this._moveTo(el);
          this._click(el, step.clicks || 1);
          this._lastTarget = step.target;
          await ctrl.sleep(260);
          return { ok: true };
        }
        case "type": {
          const el = this._typingTarget();
          if (!el) return { ok: false, message: "There is no text box selected to type into." };
          for (const ch of String(step.text)) {
            await ctrl.sleep(45 + Math.random() * 55);
            insertText(el, ch);
          }
          return { ok: true };
        }
        case "key": {
          this.cursor.say(`${step.ctrl ? "Ctrl + " : ""}${step.shift ? "Shift + " : ""}${step.key.length === 1 ? step.key.toUpperCase() : step.key}`);
          this.cursor.pulse("key");
          pressKey(this._keyTarget(), step);
          await ctrl.sleep(300);
          return { ok: true };
        }
        case "slider": {
          const el = await this._find(step.target);
          if (!el || !el.slider) return { ok: false, message: "I couldn't find that slider." };
          const from = el.slider.pointFor(el.slider.value);
          const to = el.slider.pointFor(step.value);
          await this.cursor.moveTo(from.x, from.y, ctrl);
          const c0 = os.toClient(from.x, from.y);
          firePointer(el, "pointerdown", c0);
          this.cursor.pulse("press");
          await ctrl.sleep(150);
          await this.cursor.moveTo(to.x, to.y, ctrl, (p) => firePointer(document, "pointermove", os.toClient(p.x, p.y)));
          firePointer(document, "pointermove", os.toClient(to.x, to.y));
          firePointer(document, "pointerup", os.toClient(to.x, to.y));
          await ctrl.sleep(200);
          return { ok: true };
        }
        case "confirm": return this._ask(step.say);
        case "verify": {
          await ctrl.sleep(250);
          const ok = !!step.test(os);
          await this._narrate(ok ? step.say : step.failSay, null, 0.8);
          return { ok, message: step.failSay };
        }
        case "wait": await ctrl.sleep(step.ms || 500); return { ok: true };
        default: return { ok: false, message: `Unknown step "${step.do}".` };
      }
    }

    async _ask(text) {
      this.emit("confirm", { text });
      const allow = await new Promise((resolve) => { this._confirm = resolve; });
      this._confirm = null;
      this.ctrl.check();
      this.emit("confirmed", { allow });
      return allow ? { ok: true } : { ok: true, declined: true };
    }

    async _find(name, timeout = 1500) {
      const t0 = performance.now();
      for (;;) {
        const el = byTarget(this.os.root, name);
        if (el && el.getClientRects().length) return el;
        if (performance.now() - t0 > timeout) return null;
        await this.ctrl.sleep(80);
      }
    }

    async _moveTo(el) {
      const p = this.os.centerOf(el);
      await this.cursor.moveTo(p.x, p.y, this.ctrl);
    }

    /** Press at the pointer: the event goes to the element under it (or the intended target when it's off-screen). */
    _click(target, count) {
      const os = this.os;
      const c = os.toClient(this.cursor.x, this.cursor.y);
      const hits = document.elementsFromPoint(c.x, c.y).filter((e) => os.root.contains(e) && e !== os.root);
      let el = hits[0];
      if (!el || !(target.contains(el) || el.contains(target))) el = target;
      for (let n = 1; n <= count; n++) {
        firePointer(el, "pointerdown", c);
        fireMouse(el, "mousedown", c, n);
        if (n === 1) focusFor(el, os);
        firePointer(el, "pointerup", c);
        fireMouse(el, "mouseup", c, n);
        fireMouse(el, "click", c, n);
        this.cursor.pulse("click");
      }
      if (count === 2) fireMouse(el, "dblclick", c, 2);
      if (count === 3) {
        const field = el.closest("input, textarea");
        if (field) field.select();
      }
    }

    _typingTarget() {
      const a = document.activeElement;
      if (a && this.os.root.contains(a) && /^(INPUT|TEXTAREA)$/.test(a.tagName)) return a;
      const last = this._lastTarget && byTarget(this.os.root, this._lastTarget);
      if (last && /^(INPUT|TEXTAREA)$/.test(last.tagName)) { last.focus({ preventScroll: true }); return last; }
      const renaming = byTarget(this.os.root, "files.rename.input");
      if (renaming) { renaming.focus({ preventScroll: true }); return renaming; }
      return null;
    }

    _keyTarget() {
      const a = document.activeElement;
      if (a && this.os.root.contains(a)) return a;
      return this.os.root.querySelector(".sim-win.is-focused") || this.os.root;
    }

    /* ------------------------------------------------------------ */
    /* Practice mode: the person acts, the agent coaches             */
    /* ------------------------------------------------------------ */
    async _practice(step, say) {
      const os = this.os, ctrl = this.ctrl;
      const coach = (el, text) => {
        if (el) {
          const p = os.centerOf(el);
          this.cursor.x = p.x; this.cursor.y = p.y;
          this.cursor.spotlight(el);
        } else this.cursor.spotlight(null);
        this.cursor.say(text);
      };
      switch (step.do) {
        case "open": {
          const el = os.isOpen(step.app) ? os.target(`win.${step.app}.title`) : os.target(`taskbar.${step.app}`);
          coach(el, `Your turn: ${os.isOpen(step.app) ? "click the title bar" : "click this icon"}.`);
          await this._waitFor(() => os.isOpen(step.app) && os.isFocused(step.app));
          return { ok: true };
        }
        case "click": {
          const el = await this._find(step.target, 4000);
          if (!el) return { ok: false, message: "The thing to click is not on screen. Try Watch again." };
          const verb = step.clicks === 2 ? "Double-click" : step.clicks === 3 ? "Click" : "Click";
          coach(el, `Your turn: ${verb.toLowerCase()} the outlined spot.`);
          const want = step.target;
          let hit = false;
          const evt = step.clicks === 2 ? "dblclick" : step.clicks === 3 ? "focusin" : "click";
          const handler = (e) => {
            if (!e.isTrusted) return;
            const t = e.target.closest && e.target.closest("[data-target]");
            if (t && t.dataset.target === want) hit = true;
          };
          document.addEventListener(evt, handler, true);
          try {
            await this._waitFor(() => hit, () => { const now = byTarget(os.root, want); if (now) this.cursor.spotlight(now); });
          } finally { document.removeEventListener(evt, handler, true); }
          if (step.clicks === 3) { const f = byTarget(os.root, want); if (f && f.select) f.select(); }
          this._lastTarget = want;
          return { ok: true };
        }
        case "type": {
          const field = this._typingTarget();
          coach(field, `Your turn: type ${step.text}`);
          const want = String(step.text).toLowerCase().trim();
          await this._waitFor(() => {
            const f = this._typingTarget();
            return f && f.value.toLowerCase().includes(want);
          });
          return { ok: true };
        }
        case "key": {
          const label = `${step.ctrl ? "Ctrl + " : ""}${step.key.length === 1 ? step.key.toUpperCase() : step.key}`;
          coach(null, `Your turn: press ${label} on your keyboard.`);
          let hit = false;
          const handler = (e) => {
            if (!e.isTrusted || !os.root.contains(e.target)) return;
            if (e.key.toLowerCase() === step.key.toLowerCase() && !!step.ctrl === (e.ctrlKey || e.metaKey)) hit = true;
          };
          document.addEventListener("keydown", handler, true);
          try { await this._waitFor(() => hit); } finally { document.removeEventListener("keydown", handler, true); }
          return { ok: true };
        }
        case "slider": {
          const el = await this._find(step.target, 4000);
          if (!el || !el.slider) return { ok: false, message: "The slider is not on screen." };
          coach(el, `Your turn: drag the slider to ${step.value}.`);
          const min = Number(el.getAttribute("aria-valuemin")), max = Number(el.getAttribute("aria-valuemax"));
          const tol = Math.max((max - min) * 0.03, max - min <= 3 ? 0.1 : 1);
          await this._waitFor(() => {
            const live = byTarget(os.root, step.target);
            return live && live.slider && Math.abs(live.slider.value - step.value) <= tol + 1e-9;
          });
          return { ok: true };
        }
        case "confirm": return this._ask(say);
        case "verify": {
          this.cursor.spotlight(null);
          const ok = !!step.test(os);
          await this._narrate(ok ? `Well done! ${step.say}` : step.failSay, null, 0.8);
          await ctrl.sleep(900);
          return { ok, message: step.failSay };
        }
        default: return { ok: true };
      }
    }

    async _waitFor(pred, onTick) {
      const ctrl = this.ctrl;
      for (;;) {
        await ctrl.gate();
        if (pred()) return;
        if (onTick) onTick();
        await sleepRaw(120);
      }
    }
  }

  /* ---------------- synthetic input helpers ---------------- */
  function fireMouse(el, type, c, detail = 1) {
    el.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true, composed: true, view: window, clientX: c.x, clientY: c.y, button: 0, buttons: type === "mousedown" ? 1 : 0, detail }));
  }
  function firePointer(el, type, c) {
    const init = { bubbles: true, cancelable: true, composed: true, clientX: c.x, clientY: c.y, button: 0, buttons: type === "pointerup" ? 0 : 1, pointerId: 1, pointerType: "mouse", isPrimary: true };
    let ev;
    try { ev = new PointerEvent(type, init); } catch { ev = new MouseEvent(type, init); }
    el.dispatchEvent(ev);
  }
  function focusFor(el, os) {
    const f = el.closest("input, textarea, [tabindex]");
    const active = document.activeElement;
    if (f && os.root.contains(f)) {
      if (f !== active) f.focus({ preventScroll: true });
    } else if (active && os.root.contains(active) && active !== el) {
      active.blur();
    }
  }
  function insertText(el, text) {
    const start = el.selectionStart == null ? el.value.length : el.selectionStart;
    const end = el.selectionEnd == null ? el.value.length : el.selectionEnd;
    el.setRangeText(text, start, end, "end");
    el.dispatchEvent(new InputEvent("input", { bubbles: true, data: text, inputType: "insertText" }));
  }
  function pressKey(el, step) {
    const init = { key: step.key, code: step.key.length === 1 ? `Key${step.key.toUpperCase()}` : step.key, bubbles: true, cancelable: true, ctrlKey: !!step.ctrl, shiftKey: !!step.shift, altKey: !!step.alt };
    el.dispatchEvent(new KeyboardEvent("keydown", init));
    el.dispatchEvent(new KeyboardEvent("keyup", init));
  }

  LW.Agent = Agent;
})();
