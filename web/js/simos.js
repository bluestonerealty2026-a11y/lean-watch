/* SimOS: the pretend computer the agent practises on.
   It has windows, a taskbar, a shared file system and system settings.
   Everything is laid out in a fixed 1000x625 "logical pixel" screen that is
   scaled to fit, so the agent can think in screen coordinates like a real
   computer-use model does. */
(function () {
  "use strict";
  const LW = window.LW;
  const { h, clamp, Emitter, deepClone } = LW.util;

  const SCREEN_W = 1000;
  const SCREEN_H = 625;
  const TASKBAR_H = 44;

  LW.apps = LW.apps || [];

  const DEFAULT_SETTINGS = {
    dark: false,
    nightlight: false,
    brightness: 80,
    wifi: true,
    bluetooth: true,
    volume: 60,
  };

  const DEFAULT_FS = {
    type: "dir",
    children: {
      Documents: { type: "dir", children: { "notes.txt": { type: "file", size: 2 }, "budget.xlsx": { type: "file", size: 18 } } },
      Pictures: { type: "dir", children: { "landscape.jpg": { type: "file", size: 412 } } },
      Music: { type: "dir", children: {} },
      "todo.txt": { type: "file", size: 1 },
    },
  };

  const TRAY_ICONS = {
    wifi: '<svg viewBox="0 0 24 24"><path d="M2 9a15 15 0 0 1 20 0M5 12.5a10 10 0 0 1 14 0M8.5 16a5 5 0 0 1 7 0"/><circle cx="12" cy="19.5" r="1.3"/></svg>',
    wifiOff: '<svg viewBox="0 0 24 24"><path d="M2 9a15 15 0 0 1 20 0M5 12.5a10 10 0 0 1 14 0M8.5 16a5 5 0 0 1 7 0M3 3l18 18"/></svg>',
    bt: '<svg viewBox="0 0 24 24"><path d="M7 7l10 10-5 4V3l5 4L7 17"/></svg>',
    btOff: '<svg viewBox="0 0 24 24"><path d="M7 7l10 10-5 4V3l5 4L7 17M3 3l18 18"/></svg>',
    vol: '<svg viewBox="0 0 24 24"><path d="M4 9h4l5-4v14l-5-4H4zM16 9a4 4 0 0 1 0 6M18.5 6.5a8 8 0 0 1 0 11"/></svg>',
    mute: '<svg viewBox="0 0 24 24"><path d="M4 9h4l5-4v14l-5-4H4zM16 9l5 6M21 9l-5 6"/></svg>',
  };

  class SimOS extends Emitter {
    constructor(mount) {
      super();
      this.W = SCREEN_W;
      this.H = SCREEN_H;
      this.settings = deepClone(DEFAULT_SETTINGS);
      this.fsRoot = deepClone(DEFAULT_FS);
      this.homeLabel = "C:\\Users\\You";
      this.apps = new Map();
      this.z = 10;

      this.root = h("div.simos", { "aria-label": "Practice computer screen" });
      this.wallpaper = h("div.sim-wallpaper");
      this.windowsLayer = h("div.sim-windows");
      this.taskbar = h("div.sim-taskbar");
      this.nightOverlay = h("div.sim-overlay.sim-overlay--night");
      this.dimOverlay = h("div.sim-overlay.sim-overlay--dim");
      this.root.append(this.wallpaper, this.windowsLayer, this.taskbar, this.nightOverlay, this.dimOverlay);
      mount.append(this.root);
      this.mount = mount;

      this.ui = makeUi(this);
      this._buildTaskbar();
      for (const def of LW.apps) this._createApp(def);
      this.applySettings();
      this._initialSnapshot = this.snapshot();
      this._fit();
      if ("ResizeObserver" in window) new ResizeObserver(() => this._fit()).observe(mount);
      window.addEventListener("resize", () => this._fit());
    }

    /* ---------- geometry ---------- */
    get scale() {
      return this.root.getBoundingClientRect().width / SCREEN_W || 1;
    }
    _fit() {
      const w = this.mount.clientWidth;
      if (!w) return;
      const s = w / SCREEN_W;
      this.root.style.transform = `scale(${s})`;
      this.mount.style.height = `${SCREEN_H * s}px`;
    }
    /** Logical screen point -> viewport (client) point. */
    toClient(x, y) {
      const r = this.root.getBoundingClientRect();
      const s = r.width / SCREEN_W;
      return { x: r.left + x * s, y: r.top + y * s };
    }
    /** Viewport point -> logical screen point. */
    toLogical(cx, cy) {
      const r = this.root.getBoundingClientRect();
      const s = r.width / SCREEN_W;
      return { x: (cx - r.left) / s, y: (cy - r.top) / s };
    }
    /** Centre of an element in logical screen coordinates. */
    centerOf(el) {
      const r = el.getBoundingClientRect();
      return this.toLogical(r.left + r.width / 2, r.top + r.height / 2);
    }
    rectOf(el) {
      const r = el.getBoundingClientRect();
      const a = this.toLogical(r.left, r.top);
      const b = this.toLogical(r.right, r.bottom);
      return { x: a.x, y: a.y, w: b.x - a.x, h: b.y - a.y };
    }
    target(name) {
      return LW.util.byTarget(this.root, name);
    }

    /* ---------- taskbar ---------- */
    _buildTaskbar() {
      const start = h("div.sim-start", { title: "SimOS" }, h("span.sim-start__logo", { html: '<svg viewBox="0 0 20 20"><rect x="2" y="2" width="7" height="7" rx="1.5"/><rect x="11" y="2" width="7" height="7" rx="1.5"/><rect x="2" y="11" width="7" height="7" rx="1.5"/><rect x="11" y="11" width="7" height="7" rx="1.5"/></svg>' }));
      this.taskApps = h("div.sim-taskapps");
      this.tray = h("div.sim-tray");
      this.trayWifi = h("span.sim-tray__icon", { title: "Wi-Fi" });
      this.trayBt = h("span.sim-tray__icon", { title: "Bluetooth" });
      this.trayVol = h("span.sim-tray__icon", { title: "Volume" });
      this.clock = h("span.sim-tray__clock");
      this.tray.append(this.trayWifi, this.trayBt, this.trayVol, this.clock);
      this.taskbar.append(start, this.taskApps, this.tray);
      const tick = () => {
        const d = new Date();
        this.clock.textContent = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      };
      tick();
      setInterval(tick, 20000);
    }

    /* ---------- apps & windows ---------- */
    _createApp(def) {
      const r = def.rect;
      const closeBtn = h("button.sim-win__close", { "aria-label": `Close ${def.title}`, dataset: { target: `win.${def.id}.close` }, html: '<svg viewBox="0 0 12 12"><path d="M2 2l8 8M10 2l-8 8"/></svg>' });
      const title = h("div.sim-win__title", { dataset: { target: `win.${def.id}.title` } },
        h("span.sim-win__icon", { html: def.icon }), h("span", {}, def.title));
      const titlebar = h("div.sim-win__bar", {}, title, closeBtn);
      const body = h("div.sim-win__body");
      const win = h("section.sim-win", { hidden: true, "aria-label": def.title, dataset: { app: def.id }, style: { left: `${r.x}px`, top: `${r.y}px`, width: `${r.w}px`, height: `${r.h}px` } }, titlebar, body);
      this.windowsLayer.append(win);

      const btn = h("button.sim-taskapp", { title: def.title, "aria-label": def.title, dataset: { target: `taskbar.${def.id}` }, html: def.icon + '<span class="sim-taskapp__dot"></span>' });
      this.taskApps.append(btn);

      const app = { def, win, body, btn, instance: null };
      this.apps.set(def.id, app);
      app.instance = def.create(this, body) || {};

      btn.addEventListener("click", () => {
        if (this.isOpen(def.id) && this.isFocused(def.id)) this.close(def.id);
        else this.open(def.id);
      });
      closeBtn.addEventListener("click", (e) => { e.stopPropagation(); this.close(def.id); });
      win.addEventListener("pointerdown", () => this.focus(def.id));
      this._makeDraggable(win, titlebar);
    }

    _makeDraggable(win, handle) {
      let start = null;
      handle.addEventListener("pointerdown", (e) => {
        if (e.target.closest("button")) return;
        const p = this.toLogical(e.clientX, e.clientY);
        start = { px: p.x, py: p.y, x: win.offsetLeft, y: win.offsetTop };
      });
      document.addEventListener("pointermove", (e) => {
        if (!start) return;
        const p = this.toLogical(e.clientX, e.clientY);
        const x = clamp(start.x + p.x - start.px, -win.offsetWidth + 80, SCREEN_W - 80);
        const y = clamp(start.y + p.y - start.py, 0, SCREEN_H - TASKBAR_H - 30);
        win.style.left = `${x}px`;
        win.style.top = `${y}px`;
      });
      document.addEventListener("pointerup", () => { start = null; });
    }

    isOpen(id) { const a = this.apps.get(id); return !!a && !a.win.hidden; }
    isFocused(id) { const a = this.apps.get(id); return !!a && a.win.classList.contains("is-focused"); }
    focusedApp() {
      for (const [id, a] of this.apps) if (!a.win.hidden && a.win.classList.contains("is-focused")) return id;
      return null;
    }
    app(id) { const a = this.apps.get(id); return a && a.instance; }

    open(id) {
      const a = this.apps.get(id);
      if (!a) return;
      const wasOpen = !a.win.hidden;
      a.win.hidden = false;
      a.btn.classList.add("is-open");
      this.focus(id);
      if (!wasOpen && a.instance.onOpen) a.instance.onOpen();
      this.emit("windows", { id, open: true });
    }
    close(id) {
      const a = this.apps.get(id);
      if (!a) return;
      a.win.hidden = true;
      a.win.classList.remove("is-focused");
      a.btn.classList.remove("is-open", "is-focused");
      this.emit("windows", { id, open: false });
    }
    focus(id) {
      for (const [other, a] of this.apps) {
        const on = other === id;
        a.win.classList.toggle("is-focused", on);
        a.btn.classList.toggle("is-focused", on && !a.win.hidden);
      }
      const a = this.apps.get(id);
      if (a) a.win.style.zIndex = String(++this.z);
    }

    /* ---------- settings ---------- */
    setSetting(key, value) {
      if (!(key in this.settings)) return;
      this.settings[key] = value;
      this.applySettings();
      this.emit("settings", { key, value });
    }
    applySettings() {
      const s = this.settings;
      this.root.classList.toggle("is-dark", s.dark);
      this.nightOverlay.style.opacity = s.nightlight ? "1" : "0";
      // brightness 10..100 -> dim overlay 0.63..0
      this.dimOverlay.style.opacity = String(((100 - s.brightness) / 100) * 0.7);
      this.trayWifi.innerHTML = s.wifi ? TRAY_ICONS.wifi : TRAY_ICONS.wifiOff;
      this.trayBt.innerHTML = s.bluetooth ? TRAY_ICONS.bt : TRAY_ICONS.btOff;
      this.trayVol.innerHTML = s.volume > 0 ? TRAY_ICONS.vol : TRAY_ICONS.mute;
      this.trayWifi.classList.toggle("is-off", !s.wifi);
      this.trayBt.classList.toggle("is-off", !s.bluetooth);
    }

    /* ---------- file system (shared by Files and Terminal) ---------- */
    get fs() {
      const os = this;
      return {
        resolve(path) {
          let node = os.fsRoot;
          for (const part of path) {
            const next = node.children && findChild(node, part);
            if (!next || next.node.type !== "dir") return null;
            node = next.node;
          }
          return node;
        },
        list(path) {
          const dir = this.resolve(path);
          if (!dir) return [];
          return Object.entries(dir.children)
            .map(([name, n]) => ({ name, type: n.type, size: n.size || 0 }))
            .sort((a, b) => (a.type === b.type ? a.name.localeCompare(b.name) : a.type === "dir" ? -1 : 1));
        },
        exists(path, name) {
          const dir = this.resolve(path);
          return !!(dir && findChild(dir, name));
        },
        realName(path, name) {
          const dir = this.resolve(path);
          const hit = dir && findChild(dir, name);
          return hit ? hit.name : null;
        },
        mkdir(path, name) {
          const dir = this.resolve(path);
          name = (name || "").trim();
          if (!dir) return { ok: false, error: "The folder path does not exist." };
          if (!name || /[\\/:*?"<>|]/.test(name)) return { ok: false, error: "That folder name is not allowed." };
          if (findChild(dir, name)) return { ok: false, error: `A file or folder called "${name}" already exists.` };
          dir.children[name] = { type: "dir", children: {} };
          os.emit("fs", { op: "mkdir", path, name });
          return { ok: true };
        },
        remove(path, name) {
          const dir = this.resolve(path);
          const hit = dir && findChild(dir, name);
          if (!hit) return { ok: false, error: `Could not find "${name}".` };
          delete dir.children[hit.name];
          os.emit("fs", { op: "remove", path, name: hit.name });
          return { ok: true };
        },
        label(path) {
          return [os.homeLabel, ...path].join("\\");
        },
      };
    }

    /* ---------- snapshot / restore (for "Now you try") ---------- */
    snapshot() {
      const apps = {};
      const windows = {};
      for (const [id, a] of this.apps) {
        apps[id] = a.instance.snapshot ? a.instance.snapshot() : null;
        windows[id] = { open: !a.win.hidden, left: a.win.style.left, top: a.win.style.top };
      }
      return deepClone({ settings: this.settings, fs: this.fsRoot, apps, windows, focused: this.focusedApp() });
    }
    restore(snap) {
      snap = deepClone(snap);
      this.settings = snap.settings;
      this.fsRoot = snap.fs;
      this.applySettings();
      for (const [id, a] of this.apps) {
        if (a.instance.restore && snap.apps[id] != null) a.instance.restore(snap.apps[id]);
        const w = snap.windows[id];
        a.win.style.left = w.left;
        a.win.style.top = w.top;
        if (w.open) this.open(id); else this.close(id);
      }
      if (snap.focused) this.focus(snap.focused);
      this.emit("settings", {});
      this.emit("fs", { op: "restore" });
    }
    reset() { this.restore(this._initialSnapshot); }
  }

  function findChild(dir, name) {
    const want = String(name).toLowerCase();
    for (const [n, node] of Object.entries(dir.children)) {
      if (n.toLowerCase() === want) return { name: n, node };
    }
    return null;
  }

  /* ---------- shared widgets used by the apps ---------- */
  function makeUi(os) {
    function toggle({ target, label, checked, onChange }) {
      const btn = h("button.sim-toggle", { role: "switch", "aria-checked": String(!!checked), "aria-label": label, dataset: { target } },
        h("span.sim-toggle__knob"));
      btn.addEventListener("click", () => {
        const next = btn.getAttribute("aria-checked") !== "true";
        btn.setAttribute("aria-checked", String(next));
        onChange && onChange(next);
      });
      btn.setChecked = (v) => btn.setAttribute("aria-checked", String(!!v));
      return btn;
    }

    /** A slider driven by pointer position, so a synthetic drag from the agent works exactly like a real one. */
    function slider({ target, label, min, max, step = 1, value, onChange, format }) {
      const fill = h("span.sim-slider__fill");
      const thumb = h("span.sim-slider__thumb", { dataset: { target: `${target}.thumb` } });
      const track = h("span.sim-slider__track", {}, fill, thumb);
      const out = h("output.sim-slider__out");
      const root = h("div.sim-slider", { role: "slider", tabindex: "0", "aria-label": label, "aria-valuemin": min, "aria-valuemax": max, dataset: { target } }, track, out);
      let v = value;
      let dragging = false;
      function set(nv, fire = true) {
        nv = clamp(Math.round(nv / step) * step, min, max);
        nv = Number(nv.toFixed(4));
        const changed = nv !== v;
        v = nv;
        const pct = ((v - min) / (max - min)) * 100;
        fill.style.width = `${pct}%`;
        thumb.style.left = `${pct}%`;
        out.textContent = format ? format(v) : String(v);
        root.setAttribute("aria-valuenow", String(v));
        if (fire && changed && onChange) onChange(v);
      }
      function fromClientX(cx) {
        const r = track.getBoundingClientRect();
        return min + clamp((cx - r.left) / r.width, 0, 1) * (max - min);
      }
      root.addEventListener("pointerdown", (e) => { dragging = true; set(fromClientX(e.clientX)); e.preventDefault(); });
      document.addEventListener("pointermove", (e) => { if (dragging) set(fromClientX(e.clientX)); });
      document.addEventListener("pointerup", () => { dragging = false; });
      root.addEventListener("keydown", (e) => {
        if (e.key === "ArrowRight" || e.key === "ArrowUp") { set(v + step * 5); e.preventDefault(); }
        if (e.key === "ArrowLeft" || e.key === "ArrowDown") { set(v - step * 5); e.preventDefault(); }
      });
      set(value, false);
      root.slider = {
        get value() { return v; },
        set,
        /** Logical screen point where the thumb sits for a given value. */
        pointFor(val) {
          const r = os.rectOf(track);
          const f = (clamp(val, min, max) - min) / (max - min);
          return { x: r.x + f * r.w, y: r.y + r.h / 2 };
        },
      };
      return root;
    }
    return { toggle, slider };
  }

  LW.SimOS = SimOS;
  LW.SimOS.SCREEN_W = SCREEN_W;
  LW.SimOS.SCREEN_H = SCREEN_H;
  LW.SimOS.TASKBAR_H = TASKBAR_H;
})();
