/* PhotoLab: a small photo editor laid out like Photoshop
   (menu bar, tool column, canvas, History panel). */
(function () {
  "use strict";
  const LW = window.LW;
  const { h, deepClone } = LW.util;

  const ICON = '<svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 16l5-5 4 4 3-3 6 6"/><circle cx="15.5" cy="8.5" r="1.6"/></svg>';

  const MENUS = [
    { id: "file", label: "File", items: [{ id: "revert", label: "Revert to original" }] },
    { id: "edit", label: "Edit", items: [{ id: "undo", label: "Undo", keys: "Ctrl+Z" }] },
    { id: "image", label: "Image", items: [
      { id: "brightness", label: "Brightness/Contrast…" },
      { id: "grayscale", label: "Black & White" },
      { id: "rotate", label: "Rotate 90° clockwise" },
      { id: "flip", label: "Flip horizontal" },
    ] },
    { id: "filter", label: "Filter", items: [
      { id: "blur", label: "Blur" },
      { id: "sepia", label: "Vintage (sepia)" },
    ] },
  ];

  const TOOLS = [
    { id: "move", label: "Move", svg: '<path d="M12 3v18M3 12h18M12 3l-3 3M12 3l3 3M12 21l-3-3M12 21l3-3M3 12l3-3M3 12l3 3M21 12l-3-3M21 12l-3 3"/>' },
    { id: "crop", label: "Crop", svg: '<path d="M6 2v16h16M2 6h16v16"/>' },
    { id: "brush", label: "Brush", svg: '<path d="M14 4l6 6-8 8H6v-6z"/>' },
    { id: "text", label: "Text", svg: '<path d="M5 5h14M12 5v14"/>' },
  ];

  const ORIGINAL = { b: 0, c: 0, gray: false, sepia: false, blur: 0, rot: 0, flip: false };

  function paintLandscape(canvas) {
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    const sky = ctx.createLinearGradient(0, 0, 0, H * 0.62);
    sky.addColorStop(0, "#3c6fb4");
    sky.addColorStop(0.55, "#f2a35e");
    sky.addColorStop(1, "#f7d48b");
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = "#fff3c4";
    ctx.beginPath(); ctx.arc(W * 0.7, H * 0.42, 26, 0, Math.PI * 2); ctx.fill();
    const ridge = (base, amp, color, seed) => {
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.moveTo(0, H);
      for (let x = 0; x <= W; x += 8) {
        const y = base - amp * (0.55 * Math.sin(x / 61 + seed) + 0.3 * Math.sin(x / 23 + seed * 2) + 0.15 * Math.sin(x / 9 + seed));
        ctx.lineTo(x, y);
      }
      ctx.lineTo(W, H); ctx.closePath(); ctx.fill();
    };
    ridge(H * 0.52, 42, "#6a5b8c", 1.3);
    ridge(H * 0.6, 30, "#3f4f6e", 4.1);
    const lake = ctx.createLinearGradient(0, H * 0.64, 0, H);
    lake.addColorStop(0, "#e9b67a");
    lake.addColorStop(1, "#2e4a6b");
    ctx.fillStyle = lake;
    ctx.fillRect(0, H * 0.64, W, H * 0.36);
    ctx.fillStyle = "rgba(255,243,196,.55)";
    for (let i = 0; i < 7; i++) ctx.fillRect(W * 0.7 - 30 + i * 3, H * 0.68 + i * 9, 60 - i * 6, 2);
    ctx.fillStyle = "#1f3326";
    ctx.fillRect(0, H * 0.86, W, H * 0.14);
    for (let i = 0; i < 9; i++) {
      const x = 14 + i * 22 + (i % 3) * 5, hgt = 40 + (i % 4) * 12, y = H * 0.9;
      ctx.beginPath(); ctx.moveTo(x, y - hgt); ctx.lineTo(x - 11, y); ctx.lineTo(x + 11, y); ctx.closePath(); ctx.fill();
    }
  }

  LW.apps.push({
    id: "photolab",
    title: "PhotoLab — landscape.jpg",
    icon: ICON,
    rect: { x: 110, y: 26, w: 680, h: 452 },
    create(os, body) {
      const win = body.parentElement;
      body.classList.add("app-photo");
      let state = deepClone(ORIGINAL);
      let history = [{ label: "Open", state: deepClone(ORIGINAL) }];
      let openMenu = null;
      let tool = "move";

      /* menu bar */
      const menubar = h("div.ph-menubar", { role: "menubar" });
      const dropdown = h("div.ph-dropdown", { role: "menu", hidden: true });
      for (const m of MENUS) {
        const b = h("button.ph-menu", { role: "menuitem", dataset: { target: `photolab.menu.${m.id}` } }, m.label);
        b.addEventListener("click", (e) => { e.stopPropagation(); toggleMenu(m, b); });
        menubar.append(b);
      }
      function toggleMenu(m, b) {
        if (openMenu === m.id) return closeMenu();
        openMenu = m.id;
        dropdown.replaceChildren(...m.items.map((it) => {
          const item = h("button.ph-item", { role: "menuitem", dataset: { target: `photolab.item.${it.id}` } },
            h("span", {}, it.label), it.keys ? h("kbd.ph-keys", {}, it.keys) : null);
          item.addEventListener("click", (e) => { e.stopPropagation(); closeMenu(); command(it.id); });
          return item;
        }));
        dropdown.style.left = `${b.offsetLeft}px`;
        dropdown.hidden = false;
        menubar.querySelectorAll(".ph-menu").forEach((x) => x.classList.toggle("is-open", x === b));
      }
      function closeMenu() {
        openMenu = null;
        dropdown.hidden = true;
        menubar.querySelectorAll(".ph-menu").forEach((x) => x.classList.remove("is-open"));
      }
      body.addEventListener("click", () => closeMenu());

      /* tools */
      const tools = h("div.ph-tools", { "aria-label": "Tools" });
      for (const t of TOOLS) {
        const b = h("button.ph-tool", { title: t.label, "aria-label": t.label, dataset: { target: `photolab.tool.${t.id}` }, html: `<svg viewBox="0 0 24 24">${t.svg}</svg>` });
        b.addEventListener("click", () => { tool = t.id; renderTools(); });
        tools.append(b);
      }
      const renderTools = () => tools.querySelectorAll(".ph-tool").forEach((b, i) => b.classList.toggle("is-active", TOOLS[i].id === tool));

      /* canvas */
      const canvas = h("canvas.ph-canvas", { width: "480", height: "320", "aria-label": "Photo of a lake at sunset" });
      paintLandscape(canvas);
      const canvasWrap = h("div.ph-stage", {}, h("div.ph-frame", {}, canvas));

      /* history */
      const histList = h("ol.ph-history__list");
      const historyPanel = h("aside.ph-history", {}, h("div.ph-panel-title", {}, "History"), histList);

      const status = h("div.ph-status", {}, "landscape.jpg · 1200 × 800 px · 100%");

      /* brightness/contrast dialog */
      const dlgB = os.ui.slider({ target: "photolab.dialog.brightness", label: "Brightness", min: -100, max: 100, value: 0, onChange: preview });
      const dlgC = os.ui.slider({ target: "photolab.dialog.contrast", label: "Contrast", min: -100, max: 100, value: 0, onChange: preview });
      const okBtn = h("button.ph-btn.ph-btn--primary", { dataset: { target: "photolab.dialog.ok" } }, "OK");
      const cancelBtn = h("button.ph-btn", { dataset: { target: "photolab.dialog.cancel" } }, "Cancel");
      const dialog = h("div.ph-dialog", { hidden: true, role: "dialog", "aria-label": "Brightness/Contrast" },
        h("div.ph-dialog__panel", {},
          h("div.ph-dialog__title", {}, "Brightness/Contrast"),
          h("label.ph-field", {}, h("span", {}, "Brightness"), dlgB),
          h("label.ph-field", {}, h("span", {}, "Contrast"), dlgC),
          h("div.ph-dialog__actions", {}, cancelBtn, okBtn)));
      okBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        dialog.hidden = true;
        commit("Brightness/Contrast", { ...state, b: dlgB.slider.value, c: dlgC.slider.value });
      });
      cancelBtn.addEventListener("click", (e) => { e.stopPropagation(); dialog.hidden = true; render(); });
      function preview() { render({ ...state, b: dlgB.slider.value, c: dlgC.slider.value }); }

      const main = h("div.ph-main", {}, tools, canvasWrap, historyPanel);
      body.append(menubar, dropdown, main, status, dialog);

      function command(id) {
        switch (id) {
          case "brightness":
            dlgB.slider.set(state.b, false);
            dlgC.slider.set(state.c, false);
            dialog.hidden = false;
            break;
          case "grayscale": commit("Black & White", { ...state, gray: true }); break;
          case "sepia": commit("Vintage", { ...state, sepia: true }); break;
          case "blur": commit("Blur", { ...state, blur: Math.min(6, state.blur + 2) }); break;
          case "rotate": commit("Rotate 90°", { ...state, rot: (state.rot + 90) % 360 }); break;
          case "flip": commit("Flip horizontal", { ...state, flip: !state.flip }); break;
          case "undo": undo(); break;
          case "revert": commit("Revert", deepClone(ORIGINAL)); break;
        }
      }
      function commit(label, next) {
        state = next;
        history.push({ label, state: deepClone(next) });
        render();
        os.emit("photolab", { label, state });
      }
      function undo() {
        if (history.length <= 1) return;
        history.pop();
        state = deepClone(history[history.length - 1].state);
        render();
        os.emit("photolab", { label: "Undo", state });
      }
      win.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); undo(); }
        if (e.key === "Escape") { closeMenu(); dialog.hidden = true; render(); }
      });

      function render(view = state) {
        const f = [
          `brightness(${(1 + view.b / 100).toFixed(2)})`,
          `contrast(${(1 + view.c / 100).toFixed(2)})`,
          view.gray ? "grayscale(1)" : "",
          view.sepia ? "sepia(.85)" : "",
          view.blur ? `blur(${view.blur}px)` : "",
        ].filter(Boolean).join(" ");
        canvas.style.filter = f;
        canvas.style.transform = `rotate(${view.rot}deg) scaleX(${view.flip ? -1 : 1})`;
        histList.replaceChildren(...history.map((it, i) =>
          h("li.ph-history__item", { class: i === history.length - 1 ? "ph-history__item is-current" : "ph-history__item" }, it.label)));
      }
      renderTools();
      render();

      return {
        get state() { return deepClone(state); },
        get historyLabels() { return history.map((x) => x.label); },
        get dialogOpen() { return !dialog.hidden; },
        snapshot: () => ({ state, history }),
        restore(s) { state = deepClone(s.state); history = deepClone(s.history); closeMenu(); dialog.hidden = true; render(); },
      };
    },
  });
})();
