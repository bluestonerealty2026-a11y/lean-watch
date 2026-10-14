/* Files: a file explorer for the practice computer's home folder. */
(function () {
  "use strict";
  const LW = window.LW;
  const { h, deepClone } = LW.util;

  const ICON = '<svg viewBox="0 0 24 24"><path d="M3 6.5A1.5 1.5 0 0 1 4.5 5H10l2 2.2h7.5A1.5 1.5 0 0 1 21 8.7v9.8a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 18.5z"/></svg>';
  const FOLDER = '<svg viewBox="0 0 48 40" class="fx-ico fx-ico--dir"><path d="M2 6a3 3 0 0 1 3-3h13l4 4h21a3 3 0 0 1 3 3v25a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3z"/></svg>';
  const FILE = '<svg viewBox="0 0 40 48" class="fx-ico fx-ico--file"><path d="M4 3h22l10 10v32H4z"/><path d="M26 3v10h10" class="fx-fold"/></svg>';

  LW.apps.push({
    id: "files",
    title: "Files",
    icon: ICON,
    rect: { x: 250, y: 84, w: 560, h: 370 },
    create(os, body) {
      body.classList.add("app-files");
      let path = [];
      let selected = null;
      let editing = false;
      let confirming = false;

      const upBtn = h("button.fx-btn", { dataset: { target: "files.up" }, "aria-label": "Up one folder", html: '<svg viewBox="0 0 16 16"><path d="M8 13V3M3.5 7.5L8 3l4.5 4.5"/></svg>' });
      const newBtn = h("button.fx-btn.fx-btn--label", { dataset: { target: "files.newfolder" } }, "New folder");
      const delBtn = h("button.fx-btn.fx-btn--label", { dataset: { target: "files.delete" } }, "Delete");
      const crumb = h("div.fx-path");
      const grid = h("div.fx-grid", { role: "listbox", "aria-label": "Files and folders" });
      const status = h("div.fx-status");
      const confirmBar = h("div.fx-confirm", { hidden: true, role: "alertdialog" });
      body.append(h("div.fx-toolbar", {}, upBtn, newBtn, delBtn, crumb), grid, confirmBar, status);

      upBtn.addEventListener("click", () => { if (path.length) { path = path.slice(0, -1); selected = null; render(); } });
      newBtn.addEventListener("click", () => { editing = true; confirming = false; render(); });
      delBtn.addEventListener("click", () => { if (selected) { confirming = true; render(); } });

      function startEditor() {
        let base = "New folder", n = 2, name = base;
        while (os.fs.exists(path, name)) name = `${base} (${n++})`;
        const input = h("input.fx-rename", { type: "text", value: name, "aria-label": "New folder name", dataset: { target: "files.rename.input" } });
        const tile = h("div.fx-item.is-editing", {}, h("span", { html: FOLDER }), input);
        let done = false;
        const commit = (keep) => {
          if (done) return;
          done = true;
          editing = false;
          if (keep) {
            const r = os.fs.mkdir(path, input.value);
            if (r.ok) selected = input.value.trim();
            else flash(r.error);
          }
          render();
        };
        input.addEventListener("keydown", (e) => {
          if (e.key === "Enter") commit(true);
          if (e.key === "Escape") commit(false);
        });
        input.addEventListener("blur", () => setTimeout(() => commit(true), 0));
        setTimeout(() => { input.focus({ preventScroll: true }); input.select(); }, 0);
        return tile;
      }

      let flashMsg = "";
      function flash(msg) { flashMsg = msg; setTimeout(() => { flashMsg = ""; render(); }, 3500); }

      function render() {
        const items = os.fs.list(path);
        crumb.textContent = os.fs.label(path);
        upBtn.disabled = !path.length;
        delBtn.disabled = !selected;
        const tiles = items.map((it) => {
          const tile = h("button.fx-item", { role: "option", "aria-selected": String(selected === it.name), dataset: { target: `files.item:${it.name}` } },
            h("span", { html: it.type === "dir" ? FOLDER : FILE }), h("span.fx-name", {}, it.name));
          tile.addEventListener("click", () => { selected = it.name; confirming = false; render(); });
          tile.addEventListener("dblclick", () => {
            if (it.type === "dir") { path = [...path, it.name]; selected = null; render(); }
          });
          return tile;
        });
        if (editing) tiles.push(startEditor());
        grid.replaceChildren(...tiles);
        if (!items.length && !editing) grid.append(h("p.fx-empty", {}, "This folder is empty."));

        confirmBar.hidden = !confirming;
        if (confirming) {
          const cancel = h("button.fx-btn.fx-btn--label", { dataset: { target: "files.confirm.cancel" } }, "Cancel");
          const ok = h("button.fx-btn.fx-btn--label.fx-btn--danger", { dataset: { target: "files.confirm.delete" } }, "Delete");
          cancel.addEventListener("click", () => { confirming = false; render(); });
          ok.addEventListener("click", () => {
            const r = os.fs.remove(path, selected);
            if (!r.ok) flash(r.error);
            confirming = false;
            selected = null;
            render();
          });
          confirmBar.replaceChildren(h("span", {}, `Delete "${selected}"? There is no recycle bin on this computer.`), cancel, ok);
        }
        status.textContent = flashMsg || `${items.length} item${items.length === 1 ? "" : "s"}${selected ? ` · "${selected}" selected` : ""}`;
      }

      os.on("fs", () => {
        if (path.length && !os.fs.resolve(path)) path = [];
        if (selected && !os.fs.exists(path, selected)) selected = null;
        if (!editing) render();
      });
      render();

      return {
        get path() { return path.slice(); },
        get selected() { return selected; },
        snapshot: () => ({ path, selected: null }),
        restore(s) { path = deepClone(s.path); selected = null; editing = false; confirming = false; render(); },
      };
    },
  });
})();
