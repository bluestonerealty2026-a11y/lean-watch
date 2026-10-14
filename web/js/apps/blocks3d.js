/* Blocks3D: a tiny 3D editor laid out like Blender
   (viewport on the left, Object properties on the right). */
(function () {
  "use strict";
  const LW = window.LW;
  const { h, deepClone } = LW.util;

  const ICON = '<svg viewBox="0 0 24 24"><path d="M12 2.8l8 4.6v9.2l-8 4.6-8-4.6V7.4z"/><path d="M4 7.4l8 4.6 8-4.6M12 12v9.2"/></svg>';

  const COLORS = {
    gray: "#9aa3ad",
    red: "#d9534f",
    blue: "#3d7be0",
    green: "#3ca66b",
    orange: "#ee8a2c",
    purple: "#8a5cd1",
  };
  const START = { rx: 0, ry: 0, rz: 0, scale: 1, color: "gray", wire: false };

  const VERTS = [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]];
  const FACES = [[0, 1, 2, 3], [5, 4, 7, 6], [4, 0, 3, 7], [1, 5, 6, 2], [3, 2, 6, 7], [4, 5, 1, 0]];
  const EDGES = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
  const rad = (d) => (d * Math.PI) / 180;

  function rotX(v, a) { const c = Math.cos(a), s = Math.sin(a); return [v[0], v[1] * c - v[2] * s, v[1] * s + v[2] * c]; }
  function rotY(v, a) { const c = Math.cos(a), s = Math.sin(a); return [v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c]; }
  function rotZ(v, a) { const c = Math.cos(a), s = Math.sin(a); return [v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2]]; }
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
  function shade(hex, k) {
    const n = parseInt(hex.slice(1), 16);
    const r = Math.round(((n >> 16) & 255) * k), g = Math.round(((n >> 8) & 255) * k), b = Math.round((n & 255) * k);
    return `rgb(${Math.min(255, r)},${Math.min(255, g)},${Math.min(255, b)})`;
  }

  LW.apps.push({
    id: "blocks3d",
    title: "Blocks3D — scene.b3d",
    icon: ICON,
    rect: { x: 170, y: 64, w: 660, h: 430 },
    create(os, body) {
      body.classList.add("app-3d");
      let state = deepClone(START);

      const solidBtn = h("button.b3-mode", { dataset: { target: "cube.shading.solid" } }, "Solid");
      const wireBtn = h("button.b3-mode", { dataset: { target: "cube.shading.wire" } }, "Wireframe");
      solidBtn.addEventListener("click", () => set({ wire: false }));
      wireBtn.addEventListener("click", () => set({ wire: true }));
      const header = h("div.b3-header", {}, h("span.b3-pill", {}, "Object Mode"), h("span.b3-spacer"), h("span.b3-label", {}, "Viewport shading"), solidBtn, wireBtn);

      const canvas = h("canvas.b3-canvas", { width: "840", height: "660", "aria-label": "3D viewport showing a cube" });
      const viewport = h("div.b3-viewport", {}, canvas, h("div.b3-hint", {}, "Tip: in Blender, press R then Z then a number to rotate"));

      const rotInputs = {};
      const rotRow = ["x", "y", "z"].map((axis) => {
        const input = h("input.b3-num", { type: "text", inputmode: "decimal", "aria-label": `Rotation ${axis.toUpperCase()}`, dataset: { target: `cube.rot.${axis}` } });
        const apply = () => {
          const v = parseFloat(input.value);
          if (Number.isFinite(v)) set({ [`r${axis}`]: ((v % 360) + 360) % 360 });
          else input.value = state[`r${axis}`];
        };
        input.addEventListener("change", apply);
        input.addEventListener("keydown", (e) => { if (e.key === "Enter") { apply(); input.blur(); } });
        rotInputs[axis] = input;
        return h("label.b3-field", {}, h("span.b3-axis", { class: `b3-axis b3-axis--${axis}` }, axis.toUpperCase()), input, h("span.b3-unit", {}, "°"));
      });
      const scaleSlider = os.ui.slider({ target: "cube.scale", label: "Scale", min: 0.5, max: 2, step: 0.1, value: 1, format: (v) => v.toFixed(1), onChange: (v) => set({ scale: v }, false) });
      const swatches = h("div.b3-swatches", { role: "group", "aria-label": "Material colour" },
        ...Object.entries(COLORS).map(([name, hex]) => {
          const b = h("button.b3-swatch", { title: name, "aria-label": `${name} material`, style: { background: hex }, dataset: { target: `cube.color.${name}` } });
          b.addEventListener("click", () => set({ color: name }));
          return b;
        }));
      const props = h("aside.b3-props", {},
        h("div.b3-props__title", {}, "Cube"),
        h("div.b3-group", {}, h("div.b3-group__label", {}, "Rotation"), ...rotRow),
        h("div.b3-group", {}, h("div.b3-group__label", {}, "Scale"), scaleSlider),
        h("div.b3-group", {}, h("div.b3-group__label", {}, "Material"), swatches));

      body.append(header, h("div.b3-main", {}, viewport, props));

      function set(patch, syncSlider = true) {
        state = { ...state, ...patch };
        render(syncSlider);
        os.emit("blocks3d", { state });
      }

      function render(syncSlider = true) {
        for (const a of ["x", "y", "z"]) if (document.activeElement !== rotInputs[a]) rotInputs[a].value = String(Math.round(state[`r${a}`]));
        if (syncSlider) scaleSlider.slider.set(state.scale, false);
        solidBtn.classList.toggle("is-active", !state.wire);
        wireBtn.classList.toggle("is-active", state.wire);
        body.querySelectorAll(".b3-swatch").forEach((b) => b.classList.toggle("is-active", b.title === state.color));
        draw();
      }

      function draw() {
        const ctx = canvas.getContext("2d");
        const W = canvas.width, H = canvas.height;
        ctx.fillStyle = "#2b2f36";
        ctx.fillRect(0, 0, W, H);
        const camYaw = rad(-35), camPitch = rad(-24), dist = 7.5, f = 760; // camera above the cube, looking down
        const toView = (v) => { let p = rotY(v, camYaw); p = rotX(p, camPitch); return [p[0], p[1], p[2] + dist]; };
        const project = (p) => [W / 2 + (f * p[0]) / p[2], H / 2 + 40 - (f * p[1]) / p[2]];

        // ground grid
        ctx.lineWidth = 1.5;
        for (let i = -5; i <= 5; i++) {
          for (const [a, b] of [[[i, -1.6, -5], [i, -1.6, 5]], [[-5, -1.6, i], [5, -1.6, i]]]) {
            const pa = toView(a), pb = toView(b);
            if (pa[2] < 0.5 || pb[2] < 0.5) continue;
            const A = project(pa), B = project(pb);
            ctx.strokeStyle = i === 0 ? "rgba(255,255,255,.28)" : "rgba(255,255,255,.08)";
            ctx.beginPath(); ctx.moveTo(A[0], A[1]); ctx.lineTo(B[0], B[1]); ctx.stroke();
          }
        }

        // Blender is Z-up: its Z rotation spins around the vertical axis (our math Y),
        // and its Y axis points into the screen (our math Z).
        const verts = VERTS.map((v) => {
          let p = v.map((c) => c * state.scale);
          p = rotX(p, rad(state.rx)); p = rotZ(p, rad(state.ry)); p = rotY(p, rad(state.rz));
          return toView(p);
        });
        const pts = verts.map(project);
        const light = norm([-0.45, 0.8, -0.6]); // upper-left-front, so top, left and right faces read apart
        const base = COLORS[state.color];

        if (state.wire) {
          ctx.strokeStyle = "#f4b042";
          ctx.lineWidth = 3;
          for (const [a, b] of EDGES) { ctx.beginPath(); ctx.moveTo(pts[a][0], pts[a][1]); ctx.lineTo(pts[b][0], pts[b][1]); ctx.stroke(); }
        } else {
          const faces = FACES.map((idx) => {
            const [a, b, c] = idx.map((i) => verts[i]);
            const n = norm(cross(sub(c, a), sub(b, a))); // outward normal for this winding
            const depth = idx.reduce((s, i) => s + verts[i][2], 0) / 4;
            return { idx, n, depth };
          }).sort((p, q) => q.depth - p.depth);
          for (const face of faces) {
            if (face.n[2] > 0) continue; // facing away from the camera
            const k = 0.4 + 0.65 * Math.max(0, dot(face.n, light));
            ctx.fillStyle = shade(base, k);
            ctx.strokeStyle = "rgba(0,0,0,.35)";
            ctx.lineWidth = 2;
            ctx.beginPath();
            face.idx.forEach((i, j) => (j ? ctx.lineTo(pts[i][0], pts[i][1]) : ctx.moveTo(pts[i][0], pts[i][1])));
            ctx.closePath(); ctx.fill(); ctx.stroke();
          }
          ctx.strokeStyle = "#f4b042";
          ctx.lineWidth = 2;
          ctx.globalAlpha = 0.9;
          // selection outline like Blender's orange highlight
          for (const face of faces) {
            if (face.n[2] > 0) continue;
            ctx.beginPath();
            face.idx.forEach((i, j) => (j ? ctx.lineTo(pts[i][0], pts[i][1]) : ctx.moveTo(pts[i][0], pts[i][1])));
            ctx.closePath(); ctx.stroke();
          }
          ctx.globalAlpha = 1;
        }

        // axis gizmo
        const o = [70, H - 70];
        const axes = [[[1, 0, 0], "#e5534b", "X"], [[0, 0, 1], "#6cc644", "Y"], [[0, 1, 0], "#4a90e2", "Z"]];
        ctx.font = "bold 22px sans-serif";
        for (const [v, col, label] of axes) {
          let p = rotY(v, camYaw); p = rotX(p, camPitch);
          const e = [o[0] + p[0] * 44, o[1] - p[1] * 44];
          ctx.strokeStyle = col; ctx.lineWidth = 4;
          ctx.beginPath(); ctx.moveTo(o[0], o[1]); ctx.lineTo(e[0], e[1]); ctx.stroke();
          ctx.fillStyle = col; ctx.fillText(label, e[0] + 4, e[1] + 4);
        }
      }

      render();
      return {
        get state() { return deepClone(state); },
        snapshot: () => state,
        restore(s) { state = deepClone(s); render(); },
      };
    },
  });
})();
