/* Wires the page together: tabs, command box, permission, step-by-step notes,
   and the switch between the practice screen and your real computer. */
(function () {
  "use strict";
  const LW = window.LW;
  const { h, store, clamp } = LW.util;
  const $ = (id) => document.getElementById(id);

  const os = new LW.SimOS($("screen-fit"));
  const cursor = new LW.VirtualCursor(os);
  const agent = new LW.Agent(os, cursor);
  const bridge = new LW.Bridge();
  const ctx = { os, cursor, agent, bridge };

  const state = {
    mode: "sim",                       // "sim" = practice screen, "device" = real computer
    permission: { sim: null, device: null },
    pending: null,                     // command waiting for permission
    lastRun: null,                     // { plan, before } for Watch again / Now you try
    deviceRunning: false,
    devicePaused: false,
  };

  /* ================= tabs ================= */
  const VIEWS = ["watch", "progress", "how"];
  function showView(name) {
    if (!VIEWS.includes(name)) name = "watch";
    for (const v of VIEWS) {
      $(`view-${v}`).hidden = v !== name;
      $(`tab-${v}`).setAttribute("aria-selected", String(v === name));
    }
    if (name === "progress") LW.progress.renderChecks(ctx);
  }
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
    const v = t.dataset.view;
    try { history.replaceState(null, "", `#${v}`); } catch { /* sandboxed */ }
    showView(v);
  }));
  window.addEventListener("hashchange", () => showView(location.hash.slice(1)));
  showView(location.hash.slice(1));

  /* ================= readout, speed, speech ================= */
  cursor.onMove(({ x, y, ms }) => {
    $("ro-x").textContent = String(Math.round(x));
    $("ro-y").textContent = String(Math.round(y));
    $("ro-ms").textContent = String(ms);
  });
  document.querySelectorAll(".seg__btn").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(".seg__btn").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
    agent.setSpeed(Number(b.dataset.speed));
    store.set("lw-speed", b.dataset.speed);
  }));
  const savedSpeed = store.get("lw-speed", null);
  if (savedSpeed) { const b = document.querySelector(`.seg__btn[data-speed="${savedSpeed}"]`); if (b) b.click(); }
  $("opt-speak").addEventListener("change", (e) => { agent.speakEnabled = e.target.checked; });
  if (!("speechSynthesis" in window)) { $("opt-speak").disabled = true; $("opt-speak").closest("label").title = "This browser has no speech voice"; }

  /* ================= notes panel ================= */
  const notes = $("notes-body");
  let stepEls = [];
  const setNotes = (...nodes) => notes.replaceChildren(...nodes.flat().filter(Boolean));

  function setRunState(kind, label) {
    const el = $("run-state");
    el.dataset.state = kind;
    el.textContent = label;
  }
  function setCmdStatus(text) { $("cmd-status").textContent = text; }

  function exampleChip(text) {
    return h("button.chip-ex", { type: "button", onclick: () => submit(text) }, text);
  }

  function renderIdle() {
    stepEls = [];
    const groups = [
      ["Settings", ["Turn on dark mode", "Set brightness to 40", "Turn off Bluetooth"]],
      ["PhotoLab", ["Make my photo black and white", "Make the photo brighter"]],
      ["Blocks3D", ["Rotate the cube 45 degrees", "Make the cube red"]],
      ["Command Prompt", ["Check my IP address", "Create a folder called Homework using the terminal"]],
      ["Files", ["Create a new folder called Recipes"]],
    ];
    setNotes(
      h("p.notes__intro", {}, "Type what you want to do in the box at the bottom right. I move the pointer, click and type while you watch, and explain every step."),
      h("p.notes__sub", {}, "Then press Now you try and do it yourself with my help."),
      h("div.groups", {}, ...groups.map(([title, ex]) =>
        h("div.group", {}, h("div.group__title", {}, title), h("div.group__chips", {}, ...ex.map(exampleChip))))),
    );
    setRunState("idle", "Idle");
  }

  function pctOf(score) { return `${clamp(Math.round(score * 100), 0, 100)}% match`; }

  function renderPlan(plan, flat, mode) {
    stepEls = [];
    let n = 0;
    const items = plan.items.map((item) => {
      const why = h("details.why", {},
        h("summary", {}, "Why this?"),
        h("p.small.muted", {}, "I compared your words with example sentences for every skill. Closest matches:"),
        h("ol.why__list", {}, ...item.alternatives.map((a) =>
          h("li", {}, h("span.why__name", {}, a.title), h("span.why__bar", { "aria-hidden": "true" }, h("span", { style: { width: `${clamp(a.score * 100, 2, 100)}%` } })), h("span.why__score", {}, a.score.toFixed(2))))));
      const steps = h("ol.steps", {}, ...item.steps.map((s) => {
        n += 1;
        const li = h("li.step", { dataset: { status: "pending" } },
          h("span.step__n", {}, String(n)),
          h("span.step__text", {}, s.do === "verify" ? "Check that it worked." : s.say || ""));
        stepEls.push(li);
        return li;
      }));
      return h("div.plan__item", {},
        h("div.plan__intent", {}, h("strong", {}, item.summary), h("span.match", {}, pctOf(item.score))),
        why, steps,
        item.tip ? h("div.tip", {}, h("span.tip__label", {}, "On your own computer"), h("span", {}, item.tip)) : null);
    });
    setNotes(
      h("p.plan__asked", {}, mode === "practice" ? "Your turn. Follow the outlined spots for: " : "You asked: ", h("q", {}, plan.input)),
      ...items,
      h("div.confirm-slot", { id: "confirm-slot" }),
      h("div.done-slot", { id: "done-slot" }));
  }

  function renderUnknown(result, text) {
    stepEls = [];
    const sug = (result.suggestions || []).filter(Boolean);
    setNotes(
      h("p.plan__asked", {}, "You asked: ", h("q", {}, text)),
      h("div.callout.callout--warn", {}, h("strong", {}, result.message)),
      sug.length ? h("p.notes__sub", {}, "Closest things I can do here:") : null,
      sug.length ? h("div.group__chips", {}, ...sug.map((s) => exampleChip(s.examples[0].replace(/^./, (c) => c.toUpperCase())))) : null,
      h("p.small.muted", {}, "The practice screen knows a fixed set of skills. Switch to My computer and the local agent can try anything, because Claude reads the real screen."));
    setRunState("idle", "Not understood");
  }

  function confirmCard(text, onAnswer) {
    const slot = $("confirm-slot") || notes;
    const yes = h("button.btn.btn--go", { type: "button" }, "Allow");
    const no = h("button.btn.btn--ghost", { type: "button" }, "No, skip");
    const card = h("div.callout.callout--ask", { role: "alertdialog", "aria-label": "Confirm a risky step" },
      h("strong", {}, "Your OK, please"), h("p", {}, text), h("div.callout__actions", {}, no, yes));
    const answer = (allow) => { card.remove(); onAnswer(allow); };
    yes.addEventListener("click", () => answer(true));
    no.addEventListener("click", () => answer(false));
    slot.append(card);
    no.focus({ preventScroll: true });
    setRunState("ask", "Waiting for you");
    setCmdStatus("Waiting for your answer in the step list.");
  }

  /* ================= running on the practice screen ================= */
  function needsKeyboard(plan) {
    return plan.items.some((it) => it.steps.some((s) => s.do === "type" || s.do === "key"));
  }

  function runSim(text) {
    if (agent.running) return;
    const plan = LW.planner.plan(text, os);
    if (!plan.ok) { renderUnknown(plan, text); setCmdStatus("I didn't understand that one. Try an example."); return; }
    const perm = state.permission.sim;
    if (needsKeyboard(plan) && !perm.keyboard) {
      renderUnknown({ message: "This task needs typing, and keyboard control is switched off. Click the permission button at the top to allow it again.", suggestions: [] }, text);
      return;
    }
    $("screen-frame").scrollIntoView({ block: "nearest", behavior: "smooth" });
    agent.run(plan, "watch");
  }

  agent.on("start", ({ plan, flat, mode }) => {
    renderPlan(plan, flat, mode);
    setRunState("run", mode === "practice" ? "Your turn" : "Working");
    $("btn-pause").disabled = mode === "practice";
    $("btn-stop").disabled = false;
    $("btn-pause").textContent = "Pause";
    setBusy(true);
    setCmdStatus(mode === "practice" ? "Do each outlined step yourself. Press Esc to stop." : `Working on it. Step 1 of ${flat.length}.`);
  });
  agent.on("step", ({ index, status, note }) => {
    const li = stepEls[index];
    if (!li) return;
    li.dataset.status = status;
    if (note && status !== "active") {
      const t = li.querySelector(".step__text");
      if (status === "skipped" || status === "failed") t.textContent = note;
    }
    if (status === "active") {
      li.scrollIntoView({ block: "nearest", behavior: "smooth" });
      if (agent.mode === "watch") setCmdStatus(`Working on it. Step ${index + 1} of ${stepEls.length}.`);
    }
  });
  agent.on("confirm", ({ text }) => confirmCard(text, (allow) => agent.respond(allow)));
  agent.on("confirmed", () => setRunState("run", agent.mode === "practice" ? "Your turn" : "Working"));
  agent.on("paused", ({ reason }) => {
    setRunState("pause", "Paused");
    $("btn-pause").textContent = "Resume";
    setCmdStatus(reason || "Paused. Press Resume to continue.");
  });
  agent.on("resumed", () => {
    setRunState("run", "Working");
    $("btn-pause").textContent = "Pause";
    setCmdStatus("Carrying on.");
  });
  agent.on("done", (out) => {
    $("btn-pause").disabled = true;
    $("btn-stop").disabled = true;
    setBusy(false);
    const slot = $("done-slot") || notes;
    if (out.mode === "watch" && out.before) state.lastRun = { plan: lastPlanFrom(), before: out.before };
    const actions = [];
    if (state.lastRun) {
      actions.push(h("button.btn.btn--ghost", { type: "button", onclick: () => replay("watch") }, "Watch again"));
      actions.push(h("button.btn.btn--go", { type: "button", onclick: () => replay("practice") }, "Now you try"));
    }
    if (out.ok) {
      slot.replaceChildren(h("div.callout.callout--ok", {},
        h("strong", {}, out.mode === "practice" ? "You did it yourself!" : "Done."),
        h("p", {}, out.mode === "practice" ? "Every step was yours. Try another task, or watch it again." : "Want to try it with your own hands? I'll rewind the screen and point the way."),
        h("div.callout__actions", {}, ...actions)));
      setRunState("done", "Done");
      setCmdStatus(out.mode === "practice" ? "Nicely done." : "Done. Press Now you try to practise it.");
    } else {
      slot.replaceChildren(h("div.callout.callout--warn", {}, h("strong", {}, out.message), actions.length ? h("div.callout__actions", {}, ...actions) : null));
      setRunState(out.reason === "stopped" ? "stop" : "warn", out.reason === "stopped" ? "Stopped" : out.reason === "declined" ? "Skipped" : "Didn't finish");
      setCmdStatus(out.message);
    }
  });
  let currentPlan = null;
  agent.on("start", ({ plan, mode }) => { if (mode === "watch") currentPlan = plan; });
  const lastPlanFrom = () => currentPlan;

  function replay(mode) {
    if (agent.running || !state.lastRun) return;
    os.restore(state.lastRun.before);
    if (mode === "watch" && !state.permission.sim) return askPermission("sim", null);
    agent.run(state.lastRun.plan, mode);
  }

  // If the person clicks the practice screen while the agent works, hand control back.
  os.root.addEventListener("pointerdown", (e) => {
    if (e.isTrusted && agent.running && agent.mode === "watch") agent.pause("You clicked the screen, so I paused. Press Resume when you want me to continue.");
  }, true);

  /* ================= permission ================= */
  function askPermission(mode, command) {
    state.pending = command ? { mode, command } : null;
    const device = mode === "device";
    $("consent-lede").textContent = device
      ? "The local agent will move your real pointer, click and type on your computer while you watch. You can stop it at any time."
      : "I will move the pointer, click and type on the practice screen to show you how it is done. You can stop me at any time.";
    $("scope-device-row").hidden = !device;
    $("scope-device").checked = false;
    $("consent").dataset.mode = mode;
    updateConsentButton();
    $("consent").hidden = false;
    $("consent-yes").focus({ preventScroll: true });
  }
  function updateConsentButton() {
    const device = $("consent").dataset.mode === "device";
    $("consent-yes").disabled = !$("scope-mouse").checked || (device && !$("scope-device").checked);
  }
  ["scope-mouse", "scope-keyboard", "scope-device"].forEach((id) => $(id).addEventListener("change", updateConsentButton));
  $("consent-no").addEventListener("click", () => {
    $("consent").hidden = true;
    state.pending = null;
    setCmdStatus("Okay. Nothing will move until you allow it.");
  });
  $("consent-yes").addEventListener("click", () => {
    const mode = $("consent").dataset.mode;
    const scopes = { mouse: $("scope-mouse").checked, keyboard: $("scope-keyboard").checked };
    state.permission[mode] = scopes;
    if (mode === "device") bridge.send({ type: "grant", scopes });
    $("consent").hidden = true;
    updatePills();
    const p = state.pending;
    state.pending = null;
    if (p) submit(p.command);
  });
  $("pill-permission").addEventListener("click", () => {
    const m = state.mode;
    if (state.permission[m]) {
      state.permission[m] = null;
      if (m === "device") bridge.send({ type: "revoke" });
      if (agent.running) agent.stop();
      updatePills();
      setCmdStatus("Permission revoked. Nothing will move until you allow it again.");
    } else {
      askPermission(m, null);
    }
  });

  function updatePills() {
    const perm = state.permission[state.mode];
    const pill = $("pill-permission");
    pill.dataset.state = perm ? "on" : "off";
    pill.querySelector(".pill__text").textContent = perm
      ? `Allowed on ${state.mode === "device" ? "my computer" : "practice screen"}`
      : "No permission yet";
    pill.title = perm ? "Click to revoke permission" : "Click to give permission";
    const a = $("pill-agent");
    const online = bridge.state === "online";
    a.dataset.state = online ? "on" : bridge.state === "connecting" ? "wait" : "off";
    a.querySelector(".pill__text").textContent = online
      ? `Local agent ${bridge.info && bridge.info.dry_run ? "online (dry run)" : "online"}`
      : bridge.state === "connecting" ? "Connecting…" : "Local agent offline";
  }

  /* ================= command box ================= */
  const input = $("cmd-input");
  function submit(text) {
    text = String(text || "").trim();
    if (!text) { input.focus(); return; }
    input.value = "";
    $("cmd-suggest").hidden = true;
    showView("watch");
    if (agent.running || state.deviceRunning) { setCmdStatus("I'm still busy. Press Stop first."); return; }
    if (!state.permission[state.mode]) {
      if (state.mode === "device" && bridge.blocker) return renderDeviceHelp(text);
      setCmdStatus("I need your permission first.");
      return askPermission(state.mode, text);
    }
    if (state.mode === "sim") runSim(text);
    else runDevice(text);
  }
  $("cmd-form").addEventListener("submit", (e) => {
    e.preventDefault();
    if (agent.running) return agent.stop();
    if (state.deviceRunning) return void bridge.send({ type: "stop" });
    submit(input.value);
  });
  function setBusy(busy) {
    const go = $("cmd-go");
    go.textContent = busy ? "Stop" : "Go";
    go.classList.toggle("btn--go", !busy);
    go.classList.toggle("btn--stop", busy);
    go.title = busy ? "Stop (Esc)" : "Run this command";
  }

  $("cmd-chips").replaceChildren(...LW.planner.EXAMPLES.map((t) => h("button.chip-ex", { type: "button", onmousedown: (e) => e.preventDefault(), onclick: () => submit(t) }, t)));
  input.addEventListener("focus", () => { if (!input.value) $("cmd-suggest").hidden = false; });
  input.addEventListener("input", () => { $("cmd-suggest").hidden = !!input.value; });
  input.addEventListener("blur", () => setTimeout(() => { $("cmd-suggest").hidden = true; }, 150));

  function setCollapsed(c) {
    $("cmd").classList.toggle("is-collapsed", c);
    $("cmd-form").hidden = c;
    $("cmd-fab").hidden = !c;
    $("cmd-collapse").setAttribute("aria-expanded", String(!c));
    store.set("lw-cmd-collapsed", c);
  }
  $("cmd-collapse").addEventListener("click", () => setCollapsed(true));
  $("cmd-fab").addEventListener("click", () => { setCollapsed(false); input.focus(); });
  if (store.get("lw-cmd-collapsed", false)) setCollapsed(true);

  document.querySelectorAll(".target-switch__btn").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.targetMode)));

  function setMode(mode) {
    if (agent.running || state.deviceRunning) { setCmdStatus("Stop the current task before switching."); return; }
    state.mode = mode;
    document.querySelectorAll(".target-switch__btn").forEach((x) => x.setAttribute("aria-checked", String(x.dataset.targetMode === mode)));
    const device = mode === "device";
    $("screen-fit").hidden = device;
    $("live-view").hidden = !device;
    $("stage-eyebrow").textContent = device ? "Your computer" : "Practice screen";
    $("stage-hint").textContent = device
      ? "What the local agent sees on your real screen. The pointer moves on your computer, not here."
      : "A safe pretend computer. Nothing here touches your real files.";
    input.placeholder = device ? "What should I do on your computer?" : "What should I do? e.g. turn on dark mode";
    updatePills();
    if (device) {
      if (bridge.blocker) renderDeviceHelp(null);
      else if (!state.deviceRunning) renderDeviceIdle();
    } else renderIdle();
  }

  /* ================= keyboard ================= */
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (!$("consent").hidden) { $("consent-no").click(); return; }
      if (agent.running) { agent.stop(); e.preventDefault(); return; }
      if (state.deviceRunning) { bridge.send({ type: "stop" }); e.preventDefault(); }
      return;
    }
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName) || document.activeElement.isContentEditable;
    if (e.key === "/" && !typing) {
      e.preventDefault();
      setCollapsed(false);
      input.focus();
    }
  });
  $("btn-stop").addEventListener("click", () => {
    if (agent.running) agent.stop();
    else if (state.deviceRunning) bridge.send({ type: "stop" });
  });
  $("btn-pause").addEventListener("click", () => {
    if (agent.running) {
      if (agent.ctrl && agent.ctrl.paused) agent.resume(); else agent.pause("Paused. Press Resume to continue.");
    } else if (state.deviceRunning) {
      bridge.send({ type: state.devicePaused ? "resume" : "pause" });
    }
  });

  /* ================= real computer (local agent) ================= */
  let deviceLog = null;

  function renderDeviceHelp(text) {
    stepEls = [];
    deviceLog = null;
    setNotes(
      text ? h("p.plan__asked", {}, "You asked: ", h("q", {}, text)) : null,
      h("div.callout.callout--warn", {}, h("strong", {}, "The local agent is not connected."), h("p", {}, bridge.blocker)),
      h("ol.steps-plain", {},
        h("li", {}, "Open a terminal in the project folder and run ", h("code", {}, "cd agent")),
        h("li", {}, h("code", {}, "pip install -r requirements.txt")),
        h("li", {}, "Set ", h("code", {}, "ANTHROPIC_API_KEY"), " or run ", h("code", {}, "ant auth login")),
        h("li", {}, h("code", {}, "python -m leanwatch"), " opens this page, connected"),
        h("li", {}, "Add ", h("code", {}, "--dry-run"), " to watch it plan without moving your mouse")),
      h("p.small.muted", {}, "Until then, the practice screen works fully in this browser."));
    setRunState("idle", "Offline");
    setCmdStatus("The local agent is offline. Switch to Practice screen or start it.");
  }

  function renderDeviceIdle() {
    stepEls = [];
    deviceLog = null;
    const i = bridge.info || {};
    setNotes(
      h("p.notes__intro", {}, "Connected to your computer. Tell me a task, for example: open the calculator and work out 15% of 80."),
      h("ul.kv", {},
        h("li", {}, h("span", {}, "Model"), h("code", {}, i.model || "unknown")),
        h("li", {}, h("span", {}, "Screen"), h("code", {}, i.screen ? `${i.screen.width} × ${i.screen.height}` : "unknown")),
        h("li", {}, h("span", {}, "Mode"), h("code", {}, i.dry_run ? "dry run, the mouse won't move" : "live"))),
      i.api_ready === false ? h("div.callout.callout--warn", {}, h("strong", {}, "Claude API is not set up."), h("p", {}, i.api_hint || "Set ANTHROPIC_API_KEY and restart the agent.")) : null);
    setRunState("idle", "Ready");
  }

  function runDevice(text) {
    if (bridge.state !== "online") return renderDeviceHelp(text);
    bridge.send({ type: "grant", scopes: state.permission.device });
    if (!bridge.send({ type: "command", text })) return renderDeviceHelp(text);
    deviceLog = h("ol.devlog");
    setNotes(h("p.plan__asked", {}, "You asked: ", h("q", {}, text)), deviceLog, h("div.confirm-slot", { id: "confirm-slot" }), h("div.done-slot", { id: "done-slot" }));
    $("live-empty").hidden = false;
  }

  function logLine(kind, text) {
    if (!deviceLog) return;
    const li = h("li.devlog__item", { dataset: { kind } }, text);
    deviceLog.append(li);
    li.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  // Status updates refresh the idle panel, but never wipe a finished run's log.
  bridge.on("state", () => {
    updatePills();
    if (state.mode === "device" && !state.deviceRunning && !deviceLog) {
      if (bridge.state === "online") renderDeviceIdle(); else if (bridge.state === "offline") renderDeviceHelp(null);
    }
  });
  bridge.on("status", (s) => {
    updatePills();
    if (state.mode === "device" && !state.deviceRunning && !s.running && !deviceLog) renderDeviceIdle();
  });
  bridge.on("run_started", () => {
    state.deviceRunning = true;
    state.devicePaused = false;
    $("btn-pause").disabled = false;
    $("btn-stop").disabled = false;
    setBusy(true);
    setRunState("run", "Working");
    setCmdStatus("Working on your computer. Press Esc here, or slam the mouse into a screen corner, to stop.");
  });
  bridge.on("narration", (m) => { logLine(m.kind || "say", m.text); if (m.kind !== "final") setCmdStatus(m.text); });
  bridge.on("action", (m) => logLine("action", m.summary));
  bridge.on("screenshot", (m) => {
    $("live-img").src = m.image;
    $("live-empty").hidden = true;
    const dot = $("live-cursor");
    if (m.cursor) {
      dot.hidden = false;
      dot.style.left = `${m.cursor.fx * 100}%`;
      dot.style.top = `${m.cursor.fy * 100}%`;
    }
  });
  bridge.on("confirm_request", (m) => confirmCard(`${m.action}. ${m.reason}`, (allow) => {
    bridge.send({ type: "confirm", id: m.id, allow });
    setRunState("run", "Working");
  }));
  bridge.on("paused", (m) => { state.devicePaused = true; $("btn-pause").textContent = "Resume"; setRunState("pause", "Paused"); setCmdStatus(m.reason || "Paused."); logLine("pause", m.reason || "Paused."); });
  bridge.on("resumed", () => { state.devicePaused = false; $("btn-pause").textContent = "Pause"; setRunState("run", "Working"); });
  bridge.on("run_finished", (m) => {
    state.deviceRunning = false;
    state.devicePaused = false;
    $("btn-pause").disabled = true;
    $("btn-pause").textContent = "Pause";
    $("btn-stop").disabled = true;
    setBusy(false);
    const slot = $("done-slot") || notes;
    slot.replaceChildren(h("div.callout", { class: `callout ${m.ok ? "callout--ok" : "callout--warn"}` }, h("strong", {}, m.ok ? "Done." : "Stopped."), m.message ? h("p", {}, m.message) : null));
    setRunState(m.ok ? "done" : "stop", m.ok ? "Done" : "Stopped");
    setCmdStatus(m.ok ? "Done." : m.message || "Stopped.");
  });
  bridge.on("error", (m) => { logLine("error", m.message); setCmdStatus(m.message); });

  /* ================= boot ================= */
  renderIdle();
  updatePills();
  LW.progress.init(ctx);
  bridge.connect();

  window.LeanWatch = Object.assign(ctx, {
    state,
    submit,
    setMode,
    grant(mode = "sim", scopes = { mouse: true, keyboard: true }) { state.permission[mode] = scopes; updatePills(); },
  });
})();
