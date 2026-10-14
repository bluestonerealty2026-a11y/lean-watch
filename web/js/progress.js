/* Build progress tracker. The statuses below describe this repository as
   committed; click a chip on the page to update it as you keep building
   (your changes are remembered in this browser only). */
(function () {
  "use strict";
  const LW = window.LW;
  const { h, store } = LW.util;

  const UPDATED = "14 Oct 2026";
  const STORE_KEY = "lw-progress-v1";
  const STATUSES = ["planned", "next", "doing", "done"];
  const LABEL = { planned: "Planned", next: "Next", doing: "In progress", done: "Done" };
  const WEIGHT = { planned: 0, next: 0, doing: 0.5, done: 1 };

  const PHASES = [
    {
      id: "web", title: "Control center in the browser",
      tasks: [
        { id: "cmdbox", title: "Command box in the bottom-right of every view", where: "web/index.html", status: "done", tags: ["UI"] },
        { id: "consent", title: "Permission screen before the cursor moves", where: "web/index.html #consent", status: "done", tags: ["Safety"] },
        { id: "simos", title: "Practice computer with Settings, PhotoLab, Blocks3D, Command Prompt and Files", where: "web/js/simos.js, web/js/apps/", status: "done", tags: ["UI"] },
        { id: "cursor", title: "Human-like pointer: Fitts's law timing, minimum-jerk path", where: "web/js/cursor.js", status: "done", tags: ["HCI", "Robotics"] },
        { id: "planner", title: "Offline intent planner: TF-IDF classifier and slot filling", where: "web/js/planner.js", status: "done", tags: ["NLP"] },
        { id: "narrate", title: "Step-by-step narration, read-aloud, tips for the real app", where: "web/js/agent.js", status: "done", tags: ["UI"] },
        { id: "practice", title: "Now you try: rewind the screen and coach the person through it", where: "web/js/agent.js", status: "done", tags: ["Learning"] },
        { id: "control", title: "Pause, Stop with Esc, pause when the person clicks", where: "web/js/main.js", status: "done", tags: ["Safety"] },
        { id: "tracker", title: "This progress page with live system checks", where: "web/js/progress.js", status: "done", tags: ["UI"] },
        { id: "smoke", title: "Browser smoke test that runs real commands end to end", where: "web/tests/smoke.mjs", status: "done", tags: ["Testing"] },
      ],
    },
    {
      id: "agent", title: "Local agent that controls your real computer",
      tasks: [
        { id: "screen", title: "Screenshots scaled to the model's image limits, coordinates mapped back", where: "agent/leanwatch/screen.py", status: "done", tags: ["CV"] },
        { id: "controller", title: "Mouse and keyboard for all 17 computer-use actions", where: "agent/leanwatch/controller.py", status: "done", tags: ["Robotics"] },
        { id: "brain", title: "Open-model brain on your computer: screenshot, Thought, Action, no API key", where: "agent/leanwatch/local_brain.py", status: "done", tags: ["LLM", "Agents", "CV"] },
        { id: "actions", title: "UI-TARS action language: parser, writer, coordinate spaces", where: "agent/leanwatch/actions.py", status: "done", tags: ["NLP"] },
        { id: "claude", title: "Optional Claude brain for comparison", where: "agent/leanwatch/brain.py", status: "done", tags: ["LLM"] },
        { id: "server", title: "Local server: serves this page, WebSocket bridge, token and host checks", where: "agent/leanwatch/server.py", status: "done", tags: ["Safety"] },
        { id: "overlay", title: "Always-on-top command box in the corner of every screen", where: "agent/leanwatch/overlay.py", status: "done", tags: ["UI"] },
        { id: "safety", title: "Scopes, risky-action confirmation, take-over pause, fail-safe corner", where: "agent/leanwatch/permissions.py", status: "done", tags: ["Safety"] },
        { id: "unit", title: "Unit tests that need no API key, display or model download", where: "agent/tests/", status: "done", tags: ["Testing"] },
        { id: "firstrun", title: "First live run with UI-TARS-1.5-7B on a real desktop", where: "python -m leanwatch", status: "next", tags: ["Testing"] },
        { id: "platforms", title: "Check Windows, macOS (Retina scaling) and Linux", where: "agent/leanwatch/screen.py", status: "next", tags: ["Testing"] },
        { id: "osperms", title: "Guide for macOS Accessibility and Screen Recording permissions", where: "README.md", status: "doing", tags: ["Docs"] },
      ],
    },
    {
      id: "product", title: "Ready for people who are not programmers",
      tasks: [
        { id: "installer", title: "One-click installer (PyInstaller) and desktop shortcut", status: "planned", tags: ["Packaging"] },
        { id: "autostart", title: "Start the corner command box when the computer starts", status: "planned", tags: ["UI"] },
        { id: "voice", title: "Speak commands instead of typing (speech-to-text, e.g. Whisper)", status: "planned", tags: ["Speech", "NLP"] },
        { id: "lessons", title: "Save a finished task as a lesson to replay later", status: "planned", tags: ["Learning"] },
        { id: "i18n", title: "Narration in other languages", status: "planned", tags: ["NLP"] },
      ],
    },
    {
      id: "model", title: "Fine-tune an open model from Hugging Face",
      tasks: [
        { id: "dataset", title: "Data engine: record screenshot, instruction, thought and action on the practice screen", where: "training/record_practice.mjs", status: "done", tags: ["Data"] },
        { id: "import", title: "Import public GUI datasets (GroundCUA, OS-Atlas, Aguvis, AgentNet, ScreenSpot)", where: "training/import_hf.py", status: "done", tags: ["Data"] },
        { id: "build", title: "Build chat-format examples: grounding and multi-step actions", where: "training/build_dataset.py", status: "done", tags: ["NLP", "CV"] },
        { id: "sft", title: "LoRA / QLoRA supervised fine-tuning with TRL (loss on answers only)", where: "training/train_sft.py", status: "done", tags: ["DL", "PyTorch"] },
        { id: "eval", title: "Click-inside-target accuracy before and after training", where: "training/evaluate.py", status: "done", tags: ["Evaluation"] },
        { id: "pipetest", title: "End-to-end pipeline test with a tiny Qwen2.5-VL on CPU", where: "training/tests/test_pipeline.py", status: "done", tags: ["Testing"] },
        { id: "gpurun", title: "First real fine-tune of UI-TARS-1.5-7B on a GPU", status: "next", tags: ["DL"] },
        { id: "qwen35", title: "Support Qwen3.5-based agents (UI-Mate, UI-Venus-2) and their tool-call format", status: "planned", tags: ["LLM"] },
        { id: "rl", title: "Reinforcement learning (GRPO) with the practice screen as the environment; reward when a verify step passes", status: "planned", tags: ["RL"] },
        { id: "bench", title: "Benchmark: ScreenSpot-Pro, OSWorld-G, task success rate, wrong-click rate", status: "planned", tags: ["Evaluation"] },
      ],
    },
  ];

  let overrides = store.get(STORE_KEY, {});
  const statusOf = (t) => overrides[t.id] || t.status;
  const pct = (tasks) => Math.round((tasks.reduce((s, t) => s + WEIGHT[statusOf(t)], 0) / tasks.length) * 100);

  function render() {
    const all = PHASES.flatMap((p) => p.tasks);
    const counts = Object.fromEntries(STATUSES.map((s) => [s, all.filter((t) => statusOf(t) === s).length]));
    const total = pct(all);

    document.getElementById("overall").replaceChildren(
      h("div.overall__num", {}, h("span.overall__value", {}, String(total)), h("span.overall__unit", {}, "%")),
      h("div.meter", { role: "img", "aria-label": `${total} percent complete` }, h("span.meter__fill", { style: { width: `${total}%` } })),
      h("dl.overall__counts", {},
        ...["done", "doing", "next", "planned"].map((s) => h("div.overall__count", {}, h("dt", {}, LABEL[s]), h("dd", {}, String(counts[s]))))),
      h("p.muted.small", {}, `Updated ${UPDATED}`));

    document.getElementById("phases").replaceChildren(...PHASES.map((phase, i) => {
      const p = pct(phase.tasks);
      return h("section.phase", { "aria-labelledby": `phase-${phase.id}` },
        h("header.phase__head", {},
          h("span.phase__num", {}, `Phase ${i + 1}`),
          h("h2.phase__title", { id: `phase-${phase.id}` }, phase.title),
          h("span.phase__pct", {}, `${p}%`)),
        h("div.meter.meter--thin", { "aria-hidden": "true" }, h("span.meter__fill", { style: { width: `${p}%` } })),
        h("ul.tasks", {}, ...phase.tasks.map((t) => {
          const s = statusOf(t);
          const chip = h("button.chip", { class: `chip chip--${s}`, title: "Click to change status", "aria-label": `${t.title}. Status: ${LABEL[s]}. Click to change.` }, LABEL[s]);
          chip.addEventListener("click", () => {
            const next = STATUSES[(STATUSES.indexOf(s) + 1) % STATUSES.length];
            if (next === t.status) delete overrides[t.id]; else overrides[t.id] = next;
            store.set(STORE_KEY, overrides);
            render();
          });
          return h("li.task", { dataset: { status: s } }, chip,
            h("div.task__text", {},
              h("div.task__title", {}, t.title),
              h("div.task__meta", {},
                t.where ? h("code.task__where", {}, t.where) : null,
                ...(t.tags || []).map((tag) => h("span.tag", {}, tag)))));
        })));
    }));

    const upcoming = all.filter((t) => ["doing", "next"].includes(statusOf(t))).slice(0, 4);
    const list = upcoming.length ? upcoming : all.filter((t) => statusOf(t) === "planned").slice(0, 3);
    document.getElementById("next-list").replaceChildren(...list.map((t) => h("li", {}, t.title)));
  }

  /* ---------------- live checks ---------------- */
  function checks(ctx) {
    const out = [];
    const add = (state, label, detail) => out.push({ state, label, detail });
    try {
      const n = ctx.os ? ctx.os.apps.size : 0;
      add(n === 5 ? "pass" : "fail", "Practice computer", `${n} apps loaded on a ${ctx.os.W}×${ctx.os.H} screen`);
    } catch (e) { add("fail", "Practice computer", e.message); }
    try {
      const ms = Math.round(LW.motion.movementTime(300));
      add("pass", "Cursor engine", `A 300 px move takes about ${ms} ms at normal speed`);
    } catch (e) { add("fail", "Cursor engine", e.message); }
    try {
      const st = LW.planner.stats;
      const probe = LW.planner.classify("please turn on dark mode").best.skill.id;
      add(probe === "settings.darkmode" ? "pass" : "fail", "Intent planner",
        `${st.skills} skills, ${st.examples} examples, ${st.vocabulary} words. Self-test ${probe === "settings.darkmode" ? "passed" : "failed"}`);
    } catch (e) { add("fail", "Intent planner", e.message); }
    add("speechSynthesis" in window ? "pass" : "warn", "Read aloud", "speechSynthesis" in window ? "Your browser can read steps aloud" : "This browser has no speech voice");
    const b = ctx.bridge;
    if (b.state === "online") add("pass", "Local agent", b.info && b.info.model ? `Connected. Model ${b.info.model}${b.info.dry_run ? ", dry run" : ""}` : "Connected");
    else add("warn", "Local agent", b.blocker || "Not connected yet. Retrying.");
    let storageOk = false;
    try { localStorage.setItem("lw-probe", "1"); localStorage.removeItem("lw-probe"); storageOk = true; } catch { /* blocked */ }
    add(storageOk ? "pass" : "warn", "Browser storage", storageOk ? "Status chips are remembered" : "Blocked here, so chip changes reset on reload");
    return out;
  }

  function renderChecks(ctx) {
    const items = checks(ctx);
    document.getElementById("diag-list").replaceChildren(...items.map((c) =>
      h("li.diag__item", { dataset: { state: c.state } },
        h("span.diag__mark", { "aria-hidden": "true" }, c.state === "pass" ? "✓" : c.state === "warn" ? "!" : "✕"),
        h("div", {}, h("div.diag__label", {}, h("span.sr-only", {}, `${c.state}: `), c.label), h("div.diag__detail", {}, c.detail)))));
  }

  function init(ctx) {
    render();
    renderChecks(ctx);
    document.getElementById("btn-recheck").addEventListener("click", () => renderChecks(ctx));
    document.getElementById("btn-reset-progress").addEventListener("click", () => {
      overrides = {};
      store.remove(STORE_KEY);
      render();
    });
    ctx.bridge.on("state", () => renderChecks(ctx));
  }

  LW.progress = { init, renderChecks, PHASES };
})();
