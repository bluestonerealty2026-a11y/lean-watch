// LeanWatch data engine: records labelled training data from the practice computer.
//
// For every episode it randomises the screen (window positions, open apps, dark mode,
// brightness, extra folders), picks a task sentence, lets the agent do it, and saves a
// screenshot just before each action together with:
//   - the action (click / type / hotkey / drag / confirm / finished) in image pixels
//   - the box of the element being clicked or dragged
//   - the tutor explanation for that step (used as the model's "Thought")
// It also saves grounding samples: (screenshot, "Dark mode") -> box, for every
// visible, uniquely named element.
//
//   npm install                                   # once, installs Playwright
//   node training/record_practice.mjs --episodes 300 --out data/practice --seed 1 --scale 1.5
//
// Output (model-agnostic; training/build_dataset.py turns it into chat data for a model):
//   <out>/images/*.png
//   <out>/episodes.jsonl    one task per line, with its steps
//   <out>/grounding.jsonl   one (image, label, bbox) per line
//   <out>/meta.json
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
import path from "node:path";
import fs from "node:fs";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

/* ---------------- arguments ---------------- */
function parseArgs(argv) {
  const out = { episodes: 50, out: "data/practice", seed: 1, scale: 1, groundingPerShot: 4 };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    if (a === "--episodes") out.episodes = parseInt(next(), 10);
    else if (a === "--out") out.out = next();
    else if (a === "--seed") out.seed = parseInt(next(), 10);
    else if (a === "--scale") out.scale = parseFloat(next());
    else if (a === "--grounding-per-shot") out.groundingPerShot = parseInt(next(), 10);
    else if (a === "--help" || a === "-h") {
      console.log("node training/record_practice.mjs [--episodes N] [--out DIR] [--seed N] [--scale 1|1.5|2] [--grounding-per-shot N]");
      process.exit(0);
    } else throw new Error(`Unknown argument ${a}`);
  }
  return out;
}
const args = parseArgs(process.argv.slice(2));

/* ---------------- seeded randomness ---------------- */
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rng = mulberry32(args.seed);
const pick = (arr) => arr[Math.floor(rng() * arr.length)];
const num = (lo, hi, step = 1) => lo + step * Math.floor(rng() * (Math.floor((hi - lo) / step) + 1));
const chance = (p) => rng() < p;

/* ---------------- task sentences ---------------- */
const NAMES = ["Homework", "Recipes", "Taxes 2026", "Holiday Photos", "Projects", "Music Lessons", "Invoices", "School",
  "Drafts", "Garden Plans", "Scans", "Receipts", "Notes", "Old Files", "Family", "Work", "Archive", "Ideas", "Letters", "Budget"];
const quote = (name) => (/\s/.test(name) ? `"${name}"` : name);
const COLORS = ["red", "blue", "green", "orange", "purple", "gray"];
const APPS = ["Settings", "the photo editor", "PhotoLab", "the terminal", "the command prompt", "Blender", "Blocks3D", "the file explorer", "Files"];

const TEMPLATES = [
  () => pick(["Turn on dark mode", "Switch to dark mode", "Enable the dark theme", "Turn off dark mode", "Go back to light mode", "Use the light theme"]),
  () => pick(["Turn on night light", "Turn off night light", "Enable night light to reduce blue light", "Make the screen colours warmer"]),
  () => pick(["Turn off Wi-Fi", "Turn on Wi-Fi", "Connect to wifi", "Disconnect from the wireless network"]),
  () => pick(["Turn on Bluetooth", "Turn off Bluetooth", "Disable bluetooth", "Enable bluetooth so I can pair my headphones"]),
  () => pick([`Set screen brightness to ${num(20, 100, 5)}`, "Make the screen brighter", "Dim the screen", `Set the brightness to ${num(20, 100, 5)}`]),
  () => pick([`Set the volume to ${num(0, 100, 5)}`, "Make it louder", "Mute the sound", "Turn the volume down"]),
  () => pick(["Make my photo black and white", "Turn the picture grayscale", "Remove the colour from the image"]),
  () => pick(["Make the photo brighter", "Darken the photo", `Increase the brightness of the photo by ${num(10, 60, 5)}`, `Set the photo brightness to ${num(-60, 60, 5)}`]),
  () => pick(["Increase the contrast of the photo", "Reduce the contrast of the picture", `Set the photo contrast to ${num(-50, 50, 5)}`]),
  () => pick(["Blur the photo", "Give the photo a vintage look", "Flip the photo horizontally", `Rotate the photo ${pick([90, 180, 270])} degrees`, "Undo the last edit", "Reset the photo"]),
  () => pick([`Rotate the cube ${num(15, 180, 15)} degrees`, `Rotate the cube on the x axis by ${num(15, 90, 15)}`, `Rotate the cube to ${num(0, 330, 30)} degrees on the y axis`]),
  () => `Make the cube ${pick(COLORS)}`,
  () => pick(["Show the wireframe", "Switch to solid view", "Make the cube bigger", "Make the cube smaller", `Scale the cube to ${pick([0.5, 0.8, 1.2, 1.5, 2])}`]),
  () => pick([`Create a folder called ${quote(pick(NAMES))} using the terminal`, "Check my IP address", "List the files in this folder using the terminal", `Ping ${pick(["google", "wikipedia", "example.com"])}`]),
  () => `Create a new folder called ${quote(pick(NAMES))}`,
  (st) => (st.folders.length ? `Delete the folder ${quote(pick(st.folders))}` : null),
  (st) => (st.folders.length ? `Delete the folder ${quote(pick(st.folders))} using the terminal` : null),
  () => pick(["Open the Documents folder", "Show me my pictures", "Open the music folder"]),
  () => `Open ${pick(APPS)}`,
  (st) => (st.open.length ? `Close ${pick(st.open)}` : null),
  () => `${pick(["Set brightness to 40", "Turn on dark mode", "Turn off bluetooth"])} and ${pick(["make my photo black and white", "make the cube red", "check my IP address"])}`,
];

/* ---------------- browser ---------------- */
const here = path.dirname(fileURLToPath(import.meta.url));
const pageUrl = pathToFileURL(path.resolve(here, "..", "web", "index.html")).href;
const outDir = path.resolve(args.out);
fs.mkdirSync(path.join(outDir, "images"), { recursive: true });
const epFile = fs.createWriteStream(path.join(outDir, "episodes.jsonl"));
const grFile = fs.createWriteStream(path.join(outDir, "grounding.jsonl"));

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1100, height: 900 }, deviceScaleFactor: args.scale });
page.on("pageerror", (e) => console.error("page error:", e.message));
await page.goto(pageUrl);
await page.waitForFunction(() => window.LeanWatch && window.LeanWatch.os);
// A clean 1000x625 screen: no notes panel, command box, pointer, captions or animations.
await page.addStyleTag({ content: `
  .notes, .cmd, .stage__bar, .stage__controls, .topbar { display: none !important; }
  .view--watch { grid-template-columns: 1fr !important; }
  .screen-frame { padding: 0 !important; width: 1000px; }
  .screen-fit { width: 1000px !important; border-radius: 0 !important; }
  .lw-cursor, .lw-bubble, .lw-spot, .lw-ripple { display: none !important; }
  *, *::before, *::after { transition: none !important; animation: none !important; }
` });
await page.evaluate(() => {
  window.dispatchEvent(new Event("resize"));
  const lw = window.LeanWatch;
  lw.grant("sim");
  lw.agent.setSpeed(25);
  lw.agent.on("confirm", () => setTimeout(() => lw.agent.respond(true), 0));
});
await page.waitForFunction(() => Math.abs(window.LeanWatch.os.scale - 1) < 0.01);

const screen = page.locator(".simos");
const S = args.scale;
const px = (v) => Math.round(v * S);
const box = (b) => b && b.map(px);

/* ---------------- recording ---------------- */
let current = null; // episode being recorded
let shotIndex = 0;

async function takeShot() {
  const name = `${current.id}_s${String(current.steps.length).padStart(2, "0")}.png`;
  await screen.screenshot({ path: path.join(outDir, "images", name), animations: "disabled" });
  current.files.push(name);
  shotIndex++;
  return `images/${name}`;
}

async function groundingFor(image) {
  if (args.groundingPerShot <= 0) return;
  const targets = await page.evaluate(() => {
    const lw = window.LeanWatch;
    return lw.os.visibleTargets().map((t) => ({
      label: window.LW.describeElement(t.el),
      role: t.el.getAttribute("role") || t.el.tagName.toLowerCase(),
      target: t.target,
      bbox: [t.rect.x, t.rect.y, t.rect.x + t.rect.w, t.rect.y + t.rect.h],
    }));
  });
  const counts = new Map();
  for (const t of targets) counts.set(t.label.toLowerCase(), (counts.get(t.label.toLowerCase()) || 0) + 1);
  const usable = targets.filter((t) => t.label && !t.label.includes(".") && t.label.length <= 60 && counts.get(t.label.toLowerCase()) === 1);
  for (let i = usable.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [usable[i], usable[j]] = [usable[j], usable[i]]; }
  // Kept with the episode and written only if the episode is kept: a discarded episode's
  // screenshots are deleted, and its id (and image names) are reused by the next attempt.
  for (const t of usable.slice(0, args.groundingPerShot)) {
    current.groundingRows.push({ image, label: t.label, role: t.role, target: t.target, bbox: box(t.bbox), width: px(1000), height: px(625) });
  }
}

await page.exposeFunction("__lwBeforeAct", async (info) => {
  if (!current) return;
  const image = await takeShot();
  const step = { image, thought: info.thought, action: null };
  if (info.type === "click") step.action = { type: "click", x: px(info.x), y: px(info.y), clicks: info.clicks || 1, button: "left" };
  else if (info.type === "drag") step.action = { type: "drag", x1: px(info.x1), y1: px(info.y1), x2: px(info.x2), y2: px(info.y2) };
  else if (info.type === "type") step.action = { type: "type", text: info.text };
  else if (info.type === "hotkey") step.action = { type: "hotkey", keys: info.keys };
  else if (info.type === "confirm") step.action = { type: "confirm", text: info.text };
  if (info.bbox) step.bbox = box(info.bbox);
  if (info.label) step.label = info.label;
  current.steps.push(step);
  await groundingFor(image);
});
await page.evaluate(() => { window.LeanWatch.agent.beforeAct = (info) => window.__lwBeforeAct(info); });

async function randomiseScreen(seed) {
  return page.evaluate(({ seed, names }) => {
    let a = seed >>> 0;
    const r = () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    const lw = window.LeanWatch, os = lw.os;
    os.reset();
    for (const [, app] of os.apps) {
      const { w, h } = app.def.rect; // hidden windows measure 0x0, so use the app's declared size
      app.win.style.left = `${Math.round(r() * Math.max(0, 1000 - w))}px`;
      app.win.style.top = `${Math.round(r() * Math.max(0, 625 - 44 - h))}px`;
    }
    const ids = [...os.apps.keys()];
    const openCount = Math.floor(r() * 3);
    for (let i = 0; i < openCount; i++) os.open(ids[Math.floor(r() * ids.length)]);
    if (r() < 0.3) os.setSetting("dark", true);
    if (r() < 0.25) os.setSetting("brightness", 60 + Math.round(r() * 8) * 5);
    if (r() < 0.2) os.setSetting("bluetooth", false);
    const extra = Math.floor(r() * 3);
    for (let i = 0; i < extra; i++) os.fs.mkdir([], names[Math.floor(r() * names.length)]);
    const titles = { settings: "Settings", photolab: "the photo editor", blocks3d: "Blender", terminal: "the terminal", files: "Files" };
    return {
      folders: os.fs.list([]).filter((x) => x.type === "dir").map((x) => x.name),
      open: ids.filter((id) => os.isOpen(id)).map((id) => titles[id]),
    };
  }, { seed, names: NAMES });
}

/* ---------------- main loop ---------------- */
const stats = { episodes: 0, steps: 0, grounding: 0, skippedUnplanned: 0, skippedFailed: 0, bySkill: {} };
let attempts = 0;
while (stats.episodes < args.episodes && attempts < args.episodes * 4) {
  attempts++;
  const state = await randomiseScreen(Math.floor(rng() * 2 ** 31));
  const instruction = pick(TEMPLATES)(state);
  if (!instruction) continue;
  const plan = await page.evaluate((text) => {
    const p = window.LW.planner.plan(text, window.LeanWatch.os);
    return p.ok ? { ok: true, skills: p.items.map((i) => i.skill.id), summary: p.items.map((i) => i.summary).join("; ") } : { ok: false };
  }, instruction);
  if (!plan.ok) { stats.skippedUnplanned++; continue; }

  current = { id: `e${String(stats.episodes + 1).padStart(5, "0")}`, steps: [], files: [], groundingRows: [] };
  const result = await page.evaluate(async (text) => {
    const lw = window.LeanWatch;
    const says = [];
    const off = lw.agent.on("say", (e) => says.push(e.text));
    const out = await lw.agent.run(window.LW.planner.plan(text, lw.os), "watch");
    off();
    return { ok: out.ok, message: out.message, lastSay: says[says.length - 1] || "" };
  }, instruction);

  // Every point and box must lie on the screen image, or the sample would teach wrong coordinates.
  const W = px(1000), H = px(625);
  const inside = (x, y) => x >= 0 && y >= 0 && x < W && y < H;
  const onScreen = current.steps.every((s) => {
    const a = s.action || {};
    if (a.type === "click" && !inside(a.x, a.y)) return false;
    if (a.type === "drag" && !(inside(a.x1, a.y1) && inside(a.x2, a.y2))) return false;
    return !s.bbox || (s.bbox[0] >= 0 && s.bbox[1] >= 0 && s.bbox[2] <= W && s.bbox[3] <= H);
  });
  if (!onScreen) stats.skippedOffscreen = (stats.skippedOffscreen || 0) + 1;

  if (!result.ok || current.steps.length === 0 || !onScreen) {
    for (const f of current.files) fs.rmSync(path.join(outDir, "images", f), { force: true });
    stats.skippedFailed++;
    current = null;
    continue;
  }
  const finalImage = await takeShot();
  current.steps.push({ image: finalImage, thought: result.lastSay, action: { type: "finished", text: result.lastSay } });
  epFile.write(JSON.stringify({
    id: current.id,
    instruction,
    skills: plan.skills,
    summary: plan.summary,
    width: px(1000),
    height: px(625),
    steps: current.steps,
  }) + "\n");
  for (const row of current.groundingRows) grFile.write(JSON.stringify(row) + "\n");
  stats.episodes++;
  stats.steps += current.steps.length;
  stats.grounding += current.groundingRows.length;
  for (const s of plan.skills) stats.bySkill[s] = (stats.bySkill[s] || 0) + 1;
  if (stats.episodes % 25 === 0) console.log(`${stats.episodes}/${args.episodes} episodes, ${stats.steps} steps, ${stats.grounding} grounding samples`);
  current = null;
}

await browser.close();
await new Promise((r) => epFile.end(r));
await new Promise((r) => grFile.end(r));
const meta = { ...stats, seed: args.seed, scale: args.scale, screen: { width: px(1000), height: px(625) }, created: new Date().toISOString(), source: "LeanWatch practice computer" };
fs.writeFileSync(path.join(outDir, "meta.json"), JSON.stringify(meta, null, 2));
console.log(`Done: ${stats.episodes} episodes, ${stats.steps} steps, ${stats.grounding} grounding samples -> ${outDir}`);
if (stats.skippedFailed) console.log(`(${stats.skippedFailed} runs did not finish and were discarded)`);
