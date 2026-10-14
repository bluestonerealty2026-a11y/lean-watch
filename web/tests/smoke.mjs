// End-to-end smoke test for the LeanWatch control center.
// Opens web/index.html in Chromium, grants permission through the real
// consent dialog, runs commands from the command box and checks that the
// practice computer changed. Also drives "Now you try" with real mouse clicks.
//
//   npm install            (installs Playwright)
//   npm test               (or: node web/tests/smoke.mjs)
//
// Set SHOTS=some/dir to also save screenshots, and LEANWATCH_URL to test a
// served copy instead of the local file.
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
import path from "node:path";
import fs from "node:fs";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const here = path.dirname(fileURLToPath(import.meta.url));
// LEANWATCH_URL tests a served copy (e.g. the GitHub Pages build); otherwise open the file directly.
const pageUrl = process.env.LEANWATCH_URL || pathToFileURL(path.resolve(here, "..", "index.html")).href;
const shots = process.env.SHOTS;
if (shots) fs.mkdirSync(shots, { recursive: true });

let failures = 0;
const results = [];
function check(name, ok, detail = "") {
  results.push(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  (${detail})` : ""}`);
  if (!ok) failures++;
}

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1360, height: 900 } });
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
page.on("console", (m) => { if (m.type() === "error" && !/fonts\.(googleapis|gstatic)/.test(m.text())) errors.push(m.text()); });

await page.goto(pageUrl);
await page.waitForFunction(() => window.LeanWatch && window.LeanWatch.os);
await page.evaluate(() => window.LeanWatch.agent.setSpeed(8));
if (shots) await page.screenshot({ path: path.join(shots, "1-idle.png") });

const runState = () => page.locator("#run-state").textContent();
async function command(text) {
  await page.fill("#cmd-input", text);
  await page.press("#cmd-input", "Enter");
}
async function waitIdle(timeout = 60000) {
  await page.waitForFunction(() => !window.LeanWatch.agent.running, null, { timeout });
}
const os = (fn) => page.evaluate(fn);

// 1. Consent appears before anything moves.
await command("Turn on dark mode");
check("consent dialog shows before first run", await page.locator("#consent").isVisible());
check("nothing ran without permission", (await os(() => window.LeanWatch.os.settings.dark)) === false);
await page.click("#consent-yes");
await page.waitForFunction(() => window.LeanWatch.agent.running);
if (shots) {
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(shots, "2-running.png") });
}
await waitIdle();
check("dark mode turned on", await os(() => window.LeanWatch.os.settings.dark === true));
check("run reports Done", (await runState()) === "Done", await runState());

// 2. Now you try: rewind and let the person do it with real clicks.
await page.getByRole("button", { name: "Now you try" }).click();
await page.waitForFunction(() => window.LeanWatch.agent.running && window.LeanWatch.agent.mode === "practice");
check("practice rewinds the screen", await os(() => window.LeanWatch.os.settings.dark === false));
for (const target of ["taskbar.settings", "settings.nav.display", "settings.darkmode"]) {
  const sel = `[data-target="${target}"]`;
  await page.waitForSelector(`.lw-spot:not([hidden])`);
  await page.waitForTimeout(250);
  await page.locator(sel).first().click();
}
await waitIdle();
check("practice completed by the person", await os(() => window.LeanWatch.os.settings.dark === true));
check("practice shows praise", await page.getByText("You did it yourself!").isVisible());

// 3. More commands across every app.
const cases = [
  ["Make my photo black and white", () => window.LeanWatch.os.app("photolab").state.gray === true],
  ["Make the photo brighter", () => window.LeanWatch.os.app("photolab").state.b === 35],
  ["Rotate the cube 45 degrees", () => window.LeanWatch.os.app("blocks3d").state.rz === 45],
  ["Make the cube red", () => window.LeanWatch.os.app("blocks3d").state.color === "red"],
  ["Create a folder called Homework using the terminal", () => window.LeanWatch.os.fs.exists([], "Homework")],
  ["Check my IP address", () => window.LeanWatch.os.app("terminal").lines.some((l) => l.includes("192.168.1.42"))],
  ["Set brightness to 40 and turn off bluetooth", () => Math.abs(window.LeanWatch.os.settings.brightness - 40) <= 1 && window.LeanWatch.os.settings.bluetooth === false],
  ["Create a new folder called Recipes", () => window.LeanWatch.os.fs.exists([], "Recipes")],
  ["Undo the last edit", () => window.LeanWatch.os.app("photolab").state.b === 0],
];
for (const [text, test] of cases) {
  await command(text);
  await waitIdle();
  const ok = await page.evaluate(test);
  check(text, ok, await runState());
}
if (shots) await page.screenshot({ path: path.join(shots, "3-after-commands.png") });

// 4. Risky step waits for an explicit Allow.
await command("Delete the folder Homework");
await page.waitForSelector(".callout--ask");
check("delete asks first", await os(() => window.LeanWatch.os.fs.exists([], "Homework")));
await page.locator(".callout--ask").getByRole("button", { name: "Allow" }).click();
await waitIdle();
check("delete after Allow", await os(() => !window.LeanWatch.os.fs.exists([], "Homework")));

// 5. Declining a risky step leaves things alone.
await command('Delete the folder "Recipes"');
await page.waitForSelector(".callout--ask");
await page.locator(".callout--ask").getByRole("button", { name: "No, skip" }).click();
await waitIdle();
check("declined delete keeps the folder", await os(() => window.LeanWatch.os.fs.exists([], "Recipes")));

// 6. Esc stops a run.
await page.evaluate(() => window.LeanWatch.agent.setSpeed(1));
await command("Make the cube bigger");
await page.waitForFunction(() => window.LeanWatch.agent.running);
await page.waitForTimeout(300);
await page.keyboard.press("Escape");
await waitIdle(5000);
check("Esc stops the agent", (await runState()) === "Stopped", await runState());
await page.evaluate(() => window.LeanWatch.agent.setSpeed(8));

// 7. Unknown request is explained, not guessed.
await command("Bake me a chocolate cake");
await page.waitForTimeout(200);
check("unknown request is not executed", (await runState()) === "Not understood", await runState());

// 8. Other views render.
await page.click("#tab-progress");
const diag = await page.locator("#diag-list li").count();
check("progress page shows live checks", diag >= 5, `${diag} checks`);
check("overall percent renders", /\d+/.test(await page.locator(".overall__value").textContent()));
if (shots) await page.screenshot({ path: path.join(shots, "4-progress.png"), fullPage: true });
await page.click("#tab-how");
check("how-it-works renders", await page.locator(".loop-svg").isVisible());

// 9. Phone width: no sideways scroll.
await page.setViewportSize({ width: 390, height: 844 });
await page.click("#tab-watch");
await page.waitForTimeout(300);
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check("no horizontal scroll at 390px", overflow <= 0, `${overflow}px`);
if (shots) await page.screenshot({ path: path.join(shots, "5-phone.png"), fullPage: true });

check("no console or page errors", errors.length === 0, errors.slice(0, 3).join(" | "));
await browser.close();

console.log(results.join("\n"));
console.log(failures ? `\n${failures} check(s) failed` : `\nAll ${results.length} checks passed`);
process.exit(failures ? 1 : 0);
