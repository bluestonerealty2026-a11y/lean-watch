/* Offline planner for the practice screen.
   1. Intent classification: every skill has example sentences. We turn text
      into TF-IDF vectors and pick the skill whose closest example has the
      highest cosine similarity (a nearest-neighbour classifier).
   2. Slot filling: regular expressions pull out numbers, names, colours,
      on/off and direction words.
   3. Planning: the chosen skill turns slots into a list of cursor steps,
      each with a plain-language explanation.
   On a real computer the local agent replaces all of this with Claude's
   computer-use toolset, which reads screenshots instead of known targets. */
(function () {
  "use strict";
  const LW = window.LW;
  const { clamp } = LW.util;

  /* ------------------------------------------------------------------ */
  /* Text normalisation                                                  */
  /* ------------------------------------------------------------------ */
  const PHRASES = [
    [/\bblack\s*(?:and|&|n)\s*white\b|\bb\s*&\s*w\b|\bgr[ae]y\s*scale\b|\bgr[ae]yscale\b/g, " monochrome "],
    [/\bcommand\s*prompt\b|\bcommand\s*line\b|\bcmd\b|\bpowershell\b|\bconsole\b/g, " terminal "],
    [/\bwi[\s-]?fi\b|\bwireless\b/g, " wifi "],
    [/\bblue\s*tooth\b/g, " bluetooth "],
    [/\bdark\s*(?:mode|theme)\b|\bnight\s*mode\b|\blight\s*(?:mode|theme)\b/g, " darkmode "],
    [/\bnight\s*(?:light|shift)\b|\bblue\s*light\b/g, " nightlight "],
    [/\bip\s*address\b|\bipconfig\b|\bip\b/g, " ipaddress "],
    [/\bphoto\s*(?:editor|lab)\b|\bphotoshop\b|\bphotolab\b/g, " photo editor "],
    [/\bblender\b|\bblocks\s*3d\b|\b3d\b/g, " cube editor3d "],
    [/\bfile\s*(?:explorer|manager|browser)\b|\bfinder\b|\bexplorer\b/g, " files "],
    [/\bdirector(?:y|ies)\b/g, " folder "],
  ];
  const WORDS = {
    picture: "photo", pic: "photo", pics: "photo", image: "photo", img: "photo", photograph: "photo",
    box: "cube", object: "cube", block: "cube", colour: "color", grey: "gray",
    sound: "volume", audio: "volume", speaker: "volume",
    bigger: "big", larger: "big", large: "big", grow: "big", smaller: "small", shrink: "small",
    louder: "loud", quieter: "quiet", softer: "quiet", dimmer: "dim", brighter: "bright", brighten: "bright",
    darker: "dark", darken: "dark", sideways: "rotate", spin: "rotate", tilt: "rotate", mirror: "flip",
    blurry: "blur", soften: "blur", vintage: "sepia", retro: "sepia", launch: "open", erase: "delete",
    remove: "delete", trash: "delete", display: "screen", monitor: "screen", setting: "settings",
    revert: "reset", original: "reset", rmdir: "delete", rd: "delete",
  };
  const STOP = new Set(("a an the my me i you your please can could would will to of for on off in into with using use " +
    "be so up down and then some it its is are do does just now want like lets let's help how what this that " +
    "there here by from via at all bit little lot much very really make set turn switch get go").split(" "));

  function stem(w) {
    if (w.length > 5 && w.endsWith("ing")) return w.slice(0, -3);
    if (w.length > 4 && w.endsWith("ed")) return w.slice(0, -2);
    if (w.length > 4 && w.endsWith("er")) return w.slice(0, -2);
    if (w.length > 3 && w.endsWith("s") && !w.endsWith("ss")) return w.slice(0, -1);
    return w;
  }

  function tokenize(text) {
    let t = ` ${String(text).toLowerCase()} `;
    for (const [re, rep] of PHRASES) t = t.replace(re, rep);
    return t
      .replace(/[^a-z0-9 ]+/g, " ")
      .split(/\s+/)
      .filter(Boolean)
      .map((w) => WORDS[w] || w)
      .filter((w) => !STOP.has(w) && !/^\d+(\.\d+)?$/.test(w))
      .map(stem);
  }

  /* ------------------------------------------------------------------ */
  /* TF-IDF nearest-neighbour intent model                               */
  /* ------------------------------------------------------------------ */
  class IntentModel {
    constructor(skills) {
      this.skills = skills;
      const df = new Map();
      const rows = [];
      skills.forEach((s, i) => s.examples.forEach((ex) => {
        const toks = tokenize(ex);
        rows.push({ skill: i, toks });
        new Set(toks).forEach((w) => df.set(w, (df.get(w) || 0) + 1));
      }));
      const N = rows.length;
      this.idf = new Map([...df].map(([w, d]) => [w, Math.log((1 + N) / (1 + d)) + 1]));
      this.rows = rows.map((r) => ({ skill: r.skill, vec: this.vectorize(r.toks) }));
      this.stats = { skills: skills.length, examples: N, vocabulary: df.size };
    }
    vectorize(toks) {
      const v = new Map();
      for (const w of toks) if (this.idf.has(w)) v.set(w, (v.get(w) || 0) + this.idf.get(w));
      const norm = Math.hypot(...v.values()) || 1;
      for (const [w, x] of v) v.set(w, x / norm);
      return v;
    }
    /** Skills ranked by their best-matching example (cosine similarity). */
    rank(text) {
      const q = this.vectorize(tokenize(text));
      const best = new Array(this.skills.length).fill(0);
      for (const row of this.rows) {
        let s = 0;
        for (const [w, x] of q) { const y = row.vec.get(w); if (y) s += x * y; }
        if (s > best[row.skill]) best[row.skill] = s;
      }
      return best.map((score, i) => ({ skill: this.skills[i], score })).sort((a, b) => b.score - a.score);
    }
  }

  /* ------------------------------------------------------------------ */
  /* Slot filling                                                        */
  /* ------------------------------------------------------------------ */
  const NOT_NAMES = new Set(["folder", "directory", "file", "the", "it", "a", "new", "my", "using", "with", "in", "this", "that"]);
  function firstIndex(t, re) { const m = t.match(re); return m ? m.index : -1; }

  function extractSlots(raw) {
    const text = String(raw);
    const t = text.toLowerCase().replace(/\b3d\b/g, " ");
    const slots = { raw: text };

    const num = t.match(/-?\d+(?:\.\d+)?/);
    slots.number = num ? parseFloat(num[0]) : null;
    slots.relative = /\bby\s+-?\d/.test(t);
    slots.absolute = /\bto\s+-?\d/.test(t);

    if (/\b(off|disable|deactivate|disconnect|light\s*(?:mode|theme))\b/.test(t)) slots.state = false;
    else if (/\b(on|enable|activate|connect|dark\s*(?:mode|theme)|night\s*mode)\b/.test(t)) slots.state = true;
    else slots.state = null;

    const upAt = firstIndex(t, /\b(bright(?:er|en)?|increase|raise|up|higher|more|louder|bigger|larger|grow|enlarge)\b/);
    const downAt = firstIndex(t, /\b(dim(?:mer)?|dark(?:er|en)?|decrease|lower|down|less|reduce|quieter|softer|smaller|shrink)\b/);
    slots.direction = upAt < 0 && downAt < 0 ? null : downAt < 0 || (upAt >= 0 && upAt < downAt) ? "up" : "down";
    if (/\btoo\s+(dark|dim)\b/.test(t)) slots.direction = "up";
    if (/\btoo\s+(bright|light)\b/.test(t)) slots.direction = "down";
    if (/\bmute\b/.test(t)) slots.mute = true;

    const color = t.match(/\b(red|blue|green|orange|purple|gr[ae]y)\b/);
    slots.color = color ? color[1].replace("grey", "gray") : null;

    const axis = t.match(/\b([xyz])[\s-]*axis\b|\baxis[\s-]*([xyz])\b|\baround\s+(?:the\s+)?([xyz])\b|\bon\s+([xyz])\b/);
    slots.axis = axis ? (axis[1] || axis[2] || axis[3] || axis[4]) : null;

    slots.solid = /\bsolid\b/.test(t);

    let name = null;
    const quoted = text.match(/["“”'‘’]([^"“”'‘’]{1,40})["“”'‘’]/);
    const called = text.match(/\b(?:called|named|titled|name it)\s+([A-Za-z0-9_\-.]+(?:\s+[A-Za-z0-9_\-.]+)?)/i);
    const verbObj = text.match(/\b(?:delete|remove|erase|trash|rmdir|rd|del|mkdir|md)\s+(?:the\s+|my\s+)?(?:folder\s+|directory\s+|file\s+)?([A-Za-z0-9_\-.]+)/i);
    if (quoted) name = quoted[1].trim();
    else if (called) name = called[1].replace(/\s+(?:using|with|in|on|from|via|and|then|folder|directory)$/i, "").trim();
    else if (verbObj && !NOT_NAMES.has(verbObj[1].toLowerCase())) name = verbObj[1];
    slots.name = name;

    const host = t.match(/\bping\s+([a-z0-9-]+(?:\.[a-z0-9-]+)+)/) || t.match(/\b(google|youtube|wikipedia|bing)\b/);
    slots.host = host ? (host[1].includes(".") ? host[1] : `${host[1]}.com`) : null;

    const toks = new Set(tokenize(text));
    slots.app = toks.has("photo") || toks.has("editor") && !toks.has("editor3d") ? "photolab"
      : toks.has("terminal") ? "terminal"
      : toks.has("cube") || toks.has("editor3d") ? "blocks3d"
      : toks.has("files") || toks.has("folder") ? "files"
      : toks.has("settings") ? "settings" : null;
    return slots;
  }

  /* ------------------------------------------------------------------ */
  /* Step builders                                                       */
  /* ------------------------------------------------------------------ */
  const APP_NAMES = { settings: "Settings", photolab: "PhotoLab", blocks3d: "Blocks3D", terminal: "Command Prompt", files: "Files" };
  const open = (app, say) => ({ do: "open", app, say: say || `First I open ${APP_NAMES[app]} from the taskbar at the bottom of the screen.` });
  const click = (target, say, extra) => ({ do: "click", target, say, ...extra });
  const type = (text, say) => ({ do: "type", text, say });
  const key = (k, say, mods) => ({ do: "key", key: k, say, ...mods });
  const slide = (target, value, say) => ({ do: "slider", target, value, say });
  const confirm = (say) => ({ do: "confirm", say });
  const verify = (test, say, failSay) => ({ do: "verify", test, say, failSay: failSay || "That did not work as expected." });
  const onOff = (v) => (v ? "on" : "off");
  const quoteArg = (s) => (/\s/.test(s) ? `"${s}"` : s);

  function toggleSkill({ id, title, setting, target, nav, navLabel, examples, tip, label }) {
    return {
      id, app: "settings", title, examples, tip,
      plan(slots, os) {
        let want = slots.state;
        if (want == null) want = !os.settings[setting];
        return {
          summary: `Turn ${label} ${onOff(want)}`,
          steps: [
            open("settings"),
            click(`settings.nav.${nav}`, `${label} lives in the ${navLabel} section, so I click ${navLabel} in the left column.`),
            click(target, `Now I click the ${label} switch to turn it ${onOff(want)}.`, {
              skipIf: (o) => o.settings[setting] === want,
              skipSay: `${label} is already ${onOff(want)}, so there is nothing to click.`,
            }),
            verify((o) => o.settings[setting] === want, `Done. ${label} is ${onOff(want)}.`),
          ],
        };
      },
    };
  }

  function sliderSetting({ id, title, setting, target, nav, navLabel, min, max, stepBy, examples, tip, label, unit = "%" }) {
    return {
      id, app: "settings", title, examples, tip,
      plan(slots, os) {
        const cur = os.settings[setting];
        let v = slots.number;
        if (slots.mute && setting === "volume") v = 0;
        else if (v == null) v = cur + (slots.direction === "down" ? -stepBy : stepBy);
        else if (slots.relative) v = cur + (slots.direction === "down" ? -Math.abs(v) : v);
        v = clamp(Math.round(v), min, max);
        return {
          summary: `Set ${label.toLowerCase()} to ${v}${unit}`,
          steps: [
            open("settings"),
            click(`settings.nav.${nav}`, `The ${label} slider is in the ${navLabel} section.`),
            slide(target, v, `I drag the ${label} slider ${v >= cur ? "right" : "left"} to ${v}${unit}. Dragging means: press, move, then let go.`),
            verify((o) => Math.abs(o.settings[setting] - v) <= 1, `${label} is now ${v}${unit}.`),
          ],
        };
      },
    };
  }

  function photoMenuSkill({ id, title, menu, item, menuLabel, itemLabel, examples, tip, test, done }) {
    return {
      id, app: "photolab", title, examples, tip,
      plan() {
        return {
          summary: title,
          steps: [
            open("photolab"),
            click(`photolab.menu.${menu}`, `${itemLabel} is in the ${menuLabel} menu at the top, so I open ${menuLabel}.`),
            click(`photolab.item.${item}`, `Then I choose "${itemLabel}".`),
            verify((o) => test(o.app("photolab").state, o), done),
          ],
        };
      },
    };
  }

  function photoAdjustSkill({ id, title, field, label, examples, tip }) {
    return {
      id, app: "photolab", title, examples, tip,
      plan(slots, os) {
        const cur = os.app("photolab").state[field];
        let v = slots.number;
        const down = slots.direction === "down";
        if (v == null) v = cur + (down ? -35 : 35);
        else if (slots.relative || !slots.absolute) v = cur + (down ? -Math.abs(v) : Math.abs(v));
        v = clamp(Math.round(v), -100, 100);
        return {
          summary: `Set photo ${label.toLowerCase()} to ${v > 0 ? "+" : ""}${v}`,
          steps: [
            open("photolab"),
            click("photolab.menu.image", `${label} controls are under the Image menu.`),
            click("photolab.item.brightness", 'I pick "Brightness/Contrast…". The dots mean a settings box will open.'),
            slide(`photolab.dialog.${field === "b" ? "brightness" : "contrast"}`, v, `I drag the ${label} slider to ${v > 0 ? "+" : ""}${v}. Watch the photo change as I move it.`),
            click("photolab.dialog.ok", "I click OK to keep the change."),
            verify((o) => o.app("photolab").state[field] === v, `Done. ${label} is ${v > 0 ? "+" : ""}${v}. You can press Ctrl+Z to undo it.`),
          ],
        };
      },
    };
  }

  function terminalSkill({ id, title, examples, tip, command, risky, check, done, explain }) {
    return {
      id, app: "terminal", title, examples, tip, risky,
      plan(slots, os) {
        const cmd = command(slots, os);
        if (cmd && cmd.error) return { error: cmd.error };
        const steps = [
          open("terminal"),
          click("terminal.input", "I click next to the blinking prompt so my typing goes into the terminal."),
          type(cmd, explain ? explain(cmd) : `I type the command: ${cmd}`),
        ];
        if (risky) steps.push(confirm(risky(cmd)));
        steps.push(key("Enter", "I press Enter to run it."));
        steps.push(verify((o) => check(o, slots, cmd), typeof done === "function" ? done(slots, cmd) : done));
        return { summary: `${title}: ${cmd}`, steps };
      },
    };
  }

  /* ------------------------------------------------------------------ */
  /* Skills                                                              */
  /* ------------------------------------------------------------------ */
  const SKILLS = [
    toggleSkill({
      id: "settings.darkmode", title: "Dark mode on or off", setting: "dark", target: "settings.darkmode", nav: "display", navLabel: "Display", label: "Dark mode",
      examples: ["turn on dark mode", "enable dark theme", "switch to dark mode", "turn off dark mode", "go back to light mode", "use the light theme", "night mode please", "make windows dark"],
      tip: "Windows 11: Settings, Personalization, Colors, Choose your mode. Mac: System Settings, Appearance.",
    }),
    toggleSkill({
      id: "settings.nightlight", title: "Night light on or off", setting: "nightlight", target: "settings.nightlight", nav: "display", navLabel: "Display", label: "Night light",
      examples: ["turn on night light", "enable night light", "reduce blue light", "make the screen colours warmer", "turn off night light", "warm screen at night"],
      tip: "Windows: Settings, System, Display, Night light. Mac: System Settings, Displays, Night Shift.",
    }),
    toggleSkill({
      id: "settings.wifi", title: "Wi-Fi on or off", setting: "wifi", target: "settings.wifi", nav: "network", navLabel: "Network", label: "Wi-Fi",
      examples: ["turn off wifi", "turn on wi-fi", "disconnect from the wireless network", "connect to wifi", "enable wifi", "disable wireless"],
      tip: "Windows: click the network icon at the right end of the taskbar. Mac: the Wi-Fi icon in the menu bar.",
    }),
    toggleSkill({
      id: "settings.bluetooth", title: "Bluetooth on or off", setting: "bluetooth", target: "settings.bluetooth", nav: "network", navLabel: "Network", label: "Bluetooth",
      examples: ["turn on bluetooth", "disable bluetooth", "switch bluetooth off", "enable bluetooth so i can pair my headphones", "bluetooth on"],
      tip: "Windows: Settings, Bluetooth & devices. Mac: System Settings, Bluetooth.",
    }),
    sliderSetting({
      id: "settings.brightness", title: "Screen brightness", setting: "brightness", target: "settings.brightness", nav: "display", navLabel: "Display", label: "Brightness",
      min: 10, max: 100, stepBy: 20,
      examples: ["set screen brightness to 50", "make the screen brighter", "dim the screen", "lower the display brightness", "increase screen brightness", "my screen is too dark", "turn the brightness up", "screen brightness"],
      tip: "Windows: Settings, System, Display, Brightness, or the sun keys on your keyboard. Mac: the sun keys or Control Centre.",
    }),
    sliderSetting({
      id: "settings.volume", title: "Volume", setting: "volume", target: "settings.volume", nav: "sound", navLabel: "Sound", label: "Volume",
      min: 0, max: 100, stepBy: 20,
      examples: ["set the volume to 30", "turn the volume up", "make it louder", "lower the sound", "mute the sound", "make it quieter", "volume down", "the speaker is too loud"],
      tip: "Windows: click the speaker icon on the taskbar. Mac: the speaker keys or Control Centre.",
    }),

    photoMenuSkill({
      id: "photo.grayscale", title: "Make the photo black and white", menu: "image", item: "grayscale", menuLabel: "Image", itemLabel: "Black & White",
      examples: ["make my photo black and white", "turn the picture grayscale", "remove the colour from the image", "desaturate the photo", "make the picture monochrome", "black and white photo"],
      tip: "Photoshop: Image, Adjustments, Black & White (Alt+Shift+Ctrl+B).",
      test: (s) => s.gray, done: "The photo is now black and white. The History panel on the right lists the change.",
    }),
    photoAdjustSkill({
      id: "photo.brightness", title: "Photo brightness", field: "b", label: "Brightness",
      examples: ["make the photo brighter", "brighten the picture", "increase the brightness of the image by 40", "the photo is too dark", "darken the photo", "lower the brightness of the picture", "photo brightness"],
      tip: "Photoshop: Image, Adjustments, Brightness/Contrast.",
    }),
    photoAdjustSkill({
      id: "photo.contrast", title: "Photo contrast", field: "c", label: "Contrast",
      examples: ["increase the contrast of the photo", "more contrast on the picture", "reduce contrast", "make the image punchier", "photo contrast"],
      tip: "Photoshop: Image, Adjustments, Brightness/Contrast.",
    }),
    photoMenuSkill({
      id: "photo.blur", title: "Blur the photo", menu: "filter", item: "blur", menuLabel: "Filter", itemLabel: "Blur",
      examples: ["blur the photo", "make the picture blurry", "soften the image", "add a blur effect"],
      tip: "Photoshop: Filter, Blur, Gaussian Blur.",
      test: (s) => s.blur > 0, done: "The photo is blurred.",
    }),
    photoMenuSkill({
      id: "photo.sepia", title: "Vintage look", menu: "filter", item: "sepia", menuLabel: "Filter", itemLabel: "Vintage (sepia)",
      examples: ["give the photo a vintage look", "make the picture sepia", "old fashioned photo effect", "retro filter on the image"],
      tip: "Photoshop: Image, Adjustments, Photo Filter, Sepia.",
      test: (s) => s.sepia, done: "The photo now has a warm vintage tone.",
    }),
    {
      id: "photo.rotate", app: "photolab", title: "Rotate the photo",
      examples: ["rotate the photo", "turn the picture sideways", "rotate the image 90 degrees", "the photo is the wrong way round", "rotate picture 180"],
      tip: "Photoshop: Image, Image Rotation, 90° Clockwise.",
      plan(slots, os) {
        const deg = slots.number && [90, 180, 270].includes(Math.abs(slots.number)) ? Math.abs(slots.number) : 90;
        const times = deg / 90;
        const start = os.app("photolab").state.rot;
        const steps = [open("photolab")];
        for (let i = 0; i < times; i++) {
          steps.push(click("photolab.menu.image", i ? "Again: Image menu." : "Rotation is in the Image menu."));
          steps.push(click("photolab.item.rotate", i ? "And Rotate 90° clockwise once more." : 'I choose "Rotate 90° clockwise". Each click turns it a quarter turn.'));
        }
        steps.push(verify((o) => o.app("photolab").state.rot === (start + deg) % 360, `Rotated ${deg}°.`));
        return { summary: `Rotate the photo ${deg}°`, steps };
      },
    },
    photoMenuSkill({
      id: "photo.flip", title: "Flip the photo", menu: "image", item: "flip", menuLabel: "Image", itemLabel: "Flip horizontal",
      examples: ["flip the photo", "mirror the image", "flip the picture horizontally"],
      tip: "Photoshop: Image, Image Rotation, Flip Canvas Horizontal.",
      test: () => true, done: "Flipped left to right.",
    }),
    {
      id: "photo.undo", app: "photolab", title: "Undo the last photo edit",
      examples: ["undo the last edit", "undo that", "go back one step", "cancel my last change to the photo", "undo"],
      tip: "Ctrl+Z on Windows and Linux, Cmd+Z on a Mac. Works in almost every program.",
      plan(slots, os) {
        const n = os.app("photolab").historyLabels.length;
        return {
          summary: "Undo the last photo edit",
          steps: [
            open("photolab"),
            key("z", "I press Ctrl+Z, the keyboard shortcut for Undo. Hold Ctrl, tap Z.", { ctrl: true }),
            verify((o) => n <= 1 || o.app("photolab").historyLabels.length === n - 1, n <= 1 ? "There was nothing to undo yet." : "The last edit is undone. It vanished from History."),
          ],
        };
      },
    },
    photoMenuSkill({
      id: "photo.revert", title: "Reset the photo", menu: "file", item: "revert", menuLabel: "File", itemLabel: "Revert to original",
      examples: ["reset the photo", "remove all edits from the picture", "go back to the original image", "revert the photo"],
      tip: "Photoshop: File, Revert (F12).",
      test: (s) => !s.gray && !s.sepia && !s.blur && !s.b && !s.c && !s.rot && !s.flip, done: "The photo is back to the original.",
    }),

    {
      id: "cube.rotate", app: "blocks3d", title: "Rotate the cube",
      examples: ["rotate the cube 45 degrees", "spin the cube by 90", "turn the cube on the x axis 30 degrees", "rotate the object", "tilt the box", "rotate the 3d model"],
      tip: "Blender: select the object, press R, then the axis letter (X, Y or Z), type the angle and press Enter.",
      plan(slots, os) {
        const axis = slots.axis || "z";
        const cur = os.app("blocks3d").state[`r${axis}`];
        const n = slots.number == null ? 45 : slots.number;
        const v = Math.round((((slots.absolute ? n : cur + n) % 360) + 360) % 360);
        return {
          summary: `Rotate the cube to ${v}° on ${axis.toUpperCase()}`,
          steps: [
            open("blocks3d"),
            click(`cube.rot.${axis}`, `In the Object panel on the right I triple-click the ${axis.toUpperCase()} rotation box. Triple-click selects everything in it.`, { clicks: 3 }),
            type(String(v), `I type ${v}. It replaces the selected number.`),
            key("Enter", "Enter confirms the new value."),
            verify((o) => Math.round(o.app("blocks3d").state[`r${axis}`]) === v, `The cube is rotated to ${v}° around ${axis.toUpperCase()}.`),
          ],
        };
      },
    },
    {
      id: "cube.color", app: "blocks3d", title: "Change the cube colour",
      examples: ["make the cube red", "change the cube colour to blue", "paint the box green", "set the material to orange", "purple cube", "color the object"],
      tip: "Blender: Material Properties tab (the red sphere icon), then Base Color.",
      plan(slots) {
        const c = slots.color || "red";
        return {
          summary: `Make the cube ${c}`,
          steps: [
            open("blocks3d"),
            click(`cube.color.${c}`, `Colours live in the Material section. I click the ${c} swatch.`),
            verify((o) => o.app("blocks3d").state.color === c, `The cube is ${c} now.`),
          ],
        };
      },
    },
    {
      id: "cube.wireframe", app: "blocks3d", title: "Wireframe or solid view",
      examples: ["show the wireframe", "switch to wireframe view", "see through the cube", "show solid view", "go back to solid shading", "x-ray the model"],
      tip: "Blender: press Z for the shading pie menu, or use the round shading buttons at the top right of the viewport.",
      plan(slots) {
        const wire = !slots.solid;
        return {
          summary: wire ? "Switch to wireframe view" : "Switch to solid view",
          steps: [
            open("blocks3d"),
            click(wire ? "cube.shading.wire" : "cube.shading.solid", `Viewport shading buttons are at the top. I click ${wire ? "Wireframe" : "Solid"}.`),
            verify((o) => o.app("blocks3d").state.wire === wire, wire ? "Now you can see every edge of the cube." : "Back to solid shading."),
          ],
        };
      },
    },
    {
      id: "cube.scale", app: "blocks3d", title: "Scale the cube",
      examples: ["make the cube bigger", "scale the cube up", "shrink the cube", "make the box smaller", "scale the cube to 1.5", "resize the object"],
      tip: "Blender: press S, type a number such as 1.5, then press Enter.",
      plan(slots, os) {
        const cur = os.app("blocks3d").state.scale;
        let v = slots.number;
        if (v != null && v > 2) v = v / 100;
        if (v == null) v = cur + (slots.direction === "down" ? -0.3 : 0.5);
        v = Math.round(clamp(v, 0.5, 2) * 10) / 10;
        return {
          summary: `Scale the cube to ${v.toFixed(1)}`,
          steps: [
            open("blocks3d"),
            slide("cube.scale", v, `I drag the Scale slider to ${v.toFixed(1)}. 1.0 is the original size.`),
            verify((o) => Math.abs(o.app("blocks3d").state.scale - v) < 0.05, `The cube is at scale ${v.toFixed(1)}.`),
          ],
        };
      },
    },

    terminalSkill({
      id: "terminal.mkdir", title: "Make a folder with the terminal",
      examples: ["create a folder using the terminal", "make a new directory with cmd", "use the command prompt to make a folder", "mkdir", "terminal new folder"],
      tip: "Windows Command Prompt and Mac/Linux Terminal: mkdir NAME. Put quotes around names with spaces.",
      command: (s) => `mkdir ${quoteArg(s.name || "NewFolder")}`,
      explain: (cmd) => `I type "${cmd}". mkdir is short for "make directory", which is the terminal word for folder.`,
      check: (o, s) => o.fs.exists([], s.name || "NewFolder") || o.fs.exists(o.app("terminal").path, s.name || "NewFolder"),
      done: (s) => `The folder "${s.name || "NewFolder"}" exists now. Open Files and you will see it there too.`,
    }),
    terminalSkill({
      id: "terminal.ip", title: "Find my IP address",
      examples: ["check my ip address", "what is my ip", "show network information in the terminal", "run ipconfig", "find my computer's address on the network"],
      tip: "Windows: ipconfig. Mac: ipconfig getifaddr en0. Linux: ip addr.",
      command: () => "ipconfig",
      explain: () => 'I type "ipconfig", which prints your network details.',
      check: (o) => o.app("terminal").lines.some((l) => /IPv4|Media State/.test(l)),
      done: "Look for the line that says IPv4 Address. That is this computer's address on your home network.",
    }),
    terminalSkill({
      id: "terminal.list", title: "List files with the terminal",
      examples: ["list the files in this folder using the terminal", "show what is in the folder with cmd", "run dir", "list directory contents in the terminal", "ls"],
      tip: "Windows: dir. Mac and Linux: ls.",
      command: () => "dir",
      explain: () => 'I type "dir", short for directory listing. On a Mac you would type ls.',
      check: (o) => o.app("terminal").lines.some((l) => /Directory of/.test(l)),
      done: "Folders are marked <DIR>. Files show their size.",
    }),
    terminalSkill({
      id: "terminal.ping", title: "Test the internet connection",
      examples: ["check if the internet works", "ping google", "test my internet connection", "is the website reachable"],
      tip: "Any terminal: ping example.com. Press Ctrl+C to stop it on Mac and Linux.",
      command: (s) => `ping ${s.host || "example.com"}`,
      explain: (cmd) => `I type "${cmd}". Ping sends a tiny message and times how long the reply takes.`,
      check: (o) => o.app("terminal").lines.some((l) => /Reply from|could not find host/.test(l)),
      done: "Replies with a time in ms mean the connection works. Smaller numbers are faster.",
    }),
    terminalSkill({
      id: "terminal.rmdir", title: "Delete a folder with the terminal",
      examples: ["delete a folder using the terminal", "remove a directory with cmd", "rmdir", "use the command prompt to delete a folder"],
      tip: "Windows: rmdir NAME (add /s to delete what is inside). Mac/Linux: rm -r NAME. There is no recycle bin.",
      command: (s, os) => {
        if (!s.name) return { error: 'Which folder should I delete? For example: delete the folder "Homework" using the terminal.' };
        const real = os.fs.realName([], s.name);
        if (!real) return { error: `I can't find a folder called "${s.name}" in your home folder.` };
        return `rmdir ${quoteArg(real)}`;
      },
      risky: (cmd) => `"${cmd}" deletes the folder for good. The terminal has no recycle bin. Shall I press Enter?`,
      check: (o, s) => !o.fs.exists([], s.name),
      done: (s) => `"${s.name}" is deleted.`,
    }),

    {
      id: "files.newfolder", app: "files", title: "Make a new folder",
      examples: ["create a new folder called", "make a folder on my computer", "new folder named", "add a new folder", "create folder"],
      tip: "Windows File Explorer: Ctrl+Shift+N or right-click, New, Folder. Mac Finder: Shift+Cmd+N.",
      plan(slots) {
        const name = slots.name || "New folder";
        return {
          summary: `Create the folder "${name}"`,
          steps: [
            open("files"),
            ...homeSteps(),
            click("files.newfolder", 'I click "New folder" in the toolbar.'),
            type(name, `A name box appears with the text already selected, so I just type "${name}".`),
            key("Enter", "Enter saves the name."),
            verify((o) => o.fs.exists([], name), `The folder "${name}" is ready.`),
          ],
        };
      },
    },
    {
      id: "files.delete", app: "files", title: "Delete a file or folder",
      examples: ["delete the folder", "remove this folder", "throw away the file", "trash a folder", "delete a file"],
      tip: "Windows: select it and press Delete. Mac: select it and press Cmd+Delete. Deleted items go to the Recycle Bin or Trash.",
      risky: true,
      plan(slots, os) {
        if (!slots.name) return { error: 'Which file or folder should I delete? For example: delete the folder "Music".' };
        const real = os.fs.realName([], slots.name);
        if (!real) return { error: `I can't find "${slots.name}" in your home folder.` };
        return {
          summary: `Delete "${real}"`,
          steps: [
            open("files"),
            ...homeSteps(),
            click(`files.item:${real}`, `I click "${real}" once to select it.`),
            confirm(`Deleting "${real}" can't be undone on this computer. Shall I continue?`),
            click("files.delete", "I click Delete in the toolbar."),
            click("files.confirm.delete", "Files asks me to confirm. I click Delete."),
            verify((o) => !o.fs.exists([], real), `"${real}" is gone.`),
          ],
        };
      },
    },
    {
      id: "files.open", app: "files", title: "Open a folder",
      examples: ["open the documents folder", "show me my pictures", "go into the music folder", "browse my files"],
      tip: "Double-click a folder to open it. The Back or Up arrow takes you out again.",
      plan(slots, os) {
        const known = ["Documents", "Pictures", "Music"].find((n) => new RegExp(`\\b${n.slice(0, -1)}`, "i").test(slots.raw));
        const name = slots.name && os.fs.realName([], slots.name) ? os.fs.realName([], slots.name) : known;
        if (!name) {
          return { summary: "Open Files", steps: [open("files"), verify(() => true, "Here are your files. Double-click a folder to open it.")] };
        }
        return {
          summary: `Open the ${name} folder`,
          steps: [
            open("files"),
            ...homeSteps(),
            click(`files.item:${name}`, `I double-click "${name}". Two quick clicks open a folder.`, { clicks: 2 }),
            verify((o) => o.app("files").path.join("/") === name, `You are inside ${name}.`),
          ],
        };
      },
    },

    {
      id: "app.open", app: null, title: "Open an app",
      examples: ["open settings", "launch the photo editor", "start the terminal", "open blender", "show me the file explorer", "open the 3d app", "open photoshop", "open command prompt"],
      tip: "Windows: press the Windows key and type the app name. Mac: Cmd+Space opens Spotlight; type the name and press Return.",
      plan(slots) {
        if (!slots.app) return { error: "Which app should I open? Try Settings, PhotoLab, Blocks3D, Command Prompt or Files." };
        return {
          summary: `Open ${APP_NAMES[slots.app]}`,
          steps: [open(slots.app, `I click the ${APP_NAMES[slots.app]} icon on the taskbar.`), verify((o) => o.isOpen(slots.app), `${APP_NAMES[slots.app]} is open.`)],
        };
      },
    },
    {
      id: "app.close", app: null, title: "Close an app",
      examples: ["close the terminal", "close the settings window", "exit the photo editor", "close blender", "close files"],
      tip: "Click the X in the window's corner. Windows: Alt+F4. Mac: Cmd+Q quits the app.",
      plan(slots, os) {
        const app = slots.app || os.focusedApp();
        if (!app) return { error: "No window is open right now." };
        return {
          summary: `Close ${APP_NAMES[app]}`,
          steps: [
            click(`win.${app}.close`, `I click the X in the top-right corner of the ${APP_NAMES[app]} window.`, { skipIf: (o) => !o.isOpen(app), skipSay: `${APP_NAMES[app]} is not open.` }),
            verify((o) => !o.isOpen(app), `${APP_NAMES[app]} is closed.`),
          ],
        };
      },
    },
  ];

  function homeSteps() {
    return [0, 1, 2].map((i) => click("files.up", i ? "Up once more." : "I go back to your home folder with the Up arrow.", {
      skipIf: (o) => o.app("files").path.length === 0,
      skipSay: null,
    }));
  }

  /* ------------------------------------------------------------------ */
  /* Planner                                                             */
  /* ------------------------------------------------------------------ */
  const THRESHOLD = 0.24;
  const model = new IntentModel(SKILLS);

  function classify(text) {
    const slots = extractSlots(text);
    // Names are slot values, not evidence for an intent ("Homework" says nothing about Files vs Terminal).
    let rankText = String(text).replace(/["“”'‘’][^"“”'‘’]*["“”'‘’]/g, " ");
    if (slots.name) rankText = rankText.split(slots.name).join(" ");
    const ranked = model.rank(rankText);
    // Nudge toward skills of an app the user named explicitly, away from other apps' skills.
    if (slots.app) {
      for (const r of ranked) {
        if (r.skill.app === slots.app) r.score += 0.06;
        else if (r.skill.app) r.score -= 0.06;
      }
    }
    ranked.sort((a, b) => b.score - a.score);
    return { text, slots, best: ranked[0], alternatives: ranked.slice(0, 3).map((r) => ({ id: r.skill.id, title: r.skill.title, score: r.score })) };
  }

  function splitCommands(text) {
    const parts = String(text).split(/\s*(?:;|,?\s+and\s+then\s+|,?\s+then\s+|\.\s+|,\s+(?=(?:and\s+)?(?:turn|make|set|open|close|create|delete|rotate|blur|flip|mute|show|check|run)\b))\s*/i)
      .map((p) => p.replace(/^and\s+/i, "").trim()).filter(Boolean);
    const out = [];
    for (const p of parts) {
      const pieces = p.split(/\s+and\s+/i);
      if (pieces.length > 1) {
        const scored = pieces.map(classify);
        const confident = scored.every((s) => s.best.score >= 0.34);
        const distinct = new Set(scored.map((s) => s.best.skill.id)).size === pieces.length;
        if (confident && distinct) { out.push(...pieces); continue; }
      }
      out.push(p);
    }
    return out.length ? out : [String(text)];
  }

  /** Turn a sentence into a plan the agent can run on the practice screen. */
  function plan(text, os) {
    const pieces = splitCommands(text);
    const items = [];
    for (const piece of pieces) {
      const c = classify(piece);
      if (!c.best || c.best.score < THRESHOLD) {
        return {
          ok: false,
          message: pieces.length > 1 ? `I understood some of that, but not "${piece}".` : "I don't know how to do that on the practice screen yet.",
          suggestions: c.alternatives.filter((a) => a.score > 0.05).map((a) => SKILLS.find((s) => s.id === a.id)),
        };
      }
      const skill = c.best.skill;
      const p = skill.plan(c.slots, os);
      if (p.error) return { ok: false, message: p.error, suggestions: [] };
      items.push({ text: piece, skill, score: c.best.score, alternatives: c.alternatives, slots: c.slots, summary: p.summary, steps: p.steps, tip: skill.tip });
    }
    return { ok: true, input: text, items };
  }

  const EXAMPLES = [
    "Turn on dark mode",
    "Make my photo black and white",
    "Rotate the cube 45 degrees",
    "Create a folder called Homework using the terminal",
    "Set brightness to 40 and turn off bluetooth",
    "Check my IP address",
  ];

  LW.planner = { plan, classify, tokenize, extractSlots, splitCommands, SKILLS, EXAMPLES, stats: model.stats, THRESHOLD };
})();
