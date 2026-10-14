/* Terminal: a Command Prompt look-alike sharing the same files as the Files app. */
(function () {
  "use strict";
  const LW = window.LW;
  const { h, deepClone } = LW.util;

  const ICON = '<svg viewBox="0 0 24 24"><rect x="2.5" y="4" width="19" height="16" rx="2"/><path d="M6 9l3 3-3 3M11 15h6"/></svg>';

  const HELP = [
    "Commands you can practise:",
    "  dir            list files and folders here (ls also works)",
    "  cd NAME        go into a folder   (cd .. goes back up)",
    "  mkdir NAME     make a new folder",
    "  rmdir NAME     delete a folder (careful: no recycle bin!)",
    "  echo TEXT      print some text",
    "  ipconfig       show this computer's network address",
    "  ping HOST      check whether a website answers",
    "  whoami, date, cls",
  ];

  LW.apps.push({
    id: "terminal",
    title: "Command Prompt",
    icon: ICON,
    rect: { x: 210, y: 110, w: 580, h: 340 },
    create(os, body) {
      body.classList.add("app-term");
      let path = [];
      let lines = ["SimOS [Version 1.0]  Practice terminal. Type help and press Enter.", ""];
      const recall = [];
      let recallAt = 0;

      const out = h("div.term-out", { role: "log", "aria-live": "off" });
      const promptEl = h("span.term-prompt");
      const input = h("input.term-input", { type: "text", spellcheck: "false", autocapitalize: "off", "aria-label": "Terminal command", dataset: { target: "terminal.input" } });
      const line = h("div.term-line", {}, promptEl, input);
      body.append(out, line);
      body.addEventListener("click", () => input.focus({ preventScroll: true }));

      const prompt = () => `${os.fs.label(path)}>`;
      const print = (...l) => { lines.push(...l); if (lines.length > 300) lines = lines.slice(-300); };

      function run(raw) {
        const cmdline = raw.trim();
        print(`${prompt()}${raw}`);
        if (!cmdline) return render();
        recall.push(cmdline);
        recallAt = recall.length;
        const [cmd, ...rest] = cmdline.split(/\s+/);
        const arg = rest.join(" ").replace(/^["']|["']$/g, "");
        switch (cmd.toLowerCase()) {
          case "help": print(...HELP); break;
          case "dir":
          case "ls": {
            const items = os.fs.list(path);
            print(` Directory of ${os.fs.label(path)}`, "");
            if (!items.length) print("  (empty folder)");
            for (const it of items) print(`  ${it.type === "dir" ? "<DIR>     " : `${String(it.size).padStart(6)} KB `}  ${it.name}`);
            print(`  ${items.filter((i) => i.type === "file").length} file(s)   ${items.filter((i) => i.type === "dir").length} folder(s)`);
            break;
          }
          case "cd": {
            if (!arg || arg === "~" || arg === "\\") { path = []; break; }
            if (arg === "..") { path = path.slice(0, -1); break; }
            const real = os.fs.realName(path, arg);
            const next = real ? [...path, real] : null;
            if (next && os.fs.resolve(next)) path = next;
            else print("The system cannot find the path specified.");
            break;
          }
          case "mkdir":
          case "md": {
            if (!arg) { print("Usage: mkdir FOLDER_NAME"); break; }
            const r = os.fs.mkdir(path, arg);
            print(r.ok ? `Created folder ${arg}` : r.error);
            break;
          }
          case "rmdir":
          case "rd":
          case "del": {
            if (!arg) { print(`Usage: ${cmd} NAME`); break; }
            const r = os.fs.remove(path, arg);
            print(r.ok ? `Deleted ${arg}` : r.error);
            break;
          }
          case "echo": print(arg); break;
          case "whoami": print("simos\\you"); break;
          case "date": print(new Date().toDateString()); break;
          case "ipconfig":
          case "ifconfig":
            print("Wireless LAN adapter Wi-Fi:", "");
            if (!os.settings.wifi) print("   Media State . . . . . . . . . . : Media disconnected (Wi-Fi is off)");
            else print(
              "   IPv4 Address. . . . . . . . . . : 192.168.1.42",
              "   Subnet Mask . . . . . . . . . . : 255.255.255.0",
              "   Default Gateway . . . . . . . . : 192.168.1.1",
              "   (practice data, not your real network)");
            break;
          case "ping": {
            const host = arg || "example.com";
            if (!os.settings.wifi) { print(`Ping request could not find host ${host}. Wi-Fi is off.`); break; }
            print(`Pinging ${host} with 32 bytes of data:`);
            for (let i = 0; i < 4; i++) print(`Reply from 93.184.216.34: bytes=32 time=${18 + i * 3}ms TTL=56`);
            break;
          }
          case "cls":
          case "clear": lines = []; break;
          default:
            print(`'${cmd}' is not recognized as a command. Type help to see what you can practise.`);
        }
        render();
      }

      input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") { const v = input.value; input.value = ""; run(v); }
        else if (e.key === "ArrowUp" && recall.length) { recallAt = Math.max(0, recallAt - 1); input.value = recall[recallAt]; e.preventDefault(); }
        else if (e.key === "ArrowDown" && recall.length) { recallAt = Math.min(recall.length, recallAt + 1); input.value = recall[recallAt] || ""; e.preventDefault(); }
      });

      function render() {
        out.textContent = lines.join("\n");
        promptEl.textContent = prompt();
        out.scrollTop = out.scrollHeight;
      }
      render();

      return {
        get lines() { return lines.slice(); },
        get path() { return path.slice(); },
        onOpen() { setTimeout(() => (out.scrollTop = out.scrollHeight), 0); },
        snapshot: () => ({ path, lines }),
        restore(s) { path = deepClone(s.path); lines = deepClone(s.lines); input.value = ""; render(); },
      };
    },
  });
})();
