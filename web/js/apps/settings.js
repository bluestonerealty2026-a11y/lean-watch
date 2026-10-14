/* Settings app: display, network and sound, like a laptop's settings panel. */
(function () {
  "use strict";
  const LW = window.LW;
  const { h } = LW.util;

  const ICON = '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3.2"/><path d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3M5.3 5.3l2.1 2.1M16.6 16.6l2.1 2.1M5.3 18.7l2.1-2.1M16.6 7.4l2.1-2.1"/></svg>';

  const SECTIONS = [
    { id: "display", label: "Display" },
    { id: "network", label: "Network" },
    { id: "sound", label: "Sound" },
  ];

  LW.apps.push({
    id: "settings",
    title: "Settings",
    icon: ICON,
    rect: { x: 240, y: 56, w: 520, h: 360 },
    create(os, body) {
      let section = "display";
      const nav = h("nav.set-nav", { "aria-label": "Settings sections" });
      const panel = h("div.set-panel");
      body.classList.add("app-settings");
      body.append(nav, panel);

      const navBtns = SECTIONS.map((s) => {
        const b = h("button.set-nav__btn", { dataset: { target: `settings.nav.${s.id}` } }, s.label);
        b.addEventListener("click", () => { section = s.id; render(); });
        nav.append(b);
        return b;
      });

      const row = (title, sub, control) =>
        h("div.set-row", {}, h("div.set-row__text", {}, h("div.set-row__title", {}, title), sub ? h("div.set-row__sub", {}, sub) : null), control);

      function render() {
        navBtns.forEach((b, i) => b.classList.toggle("is-active", SECTIONS[i].id === section));
        panel.replaceChildren();
        const s = os.settings;
        if (section === "display") {
          panel.append(
            h("h3.set-h", {}, "Display"),
            row("Dark mode", "Dark colours for windows and menus", os.ui.toggle({ target: "settings.darkmode", label: "Dark mode", checked: s.dark, onChange: (v) => os.setSetting("dark", v) })),
            row("Night light", "Warmer colours to help you sleep", os.ui.toggle({ target: "settings.nightlight", label: "Night light", checked: s.nightlight, onChange: (v) => os.setSetting("nightlight", v) })),
            row("Brightness", null, os.ui.slider({ target: "settings.brightness", label: "Brightness", min: 10, max: 100, value: s.brightness, format: (v) => `${v}%`, onChange: (v) => os.setSetting("brightness", v) })),
          );
        } else if (section === "network") {
          panel.append(
            h("h3.set-h", {}, "Network"),
            row("Wi-Fi", s.wifi ? "Connected to HomeNet" : "Off", os.ui.toggle({ target: "settings.wifi", label: "Wi-Fi", checked: s.wifi, onChange: (v) => os.setSetting("wifi", v) })),
            row("Bluetooth", s.bluetooth ? "On, discoverable as SimOS-Laptop" : "Off", os.ui.toggle({ target: "settings.bluetooth", label: "Bluetooth", checked: s.bluetooth, onChange: (v) => os.setSetting("bluetooth", v) })),
          );
        } else {
          panel.append(
            h("h3.set-h", {}, "Sound"),
            row("Volume", null, os.ui.slider({ target: "settings.volume", label: "Volume", min: 0, max: 100, value: s.volume, format: (v) => (v === 0 ? "Muted" : `${v}%`), onChange: (v) => os.setSetting("volume", v) })),
          );
        }
      }

      // Re-render when settings change from elsewhere, but not while a slider is being dragged.
      os.on("settings", (e) => { if (!e || !e.key || e.key === "wifi" || e.key === "bluetooth") render(); });
      render();

      return {
        get section() { return section; },
        snapshot: () => ({ section }),
        restore(s) { section = s.section; render(); },
      };
    },
  });
})();
