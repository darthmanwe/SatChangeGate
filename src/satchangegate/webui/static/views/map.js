import { api, el, chip, commandLine, spinner, setStatus } from "../app.js";

export const title = "Map";
export const group = "Explore";

/* Scenes on a real map, from the transform the rasters carry rather than a
   bounding box interpolated from a polygon.

   The basemap is off until someone turns it on, and turning it on is the only
   thing in this entire UI that touches the network. A local tool that silently
   phones a tile server is a local tool that leaks where you are looking. */

let leafletReady = null;

function loadLeaflet() {
  if (leafletReady) return leafletReady;
  leafletReady = new Promise((resolve, reject) => {
    const css = el("link", { rel: "stylesheet", href: "/static/vendor/leaflet/leaflet.css" });
    document.head.append(css);
    const s = document.createElement("script");
    s.src = "/static/vendor/leaflet/leaflet.js";
    s.onload = () => resolve(window.L);
    s.onerror = () => reject(new Error("Vendored Leaflet failed to load"));
    document.head.append(s);
  });
  return leafletReady;
}

export async function render(mount) {
  mount.append(el("header", {},
    el("h1", {}, "Where the imagery is"),
    el("p", { class: "lede" },
      "Twenty-four bitemporal AOIs, coloured by split. Click one to overlay its " +
      "before and after imagery and step through its tiles.")));

  const host = el("div", {}, spinner("reading scene headers…"));
  mount.append(host);

  let L, scenes, meta;
  try {
    [L, meta] = await Promise.all([loadLeaflet(), api.get("/api/scenes")]);
    scenes = meta.scenes;
  } catch (err) {
    host.replaceChildren(el("div", { class: "notice stop" },
      el("strong", {}, "Could not build the map. "), String(err.message || err)));
    return;
  }
  const located = scenes.filter((s) => s.bbox);
  if (!located.length) {
    host.replaceChildren(el("div", { class: "notice stop" },
      "No scene carries a georeference on this machine."));
    return;
  }

  host.replaceChildren();

  const controls = el("div", { style: "display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:10px" },
    toggle("basemap", false, (on) => setBasemap(on)),
    toggle("imagery overlay", true, (on) => { showOverlay = on; refreshOverlay(); }),
    el("span", { class: "chip" }, "t1/t2:"),
    el("button", { id: "which", class: "btn", onClick: () => {
      which = which === "t1" ? "t2" : "t1";
      document.getElementById("which").textContent = which;
      refreshOverlay();
    } }, "t1"),
    el("span", { class: "dim", style: "margin-left:auto" },
      `${located.length} located · bounds from ${located[0].bbox_source.replace("_", " ")}`));
  host.append(controls);

  const mapEl = el("div", { style: "height:min(62vh,540px); border:1px solid var(--rule); border-radius:6px" });
  host.append(mapEl);

  const info = el("div", { class: "card", style: "margin-top:12px" },
    el("p", { class: "dim" }, "Click a scene."));
  host.append(info);

  const map = L.map(mapEl, { zoomControl: true, attributionControl: true }).setView([20, 10], 2);
  let basemapLayer = null;
  let overlay = null;
  let selected = null;
  let which = "t1";
  let showOverlay = true;

  function setBasemap(on) {
    if (on && !basemapLayer) {
      basemapLayer = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19,
        attribution: "&copy; OpenStreetMap contributors",
      }).addTo(map);
      setStatus(undefined, "basemap on — this is the only network request the UI makes");
    } else if (!on && basemapLayer) {
      map.removeLayer(basemapLayer);
      basemapLayer = null;
      setStatus(undefined, "basemap off — no network requests");
    }
  }

  function refreshOverlay() {
    if (overlay) { map.removeLayer(overlay); overlay = null; }
    if (!selected || !showOverlay) return;
    const [w, s, e, n] = selected.bbox;
    overlay = L.imageOverlay(
      `/api/scenes/${selected.city}/preview/${which}`,
      [[s, w], [n, e]],
      { opacity: 0.95 },
    ).addTo(map);
  }

  const colors = { train: "#8a5420", test: "#0d6b75" };
  const rects = new Map();
  for (const scene of located) {
    const [w, s, e, n] = scene.bbox;
    const rect = L.rectangle([[s, w], [n, e]], {
      color: colors[scene.split] || "#888",
      weight: 1.5,
      fillOpacity: 0.12,
    }).addTo(map);
    rect.bindTooltip(`${scene.city} (${scene.split})`, { sticky: true });
    rect.on("click", () => selectScene(scene));
    rects.set(scene.city, rect);
  }
  map.fitBounds(L.latLngBounds(located.map((s) => [[s.bbox[1], s.bbox[0]], [s.bbox[3], s.bbox[2]]])
    .flat()), { padding: [24, 24] });

  function selectScene(scene) {
    selected = scene;
    for (const [city, rect] of rects) {
      rect.setStyle({ weight: city === scene.city ? 3 : 1.5,
        fillOpacity: city === scene.city ? 0.05 : 0.12 });
    }
    const [w, s, e, n] = scene.bbox;
    map.fitBounds([[s, w], [n, e]], { padding: [40, 40] });
    refreshOverlay();
    renderInfo(scene);
  }

  function renderInfo(scene) {
    info.replaceChildren(
      el("h3", {}, scene.city, " ",
        chip(scene.split, scene.split === "test" ? "accent" : undefined),
        scene.has_label ? chip("labelled", "ok") : chip("unlabelled", "warn")),
      el("div", { class: "grid cols-4" },
        kv("grid", `${scene.height} × ${scene.width} px`),
        kv("acquired", `${scene.date_t1 || "?"} → ${scene.date_t2 || "?"}`),
        kv("bounds from", scene.bbox_source.replace(/_/g, " ")),
        kv("CRS", scene.crs || "—")),
      el("p", { class: "dim", style: "margin-top:10px" },
        `Drawing this equirectangular grid on a Web Mercator basemap displaces it ` +
        `by about ${scene.mercator_skew_px} px vertically — measured, not assumed. ` +
        `At ${scene.height} px tall that is ` +
        `${((scene.mercator_skew_px / scene.height) * 100).toFixed(3)}% of the scene.`),
      ...(scene.notes || []).map((n) => el("div", { class: "notice warn" }, n)),
      el("div", { style: "display:flex; gap:8px; margin-top:10px; flex-wrap:wrap" },
        el("a", { class: "btn", href: `#compare?city=${scene.city}` }, "open in compare"),
        el("a", { class: "btn", href: `/api/scenes/${scene.city}/preview/t1`, target: "_blank" },
          "t1 preview"),
        el("a", { class: "btn", href: `/api/scenes/${scene.city}/preview/t2`, target: "_blank" },
          "t2 preview")),
      commandLine(`satchangegate run --pair ${scene.city}`),
    );
  }

  host.append(el("div", { class: "notice" },
    el("strong", {}, "Georeferencing. "), meta.georeferencing));
}

function kv(k, v) {
  return el("div", { class: "card stat" },
    el("span", { class: "k" }, k), el("span", { class: "v", style: "font-size:14px" }, v));
}

function toggle(label, initial, onChange) {
  const input = el("input", { type: "checkbox" });
  input.checked = initial;
  input.style.width = "auto";
  input.addEventListener("change", () => onChange(input.checked));
  return el("label", { style: "display:flex; gap:6px; align-items:center; font-size:12px" },
    input, label);
}
