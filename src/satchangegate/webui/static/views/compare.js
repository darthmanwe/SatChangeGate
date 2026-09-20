import { api, el, chip, commandLine, spinner } from "../app.js";

export const title = "Compare";
export const group = "Explore";

/* Before and after, four ways: swipe, blend, side by side, and a difference
   image computed in the browser. Either side can be any rendered layer, so you
   can swipe the change mask over the imagery it came from.

   The difference view is labelled as what it is -- an RGB residual of two
   display renderings -- and never as the detector's output. The gate works on
   calibrated reflectance across six bands against a robust per-scene threshold;
   it sees things this does not and ignores things this shows. Conflating them
   would flatter the gate with a picture it did not produce. */

const MODES = ["swipe", "blend", "side by side", "difference"];

export async function render(mount) {
  mount.append(el("header", {},
    el("h1", {}, "Before and after"),
    el("p", { class: "lede" },
      "Whole scenes with every derived layer, or a single scored tile with the " +
      "exact evidence the verifier was shown.")));

  const host = el("div", {}, spinner());
  mount.append(host);

  let scenes;
  try {
    scenes = (await api.get("/api/scenes")).scenes;
  } catch (err) {
    host.replaceChildren(el("div", { class: "notice stop" }, String(err.message || err)));
    return;
  }

  const params = new URLSearchParams(location.hash.split("?")[1] || "");
  const state = {
    source: "scene",
    city: params.get("city") || scenes[0]?.city,
    left: "rgb_t1",
    right: "rgb_t2",
    tile: null,
    tiles: [],
    mode: "swipe",
    position: 50,
    layerMeta: null,
  };

  host.replaceChildren();
  const bar = el("div", { class: "grid cols-4", style: "margin-bottom:12px" });
  const stage = el("div", {});
  const caption = el("div", {});
  host.append(bar, stage, caption);

  async function ensureLayers() {
    if (state.layerMeta?.city === state.city) return;
    try {
      state.layerMeta = await api.get(`/api/scenes/${state.city}/layers`);
      state.layerMeta.city = state.city;
    } catch {
      state.layerMeta = null;
    }
  }

  function layerOptions() {
    const groups = new Map();
    for (const l of state.layerMeta?.layers || []) {
      if (!groups.has(l.group)) groups.set(l.group, []);
      groups.get(l.group).push(l);
    }
    return groups;
  }

  function rebuildBar() {
    const controls = [
      select("source", ["scene", "scored tile"], state.source, async (v) => {
        state.source = v;
        if (v === "scored tile" && !state.tiles.length) {
          try {
            state.tiles = (await api.get("/api/evidence")).tiles.map((t) => t.tile_id);
          } catch { state.tiles = []; }
          state.tile = state.tiles[0] || null;
        }
        await refresh();
      }),
    ];

    if (state.source === "scene") {
      controls.push(select("scene", scenes.map((s) => s.city), state.city, async (v) => {
        state.city = v;
        state.layerMeta = null;
        await refresh();
      }));
      const groups = layerOptions();
      controls.push(grouped("left layer", groups, state.left, (v) => { state.left = v; draw(); }));
      controls.push(grouped("right layer", groups, state.right, (v) => { state.right = v; draw(); }));
    } else {
      controls.push(select("tile", state.tiles.length ? state.tiles : ["(none)"], state.tile,
        (v) => { state.tile = v; draw(); }));
    }

    controls.push(select("mode", MODES, state.mode, (v) => { state.mode = v; draw(); }));
    controls.push(sliderField("position", state.position, (v) => {
      state.position = v;
      const stageEl = stage.querySelector("[data-stage]");
      if (stageEl) applyPosition(stageEl, state);
    }));
    bar.replaceChildren(...controls);
  }

  async function refresh() {
    if (state.source === "scene") await ensureLayers();
    rebuildBar();
    draw();
  }

  function draw() {
    const pair = sourcesFor(state);
    if (!pair) {
      stage.replaceChildren(el("div", { class: "notice warn" },
        "Nothing to compare. Run the funnel to produce scored tiles."));
      caption.replaceChildren();
      return;
    }
    stage.replaceChildren(buildStage(pair, state));
    caption.replaceChildren(...describe(state, scenes, pair));
  }

  await refresh();
}

function sourcesFor(state) {
  if (state.source === "scene") {
    if (!state.city) return null;
    const base = `/api/scenes/${state.city}/layers`;
    return {
      before: `${base}/${state.left}`,
      after: `${base}/${state.right}`,
      extra: null,
      label: state.city,
    };
  }
  if (!state.tile) return null;
  return {
    before: `/api/evidence/${state.tile}/before_rgb.png`,
    after: `/api/evidence/${state.tile}/after_rgb.png`,
    extra: `/api/evidence/${state.tile}/change_overlay.png`,
    label: state.tile,
  };
}

function buildStage(pair, state) {
  if (state.mode === "side by side") {
    return el("div", { class: "grid cols-2" },
      figure(pair.before, leftLabel(state)),
      figure(pair.after, rightLabel(state)),
      pair.extra ? figure(pair.extra, "change overlay, as sent to the model") : null);
  }
  if (state.mode === "difference") return differenceStage(pair);

  const wrap = el("div", {
    "data-stage": state.mode,
    style: "position:relative; overflow:hidden; border:1px solid var(--rule);" +
      "border-radius:6px; background:var(--surface-2)",
  },
    el("img", { src: pair.before, alt: leftLabel(state),
      style: "display:block; width:100%; height:auto" }),
    el("img", { src: pair.after, alt: rightLabel(state), "data-top": true,
      style: "position:absolute; inset:0; width:100%; height:100%; object-fit:fill" }),
    el("div", { "data-handle": true, style:
      "position:absolute; top:0; bottom:0; width:2px; background:var(--accent); pointer-events:none" }));
  applyPosition(wrap, state);
  return wrap;
}

function applyPosition(wrap, state) {
  const top = wrap.querySelector("[data-top]");
  const handle = wrap.querySelector("[data-handle]");
  if (!top) return;
  if (state.mode === "blend") {
    top.style.clipPath = "";
    top.style.opacity = String(state.position / 100);
    if (handle) handle.style.display = "none";
    return;
  }
  top.style.opacity = "1";
  top.style.clipPath = `inset(0 0 0 ${state.position}%)`;
  if (handle) {
    handle.style.display = "block";
    handle.style.left = `${state.position}%`;
  }
}

function differenceStage(pair) {
  const canvas = el("canvas", {
    style: "display:block; width:100%; border:1px solid var(--rule); border-radius:6px",
  });
  const note = el("p", { class: "dim" }, "computing residual…");
  const wrap = el("div", {}, canvas, note);

  const a = new Image();
  const b = new Image();
  let loaded = 0;
  const onLoad = () => {
    if (++loaded < 2) return;
    const w = Math.min(a.naturalWidth, b.naturalWidth);
    const h = Math.min(a.naturalHeight, b.naturalHeight);
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(a, 0, 0, w, h);
    const A = ctx.getImageData(0, 0, w, h);
    ctx.drawImage(b, 0, 0, w, h);
    const B = ctx.getImageData(0, 0, w, h);
    const out = ctx.createImageData(w, h);
    const mag = new Float32Array(w * h);
    let peak = 1;
    for (let i = 0, p = 0; i < A.data.length; i += 4, p++) {
      const d = Math.abs(A.data[i] - B.data[i])
        + Math.abs(A.data[i + 1] - B.data[i + 1])
        + Math.abs(A.data[i + 2] - B.data[i + 2]);
      mag[p] = d;
      if (d > peak) peak = d;
    }
    for (let p = 0, i = 0; p < mag.length; p++, i += 4) {
      const t = Math.min(1, (mag[p] / peak) * 1.6);
      out.data[i] = Math.round(255 * t);
      out.data[i + 1] = Math.round(255 * Math.max(0, t - 0.45));
      out.data[i + 2] = Math.round(255 * Math.max(0, t - 0.85));
      out.data[i + 3] = 255;
    }
    ctx.putImageData(out, 0, 0);
    note.textContent =
      `RGB residual of the two rendered layers, scaled to its own maximum ` +
      `(${peak.toFixed(0)} of a possible 765). Brighter is a larger difference.`;
  };
  a.onload = onLoad; b.onload = onLoad;
  a.onerror = b.onerror = () => { note.textContent = "Could not load one of the images."; };
  a.src = pair.before; b.src = pair.after;
  return wrap;
}

function labelFor(state, name) {
  const found = (state.layerMeta?.layers || []).find((l) => l.name === name);
  return found ? found.label : name;
}
const leftLabel = (s) => (s.source === "scene" ? labelFor(s, s.left) : "before");
const rightLabel = (s) => (s.source === "scene" ? labelFor(s, s.right) : "after");

function describe(state, scenes, pair) {
  const out = [];

  if (state.mode === "difference") {
    out.push(el("div", { class: "notice warn" },
      el("strong", {}, "This is not the detector's output. "),
      "It is an RGB residual of two rendered layers, computed in the browser. " +
      "The gate works on calibrated reflectance across six bands against a " +
      "robust per-scene threshold. Use this to orient, not to judge — the " +
      "change mask layer is the detector's actual answer."));
  }

  if (state.source === "scene") {
    const scene = scenes.find((s) => s.city === state.city);
    if (scene) {
      out.push(el("p", { class: "dim" },
        `${scene.city}: ${scene.height} × ${scene.width} px, `,
        `${scene.date_t1 || "?"} → ${scene.date_t2 || "?"}, `,
        chip(scene.split, scene.split === "test" ? "accent" : undefined)));
    }
    for (const note of state.layerMeta?.rendering_notes || []) {
      out.push(el("p", { class: "dim" }, "· ", note));
    }
    if (state.layerMeta?.fingerprint) {
      out.push(el("p", { class: "dim mono" },
        `rendered under config ${state.layerMeta.fingerprint}`));
    }
    out.push(commandLine(`satchangegate run --pair ${state.city}`));
  } else if (state.tile) {
    out.push(el("p", { class: "dim" },
      `${pair.label}: the scored 64 px tile inside a 3× context window, upsampled ` +
      "to a 512 px long edge with nearest-neighbour so the sensor grid stays crisp. " +
      "These are the exact bytes the model received, served rather than re-rendered."));
    out.push(commandLine("satchangegate e2e --split test --vlm --batch"));
  }
  return out;
}

function figure(src, caption) {
  return el("figure", { style: "margin:0" },
    el("img", { src, alt: caption,
      style: "width:100%; border:1px solid var(--rule); border-radius:6px" }),
    el("figcaption", { class: "dim", style: "font-size:11px; margin-top:4px" }, caption));
}

function select(label, options, value, onChange) {
  const sel = el("select", {}, ...options.map((o) =>
    el("option", { value: o, selected: o === value }, o)));
  sel.addEventListener("change", () => onChange(sel.value));
  return el("label", { class: "field" }, el("span", {}, label), sel);
}

function grouped(label, groups, value, onChange) {
  const sel = el("select", {});
  for (const [name, items] of groups) {
    const og = el("optgroup", { label: name });
    for (const item of items) {
      og.append(el("option", { value: item.name, selected: item.name === value }, item.label));
    }
    sel.append(og);
  }
  sel.addEventListener("change", () => onChange(sel.value));
  return el("label", { class: "field" }, el("span", {}, label), sel);
}

function sliderField(label, value, onChange) {
  const input = el("input", { type: "range", min: "0", max: "100", value: String(value) });
  input.addEventListener("input", () => onChange(Number(input.value)));
  return el("label", { class: "field" }, el("span", {}, label), input);
}
