import { api, el, chip, commandLine, spinner } from "../app.js";

export const title = "Compare";
export const group = "Explore";

/* Before and after, four ways: swipe, blend, side by side, and a difference
   image computed in the browser.

   The difference view is labelled as what it is -- an RGB residual of two
   display-stretched previews -- and not as the detector's output. They are not
   the same thing and conflating them would flatter the gate: the change mask is
   computed on calibrated reflectance across six bands against a scene-level
   robust threshold, not on two JPEG-ish renderings. */

const MODES = ["swipe", "blend", "side by side", "difference"];

export async function render(mount) {
  mount.append(el("header", {},
    el("h1", {}, "Before and after"),
    el("p", { class: "lede" },
      "Whole scenes, or a single scored tile with the evidence the verifier saw.")));

  const host = el("div", {}, spinner());
  mount.append(host);

  let scenes;
  try {
    scenes = (await api.get("/api/scenes")).scenes;
  } catch (err) {
    host.replaceChildren(el("div", { class: "notice stop" }, String(err.message || err)));
    return;
  }

  const params = new URLSearchParams((location.hash.split("?")[1] || ""));
  const state = {
    source: "scene",
    city: params.get("city") || scenes[0]?.city,
    tile: null,
    mode: "swipe",
    position: 50,
    tiles: [],
  };

  host.replaceChildren();
  const bar = el("div", { class: "grid cols-4", style: "margin-bottom:12px" });
  const stage = el("div", {});
  const caption = el("div", {});
  host.append(bar, stage, caption);

  function rebuildBar() {
    bar.replaceChildren(
      select("source", ["scene", "scored tile"], state.source, async (v) => {
        state.source = v;
        if (v === "scored tile" && !state.tiles.length) {
          try {
            state.tiles = (await api.get("/api/evidence")).tiles.map((t) => t.tile_id);
          } catch { state.tiles = []; }
        }
        state.tile = state.tiles[0] || null;
        rebuildBar(); draw();
      }),
      state.source === "scene"
        ? select("scene", scenes.map((s) => s.city), state.city, (v) => { state.city = v; draw(); })
        : select("tile", state.tiles.length ? state.tiles : ["(none)"], state.tile,
            (v) => { state.tile = v; draw(); }),
      select("mode", MODES, state.mode, (v) => { state.mode = v; draw(); }),
      sliderField("position", state.position, (v) => {
        state.position = v;
        const stageEl = stage.querySelector("[data-stage]");
        if (stageEl) applyPosition(stageEl, state);
      }),
    );
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

  rebuildBar();
  draw();
}

function sourcesFor(state) {
  if (state.source === "scene") {
    if (!state.city) return null;
    return {
      before: `/api/scenes/${state.city}/preview/t1`,
      after: `/api/scenes/${state.city}/preview/t2`,
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
      figure(pair.before, "before"),
      figure(pair.after, "after"),
      pair.extra ? figure(pair.extra, "change overlay (as sent to the model)") : null);
  }
  if (state.mode === "difference") {
    return differenceStage(pair);
  }
  const wrap = el("div", {
    "data-stage": state.mode,
    style: "position:relative; overflow:hidden; border:1px solid var(--rule); border-radius:6px; background:var(--surface-2)",
  },
    el("img", { src: pair.before, alt: "before",
      style: "display:block; width:100%; height:auto" }),
    el("img", { src: pair.after, alt: "after", "data-top": true,
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
    let peak = 1;
    const mag = new Float32Array(w * h);
    for (let i = 0, p = 0; i < A.data.length; i += 4, p++) {
      const d = Math.abs(A.data[i] - B.data[i])
        + Math.abs(A.data[i + 1] - B.data[i + 1])
        + Math.abs(A.data[i + 2] - B.data[i + 2]);
      mag[p] = d;
      if (d > peak) peak = d;
    }
    for (let p = 0, i = 0; p < mag.length; p++, i += 4) {
      const t = mag[p] / peak;
      out.data[i] = Math.round(255 * Math.min(1, t * 1.6));
      out.data[i + 1] = Math.round(255 * Math.min(1, Math.max(0, t * 1.6 - 0.45)));
      out.data[i + 2] = Math.round(255 * Math.min(1, Math.max(0, t * 1.6 - 0.85)));
      out.data[i + 3] = 255;
    }
    ctx.putImageData(out, 0, 0);
    note.replaceChildren(document.createTextNode(
      `RGB residual of the two display renderings, scaled to its own maximum ` +
      `(${peak.toFixed(0)}/765). Brighter is a larger difference.`));
  };
  a.onload = onLoad; b.onload = onLoad;
  a.onerror = b.onerror = () => note.textContent = "Could not load one of the images.";
  a.src = pair.before; b.src = pair.after;
  return wrap;
}

function describe(state, scenes, pair) {
  const out = [];
  if (state.mode === "difference") {
    out.push(el("div", { class: "notice warn" },
      el("strong", {}, "This is not the detector's output. "),
      "It is an RGB residual of two display-stretched previews, computed in the " +
      "browser. The gate works on calibrated reflectance across six bands, " +
      "against a robust per-scene threshold — it sees things this does not, and " +
      "ignores things this shows. Use it to orient, not to judge."));
  }
  if (state.source === "scene") {
    const scene = scenes.find((s) => s.city === state.city);
    if (scene) {
      out.push(el("p", { class: "dim" },
        `${scene.city}: ${scene.height} × ${scene.width} px, `,
        `${scene.date_t1 || "?"} → ${scene.date_t2 || "?"}, `,
        chip(scene.split, scene.split === "test" ? "accent" : undefined)));
      out.push(el("p", { class: "dim" },
        "Both frames use a joint 2–98 percentile stretch computed across the pair, " +
        "so a brightness shift between acquisitions does not read as change."));
      out.push(commandLine(`satchangegate run --pair ${scene.city}`));
    }
  } else if (state.tile) {
    out.push(el("p", { class: "dim" },
      `${pair.label}: the scored 64 px tile inside a 3× context window, upsampled ` +
      "to a 512 px long edge with nearest-neighbour so the sensor grid stays crisp."));
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

function sliderField(label, value, onChange) {
  const input = el("input", { type: "range", min: "0", max: "100", value: String(value) });
  input.addEventListener("input", () => onChange(Number(input.value)));
  return el("label", { class: "field" }, el("span", {}, label), input);
}
