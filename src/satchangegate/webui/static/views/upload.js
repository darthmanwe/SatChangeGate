import { api, el, chip, table, commandLine, spinner, setStatus } from "../app.js";

export const title = "Upload";
export const group = "Explore";

/* Bring your own imagery — under a contract.

   This view asks for more than a file picker does, and the asking is the point.
   Band 4 of an arbitrary raster is not red. Dividing by 10000 is right for
   Sentinel-2 L1C and wrong for almost everything else. Two same-sized rasters of
   different continents pass every shape check. None of that is recoverable from
   the bytes, so it is declared and then verified, and anything unverifiable is
   refused rather than approximated. */

const PRESETS = {
  "Sentinel-2 L1C, 13 bands": {
    band_map: { B01: 1, B02: 2, B03: 3, B04: 4, B05: 5, B06: 6, B07: 7, B08: 8,
      B8A: 9, B09: 10, B10: 11, B11: 12, B12: 13 },
    reflectance_scale: 10000, alignment: "georeferenced", target_resolution_m: 10,
  },
  "Six gate bands only": {
    band_map: { B02: 1, B03: 2, B04: 3, B08: 4, B11: 5, B12: 6 },
    reflectance_scale: 10000, alignment: "georeferenced", target_resolution_m: 10,
  },
  "RGB render (degraded lane)": {
    band_map: { B04: 1, B03: 2, B02: 3 },
    reflectance_scale: 255, alignment: "already_aligned", target_resolution_m: null,
  },
};

let state = { uploadId: null, files: [], pairs: [], preset: "Six gate bands only" };

export async function render(mount, ctx) {
  const contract = await api.get("/api/uploads/contract");

  mount.append(el("header", {},
    el("h1", {}, "Upload imagery"),
    el("p", { class: "lede" },
      "Two images, or many pairs. What the files are has to be declared — none " +
      "of it is recoverable from the bytes, and guessing produces a confident " +
      "answer about the wrong thing.")));

  mount.append(el("h2", {}, "What you have to declare, and why"));
  mount.append(table(
    [
      { label: "requirement", get: (r) => el("strong", {}, r.what) },
      { label: "why", get: (r) => r.why },
    ],
    contract.requirements));

  const limits = contract.limits;
  mount.append(el("p", { class: "dim" },
    `Limits: ${limits.max_file_mb} MB per file, ${limits.max_total_mb} MB total, ` +
    `${limits.max_megapixels} Mpx and ${limits.max_bands} bands per image, ` +
    `${limits.max_files} files. Formats: ${contract.allowed_drivers.join(", ")} — ` +
    "anything that can reference other files or URLs is refused outright. " +
    "Compressed size is not a memory bound, so pixels are checked from the header."));

  /* ---- 1. files -------------------------------------------------------- */
  mount.append(el("h2", {}, "1 · Files"));
  const filesHost = el("div", { class: "card" });
  mount.append(filesHost);
  renderFilePicker(filesHost);

  /* ---- 2. declaration --------------------------------------------------- */
  mount.append(el("h2", {}, "2 · What these images are"));
  const declHost = el("div", { class: "card" });
  mount.append(declHost);
  renderDeclaration(declHost);

  /* ---- 3. pairing ------------------------------------------------------- */
  mount.append(el("h2", {}, "3 · Which is before, which is after"));
  const pairHost = el("div", { class: "card" });
  mount.append(pairHost);
  renderPairing(pairHost);

  /* ---- 4. verify + run -------------------------------------------------- */
  mount.append(el("h2", {}, "4 · Verify, then run"));
  const runHost = el("div", { class: "card" });
  mount.append(runHost);
  renderActions(runHost, ctx);
}

function renderFilePicker(host) {
  const input = el("input", { type: "file", multiple: true,
    accept: ".tif,.tiff,.png,.jpg,.jpeg" });
  const status = el("div", { class: "dim", style: "margin-top:8px" }, "No files staged.");

  input.addEventListener("change", async () => {
    if (!input.files.length) return;
    status.replaceChildren(spinner("uploading…"));
    const body = new FormData();
    for (const file of input.files) body.append("files", file);
    try {
      const res = await fetch("/api/uploads", {
        method: "POST",
        headers: { "X-SCG-Token": document.querySelector('meta[name="scg-token"]').content },
        body,
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const data = await res.json();
      state.uploadId = data.upload_id;
      state.files = data.files;
      status.replaceChildren(
        el("p", {}, `${data.files.length} file(s) staged.`),
        table(
          [
            { label: "your name", get: (f) => f.client_name },
            { label: "stored as", get: (f) => el("code", {}, f.stored) },
          ],
          data.files),
        el("p", { class: "dim" }, data.note));
      document.dispatchEvent(new CustomEvent("scg-files-staged"));
    } catch (err) {
      status.replaceChildren(el("div", { class: "notice stop" },
        el("strong", {}, "Refused. "), String(err.message || err)));
    }
  });

  host.replaceChildren(
    el("label", { class: "field" },
      el("span", {}, "images"), input,
      el("span", { class: "hint" },
        "Nothing is decoded on upload beyond the header checks the limits need.")),
    status);
}

function renderDeclaration(host) {
  const select = el("select", {}, ...Object.keys(PRESETS).map((k) =>
    el("option", { value: k, selected: k === state.preset }, k)));
  const detail = el("pre", { class: "mono", style: "overflow-x:auto; font-size:12px" });
  const note = el("div", {});

  const paint = () => {
    const preset = PRESETS[state.preset];
    detail.textContent = JSON.stringify(preset, null, 2);
    note.replaceChildren(
      state.preset.includes("RGB")
        ? el("div", { class: "notice warn" },
            el("strong", {}, "This is the degraded lane. "),
            "An 8-bit RGB render is not reflectance, and NDVI needs near-infrared " +
            "while NDBI needs short-wave infrared. The gate will refuse and say so. " +
            "You still get structural evidence — SSIM, a perceptual hash distance, " +
            "a change-vector magnitude — and you can still ask the vision model, " +
            "which only ever sees pictures anyway.")
        : el("div", { class: "notice" },
            el("strong", {}, "Full lane. "),
            "All six bands the gate needs are declared, so it will produce a real " +
            "decision with all eighteen features."));
  };
  select.addEventListener("change", () => { state.preset = select.value; paint(); });
  paint();

  host.replaceChildren(
    el("label", { class: "field" }, el("span", {}, "preset"), select),
    el("details", {}, el("summary", { class: "dim" }, "declaration"), detail),
    note);
}

function renderPairing(host) {
  const paint = () => {
    if (!state.files.length) {
      host.replaceChildren(el("p", { class: "dim" }, "Stage some files first."));
      return;
    }
    const options = state.files.map((f) => f.stored);
    const rows = el("div", { class: "grid" });
    state.pairs = state.pairs.length ? state.pairs : [{ key: "pair_000", t1: options[0], t2: options[1] || options[0] }];

    const draw = () => {
      rows.replaceChildren(...state.pairs.map((pair, i) =>
        el("div", { class: "grid cols-4" },
          field("key", textInput(pair.key, (v) => { pair.key = v; })),
          field("before (t1)", picker(options, pair.t1, (v) => { pair.t1 = v; })),
          field("after (t2)", picker(options, pair.t2, (v) => { pair.t2 = v; })),
          el("label", { class: "field" }, el("span", {}, " "),
            el("button", { onClick: () => { state.pairs.splice(i, 1); draw(); } }, "remove")))));
    };
    draw();

    host.replaceChildren(
      el("p", { class: "dim" },
        "Pairing is explicit. Sorting filenames and pairing neighbours would " +
        "mis-pair the moment someone uploads three files for two sites, and a " +
        "mis-paired before/after is not detectable downstream — it just answers " +
        "about the wrong comparison."),
      rows,
      el("button", { style: "margin-top:8px", onClick: () => {
        state.pairs.push({
          key: `pair_${String(state.pairs.length).padStart(3, "0")}`,
          t1: options[0], t2: options[1] || options[0],
        });
        draw();
      } }, "add a pair"));
  };
  document.addEventListener("scg-files-staged", paint);
  paint();
}

function renderActions(host, ctx) {
  const out = el("div", {});
  const spendOn = ctx.capabilities?.spend?.allowed;

  const verify = el("button", { class: "primary", onClick: async () => {
    if (!state.uploadId) {
      out.replaceChildren(el("div", { class: "notice warn" }, "Stage some files first."));
      return;
    }
    out.replaceChildren(spinner("verifying…"));
    try {
      const res = await api.post(`/api/uploads/${state.uploadId}/inspect`, manifest());
      out.replaceChildren(...renderInspection(res));
    } catch (err) {
      out.replaceChildren(el("div", { class: "notice stop" },
        el("strong", {}, "Refused. "), String(err.message || err)));
    }
  } }, "Verify without running");

  const vlmBox = el("input", { type: "checkbox" });
  vlmBox.style.width = "auto";
  vlmBox.disabled = !spendOn;

  const run = el("button", { onClick: async () => {
    if (!state.uploadId) return;
    out.replaceChildren(spinner("queueing…"));
    try {
      const res = await api.post(`/api/uploads/${state.uploadId}/run`, {
        ...manifest(), pair: state.pairs[0]?.key, vlm: vlmBox.checked,
      });
      out.replaceChildren(
        el("p", { class: "dim mono" }, `run ${res.run_id}`),
        commandLine(res.command),
        el("p", { class: "dim" },
          "Watch it in the Console, or copy the command above — it reproduces " +
          "this exactly, which is the point of asking you to declare all of it."));
      setStatus(undefined, `upload run ${res.run_id} queued`);
    } catch (err) {
      out.replaceChildren(el("div", { class: "notice stop" },
        el("strong", {}, "Refused. "), String(err.message || err)));
    }
  } }, "Run the first pair");

  host.replaceChildren(
    el("div", { style: "display:flex; gap:10px; align-items:center; flex-wrap:wrap" },
      verify, run,
      el("label", { style: "display:flex; gap:6px; align-items:center; font-size:12px" },
        vlmBox, spendOn ? "also verify with the vision model (paid)"
          : "verification disabled: server started without --allow-spend")),
    out);
}

function manifest() {
  return { declaration: PRESETS[state.preset], pairs: state.pairs };
}

function renderInspection(res) {
  const out = [];
  out.push(el("p", {},
    `${res.n_usable} of ${res.n_pairs} pair(s) verified. `,
    chip(`full lane: ${res.lanes.full}`, res.lanes.full ? "ok" : undefined), " ",
    chip(`degraded: ${res.lanes.rgb_only}`, res.lanes.rgb_only ? "warn" : undefined)));

  out.push(table(
    [
      { label: "pair", get: (p) => el("code", {}, p.key) },
      { label: "", get: (p) => p.ok ? chip(p.capability.lane, p.capability.lane === "full" ? "ok" : "warn") : chip("refused", "stop") },
      { label: "size", num: true, get: (p) => p.ok ? `${p.width}×${p.height}` : "—" },
      { label: "valid", num: true, get: (p) => p.ok ? `${(p.valid_fraction * 100).toFixed(1)}%` : "—" },
      { label: "CRS", get: (p) => p.ok ? (p.crs || "none") : "—" },
      { label: "detail", get: (p) => p.ok
          ? el("span", { class: "dim" }, (p.warnings || []).join(" ") || "no notes")
          : el("span", { class: "chip stop" }, p.refused) },
    ],
    res.pairs));

  for (const pair of res.pairs.filter((p) => p.ok && p.capability.lane !== "full")) {
    out.push(el("div", { class: "notice warn" },
      el("strong", {}, `${pair.key}: degraded lane. `),
      `Missing for the gate: ${pair.capability.missing_for_gate.join(", ")}. `,
      pair.capability.explanation));
  }
  return out;
}

function field(label, control) {
  return el("label", { class: "field" }, el("span", {}, label), control);
}

function textInput(value, onChange) {
  const i = el("input", { type: "text", value });
  i.addEventListener("input", () => onChange(i.value));
  return i;
}

function picker(options, value, onChange) {
  const s = el("select", {}, ...options.map((o) =>
    el("option", { value: o, selected: o === value }, o)));
  s.addEventListener("change", () => onChange(s.value));
  return s;
}
