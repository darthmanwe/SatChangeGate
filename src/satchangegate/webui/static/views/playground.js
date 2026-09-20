import { api, el, chip, stat, table, fmt, commandLine, missing, spinner, setStatus } from "../app.js";

export const title = "Thresholds";
export const group = "Lab";

/* Move a threshold, watch the trade move.

   Two honesty constraints are wired into this view rather than written beside
   it. The default population is the *training* split, because optimising
   sliders against held-out rows converts the test set into development data and
   no later assertion can undo that. And the eight thresholds that change the
   change mask are shown separately and greyed: the cached matrix cannot answer
   for them, so the view says so instead of serving a number that looks live. */

let seq = 0;

export async function render(mount) {
  mount.append(el("header", {},
    el("h1", {}, "Threshold playground"),
    el("p", { class: "lede" },
      "Every re-score runs the pipeline's own decide() over a cached feature " +
      "matrix. Nothing here reimplements the gate.")));

  const host = el("div", {}, spinner("loading thresholds…"));
  mount.append(host);

  let meta;
  try {
    meta = await api.get("/api/playground/thresholds");
  } catch (err) {
    host.replaceChildren(el("div", { class: "notice stop" }, String(err.message || err)));
    return;
  }

  const state = { split: "train", overrides: {} };
  host.replaceChildren();

  const top = el("div", { style: "display:flex; gap:12px; align-items:end; flex-wrap:wrap; margin-bottom:14px" },
    el("label", { class: "field", style: "max-width:200px" },
      el("span", {}, "population"),
      (() => {
        const s = el("select", {},
          el("option", { value: "train", selected: true }, "train (development)"),
          el("option", { value: "test" }, "test (held out)"));
        s.addEventListener("change", () => { state.split = s.value; run(); });
        return s;
      })()),
    el("button", { onClick: () => { state.overrides = {}; resetInputs(); run(); } }, "reset to shipped"));
  host.append(top);

  const resultHost = el("div", {});
  host.append(resultHost);

  const decision = meta.thresholds.filter((t) => t.kind === "decision");
  const mask = meta.thresholds.filter((t) => t.kind === "mask");
  const other = meta.thresholds.filter((t) => t.kind === "other");

  const inputs = [];
  function resetInputs() { for (const fn of inputs) fn(); }

  host.append(el("h2", {}, "Decision thresholds"));
  host.append(el("p", { class: "dim" },
    "Read only inside decide(), so the cached matrix answers for them exactly."));
  host.append(el("div", { class: "grid cols-3" },
    ...decision.map((t) => sliderFor(t, state, run, inputs))));

  host.append(el("h2", {}, "Mask thresholds"));
  host.append(el("div", { class: "notice warn" },
    el("strong", {}, "These cannot be answered from cache. "), meta.note, " ",
    el("span", { class: "mono" }, meta.invalidates.join(", ")), " are recomputed."));
  host.append(el("div", { class: "grid cols-3" },
    ...mask.map((t) => el("div", { class: "card" },
      el("div", { class: "k mono", style: "font-size:12px" }, t.name),
      el("div", { class: "mono", style: "font-size:18px" }, String(t.value)),
      el("div", { class: "dim", style: "font-size:11px" },
        "changes the mask — set it in thresholds.yaml and re-run eval")))));

  if (other.length) {
    host.append(el("h2", {}, "Other"));
    host.append(el("div", { class: "grid cols-4" },
      ...other.map((t) => el("div", { class: "card stat" },
        el("span", { class: "k" }, t.name),
        el("span", { class: "v", style: "font-size:15px" }, String(t.value))))));
  }

  async function run() {
    const mine = ++seq;
    resultHost.replaceChildren(spinner("re-scoring…"));
    let res;
    try {
      res = await api.post("/api/playground/score", {
        split: state.split, thresholds: state.overrides,
      });
    } catch (err) {
      if (mine !== seq) return;
      const detail = err.detail && typeof err.detail === "object" ? err.detail : {};
      resultHost.replaceChildren(missing(
        detail.detail || "Cannot re-score here.",
        detail.produced_by || "", String(err.message || err)));
      return;
    }
    // A slower earlier request must never overwrite a later choice.
    if (mine !== seq) return;
    resultHost.replaceChildren(...renderResult(res, state));
    setStatus(undefined, `${res.population.n} rows · ${Object.keys(state.overrides).length} overridden`);
  }

  run();
}

function renderResult(res, state) {
  const m = res.metrics;
  const out = [];

  if (res.test_informed) {
    out.push(el("div", { class: "notice warn" },
      el("strong", {}, "Held-out population. "), res.test_informed));
  }
  if (res.stale) {
    out.push(el("div", { class: "notice stop" },
      el("strong", {}, "Ignored: "),
      el("span", { class: "mono" }, res.ignored_feature_changing.join(", ")), ". ",
      res.stale_note));
  }

  out.push(el("div", { class: "grid cols-4" },
    stat("precision", fmt(m.precision)),
    stat("recall", fmt(m.recall)),
    stat("F1", fmt(m.f1)),
    stat("balanced acc.", fmt(m.balanced_accuracy)),
    stat("candidates", m.n_candidates, `${(m.candidate_rate * 100).toFixed(1)}% of assessable`),
    stat("gate filters", `${m.gate_filtered_pct.toFixed(1)}%`, "of assessable tiles"),
    stat("tier 0 refuses", `${m.tier0_refused_pct.toFixed(1)}%`, `${m.n_refused} tiles`),
    stat("total reduction", `${m.total_reduction_pct.toFixed(1)}%`, "before any API call")));

  out.push(el("h3", {}, "Confusion"));
  out.push(table(
    [
      { label: "", get: (r) => r[0] },
      { label: "predicted change", num: true, get: (r) => r[1] },
      { label: "predicted no change", num: true, get: (r) => r[2] },
    ],
    [["actually change", m.confusion.tp, m.confusion.fn],
     ["actually no change", m.confusion.fp, m.confusion.tn]]));
  out.push(el("p", { class: "dim" },
    `${m.n_scored} scored of ${m.n_total}; ${m.n_refused} refused by the quality ` +
    "tier and excluded from every rate above."));

  out.push(el("h3", {}, "Which rule fired"));
  out.push(table(
    [
      { label: "reason", get: (r) => r[0] },
      { label: "tiles", num: true, get: (r) => r[1] },
    ],
    Object.entries(m.reasons || {})));

  if (res.yaml) {
    out.push(el("h3", {}, "Export"));
    out.push(el("pre", { class: "mono", style: "overflow-x:auto; font-size:12px" }, res.yaml));
    out.push(el("p", { class: "dim" },
      "Same shape as `satchangegate tune` writes, but these were chosen by eye: " +
      "they carry no optimality claim" +
      (state.split === "test" ? ", and they are test-informed." : ".")));
  }
  out.push(commandLine(res.command));
  out.push(el("p", { class: "dim" },
    `Population from ${res.population.source === "sample" ? "the committed sample" : "a local run"}: ` +
    res.population.produced_by));
  return out;
}

function sliderFor(t, state, onChange, inputs) {
  const isInt = t.type === "integer";
  const lo = t.minimum ?? 0;
  const hi = t.maximum ?? guessMax(t.value, isInt);
  const step = isInt ? 1 : Math.max((hi - lo) / 200, 0.001);

  const value = el("span", { class: "mono", style: "font-size:15px" }, String(t.value));
  const input = el("input", {
    type: "range", min: String(lo), max: String(hi), step: String(step),
    value: String(t.value),
  });
  const shipped = t.value;

  input.addEventListener("input", () => {
    const v = isInt ? Math.round(Number(input.value)) : Number(Number(input.value).toFixed(4));
    value.textContent = String(v);
    value.style.color = v === shipped ? "" : "var(--accent)";
    if (v === shipped) delete state.overrides[t.name];
    else state.overrides[t.name] = v;
    onChange();
  });
  inputs.push(() => {
    input.value = String(shipped);
    value.textContent = String(shipped);
    value.style.color = "";
  });

  return el("div", { class: "card" },
    el("div", { style: "display:flex; justify-content:space-between; align-items:baseline; gap:8px" },
      el("span", { class: "k mono", style: "font-size:12px" }, t.name), value),
    input,
    el("div", { class: "dim", style: "font-size:11px" }, `shipped ${shipped}`));
}

function guessMax(value, isInt) {
  if (value === 0) return isInt ? 100 : 1;
  const scaled = Math.abs(value) * 4;
  return isInt ? Math.ceil(scaled) : Number(scaled.toPrecision(2));
}
