import { api, el, chip, stat, commandLine, missing, spinner, setStatus, fmt } from "../app.js";

export const title = "Funnel";
export const group = "Explore";

/* The funnel as it fills, and what it costs while it does.

   Progress here comes from events the server persisted before streaming them,
   not from tailing the run ledger. That distinction is the whole reason this
   view works: the expensive gate pass completes in full before a single ledger
   row is written, so a tailer would show minutes of nothing and then a burst of
   five hundred rows. */

const STAGES = [
  { key: "input", label: "Input tiles", note: "everything in the split" },
  { key: "tier0", label: "Refused by Tier 0", note: "unusable imagery, not judged" },
  { key: "filtered", label: "Filtered by the gate", note: "judged, and found unchanged" },
  { key: "candidates", label: "Candidates", note: "forwarded for verification" },
  { key: "verified", label: "Verified", note: "what the budget actually bought" },
];

let source = null;

export async function render(mount, ctx) {
  mount.append(el("header", {},
    el("h1", {}, "The funnel"),
    el("p", { class: "lede" },
      "Four tiers, each cheaper than the one after it. The point is not what " +
      "the gate catches — it is what it stops you paying to look at.")));

  const host = el("div", {});
  mount.append(host);

  const live = el("div", { class: "card", style: "margin-bottom:16px" });
  host.append(live);
  renderLauncher(live, ctx);

  host.append(el("h2", {}, "Last recorded run"));
  const recorded = el("div", {});
  host.append(recorded);
  await renderRecorded(recorded);
}

/* ------------------------------------------------------------------- live -- */

function renderLauncher(host, ctx) {
  const caps = ctx.capabilities || {};
  const e2e = (caps.operations || []).find((o) => o.name === "e2e") || {};
  const spendOn = caps.spend?.allowed;

  const state = { n: 120, vlm: false, split: "test" };

  const controls = el("div", { class: "grid cols-4" },
    field("split", (() => {
      const s = el("select", {},
        el("option", { value: "test", selected: true }, "test (held out)"),
        el("option", { value: "train" }, "train"));
      s.addEventListener("change", () => { state.split = s.value; });
      return s;
    })()),
    field("tiles (blank = all)", (() => {
      const i = el("input", { type: "number", min: "1", value: "120" });
      i.addEventListener("input", () => { state.n = i.value ? Number(i.value) : null; });
      return i;
    })()),
    field("verification", (() => {
      const wrap = el("div", { style: "display:flex; align-items:center; gap:6px" });
      const c = el("input", { type: "checkbox" });
      c.style.width = "auto";
      c.disabled = !spendOn;
      c.addEventListener("change", () => { state.vlm = c.checked; });
      wrap.append(c, el("span", { class: "dim", style: "font-size:12px" },
        spendOn ? "call the vision model (paid)" : "disabled: server started without --allow-spend"));
      return wrap;
    })()),
    el("label", { class: "field" }, el("span", {}, " "),
      el("button", { class: "primary", onClick: () => launch(host, state) }, "Run the funnel")));

  host.replaceChildren(
    el("h3", {}, "Run it now"),
    el("p", { class: "dim" },
      e2e.summary || "Run the funnel over a split and measure its cost."),
    controls,
    el("div", { id: "live-progress" }));
}

async function launch(host, state) {
  const slot = host.querySelector("#live-progress");
  slot.replaceChildren(spinner("queueing…"));
  let res;
  try {
    res = await api.post("/api/runs", {
      operation: "e2e",
      params: { split: state.split, n: state.n || null, vlm: state.vlm },
    });
  } catch (err) {
    slot.replaceChildren(el("div", { class: "notice stop" },
      el("strong", {}, "Refused. "), String(err.message || err)));
    return;
  }
  watch(slot, res.run_id, res.command);
}

function watch(slot, runId, command) {
  const bars = new Map();
  const counters = {
    tiles: 0, cities: 0, citiesDone: 0, candidates: 0, verified: 0, cost: 0,
  };

  const headline = el("div", { class: "grid cols-4" });
  const log = el("div", {
    class: "mono",
    style: "max-height:180px; overflow:auto; font-size:11px; margin-top:10px;" +
      "border-top:1px solid var(--rule); padding-top:8px",
  });
  const bar = el("div", {
    style: "height:6px; background:var(--surface-3); border-radius:3px; overflow:hidden; margin:10px 0",
  }, el("div", { "data-fill": true, style: "height:100%; width:0%; background:var(--accent); transition:width .3s" }));

  slot.replaceChildren(
    el("p", { class: "dim mono" }, `run ${runId}`),
    headline, bar, log, commandLine(command));

  const update = () => {
    headline.replaceChildren(
      stat("cities", `${counters.citiesDone}/${counters.cities || "?"}`),
      stat("tiles", counters.tiles || "—"),
      stat("candidates", counters.candidates),
      stat("spend", `$${counters.cost.toFixed(4)}`,
        counters.verified ? `${counters.verified} verified` : "nothing paid for yet"));
    const pct = counters.cities ? (counters.citiesDone / counters.cities) * 100 : 0;
    bar.querySelector("[data-fill]").style.width = `${pct}%`;
  };
  update();

  const note = (text, kind) => {
    log.prepend(el("div", { class: kind === "warn" ? "chip warn" : "" }, text));
  };

  if (source) source.close();
  source = new EventSource(`/api/runs/${runId}/stream`);

  source.addEventListener("indexed", (e) => {
    const d = JSON.parse(e.data);
    counters.tiles = d.n_tiles;
    note(`indexed ${d.n_tiles} tiles in ${d.split}${d.resumed ? `, ${d.resumed} already done` : ""}`);
    update();
  });
  source.addEventListener("gate_pass_started", (e) => {
    const d = JSON.parse(e.data);
    counters.cities = d.n_cities;
    note(`gate pass over ${d.n_cities} cities`);
    update();
  });
  source.addEventListener("city_started", (e) => {
    const d = JSON.parse(e.data);
    note(`  ${d.city}: building scene cache (${d.index}/${d.of})`);
  });
  source.addEventListener("city_finished", (e) => {
    const d = JSON.parse(e.data);
    counters.citiesDone += 1;
    counters.candidates = d.candidates;
    note(`  ${d.city}: done — ${d.done}/${d.total} tiles, ${d.candidates} candidates so far`);
    update();
  });
  source.addEventListener("selected", (e) => {
    const d = JSON.parse(e.data);
    note(`selected ${d.n_selected} of ${d.n_candidates} candidates (${d.strategy}` +
      `${d.cap ? `, cap ${d.cap}` : ""})`);
  });
  source.addEventListener("verified", (e) => {
    const d = JSON.parse(e.data);
    counters.verified = d.done;
    counters.cost = d.cumulative_cost_usd || 0;
    note(`  verified ${d.tile_id}: ${d.verdict} (${d.done}/${d.total})`);
    update();
  });
  source.addEventListener("finished", (e) => {
    const d = JSON.parse(e.data);
    note(`finished: ${d.status} in ${d.duration_s}s`, d.status === "failed" ? "warn" : null);
  });
  source.addEventListener("done", async () => {
    source.close();
    source = null;
    setStatus(undefined, `run ${runId} complete`);
    try {
      const run = await api.get(`/api/runs/${runId}`);
      slot.append(renderRunResult(run));
    } catch { /* the log above already says what happened */ }
  });
  source.onerror = () => {
    note("stream interrupted — the run continues; reload to resume from where it left off", "warn");
  };
}

function renderRunResult(run) {
  const data = run.result?.data || {};
  const cost = data.funnel_cost || {};
  if (!cost.n_pairs) {
    return el("div", { class: "notice" },
      run.status === "failed"
        ? el("span", {}, el("strong", {}, "Failed. "), run.error || "")
        : "Finished.");
  }
  return el("div", {},
    el("h3", {}, "Result"),
    funnelBars(cost),
    el("p", { class: "dim" },
      `Wrote into its own run directory, not the benchmark: ${run.request.out}`));
}

/* --------------------------------------------------------------- recorded -- */

async function renderRecorded(host) {
  let res;
  try {
    res = await api.get("/api/artifacts/e2e_test");
  } catch (err) {
    const detail = err.detail && typeof err.detail === "object" ? err.detail : {};
    host.replaceChildren(missing(
      "No funnel run recorded on this machine.",
      detail.produced_by || "satchangegate e2e --split test --vlm --batch"));
    return;
  }
  const data = res.data;
  const cost = data.funnel_cost || {};
  const verdicts = data.vlm_verdicts || {};

  host.replaceChildren(
    funnelBars(cost),
    el("div", { class: "grid cols-4", style: "margin-top:12px" },
      stat("measured spend", `$${fmt(cost.cost_usd?.total, 4)}`,
        `$${fmt(cost.cost_usd?.mean_per_vlm_call, 6)} per verification`),
      stat("review everything", `$${fmt(cost.counterfactual_review_everything_usd, 3)}`,
        "if nothing were filtered"),
      stat("batch saving", `$${fmt(cost.batch_saving_usd, 3)}`,
        `${fmt(cost.batch_saving_pct, 1)}% — independent of the gate`),
      stat("errors", cost.n_errors ?? "—", "calls that did not return")),

    el("h3", {}, "What the verifier said"),
    el("div", { style: "display:flex; gap:8px; flex-wrap:wrap" },
      ...Object.entries(verdicts).map(([k, v]) =>
        chip(`${k.replace(/_/g, " ")}: ${v}`,
          k === "real_change" ? "ok" : k === "likely_artifact" ? "stop" : "warn"))),

    cost.savings_pct_note
      ? el("div", { class: "notice warn" },
          el("strong", {}, "Read the saving carefully. "), cost.savings_pct_note)
      : null,
    commandLine(res.produced_by));
}

function funnelBars(cost) {
  const total = cost.n_pairs || 1;
  const values = {
    input: cost.n_pairs,
    tier0: cost.n_tier0_refused,
    filtered: cost.n_gate_filtered,
    candidates: cost.n_candidates,
    verified: cost.n_vlm_calls,
  };
  return el("div", { class: "grid" },
    ...STAGES.map((s) => {
      const n = values[s.key] ?? 0;
      const pct = (n / total) * 100;
      return el("div", {},
        el("div", { style: "display:flex; justify-content:space-between; font-size:12px" },
          el("span", {}, s.label, " ", el("span", { class: "dim" }, `· ${s.note}`)),
          el("span", { class: "mono" }, `${n} (${pct.toFixed(1)}%)`)),
        el("div", { style: "height:14px; background:var(--surface-3); border-radius:3px; overflow:hidden; margin-top:3px" },
          el("div", { style: `height:100%; width:${Math.max(pct, 0.6)}%; background:${barColour(s.key)}` })));
    }));
}

function barColour(key) {
  return {
    input: "var(--ink-3)",
    tier0: "var(--warn)",
    filtered: "var(--accent)",
    candidates: "var(--ok)",
    verified: "var(--stop)",
  }[key] || "var(--ink-3)";
}

function field(label, control) {
  return el("label", { class: "field" }, el("span", {}, label), control);
}
