import { api, el, chip, stat, commandLine, missing, table, fmt } from "../app.js";

export const title = "Overview";
export const group = "";

export async function render(mount, ctx) {
  const caps = ctx.capabilities || (await api.get("/api/capabilities"));
  mount.append(
    el("header", {},
      el("h1", {}, "What this installation can do"),
      el("p", { class: "lede" },
        "Every panel in this UI shows the command that produces it. Anything not " +
        "computed here says so, and names the command that would compute it, " +
        "rather than rendering a blank or borrowing a number from somewhere else.")),
  );

  /* ---- what is present ------------------------------------------------- */
  const present = caps.present || {};
  const labels = {
    dataset: ["OSCD imagery", "satchangegate download-oscd"],
    "baseline-extra": ["scikit-learn + matplotlib", 'pip install -e ".[baseline]"'],
    "api-key": ["ANTHROPIC_API_KEY", "set it in .env"],
    ledger: ["Run ledger", "satchangegate e2e --split test"],
  };
  mount.append(el("h2", {}, "Present"));
  mount.append(el("div", { class: "grid cols-4" },
    ...Object.entries(labels).map(([key, [label, how]]) =>
      el("div", { class: "card stat" },
        el("span", { class: "k" }, label),
        el("span", { class: "v" }, present[key] ? "yes" : "no"),
        el("span", { class: "n" }, present[key] ? `${caps.n_scenes || ""} scenes`.trim() || "available" : how)))));

  /* ---- spend ------------------------------------------------------------ */
  const spend = caps.spend || {};
  mount.append(el("h2", {}, "Spend"));
  mount.append(el("div", { class: spend.allowed ? "notice warn" : "notice" },
    el("strong", {}, spend.allowed
      ? `Paid calls are enabled, capped at $${Number(spend.cap_usd).toFixed(2)} for this session. `
      : "Paid calls are disabled for this session. "),
    spend.note));

  /* ---- headline funnel, if a run exists --------------------------------- */
  mount.append(el("h2", {}, "Last measured funnel"));
  try {
    const { data, produced_by, source } = await api.get("/api/artifacts/e2e_test");
    const c = data.funnel_cost || {};
    const m = data.gate_metrics || {};
    mount.append(el("div", { class: "grid cols-4" },
      stat("input tiles", c.n_pairs ?? "—"),
      stat("tier 0 refused", c.n_tier0_refused ?? "—", `${fmt(c.tier0_refused_pct, 1)}% of input`),
      stat("gate filtered", c.n_gate_filtered ?? "—", `${fmt(c.gate_filtered_pct, 1)}% of assessable`),
      stat("total reduction", `${fmt(c.total_reduction_pct, 1)}%`, "before any API call"),
      stat("verifications", c.n_vlm_calls ?? "—", `${fmt(c.vlm_call_rate_pct, 1)}% of input`),
      stat("measured spend", `$${fmt(c.cost_usd?.total, 4)}`, `$${fmt(c.cost_usd?.mean_per_vlm_call, 6)}/call`),
      stat("gate precision", fmt(m.precision), ciNote(m.precision_ci95)),
      stat("gate recall", fmt(m.recall), ciNote(m.recall_ci95))));
    if (c.savings_pct_note) {
      mount.append(el("div", { class: "notice warn" },
        el("strong", {}, "Read the saving carefully. "), c.savings_pct_note));
    }
    mount.append(commandLine(produced_by));
    if (source === "sample") {
      mount.append(el("p", { class: "dim" },
        "From the committed sample, not a run made here."));
    }
  } catch (err) {
    mount.append(missing("No funnel run on this machine.",
      "satchangegate e2e --split test", String(err.message || err)));
  }

  /* ---- operations ------------------------------------------------------- */
  mount.append(el("h2", {}, "Operations"));
  mount.append(el("p", { class: "lede" },
    "Availability is computed from what is actually on disk, so nothing here " +
    "offers a button that cannot run."));
  mount.append(table(
    [
      { label: "operation", get: (o) => el("code", {}, o.name) },
      { label: "", get: (o) => o.spends_money ? chip("spends", "warn") : "" },
      { label: "speed", get: (o) => chip(o.speed) },
      { label: "status", get: (o) => o.available ? chip("ready", "ok")
          : chip(`blocked: ${o.blocked_by.join(", ")}`, "stop") },
      { label: "what it does", get: (o) => o.summary },
    ],
    caps.operations || [],
  ));
}

function ciNote(ci) {
  return Array.isArray(ci) ? `95% CI ${fmt(ci[0])}–${fmt(ci[1])}` : undefined;
}
