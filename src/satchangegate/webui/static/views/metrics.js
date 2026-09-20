import { api, el, chip, stat, table, fmt, commandLine, missing } from "../app.js";

export const title = "Metrics";
export const group = "Lab";

/* Every number here carries its denominator and its exclusions, because the two
   populations in this repo are easy to confuse: the baselines file scores 621
   tiles and the headline confusion matrix scores 534, the difference being the
   87 the quality tier refused to judge. Putting both on one screen without
   saying so would invite exactly the conflation the 0.3.0 audit corrected. */

export async function render(mount) {
  mount.append(el("header", {},
    el("h1", {}, "Metrics and integrity"),
    el("p", { class: "lede" },
      "Held-out numbers, the guarantee attached to them, and the corrections " +
      "that got them here.")));

  await section(mount, "Held-out gate", "eval_test", (data, cmd) => {
    const m = data.metrics || {};
    const out = [];
    out.push(el("div", { class: "grid cols-4" },
      stat("precision", fmt(m.precision), ci(m.precision_ci95)),
      stat("recall", fmt(m.recall), ci(m.recall_ci95)),
      stat("specificity", fmt(m.specificity), ci(m.specificity_ci95)),
      stat("F1", fmt(m.f1)),
      stat("balanced acc.", fmt(m.balanced_accuracy)),
      stat("scored", m.n ?? "—", `${m.n_positive ?? "—"} change / ${m.n_negative ?? "—"} not`),
      stat("refused by tier 0", data.n_low_quality ?? "—", `${fmt(data.tier0_refused_pct, 1)}% of all tiles`),
      stat("cities scored", (data.cities_scored || []).length,
        `of ${(data.cities || []).length} in the split`)));

    const cm = m.confusion || {};
    out.push(el("h3", {}, "Confusion matrix"));
    out.push(table(
      [
        { label: "", get: (r) => r[0] },
        { label: "predicted change", num: true, get: (r) => r[1] },
        { label: "predicted no change", num: true, get: (r) => r[2] },
      ],
      [["actually change", cm.tp ?? "—", cm.fn ?? "—"],
       ["actually no change", cm.fp ?? "—", cm.tn ?? "—"]]));

    out.push(el("div", { class: "notice" },
      el("strong", {}, "Denominator. "),
      `${m.n} scored tiles, not ${data.n_tiles}. The ${data.n_low_quality} difference is ` +
      "tiles the quality tier refused to judge; they are excluded from every rate " +
      "above rather than counted as correct or incorrect. The scene-level flag " +
      "means a city passes or vanishes whole, which is why " +
      `${(data.cities_scored || []).length} of ${(data.cities || []).length} cities appear.`));

    if (data.pixel_metrics) {
      const p = data.pixel_metrics;
      out.push(el("h3", {}, "Pixel level"));
      out.push(el("div", { class: "grid cols-4" },
        stat("precision", fmt(p.precision)), stat("recall", fmt(p.recall)),
        stat("F1", fmt(p.f1)), stat("IoU", fmt(p.iou))));
      out.push(el("div", { class: "notice warn" },
        el("strong", {}, "This is a triage gate, not a segmentation model. "),
        `Scored over ${(p.n / 1e6).toFixed(2)} M observed pixels against roughly ` +
        "0.50–0.60 F1 for supervised deep models on this benchmark. The tile-level " +
        "number above is the one that matters for deciding whether to spend."));
    }
    out.push(commandLine(cmd));
    return out;
  });

  await section(mount, "Rule gate against learned models", "baselines", (data, cmd) => {
    const out = [];
    out.push(table(
      [
        { label: "model", get: (m) => el("code", {}, m.name) },
        { label: "avg precision", num: true, get: (m) => fmt(m.average_precision) },
        { label: "ROC AUC", num: true, get: (m) => fmt(m.roc_auc) },
        { label: "precision @ gate recall", num: true, get: (m) => fmt(m.precision_at_gate_recall) },
      ],
      data.models || []));
    out.push(prCurves(data));
    out.push(el("div", { class: "notice" },
      el("strong", {}, "Different denominator. "),
      `These score all ${data.n_test} held-out tiles, including the ones the ` +
      "quality tier refused, because a ranking metric needs the whole population. " +
      "The gate table above scores only the assessable subset. The two are not " +
      "interchangeable."));
    out.push(commandLine(cmd));
    return out;
  });

  await section(mount, "The recall guarantee, and where it breaks", "conformal", (data, cmd) => {
    const c = data.calibration || {};
    const v = data.held_out_test || {};
    const pr = data.predictor || {};
    const out = [];

    out.push(el("div", { class: "grid cols-4" },
      stat("lambda", fmt(c.lambda)),
      stat("target FNR", fmt(c.target_fnr_alpha, 2), `${Math.round((1 - c.confidence_delta) * 100)}% confidence`),
      stat("calibration FNR", fmt(c.calibration_fnr, 4), `bound ${fmt(c.calibration_fnr_upper_bound, 4)}`),
      stat("held-out FNR", fmt(v.observed_fnr, 4), `recall ${fmt(v.observed_recall, 4)}`)));

    out.push(el("div", { class: pr.matches_shipped_thresholds ? "notice ok" : "notice warn" },
      el("strong", {}, "Predictor independence. "),
      `The gate was refit on ${(pr.fitted_on_cities || []).length} training cities with the ` +
      `calibration cities withheld: ${(pr.fitted_on_cities || []).join(", ")}. `,
      pr.matches_shipped_thresholds
        ? "That refit reproduced the shipped thresholds exactly, so this guarantee " +
          "applies to the gate this repo actually ships."
        : "It differs from the shipped thresholds, so it certifies the refit gate, " +
          "not the shipped one."));

    out.push(el("h3", {}, `Falsification: ${v.cities_where_it_held} of ${v.cities_tested} held-out cities`));
    out.push(table(
      [
        { label: "city", get: (r) => r[0] },
        { label: "n", num: true, get: (r) => r[1].n },
        { label: "positives", num: true, get: (r) => r[1].n_positive },
        { label: "recall", num: true, get: (r) => fmt(r[1].recall) },
        { label: "within alpha", get: (r) => r[1].held === null ? chip("n/a")
            : r[1].held ? chip("yes", "ok") : chip("no", "stop") },
      ],
      Object.entries(v.per_city || {})));
    out.push(el("div", { class: "notice" }, v.reading || ""));
    out.push(commandLine(cmd));
    return out;
  });

  await section(mount, "What a budget buys", "operating_points", (data, cmd) => {
    const out = [];
    out.push(table(
      [
        { label: "budget", num: true, get: (p) => `$${p.budget_usd.toFixed(2)}` },
        { label: "calls", num: true, get: (p) => p.affordable_calls },
        { label: "threshold", num: true, get: (p) => p.threshold === null ? "none" : fmt(p.threshold, 4) },
        { label: "flagged", num: true, get: (p) => p.n_flagged },
        { label: "unused", num: true, get: (p) => p.unused_calls ?? 0 },
        { label: "recall", num: true, get: (p) => fmt(p.recall) },
        { label: "precision", num: true, get: (p) => fmt(p.precision) },
        { label: "spend", num: true, get: (p) => `$${p.spend_usd.toFixed(4)}` },
      ],
      data.operating_points || []));
    if (data.selection_rule) out.push(el("p", { class: "dim" }, data.selection_rule));
    if (data.not_a_spending_authority) {
      out.push(el("div", { class: "notice warn" },
        el("strong", {}, "Not a spending authority. "), data.not_a_spending_authority));
    }
    out.push(commandLine(cmd));
    return out;
  });

  await section(mount, "Offline control battery", "dev_tests", (data, cmd) => [
    table(
      [
        { label: "check", get: (c) => el("code", {}, c.name) },
        { label: "expected", get: (c) => c.expected },
        { label: "actual", get: (c) => c.actual },
        { label: "", get: (c) => c.passed ? chip("pass", "ok") : chip("fail", "stop") },
      ],
      data.checks || []),
    el("p", { class: "dim" }, `${data.n_passed}/${data.n_checks} passed.`),
    commandLine(cmd),
  ]);

  await section(mount, "Learned scorer model card", "scorer_card", (data, cmd) => [
    el("div", { class: "grid cols-4" },
      stat("model", data.model || "—"),
      stat("avg precision", fmt(data.held_out_metrics?.average_precision)),
      stat("ROC AUC", fmt(data.held_out_metrics?.roc_auc)),
      stat("training tiles", data.n_train ?? "—", `${(data.train_cities || []).length} cities`)),
    el("h3", {}, "Intended use"),
    el("p", { class: "lede" }, data.intended_use || "—"),
    el("h3", {}, "Limitations"),
    el("ul", {}, ...(data.limitations || []).map((l) => el("li", {}, l))),
    commandLine(cmd),
  ]);
}

/* ------------------------------------------------------------------ helpers */

async function section(mount, heading, key, build) {
  mount.append(el("h2", {}, heading));
  const host = el("div", {});
  mount.append(host);
  try {
    const res = await api.get(`/api/artifacts/${key}`);
    host.append(...build(res.data, res.produced_by));
    if (res.source === "sample") {
      host.append(el("p", { class: "dim" },
        "Read from the committed sample, not from a run made on this machine."));
    }
  } catch (err) {
    const detail = err.detail && typeof err.detail === "object" ? err.detail : {};
    host.append(missing(
      detail.detail || "Not computed here.",
      detail.produced_by || "",
      String(err.message || err)));
  }
}

function ci(pair) {
  return Array.isArray(pair) ? `95% CI ${fmt(pair[0])}–${fmt(pair[1])}` : undefined;
}

/** Precision-recall curves, drawn from the full curve the artifact already
 *  carries. No matplotlib in the UI path: the numbers are right there. */
function prCurves(data) {
  const W = 520, H = 320, P = 42;
  const colors = ["var(--ink-3)", "var(--warn)", "var(--accent)"];
  const x = (r) => P + r * (W - P - 12);
  const y = (p) => H - P - p * (H - P - 14);

  const parts = [];
  parts.push(`<rect x="${P}" y="14" width="${W - P - 12}" height="${H - P - 14}"
    fill="none" stroke="var(--rule)"/>`);
  for (let t = 0; t <= 1.0001; t += 0.25) {
    parts.push(`<line x1="${x(t)}" y1="14" x2="${x(t)}" y2="${H - P}" stroke="var(--rule)" stroke-dasharray="2 4"/>`);
    parts.push(`<line x1="${P}" y1="${y(t)}" x2="${W - 12}" y2="${y(t)}" stroke="var(--rule)" stroke-dasharray="2 4"/>`);
    parts.push(`<text x="${x(t)}" y="${H - P + 16}" fill="var(--ink-3)" font-size="10" text-anchor="middle">${t.toFixed(2)}</text>`);
    parts.push(`<text x="${P - 8}" y="${y(t) + 3}" fill="var(--ink-3)" font-size="10" text-anchor="end">${t.toFixed(2)}</text>`);
  }
  const prevalence = data.positive_prevalence_test;
  if (prevalence) {
    parts.push(`<line x1="${P}" y1="${y(prevalence)}" x2="${W - 12}" y2="${y(prevalence)}"
      stroke="var(--stop)" stroke-dasharray="5 3"/>`);
    parts.push(`<text x="${W - 16}" y="${y(prevalence) - 5}" fill="var(--stop)" font-size="10"
      text-anchor="end">chance ${prevalence.toFixed(3)}</text>`);
  }
  (data.models || []).forEach((m, i) => {
    const pts = (m.curve || [])
      .filter((pt) => Number.isFinite(pt.recall) && Number.isFinite(pt.precision))
      .map((pt) => `${x(pt.recall).toFixed(1)},${y(pt.precision).toFixed(1)}`)
      .join(" ");
    parts.push(`<polyline points="${pts}" fill="none" stroke="${colors[i % colors.length]}"
      stroke-width="1.8" stroke-linejoin="round"/>`);
  });
  const op = data.shipped_gate_operating_point;
  if (op) {
    parts.push(`<circle cx="${x(op.recall)}" cy="${y(op.precision)}" r="4"
      fill="var(--accent)" stroke="var(--surface)" stroke-width="1.5"/>`);
    parts.push(`<text x="${x(op.recall) + 8}" y="${y(op.precision) - 6}" fill="var(--ink-2)"
      font-size="10">shipped operating point</text>`);
  }
  parts.push(`<text x="${(W + P) / 2}" y="${H - 6}" fill="var(--ink-3)" font-size="11" text-anchor="middle">recall</text>`);
  parts.push(`<text x="12" y="${H / 2}" fill="var(--ink-3)" font-size="11" text-anchor="middle"
    transform="rotate(-90 12 ${H / 2})">precision</text>`);

  const legend = el("div", { style: "display:flex; gap:14px; flex-wrap:wrap; margin-top:6px" },
    ...(data.models || []).map((m, i) => el("span", { class: "chip" },
      el("span", { style: `width:10px;height:2px;display:inline-block;background:${colors[i % colors.length]}` }),
      m.name)));

  return el("div", {},
    el("div", { class: "scroll" }, el("div", {
      html: `<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${W}px" role="img"
        aria-label="Precision-recall curves">${parts.join("")}</svg>`,
    })),
    legend);
}
