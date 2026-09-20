import { api, el, chip, table, fmt, commandLine, missing, spinner } from "../app.js";

export const title = "Evidence";
export const group = "Explore";

/* What the verifier was actually shown, beside what the gate had said and what
   the label says. Three things this view refuses to do:

   - invent the verdict fields the recorded run never persisted;
   - serve the unredacted gate features that sit beside the metadata on disk;
   - pretend the 221 packaged candidates are the 100 that were verified.        */

let cache = null;

export async function render(mount) {
  mount.append(el("header", {},
    el("h1", {}, "Evidence and verdicts"),
    el("p", { class: "lede" },
      "Each package is exactly what the vision tier received: before, after, a " +
      "change overlay, and metadata with the gate's own answer deliberately " +
      "withheld.")));

  const host = el("div", {}, spinner("loading packages…"));
  mount.append(host);

  let data;
  try {
    data = cache || (cache = await api.get("/api/evidence"));
  } catch (err) {
    host.replaceChildren(missing("No evidence packages on this machine.",
      "satchangegate e2e --split test", String(err.message || err)));
    return;
  }
  if (!data.tiles.length) {
    host.replaceChildren(missing("No evidence packages on this machine.",
      "satchangegate e2e --split test"));
    return;
  }

  const verified = data.tiles.filter((t) => t.vlm_called);
  host.replaceChildren();

  host.append(el("div", { class: "grid cols-4" },
    box("packaged candidates", data.n_packages),
    box("verified", verified.length, "the rest were outside the call budget"),
    box("ledger rows", data.n_ledger_rows, "every tile, not just candidates"),
    box("agreed with the gate", verified.filter((t) => t.vlm_verdict === "real_change").length,
      "verdict real_change")));

  host.append(el("div", { class: "notice warn" },
    el("strong", {}, "Three verdict fields were never persisted. "),
    data.not_retained_note, " ",
    el("span", { class: "mono" }, data.not_retained.join(", "))));

  /* ---- filters ---------------------------------------------------------- */
  const filters = { city: "", verdict: "", outcome: "" };
  const cities = [...new Set(data.tiles.map((t) => t.city).filter(Boolean))].sort();
  const bar = el("div", { class: "grid cols-4", style: "margin:14px 0" },
    picker("city", ["", ...cities], (v) => { filters.city = v; draw(); }),
    picker("verdict", ["", "real_change", "likely_artifact", "uncertain", "(not verified)"],
      (v) => { filters.verdict = v; draw(); }),
    picker("outcome", ["", "true positive", "false positive", "true negative", "false negative"],
      (v) => { filters.outcome = v; draw(); }),
    el("label", { class: "field" }, el("span", {}, " "),
      el("button", { onClick: () => { Object.keys(filters).forEach((k) => (filters[k] = ""));
        for (const s of bar.querySelectorAll("select")) s.value = ""; draw(); } }, "clear")));
  host.append(bar);

  const listHost = el("div", {});
  const detailHost = el("div", { class: "card" },
    el("p", { class: "dim" }, "Pick a tile to see what the model was shown."));
  host.append(el("div", { class: "grid cols-2" }, listHost, detailHost));

  function outcomeOf(t) {
    if (t.gate === "low_quality") return "refused";
    const flagged = t.gate === "candidate_change";
    if (t.label === 1) return flagged ? "true positive" : "false negative";
    return flagged ? "false positive" : "true negative";
  }

  function draw() {
    const rows = data.tiles.filter((t) => {
      if (filters.city && t.city !== filters.city) return false;
      if (filters.verdict === "(not verified)" && t.vlm_called) return false;
      if (filters.verdict && filters.verdict !== "(not verified)"
        && t.vlm_verdict !== filters.verdict) return false;
      if (filters.outcome && outcomeOf(t) !== filters.outcome) return false;
      return true;
    });
    listHost.replaceChildren(
      el("p", { class: "dim" }, `${rows.length} of ${data.tiles.length} packages`),
      table(
        [
          { label: "tile", get: (t) => el("code", {}, t.tile_id) },
          { label: "gate", get: (t) => chip(t.gate === "candidate_change" ? "flagged" : t.gate,
              t.gate === "candidate_change" ? "accent" : undefined) },
          { label: "conf", num: true, get: (t) => fmt(t.gate_confidence) },
          { label: "truth", get: (t) => t.label === 1 ? chip("change", "ok") : chip("none") },
          { label: "verdict", get: (t) => t.vlm_called
              ? chip(t.vlm_verdict || "error", verdictKind(t.vlm_verdict))
              : el("span", { class: "dim" }, "not verified") },
        ],
        rows.slice(0, 300),
        { onRow: (t) => showTile(detailHost, t, outcomeOf(t)) }),
      rows.length > 300 ? el("p", { class: "dim" }, "Showing the first 300.") : null);
  }
  draw();
}

async function showTile(host, row, outcome) {
  host.replaceChildren(spinner());
  let detail;
  try {
    detail = await api.get(`/api/evidence/${row.tile_id}`);
  } catch (err) {
    host.replaceChildren(el("div", { class: "notice stop" }, String(err.message || err)));
    return;
  }
  const imgs = ["before_rgb.png", "after_rgb.png", "change_overlay.png"];
  host.replaceChildren(
    el("h3", {}, el("code", {}, row.tile_id), " ", chip(outcome,
      outcome.startsWith("true") ? "ok" : outcome === "refused" ? undefined : "stop")),
    el("div", { class: "grid cols-3" },
      ...imgs.map((name) => el("figure", { style: "margin:0" },
        el("img", { src: `/api/evidence/${row.tile_id}/${name}`, alt: name,
          style: "width:100%; border-radius:4px; border:1px solid var(--rule)" }),
        el("figcaption", { class: "dim", style: "font-size:11px" },
          name.replace("_rgb.png", "").replace(".png", ""))))),
    el("p", { class: "dim", style: "margin-top:4px" },
      "The cyan box is the 64 px tile that was scored. The surrounding frame is " +
      "3x context, which measurably changed what the model could resolve."),

    el("h3", {}, "What the gate said"),
    table([{ label: "field", get: (r) => r[0] }, { label: "value", get: (r) => r[1] }],
      [["decision", row.gate], ["reason", row.gate_reason || "—"],
       ["confidence", fmt(row.gate_confidence)], ["ground truth", row.label === 1 ? "change" : "no change"]]),

    el("h3", {}, "What the verifier said"),
    row.vlm_called
      ? table([{ label: "field", get: (r) => r[0] }, { label: "value", get: (r) => r[1] }],
          [["verdict", row.vlm_verdict || "—"], ["change type", row.vlm_change_type || "—"],
           ["confidence", fmt(row.vlm_confidence)],
           ["cost", row.cost_usd != null ? `$${Number(row.cost_usd).toFixed(6)}` : "—"],
           ["artifact_risk", notRetained()], ["visual_evidence", notRetained()],
           ["requires_human_review", notRetained()]])
      : el("p", { class: "dim" },
          "Not verified. It was packaged as a candidate but fell outside the call " +
          "budget, so no verdict exists — which is not the same as a negative one."),

    el("h3", {}, "Metadata the model was shown"),
    el("pre", { class: "mono", style: "overflow-x:auto; font-size:12px" },
      JSON.stringify(detail.metadata_shown_to_model, null, 2)),
    el("div", { class: "notice" }, el("strong", {}, "Redaction. "), detail.redaction_note),
    commandLine("satchangegate e2e --split test --vlm --batch --sample stratified"),
  );
}

function notRetained() {
  return el("span", { class: "chip warn" }, "not retained");
}

function verdictKind(v) {
  if (v === "real_change") return "ok";
  if (v === "likely_artifact") return "stop";
  return "warn";
}

function box(k, v, n) {
  return el("div", { class: "card stat" },
    el("span", { class: "k" }, k), el("span", { class: "v" }, v),
    n ? el("span", { class: "n" }, n) : null);
}

function picker(label, options, onChange) {
  const sel = el("select", {}, ...options.map((o) =>
    el("option", { value: o }, o === "" ? `all ${label}` : o)));
  sel.addEventListener("change", () => onChange(sel.value));
  return el("label", { class: "field" }, el("span", {}, label), sel);
}
