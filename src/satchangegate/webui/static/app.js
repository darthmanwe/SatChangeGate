/* Shell: routing, the API client, and the shared pieces every view uses.
   Views live in ./views/ and export { title, group, render(mount, ctx) }. */

const TOKEN = document.querySelector('meta[name="scg-token"]')?.content || "";

/* ----------------------------------------------------------------- api --- */

export const api = {
  async get(path) {
    const res = await fetch(path, { headers: { Accept: "application/json" } });
    if (!res.ok) {
      let detail;
      try { detail = (await res.json()).detail; } catch { detail = res.statusText; }
      throw new ApiError(detail, res.status);
    }
    return res.json();
  },
  async post(path, body) {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-SCG-Token": TOKEN },
      body: JSON.stringify(body ?? {}),
    });
    if (!res.ok) {
      let detail;
      try { detail = (await res.json()).detail; } catch { detail = res.statusText; }
      throw new ApiError(detail, res.status);
    }
    return res.json();
  },
};

export class ApiError extends Error {
  constructor(detail, status) {
    super(typeof detail === "string" ? detail : detail?.detail || "Request failed");
    this.status = status;
    this.detail = detail;
  }
}

/* -------------------------------------------------------------- helpers --- */

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2).toLowerCase(), v);
    else node.setAttribute(k, v === true ? "" : String(v));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

/** The command that produced what is on screen. Never decorative: it is
 *  rendered from the same validated request the server ran. */
export function commandLine(text) {
  const tpl = document.getElementById("tpl-command");
  const node = tpl.content.firstElementChild.cloneNode(true);
  node.querySelector(".command-text").textContent = text;
  node.querySelector(".copy").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(text);
      const b = node.querySelector(".copy");
      b.textContent = "copied";
      setTimeout(() => (b.textContent = "copy"), 1200);
    } catch { /* clipboard blocked; the text is selectable anyway */ }
  });
  return node;
}

/** Missing is a state, not an error: say what is absent and how to make it. */
export function missing(what, producedBy, extra) {
  return el("div", { class: "missing" },
    el("span", { class: "what" }, what),
    extra ? el("span", { class: "dim" }, extra) : null,
    producedBy && producedBy !== "(no producing command yet)"
      ? commandLine(producedBy)
      : el("span", { class: "dim" }, "No command produces this yet."));
}

export function spinner(label = "working…") {
  return el("div", { class: "spinner mono" }, label);
}

export function chip(text, kind) {
  return el("span", { class: kind ? `chip ${kind}` : "chip" }, text);
}

export function stat(key, value, note) {
  return el("div", { class: "card stat" },
    el("span", { class: "k" }, key),
    el("span", { class: "v" }, value),
    note ? el("span", { class: "n" }, note) : null);
}

export function table(columns, rows, opts = {}) {
  const thead = el("thead", {}, el("tr", {},
    ...columns.map((c) => el("th", { class: c.num ? "num" : null }, c.label))));
  const tbody = el("tbody", {}, ...rows.map((row) => {
    const tr = el("tr", { class: opts.onRow ? "is-clickable" : null },
      ...columns.map((c) => el("td", { class: c.num ? "num" : null }, c.get(row))));
    if (opts.onRow) tr.addEventListener("click", () => opts.onRow(row));
    return tr;
  }));
  return el("div", { class: "scroll" }, el("table", {}, thead, tbody));
}

export function fmt(value, digits = 3) {
  if (value === null || value === undefined) return "—";
  if (typeof value !== "number") return String(value);
  return Number.isInteger(value) ? String(value) : value.toFixed(digits);
}

export function setStatus(left, right) {
  if (left !== undefined) document.getElementById("status-left").textContent = left;
  if (right !== undefined) document.getElementById("status-right").textContent = right;
}

/* --------------------------------------------------------------- routing -- */

const VIEWS = [
  ["overview", "./views/overview.js"],
  ["map", "./views/map.js"],
  ["compare", "./views/compare.js"],
  ["upload", "./views/upload.js"],
  ["funnel", "./views/funnel.js"],
  ["evidence", "./views/evidence.js"],
  ["metrics", "./views/metrics.js"],
  ["playground", "./views/playground.js"],
  ["console", "./views/console.js"],
];

const loaded = new Map();
const ctx = { capabilities: null };

async function loadView(name) {
  if (!loaded.has(name)) {
    const entry = VIEWS.find(([n]) => n === name);
    if (!entry) throw new Error(`No view ${name}`);
    loaded.set(name, await import(entry[1]));
  }
  return loaded.get(name);
}

async function show(name) {
  const mount = document.getElementById("view");
  mount.replaceChildren(spinner());
  for (const btn of document.querySelectorAll(".tab")) {
    btn.setAttribute("aria-selected", String(btn.dataset.view === name));
  }
  try {
    const view = await loadView(name);
    mount.replaceChildren();
    await view.render(mount, ctx);
  } catch (err) {
    mount.replaceChildren(el("div", { class: "notice stop" },
      el("strong", {}, "This view failed to load. "), String(err.message || err)));
  }
  mount.focus({ preventScroll: true });
  if (location.hash.slice(1) !== name) history.replaceState(null, "", `#${name}`);
}

async function buildTabs() {
  const nav = document.getElementById("tabs");
  const groups = new Map();
  for (const [name] of VIEWS) {
    const view = await loadView(name);
    const group = view.group || "";
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push([name, view.title || name]);
  }
  nav.replaceChildren();
  for (const [group, items] of groups) {
    if (group) nav.append(el("span", { class: "tab-group" }, group));
    for (const [name, title] of items) {
      nav.append(el("button", {
        class: "tab", role: "tab", "data-view": name, "aria-selected": "false",
        onClick: () => show(name),
      }, title));
    }
  }
}

function initTheme() {
  const saved = (() => { try { return localStorage.getItem("scg-theme"); } catch { return null; } })();
  if (saved === "light" || saved === "dark") document.documentElement.dataset.theme = saved;
  else document.documentElement.removeAttribute("data-theme");
  document.getElementById("theme-toggle").addEventListener("click", () => {
    const now = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = now;
    try { localStorage.setItem("scg-theme", now); } catch { /* private window */ }
  });
}

async function main() {
  initTheme();
  await buildTabs();
  try {
    ctx.capabilities = await api.get("/api/capabilities");
    const p = ctx.capabilities.provenance || {};
    setStatus(
      `satchangegate ${p.satchangegate_version} · python ${p.python} · config ${p.config_sha256}`,
      ctx.capabilities.spend.allowed
        ? `spend enabled · cap $${ctx.capabilities.spend.cap_usd.toFixed(2)}`
        : "spend disabled",
    );
  } catch (err) {
    setStatus("could not read capabilities", String(err.message || err));
  }
  window.addEventListener("hashchange", () => show(location.hash.slice(1) || "overview"));
  await show(location.hash.slice(1) || "overview");
}

main();
