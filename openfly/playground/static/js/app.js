// openfly playground: wires the UI, the live stream and the 3D viewer together.
import { BrainViewer, hexToRgb } from "./viewer.js";
import { RateChart, Meters } from "./charts.js";
import { Sonifier } from "./audio.js";
import { ArenaView } from "./arena.js";

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const state = { info: null, running: true, mode: "brain", highlight: null, silenced: null, lastFrame: null, colorMode: "class", path: null };

// ------------------------------------------------------------- helpers
function b64ToU32(s) {
  if (!s) return new Uint32Array(0);
  const bin = atob(s), buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return new Uint32Array(buf.buffer);
}
function b64ToF32(s) {
  const bin = atob(s), buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return new Float32Array(buf.buffer);
}
function toast(msg, kind = "") {
  const t = document.createElement("div");
  t.className = "toast " + kind;
  t.textContent = msg;
  $("#toast").appendChild(t);
  setTimeout(() => t.remove(), kind === "err" ? 6000 : 3200);
}
async function api(cmd) {
  const r = await fetch("/api/command", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(cmd) });
  const j = await r.json();
  if (!j.ok) toast(j.error || "command failed", "err");
  return j;
}
function esc(s) { return String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
const ICONS = { taste: "\u{1F36C}", smell: "\u{1F443}", vision: "\u{1F441}", danger: "⚠️", hearing: "\u{1F3B5}", wind: "\u{1F32C}",
  touch: "✋", temperature: "\u{1F525}", food: "\u{1F34C}", social: "\u{1F98B}", motion: "\u{1F300}", feeding: "\u{1F36F}", navigation: "\u{1F9ED}", conflict: "⚔️", escape: "\u{1F4A8}" };

// ---------------------------------------------------------------- boot
const viewer = new BrainViewer($("#gl"));
const audio = new Sonifier();
const arena = new ArenaView($("#arena"));
let chart, meters;

async function boot() {
  const info = await (await fetch("/api/info")).json();
  state.info = info;
  $("#brainname").textContent = `${info.title} · ${info.N.toLocaleString()} neurons · ${info.n_connections.toLocaleString()} connections`;
  $("#about").innerHTML = `<div>${esc(info.description)}</div><div style="margin-top:6px"><b>Cite:</b> ${esc(info.citation)}</div>
    <div style="margin-top:6px"><b>Transmitter signs:</b> ${esc(info.transmitter_policy)}</div><div style="margin-top:6px">${esc(info.license)}</div>`;
  if (info.layout === "schematic") $("#layoutnote").style.display = "block";
  $("#splashmsg").textContent = "loading " + info.N.toLocaleString() + " neurons...";

  const geo = await (await fetch("/api/geometry")).arrayBuffer();
  const N = info.N;
  const pos = new Float32Array(geo, 0, N * 3);
  const codes = new Uint8Array(geo, N * 12, N * 3);
  const cls = new Uint8Array(N), nt = new Uint8Array(N), side = new Uint8Array(N);
  for (let i = 0; i < N; i++) { cls[i] = codes[3 * i]; nt[i] = codes[3 * i + 1]; side[i] = codes[3 * i + 2]; }
  state.codes = { cls, nt, side };
  viewer.setGeometry(new Float32Array(pos), state.codes, {
    cls: info.classes.map((c) => hexToRgb(c.color)), nt: info.nts.map((c) => hexToRgb(c.color)), side: info.sides.map((c) => hexToRgb(c.color)),
  });
  $("#dots").value = viewer.params.size;
  state.classIndex = Object.fromEntries(info.classes.map((c, k) => [c.name, k]));
  chart = new RateChart($("#chart"), Object.fromEntries(info.classes.map((c) => [c.name, c.color])));
  meters = new Meters($("#meters"), info.channels);
  buildSenses(info);
  buildLessons(info);
  buildLab(info);
  legend();
  viewer.onPick = inspect;
  connect();
  requestAnimationFrame(loop);
  setTimeout(() => { $("#splash").style.opacity = 0; setTimeout(() => $("#splash").remove(), 600); }, 200);
}

// ------------------------------------------------------------- stream
function connect() {
  const es = new EventSource("/api/stream");
  es.onmessage = (e) => onFrame(JSON.parse(e.data));
  es.onerror = () => { $("#stat").textContent = "reconnecting..."; };
}

function onFrame(f) {
  if (f.type === "error") { toast("simulation error: " + f.message, "err"); return; }
  state.lastFrame = f;
  const spikes = b64ToU32(f.spikes);
  viewer.time = f.t;
  viewer.spike(spikes, f.t);
  $("#clock").textContent = `t = ${(f.t / 1000).toFixed(3)} s`;
  $("#stat").textContent = `${f.n_active.toLocaleString()} firing · ${f.awake.toLocaleString()} awake · ${f.wall_ms} ms/step`;
  chart.push(f.t, f.class_rates);
  meters.update(f.drive);
  if (audio.on) {
    const counts = {};
    const names = state.info.classes.map((c) => c.name);
    for (let k = 0; k < spikes.length; k++) { const n = names[state.codes.cls[spikes[k]]]; counts[n] = (counts[n] || 0) + 1; }
    audio.frame(counts, state.classIndex, f.drive);
  }
  renderChips(f.stimuli);
  if (f.fly) {
    arena.update(f.fly);
    const s = f.fly;
    $("#arenastate").textContent = `t=${s.t_s.toFixed(1)}s  ${s.state}  energy ${s.energy.toFixed(2)}` + (s.events && s.events.length ? `  · ${s.events[s.events.length - 1].event}` : "");
  }
  if (f.running !== state.running) { state.running = f.running; $("#playbtn").textContent = f.running ? "Pause" : "Play"; }
}

let lastT = performance.now();
function loop(now) {
  const dt = now - lastT; lastT = now;
  viewer.render(now, dt);
  chart && chart.draw();
  if (state.mode === "fly") arena.draw();
  requestAnimationFrame(loop);
}

// ---------------------------------------------------------- stimuli
function stimMode() { return $("#stimmode").value === "replace"; }
async function stim(cmd) {
  if (cmd.replace === undefined && cmd.cmd !== "stimulate") cmd.replace = stimMode();
  const r = await api(cmd);
  if (r.ok && r.targets) { state.highlight = b64ToU32(r.targets); applyFlags(); toast(`${r.label}: ${r.n.toLocaleString()} neurons driven`, "good"); }
  return r;
}
function renderChips(list) {
  const root = $("#stimchips");
  const key = JSON.stringify(list.map((s) => s.id)) + (state.silenced ? state.silenced.length : 0);
  if (root.dataset.key === key) return;
  root.dataset.key = key;
  root.innerHTML = "";
  for (const s of list) {
    const c = document.createElement("div");
    c.className = "chip";
    c.innerHTML = `<span>${esc(s.label)} <span class="stat">${s.n.toLocaleString()}</span></span><span class="x" title="remove">×</span>`;
    c.querySelector(".x").onclick = () => api({ cmd: "remove", id: s.id });
    root.appendChild(c);
  }
  if (state.silenced && state.silenced.length) {
    const c = document.createElement("div");
    c.className = "chip silence";
    c.innerHTML = `<span>silenced <span class="stat">${state.silenced.length.toLocaleString()}</span></span><span class="x">×</span>`;
    c.querySelector(".x").onclick = unsilence;
    root.appendChild(c);
  }
}
function applyFlags() { viewer.setFlags(state.highlight, state.silenced); }

function buildSenses(info) {
  const sc = $("#scenarios");
  for (const s of info.scenarios) {
    const d = document.createElement("div");
    d.className = "card";
    const ico = ICONS[s.tags.find((t) => ICONS[t])] || "✨";
    d.innerHTML = `<div class="ico">${ico}</div><div class="t">${esc(s.title)}</div>`;
    d.title = `${s.story}\n\nA real fly: ${s.real_fly}`;
    d.onclick = async () => {
      d.classList.remove("flash"); void d.offsetWidth; d.classList.add("flash");
      const r = await stim({ cmd: "scenario", name: s.name });
      if (r.ok) toast(s.real_fly ? `Real flies: ${s.real_fly}` : s.story);
    };
    sc.appendChild(d);
  }
  const od = $("#odor");
  for (const o of info.odors) od.insertAdjacentHTML("beforeend", `<option value="${o.name}">${o.name.replaceAll("_", " ")} (${o.valence})</option>`);
  const showOdor = () => { const o = info.odors.find((x) => x.name === od.value); $("#odordesc").textContent = o ? o.description : ""; };
  od.onchange = showOdor; showOdor();
  $("#conc").oninput = () => ($("#concv").textContent = (+$("#conc").value).toFixed(2));
  $("#lr").oninput = () => { const v = +$("#lr").value; $("#lrv").textContent = Math.abs(v) < 0.05 ? "both" : v < 0 ? `left ${(-v * 100).toFixed(0)}%` : `right ${(v * 100).toFixed(0)}%`; };
  $("#smellbtn").onclick = () => {
    const c = +$("#conc").value, lr = +$("#lr").value;
    const cmd = { cmd: "smell", odor: od.value, concentration: c };
    if (Math.abs(lr) > 0.05) { cmd.left = c * (lr < 0 ? 1 : 1 - lr); cmd.right = c * (lr > 0 ? 1 : 1 + lr); }
    stim(cmd);
  };
  const ts = $("#tastes");
  for (const t of info.tastants) {
    const b = document.createElement("button");
    b.className = "btn"; b.textContent = t.replaceAll("_", " ");
    b.onclick = () => stim({ cmd: "taste", tastant: t });
    ts.appendChild(b);
  }
  for (const b of $$("[data-vis]")) b.onclick = () => stim({ cmd: "vision", kind: b.dataset.vis, azimuth: +(b.dataset.az || 0) });
  for (const b of $$("[data-cmd]")) b.onclick = () => stim(JSON.parse(b.dataset.cmd));
  $("#clearbtn").onclick = () => { api({ cmd: "clear" }); state.highlight = null; applyFlags(); };

  // neurons tab
  $("#rate").oninput = () => ($("#ratev").textContent = $("#rate").value + " Hz");
  const doSearch = async () => {
    const q = $("#search").value.trim();
    if (!q) return;
    const r = await api({ cmd: "search", q, limit: 30 });
    const root = $("#results");
    root.innerHTML = "";
    const rows = r.results || [];
    if (!rows.length) {
      // maybe it's a valid name that search() doesn't list (e.g. a class); offer it anyway
      rows.push({ name: q, kind: "name", n: "?", description: "" });
    }
    for (const row of rows) {
      const d = document.createElement("div");
      d.className = "r";
      d.innerHTML = `<span class="n" title="${esc(row.description)}">${esc(row.name)}</span><span class="c">${row.n}</span>
        <button class="btn small" data-a="go">Activate</button><button class="btn small danger" data-a="sil">Silence</button><button class="btn small" data-a="hl">Show</button>`;
      d.querySelector('[data-a="go"]').onclick = () => stim({ cmd: "stimulate", name: row.name, rate: +$("#rate").value, side: $("#nside").value, replace: stimMode() });
      d.querySelector('[data-a="sil"]').onclick = () => silence(row.name);
      d.querySelector('[data-a="hl"]').onclick = () => highlight(row.name);
      root.appendChild(d);
    }
  };
  $("#searchbtn").onclick = doSearch;
  $("#search").onkeydown = (e) => { if (e.key === "Enter") doSearch(); };
  $("#unsilbtn").onclick = unsilence;
}

async function silence(name) {
  const r = await api({ cmd: "silence", name });
  if (r.ok) { state.silenced = b64ToU32(r.silenced); applyFlags(); $("#silinfo").textContent = `${state.silenced.length.toLocaleString()} neurons silenced`; toast(`silenced ${name} (${r.n})`); }
}
async function unsilence() {
  await api({ cmd: "unsilence" });
  state.silenced = null; applyFlags(); $("#silinfo").textContent = "none";
}
async function highlight(name) {
  const r = await api({ cmd: "highlight", name });
  if (r.ok) { state.highlight = b64ToU32(r.indices); applyFlags(); toast(`${name}: ${r.n.toLocaleString()} neurons`); }
}

// --------------------------------------------------------- inspector
async function inspect(i) {
  if (i < 0) return;
  const r = await api({ cmd: "neuron", index: i });
  if (!r.ok) return;
  const n = r.neuron;
  state.highlight = new Uint32Array([i]); applyFlags();
  const partners = (rows, title) => `<div class="k" style="margin-top:6px">${title}</div><table>` +
    rows.map((p) => `<tr><td>${esc(p.cell_type || "?")}</td><td>${esc(p.side || "")}</td><td class="num">${Math.round(p.synapses)}</td></tr>`).join("") + "</table>";
  $("#inspector").innerHTML = `<div style="font-size:15px;font-weight:700">${esc(n.cell_type || "untyped neuron")}</div>
    <div><span class="pill">${esc(n.super_class)}</span><span class="pill">${esc(n.cell_class)}</span><span class="pill">${esc(n.side)}</span><span class="pill">${esc(n.nt)}</span></div>
    <div class="stat" style="margin-top:4px">id ${esc(n.root_id)} · ${n.n_inputs} in / ${n.n_outputs} out synapses${n.neuropil ? " · " + esc(n.neuropil) : ""}</div>
    ${partners(n.top_inputs, "strongest inputs")}${partners(n.top_outputs, "strongest outputs")}
    <div class="row"><button class="btn small" id="ins-go">Activate type</button><button class="btn small danger" id="ins-sil">Silence type</button><button class="btn small" id="ins-hl">Show type</button></div>`;
  const type = n.cell_type || String(n.root_id);
  $("#ins-go").onclick = () => stim({ cmd: "stimulate", name: type, rate: +$("#rate").value, replace: stimMode() });
  $("#ins-sil").onclick = () => silence(type);
  $("#ins-hl").onclick = () => highlight(type);
}

// ------------------------------------------------------------ legend
function legend() {
  const info = state.info, mode = state.colorMode, root = $("#legend");
  if (mode === "influence") {
    root.innerHTML = `<div class="li"><span class="sw" style="color:#ff7333;background:#ff7333"></span>excited by source</div>
      <div class="li"><span class="sw" style="color:#408cff;background:#408cff"></span>inhibited by source</div>`;
    return;
  }
  const items = mode === "nt" ? info.nts : mode === "side" ? info.sides : info.classes;
  const key = mode === "nt" ? "nt" : mode === "side" ? "side" : "cls";
  root.innerHTML = (mode === "activity" ? `<div class="stat" style="margin-bottom:4px">only firing neurons glow</div>` : "") +
    items.filter((c) => c.count === undefined || c.count > 0).map((c, k) =>
      `<div class="li" data-k="${items.indexOf(c)}"><span class="sw" style="color:${c.color};background:${c.color}"></span>${esc(c.name.replaceAll("_", " "))}<span class="c">${c.count ? c.count.toLocaleString() : ""}</span></div>`).join("");
  for (const li of root.querySelectorAll(".li")) li.onclick = () => {
    const k = +li.dataset.k, codes = state.codes[key], idx = [];
    for (let i = 0; i < codes.length; i++) if (codes[i] === k) idx.push(i);
    state.highlight = new Uint32Array(idx); applyFlags();
    toast(`${items[k].name}: ${idx.length.toLocaleString()} neurons`);
  };
}
function setColor(mode, values) {
  state.colorMode = mode;
  viewer.setColorMode(mode, values);
  for (const b of $$("#colorseg button")) b.classList.toggle("on", b.dataset.c === mode);
  legend();
}

// ------------------------------------------------------------ lessons
async function runAction(a) {
  if (!a) return;
  if (a.cmd === "color") return setColor(a.mode);
  if (a.cmd === "highlight") return highlight(a.name);
  if (a.cmd === "path") return findPath(a.source, a.target);
  if (a.cmd === "mode") return setMode(a.mode, a.arena);
  if (a.cmd === "silence") return silence(a.name);
  if (a.cmd === "unsilence") return unsilence();
  return stim({ ...a });
}
function buildLessons(info) {
  const root = $("#lessons");
  for (const L of info.lessons) {
    const d = document.createElement("div");
    d.className = "lesson";
    d.innerHTML = `<div class="lt"><span>${esc(L.title)}</span><button class="btn small">Start</button></div>` +
      L.steps.map((s, k) => `<div class="step" data-k="${k}"><div class="st">${k + 1}/${L.steps.length} · ${esc(s.title)}</div><p>${esc(s.text)}</p>
        <div class="row">${k ? '<button class="btn small" data-a="prev">Back</button>' : ""}<button class="btn small primary" data-a="do">Show me</button>
        ${k < L.steps.length - 1 ? '<button class="btn small" data-a="next">Next</button>' : '<button class="btn small" data-a="done">Done</button>'}</div></div>`).join("");
    let k = 0;
    const show = (i) => { k = i; d.querySelectorAll(".step").forEach((s) => s.classList.toggle("on", +s.dataset.k === k)); runAction(L.steps[k].action); };
    d.querySelector(".lt button").onclick = () => { $$(".lesson").forEach((x) => x.classList.remove("open")); d.classList.add("open"); show(0); };
    d.querySelectorAll('[data-a="next"]').forEach((b) => (b.onclick = () => show(k + 1)));
    d.querySelectorAll('[data-a="prev"]').forEach((b) => (b.onclick = () => show(k - 1)));
    d.querySelectorAll('[data-a="do"]').forEach((b) => (b.onclick = () => runAction(L.steps[k].action)));
    d.querySelectorAll('[data-a="done"]').forEach((b) => (b.onclick = () => { d.classList.remove("open"); toast("lesson complete ✨", "good"); }));
    root.appendChild(d);
  }
  const done = JSON.parse(localStorage.getItem("openfly-challenges") || "{}");
  const cr = $("#challenges");
  for (const c of info.challenges) {
    const d = document.createElement("div");
    d.className = "challenge" + (done[c.id] ? " done" : "");
    d.innerHTML = `<div class="ct">${done[c.id] ? "✅ " : ""}${esc(c.title)}</div><div class="cg">${esc(c.goal)}</div><div class="ch">hint: ${esc(c.hint)}</div>
      <div class="row"><button class="btn small primary">Check now</button><span class="stat"></span></div>`;
    d.querySelector("button").onclick = async () => {
      const r = await api({ cmd: "challenge", id: c.id });
      d.querySelector(".stat").textContent = r.detail || "";
      if (r.ok) {
        done[c.id] = true; localStorage.setItem("openfly-challenges", JSON.stringify(done));
        d.classList.add("done"); d.querySelector(".ct").textContent = "✅ " + c.title; toast(`Challenge solved: ${c.title}!`, "good");
      } else toast("not yet: " + (r.detail || ""));
    };
    cr.appendChild(d);
  }
}

// --------------------------------------------------------------- lab
async function findPath(src, tgt) {
  $("#psrc").value = src; $("#ptgt").value = tgt;
  const r = await api({ cmd: "path", source: src, target: tgt });
  if (!r.ok) return;
  const idx = r.path.map((p) => p.index);
  viewer.setPath(idx);
  state.highlight = new Uint32Array(idx); applyFlags();
  $("#pathout").innerHTML = r.path.map((p, k) => `<div class="p">${k ? '<span class="arrow">↓</span>' : '<span class="arrow">●</span>'}
    <b>${esc(p.cell_type)}</b><span class="stat">${esc(p.side)} · ${esc(p.nt)}${k ? ` · ${Math.round(p.synapses)} syn (${(p.input_fraction * 100).toFixed(1)}% of input)` : ""}</span></div>`).join("");
}
async function poll(job, out, render) {
  out.style.display = "block";
  out.textContent = "running...";
  for (;;) {
    await new Promise((r) => setTimeout(r, 1000));
    const j = await (await fetch(`/api/job/${job}`)).json();
    if (j.status === "running") { out.textContent = j.progress; continue; }
    if (j.status === "error") { out.textContent = j.result.error; return; }
    render(j.result); return;
  }
}
function buildLab(info) {
  $("#pathbtn").onclick = () => findPath($("#psrc").value, $("#ptgt").value);
  $("#pathclear").onclick = () => { viewer.setPath(null); $("#pathout").innerHTML = ""; state.highlight = null; applyFlags(); };
  $("#inflbtn").onclick = async () => {
    const r = await api({ cmd: "influence", source: $("#isrc").value });
    if (r.ok) setColor("influence", b64ToF32(r.values));
  };
  $("#screenbtn").onclick = async () => {
    const r = await api({ cmd: "job", kind: "screen", target: $("#scrtgt").value });
    if (r.ok) poll(r.job, $("#screenout"), (res) => {
      $("#screenout").textContent = "rank  cell type        neurons  target rate\n" +
        res.table.map((t, k) => `${String(k + 1).padStart(3)}   ${String(t.group).padEnd(16)} ${String(t.neurons).padStart(6)}  ${t.target_rate_hz.toFixed(1)} Hz`).join("\n");
    });
  };
  for (const id of ["#csp", "#csm"]) for (const o of info.odors) $(id).insertAdjacentHTML("beforeend", `<option value="${o.name}">${o.name.replaceAll("_", " ")}</option>`);
  $("#csp").value = "banana"; $("#csm").value = "acetic_acid";
  $("#condbtn").onclick = async () => {
    const r = await api({ cmd: "job", kind: "conditioning", cs_plus: $("#csp").value, cs_minus: $("#csm").value, reinforcer: $("#reinf").value });
    if (r.ok) poll(r.job, $("#condout"), (res) => { $("#condout").textContent = res.summary; toast(`learning index ${res.learning_index}`, "good"); });
  };
}

// --------------------------------------------------------------- modes
async function setMode(mode, arenaName) {
  const r = await api({ cmd: "mode", mode, arena: arenaName || $("#arenasel").value });
  if (!r.ok) return;
  state.mode = mode;
  for (const b of $$("#modeseg button")) b.classList.toggle("on", b.dataset.mode === mode);
  $("#arenawrap").classList.toggle("on", mode === "fly");
  if (mode === "fly") { arena.setArena(r.arena, r.field); if (arenaName) $("#arenasel").value = arenaName; toast("The brain is now driving a virtual fly"); }
}

// ---------------------------------------------------------- controls
$("#playbtn").onclick = () => api({ cmd: state.running ? "pause" : "play" });
$("#speed").onchange = () => api({ cmd: "speed", value: +$("#speed").value });
$("#resetbtn").onclick = () => { api({ cmd: "reset" }); viewer.clearSpikes(); chart.clear(); state.highlight = null; applyFlags(); };
$("#soundbtn").onclick = () => { if (audio.on) { audio.disable(); $("#soundbtn").textContent = "Sound off"; } else { audio.enable(); $("#soundbtn").textContent = "Sound on"; } $("#soundbtn").classList.toggle("on", audio.on); };
for (const b of $$("#modeseg button")) b.onclick = () => setMode(b.dataset.mode);
for (const b of $$("#colorseg button")) b.onclick = () => setColor(b.dataset.c);
for (const b of $$("#viewseg button")) b.onclick = () => viewer.view(b.dataset.v);
for (const b of $$(".tabs button")) b.onclick = () => {
  $$(".tabs button").forEach((x) => x.classList.toggle("on", x === b));
  $$(".tabpane").forEach((p) => { if (p.id.startsWith("tab-")) p.classList.toggle("on", p.id === "tab-" + b.dataset.tab); });
};
$("#rotbtn").onclick = () => { viewer.params.autoRotate = !viewer.params.autoRotate; $("#rotbtn").classList.toggle("on", viewer.params.autoRotate); };
$("#glow").oninput = () => (viewer.params.decay = +$("#glow").value);
$("#dots").oninput = () => (viewer.params.size = +$("#dots").value);
for (const b of $$("#toolseg button")) b.onclick = () => { arena.tool = b.dataset.t; $$("#toolseg button").forEach((x) => x.classList.toggle("on", x === b)); };
$("#predbtn").onclick = () => api({ cmd: "arena_add", kind: "predator", bearing: Math.random() < 0.5 ? -60 : 60 });
arena.onClick = async (tool, x, y) => {
  let r;
  if (tool === "move") r = await api({ cmd: "fly_place", x, y });
  else if (tool === "food") r = await api({ cmd: "arena_add", kind: "food", x, y, odor: "banana", taste: "sugar", name: "banana" });
  else if (tool === "bitter") r = await api({ cmd: "arena_add", kind: "food", x, y, odor: "apple_cider_vinegar", taste: "bitter", name: "bitter bait" });
  else if (tool === "heat") r = await api({ cmd: "arena_add", kind: "heat", x, y });
  if (r && r.arena) arena.setArena(r.arena, r.field);
};
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.code === "Space") { e.preventDefault(); $("#playbtn").click(); }
  if (e.key === "1") viewer.view("back"); if (e.key === "2") viewer.view("front"); if (e.key === "3") viewer.view("top"); if (e.key === "4") viewer.view("left");
});
fetch("/api/info").then((r) => r.json()).then((info) => {
  for (const a of info.arenas) $("#arenasel").insertAdjacentHTML("beforeend", `<option value="${a}">${a.replaceAll("_", " ")}</option>`);
  $("#arenasel").onchange = () => setMode("fly", $("#arenasel").value);
});

boot().catch((e) => { $("#splashmsg").textContent = "failed to start: " + e.message; console.error(e); });
