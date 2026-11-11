/* Soundxx front end: no build step, no network beyond 127.0.0.1. */
"use strict";

// ---------------------------------------------------------------- plumbing
const TOKEN = (() => {
  const m = location.hash.match(/token=([\w-]+)(?:&to=([\w/%.-]+))?/);
  if (m) { sessionStorage.setItem("token", m[1]); history.replaceState(null, "", "#" + (m[2] || "/record")); }
  return sessionStorage.getItem("token") || "";
})();

async function api(path, opts = {}) {
  const init = { method: opts.method || (opts.body ? "POST" : "GET"), headers: { "X-Token": TOKEN } };
  if (opts.body !== undefined) { init.headers["Content-Type"] = "application/json"; init.body = JSON.stringify(opts.body); }
  const r = await fetch(path, init);
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = j.detail || msg; } catch (_) { msg = await r.text().catch(() => msg); }
    throw new Error(msg);
  }
  const type = r.headers.get("content-type") || "";
  return type.includes("json") ? r.json() : r.text();
}
const post = (path, body = {}) => api(path, { body });

function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === false || v == null) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "class") el.className = v;
    else if (k === "html") el.innerHTML = v;
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid.nodeType ? kid : String(kid));
  return el;
}
const $ = (sel, root = document) => root.querySelector(sel);

function clock(s) {
  s = Math.max(0, Math.floor(s || 0));
  const hh = Math.floor(s / 3600), mm = Math.floor(s / 60) % 60, ss = s % 60;
  return (hh ? hh + ":" + String(mm).padStart(2, "0") : String(mm)) + ":" + String(ss).padStart(2, "0");
}
function when(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" }) + ", " +
    d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}
function meterPct(db) { return Math.max(0, Math.min(100, (db + 70) / 60 * 100)); }

function toast(message, level = "info", ms = 4500) {
  const t = h("div", { class: "toast " + level, role: level === "error" ? "alert" : "status" }, message);
  $("#toasts").append(t);
  setTimeout(() => t.remove(), level === "error" ? ms * 2 : ms);
}
async function guarded(fn) {
  try { return await fn(); } catch (e) { toast(e.message, "error"); }
}

function confirmDialog(title, text, okLabel, danger) {
  return new Promise(resolve => {
    const d = h("dialog", {}, h("h2", {}, title), h("p", {}, text),
      h("div", { class: "row", style: "justify-content:flex-end" },
        h("button", { class: "btn", onclick: () => { d.close(); resolve(false); } }, "Cancel"),
        h("button", { class: "btn " + (danger ? "rec" : "primary"), onclick: () => { d.close(); resolve(true); } }, okLabel)));
    document.body.append(d);
    d.addEventListener("close", () => { d.remove(); resolve(false); });
    d.showModal();
  });
}

// ---------------------------------------------------------------- live push
const store = { engine: null, jobs: null, state: null, lines: [], listeners: new Set() };
function emit(msg) { for (const fn of store.listeners) fn(msg); }

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws?token=${TOKEN}`);
  ws.onmessage = ev => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "hello") { store.engine = msg.engine; store.jobs = msg.jobs; store.state = msg.engine.state; }
    else if (msg.type === "state") store.state = msg.state;
    else if (msg.type === "jobs") { store.jobs = msg; renderJobline(); }
    else if (msg.type === "line") store.lines.push(msg);
    else if (msg.type === "event" && msg.level !== "info") toast(msg.message, msg.level);
    else if (msg.type === "event") toast(msg.message, "info", 2500);
    emit(msg);
  };
  ws.onclose = () => setTimeout(connect, 1000);
}

function renderJobline() {
  const el = $("#jobline");
  const j = store.jobs;
  el.replaceChildren();
  if (!j) return;
  const cur = j.current;
  if (cur && cur.kind === "process") {
    const p = cur.progress || {};
    el.append(h("a", { href: "#/session/" + cur.session, class: "small" }, p.message || "Processing a lecture"),
      h("span", { class: "meter" }, h("i", { style: `width:${Math.round((p.fraction || 0) * 100)}%` })));
  } else if (cur && cur.kind === "notes") {
    el.append(h("a", { href: "#/session/" + cur.session }, "Writing lecture notes"), h("span", { class: "loading" }));
  } else if (j.queue && j.queue.length) {
    el.append(j.waiting_for_engine ? `${j.queue.length} waiting until the recording screen closes`
      : `${j.queue.length} queued`);
  }
}

// ---------------------------------------------------------------- router
let cleanup = null;
async function route() {
  if (cleanup) { cleanup(); cleanup = null; }
  const view = $("#view");
  view.className = "";
  view.replaceChildren();
  const [, page, arg] = (location.hash.match(/^#\/(\w+)\/?(.*)$/) || [null, "record", ""]);
  for (const a of document.querySelectorAll("[data-nav]"))
    a.classList.toggle("on", a.dataset.nav === (page === "session" ? "sessions" : page === "live" ? "record" : page));
  if (page !== "record" && page !== "live") post("/api/engine/release").catch(() => {});
  if (page === "sessions") cleanup = await viewSessions(view);
  else if (page === "session") cleanup = await viewSession(view, decodeURIComponent(arg));
  else if (page === "live") cleanup = await viewLive(view);
  else cleanup = await viewRecord(view);
  view.focus({ preventScroll: true });
}
window.addEventListener("hashchange", route);

// ---------------------------------------------------------------- record (start)
async function viewRecord(view) {
  const status = await api("/api/status");
  store.engine = status.engine;
  if (status.engine.session) { location.hash = "#/live"; return viewLive(view); }
  const settings = status.settings;
  let output = settings.output_device || "";
  let mode = settings.mode || "listen";

  const meter = h("i");
  const micName = h("span", { class: "muted small" }, "");
  const micDb = h("span", { class: "muted small" }, "");
  const outSel = h("select", { id: "out", "aria-describedby": "out-help" });
  const outHelp = h("div", { id: "out-help", class: "small muted" });
  const checksEl = h("ul", { class: "checks" });
  const startBtn = h("button", { class: "btn rec big", disabled: true }, "Start recording");
  const ready = h("p", { class: "loading" }, "Getting the microphone and models ready");
  const title = h("input", { type: "text", id: "title", placeholder: "e.g. Linked lists" });
  const course = h("input", { type: "text", id: "course", list: "courses", value: settings.courses[0] || "" });
  const dl = h("datalist", { id: "courses" }, settings.courses.map(c => h("option", { value: c })));
  const segBtns = [["listen", "Listen to everyone"], ["full", "Focus on a speaker"]].map(([m, label]) =>
    h("button", { class: m === mode ? "on" : "", "aria-pressed": String(m === mode), onclick: () => setMode(m) }, label));
  const modeHelp = h("div", { class: "small muted" });

  function setMode(m) {
    mode = m;
    segBtns.forEach((b, i) => { const on = ["listen", "full"][i] === m; b.classList.toggle("on", on); b.setAttribute("aria-pressed", String(on)); });
    modeHelp.textContent = m === "listen"
      ? "The whole room, cleaned up, with the least delay in your ears."
      : "The person you pick is louder and everyone else quieter. About a quarter second more delay.";
    post("/api/live/command", { cmd: "set_mode", mode: m }).catch(() => {});
  }
  setMode(mode);

  async function loadDevices() {
    const d = await api("/api/devices").catch(() => ({ outputs: [] }));
    outSel.replaceChildren(h("option", { value: "" }, "No earbuds (record only)"),
      ...d.outputs.map(o => h("option", { value: o.name, disabled: o.loudspeaker, selected: o.name === output },
        o.loudspeaker ? `${o.name} (would feed back into the mic)` : o.name)));
    if (output && !d.outputs.some(o => o.name === output))
      outSel.append(h("option", { value: output, selected: true }, `${output} (not connected)`));
    outSel.value = output;
  }
  outSel.addEventListener("change", () => {
    output = outSel.value;
    post("/api/live/command", { cmd: "set_output", device: output || null }).catch(() => {});
    refreshChecks();
  });

  async function refreshChecks() {
    const pf = await api("/api/preflight?output=" + encodeURIComponent(output)).catch(() => null);
    if (!pf) return;
    const shown = pf.checks.filter(c => c.level !== "ok");
    checksEl.replaceChildren(...(shown.length ? shown : pf.checks.filter(c => ["output", "disk", "model_final"].includes(c.id)))
      .map(c => h("li", { class: c.level }, h("div", { class: "t" }, c.title), h("div", {}, c.message),
        c.fix ? h("div", { class: "fix" }, c.fix) : null)));
    startBtn.disabled = !pf.can_start || !(store.engine && store.engine.ready);
    startBtn.dataset.blocked = pf.can_start ? "" : "1";
  }

  function onMsg(msg) {
    if (msg.type === "state") {
      const s = msg.state;
      meter.style.width = meterPct(s.mic_db) + "%";
      meter.parentElement.classList.toggle("clip", !!s.clipping);
      micName.textContent = s.input || "";
      micDb.textContent = s.mic_silent ? "no signal" : Math.round(s.mic_db) + " dB";
      outHelp.textContent = !s.wanted_output ? "Nothing will play; the lecture is still recorded."
        : s.output_state === "running" ? `${s.output} connected` : `Waiting for ${s.wanted_output} to connect`;
    } else if (msg.type === "ready") {
      store.engine = { ...(store.engine || {}), ready: true };
      ready.textContent = "Ready.";
      ready.className = "muted";
      refreshChecks();
    } else if (msg.type === "loading") {
      ready.textContent = `Loading ${msg.what}`;
    } else if (msg.type === "fatal") {
      ready.className = "";
      ready.textContent = "The audio engine couldn't start: " + msg.message;
      refreshChecks();
    } else if (msg.type === "started") {
      location.hash = "#/live";
    } else if (msg.type === "engine_exit" && !store.engine?.session) {
      ready.className = "";
      ready.replaceChildren("The audio engine closed. ", h("button", { class: "btn", onclick: warm }, "Try again"));
    }
  }
  async function warm() {
    ready.className = "loading";
    ready.textContent = "Getting the microphone and models ready";
    store.engine = await post("/api/engine/warm", { output: output || null, mode });
    if (store.engine.ready) { ready.className = "muted"; ready.textContent = "Ready."; }
    refreshChecks();
  }
  startBtn.addEventListener("click", () => guarded(async () => {
    startBtn.disabled = true;
    await post("/api/live/start", { title: title.value, course: course.value });
  }));

  view.append(h("div", { class: "start" },
    h("section", {},
      h("h1", {}, "Record a lecture"),
      h("p", { class: "lede" }, "Soundxx plays the room into your earbuds, cleaned up, while it records and " +
        "writes down who says what. After class it makes a careful transcript and notes."),
      h("div", { class: "mic" }, h("div", { class: "row" }, h("b", {}, "Microphone"), micName, micDb),
        h("div", { class: "meter big", role: "meter", "aria-label": "Microphone level" }, meter)),
      h("div", { class: "field" }, h("label", { for: "out" }, "Play into"), outSel, outHelp),
      h("div", { class: "field" }, h("label", {}, "In your ears"), h("div", { class: "seg", role: "group" }, segBtns), modeHelp),
      h("div", { class: "row" },
        h("div", { class: "field", style: "flex:2" }, h("label", { for: "title" }, "Lecture (optional)"), title),
        h("div", { class: "field", style: "flex:1" }, h("label", { for: "course" }, "Course (optional)"), course, dl)),
      h("div", { class: "row" }, startBtn, ready)),
    h("aside", { class: "panel" }, h("h2", {}, "Before you start"), checksEl)));

  store.listeners.add(onMsg);
  await loadDevices();
  await warm();
  const timer = setInterval(() => { refreshChecks(); }, 4000);
  const devTimer = setInterval(loadDevices, 10000);
  return () => { store.listeners.delete(onMsg); clearInterval(timer); clearInterval(devTimer); };
}

// ---------------------------------------------------------------- live
async function viewLive(view) {
  view.className = "full";
  const st = await api("/api/status");
  if (!st.engine.session) { location.hash = "#/record"; return null; }
  const sid = st.engine.session;
  const prior = await api("/api/live/lines");
  store.lines = prior.lines || [];
  let editing = null;

  const recEl = h("span", { class: "recdot" }, "0:00");
  const chips = h("span", { class: "row" });
  const pauseBtn = h("button", { class: "btn", onclick: () => post("/api/live/command", { cmd: store.state?.phase === "paused" ? "resume" : "pause" }) }, "Pause");
  const stopBtn = h("button", { class: "btn primary", onclick: stop }, "Stop recording");
  const segBtns = [["listen", "Everyone"], ["full", "Focus"]].map(([m, label]) =>
    h("button", { title: m === "listen" ? "The whole room, least delay" : "The selected speaker louder (about 250 ms more delay)",
      onclick: () => post("/api/live/command", { cmd: "set_mode", mode: m }) }, label));
  const spkList = h("div");
  const feed = h("div", { class: "feed", "aria-live": "off" });
  let stick = true;
  feed.addEventListener("scroll", () => { stick = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 60; });

  async function stop() {
    if (!(await confirmDialog("Stop recording?", "The recording and the live transcript are saved. The careful " +
      "transcript is made next, in the background.", "Stop recording", true))) return;
    stopBtn.disabled = true;
    stopBtn.textContent = "Finishing the transcript…";
    await post("/api/live/stop");
  }

  function lineEl(l) {
    return h("p", { class: "ln" }, h("span", { class: "ts" }, clock(l.start)),
      h("span", { class: "who" }, l.speaker || "Unknown"), l.text);
  }
  function renderLines() {
    feed.replaceChildren(...(store.lines.length ? store.lines.map(lineEl)
      : [h("p", { class: "empty" }, "Words appear here a few seconds after they're spoken. Everyone is " +
        "written down, with who said it.")]));
    if (stick) feed.scrollTop = feed.scrollHeight;
  }
  renderLines();

  function renderSpeakers(s) {
    if (editing) return;
    const rows = [h("div", { class: "spk" + (s.selected == null ? " sel" : ""), tabindex: 0,
      onclick: () => post("/api/live/command", { cmd: "select", speaker: null }) },
      h("span", { class: "name" }, "Everyone"), h("span", { class: "time" }, ""))];
    for (const p of s.speakers) {
      const row = h("div", { class: "spk" + (p.selected ? " sel" : "") + (p.talking ? " talking" : ""), tabindex: 0,
        onclick: ev => {
          if (ev.target.closest("button,input,label")) return;
          post("/api/live/command", { cmd: "select", speaker: p.id });
          if (s.mode !== "full") post("/api/live/command", { cmd: "set_mode", mode: "full" });
        },
        onkeydown: ev => { if (ev.key === "Enter" && ev.target === row) row.click(); } },
        h("span", { class: "name" }, p.label + (p.enrolled ? " ★" : "")),
        h("span", { class: "time" }, clock(p.speaking_s)),
        h("div", { class: "meter" }, h("i", { style: `width:${p.level_db > -99 ? meterPct(p.level_db) : 0}%` })),
        h("div", { class: "acts" }, h("button", { class: "btn quiet small", onclick: () => rename(p) }, "Name")));
      rows.push(row);
    }
    if (!s.speakers.length) rows.push(h("p", { class: "muted small" }, "People appear here once they've spoken for a few seconds."));
    rows.push(h("div", { class: "row", style: "margin-top:16px" },
      h("button", { class: "btn quiet small", onclick: () => post("/api/live/command", { cmd: "reset" }) }, "Start over with speakers")));
    spkList.replaceChildren(h("h2", {}, "Speakers"), ...rows);
  }
  function rename(p) {
    editing = p.id;
    const inp = h("input", { type: "text", value: p.label.startsWith("Speaker ") ? "" : p.label, placeholder: "Name, e.g. Pak Budi" });
    const keep = h("input", { type: "checkbox", id: "keep", checked: true });
    const done = () => { editing = null; };
    const box = h("div", { class: "rename" }, inp,
      h("label", { class: "small" }, keep, " Remember this voice next time"),
      h("div", { class: "row" },
        h("button", { class: "btn primary", onclick: async () => {
          if (inp.value.trim()) await post("/api/live/command", { cmd: "rename", speaker: p.id, name: inp.value, remember: keep.checked });
          done();
        } }, "Save"),
        h("button", { class: "btn", onclick: done }, "Cancel")));
    inp.addEventListener("keydown", ev => { if (ev.key === "Enter") box.querySelector(".primary").click(); if (ev.key === "Escape") done(); });
    spkList.replaceChildren(h("h2", {}, "Name " + p.label), box);
    inp.focus();
  }

  function renderStatus(s) {
    recEl.textContent = clock(s.recording_s);
    recEl.classList.toggle("paused", s.phase === "paused");
    pauseBtn.textContent = s.phase === "paused" ? "Resume" : "Pause";
    pauseBtn.disabled = s.phase === "stopping";
    segBtns.forEach((b, i) => b.classList.toggle("on", ["listen", "full"][i] === s.mode));
    const c = [];
    if (s.phase === "paused") c.push(h("span", { class: "chip warn" }, "Paused: nothing is recorded"));
    if (s.phase === "stopping") c.push(h("span", { class: "chip" }, "Finishing the transcript…"));
    if (!s.wanted_output) c.push(h("span", { class: "chip" }, "No earbuds: recording only"));
    else if (s.output_state === "running") c.push(h("span", { class: "chip good" }, s.output + (s.latency_ms ? ` (${s.latency_ms} ms)` : "")));
    else c.push(h("span", { class: "chip warn" }, `Waiting for ${s.wanted_output}`));
    if (s.mic_silent) c.push(h("span", { class: "chip bad" }, "Mic gives no sound: check microphone access"));
    if (s.clipping) c.push(h("span", { class: "chip warn" }, "Mic too loud"));
    if (s.asr_pending > 2) c.push(h("span", { class: "chip warn" }, "Transcript is behind"));
    if (!s.asr_on) c.push(h("span", { class: "chip warn" }, "Live transcript off"));
    if (s.recorder_error) c.push(h("span", { class: "chip bad" }, "Recording stopped: " + s.recorder_error));
    if (s.errors) c.push(h("span", { class: "chip bad" }, "Audio processing failed; playing unprocessed"));
    chips.replaceChildren(...c);
  }

  function onMsg(msg) {
    if (msg.type === "state") { renderStatus(msg.state); renderSpeakers(msg.state); }
    else if (msg.type === "line") { feed.append(lineEl(msg)); feed.querySelector(".empty")?.remove(); if (stick) feed.scrollTop = feed.scrollHeight; }
    else if (msg.type === "engine_exit") location.hash = "#/session/" + encodeURIComponent(sid);
  }
  store.listeners.add(onMsg);
  if (store.state) { renderStatus(store.state); renderSpeakers(store.state); }

  view.append(h("div", { class: "status" }, recEl, chips, h("span", { class: "spacer" }),
    h("div", { class: "seg", role: "group", "aria-label": "In your ears" }, segBtns), pauseBtn, stopBtn),
    h("div", { class: "live" }, h("section", { class: "speakers" }, spkList), feed));
  return () => store.listeners.delete(onMsg);
}

// ---------------------------------------------------------------- sessions
const STATUS = { recording: "Recording", recorded: "Not processed yet", processing: "Processing", ready: "Ready" };

async function viewSessions(view) {
  const data = await api("/api/sessions");
  let course = sessionStorage.getItem("course-filter") || "";
  const courses = [...new Set(data.sessions.map(s => s.course).filter(Boolean))].sort();
  const sel = h("select", { "aria-label": "Course" }, h("option", { value: "" }, "All courses"),
    courses.map(c => h("option", { value: c, selected: c === course }, c)));
  const list = h("ul", { class: "list" });
  sel.addEventListener("change", () => { course = sel.value; sessionStorage.setItem("course-filter", course); render(); });

  function chip(s) {
    if (s.needs_recovery) return h("span", { class: "chip bad" }, "Interrupted");
    if (s.status === "processing") return h("span", { class: "chip" }, "Processing");
    if (s.status === "ready") return h("span", { class: "chip good" }, s.has_notes ? "Ready, with notes" : "Ready");
    return h("span", { class: "chip" + (s.status === "recording" ? " bad" : "") }, STATUS[s.status] || s.status);
  }
  function render() {
    const shown = data.sessions.filter(s => !course || s.course === course);
    list.replaceChildren(...(shown.length ? shown.map(s => h("li", {}, h("a", { href: "#/session/" + encodeURIComponent(s.id) },
      h("span", { class: "when" }, when(s.started)),
      h("span", {}, h("span", { class: "title" }, s.title || "Untitled lecture"), s.course ? h("span", { class: "course" }, s.course) : null),
      h("span", { class: "dur" }, clock(s.duration_s)), chip(s))))
      : [h("li", { class: "empty", style: "padding:20px 0" }, data.sessions.length ? "No sessions for this course."
        : "No sessions yet. Record a lecture, or bring in a recording you already have.")]));
  }
  const interrupted = data.sessions.filter(s => s.needs_recovery);
  const banner = interrupted.length ? h("div", { class: "banner" },
    h("span", {}, `${interrupted.length} session${interrupted.length > 1 ? "s" : ""} stopped unexpectedly. The audio up to that moment is saved.`),
    h("button", { class: "btn", onclick: () => guarded(async () => {
      for (const s of interrupted) await post(`/api/sessions/${encodeURIComponent(s.id)}/recover`);
      toast("Recovered. They're processed next.");
      route();
    }) }, "Recover")) : null;

  async function importFile() {
    let path = null;
    if (window.pywebview?.api?.pick_audio) path = await window.pywebview.api.pick_audio();
    else path = prompt("Path of the audio file to bring in (wav, m4a, mp3):");
    if (!path) return;
    await guarded(async () => {
      const s = await post("/api/import", { path, course });
      toast("Imported. It's processed next.");
      location.hash = "#/session/" + encodeURIComponent(s.id);
    });
  }

  view.append(h("div", { class: "row", style: "justify-content:space-between;max-width:980px" },
    h("h1", {}, "Sessions"), h("div", { class: "row" }, sel, h("button", { class: "btn", onclick: importFile }, "Bring in a recording"))),
    banner, list);
  render();
  const onMsg = msg => { if (msg.type === "session_changed" || msg.type === "engine_exit") route(); };
  store.listeners.add(onMsg);
  return () => store.listeners.delete(onMsg);
}

// ---------------------------------------------------------------- one session
async function viewSession(view, sid) {
  view.className = "full";
  const enc = encodeURIComponent(sid);
  let data = await api(`/api/sessions/${enc}`);
  const audio = h("audio", { controls: true, preload: "metadata", src: `/api/sessions/${enc}/audio?token=${TOKEN}` });
  const rates = [1, 1.25, 1.5, 2];
  const rateBtn = h("button", { class: "btn quiet", title: "Playback speed", onclick: () => {
    const i = (rates.indexOf(audio.playbackRate) + 1) % rates.length; audio.playbackRate = rates[i]; rateBtn.textContent = rates[i] + "×";
  } }, "1×");
  const back = h("button", { class: "btn quiet", title: "Back 10 seconds", onclick: () => { audio.currentTime -= 10; } }, "−10 s");
  const head = h("div", { class: "shead" });
  const spkbar = h("div", { class: "spkbar" });
  const tx = h("section", { class: "tx", "aria-label": "Transcript" });
  const notes = h("aside", { class: "notes", "aria-label": "Notes" });
  let lineEls = [];
  let editingLine = null;

  function play(t) { audio.currentTime = Math.max(0, t - 0.3); audio.play().catch(() => {}); }
  async function reload() { data = await api(`/api/sessions/${enc}`); render(); }
  async function edit(op) { await guarded(async () => { await post(`/api/sessions/${enc}/edit`, op); await reload(); }); }

  function renderHead() {
    const s = data.session;
    const titleIn = h("input", { type: "text", class: "title-in", value: s.title || "", placeholder: "Untitled lecture", "aria-label": "Title" });
    titleIn.addEventListener("change", () => guarded(() => api(`/api/sessions/${enc}`, { method: "PATCH", body: { title: titleIn.value } })));
    const courseIn = h("input", { type: "text", value: s.course || "", placeholder: "Course", "aria-label": "Course", style: "width:200px" });
    courseIn.addEventListener("change", () => guarded(() => api(`/api/sessions/${enc}`, { method: "PATCH", body: { course: courseIn.value } })));
    const acts = [];
    if (s.needs_recovery) acts.push(h("button", { class: "btn primary", onclick: () => guarded(async () => { await post(`/api/sessions/${enc}/recover`); reload(); }) }, "Recover"));
    if (s.has_audio && s.status === "recorded") acts.push(h("button", { class: "btn primary", onclick: () => guarded(async () => {
      const r = await post(`/api/sessions/${enc}/process`); toast(r.waiting_for_engine ? "Queued: it starts when you leave the recording screen." : "Processing started."); reload();
    }) }, "Make the careful transcript"));
    if (s.has_audio && s.status === "ready") acts.push(h("button", { class: "btn", onclick: async () => {
      if (await confirmDialog("Process again?", "The careful transcript is made again from the recording. Your corrections are kept in a backup file, not in the new transcript.", "Process again"))
        guarded(async () => { await post(`/api/sessions/${enc}/process`, { force: true }); reload(); });
    } }, "Process again"));
    if (data.transcript) {
      acts.push(h("a", { class: "btn", href: `/api/sessions/${enc}/export/transcript.md?token=${TOKEN}`, download: "" }, "Export transcript"));
      if (data.notes) acts.push(h("a", { class: "btn", href: `/api/sessions/${enc}/export/notes.md?token=${TOKEN}`, download: "" }, "Export notes"));
    }
    acts.push(h("button", { class: "btn", onclick: () => post(`/api/sessions/${enc}/reveal`) }, "Show in Finder"));
    acts.push(h("button", { class: "btn quiet danger", onclick: async () => {
      if (await confirmDialog("Move this session to the Trash?", "The recording, transcript and notes go to the Trash. You can put them back from there.", "Move to Trash", true))
        guarded(async () => { await post(`/api/sessions/${enc}/delete`); location.hash = "#/sessions"; });
    } }, "Delete"));
    const meta = [when(s.started), clock(s.duration_s)];
    if (s.interrupted) meta.push("stopped unexpectedly");
    const kids = [titleIn, h("div", { class: "row meta" }, courseIn, h("span", {}, meta.filter(Boolean).join(", "))), h("div", { class: "row" }, acts)];
    const proc = s.processing || {};
    if (s.status === "processing" || data.progress) {
      const p = data.progress || {};
      kids.push(h("div", { class: "progress" }, h("div", { class: "small muted" }, p.message || "Starting"),
        h("div", { class: "meter" }, h("i", { style: `width:${Math.round((p.fraction || 0) * 100)}%` }))));
    } else if (proc.error) {
      kids.push(h("p", { class: "small", style: "color:var(--rec)" }, "Processing stopped: " + proc.error));
    } else if (proc.queued && s.status === "recorded") {
      kids.push(h("p", { class: "small muted" }, "Queued for the careful transcript."));
    }
    head.replaceChildren(...kids);
  }

  function speakerMenu(anchor, items) {
    document.querySelector("menu.pop")?.remove();
    const r = anchor.getBoundingClientRect();
    const m = h("menu", { class: "pop", style: `left:${r.left}px;top:${r.bottom + 4}px` },
      items.map(([label, fn]) => h("li", {}, h("button", { onclick: () => { m.remove(); fn(); } }, label))));
    document.body.append(m);
    setTimeout(() => document.addEventListener("click", function off(ev) { if (!m.contains(ev.target)) { m.remove(); document.removeEventListener("click", off); } }), 0);
    m.querySelector("button")?.focus();
  }

  function renderSpeakers() {
    const doc = data.transcript;
    if (!doc) { spkbar.replaceChildren(); return; }
    const counts = {};
    for (const l of doc.lines) counts[l.speaker] = (counts[l.speaker] || 0) + 1;
    const chips = Object.entries(doc.speakers).filter(([id]) => counts[id]).map(([id, sp]) => {
      const more = h("button", { class: "btn quiet small", "aria-label": `Change ${sp.label}` }, "Change");
      more.addEventListener("click", () => speakerMenu(more, [
        ["Rename", async () => {
          const name = prompt(`New name for ${sp.label}:`, sp.label);
          if (name && name.trim()) edit({ op: "rename_speaker", speaker: id, label: name });
        }],
        ...Object.entries(doc.speakers).filter(([o]) => o !== id && counts[o]).map(([o, osp]) =>
          [`Same person as ${osp.label}`, () => edit({ op: "merge_speakers", source: id, target: o })]),
        ["Remember this voice", async () => {
          const name = prompt(`Remember ${sp.label}'s voice as:`, sp.role === "lecturer" || /^(Speaker|Mahasiswa) /.test(sp.label) ? "" : sp.label);
          if (name && name.trim()) guarded(async () => { await post(`/api/sessions/${enc}/remember`, { speaker: id, name }); toast(`${name} is recognised in the next lectures.`); reload(); });
        }],
      ]));
      return h("span", { class: "spkchip" }, h("b", {}, sp.label + (sp.remembered ? " ★" : "")),
        h("span", { class: "muted small" }, `${counts[id]} line${counts[id] > 1 ? "s" : ""}`), more);
    });
    const undo = doc.can_undo ? h("button", { class: "btn quiet", onclick: () => guarded(async () => { await post(`/api/sessions/${enc}/undo`); reload(); }) },
      "Undo " + ({ rename_speaker: "rename", merge_speakers: "merge", move_line: "move", edit_text: "text edit", mark_remembered: "remember" }[doc.last_edit?.op] || "last change")) : null;
    const src = h("span", { class: "muted small", style: "margin-left:auto" },
      doc.source === "final" ? "Careful transcript" : "Live transcript (rough)");
    spkbar.replaceChildren(...[...chips, undo, src].filter(Boolean));
  }

  function renderTranscript() {
    const doc = data.transcript;
    if (!doc || !doc.lines.length) {
      tx.replaceChildren(h("p", { class: "empty" }, data.session.status === "recording" ? "This lecture is being recorded."
        : data.session.has_audio ? "No transcript yet. Make the careful transcript to read and search this lecture." : "This session has no recording and no transcript."));
      lineEls = [];
      return;
    }
    let prev = null;
    lineEls = doc.lines.map(l => {
      const sp = doc.speakers[l.speaker] || { label: l.speaker };
      const ts = h("span", { class: "ts", role: "button", tabindex: 0, title: "Play from here", onclick: () => play(l.start),
        onkeydown: ev => { if (ev.key === "Enter") play(l.start); } }, clock(l.start));
      const whoEl = sp.label !== prev ? h("span", { class: "who" }, sp.label) : null;
      prev = sp.label;
      const editBtn = h("button", { class: "btn quiet small", onclick: () => startEdit(l, el) }, "Edit");
      const moveBtn = h("button", { class: "btn quiet small" }, "Someone else said this");
      moveBtn.addEventListener("click", () => speakerMenu(moveBtn, [
        ...Object.entries(doc.speakers).filter(([id]) => id !== l.speaker).map(([id, s2]) => [s2.label, () => edit({ op: "move_line", line: l.id, speaker: id })]),
        ["A new speaker", () => edit({ op: "move_line", line: l.id, speaker: "new" })]]));
      const el = h("div", { class: "ln" + (l.flagged && !l.edited ? " flag" : ""), "data-start": l.start, "data-end": l.end },
        ts, h("div", {}, whoEl, h("span", { class: "text", title: l.flagged && !l.edited ? "The speech model wasn't sure about this line" : null }, l.text),
          h("span", { class: "lacts" }, editBtn, moveBtn)));
      el.addEventListener("dblclick", ev => { if (!ev.target.closest("button,.ts")) startEdit(l, el); });
      return el;
    });
    tx.replaceChildren(...lineEls);
  }

  function startEdit(l, el) {
    editingLine = l.id;
    const ta = h("textarea", {}, l.text);
    const save = () => { editingLine = null; if (ta.value.trim() !== l.text) edit({ op: "edit_text", line: l.id, text: ta.value }); else renderTranscript(); };
    const box = h("div", {}, ta, h("div", { class: "row" }, h("button", { class: "btn primary", onclick: save }, "Save"),
      h("button", { class: "btn", onclick: () => { editingLine = null; renderTranscript(); } }, "Cancel"),
      h("button", { class: "btn quiet", onclick: () => play(l.start) }, "Play this line")));
    ta.addEventListener("keydown", ev => {
      if (ev.key === "Enter" && (ev.metaKey || ev.ctrlKey)) save();
      if (ev.key === "Escape") { editingLine = null; renderTranscript(); }
    });
    el.children[1].replaceWith(box);
    ta.focus();
  }

  function renderNotes() {
    const s = data.session, n = data.notes;
    const kids = [h("h2", {}, "Notes")];
    const genBtn = h("button", { class: "btn", onclick: () => guarded(async () => { await post(`/api/sessions/${enc}/notes`); toast("Writing notes…"); reload(); }) },
      n ? "Write the notes again" : "Write notes");
    if (s.notes_status === "generating") kids.push(h("p", { class: "loading" }, "Writing notes from the transcript"));
    else if (s.notes_status === "queued" && !n) kids.push(h("p", { class: "muted" }, s.notes_error ? "Waiting for an internet connection." : "Queued."));
    else if (s.notes_status === "skipped" && !n) kids.push(h("p", { class: "muted" }, "Not written automatically: " + (s.notes_error || "too little speech") + "."))
    else if (s.notes_status === "failed") kids.push(h("p", { class: "small", style: "color:var(--rec)" }, "Notes failed: " + (s.notes_error || "unknown error")));
    if (n) {
      kids.push(h("p", {}, n.summary));
      if (n.chapters.length) kids.push(h("h3", {}, "Chapters"), ...n.chapters.map(c => h("button", { class: "chap", onclick: () => play(c.start) },
        h("span", { class: "ts" }, clock(c.start)), h("span", { class: "ct" }, c.title), c.summary ? h("span", { class: "cs" }, c.summary) : null)));
      if (n.key_points.length) kids.push(h("h3", {}, "Key points"), h("ul", {}, n.key_points.map(p => h("li", {}, p))));
      if (n.tasks?.length) kids.push(h("h3", {}, "Tasks and deadlines"), h("ul", {}, n.tasks.map(p => h("li", {}, p))));
      if (n.terms.length) kids.push(h("h3", {}, "Terms"), h("ul", {}, n.terms.map(t => h("li", {}, h("b", {}, t.term), ": ", t.explanation))));
      if (n.questions?.length) kids.push(h("h3", {}, "Questions asked"), h("ul", {}, n.questions.map(q => h("li", {},
        q.start != null ? h("a", { href: "#", onclick: ev => { ev.preventDefault(); play(q.start); } }, clock(q.start)) : null, " ", h("b", {}, q.question), " ", q.answer))));
      kids.push(h("p", { class: "small muted" }, `Written ${when(n.generated)} from the ${n.corrected ? "corrected " : ""}transcript.`));
    }
    if (data.transcript && data.transcript.lines.length && s.notes_status !== "generating") kids.push(genBtn);
    kids.push(h("p", { class: "privacy" }, "Notes are written by Claude from the transcript text only. The recording never leaves this Mac."));
    notes.replaceChildren(...kids);
  }

  function render() { renderHead(); renderSpeakers(); if (!editingLine) renderTranscript(); renderNotes(); }

  let current = null;
  function follow() {
    const t = audio.currentTime;
    let idx = -1;
    for (let i = 0; i < lineEls.length; i++) {
      if (+lineEls[i].dataset.start <= t + 0.05) idx = i; else break;
    }
    const el = idx >= 0 && t <= +lineEls[idx].dataset.end + 1.5 ? lineEls[idx] : null;
    if (el === current) return;
    current?.classList.remove("now");
    current = el;
    if (el) {
      el.classList.add("now");
      const r = el.getBoundingClientRect(), tr = tx.getBoundingClientRect();
      if (!audio.paused && (r.top < tr.top || r.bottom > tr.bottom - 40)) el.scrollIntoView({ block: "center", behavior: "smooth" });
    }
  }
  audio.addEventListener("timeupdate", follow);
  const keys = ev => {
    if (ev.target.closest("input,textarea,select,button,[contenteditable]")) return;
    if (ev.key === " ") { ev.preventDefault(); audio.paused ? audio.play() : audio.pause(); }
    if (ev.key === "ArrowLeft") audio.currentTime -= 5;
    if (ev.key === "ArrowRight") audio.currentTime += 5;
  };
  document.addEventListener("keydown", keys);

  view.append(head, spkbar, h("div", { class: "review" }, tx, notes),
    data.session.has_audio ? h("div", { class: "player" }, back, audio, rateBtn) : null);
  render();
  const onMsg = msg => {
    if ((msg.type === "session_changed" && msg.session === sid) || (msg.type === "engine_exit" && msg.session === sid)) reload();
    if (msg.type === "jobs" && msg.current?.session === sid && msg.current.kind === "process") {
      data.progress = msg.current.progress; data.session.status = "processing"; renderHead();
    }
  };
  store.listeners.add(onMsg);
  return () => { store.listeners.delete(onMsg); document.removeEventListener("keydown", keys); audio.pause(); document.querySelector("menu.pop")?.remove(); };
}

// ---------------------------------------------------------------- go
if (!TOKEN) document.body.replaceChildren(h("main", {}, h("h1", {}, "Open Soundxx from the app"),
  h("p", { class: "muted" }, "This page needs the link the app opens (it carries a key so no other page can use it).")));
else { connect(); route(); }
