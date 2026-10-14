// Mohs Reference - chat page.
//
// Plain JavaScript, no build step. Talks to web_app.py's /api. Text from
// the user is always set with textContent; assistant answers arrive as HTML
// rendered on the server with raw HTML disabled, and the page runs under a
// strict Content-Security-Policy.

"use strict";

const $ = (id) => document.getElementById(id);
const els = {
  layout: $("layout"), sidebar: $("sidebar"), scrim: $("scrim"),
  conversations: $("conversations"), search: $("search"), newChat: $("new-chat"),
  thread: $("thread"), messages: $("messages"), empty: $("empty"), examples: $("examples"),
  emptySub: $("empty-sub"), title: $("title"), input: $("input"), send: $("send"),
  composer: $("composer"), counter: $("counter"), statusDot: $("status-dot"),
  statusText: $("status-text"), sourcesLine: $("sources-line"), testBadge: $("test-badge"),
  lightbox: $("lightbox"), lightboxImg: $("lightbox-img"), toasts: $("toasts"),
};

const state = {
  conversationId: null,   // null = a new, unsaved chat
  conversations: [],
  busy: false,
  maxChars: 2000,
};

const ICONS = {
  copy: '<svg viewBox="0 0 24 24"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/></svg>',
  check: '<svg viewBox="0 0 24 24"><path d="M5 12l5 5 9-10"/></svg>',
  up: '<svg viewBox="0 0 24 24"><path d="M7 11v9H4v-9h3zm0 0l4-8a2 2 0 0 1 3 2l-1 5h6a2 2 0 0 1 2 2.3l-1.2 6A2 2 0 0 1 17.8 20H7"/></svg>',
  down: '<svg viewBox="0 0 24 24"><path d="M7 13V4H4v9h3zm0 0l4 8a2 2 0 0 0 3-2l-1-5h6a2 2 0 0 0 2-2.3l-1.2-6A2 2 0 0 0 17.8 4H7"/></svg>',
  chev: '<svg viewBox="0 0 24 24" class="chev"><path d="M6 9l6 6 6-6"/></svg>',
  book: '<svg viewBox="0 0 24 24"><path d="M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2V5z"/><path d="M4 19a2 2 0 0 1 2-2h13"/></svg>',
  search: '<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg>',
  alert: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16h.01"/></svg>',
  pencil: '<svg viewBox="0 0 24 24"><path d="M4 20h4L19 9l-4-4L4 16v4z"/></svg>',
  trash: '<svg viewBox="0 0 24 24"><path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"/></svg>',
  image: '<svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="M21 16l-5-5-9 9"/></svg>',
};

// ------------------------------------------------------------------ api
async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
      body: options.body ? JSON.stringify(options.body) : undefined,
    });
  } catch {
    throw new Error("Cannot reach the server. Is web_app.py still running?");
  }
  let data = null;
  try { data = await response.json(); } catch { /* empty body */ }
  if (!response.ok) {
    const message = (data && (data.error || data.detail)) || `Request failed (${response.status})`;
    const error = new Error(typeof message === "string" ? message : "Request was not accepted");
    error.status = response.status;
    throw error;
  }
  return data;
}

// --------------------------------------------------------------- helpers
function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "text") node.textContent = value;
    else if (key === "html") node.innerHTML = value;           // trusted: our icons / server-rendered answers
    else if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of [].concat(children)) if (child) node.append(child);
  return node;
}

function toast(message, kind = "") {
  const node = el("div", { class: `toast ${kind}`, text: message, role: "status" });
  els.toasts.append(node);
  setTimeout(() => node.remove(), 4200);
}

function scrollToBottom(smooth = true) {
  els.thread.scrollTo({ top: els.thread.scrollHeight, behavior: smooth ? "smooth" : "auto" });
}

function dayGroup(iso) {
  const then = new Date(iso), now = new Date();
  const days = Math.floor((new Date(now.toDateString()) - new Date(then.toDateString())) / 86400000);
  if (days <= 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days < 7) return "Previous 7 days";
  if (days < 30) return "Previous 30 days";
  return "Older";
}

// --------------------------------------------------------------- sidebar
async function loadConversations() {
  try {
    state.conversations = await api("/api/conversations");
  } catch (e) {
    toast(e.message, "error");
    state.conversations = [];
  }
  renderConversations();
}

function renderConversations() {
  const query = els.search.value.trim().toLowerCase();
  const items = state.conversations.filter((c) => !query || c.title.toLowerCase().includes(query));
  els.conversations.replaceChildren();
  if (!items.length) {
    els.conversations.append(el("div", { class: "conv-empty",
      text: query ? "No chats match." : "Your conversations will appear here." }));
    return;
  }
  let group = null;
  for (const c of items) {
    const label = dayGroup(c.updated_at);
    if (label !== group) {
      group = label;
      els.conversations.append(el("div", { class: "group-label", text: label }));
    }
    els.conversations.append(conversationItem(c));
  }
}

function conversationItem(c) {
  const item = el("div", { class: `conv${c.id === state.conversationId ? " active" : ""}` });
  const link = el("a", { href: `/c/${c.id}`, text: c.title, title: c.title,
    onclick: (e) => { e.preventDefault(); openConversation(c.id); } });
  const rename = el("button", { class: "icon-btn small", "aria-label": "Rename", html: ICONS.pencil,
    onclick: () => startRename(item, c) });
  const remove = el("button", { class: "icon-btn small", "aria-label": "Delete", html: ICONS.trash,
    onclick: () => deleteConversation(c) });
  item.append(link, el("div", { class: "conv-actions" }, [rename, remove]));
  return item;
}

function startRename(item, c) {
  const input = el("input", { value: c.title, "aria-label": "Chat name", maxlength: "200" });
  item.replaceChildren(input);
  input.focus();
  input.select();
  let done = false;
  const finish = async (save) => {
    if (done) return;
    done = true;
    const title = input.value.trim();
    if (save && title && title !== c.title) {
      try {
        const updated = await api(`/api/conversations/${c.id}`, { method: "PATCH", body: { title } });
        c.title = updated.title;
        if (c.id === state.conversationId) setTitle(c.title);
      } catch (e) { toast(e.message, "error"); }
    }
    renderConversations();
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") finish(true);
    if (e.key === "Escape") finish(false);
  });
  input.addEventListener("blur", () => finish(true));
}

async function deleteConversation(c) {
  if (!confirm(`Delete "${c.title}"?`)) return;
  try {
    await api(`/api/conversations/${c.id}`, { method: "DELETE" });
    state.conversations = state.conversations.filter((x) => x.id !== c.id);
    if (c.id === state.conversationId) newChat();
    else renderConversations();
  } catch (e) { toast(e.message, "error"); }
}

function setSidebar(open) {
  els.layout.classList.toggle("sidebar-open", open);
  els.scrim.hidden = !open;
}

// ------------------------------------------------------------- threads
function setTitle(title) {
  els.title.textContent = title || "New chat";
  document.title = title ? `${title} · Mohs Reference` : "Mohs Reference";
}

function showEmpty(show) {
  els.empty.hidden = !show;
  els.messages.hidden = show;
}

function newChat() {
  state.conversationId = null;
  els.messages.replaceChildren();
  showEmpty(true);
  setTitle(null);
  history.pushState({}, "", "/");
  renderConversations();
  setSidebar(false);
  els.input.focus();
}

async function openConversation(id, push = true) {
  if (state.busy) { toast("Wait for the current answer to finish."); return; }
  try {
    const conv = await api(`/api/conversations/${id}`);
    state.conversationId = conv.id;
    setTitle(conv.title);
    els.messages.replaceChildren(...conv.messages.map(renderMessage));
    showEmpty(conv.messages.length === 0);
    if (push) history.pushState({ id }, "", `/c/${id}`);
    renderConversations();
    setSidebar(false);
    requestAnimationFrame(() => scrollToBottom(false));
  } catch (e) {
    toast(e.status === 404 ? "That conversation no longer exists." : e.message, "error");
    newChat();
  }
}

// ------------------------------------------------------------ messages
function renderMessage(m) {
  if (m.role === "user") {
    return el("div", { class: "msg user", "data-id": m.id }, el("div", { class: "bubble", text: m.text }));
  }
  const body = el("div", { class: "body" });
  const row = el("div", { class: "msg assistant", "data-id": m.id }, [
    el("img", { class: "avatar", src: "/static/icon.svg", alt: "" }), body]);
  if (m.status === "error") {
    body.append(errorBox(m));
    return row;
  }
  if (m.searched_as) {
    body.append(el("div", { class: "searched-as", html: ICONS.search },
      el("span", { text: `Searched as: ${m.searched_as}` })));
  }
  const answer = el("div", { class: "answer", html: m.html });
  body.append(answer);
  if (m.figures && m.figures.length) body.append(figures(m.figures));
  if (m.sources && m.sources.length) {
    const panel = sourcesPanel(m.sources);
    body.append(panel);
    wireCitations(answer, panel, m.sources);
  }
  body.append(actions(m));
  return row;
}

function errorBox(m) {
  const box = el("div", { class: "error-box", html: ICONS.alert });
  box.append(el("span", { text: m.text }));
  box.append(el("button", { class: "retry", text: "Retry", onclick: () => retry(m.id, box) }));
  return box;
}

function figures(urls) {
  return el("div", { class: "figures" }, urls.map((url) =>
    el("button", { "aria-label": "Open figure", onclick: () => openLightbox(url) },
      el("img", { src: url, alt: "Figure from the source", loading: "lazy" }))));
}

function sourcesPanel(sources) {
  const cited = sources.filter((s) => s.cited).length;
  const panel = el("div", { class: "sources" });
  const toggle = el("button", { class: "sources-toggle", "aria-expanded": "false",
    onclick: () => {
      const open = panel.classList.toggle("open");
      toggle.setAttribute("aria-expanded", String(open));
    } });
  toggle.innerHTML = ICONS.book;
  toggle.append(el("span", { text: `${sources.length} passages searched${cited ? ` · ${cited} cited` : ""}` }));
  toggle.insertAdjacentHTML("beforeend", ICONS.chev);
  const list = el("div", { class: "sources-list" }, sources.map((s) => {
    const card = el("button", { class: "source", "data-page": s.page ?? "", "data-rank": s.rank,
      onclick: () => card.classList.toggle("expanded") }, [
      el("div", { class: "source-head" }, [
        el("span", { class: "n", text: `${s.rank}` }),
        el("span", { text: [s.label, s.where].filter(Boolean).join(" · ") }),
        s.kind && s.kind !== "prose" ? el("span", { class: "kind", text: s.kind }) : null,
        s.cited ? el("span", { class: "cited", text: "cited" }) : null,
      ]),
      el("p", { class: "source-text", text: s.text }),
    ]);
    return card;
  }));
  panel.append(toggle, list);
  return panel;
}

// Clicking a citation chip opens the sources and highlights the passage.
function wireCitations(answer, panel, sources) {
  answer.querySelectorAll(".cite").forEach((chip) => {
    chip.setAttribute("role", "button");
    chip.setAttribute("tabindex", "0");
    const go = () => {
      const pages = (chip.textContent.match(/\d+/g) || []).map(Number);
      panel.classList.add("open");
      const cards = [...panel.querySelectorAll(".source")];
      const isNumbered = /^\[|【/.test(chip.textContent);
      const match = cards.find((card) => isNumbered
        ? pages.includes(Number(card.dataset.rank))
        : pages.includes(Number(card.dataset.page)));
      if (match) {
        match.scrollIntoView({ behavior: "smooth", block: "center" });
        match.classList.add("flash");
        setTimeout(() => match.classList.remove("flash"), 1400);
      } else {
        toast("That page was cited but is not among the passages shown.");
      }
    };
    chip.addEventListener("click", go);
    chip.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
  });
}

function actions(m) {
  const row = el("div", { class: "actions" });
  const copy = el("button", { class: "icon-btn small", "aria-label": "Copy answer", html: ICONS.copy,
    onclick: async () => {
      try {
        await navigator.clipboard.writeText(m.text);
        copy.innerHTML = ICONS.check;
        setTimeout(() => { copy.innerHTML = ICONS.copy; }, 1500);
      } catch { toast("Copy is not available here."); }
    } });
  const up = el("button", { class: `icon-btn small${m.feedback === 1 ? " on" : ""}`,
    "aria-label": "Helpful", html: ICONS.up });
  const down = el("button", { class: `icon-btn small${m.feedback === -1 ? " on" : ""}`,
    "aria-label": "Not helpful", html: ICONS.down });
  const note = el("span", { class: "done" });
  let value = m.feedback;

  const send = async (next, comment = null) => {
    try {
      await api(`/api/messages/${m.id}/feedback`, { method: "POST", body: { value: next, comment } });
      value = next;
      up.classList.toggle("on", value === 1);
      down.classList.toggle("on", value === -1);
      note.textContent = next ? "Thanks for the feedback" : "";
      setTimeout(() => { note.textContent = ""; }, 2500);
    } catch (e) { toast(e.message, "error"); }
  };
  up.addEventListener("click", () => send(value === 1 ? null : 1));
  down.addEventListener("click", () => {
    if (value === -1) { send(null); return; }
    send(-1);
    if (row.nextElementSibling && row.nextElementSibling.classList.contains("feedback-form")) return;
    const input = el("input", { placeholder: "What was wrong or missing? (optional)", maxlength: "2000" });
    const form = el("form", { class: "feedback-form" }, [input, el("button", { type: "submit", text: "Send" })]);
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      send(-1, input.value.trim() || null);
      form.remove();
    });
    row.after(form);
    input.focus();
  });
  row.append(copy, up, down, note);
  return row;
}

function thinking() {
  const steps = ["Searching the sources", "Reading the passages", "Writing the answer"];
  const label = el("span", { text: steps[0] });
  const node = el("div", { class: "msg assistant pending" }, [
    el("img", { class: "avatar", src: "/static/icon.svg", alt: "" }),
    el("div", { class: "body" }, el("div", { class: "thinking" }, [
      el("span", { class: "dots" }, [el("i"), el("i"), el("i")]), label])),
  ]);
  let i = 0;
  node.timer = setInterval(() => { i = Math.min(i + 1, steps.length - 1); label.textContent = steps[i]; }, 2600);
  return node;
}

// ---------------------------------------------------------------- send
function setBusy(busy) {
  state.busy = busy;
  els.send.classList.toggle("busy", busy);
  updateComposer();
}

function updateComposer() {
  const length = els.input.value.trim().length;
  const over = els.input.value.length > state.maxChars;
  els.send.disabled = state.busy || length === 0 || over;
  const near = els.input.value.length > state.maxChars * 0.8;
  els.counter.textContent = near ? `${els.input.value.length} / ${state.maxChars}` : "";
  els.counter.classList.toggle("over", over);
  fitInput();
}

// Grow with the text, up to 220px. Empty: the natural one-row height - a
// scrollHeight read before first layout can be far too large.
function fitInput() {
  els.input.style.height = "";
  if (els.input.value) els.input.style.height = `${Math.min(els.input.scrollHeight, 220)}px`;
}
window.addEventListener("resize", fitInput);

async function sendMessage(text) {
  text = text.trim();
  if (!text || state.busy) return;
  setBusy(true);
  showEmpty(false);
  const optimistic = renderMessage({ role: "user", text, id: "pending" });
  const pending = thinking();
  els.messages.append(optimistic, pending);
  els.input.value = "";
  updateComposer();
  scrollToBottom();

  try {
    const target = state.conversationId || "new";
    const data = await api(`/api/conversations/${target}/messages`, { method: "POST", body: { content: text } });
    const isNew = !state.conversationId;
    state.conversationId = data.conversation.id;
    setTitle(data.conversation.title);
    optimistic.replaceWith(renderMessage(data.user));
    pending.replaceWith(renderMessage(data.assistant));
    if (isNew) history.pushState({ id: state.conversationId }, "", `/c/${state.conversationId}`);
    loadConversations();
  } catch (e) {
    // Refused (422: empty, too long, patient data) or unreachable: nothing
    // was saved, so give the text back to edit.
    optimistic.remove();
    pending.remove();
    if (!els.messages.children.length) showEmpty(true);
    els.input.value = text;
    updateComposer();
    toast(e.message, "error");
  } finally {
    clearInterval(pending.timer);
    setBusy(false);
    scrollToBottom();
    els.input.focus();
  }
}

async function retry(messageId, box) {
  if (state.busy) return;
  setBusy(true);
  const row = box.closest(".msg");
  const pending = thinking();
  row.replaceWith(pending);
  try {
    const data = await api(`/api/messages/${messageId}/retry`, { method: "POST" });
    pending.replaceWith(renderMessage(data.assistant));
  } catch (e) {
    pending.replaceWith(row);
    toast(e.message, "error");
  } finally {
    clearInterval(pending.timer);
    setBusy(false);
    scrollToBottom();
  }
}

// ------------------------------------------------------------ lightbox
function openLightbox(url) {
  els.lightboxImg.src = url;
  els.lightbox.hidden = false;
}
function closeLightbox() {
  els.lightbox.hidden = true;
  els.lightboxImg.removeAttribute("src");
}

// -------------------------------------------------------------- status
async function pollHealth() {
  try {
    const h = await api("/api/health");
    els.statusDot.className = `dot ${h.status}`;
    const text = { ready: "Ready", warming: "Loading the search index…", error: h.error || "Index error" };
    els.statusText.textContent = h.llm_configured ? text[h.status] : "Language model not configured";
    if (!h.llm_configured) els.statusDot.className = "dot error";
    if (h.status === "warming") setTimeout(pollHealth, 1500);
  } catch {
    els.statusDot.className = "dot error";
    els.statusText.textContent = "Server unreachable";
    setTimeout(pollHealth, 5000);
  }
}

async function loadMeta() {
  try {
    const meta = await api("/api/meta");
    state.maxChars = meta.max_chars;
    els.testBadge.hidden = !meta.test_mode;
    const n = meta.sources.length;
    els.sourcesLine.textContent = n === 1 ? `Searching: ${meta.sources[0]}` : `Searching ${n} sources: ${meta.sources.join(", ")}`;
    els.emptySub.textContent = n === 1
      ? `Answers come only from the ${meta.sources[0]}, with page citations.`
      : `Answers come only from ${n} indexed sources, with citations.`;
    const blurbs = ["Site", "Technique", "Site", "Grafts"];
    els.examples.replaceChildren(...meta.examples.map((q, i) =>
      el("button", { class: "example", onclick: () => sendMessage(q) }, [
        el("strong", { text: blurbs[i] || "Example" }), el("span", { text: q })])));
  } catch (e) { toast(e.message, "error"); }
}

// ---------------------------------------------------------------- wire
els.composer.addEventListener("submit", (e) => { e.preventDefault(); sendMessage(els.input.value); });
els.input.addEventListener("input", updateComposer);
els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    if (!els.send.disabled) sendMessage(els.input.value);
  }
});
els.newChat.addEventListener("click", newChat);
els.search.addEventListener("input", renderConversations);
$("open-sidebar").addEventListener("click", () => setSidebar(true));
$("close-sidebar").addEventListener("click", () => setSidebar(false));
els.scrim.addEventListener("click", () => setSidebar(false));
$("brand").addEventListener("click", (e) => { e.preventDefault(); newChat(); });
$("lightbox-close").addEventListener("click", closeLightbox);
els.lightbox.addEventListener("click", (e) => { if (e.target === els.lightbox) closeLightbox(); });
els.thread.addEventListener("scroll", () =>
  document.querySelector(".topbar").classList.toggle("scrolled", els.thread.scrollTop > 4));
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !els.lightbox.hidden) closeLightbox();
  if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key.toLowerCase() === "o") { e.preventDefault(); newChat(); }
});
window.addEventListener("popstate", () => {
  const match = location.pathname.match(/^\/c\/([0-9a-f]+)$/);
  if (match) openConversation(match[1], false); else newChat();
});

(async function start() {
  loadMeta();
  pollHealth();
  await loadConversations();
  const match = location.pathname.match(/^\/c\/([0-9a-f]+)$/);
  if (match) openConversation(match[1], false);
  else { showEmpty(true); els.input.focus(); }
  updateComposer();
})();
