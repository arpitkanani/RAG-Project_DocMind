"use strict";

// ═══════════════════════════════════════════════════════════════════
//  STATE  —  three isolated state objects, never mixed
// ═══════════════════════════════════════════════════════════════════

/**
 * Everything that belongs to the active workspace (session).
 * @typedef {{
 *   localId: string,
 *   name: string,
 *   collection: string,
 *   type: 'doc'|'yt',
 *   status: 'processing'|'ready',
 *   isNew: boolean
 * }} WorkspaceSource
 */
const workspaceState = {
  /** @type {string|null} */
  sessionId: null,
  /** @type {WorkspaceSource[]} */
  sources: [],
};

/** Ephemeral state tied to the current input / upload cycle. */
const conversationState = {
  isLoading: false,
  uploadInProgress: false,
};

/** Sidebar-only state. */
const sidebarState = {
  chats: [],
  searchQuery: "",
  /** @type {Set<string>} */
  deletingIds: new Set(),
};

// ═══════════════════════════════════════════════════════════════════
//  API
// ═══════════════════════════════════════════════════════════════════

const API = {
  upload: "/upload",
  youtube: "/youtube",
  query: "/query",
  sessions: "/sessions",
  memory: "/memory",
};

const SESSION_KEY = "docmind.activeSessionId";

// ═══════════════════════════════════════════════════════════════════
//  BOOT
// ═══════════════════════════════════════════════════════════════════

document.addEventListener("DOMContentLoaded", async () => {
  bindUI();
  await bootApp();
});

async function bootApp() {
  const sessions = await loadSessions();
  const storedId = localStorage.getItem(SESSION_KEY);

  // 1. Try the stored session first (browser-refresh recovery)
  if (storedId) {
    const ok = await restoreSession(storedId);
    if (ok) return;
    localStorage.removeItem(SESSION_KEY);
  }

  // 2. Fall back to the most-recent available session
  const fallback = sessions.find((s) => s.session_id !== storedId);
  if (fallback) {
    const ok = await restoreSession(fallback.session_id);
    if (ok) return;
  }

  // 3. Nothing to restore — open a fresh workspace
  await createNewSession();
  renderWelcomeOnly();
}

// ═══════════════════════════════════════════════════════════════════
//  UI BINDING
// ═══════════════════════════════════════════════════════════════════

function bindUI() {
  // Sidebar
  el("newChatBtn").addEventListener("click", handleNewChat);
  el("searchInput").addEventListener("input", (e) => {
    sidebarState.searchQuery = e.target.value.toLowerCase();
    renderSidebar();
  });
  el("clearAllBtn").addEventListener("click", handleClearAll);

  // Modal triggers
  el("attachBtn").addEventListener("click", openModal);
  el("modalClose").addEventListener("click", closeModal);
  el("cancelUpload").addEventListener("click", closeModal);
  el("submitUpload").addEventListener("click", handleSubmitUpload);
  el("modalBackdrop").addEventListener("click", (e) => {
    if (e.target === el("modalBackdrop")) closeModal();
  });

  // Mobile sidebar toggle
  const sidebarToggle = document.getElementById("sidebarToggle");
  if (sidebarToggle) {
    sidebarToggle.addEventListener("click", () => {
      document.getElementById("sidebar").classList.toggle("mobile-open");
    });
  }

  // Modal tabs
  document.querySelectorAll(".modal-tab").forEach((btn) =>
    btn.addEventListener("click", () => switchTab(btn.dataset.tab))
  );

  // Drop zone
  const dropZone = el("dropZone");
  dropZone.addEventListener("click", () => {
    if (conversationState.uploadInProgress)
      return showToast("Please wait for the current upload to finish.", "error");
    el("fileInput").click();
  });
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });
  dropZone.addEventListener("dragleave", () =>
    dropZone.classList.remove("drag-over")
  );
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (conversationState.uploadInProgress)
      return showToast("Please wait for the current upload to finish.", "error");
    handleFiles(Array.from(e.dataTransfer.files || []));
    closeModal();
  });
  el("fileInput").addEventListener("change", (e) => {
    if (conversationState.uploadInProgress)
      return showToast("Please wait for the current upload to finish.", "error");
    handleFiles(Array.from(e.target.files || []));
    closeModal();
    e.target.value = "";
  });

  // Composer
  el("sendBtn").addEventListener("click", sendMessage);
  el("queryInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  el("queryInput").addEventListener("input", function () {
    autoResize.call(this);
    updateSendButton();
  });
}

// ═══════════════════════════════════════════════════════════════════
//  SESSION MANAGEMENT
// ═══════════════════════════════════════════════════════════════════

async function createNewSession() {
  try {
    const data = await apiFetch(API.sessions, { method: "POST" });
    workspaceState.sessionId = data?.session_id || "default";
  } catch {
    workspaceState.sessionId = "default";
  }
  workspaceState.sources = [];
  localStorage.setItem(SESSION_KEY, workspaceState.sessionId);
  renderComposerChips();
  updateSendButton();
}

/**
 * Load a session from the backend and restore full workspace state.
 * @returns {Promise<boolean>} true if successfully restored.
 */
async function restoreSession(sessionId) {
  const data = await apiFetch(
    `${API.sessions}/${encodeURIComponent(sessionId)}`
  );
  if (!data || data.error_code) return false;

  workspaceState.sessionId = sessionId;

  // Restore workspace sources from backend attachments
  workspaceState.sources = (data.attachments || []).map((a) => ({
    localId: uid(),
    name: a.name,
    collection: a.collection,
    type: a.type || "doc",
    status: "ready",
    isNew: false, // restored sources are NOT new — they won't attach to next message
  }));

  localStorage.setItem(SESSION_KEY, sessionId);

  // Restore message thread
  clearMessages();
  const msgs = Array.isArray(data.messages) ? data.messages : [];
  if (!msgs.length) {
    renderWelcomeOnly();
  } else {
    msgs.forEach((m) =>
      appendMessage(
        m.content,
        m.role === "human" ? "user" : "ai",
        m.role === "human" && Array.isArray(m.attachments) ? m.attachments : []
      )
    );
  }

  renderComposerChips();
  updateSendButton();
  await loadSessions();
  return true;
}

async function loadSessions() {
  const data = await apiFetch(API.sessions);
  sidebarState.chats = data?.sessions || [];
  renderSidebar();
  return sidebarState.chats;
}

async function handleNewChat() {
  await createNewSession();
  renderWelcomeOnly();
  await loadSessions();
}

async function switchSession(sessionId) {
  const ok = await restoreSession(sessionId);
  if (!ok) {
    showToast("Could not load that chat.", "error");
    localStorage.removeItem(SESSION_KEY);
    await loadSessions();
  }
}

async function deleteSession(sessionId) {
  if (!confirm("Delete this chat and its related document data?")) return;

  sidebarState.deletingIds.add(sessionId);
  renderSidebar();

  try {
    const data = await apiFetch(
      `${API.sessions}/${encodeURIComponent(sessionId)}`,
      { method: "DELETE" }
    );

    if (!data?.success) {
      showToast(data?.message || "Could not delete chat.", "error");
      return;
    }

    sidebarState.chats = sidebarState.chats.filter(
      (c) => c.session_id !== sessionId
    );

    if (workspaceState.sessionId === sessionId) {
      localStorage.removeItem(SESSION_KEY);
      await createNewSession();
      renderWelcomeOnly();
    }

    await loadSessions();
    showToast("Chat deleted.", "success");
  } catch {
    showToast("Could not delete chat.", "error");
  } finally {
    sidebarState.deletingIds.delete(sessionId);
    renderSidebar();
  }
}

async function handleClearAll() {
  if (!confirm("Clear all chats and all related document data?")) return;
  try {
    const data = await apiFetch(API.memory, { method: "DELETE" });
    if (!data?.success) {
      showToast(data?.message || "Could not clear history.", "error");
      return;
    }
    localStorage.removeItem(SESSION_KEY);
    await createNewSession();
    renderWelcomeOnly();
    await loadSessions();
    showToast("All chat history and document data cleared.", "success");
  } catch {
    showToast("Could not clear history.", "error");
  }
}

// ═══════════════════════════════════════════════════════════════════
//  UPLOAD / YOUTUBE
// ═══════════════════════════════════════════════════════════════════

let _activeTab = "file";

function openModal() {
  if (conversationState.uploadInProgress)
    return showToast("Please wait for the current upload to finish.", "error");
  el("modalBackdrop").classList.add("open");
  switchTab("file");
}

function closeModal() {
  el("modalBackdrop").classList.remove("open");
  el("ytInput").value = "";
  el("fileInput").value = "";
}

function switchTab(tab) {
  _activeTab = tab;
  document.querySelectorAll(".modal-tab").forEach((b) =>
    b.classList.toggle("active", b.dataset.tab === tab)
  );
  document.querySelectorAll(".tab-panel").forEach((p) =>
    p.classList.toggle("active", p.id === `tab-${tab}`)
  );
  const btn = document.getElementById("submitUpload");
  if (btn) {
    btn.textContent = tab === "yt" ? "Add Video" : "Browse Files";
  }
}

async function handleSubmitUpload() {
  if (conversationState.uploadInProgress)
    return showToast("Please wait for the current upload to finish.", "error");
  if (_activeTab === "yt") {
    await handleYouTube();
  } else {
    el("fileInput").click();
  }
}

async function handleFiles(files) {
  if (!files.length) return;
  for (const file of files) await uploadFile(file);
}

async function uploadFile(file) {
  const localId = uid();

  // 1. Immediately render processing chip above textarea
  workspaceState.sources.push({
    localId,
    name: file.name,
    collection: "",
    type: "doc",
    status: "processing",
    isNew: true,
  });
  setUploadBusy(true);
  renderComposerChips();

  try {
    const form = new FormData();
    form.append("file", file);
    form.append("session_id", workspaceState.sessionId || "default");

    const data = await apiFetch(API.upload, { method: "POST", body: form });

    if (!data?.success) {
      removeLocalSource(localId);
      setUploadBusy(hasProcessing());
      renderComposerChips();
      showToast(
        data?.message || data?.error || `Failed to upload ${file.name}.`,
        "error"
      );
      return;
    }

    // 2. Transition chip to ready state
    const src = workspaceState.sources.find((s) => s.localId === localId);
    if (src) {
      src.collection = data.collection_name;
      src.status = "ready";
      // isNew stays true — will attach to the NEXT message the user sends
    }

    setUploadBusy(hasProcessing());
    renderComposerChips();
    await loadSessions();
    showToast(`${file.name} indexed successfully.`, "success");
  } catch {
    removeLocalSource(localId);
    setUploadBusy(hasProcessing());
    renderComposerChips();
    showToast(`Upload failed for ${file.name}.`, "error");
  }
}

async function handleYouTube() {
  const url = el("ytInput").value.trim();
  if (!url) return showToast("Please enter a YouTube URL.", "error");
  closeModal();

  const localId = uid();

  workspaceState.sources.push({
    localId,
    name: "YouTube transcript",
    collection: "",
    type: "yt",
    status: "processing",
    isNew: true,
  });
  setUploadBusy(true);
  renderComposerChips();

  try {
    const data = await apiFetch(API.youtube, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, session_id: workspaceState.sessionId }),
    });

    if (!data?.success) {
      removeLocalSource(localId);
      setUploadBusy(hasProcessing());
      renderComposerChips();
      showToast(data?.message || "Could not process YouTube URL.", "error");
      return;
    }

    const src = workspaceState.sources.find((s) => s.localId === localId);
    if (src) {
      src.name = data.video_id ? `YouTube ${data.video_id}` : "YouTube transcript";
      src.collection = data.collection_name;
      src.status = "ready";
    }

    setUploadBusy(hasProcessing());
    renderComposerChips();
    await loadSessions();
    showToast("YouTube transcript indexed.", "success");
  } catch {
    removeLocalSource(localId);
    setUploadBusy(hasProcessing());
    renderComposerChips();
    showToast("Could not process YouTube URL.", "error");
  }
}

/**
 * Remove a source from the workspace (DELETE endpoint + UI update).
 * @param {string} collection
 */
async function removeSource(collection) {
  if (!collection) return;
  try {
    const data = await apiFetch(
      `${API.sessions}/${encodeURIComponent(
        workspaceState.sessionId
      )}/attachments/${encodeURIComponent(collection)}`,
      { method: "DELETE" }
    );

    if (!data?.success) {
      showToast(data?.message || "Could not remove the document.", "error");
      return;
    }

    workspaceState.sources = workspaceState.sources.filter(
      (s) => s.collection !== collection
    );
    renderComposerChips();
    await loadSessions();
    showToast("Source removed from this workspace.", "success");
  } catch {
    showToast("Could not remove the document.", "error");
  }
}

// ═══════════════════════════════════════════════════════════════════
//  SEND MESSAGE
// ═══════════════════════════════════════════════════════════════════

async function sendMessage() {
  const input = el("queryInput");
  const query = input.value.trim();
  if (!query) return;
  if (conversationState.uploadInProgress)
    return showToast(
      "Please wait until the source finishes processing.",
      "error"
    );
  if (conversationState.isLoading) return;

  // Guard: block query when workspace is completely empty
  const readySources = workspaceState.sources.filter(
    (s) => s.status === "ready" && s.collection
  );
  if (!readySources.length) {
    showToast(
      "Please upload a document or add a YouTube source first.",
      "error"
    );
    return;
  }

  // Collect newly-added chips — these will appear alongside this message in the chat.
  // Sources uploaded in previous messages (isNew = false) are NOT duplicated here.
  const newSources = workspaceState.sources.filter(
    (s) => s.isNew && s.status === "ready" && s.collection
  );
  const messageAttachments = newSources.map((s) => ({
    name: s.name,
    collection: s.collection,
    type: s.type,
  }));

  // Reset isNew BEFORE rendering so composer chips disappear for these sources
  workspaceState.sources.forEach((s) => {
    s.isNew = false;
  });

  hideWelcome();
  appendMessage(query, "user", messageAttachments);

  input.value = "";
  input.style.height = "auto";
  renderComposerChips(); // re-render — chips that were isNew are now gone from composer
  updateSendButton();

  conversationState.isLoading = true;
  const typingId = showTyping();

  try {
    const data = await apiFetch(API.query, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        session_id: workspaceState.sessionId,
        message_attachments: messageAttachments.length ? messageAttachments : null,
      }),
    });

    removeTyping(typingId);

    if (!data || data.error_code) {
      if (data?.error_code === "collection_not_found") {
        const missing = Array.isArray(data.missing_collections)
          ? data.missing_collections
          : [];
        workspaceState.sources = workspaceState.sources.filter(
          (s) => !missing.includes(s.collection)
        );
        renderComposerChips();
        await loadSessions();
        showToast(
          "One or more sources were removed. Please upload again if needed.",
          "error"
        );
        return;
      }
      if (data?.error_code === "knowledge_base_empty") {
        showToast(
          data.message || "Please upload a document or add a YouTube source first.",
          "error"
        );
        return;
      }
      showToast(data?.message || "Could not complete that request.", "error");
      return;
    }

    workspaceState.sessionId = data.session_id || workspaceState.sessionId;
    localStorage.setItem(SESSION_KEY, workspaceState.sessionId);
    appendMessage(
      data.answer || "I couldn't generate a response. Please try again.",
      "ai"
    );

    try {
      await loadSessions();
    } catch {
      // Non-critical — answer is already shown
    }
  } catch {
    removeTyping(typingId);
    showToast("Server is down. Please try again.", "error");
  } finally {
    conversationState.isLoading = false;
    updateSendButton();
  }
}

// ═══════════════════════════════════════════════════════════════════
//  RENDER — COMPOSER CHIPS
//
//  Shows ALL workspace sources above the textarea.
//  Each chip has an X button to remove the source from the workspace.
//  Chips with isNew=true will attach to the next sent message.
//  After send, isNew resets to false and chips remain visible for
//  ongoing workspace management (remove / reference).
// ═══════════════════════════════════════════════════════════════════

function renderComposerChips() {
  const row = el("chipsRow");

  if (!workspaceState.sources.length) {
    row.hidden = true;
    row.innerHTML = "";
    return;
  }

  row.hidden = false;
  row.innerHTML = workspaceState.sources
    .map((src) => {
      const processing = src.status === "processing";
      const typeLabel = src.type === "yt" ? "YT" : "DOC";
      const removeBtn = processing
        ? `<span class="chip-indicator processing" aria-label="Processing"></span>`
        : `<span class="chip-indicator ready" aria-label="Ready"></span>
           <button class="chip-remove" data-collection="${escHtml(src.collection)}" title="Remove source">
             <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3.5">
               <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
             </svg>
           </button>`;

      return `
        <div class="chip ${processing ? "processing" : ""}" data-local-id="${escHtml(src.localId)}">
          <span class="chip-type">${typeLabel}</span>
          <span class="chip-name" title="${escHtml(src.name)}">${escHtml(src.name)}</span>
          ${removeBtn}
        </div>`;
    })
    .join("");

  row.querySelectorAll(".chip-remove").forEach((btn) => {
    btn.addEventListener("click", () => removeSource(btn.dataset.collection));
  });
}

// ═══════════════════════════════════════════════════════════════════
//  RENDER — MESSAGES
// ═══════════════════════════════════════════════════════════════════

/**
 * Append a chat message to the thread.
 * @param {string} content
 * @param {'user'|'ai'} role
 * @param {Array} attachments  — only populated on user messages when new files were added
 */
function appendMessage(content, role, attachments = []) {
  const container = el("messages");
  const row = document.createElement("div");
  row.className = `msg-row ${role}`;

  // Only user messages can carry attachment chips; AI messages never do.
  const chipsHtml =
    role === "user" && attachments.length
      ? `<div class="message-attachments">${buildReadOnlyChips(attachments)}</div>`
      : "";

  row.innerHTML = `
    <div class="msg-avatar">${role === "user" ? "You" : "AI"}</div>
    <div class="msg-body">
      ${chipsHtml}
      <div class="msg-bubble">${formatContent(content)}</div>
    </div>`;

  container.appendChild(row);
  container.scrollTop = container.scrollHeight;
}

function buildReadOnlyChips(chips) {
  return chips
    .map(
      (c) => `
      <div class="chip message-chip">
        <span class="chip-type">${c.type === "yt" ? "YT" : "DOC"}</span>
        <span class="chip-name" title="${escHtml(c.name)}">${escHtml(c.name)}</span>
      </div>`
    )
    .join("");
}

// ═══════════════════════════════════════════════════════════════════
//  RENDER — SIDEBAR
// ═══════════════════════════════════════════════════════════════════

function renderSidebar() {
  const container = el("sidebarChats");
  const q = sidebarState.searchQuery;
  const chats = q
    ? sidebarState.chats.filter((c) =>
        (c.title || c.session_id || "").toLowerCase().includes(q)
      )
    : sidebarState.chats;

  if (!chats.length) {
    container.innerHTML = `<div class="empty-sidebar">No chats yet. Start a new conversation.</div>`;
    return;
  }

  const now = Date.now();
  const groups = { Today: [], Yesterday: [], "Previous 7 Days": [] };

  chats.forEach((c) => {
    const ms = new Date(c.last_active || 0).getTime();
    const days = (now - ms) / 86_400_000;
    if (days < 1) groups.Today.push(c);
    else if (days < 2) groups.Yesterday.push(c);
    else groups["Previous 7 Days"].push(c);
  });

  let html = "";
  Object.entries(groups).forEach(([label, items]) => {
    if (!items.length) return;
    html += `<div class="chat-group-label">${label}</div>`;
    items.forEach((c) => {
      const active = c.session_id === workspaceState.sessionId ? "active" : "";
      const deleting = sidebarState.deletingIds.has(c.session_id);
      html += `
        <div class="chat-item ${active}" data-id="${escHtml(c.session_id)}">
          <span class="chat-item-text">${escHtml(c.title || "New Chat")}</span>
          <span class="chat-item-time">${escHtml(
            c.last_active_label || formatTime(c.last_active)
          )}</span>
          <button
            class="chat-item-del"
            data-del="${escHtml(c.session_id)}"
            title="Delete chat"
            ${deleting ? "disabled" : ""}
          >${deleting ? "..." : "×"}</button>
        </div>`;
    });
  });

  container.innerHTML = html;

  container.querySelectorAll(".chat-item").forEach((item) => {
    item.addEventListener("click", (e) => {
      if (e.target.closest("[data-del]")) return;
      switchSession(item.dataset.id);
    });
  });

  container.querySelectorAll("[data-del]").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      deleteSession(btn.dataset.del);
    });
  });
}

// ═══════════════════════════════════════════════════════════════════
//  RENDER — WELCOME / MESSAGES
// ═══════════════════════════════════════════════════════════════════

function showTyping() {
  const container = el("messages");
  const id = `typing-${Date.now()}`;
  const row = document.createElement("div");
  row.className = "typing-row";
  row.id = id;
  row.innerHTML = `
    <div class="msg-avatar msg-avatar-ai">AI</div>
    <div class="typing-bubble">
      <div class="typing-label">Searching your workspace sources...</div>
      <div class="typing-dots"><span></span><span></span><span></span></div>
    </div>`;
  container.appendChild(row);
  container.scrollTop = container.scrollHeight;
  return id;
}

function removeTyping(id) {
  const node = document.getElementById(id);
  if (node) node.remove();
}

function showWelcome() {
  let welcome = el("welcome");
  if (!welcome) {
    welcome = document.createElement("div");
    welcome.id = "welcome";
    welcome.className = "welcome";
    welcome.innerHTML = `
      <div class="welcome-logo">DM</div>
      <h2>Analyze documents with a calm, focused workspace</h2>
      <p>Upload a file or YouTube transcript, then ask grounded questions without losing your document context.</p>
      <div class="suggestions">
        <button class="suggest-card" onclick="insertSuggestion('Summarize the main ideas in this document')">
          <span>01</span> Summarize the document
        </button>
        <button class="suggest-card" onclick="insertSuggestion('List the key points and action items')">
          <span>02</span> Extract key points
        </button>
      </div>`;
    el("messages").prepend(welcome);
  }
  welcome.style.display = "flex";
}

function hideWelcome() {
  const w = el("welcome");
  if (w) w.style.display = "none";
}

function clearMessages() {
  el("messages").innerHTML = "";
}

function renderWelcomeOnly() {
  clearMessages();
  showWelcome();
}

// ═══════════════════════════════════════════════════════════════════
//  UPLOAD STATE HELPERS
// ═══════════════════════════════════════════════════════════════════

function setUploadBusy(busy) {
  conversationState.uploadInProgress = busy;
  el("attachBtn").disabled = busy;
  el("submitUpload").disabled = busy;
  updateSendButton();
}

function hasProcessing() {
  return workspaceState.sources.some((s) => s.status === "processing");
}

function removeLocalSource(localId) {
  workspaceState.sources = workspaceState.sources.filter(
    (s) => s.localId !== localId
  );
}

function updateSendButton() {
  const btn = el("sendBtn");
  const query = el("queryInput").value.trim();
  const canSend =
    Boolean(query) &&
    !conversationState.uploadInProgress &&
    !conversationState.isLoading;
  btn.disabled = !canSend;
  btn.classList.toggle("ready", canSend);
}

// ═══════════════════════════════════════════════════════════════════
//  FILTER / SUGGESTIONS
// ═══════════════════════════════════════════════════════════════════

function insertSuggestion(text) {
  const input = el("queryInput");
  input.value = text;
  input.focus();
  autoResize.call(input);
  updateSendButton();
}

window.insertSuggestion = insertSuggestion;

// ═══════════════════════════════════════════════════════════════════
//  TOAST
// ═══════════════════════════════════════════════════════════════════

function showToast(message, type = "success") {
  const stack = el("toastStack");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  stack.appendChild(toast);
  // Fade out then remove
  setTimeout(() => toast.classList.add("toast-fade"), 3400);
  setTimeout(() => toast.remove(), 4000);
}

// ═══════════════════════════════════════════════════════════════════
//  CONTENT FORMATTING
// ═══════════════════════════════════════════════════════════════════

function formatContent(text) {
  if (!text) return "<p></p>";

  const lines = escHtml(text).split("\n");
  const parts = [];
  let listType = null;

  const closeList = () => {
    if (listType) {
      parts.push(listType === "ol" ? "</ol>" : "</ul>");
      listType = null;
    }
  };

  lines.forEach((rawLine) => {
    const line = rawLine.trim();
    if (!line) {
      closeList();
      return;
    }

    if (/^-\s+/.test(line)) {
      if (listType !== "ul") { closeList(); parts.push("<ul>"); listType = "ul"; }
      parts.push(`<li>${applyInline(line.replace(/^-\s+/, ""))}</li>`);
      return;
    }

    if (/^\d+\.\s+/.test(line)) {
      if (listType !== "ol") { closeList(); parts.push("<ol>"); listType = "ol"; }
      parts.push(`<li>${applyInline(line.replace(/^\d+\.\s+/, ""))}</li>`);
      return;
    }

    closeList();
    parts.push(`<p>${applyInline(line)}</p>`);
  });

  closeList();
  return parts.join("") || "<p></p>";
}

/** Apply inline markdown (bold, italic, code) to already-escaped text. */
function applyInline(text) {
  // Bold **text**
  text = text.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  // Italic *text*
  text = text.replace(/\*(.+?)\*/g, "<em>$1</em>");
  // Inline code `text`
  text = text.replace(/`([^`]+)`/g, "<code>$1</code>");
  return text;
}

// ═══════════════════════════════════════════════════════════════════
//  UTILITIES
// ═══════════════════════════════════════════════════════════════════

function el(id) {
  return document.getElementById(id);
}

function uid() {
  return `${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
}

function escHtml(v) {
  return String(v || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/**
 * Unified fetch wrapper. Always returns parsed JSON or null on network error.
 * @param {string} url
 * @param {RequestInit} [options]
 * @returns {Promise<any|null>}
 */
async function apiFetch(url, options = {}) {
  try {
    const res = await fetch(url, options);
    const text = await res.text();
    if (!text) return { ok: res.ok, status: res.status };
    try {
      return JSON.parse(text);
    } catch {
      return { ok: res.ok, status: res.status, message: text };
    }
  } catch {
    return null;
  }
}

function formatTime(iso) {
  if (!iso) return "";
  try {
    return new Date(iso).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "";
  }
}

function autoResize() {
  this.style.height = "auto";
  this.style.height = `${Math.min(this.scrollHeight, 180)}px`;
}
