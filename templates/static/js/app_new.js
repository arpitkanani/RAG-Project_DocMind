"use strict";

const state = {
  sessionId: null,
  chips: [],
  pendingChips: [],
  chats: [],
  activeTab: "file",
  uploadInProgress: false,
};

const API = {
  upload: "/upload",
  youtube: "/youtube",
  query: "/query",
  sessions: "/sessions",
  legacyNewSession: "/sessions/new",
  memory: "/memory",
};

document.addEventListener("DOMContentLoaded", async () => {
  bindUI();
  await createNewSession();
  await loadSessions();
  renderPendingChips();
});

function bindUI() {
  document.getElementById("newChatBtn").addEventListener("click", handleNewChat);
  document.getElementById("searchInput").addEventListener("input", filterChats);
  document.getElementById("clearAllBtn").addEventListener("click", handleClearAll);
  document.getElementById("attachBtn").addEventListener("click", openModal);
  document.getElementById("modalClose").addEventListener("click", closeModal);
  document.getElementById("cancelUpload").addEventListener("click", closeModal);
  document.getElementById("submitUpload").addEventListener("click", handleSubmitUpload);
  document.getElementById("sendBtn").addEventListener("click", sendMessage);
  document.getElementById("queryInput").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  });
  document.getElementById("queryInput").addEventListener("input", autoResize);
  document.getElementById("modalBackdrop").addEventListener("click", (event) => {
    if (event.target === document.getElementById("modalBackdrop")) {
      closeModal();
    }
  });

  document.querySelectorAll(".modal-tab").forEach((button) => {
    button.addEventListener("click", () => switchTab(button.dataset.tab));
  });

  const dropZone = document.getElementById("dropZone");
  dropZone.addEventListener("click", () => document.getElementById("fileInput").click());
  dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropZone.classList.add("drag-over");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropZone.classList.remove("drag-over");
    handleFiles(Array.from(event.dataTransfer.files || []));
    closeModal();
  });
  document.getElementById("fileInput").addEventListener("change", (event) => {
    handleFiles(Array.from(event.target.files || []));
    closeModal();
  });

  window.addEventListener("pagehide", handleSiteExit);
}

async function createNewSession() {
  try {
    const response = await fetch(API.sessions, { method: "POST" });
    const data = await response.json();
    state.sessionId = data.session_id;
  } catch {
    try {
      const response = await fetch(API.legacyNewSession, { method: "POST" });
      const data = await response.json();
      state.sessionId = data.session_id;
    } catch {
      state.sessionId = "default";
    }
  }

  state.chips = [];
  state.pendingChips = [];
  state.uploadInProgress = false;
  updateChatTitle("New Chat");
  renderPendingChips();
}

async function handleNewChat() {
  renderWelcomeOnly();
  await createNewSession();
  await loadSessions();
}

async function loadSessions() {
  try {
    const response = await fetch(API.sessions);
    const data = await response.json();
    state.chats = data.sessions || [];
    renderSidebar(state.chats);
  } catch {
    renderSidebar([]);
  }
}

function renderSidebar(sessions) {
  const container = document.getElementById("sidebarChats");
  if (!sessions.length) {
    container.innerHTML = `<div class="empty-sidebar">No chats yet. Start a new conversation.</div>`;
    return;
  }

  const now = Date.now();
  const groups = { Today: [], Yesterday: [], "Previous 7 Days": [] };
  sessions.forEach((session) => {
    const activeAt = new Date(session.last_active || 0).getTime();
    const days = (now - activeAt) / 86400000;
    if (days < 1) groups.Today.push(session);
    else if (days < 2) groups.Yesterday.push(session);
    else groups["Previous 7 Days"].push(session);
  });

  let html = "";
  Object.entries(groups).forEach(([label, items]) => {
    if (!items.length) return;
    html += `<div class="chat-group-label">${label}</div>`;
    items.forEach((session) => {
      const active = session.session_id === state.sessionId ? "active" : "";
      const title = escHtml(session.title || "New Chat");
      const time = escHtml(session.last_active_label || formatTime(session.last_active));
      html += `
        <div class="chat-item ${active}" data-id="${escHtml(session.session_id)}">
          <span class="chat-item-text">${title}</span>
          <span class="chat-item-time">${time}</span>
          <button class="chat-item-del" data-del="${escHtml(session.session_id)}" title="Delete chat">x</button>
        </div>`;
    });
  });

  container.innerHTML = html;
  container.querySelectorAll(".chat-item").forEach((element) => {
    element.addEventListener("click", (event) => {
      if (event.target.closest("[data-del]")) return;
      switchSession(element.dataset.id);
    });
  });
  container.querySelectorAll("[data-del]").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteSession(button.dataset.del);
    });
  });
}

function filterChats(event) {
  const query = event.target.value.toLowerCase();
  renderSidebar(
    state.chats.filter((chat) =>
      (chat.title || chat.session_id || "").toLowerCase().includes(query)
    )
  );
}

async function switchSession(sessionId) {
  try {
    const response = await fetch(`${API.sessions}/${encodeURIComponent(sessionId)}`);
    const data = await response.json();
    state.sessionId = sessionId;
    state.chips = Array.isArray(data.attachments) ? data.attachments : [];
    state.pendingChips = [];
    state.uploadInProgress = false;
    renderPendingChips();

    clearMessages();
    hideWelcome();
    const messages = data.messages || [];
    let renderedFirstHuman = false;

    if (!messages.length) {
      renderWelcomeOnly();
    } else {
      messages.forEach((message) => {
        const isHuman = message.role === "human";
        const attachmentChips = isHuman && !renderedFirstHuman ? state.chips : [];
        appendMessage(
          message.content,
          isHuman ? "user" : "ai",
          attachmentChips
        );
        if (isHuman) renderedFirstHuman = true;
      });
    }

    updateChatTitle(data.title || "New Chat");
    await loadSessions();
  } catch {
    showToast("Could not load that chat.", "error");
  }
}

async function deleteSession(sessionId) {
  if (!confirm("Delete this chat and its related document data?")) return;
  try {
    await fetch(`${API.sessions}/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
    if (state.sessionId === sessionId) {
      await handleNewChat();
    } else {
      await loadSessions();
    }
    showToast("Chat and related document data deleted.", "success");
  } catch {
    showToast("Could not delete chat.", "error");
  }
}

async function handleClearAll() {
  if (!confirm("Clear all chats and all related document data?")) return;
  try {
    await fetch(API.memory, { method: "DELETE" });
    await handleNewChat();
    showToast("All chat history and document data cleared.", "success");
  } catch {
    showToast("Could not clear history.", "error");
  }
}

function openModal() {
  document.getElementById("modalBackdrop").classList.add("open");
  switchTab("file");
}

function closeModal() {
  document.getElementById("modalBackdrop").classList.remove("open");
  document.getElementById("ytInput").value = "";
  document.getElementById("fileInput").value = "";
}

function switchTab(tab) {
  state.activeTab = tab;
  document.querySelectorAll(".modal-tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tab);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `tab-${tab}`);
  });
}

async function handleSubmitUpload() {
  if (state.activeTab === "yt") {
    await handleYouTube();
  } else {
    document.getElementById("fileInput").click();
  }
}

async function handleFiles(files) {
  for (const file of files) {
    await uploadFile(file);
  }
}

async function uploadFile(file) {
  const localChipId = `pending-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
  state.uploadInProgress = true;
  pushPendingChip({
    id: localChipId,
    name: file.name,
    collection: "",
    type: "doc",
    status: "processing",
  });

  try {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("session_id", state.sessionId || "default");
    const response = await fetch(API.upload, { method: "POST", body: formData });
    const data = await response.json();

    if (!response.ok || !data.success) {
      removePendingChip(localChipId);
      state.uploadInProgress = hasProcessingChip();
      renderPendingChips();
      showToast(data.message || data.error || data.detail || `Failed to upload ${file.name}.`, "error");
      return;
    }

    addChip({
      id: localChipId,
      name: file.name,
      collection: data.collection_name,
      type: "doc",
    });
    markPendingChipReady(localChipId, data.collection_name);
    await loadSessions();
    showToast(`${file.name} indexed successfully.`, "success");
  } catch {
    removePendingChip(localChipId);
    state.uploadInProgress = hasProcessingChip();
    renderPendingChips();
    showToast(`Upload failed for ${file.name}.`, "error");
  }
}

async function handleYouTube() {
  const url = document.getElementById("ytInput").value.trim();
  if (!url) {
    showToast("Please enter a YouTube URL.", "error");
    return;
  }

  closeModal();
  const localChipId = `pending-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
  state.uploadInProgress = true;
  pushPendingChip({
    id: localChipId,
    name: "YouTube transcript",
    collection: "",
    type: "yt",
    status: "processing",
  });

  try {
    const response = await fetch(API.youtube, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, session_id: state.sessionId }),
    });
    const data = await response.json();

    if (!response.ok || !data.success) {
      removePendingChip(localChipId);
      state.uploadInProgress = hasProcessingChip();
      renderPendingChips();
      showToast(data.message || data.detail || "Could not process YouTube URL.", "error");
      return;
    }

    addChip({
      id: localChipId,
      name: "YouTube transcript",
      collection: data.collection_name,
      type: "yt",
    });
    markPendingChipReady(localChipId, data.collection_name);
    await loadSessions();
    showToast("YouTube transcript indexed.", "success");
  } catch {
    removePendingChip(localChipId);
    state.uploadInProgress = hasProcessingChip();
    renderPendingChips();
    showToast("Could not process YouTube URL.", "error");
  }
}

function addChip(chip) {
  if (!chip.collection) return;
  state.chips = state.chips.filter((item) => item.collection !== chip.collection);
  state.chips.push(chip);
  syncAttachmentRows();
}

function pushPendingChip(chip) {
  state.pendingChips.push(chip);
  renderPendingChips();
}

function removePendingChip(chipId) {
  state.pendingChips = state.pendingChips.filter((chip) => chip.id !== chipId);
}

function markPendingChipReady(chipId, collectionName) {
  state.pendingChips = state.pendingChips.map((chip) =>
    chip.id === chipId
      ? { ...chip, collection: collectionName, status: "ready" }
      : chip
  );
  state.uploadInProgress = hasProcessingChip();
  renderPendingChips();
}

function hasProcessingChip() {
  return state.pendingChips.some((chip) => chip.status === "processing");
}

async function removeChip(index) {
  const readyChips = state.pendingChips.filter((chip) => chip.status === "ready");
  const chip = readyChips[index];
  if (!chip || !chip.collection) return;

  try {
    const response = await fetch(
      `${API.sessions}/${encodeURIComponent(state.sessionId)}/attachments/${encodeURIComponent(chip.collection)}`,
      { method: "DELETE" }
    );
    const data = await response.json();
    if (!response.ok || !data.success) {
      showToast(data.message || "Could not remove the document.", "error");
      return;
    }

    state.chips = state.chips.filter((item) => item.collection !== chip.collection);
    state.pendingChips = state.pendingChips.filter((item) => item.collection !== chip.collection);
    renderPendingChips();
    syncAttachmentRows();
    await loadSessions();
    showToast("Document removed from this chat. Upload again to continue.", "success");
  } catch {
    showToast("Could not remove the document.", "error");
  }
}

async function sendMessage() {
  const input = document.getElementById("queryInput");
  const query = input.value.trim();
  if (!query) return;

  if (state.uploadInProgress) {
    showToast("Please wait until the new document finishes processing.", "error");
    return;
  }

  if (!state.chips.length) {
    showToast("Please upload a document first.", "error");
    return;
  }

  const readyPendingChips = state.pendingChips.filter((chip) => chip.status === "ready");
  hideWelcome();
  appendMessage(query, "user", readyPendingChips);
  input.value = "";
  input.style.height = "auto";
  state.pendingChips = [];
  renderPendingChips();

  const typingId = showTyping();

  try {
    const collectionNames = state.chips.map((chip) => chip.collection).filter(Boolean);
    const response = await fetch(API.query, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        collection_names: collectionNames.length ? collectionNames : null,
        session_id: state.sessionId,
      }),
    });
    const data = await response.json();
    removeTyping(typingId);

    if (!response.ok) {
      if (data.error_code === "collection_not_found") {
        if (Array.isArray(data.missing_collections) && data.missing_collections.length) {
          state.chips = state.chips.filter(
            (chip) => !data.missing_collections.includes(chip.collection)
          );
          syncAttachmentRows();
        }
        showToast("This document was removed. Please upload the document again.", "error");
        return;
      }
      if (data.error_code === "knowledge_base_empty") {
        showToast(data.message || "Please upload a document first.", "error");
        return;
      }
      showToast(data.message || "Server is down. Please try again.", "error");
      return;
    }

    state.sessionId = data.session_id || state.sessionId;
    appendMessage(data.answer || "No response received.", "ai");
    await loadSessions();

    if (document.getElementById("chatTitle").textContent.trim() === "New Chat") {
      updateChatTitle(query.length > 48 ? `${query.slice(0, 48)}...` : query);
    }
  } catch {
    removeTyping(typingId);
    showToast("Server is down. Please try again.", "error");
  } finally {
    renderPendingChips();
  }
}

function appendMessage(content, role, attachmentChips = []) {
  const container = document.getElementById("messages");
  const row = document.createElement("div");
  row.className = `msg-row ${role}`;
  const avatar = role === "user" ? "You" : "AI";
  if (role === "user" && attachmentChips.length) {
    row.dataset.attachmentHost = "true";
  }
  row.innerHTML = `
    <div class="msg-avatar">${avatar}</div>
    <div class="msg-body">
      ${role === "user" && attachmentChips.length ? `<div class="message-attachments">${buildChipsMarkup(attachmentChips)}</div>` : ""}
      <div class="msg-bubble">${formatContent(content)}</div>
    </div>
  `;
  container.appendChild(row);
  bindChipRemoveButtons(row);
  container.scrollTop = container.scrollHeight;
}

function buildChipsMarkup(chips) {
  return chips
    .map((chip, index) => `
      <div class="chip ${chip.status === "processing" ? "processing" : ""}">
        <span class="chip-type">${chip.type === "yt" ? "YT" : "DOC"}</span>
        <span class="chip-name" title="${escHtml(chip.name)}">${escHtml(chip.name)}</span>
        ${chip.status === "processing"
          ? `<span class="chip-indicator processing" aria-label="Processing"></span>`
          : `<span class="chip-indicator ready" aria-label="Ready"></span><button class="chip-remove" data-index="${index}" title="Remove document">x</button>`}
      </div>
    `)
    .join("");
}

function bindChipRemoveButtons(scope = document) {
  scope.querySelectorAll(".chip-remove").forEach((button) => {
    button.onclick = () => removeChip(Number(button.dataset.index));
  });
}

function syncAttachmentRows() {
  document.querySelectorAll("[data-attachment-host='true'] .message-attachments").forEach((container) => {
    container.innerHTML = buildChipsMarkup(state.chips);
    container.style.display = state.chips.length ? "flex" : "none";
    bindChipRemoveButtons(container);
  });
}

function renderPendingChips() {
  const wrapper = document.querySelector(".prompt-content");
  let bar = document.getElementById("pendingChipsBar");

  if (!bar) {
    bar = document.createElement("div");
    bar.id = "pendingChipsBar";
    bar.className = "pending-chips-bar";
    wrapper.prepend(bar);
  }

  if (!state.pendingChips.length) {
    bar.innerHTML = "";
    bar.style.display = "none";
  } else {
    bar.innerHTML = buildChipsMarkup(state.pendingChips);
    bar.style.display = "flex";
    bindChipRemoveButtons(bar);
  }

  const sendBtn = document.getElementById("sendBtn");
  const canSend = !state.uploadInProgress && state.chips.length > 0;
  sendBtn.disabled = !canSend;
  sendBtn.classList.toggle("ready", canSend);
}

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
      if (listType !== "ul") {
        closeList();
        parts.push("<ul>");
        listType = "ul";
      }
      parts.push(`<li>${line.replace(/^-\s+/, "")}</li>`);
      return;
    }

    if (/^\d+\.\s+/.test(line)) {
      if (listType !== "ol") {
        closeList();
        parts.push("<ol>");
        listType = "ol";
      }
      parts.push(`<li>${line.replace(/^\d+\.\s+/, "")}</li>`);
      return;
    }

    closeList();
    parts.push(`<p>${line}</p>`);
  });

  closeList();
  return parts.join("") || "<p></p>";
}

function showTyping() {
  const container = document.getElementById("messages");
  const id = `typing-${Date.now()}`;
  const element = document.createElement("div");
  element.className = "typing-row";
  element.id = id;
  element.innerHTML = `
    <div class="msg-avatar msg-avatar-ai">AI</div>
    <div class="typing-bubble">
      <div class="typing-label">Searching your uploaded document...</div>
      <div class="typing-dots"><span></span><span></span><span></span></div>
    </div>
  `;
  container.appendChild(element);
  container.scrollTop = container.scrollHeight;
  return id;
}

function removeTyping(id) {
  const element = document.getElementById(id);
  if (element) element.remove();
}

function showWelcome() {
  const container = document.getElementById("messages");
  let welcome = document.getElementById("welcome");

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
      </div>
    `;
    container.prepend(welcome);
  }

  welcome.style.display = "flex";
}

function hideWelcome() {
  const welcome = document.getElementById("welcome");
  if (welcome) welcome.style.display = "none";
}

function clearMessages() {
  document.getElementById("messages").innerHTML = "";
}

function renderWelcomeOnly() {
  clearMessages();
  showWelcome();
}

function updateChatTitle(title) {
  document.getElementById("chatTitle").textContent = title || "New Chat";
}

function showToast(message, type = "success") {
  const stack = document.getElementById("toastStack");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  stack.appendChild(toast);
  setTimeout(() => toast.remove(), 4200);
}

function insertSuggestion(text) {
  const input = document.getElementById("queryInput");
  input.value = text;
  input.focus();
  autoResize.call(input);
}

function autoResize() {
  this.style.height = "auto";
  this.style.height = `${Math.min(this.scrollHeight, 180)}px`;
}

function formatTime(isoString) {
  if (!isoString) return "";
  try {
    return new Date(isoString).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "";
  }
}

function handleSiteExit() {
  fetch(API.memory, { method: "DELETE", keepalive: true }).catch(() => {});
}

function escHtml(value) {
  return String(value || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

window.insertSuggestion = insertSuggestion;
