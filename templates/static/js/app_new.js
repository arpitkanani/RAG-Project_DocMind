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
const API_KEY_STORAGE = "docmind.apiKey";

// ═══════════════════════════════════════════════════════════════════
//  BOOT
// ═══════════════════════════════════════════════════════════════════

document.addEventListener("DOMContentLoaded", async () => {
  bindUI();
  await ensureApiKey();
  await bootApp();
});

/**
 * Shows the styled API key modal until a non-empty key is saved.
 * Blocks bootApp() the same way the old window.prompt() loop did,
 * just without the native browser chrome.
 */
function ensureApiKey() {
  const existing = localStorage.getItem(API_KEY_STORAGE);
  if (existing) return Promise.resolve(existing);

  return new Promise((resolve) => {
    const backdrop = el("apiKeyModalBackdrop");
    const input = el("apiKeyInput");
    const errorEl = el("apiKeyError");
    const saveBtn = el("apiKeySave");

    errorEl.style.display = "none";
    input.value = "";
    backdrop.classList.add("open");
    setTimeout(() => input.focus(), 50);

    function trySave() {
      const key = input.value.trim();
      if (!key) {
        errorEl.textContent = "Please enter your API key.";
        errorEl.style.display = "block";
        return;
      }
      localStorage.setItem(API_KEY_STORAGE, key);
      backdrop.classList.remove("open");
      saveBtn.removeEventListener("click", trySave);
      input.removeEventListener("keydown", onKeydown);
      resolve(key);
    }

    function onKeydown(e) {
      if (e.key === "Enter") trySave();
    }

    saveBtn.addEventListener("click", trySave);
    input.addEventListener("keydown", onKeydown);
  });
}

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
  function isNarrowAttachViewport() {
    return window.matchMedia("(max-width: 640px)").matches;
  }

  el("attachBtn").addEventListener("click", () => {
    if (isNarrowAttachViewport()) {
      el("mobileAttachMenu").classList.toggle("open");
    } else {
      openModal();
    }
  });

  el("mobileMenuUpload").addEventListener("click", () => {
    el("mobileAttachMenu").classList.remove("open");
    openModal();
  });
  el("mobileMenuSources").addEventListener("click", () => {
    el("mobileAttachMenu").classList.remove("open");
    toggleSourcesPopover();
  });
  document.addEventListener("click", (e) => {
    const menu = el("mobileAttachMenu");
    const attachBtn = el("attachBtn");
    if (
      menu.classList.contains("open") &&
      !menu.contains(e.target) &&
      e.target !== attachBtn &&
      !attachBtn.contains(e.target)
    ) {
      menu.classList.remove("open");
    }
  });

  el("sourcesToggleBtn").addEventListener("click", toggleSourcesPopover);
  document.addEventListener("click", (e) => {
    const popover = el("sourcesPopover");
    const toggleBtn = el("sourcesToggleBtn");
    if (
      popover.classList.contains("open") &&
      !popover.contains(e.target) &&
      e.target !== toggleBtn &&
      !toggleBtn.contains(e.target)
    ) {
      popover.classList.remove("open");
    }
  });
  el("modalClose").addEventListener("click", closeModal);
  el("cancelUpload").addEventListener("click", closeModal);
  el("submitUpload").addEventListener("click", handleSubmitUpload);
  el("modalBackdrop").addEventListener("click", (e) => {
    if (e.target === el("modalBackdrop")) closeModal();
  });

  // Mobile sidebar — off-canvas overlay
  const sidebarEl = el("sidebar");
  const overlayEl = el("sidebarOverlay");
  const sidebarToggle = el("sidebarToggle");

  function isMobileViewport() {
    return window.matchMedia("(max-width: 900px)").matches;
  }

  const sidebarToggleIcon = el("sidebarToggleIcon");
  const ICON_OPEN = // sidebar is open — show "collapse" (chevrons pointing left)
    '<polyline points="11 5 4 12 11 19"/><polyline points="19 5 12 12 19 19"/>';
  const ICON_CLOSED = // sidebar is closed — show "expand" (chevrons pointing right)
    '<polyline points="13 5 20 12 13 19"/><polyline points="5 5 12 12 5 19"/>';

  function openSidebar() {
    sidebarEl.classList.remove("collapsed");
    overlayEl.classList.add("open");
    sidebarToggleIcon.innerHTML = ICON_OPEN;
  }

  function closeSidebar() {
    sidebarEl.classList.add("collapsed");
    overlayEl.classList.remove("open");
    sidebarToggleIcon.innerHTML = ICON_CLOSED;
  }

  sidebarToggle.addEventListener("click", () => {
    if (sidebarEl.classList.contains("collapsed")) openSidebar();
    else closeSidebar();
  });

  const sidebarCloseBtn = el("sidebarCloseBtn");
  if (sidebarCloseBtn) {
    sidebarCloseBtn.addEventListener("click", closeSidebar);
  }

  overlayEl.addEventListener("click", closeSidebar);

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && isMobileViewport()) closeSidebar();
  });

  // Sidebar starts collapsed on phones/tablets, open on desktop.
  // On resize (e.g. rotating a tablet), re-apply the correct default
  // rather than leaving it stuck in whatever state it was in before.
  function applyResponsiveSidebarDefault() {
    if (isMobileViewport()) closeSidebar();
    else {
      sidebarEl.classList.remove("collapsed");
      overlayEl.classList.remove("open");
    }
  }
  applyResponsiveSidebarDefault();
  window.addEventListener("resize", applyResponsiveSidebarDefault);

  // Selecting a chat on mobile should close the sidebar so the chat is visible
  el("sidebarChats").addEventListener("click", () => {
    if (isMobileViewport()) closeSidebar();
  });

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
  renderSourcesPanel();
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
    active: true, // included in query scope by default; user can toggle off
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
  renderSourcesPanel();
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

/**
 * Styled replacement for window.confirm(). Resolves true/false.
 */
function showConfirmModal(title, message, confirmLabel = "Delete") {
  return new Promise((resolve) => {
    const backdrop = el("confirmModalBackdrop");
    el("confirmModalTitle").textContent = title;
    el("confirmModalMessage").textContent = message;
    const confirmBtn = el("confirmModalConfirm");
    const cancelBtn = el("confirmModalCancel");
    const closeBtn = el("confirmModalClose");
    confirmBtn.textContent = confirmLabel;

    backdrop.classList.add("open");

    function cleanup(result) {
      backdrop.classList.remove("open");
      confirmBtn.removeEventListener("click", onConfirm);
      cancelBtn.removeEventListener("click", onCancel);
      closeBtn.removeEventListener("click", onCancel);
      backdrop.removeEventListener("click", onBackdropClick);
      resolve(result);
    }

    function onConfirm() {
      cleanup(true);
    }
    function onCancel() {
      cleanup(false);
    }
    function onBackdropClick(e) {
      if (e.target === backdrop) cleanup(false);
    }

    confirmBtn.addEventListener("click", onConfirm);
    cancelBtn.addEventListener("click", onCancel);
    closeBtn.addEventListener("click", onCancel);
    backdrop.addEventListener("click", onBackdropClick);
  });
}

async function deleteSession(sessionId) {
  const ok = await showConfirmModal(
    "Delete this chat?",
    "This will permanently delete this chat and its related document data."
  );
  if (!ok) return;

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
  const ok = await showConfirmModal(
    "Clear all chats?",
    "This will permanently delete every chat and all related document data.",
    "Clear All"
  );
  if (!ok) return;
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

/**
 * Polls a background job's status until it's ready or failed.
 * Updates the source's processingMessage live so the chip can show
 * retry/progress text (e.g. "rate limited, retrying in 20s").
 */
function pollJobStatus(jobId, localId, { onSuccess, defaultErrorMessage }) {
  const POLL_INTERVAL_MS = 3000;

  const poll = async () => {
    const job = await apiFetch(`${API.upload}/status/${encodeURIComponent(jobId)}`);

    if (!job || job.status === undefined) {
      // Network hiccup polling — try again rather than giving up on one miss
      setTimeout(poll, POLL_INTERVAL_MS);
      return;
    }

    if (job.status === "processing") {
      const src = workspaceState.sources.find((s) => s.localId === localId);
      if (src) {
        src.processingMessage = job.message || "Processing...";
        renderComposerChips();
      }
      setTimeout(poll, POLL_INTERVAL_MS);
      return;
    }

    if (job.status === "ready") {
      onSuccess(job.result || {});
      return;
    }

    // failed
    removeLocalSource(localId);
    setUploadBusy(hasProcessing());
    renderComposerChips();
    renderSourcesPanel();
    showToast(job.error || defaultErrorMessage, "error");
  };

  poll();
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
    active: true,
    processingMessage: "Uploading...",
  });
  setUploadBusy(true);
  renderComposerChips();
  renderSourcesPanel();

  try {
    const form = new FormData();
    form.append("file", file);
    form.append("session_id", workspaceState.sessionId || "default");

    const data = await apiFetch(API.upload, { method: "POST", body: form });

    if (!data?.job_id) {
      removeLocalSource(localId);
      setUploadBusy(hasProcessing());
      renderComposerChips();
      renderSourcesPanel();
      showToast(
        data?.message || data?.error || `Failed to upload ${file.name}.`,
        "error"
      );
      return;
    }

    pollJobStatus(data.job_id, localId, {
      defaultErrorMessage: `Failed to process ${file.name}.`,
      onSuccess: async (result) => {
        const src = workspaceState.sources.find((s) => s.localId === localId);
        if (src) {
          src.collection = result.collection_name;
          src.status = "ready";
          // isNew stays true — will attach to the NEXT message the user sends
        }
        setUploadBusy(hasProcessing());
        renderComposerChips();
        renderSourcesPanel();
        await loadSessions();
        showToast(`${file.name} indexed successfully.`, "success");
      },
    });
  } catch {
    removeLocalSource(localId);
    setUploadBusy(hasProcessing());
    renderComposerChips();
    renderSourcesPanel();
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
    active: true,
    processingMessage: "Fetching transcript...",
  });
  setUploadBusy(true);
  renderComposerChips();
  renderSourcesPanel();

  try {
    const data = await apiFetch(API.youtube, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, session_id: workspaceState.sessionId }),
    });

    if (!data?.job_id) {
      removeLocalSource(localId);
      setUploadBusy(hasProcessing());
      renderComposerChips();
      renderSourcesPanel();
      showToast(data?.message || "Could not process YouTube URL.", "error");
      return;
    }

    const src = workspaceState.sources.find((s) => s.localId === localId);
    if (src && data.video_id) {
      src.name = `YouTube ${data.video_id}`;
    }

    pollJobStatus(data.job_id, localId, {
      defaultErrorMessage: "Could not process YouTube URL.",
      onSuccess: async (result) => {
        const finalSrc = workspaceState.sources.find((s) => s.localId === localId);
        if (finalSrc) {
          finalSrc.name = result.video_id
            ? `YouTube ${result.video_id}`
            : "YouTube transcript";
          finalSrc.collection = result.collection_name;
          finalSrc.status = "ready";
        }
        setUploadBusy(hasProcessing());
        renderComposerChips();
        renderSourcesPanel();
        await loadSessions();
        showToast("YouTube transcript indexed.", "success");
      },
    });
  } catch {
    removeLocalSource(localId);
    setUploadBusy(hasProcessing());
    renderComposerChips();
    renderSourcesPanel();
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
    renderSourcesPanel();

    // Also strip the chip out of any message bubble it appears in
    document
      .querySelectorAll(`.message-attachments .chip-remove[data-collection="${CSS.escape(collection)}"]`)
      .forEach((btn) => btn.closest(".chip")?.remove());

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

  const activeSources = readySources.filter((s) => s.active !== false);
  if (!activeSources.length) {
    showToast(
      "All sources are excluded. Click a source chip to include it.",
      "error"
    );
    return;
  }
  // Only send an explicit scope when the user has excluded something —
  // otherwise omit it so the backend's default "search everything" behavior
  // is unchanged for anyone who never touches the toggle.
  const scopedCollectionNames =
    activeSources.length < readySources.length
      ? activeSources.map((s) => s.collection)
      : null;

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
  renderSourcesPanel();
  updateSendButton();

  conversationState.isLoading = true;
  const typingId = showTyping();

  let streamedText = "";
  let messageRow = null;
  let bubbleEl = null;

  try {
    await streamQueryAnswer(
      API.query,
      {
        query,
        session_id: workspaceState.sessionId,
        message_attachments: messageAttachments.length ? messageAttachments : null,
        collection_names: scopedCollectionNames,
      },
      {
        onToken: (token) => {
          removeTyping(typingId);
          streamedText += token;
          if (!messageRow) {
            const container = el("messages");
            messageRow = document.createElement("div");
            messageRow.className = "msg-row ai";
            messageRow.innerHTML = `
              <div class="msg-avatar">AI</div>
              <div class="msg-body">
                <div class="msg-bubble">${formatContent(streamedText)}</div>
              </div>`;
            container.appendChild(messageRow);
            bubbleEl = messageRow.querySelector(".msg-bubble");
          } else if (bubbleEl) {
            bubbleEl.innerHTML = formatContent(streamedText);
          }
          const container = el("messages");
          container.scrollTop = container.scrollHeight;
        },
        onCitations: (citations) => {
          if (citations && bubbleEl && streamedText !== FALLBACK_ANSWER_TEXT) {
            const fullContent = `${streamedText}\n\n${citations}`;
            bubbleEl.innerHTML = formatContent(fullContent);
          }
        },
        onDone: async (doneData) => {
          removeTyping(typingId);
          if (doneData.session_id) {
            workspaceState.sessionId = doneData.session_id;
            localStorage.setItem(SESSION_KEY, workspaceState.sessionId);
          }
          if (bubbleEl && doneData.final_answer) {
            bubbleEl.innerHTML = formatContent(doneData.final_answer);
          }
          try {
            await loadSessions();
          } catch {}
        },
        onError: async (errData) => {
          removeTyping(typingId);
          if (errData?.error_code === "collection_not_found") {
            const missing = Array.isArray(errData.missing_collections)
              ? errData.missing_collections
              : [];
            workspaceState.sources = workspaceState.sources.filter(
              (s) => !missing.includes(s.collection)
            );
            renderComposerChips();
            renderSourcesPanel();
            await loadSessions();
            showToast(
              "One or more sources were removed. Please upload again if needed.",
              "error"
            );
            return;
          }
          if (errData?.error_code === "knowledge_base_empty") {
            showToast(
              errData.message || "Please upload a document or add a YouTube source first.",
              "error"
            );
            return;
          }
          showToast(errData?.message || "Could not complete that request.", "error");
        },
      }
    );
  } catch {
    removeTyping(typingId);
    showToast("Server is down. Please try again.", "error");
  } finally {
    removeTyping(typingId);
    conversationState.isLoading = false;
    updateSendButton();
  }
}

const FALLBACK_ANSWER_TEXT = "I couldn't find relevant information about that in the uploaded document.";

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
  const pending = workspaceState.sources.filter((s) => s.isNew);

  if (!pending.length) {
    row.hidden = true;
    row.innerHTML = "";
    return;
  }

  row.hidden = false;
  row.innerHTML = pending
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
        <div class="chip ${processing ? "processing" : ""}" data-local-id="${escHtml(src.localId)}"
             title="${processing ? escHtml(src.processingMessage || "Processing...") : ""}">
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
//  RENDER — SOURCES PANEL (persistent toggle, separate from the
//  pending-upload chips above). Opens as a small popover from a
//  button next to the attach button; does NOT affect the composer's
//  height or the send flow.
// ═══════════════════════════════════════════════════════════════════

function renderSourcesPanel() {
  const readySources = workspaceState.sources.filter(
    (s) => s.status === "ready" && s.collection
  );
  const countBtn = el("sourcesToggleBtn");
  const countLabel = el("sourcesCount");
  const list = el("sourcesPopoverList");

  if (!readySources.length) {
    countBtn.hidden = true;
    return;
  }
  countBtn.hidden = false;

  const activeCount = readySources.filter((s) => s.active !== false).length;
  countLabel.textContent = `${activeCount}/${readySources.length}`;

  list.innerHTML = readySources
    .map((src) => {
      const inactive = src.active === false;
      const typeLabel = src.type === "yt" ? "YT" : "DOC";
      return `
        <div class="source-toggle-item ${inactive ? "inactive" : ""}" data-local-id="${escHtml(src.localId)}">
          <span class="chip-type">${typeLabel}</span>
          <span class="source-toggle-name" title="${escHtml(src.name)}">${escHtml(src.name)}</span>
          <span class="source-toggle-state">${inactive ? "Excluded" : "Included"}</span>
        </div>`;
    })
    .join("");

  list.querySelectorAll(".source-toggle-item").forEach((itemEl) => {
    itemEl.addEventListener("click", () => {
      const src = workspaceState.sources.find(
        (s) => s.localId === itemEl.dataset.localId
      );
      if (!src) return;
      src.active = src.active === false ? true : false;
      renderSourcesPanel();
    });
  });
}

function toggleSourcesPopover() {
  el("sourcesPopover").classList.toggle("open");
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

  // Wire up remove buttons on message-level chips (if any)
  row.querySelectorAll(".message-attachments .chip-remove").forEach((btn) => {
    btn.addEventListener("click", () => removeSource(btn.dataset.collection));
  });

  container.scrollTop = container.scrollHeight;

}

function truncateSourceName(name, maxLen = 12) {
  const clean = (name || "").trim();
  if (clean.length <= maxLen) return clean;
  return clean.slice(0, maxLen).trim() + "…";
}

function truncateMessageChipName(name, type) {
  const clean = (name || "").trim();

  if (type === "yt") {
    const maxLen = 14;
    return clean.length <= maxLen ? clean : clean.slice(0, maxLen).trim() + "…";
  }

  // doc: first word in full, second word trimmed
  const words = clean.split(/\s+/).filter(Boolean);
  if (words.length <= 1) {
    const maxLen = 14;
    return clean.length <= maxLen ? clean : clean.slice(0, maxLen).trim() + "…";
  }
  const first = words[0];
  const secondTrimmed = words[1].slice(0, 3);
  return `${first} ${secondTrimmed}…`;
}

function isMobileChipViewport() {
  return window.matchMedia("(max-width: 640px)").matches;
}

// function truncateChipNameDesktop(name, type) {
//   const clean = (name || "").trim();
//   if (type === "yt") {
//     return clean.length <= 10 ? clean : clean.slice(0, 10) + "…";
//   }
//   const words = clean.split(/\s+/).filter(Boolean);
//   if (words.length <= 1) {
//     return clean.length <= 14 ? clean : clean.slice(0, 14) + "…";
//   }
//   const first = words[0];
//   const secondTrimmed = words[1].slice(0, 3);
//   return `${first} ${secondTrimmed}…`;
// }

function buildReadOnlyChips(chips) {
  const mobile = isMobileChipViewport();
  return chips
    .map((c) => {
      const displayName = mobile
        ? truncateSourceName(c.name)
        : truncateMessageChipName(c.name, c.type);
      return `
      <div class="chip message-chip chip-hover-controls">
        <span class="chip-type">${c.type === "yt" ? "YT" : "DOC"}</span>
        <span class="chip-name" title="${escHtml(c.name)}">${escHtml(displayName)}</span>
        <button class="chip-remove" data-collection="${escHtml(c.collection)}" title="Remove source">
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3.5">
            <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
          </svg>
        </button>
      </div>`;
    })
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
    const apiKey = localStorage.getItem(API_KEY_STORAGE);
    const res = await fetch(url, {
      ...options,
      headers: {
        ...(options.headers || {}),
        "X-API-Key": apiKey || "",
      },
    });

    const text = await res.text();
    let parsed = null;
    if (text) {
      try {
        parsed = JSON.parse(text);
      } catch {
        parsed = { message: text };
      }
    }

    if (res.status === 401) {
      // Key missing/invalid/revoked — clear it and force re-entry
      localStorage.removeItem(API_KEY_STORAGE);
      showToast("Your API key is invalid or missing. Please re-enter it.", "error");
      ensureApiKey();
      return null;
    }

    if (res.status === 422) {
      // Framework-level validation error, not app data — surface it, don't pretend it's a result
      console.error("Request validation failed:", parsed);
      return null;
    }

    if (!res.ok) {
      console.error(`Request failed (${res.status}):`, parsed);
      return { ok: false, status: res.status, ...(parsed || {}) };
    }

    return parsed ?? { ok: res.ok, status: res.status };
  } catch (err) {
    console.error("Network error:", err);
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

/**
 * Streams SSE events from the /query endpoint token-by-token.
 * @param {string} url
 * @param {object} payload
 * @param {object} callbacks
 * @param {(token: string) => void} callbacks.onToken
 * @param {(citations: string) => void} callbacks.onCitations
 * @param {(data: object) => void} callbacks.onDone
 * @param {(err: object) => void} callbacks.onError
 */
async function streamQueryAnswer(url, payload, { onToken, onCitations, onDone, onError }) {
  try {
    const apiKey = localStorage.getItem(API_KEY_STORAGE);
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": apiKey || "",
      },
      body: JSON.stringify(payload),
    });

    if (res.status === 401) {
      localStorage.removeItem(API_KEY_STORAGE);
      showToast("Your API key is invalid or missing. Please re-enter it.", "error");
      ensureApiKey();
      onError?.({ error_code: "unauthorized", message: "API key invalid" });
      return;
    }

    if (!res.ok) {
      let errBody = null;
      try {
        errBody = await res.json();
      } catch {}
      onError?.(errBody || { error_code: "server_error", message: `Request failed with status ${res.status}` });
      return;
    }

    if (!res.body) {
      onError?.({ error_code: "no_body", message: "No response body received." });
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || ""; // Keep incomplete trailing fragment in buffer

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith("data:")) continue;

        const jsonStr = trimmed.slice(5).trim();
        if (!jsonStr) continue;

        try {
          const event = JSON.parse(jsonStr);
          if (event.type === "token") {
            onToken?.(event.content);
          } else if (event.type === "citations") {
            onCitations?.(event.citations);
          } else if (event.type === "done") {
            onDone?.(event);
          } else if (event.type === "error") {
            onError?.(event);
          }
        } catch (parseErr) {
          console.error("SSE parse error:", parseErr, jsonStr);
        }
      }
    }
  } catch (err) {
    console.error("Stream network error:", err);
    onError?.({ error_code: "network_error", message: "Server is down. Please try again." });
  }
}