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

const SESSION_KEY = "docuvortex.activeSessionId";
const API_KEY_STORAGE = "docuvortex.apiKey";
const FALLBACK_ANSWER_TEXT = "I couldn't find information about that in the uploaded document(s).";

// Server-provided last active session ID (recovered during auth check)
let _serverLastSessionId = null;

// ═══════════════════════════════════════════════════════════════════
//  BOOT
// ═══════════════════════════════════════════════════════════════════

document.addEventListener("DOMContentLoaded", async () => {
  bindUI();
  await ensureApiKey();
  await bootApp();
});

/**
 * Ensures user has an active key or prompts directly with popup on fresh load.
 * Remembers key across browser refresh, restarts, and tabs.
 */
async function ensureApiKey() {
  const existing = localStorage.getItem(API_KEY_STORAGE);
  if (existing && existing.trim()) {
    if (existing === "guest") return existing;
    try {
      const statusRes = await fetch("/api/auth/session-status", {
        headers: { "X-API-Key": existing.trim() },
        credentials: "include",
      });
      if (statusRes.ok) {
        const authData = await statusRes.json();
        if (authData && authData.authenticated) {
          _serverLastSessionId = authData.last_session_id || null;
          return existing.trim();
        }
      }
    } catch (err) {
      console.debug("API key check note:", err);
      return existing.trim(); // Network blip; preserve key
    }
    localStorage.removeItem(API_KEY_STORAGE);
  }

  // Fresh browser / new user — ask directly with popup modal
  return await openApiKeyModal({ forceRequired: true });
}

/**
 * Opens the styled API key modal so users can input/update their seed_user.py key.
 */
function openApiKeyModal({ forceRequired = false } = {}) {
  return new Promise((resolve) => {
    const backdrop = el("apiKeyModalBackdrop");
    const input = el("apiKeyInput");
    const errorEl = el("apiKeyError");
    const saveBtn = el("apiKeySave");
    const guestBtn = el("apiKeyGuest");

    if (!backdrop || !input || !saveBtn) {
      return resolve("");
    }

    if (errorEl) errorEl.style.display = "none";
    const currentKey = localStorage.getItem(API_KEY_STORAGE);
    input.value = currentKey && currentKey !== "guest" ? currentKey : "";
    backdrop.classList.add("open");
    setTimeout(() => input.focus(), 60);

    function cleanup() {
      backdrop.classList.remove("open");
      saveBtn.removeEventListener("click", trySave);
      input.removeEventListener("keydown", onKeydown);
      if (guestBtn) guestBtn.removeEventListener("click", onGuest);
      backdrop.removeEventListener("click", onBackdrop);
    }

    function onBackdrop(e) {
      if (e.target === backdrop && !forceRequired) {
        cleanup();
        resolve(localStorage.getItem(API_KEY_STORAGE) || "guest");
      }
    }

    async function trySave() {
      const key = input.value.trim();
      if (!key) {
        if (errorEl) {
          errorEl.textContent = "Please enter your API key.";
          errorEl.style.display = "block";
        }
        return;
      }

      saveBtn.disabled = true;
      saveBtn.textContent = "Verifying...";
      if (errorEl) errorEl.style.display = "none";

      try {
        const verifyRes = await fetch("/api/auth/verify-key", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ api_key: key }),
        });
        const verifyData = await verifyRes.json();

        if (!verifyData?.valid) {
          if (errorEl) {
            errorEl.textContent =
              verifyData?.message ||
              "Invalid API key. Check key generated with seed_user.py";
            errorEl.style.display = "block";
          }
          saveBtn.disabled = false;
          saveBtn.textContent = "Save Key";
          return;
        }

        // Cache last session ID returned from PostgreSQL
        if (verifyData.last_session_id) {
          _serverLastSessionId = verifyData.last_session_id;
        }

        localStorage.setItem(API_KEY_STORAGE, key);
        cleanup();
        showToast("API key verified. Welcome!", "success");
        await bootApp();
        resolve(key);
      } catch (err) {
        localStorage.setItem(API_KEY_STORAGE, key);
        cleanup();
        resolve(key);
      }
    }

    function onGuest() {
      localStorage.setItem(API_KEY_STORAGE, "guest");
      cleanup();
      showToast("Continuing in Guest mode.", "success");
      bootApp();
      resolve("guest");
    }

    function onKeydown(e) {
      if (e.key === "Enter") {
        e.preventDefault();
        trySave();
      }
      if (e.key === "Escape" && !forceRequired) {
        cleanup();
        resolve(localStorage.getItem(API_KEY_STORAGE) || "guest");
      }
    }

    saveBtn.addEventListener("click", trySave);
    input.addEventListener("keydown", onKeydown);
    if (guestBtn) guestBtn.addEventListener("click", onGuest);
    backdrop.addEventListener("click", onBackdrop);
  });
}

window.openApiKeyModal = openApiKeyModal;

async function bootApp() {
  const apiKey = localStorage.getItem(API_KEY_STORAGE);

  // Fetch latest active session from PostgreSQL if not cached
  if (!_serverLastSessionId && apiKey && apiKey !== "guest") {
    try {
      const statusRes = await fetch("/api/auth/session-status", {
        headers: { "X-API-Key": apiKey },
        credentials: "include",
      });
      if (statusRes.ok) {
        const authData = await statusRes.json();
        if (authData?.authenticated && authData.last_session_id) {
          _serverLastSessionId = authData.last_session_id;
        }
      }
    } catch {}
  }

  const sessions = await loadSessions();
  const storedId = localStorage.getItem(SESSION_KEY) || _serverLastSessionId;

  // 1. Try restoring the stored/server session first (persisted in PostgreSQL)
  if (storedId) {
    const ok = await restoreSession(storedId);
    if (ok) return;
    localStorage.removeItem(SESSION_KEY);
  }

  // 2. Fall back to the most-recent available session from the user's PostgreSQL session list
  if (sessions && sessions.length > 0) {
    const latest = sessions[0];
    const ok = await restoreSession(latest.session_id);
    if (ok) return;
  }

  // 3. No prior sessions found in PostgreSQL — start fresh
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

  const apiKeyBtn = el("apiKeyBtn");
  if (apiKeyBtn) {
    apiKeyBtn.addEventListener("click", openApiKeyModal);
  }

  // Chat input '+' button opens the upload modal with Document and YouTube options
  const attachBtn = el("attachBtn");
  if (attachBtn) {
    attachBtn.addEventListener("click", (e) => {
      e.preventDefault();
      if (conversationState.uploadInProgress) {
        return showToast("Please wait for current upload to complete.", "error");
      }
      openModal();
    });
  }

  const ytInput = el("ytInput");
  if (ytInput) {
    ytInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        handleSubmitUpload();
      }
    });
  }

  const chatFileInput = el("chatFileInput") || el("fileInput");
  if (chatFileInput) {
    chatFileInput.addEventListener("change", (e) => {
      if (conversationState.uploadInProgress) {
        return showToast("Please wait for current upload to complete.", "error");
      }
      const files = Array.from(e.target.files || []);
      if (files.length) {
        handleFiles(files);
      }
      e.target.value = "";
    });
  }

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

  // Mobile & tablet sidebar — off-canvas drawer or collapsible flex column
  const sidebarEl = el("sidebar");
  const overlayEl = el("sidebarOverlay");
  const sidebarToggle = el("sidebarToggle");

  function isMobileViewport() {
    return window.matchMedia("(max-width: 768px)").matches;
  }

  const sidebarToggleIcon = el("sidebarToggleIcon");
  const ICON_COLLAPSE = // chevrons pointing left <<<
    '<polyline points="9 4 4 10 9 16"/><polyline points="16.5 4 11.5 10 16.5 16"/><polyline points="24 4 19 10 24 16"/>';
  const ICON_EXPAND = // chevrons pointing right >>>
    '<polyline points="4 4 9 10 4 16"/><polyline points="11.5 4 16.5 10 11.5 16"/><polyline points="19 4 24 10 19 16"/>';

  function openSidebar() {
    sidebarEl.classList.remove("collapsed");
    document.body.classList.remove("sidebar-collapsed");
    if (isMobileViewport()) {
      overlayEl.classList.add("open");
    } else {
      overlayEl.classList.remove("open");
    }
    if (sidebarToggleIcon) {
      sidebarToggleIcon.setAttribute("viewBox", "0 0 28 20");
      sidebarToggleIcon.innerHTML = ICON_COLLAPSE;
    }
  }

  function closeSidebar() {
    sidebarEl.classList.add("collapsed");
    document.body.classList.add("sidebar-collapsed");
    overlayEl.classList.remove("open");
    if (sidebarToggleIcon) {
      sidebarToggleIcon.setAttribute("viewBox", "0 0 28 20");
      sidebarToggleIcon.innerHTML = ICON_EXPAND;
    }
  }

  sidebarToggle.addEventListener("click", () => {
    if (sidebarEl.classList.contains("collapsed")) openSidebar();
    else closeSidebar();
  });

  const sidebarCloseBtn = el("sidebarCloseBtn");
  if (sidebarCloseBtn) {
    sidebarCloseBtn.addEventListener("click", (e) => {
      e.preventDefault();
      closeSidebar();
    });
  }

  overlayEl.addEventListener("click", () => {
    if (isMobileViewport()) {
      closeSidebar();
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && isMobileViewport()) closeSidebar();
  });

  // Sidebar starts collapsed on phones, open on tablets/desktop
  function applyResponsiveSidebarDefault() {
    if (isMobileViewport()) {
      closeSidebar();
    } else {
      openSidebar();
    }
  }
  applyResponsiveSidebarDefault();
  window.addEventListener("resize", applyResponsiveSidebarDefault);

  // Selecting a chat on mobile should close the sidebar
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

  const msgs = Array.isArray(data.messages) ? data.messages : [];

  // Track all collections that have already been attached to a user query message in this chat
  const alreadyQueriedCollections = new Set();
  msgs.forEach((m) => {
    if (m.role === "human" && Array.isArray(m.attachments)) {
      m.attachments.forEach((att) => {
        if (typeof att === "string") {
          alreadyQueriedCollections.add(att);
        } else if (att && att.collection) {
          alreadyQueriedCollections.add(att.collection);
        }
      });
    }
  });

  // Restore workspace sources from backend attachments
  workspaceState.sources = (data.attachments || []).map((a) => {
    // If an attachment has never been attached to any user query message in this chat,
    // it remains pending (isNew = true). This keeps it visible as a chip above
    // the input area across chat switches, and attaches it to the first query sent.
    const isAlreadyQueried = Boolean(a.collection && alreadyQueriedCollections.has(a.collection));
    return {
      localId: uid(),
      name: a.name,
      collection: a.collection,
      type: a.type || "doc",
      status: "ready",
      isNew: !isAlreadyQueried,
      active: true, // included in query scope by default; user can toggle off
    };
  });

  localStorage.setItem(SESSION_KEY, sessionId);

  // Restore message thread
  clearMessages();
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
  if (conversationState.uploadInProgress) {
    return showToast("Please wait for current upload to complete.", "error");
  }
  await createNewSession();
  renderWelcomeOnly();
  await loadSessions();
}

async function switchSession(sessionId) {
  if (conversationState.uploadInProgress) {
    return showToast("Please wait for current upload to complete.", "error");
  }
  if (sessionId === workspaceState.sessionId) return;
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
          src.isNew = true;
        }
        setUploadBusy(hasProcessing());
        renderComposerChips();
        renderSourcesPanel();
        await loadSessions();
        showToast(`${file.name} uploaded successfully.`, "success");
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
          finalSrc.isNew = true;
        }
        setUploadBusy(hasProcessing());
        renderComposerChips();
        renderSourcesPanel();
        await loadSessions();
        showToast("YouTube video indexed successfully.", "success");
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
  if (
    conversationState.uploadInProgress ||
    workspaceState.sources.some((s) => s.status === "processing")
  ) {
    return showToast(
      "Please wait until the source finishes processing.",
      "error"
    );
  }
  if (conversationState.isLoading) return;

  // Check if any sources are uploaded/ready
  const readySources = workspaceState.sources.filter(
    (s) => s.status === "ready" && s.collection
  );

  // Determine scoped collections
  let scopedCollectionNames = [];

  if (readySources.length > 0) {
    const activeSources = readySources.filter((s) => s.active !== false);
    if (!activeSources.length) {
      showToast(
        "All sources are excluded. Click a source chip to include it.",
        "error"
      );
      return;
    }
    scopedCollectionNames = activeSources.map((s) => s.collection);
  } else {
    scopedCollectionNames = [];
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
  renderSourcesPanel();
  updateSendButton();

  conversationState.isLoading = true;

  let streamedText = "";
  let messageRow = null;
  let bubbleEl = null;
  let activeToolRunning = false;
  let activeToolLabel = "";
  let activeToolBadgeType = "tool";

  // Single unified circle-spinner status card created immediately
  const container = el("messages");
  messageRow = document.createElement("div");
  messageRow.className = "msg-row ai";
  messageRow.innerHTML = `
    <div class="msg-avatar">AI</div>
    <div class="msg-body">
      <div class="tool-status-badge structuring" id="currentToolBadge">
        <span class="tool-spinner"></span>
        <span class="tool-label">Structuring answer...</span>
      </div>
      <div class="msg-bubble" style="display:none;"></div>
    </div>`;
  container.appendChild(messageRow);
  bubbleEl = messageRow.querySelector(".msg-bubble");
  container.scrollTop = container.scrollHeight;

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
        onStatus: (statusData) => {
          if (activeToolRunning) return;
          const statusLabel = statusData.message || "Retrieving data...";
          const badge = messageRow ? messageRow.querySelector("#currentToolBadge") : null;
          if (badge) {
            badge.className = `tool-status-badge ${statusData.stage || "search"}`;
            badge.innerHTML = `<span class="tool-spinner"></span><span class="tool-label">${statusLabel}</span>`;
            badge.style.display = "inline-flex";
          }
          container.scrollTop = container.scrollHeight;
        },
        onToolStatus: (toolData) => {
          activeToolRunning = true;
          activeToolLabel = toolData.message || `🔧 Using ${toolData.tool || "tool"}...`;
          activeToolBadgeType = toolData.badge_type || "tool";
          const badgeClass = `tool-status-badge ${activeToolBadgeType}`;

          if (!messageRow) {
            const container = el("messages");
            messageRow = document.createElement("div");
            messageRow.className = "msg-row ai";
            messageRow.innerHTML = `
              <div class="msg-avatar">AI</div>
              <div class="msg-body">
                <div class="${badgeClass}" id="currentToolBadge">
                  <span class="tool-spinner"></span>
                  <span class="tool-label">${activeToolLabel}</span>
                </div>
                <div class="msg-bubble" style="display:none;"></div>
              </div>`;
            container.appendChild(messageRow);
            bubbleEl = messageRow.querySelector(".msg-bubble");
          } else {
            let badge = messageRow.querySelector("#currentToolBadge");
            if (!badge) {
              badge = document.createElement("div");
              badge.id = "currentToolBadge";
              messageRow.querySelector(".msg-body").prepend(badge);
            }
            badge.className = badgeClass;
            badge.innerHTML = `<span class="tool-spinner"></span><span class="tool-label">${activeToolLabel}</span>`;
            badge.style.display = "inline-flex";
          }
          const container = el("messages");
          container.scrollTop = container.scrollHeight;
        },
        onToolStart: (toolData) => {
          if (!activeToolRunning && toolData) {
            const toolDisplayMap = {
              "search_tool": ["search", "🔍 Searching the web..."],
              "duckduckgo_search": ["search", "🔍 Searching the web..."],
              "calculator_tool": ["calc", "🧮 Calculating..."],
              "calculator": ["calc", "🧮 Calculating..."],
              "stock_price_tool": ["stock", "📈 Fetching stock price..."],
              "get_stock_price": ["stock", "📈 Fetching stock price..."],
              "get_weather": ["weather", "🌤️ Checking weather..."],
              "search_arxiv": ["research", "📚 Searching academic papers..."],
              "summarize_document": ["rag", "Reading document context..."],
              "rag_query": ["rag", "Searching knowledge base..."],
            };
            const match = toolDisplayMap[toolData.name] || ["tool", `🔧 Using ${toolData.name || "tool"}...`];
            activeToolBadgeType = match[0];
            activeToolLabel = match[1];
            const badgeClass = `tool-status-badge ${activeToolBadgeType}`;

            if (!messageRow) {
              const container = el("messages");
              messageRow = document.createElement("div");
              messageRow.className = "msg-row ai";
              messageRow.innerHTML = `
                <div class="msg-avatar">AI</div>
                <div class="msg-body">
                  <div class="${badgeClass}" id="currentToolBadge">
                    <span class="tool-spinner"></span>
                    <span class="tool-label">${activeToolLabel}</span>
                  </div>
                  <div class="msg-bubble" style="display:none;"></div>
                </div>`;
              container.appendChild(messageRow);
              bubbleEl = messageRow.querySelector(".msg-bubble");
            } else {
              let badge = messageRow.querySelector("#currentToolBadge");
              if (!badge) {
                badge = document.createElement("div");
                badge.id = "currentToolBadge";
                messageRow.querySelector(".msg-body").prepend(badge);
              }
              badge.className = badgeClass;
              badge.innerHTML = `<span class="tool-spinner"></span><span class="tool-label">${activeToolLabel}</span>`;
              badge.style.display = "inline-flex";
            }
          }
        },
        onToolEnd: () => {
          // Keep tool badge visible with tool identity until tokens arrive or synthesis begins
          const badge = messageRow ? messageRow.querySelector("#currentToolBadge") : null;
          if (badge && activeToolRunning) {
            badge.className = `tool-status-badge ${activeToolBadgeType}`;
            badge.innerHTML = `<span class="tool-spinner"></span><span class="tool-label">${activeToolLabel} • Structuring...</span>`;
            badge.style.display = "inline-flex";
          }
        },
        onClarification: (data) => {
          activeToolRunning = false;
          const badge = messageRow ? messageRow.querySelector("#currentToolBadge") : null;
          if (badge) badge.style.display = "none";
          streamedText = "__clarification__"; // Mark so done event doesn't overwrite

          const optionsHtml = (data.options || [])
            .map(
              (opt) =>
                `<button type="button" class="clarify-option" data-query="${escHtml(opt)}">${escHtml(opt)}</button>`
            )
            .join("");

          if (!messageRow) {
            messageRow = document.createElement("div");
            messageRow.className = "msg-row ai";
            container.appendChild(messageRow);
          }

          messageRow.innerHTML = `
            <div class="msg-avatar">AI</div>
            <div class="msg-body">
              <div class="msg-bubble clarify-bubble">
                <p class="clarify-message">${escHtml(data.message || "Your question is a bit vague. Could you be more specific?")}</p>
                <div class="clarify-options">${optionsHtml}</div>
                <div class="clarify-divider"><span>or</span></div>
                <div class="clarify-custom">
                  <input type="text" class="clarify-input" placeholder="Type your refined question..." />
                  <button type="button" class="clarify-send" title="Send" disabled>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                      <path d="M22 2 11 13"/><path d="m22 2-7 20-4-9-9-4Z"/>
                    </svg>
                  </button>
                </div>
              </div>
            </div>`;

          bubbleEl = messageRow.querySelector(".msg-bubble");
          const bubble = messageRow.querySelector(".clarify-bubble");
          const customInput = bubble.querySelector(".clarify-input");
          const customSend = bubble.querySelector(".clarify-send");
          const optionBtns = bubble.querySelectorAll(".clarify-option");

          // Option click -> mutual exclusion, send query
          optionBtns.forEach((btn) => {
            btn.addEventListener("click", () => {
              if (bubble.classList.contains("clarify-resolved")) return;
              bubble.classList.add("clarify-resolved");
              optionBtns.forEach((b) => {
                b.disabled = true;
                b.classList.add("clarify-dimmed");
              });
              btn.classList.remove("clarify-dimmed");
              btn.classList.add("clarify-selected");
              customInput.disabled = true;
              customInput.classList.add("clarify-disabled");
              customSend.disabled = true;
              el("queryInput").value = btn.dataset.query;
              sendMessage();
            });
          });

          // Typing in input -> disable option buttons
          customInput.addEventListener("input", () => {
            const hasText = customInput.value.trim().length > 0;
            customSend.disabled = !hasText;
            optionBtns.forEach((b) => {
              b.disabled = hasText;
              b.classList.toggle("clarify-disabled", hasText);
            });
          });

          // Custom submit
          const handleCustomSubmit = () => {
            if (bubble.classList.contains("clarify-resolved")) return;
            const q = customInput.value.trim();
            if (!q) return;
            bubble.classList.add("clarify-resolved");
            customInput.disabled = true;
            customSend.disabled = true;
            optionBtns.forEach((b) => {
              b.disabled = true;
              b.classList.add("clarify-dimmed");
            });
            el("queryInput").value = q;
            sendMessage();
          };

          customSend.addEventListener("click", handleCustomSubmit);
          customInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleCustomSubmit();
            }
          });

          container.scrollTop = container.scrollHeight;
        },
        onToken: (token) => {
          activeToolRunning = false;
          const badge = messageRow ? messageRow.querySelector("#currentToolBadge") : null;
          if (badge) badge.style.display = "none";
          streamedText += token;
          if (bubbleEl) {
            bubbleEl.style.display = "block";
            bubbleEl.innerHTML = formatContent(streamedText);
          }
          container.scrollTop = container.scrollHeight;
        },
        onCitations: (citations) => {
          if (citations && bubbleEl && streamedText !== FALLBACK_ANSWER_TEXT) {
            const fullContent = `${streamedText}\n\n${citations}`;
            bubbleEl.innerHTML = formatContent(fullContent);
          }
        },
        onDone: async (doneData) => {
          activeToolRunning = false;
          const badge = messageRow ? messageRow.querySelector("#currentToolBadge") : null;
          if (badge) badge.style.display = "none";
          if (doneData.session_id) {
            workspaceState.sessionId = doneData.session_id;
            localStorage.setItem(SESSION_KEY, workspaceState.sessionId);
          }
          if (bubbleEl && doneData.final_answer && streamedText !== "__clarification__") {
            bubbleEl.style.display = "block";
            bubbleEl.innerHTML = formatContent(doneData.final_answer);
          }
          try {
            await loadSessions();
          } catch {}
        },
        onError: async (errData) => {
          if (!streamedText && messageRow) {
            messageRow.remove();
          }
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
            // Knowledge base empty should not block general conversational flow
            return;
          }
          showToast(errData?.message || "Could not complete that request.", "error");
        },
      }
    );
  } catch {
    if (!streamedText && messageRow) {
      messageRow.remove();
    }
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
        : `<button class="chip-remove" data-collection="${escHtml(src.collection)}" data-local-id="${escHtml(src.localId)}" title="Remove source">
             <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3.5">
               <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
             </svg>
           </button>`;

      const displayName = cleanSourceName(src.name);
      return `
        <div class="chip ${processing ? "processing" : ""}" data-local-id="${escHtml(src.localId)}"
             title="${processing ? escHtml(src.processingMessage || "Processing...") : ""}">
          <span class="chip-type">${typeLabel}</span>
          <span class="chip-name" title="${escHtml(displayName)}">${escHtml(displayName)}</span>
          ${removeBtn}
        </div>`;
    })
    .join("");

  row.querySelectorAll(".chip-remove").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const col = btn.dataset.collection;
      const localId = btn.dataset.localId;
      if (col) {
        removeSource(col);
      } else if (localId) {
        removeLocalSource(localId);
        setUploadBusy(hasProcessing());
        renderComposerChips();
        renderSourcesPanel();
      }
    });
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
      const displayName = cleanSourceName(src.name);
      return `
        <div class="source-toggle-item ${inactive ? "inactive" : ""}" data-local-id="${escHtml(src.localId)}">
          <span class="chip-type">${typeLabel}</span>
          <span class="source-toggle-name" title="${escHtml(displayName)}">${escHtml(displayName)}</span>
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

function cleanSourceName(name) {
  if (!name) return "";
  let clean = String(name).trim();
  clean = clean.replace(/\\/g, "/");
  if (clean.includes("/")) {
    clean = clean.split("/").pop();
  }
  clean = clean.replace(/^(?:data\/)?uploads\//i, "");
  clean = clean.replace(/_[a-f0-9]{8}(\.[a-zA-Z0-9]+)$/i, "$1");
  return clean || "Document";
}

function truncateSourceName(name, maxLen = 12) {
  const clean = cleanSourceName(name);
  if (clean.length <= maxLen) return clean;
  return clean.slice(0, maxLen).trim() + "…";
}

function truncateMessageChipName(name, type) {
  const clean = cleanSourceName(name);

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

function buildReadOnlyChips(chips) {
  const mobile = isMobileChipViewport();
  return chips
    .map((c) => {
      const clean = cleanSourceName(c.name);
      const displayName = mobile
        ? truncateSourceName(clean)
        : truncateMessageChipName(clean, c.type);
      return `
      <div class="chip message-chip chip-hover-controls">
        <span class="chip-type">${c.type === "yt" ? "YT" : "DOC"}</span>
        <span class="chip-name" title="${escHtml(clean)}">${escHtml(displayName)}</span>
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

function showWelcome() {
  let welcome = el("welcome");
  if (!welcome) {
    welcome = document.createElement("div");
    welcome.id = "welcome";
    welcome.className = "welcome";
    welcome.innerHTML = `
      <div class="welcome-logo">DV</div>
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
  let stack = el("toastStack");
  if (!stack) {
    stack = document.createElement("div");
    stack.id = "toastStack";
    stack.className = "toast-stack";
    document.body.appendChild(stack);
  }
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

  // 1. Strip any 'uploads/' or 'data/uploads/' from the text and citations
  let raw = String(text).replace(/\b(?:data\/)?uploads[\/\\]/gi, "");

  // 2. Normalize and split glued inline items onto new lines:
  // e.g. "violates the grammar. 2. Parse-Tree Construction" -> "violates the grammar.\n\n2. Parse-Tree Construction"
  raw = raw.replace(/([.!?])\s+(?=\d+[\.\)]\s+[A-Za-z\*\#])/g, "$1\n\n");
  // e.g. "according to the language's grammar. - Primary Tasks" -> "according to the language's grammar.\n\n- Primary Tasks"
  raw = raw.replace(/([.!?])\s+(?=[-•*]\s+[A-Za-z\*\#])/g, "$1\n\n");
  // e.g. "some text. ### Heading" -> "some text.\n\n### Heading"
  raw = raw.replace(/([.!?])\s+(?=#{1,6}\s+)/g, "$1\n\n");

  const lines = raw.split("\n");
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

    // Markdown headings (#, ##, ###, ####)
    const headingMatch = line.match(/^(#{1,6})\s+(.*)$/);
    if (headingMatch) {
      closeList();
      const level = Math.min(Math.max(headingMatch[1].length, 2), 4);
      const headingText = escHtml(headingMatch[2]);
      parts.push(`<h${level} class="msg-heading">${applyInline(headingText)}</h${level}>`);
      return;
    }

    // Bullet list items (- , * , • )
    if (/^[-*•]\s+/.test(line)) {
      if (listType !== "ul") {
        closeList();
        parts.push("<ul>");
        listType = "ul";
      }
      const itemText = escHtml(line.replace(/^[-*•]\s+/, ""));
      parts.push(`<li>${applyInline(itemText)}</li>`);
      return;
    }

    // Numbered list items (1. , 2. , 1) , 2) )
    if (/^\d+[\.\)]\s+/.test(line)) {
      if (listType !== "ol") {
        closeList();
        parts.push("<ol>");
        listType = "ol";
      }
      const itemText = escHtml(line.replace(/^\d+[\.\)]\s+/, ""));
      parts.push(`<li>${applyInline(itemText)}</li>`);
      return;
    }

    // Source: citation heading
    if (/^sources?\s*:?$/i.test(line)) {
      closeList();
      parts.push(`<div class="citation-header">Source:</div>`);
      return;
    }

    // Regular paragraph
    closeList();
    parts.push(`<p>${applyInline(escHtml(line))}</p>`);
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
    const headers = { ...(options.headers || {}) };
    if (apiKey && apiKey.trim() && apiKey !== "guest") {
      headers["X-API-Key"] = apiKey.trim();
    }
    const res = await fetch(url, {
      ...options,
      credentials: "include",
      headers,
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
      localStorage.removeItem(API_KEY_STORAGE);
      showToast("Invalid or expired API key. Please enter a valid API key.", "error");
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
 * @param {(data: object) => void} callbacks.onStatus
 * @param {(token: string) => void} callbacks.onToken
 * @param {(citations: string) => void} callbacks.onCitations
 * @param {(data: object) => void} callbacks.onToolStart
 * @param {(data: object) => void} callbacks.onToolEnd
 * @param {(data: object) => void} callbacks.onDone
 * @param {(err: object) => void} callbacks.onError
 * @param {(data: object) => void} [callbacks.onClarification]
 */
async function streamQueryAnswer(
  url,
  payload,
  { onStatus, onToolStatus, onToken, onCitations, onToolStart, onToolEnd, onDone, onError, onClarification }
) {
  try {
    const apiKey = localStorage.getItem(API_KEY_STORAGE);
    const headers = {
      "Content-Type": "application/json",
    };
    if (apiKey && apiKey.trim() && apiKey !== "guest") {
      headers["X-API-Key"] = apiKey.trim();
    }
    const res = await fetch(url, {
      method: "POST",
      credentials: "include",
      headers,
      body: JSON.stringify(payload),
    });

    if (res.status === 401) {
      localStorage.removeItem(API_KEY_STORAGE);
      showToast("Invalid or expired API key. Please enter a valid API key.", "error");
      ensureApiKey();
      onError?.({ error_code: "unauthorized", message: "Invalid API key" });
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
          if (event.type === "status") {
            onStatus?.(event);
          } else if (event.type === "tool_status") {
            onToolStatus?.(event);
          } else if (event.type === "token") {
            onToken?.(event.content);
          } else if (event.type === "clarification") {
            onClarification?.(event);
          } else if (event.type === "citations") {
            onCitations?.(event.citations);
          } else if (event.type === "tool_start") {
            onToolStart?.(event);
          } else if (event.type === "tool_end") {
            onToolEnd?.(event);
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