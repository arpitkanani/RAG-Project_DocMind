"use strict";

// ══════════════════════ PERFORMANCE TELEMETRY (Component 0) ══════════════════════
const _appUrlParams = new URLSearchParams(window.location.search);
const isPerf = _appUrlParams.get("perf") === "1" || sessionStorage.getItem("dv_perf") === "1";
if (_appUrlParams.get("perf") === "1") {
  sessionStorage.setItem("dv_perf", "1");
}

if (isPerf) {
  const _origAppFetch = window.fetch;
  window.fetch = async function (...args) {
    const t0 = performance.now();
    const url = typeof args[0] === "string" ? args[0] : (args[0]?.url || "unknown");
    const method = (args[1]?.method || "GET").toUpperCase();
    try {
      const res = await _origAppFetch.apply(this, args);
      const dt = (performance.now() - t0).toFixed(1);
      console.log(`[perf-fetch] ${method} ${url} ${dt}ms (status ${res.status})`);
      return res;
    } catch (err) {
      const dt = (performance.now() - t0).toFixed(1);
      console.warn(`[perf-fetch] ${method} ${url} FAILED in ${dt}ms:`, err);
      throw err;
    }
  };
}

// Emit initial load telemetry
if (window.__reportTelemetry) {
  window.__reportTelemetry({ type: "info", message: "app_new.js (v47) top-level execution started" });
}

// Global error catcher — if the script throws at the top level, show it visibly and send telemetry
window.onerror = function (msg, url, line, col, error) {
  console.error("[DocuVortex] Uncaught error:", msg, "at", url, "line", line);
  if (window.__reportTelemetry) {
    window.__reportTelemetry({
      type: "error",
      message: String(msg),
      file: String(url),
      line: line,
      col: col,
      stack: error && error.stack ? String(error.stack) : ""
    });
  }
  try {
    const stack = document.getElementById("toastStack");
    if (stack) {
      const t = document.createElement("div");
      t.className = "toast error";
      t.textContent = "[JS Error] " + msg + " (line " + line + ")";
      stack.appendChild(t);
      setTimeout(() => t.remove(), 8000);
    }
  } catch (_) {}
};

window.onunhandledrejection = function (event) {
  console.error("[DocuVortex] Unhandled promise rejection:", event.reason);
  if (window.__reportTelemetry) {
    window.__reportTelemetry({
      type: "unhandledrejection",
      message: event.reason ? (event.reason.message || String(event.reason)) : "Unknown rejection",
      stack: event.reason && event.reason.stack ? String(event.reason.stack) : ""
    });
  }
};

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
 *   isNew: boolean,
 *   needs_reindex?: boolean,
 *   embedding_model?: string,
 *   vector_dim?: number
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
  settingsEmbedding: "/api/settings/embedding",
  wsChat: "/ws/chat",
};

const SESSION_KEY = "docuvortex.activeSessionId";
const API_KEY_STORAGE = "docuvortex.apiKey";
const FALLBACK_ANSWER_TEXT = "I couldn't find information about that in the uploaded document(s).";

// Server-provided last active session ID (recovered during auth check)
let _serverLastSessionId = null;
let _checkedSessionStatus = false;

// Persistent WebSocket and background active stream manager
let _chatWs = null;
let _wsReconnectTimer = null;
let _wsPingInterval = null;
const activeStreams = {};

// ═══════════════════════════════════════════════════════════════════
//  BOOT
// ═══════════════════════════════════════════════════════════════════

async function initApp() {
  console.log("[DocuVortex] Starting application initialization...");
  if (window.__reportTelemetry) {
    window.__reportTelemetry({ type: "info", message: "initApp() started" });
  }

  // Phase 1: Bind UI event handlers
  try {
    bindUI();
    console.log("[DocuVortex] ✅ UI bindings completed.");
    if (window.__reportTelemetry) {
      window.__reportTelemetry({ type: "info", message: "bindUI() completed successfully" });
    }
  } catch (err) {
    console.error("[DocuVortex] ❌ bindUI error:", err);
    if (window.__reportTelemetry) {
      window.__reportTelemetry({ type: "error", message: "bindUI error: " + (err.message || err), stack: err.stack });
    }
    showToast("UI initialization warning: " + (err.message || err), "error");
  }

  // Phase 2: Session Bootstrap & Workspace Restoration
  try {
    const success = await ensureSessionAndBoot();
    if (success) {
      connectWebSocket();
    }
    console.log("[DocuVortex] ✅ Workspace boot complete. Authenticated session active.");
    if (window.__reportTelemetry) {
      window.__reportTelemetry({ type: "info", message: "ensureSessionAndBoot() completed successfully" });
    }
  } catch (err) {
    console.error("[DocuVortex] ❌ Workspace boot error:", err);
    if (window.__reportTelemetry) {
      window.__reportTelemetry({ type: "error", message: "Workspace boot error: " + (err.message || err), stack: err.stack });
    }
    showToast("Workspace load warning: " + (err.message || err), "error");
  }

  console.log("[DocuVortex] 🚀 Application initialization complete.");
  if (window.__reportTelemetry) {
    window.__reportTelemetry({ type: "info", message: "🚀 Application initialization complete." });
  }

  if (isPerf) {
    const clickTs = sessionStorage.getItem("dv_perf_click_ts");
    if (clickTs) {
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          const elapsed = Date.now() - parseInt(clickTs, 10);
          const action = sessionStorage.getItem("dv_perf_action") || "login";
          console.log(`[perf-ui] ${action} | next page ready=${elapsed}ms since click`);
          sessionStorage.removeItem("dv_perf_click_ts");
          sessionStorage.removeItem("dv_perf_action");
        });
      });
    }
  }
}

// Ensure DOM is ready before running initApp
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initApp);
} else {
  initApp();
}

/**
 * Ensures session is loaded directly from HTTP-only session cookie.
 * No API key prompt is ever presented to the user.
 */
async function ensureSessionAndBoot() {
  // Check if returning from landing page or direct URL with preferred session
  try {
    const urlParams = new URLSearchParams(window.location.search);
    const stagedSessionId = (urlParams.get("session_id") || "").trim();
    if (stagedSessionId) {
      _serverLastSessionId = stagedSessionId;
      localStorage.setItem(SESSION_KEY, stagedSessionId);
      if (window.location.search) {
        window.history.replaceState({}, document.title, window.location.pathname);
      }
    }
  } catch (e) {}

  return await bootApp();
}

async function bootApp() {
  const apiKey = localStorage.getItem(API_KEY_STORAGE);
  const preferredSessionId = localStorage.getItem(SESSION_KEY) || _serverLastSessionId || "";

  // 1-Step Consolidated Fast Bootstrap: user info, sidebar sessions, active session in 1 request
  try {
    const bootstrapUrl = preferredSessionId
      ? `/api/sessions/bootstrap?preferred_session_id=${encodeURIComponent(preferredSessionId)}`
      : "/api/sessions/bootstrap";

    const headers = {};
    if (apiKey) headers["X-API-Key"] = apiKey;

    let res = await fetch(bootstrapUrl, {
      headers,
      credentials: "include",
    });

    // If initial fetch returned 401 and an explicit API key was sent, retry with cookie credentials alone
    if (res.status === 401 && apiKey) {
      localStorage.removeItem(API_KEY_STORAGE);
      res = await fetch(bootstrapUrl, {
        credentials: "include",
      });
    }

    if (res.status === 401) {
      _sessionCache.clear();
      localStorage.removeItem(SESSION_KEY);
      window.location.replace("/");
      return;
    }

    if (res.ok) {
      const data = await res.json();
      if (data && data.authenticated) {
        sidebarState.chats = data.sessions || [];
        renderSidebar();



        if (data.active_session && data.active_session.session_id) {
          const activeId = data.active_session.session_id;
          _sessionCache.set(activeId, data.active_session);
          _applySessionData(activeId, data.active_session);
          return;
        } else {
          await createNewSession();
          renderWelcomeOnly();
          return;
        }
      }
    }
  } catch (err) {
    console.warn("[DocuVortex] Bootstrap fetch failed, falling back to legacy load:", err);
  }

  // Fallback: Legacy load if bootstrap failed
  const sessions = await loadSessions();
  let candidateSessionId = null;
  const userSessionIds = Array.isArray(sessions) ? new Set(sessions.map((s) => s.session_id)) : new Set();
  const storedId = localStorage.getItem(SESSION_KEY);

  if (_serverLastSessionId && userSessionIds.has(_serverLastSessionId)) {
    candidateSessionId = _serverLastSessionId;
  } else if (storedId && userSessionIds.has(storedId)) {
    candidateSessionId = storedId;
  } else if (sessions && sessions.length > 0) {
    candidateSessionId = sessions[0].session_id;
  }

  // Restore the candidate session
  if (candidateSessionId) {
    const ok = await restoreSession(candidateSessionId);
    if (ok) return;
    localStorage.removeItem(SESSION_KEY);
  }

  // Fall back to most recent available session if candidate restore failed
  if (sessions && sessions.length > 0) {
    const latest = sessions[0];
    const ok = await restoreSession(latest.session_id);
    if (ok) return;
  }

  // No prior sessions found in PostgreSQL — start fresh
  await createNewSession();
  renderWelcomeOnly();
}

// ═══════════════════════════════════════════════════════════════════
//  UI BINDING
// ═══════════════════════════════════════════════════════════════════

function safeOn(idOrEl, event, handler) {
  const elem = typeof idOrEl === "string" ? el(idOrEl) : idOrEl;
  if (elem && typeof elem.addEventListener === "function") {
    elem.addEventListener(event, handler);
    return elem;
  }
  return null;
}

function bindUI() {
  // 1. Sidebar core & footer
  try {
    safeOn("newChatBtn", "click", () => {
      closeSidebarSearch();
      handleNewChat();
    });

    const searchToggleBtn = el("sidebarSearchToggleBtn");
    const searchExpandable = el("sidebarSearchExpandable");
    const searchInput = el("searchInput");
    const searchCloseBtn = el("searchCloseBtn");
    const sidebar = el("sidebar");

    function openSidebarSearch() {
      if (!searchExpandable) return;
      searchExpandable.style.display = "block";
      if (searchToggleBtn) searchToggleBtn.classList.add("active");
      if (sidebar) sidebar.classList.add("search-active");
      if (searchInput) {
        searchInput.focus();
        if (searchInput.value) {
          sidebarState.searchQuery = searchInput.value.toLowerCase().trim();
          renderSidebar();
        }
      }
    }

    function closeSidebarSearch() {
      if (!searchExpandable) return;
      searchExpandable.style.display = "none";
      if (searchToggleBtn) searchToggleBtn.classList.remove("active");
      if (sidebar) sidebar.classList.remove("search-active");
      if (searchInput) searchInput.value = "";
      if (sidebarState.searchQuery) {
        sidebarState.searchQuery = "";
        renderSidebar();
      }
    }

    if (searchToggleBtn) {
      searchToggleBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        if (searchExpandable && searchExpandable.style.display !== "none") {
          closeSidebarSearch();
        } else {
          openSidebarSearch();
        }
      });
    }

    if (searchCloseBtn) {
      searchCloseBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        closeSidebarSearch();
      });
    }

    if (searchInput) {
      searchInput.addEventListener("input", (e) => {
        sidebarState.searchQuery = e.target.value.toLowerCase().trim();
        renderSidebar();
      });
      searchInput.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
          e.preventDefault();
          closeSidebarSearch();
        }
      });
    }

    safeOn("clearAllBtn", "click", handleClearAll);

    safeOn("settingsBtn", "click", () => openSettingsModal("embedding"));
    safeOn("settingsModalClose", "click", closeSettingsModal);
    safeOn("settingsModalCancel", "click", closeSettingsModal);
    safeOn("settingsModalSave", "click", saveSettingsModal);

  } catch (err) {
    console.warn("Sidebar core binding note:", err);
  }

  // 2. Attach button & Upload modal triggers
  try {
    safeOn("attachBtn", "click", (e) => {
      e.preventDefault();
      if (window.innerWidth <= 1024) {
        const m = el("mobileAttachMenu");
        if (m) {
          m.classList.toggle("open");
          const p = el("sourcesPopover");
          if (p) p.classList.remove("open");
        }
        return;
      }
      openModal("file");
    });

    safeOn("ytInput", "keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        handleSubmitUpload();
      }
    });

    const chatFileInput = el("chatFileInput") || el("fileInput");
    if (chatFileInput) {
      chatFileInput.addEventListener("change", (e) => {
        const files = Array.from(e.target.files || []);
        if (files.length) {
          handleFiles(files);
        }
        e.target.value = "";
      });
    }

    safeOn("mobileMenuUpload", "click", () => {
      const m = el("mobileAttachMenu");
      if (m) m.classList.remove("open");
      openModal("file");
    });
    safeOn("mobileMenuYt", "click", () => {
      const m = el("mobileAttachMenu");
      if (m) m.classList.remove("open");
      openModal("yt");
    });
    safeOn("mobileMenuSources", "click", (e) => {
      if (e) {
        e.stopPropagation();
        e.preventDefault();
      }
      const m = el("mobileAttachMenu");
      if (m) m.classList.remove("open");
      toggleSourcesPopover();
    });
    document.addEventListener("click", (e) => {
      const menu = el("mobileAttachMenu");
      const attachBtn = el("attachBtn");
      if (
        menu &&
        menu.classList.contains("open") &&
        !menu.contains(e.target) &&
        e.target !== attachBtn &&
        (!attachBtn || !attachBtn.contains(e.target))
      ) {
        menu.classList.remove("open");
      }
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        const menu = el("mobileAttachMenu");
        if (menu && menu.classList.contains("open")) {
          menu.classList.remove("open");
        }
      }
    });

    safeOn("sourcesToggleBtn", "click", (e) => {
      if (e) {
        e.stopPropagation();
      }
      toggleSourcesPopover();
    });
    document.addEventListener("click", (e) => {
      const popover = el("sourcesPopover");
      const toggleBtn = el("sourcesToggleBtn");
      const mobileSourcesBtn = el("mobileMenuSources");
      const attachMenu = el("mobileAttachMenu");
      const mobilePill = el("mobileActiveSourcesBar");
      if (
        popover &&
        popover.classList.contains("open") &&
        !popover.contains(e.target) &&
        e.target !== toggleBtn &&
        (!toggleBtn || !toggleBtn.contains(e.target)) &&
        e.target !== mobileSourcesBtn &&
        (!mobileSourcesBtn || !mobileSourcesBtn.contains(e.target)) &&
        (!attachMenu || !attachMenu.contains(e.target)) &&
        (!mobilePill || !mobilePill.contains(e.target))
      ) {
        popover.classList.remove("open");
      }
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        const popover = el("sourcesPopover");
        if (popover && popover.classList.contains("open")) {
          popover.classList.remove("open");
        }
      }
    });
    safeOn("modalClose", "click", closeModal);
    safeOn("cancelUpload", "click", closeModal);
    safeOn("submitUpload", "click", handleSubmitUpload);
    safeOn("modalBackdrop", "click", (e) => {
      if (e.target === el("modalBackdrop")) closeModal();
    });
  } catch (err) {
    console.warn("Attach / modal binding note:", err);
  }

  // 3. Mobile & tablet sidebar drawer
  try {
    const sidebarEl = el("sidebar");
    const overlayEl = el("sidebarOverlay");

    function isMobileViewport() {
      return window.matchMedia("(max-width: 768px)").matches;
    }

    const ICON_COLLAPSE =
      '<polyline points="9 4 4 10 9 16"/><polyline points="16.5 4 11.5 10 16.5 16"/><polyline points="24 4 19 10 24 16"/>';
    const ICON_EXPAND =
      '<polyline points="4 4 9 10 4 16"/><polyline points="11.5 4 16.5 10 11.5 16"/><polyline points="19 4 24 10 19 16"/>';

    /**
     * Safely update SVG icon content. Using innerHTML on SVG elements
     * can fail in strict parsers because SVG uses a different XML namespace.
     * This helper removes old children and inserts new ones via a temporary
     * HTML container, which ensures correct namespace handling.
     */
    function setSvgContent(svgEl, svgInnerHtml) {
      if (!svgEl) return;
      try {
        // Try innerHTML first (works in most modern browsers)
        svgEl.innerHTML = svgInnerHtml;
      } catch (_) {
        // Fallback: parse via DOMParser for strict SVG namespace compliance
        try {
          const parser = new DOMParser();
          const svgDoc = parser.parseFromString(
            `<svg xmlns="http://www.w3.org/2000/svg">${svgInnerHtml}</svg>`,
            "image/svg+xml"
          );
          while (svgEl.firstChild) svgEl.removeChild(svgEl.firstChild);
          const parsed = svgDoc.documentElement;
          while (parsed.firstChild) {
            svgEl.appendChild(document.importNode(parsed.firstChild, true));
          }
        } catch (e2) {
          console.warn("SVG content update fallback failed:", e2);
        }
      }
    }

    function openSidebar() {
      if (sidebarEl) sidebarEl.classList.remove("collapsed");
      document.body.classList.remove("sidebar-collapsed");
      if (overlayEl) {
        if (isMobileViewport()) {
          overlayEl.classList.add("open");
        } else {
          overlayEl.classList.remove("open");
        }
      }
      const icon = el("sidebarToggleIcon");
      if (icon) {
        icon.setAttribute("viewBox", "0 0 28 20");
        setSvgContent(icon, ICON_COLLAPSE);
      }
    }

    function closeSidebar() {
      if (sidebarEl) sidebarEl.classList.add("collapsed");
      document.body.classList.add("sidebar-collapsed");
      if (overlayEl) overlayEl.classList.remove("open");
      const icon = el("sidebarToggleIcon");
      if (icon) {
        icon.setAttribute("viewBox", "0 0 28 20");
        setSvgContent(icon, ICON_EXPAND);
      }
    }

    safeOn("sidebarToggle", "click", () => {
      if (sidebarEl && sidebarEl.classList.contains("collapsed")) openSidebar();
      else closeSidebar();
    });

    safeOn("sidebarCloseBtn", "click", (e) => {
      e.preventDefault();
      closeSidebar();
    });

    safeOn("sidebarOverlay", "click", () => {
      if (isMobileViewport()) {
        closeSidebar();
      }
    });

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && isMobileViewport()) closeSidebar();
    });

    function applyResponsiveSidebarDefault() {
      if (isMobileViewport()) {
        closeSidebar();
      } else {
        openSidebar();
      }
    }
    applyResponsiveSidebarDefault();
    window.addEventListener("resize", applyResponsiveSidebarDefault);

    safeOn("sidebarChats", "click", () => {
      if (isMobileViewport()) closeSidebar();
    });
  } catch (err) {
    console.warn("Sidebar responsive binding note:", err);
  }

  // 4. Modal tabs & drop zone
  try {
    const uploadModalEl = el("uploadModal");
    if (uploadModalEl) {
      uploadModalEl.querySelectorAll(".modal-tab").forEach((btn) =>
        btn.addEventListener("click", () => switchTab(btn.dataset.tab))
      );
    }

    const dropZone = el("dropZone");
    if (dropZone) {
      dropZone.addEventListener("click", () => {
        const fi = el("fileInput");
        if (fi) fi.click();
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
        handleFiles(Array.from(e.dataTransfer.files || []));
        closeModal();
      });
    }

    const fileInputEl = el("fileInput");
    if (fileInputEl) {
      fileInputEl.addEventListener("change", (e) => {
        handleFiles(Array.from(e.target.files || []));
        closeModal();
        e.target.value = "";
      });
    }
  } catch (err) {
    console.warn("Drop zone binding note:", err);
  }

  // 5. Query composer and Send action
  try {
    safeOn("sendBtn", "click", sendMessage);
    safeOn("queryInput", "keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
    safeOn("queryInput", "input", function () {
      autoResize.call(this);
      updateSendButton();
    });
  } catch (err) {
    console.warn("Composer binding note:", err);
  }
}

// ═══════════════════════════════════════════════════════════════════
//  SESSION MANAGEMENT
// ═══════════════════════════════════════════════════════════════════

async function createNewSession() {
  try {
    const data = await apiFetch(API.sessions, { method: "POST" });
    workspaceState.sessionId = data?.session_id || ("sess_" + Math.random().toString(36).substring(2, 10));
  } catch {
    workspaceState.sessionId = "sess_" + Math.random().toString(36).substring(2, 10);
  }
  workspaceState.sources = [];
  localStorage.setItem(SESSION_KEY, workspaceState.sessionId);
  renderComposerChips();
  renderSourcesPanel();
  updateSendButton();
}

// In-memory client-side session cache for instant (0ms) chat switching
const _sessionCache = new Map();

function _applySessionData(sessionId, data) {
  workspaceState.sessionId = sessionId;

  const msgs = Array.isArray(data.messages) ? data.messages : [];

  // Normalize attachments on messages (deserializing strings if necessary)
  msgs.forEach((m) => {
    if (typeof m.attachments === "string") {
      try {
        m.attachments = JSON.parse(m.attachments);
      } catch (e) {
        m.attachments = [];
      }
    }
  });

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

  const firstHumanMsg = msgs.find((m) => m.role === "human");
  if (firstHumanMsg && Array.isArray(data.attachments)) {
    if (!Array.isArray(firstHumanMsg.attachments)) {
      firstHumanMsg.attachments = [];
    }
    data.attachments.forEach((a) => {
      if (!a.collection) return;
      alreadyQueriedCollections.add(a.collection);

      const alreadyInAnyMsg = msgs.some(
        (m) =>
          m.role === "human" &&
          Array.isArray(m.attachments) &&
          m.attachments.some(
            (att) => (typeof att === "string" ? att : att?.collection) === a.collection
          )
      );

      if (!alreadyInAnyMsg) {
        firstHumanMsg.attachments.push({
          name: a.name,
          collection: a.collection,
          type: a.type || "doc",
        });
      }
    });
  }

  // Restore workspace sources from backend attachments
  workspaceState.sources = (data.attachments || []).map((a) => {
    const isAlreadyQueried = Boolean(a.collection && alreadyQueriedCollections.has(a.collection));
    const isExpired = Boolean(a.expired || data.is_older_than_7_days);
    return {
      localId: uid(),
      name: a.name,
      collection: a.collection,
      type: a.type || "doc",
      status: "ready",
      isNew: !isAlreadyQueried && !isExpired,
      active: !a.needs_reindex && !isExpired,
      needs_reindex: Boolean(a.needs_reindex),
      expired: isExpired,
      embedding_model: a.embedding_model,
      vector_dim: a.vector_dim,
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

  // 7-day expiration alert if chat is older than 7 days or attachments are expired
  if (data.is_older_than_7_days || data.has_expired_attachments) {
    showToast(
      "This chat is older than 7 days. If you want to chat with your document, please re-upload the document.",
      "info",
      4800
    );
  }

  renderComposerChips();
  renderSourcesPanel();

  // If this session is actively generating in background, re-attach active streaming UI
  const stream = activeStreams[sessionId];
  if (msgs.length > 0 && msgs[msgs.length - 1].role === "ai" && stream) {
    stream.isCompleted = true;
    stream.inFlight = false;
  }

  const isGenuinelyStreaming = Boolean(
    stream &&
    !stream.isCompleted &&
    stream.inFlight &&
    stream.lastActivity &&
    (Date.now() - stream.lastActivity < 25000)
  );

  if (isGenuinelyStreaming) {
    conversationState.isLoading = true;
    reAttachActiveStreamUI(sessionId);
  } else {
    if (stream) {
      stream.isCompleted = true;
      stream.inFlight = false;
    }
    conversationState.isLoading = false;
  }
  updateSendButton();

  // Instant client-side active selection update without blocking sidebar re-fetches!
  updateSidebarActiveItem(sessionId);
  updateSidebarGeneratingBadges();
}

/**
 * Load a session from the backend and restore full workspace state.
 * Uses stale-while-revalidate client caching for instantaneous switching.
 * @returns {Promise<boolean>} true if successfully restored.
 */
async function restoreSession(sessionId) {
  // 1. Instant render from memory cache if available (0ms perceived latency)
  if (_sessionCache.has(sessionId)) {
    _applySessionData(sessionId, _sessionCache.get(sessionId));
  }

  // 2. Fresh fetch from server
  const data = await apiFetch(
    `${API.sessions}/${encodeURIComponent(sessionId)}`
  );
  if (!data || data.error_code) {
    if (!_sessionCache.has(sessionId)) return false;
    return true;
  }

  _sessionCache.set(sessionId, data);
  _applySessionData(sessionId, data);
  return true;
}


function updateSidebarGeneratingBadges() {
  const container = el("sidebarChats");
  if (!container) return;
  container.querySelectorAll(".chat-generating-dot").forEach((dot) => dot.remove());
  container.querySelectorAll(".chat-item.generating").forEach((item) => {
    item.classList.remove("generating");
    const time = item.querySelector(".chat-item-time");
    if (time) time.style.display = "";
  });
}

function updateSidebarActiveItem(activeSessionId) {
  const container = el("sidebarChats");
  if (!container) return;
  container.querySelectorAll(".chat-item").forEach((item) => {
    item.classList.toggle("active", item.dataset.id === activeSessionId);
  });
}

async function loadSessions() {
  const data = await apiFetch(API.sessions);
  sidebarState.chats = data?.sessions || [];
  renderSidebar();

  // Production-grade notice for automatically expired chats (> 7 days)
  if (Array.isArray(data?.expired_sessions) && data.expired_sessions.length > 0) {
    handleExpiredSessionsNotice(data.expired_sessions);
  }

  return sidebarState.chats;
}

function handleExpiredSessionsNotice(expiredList) {
  if (!Array.isArray(expiredList) || !expiredList.length) return;

  const count = expiredList.length;
  if (count === 1) {
    const title = expiredList[0].title || "Untitled Chat";
    showToast(`Notice: Chat "${title}" was automatically cleared after 7 days of inactivity.`, "info", 4500);
  } else {
    const titles = expiredList.slice(0, 2).map((s) => `"${s.title || "Chat"}"`).join(", ");
    const remaining = count > 2 ? ` and ${count - 2} more` : "";
    showToast(`Notice: ${count} chats older than 7 days were automatically cleared (${titles}${remaining}).`, "info", 5000);
  }

  // If the active chat was expired, reset workspace to a fresh chat
  const currentExpired = expiredList.some((s) => s.session_id === workspaceState.sessionId);
  if (currentExpired) {
    localStorage.removeItem(SESSION_KEY);
    createNewSession().then(() => renderWelcomeOnly());
  }
}

async function handleNewChat() {
  if (hasProcessing()) {
    return showToast("Please wait for active document uploads to finish.", "error");
  }
  await createNewSession();
  renderWelcomeOnly();
  await loadSessions();
}

async function switchSession(sessionId) {
  if (hasProcessing()) {
    return showToast("Please wait for active document uploads to finish.", "error");
  }
  if (sessionId === workspaceState.sessionId) return;
  const ok = await restoreSession(sessionId);
  if (!ok) {
    showToast("Could not load that chat.", "error");
    localStorage.removeItem(SESSION_KEY);
    await loadSessions();
  }
  if (_chatWs && _chatWs.readyState === WebSocket.OPEN && sessionId) {
    try {
      _chatWs.send(JSON.stringify({ action: "subscribe", session_id: sessionId }));
    } catch (_) {}
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

  // Optimistic UI update: instantly prune from sidebar (<50ms user perception)
  const prevChats = [...sidebarState.chats];
  sidebarState.chats = sidebarState.chats.filter(
    (c) => c.session_id !== sessionId
  );
  _sessionCache.delete(sessionId);
  renderSidebar();

  const wasActiveSession = workspaceState.sessionId === sessionId;
  if (wasActiveSession) {
    localStorage.removeItem(SESSION_KEY);
    await createNewSession();
    renderWelcomeOnly();
  }

  try {
    const data = await apiFetch(
      `${API.sessions}/${encodeURIComponent(sessionId)}`,
      { method: "DELETE" }
    );

    if (!data?.success) {
      // Rollback on failure
      sidebarState.chats = prevChats;
      renderSidebar();
      showToast(data?.message || "Could not delete chat.", "error");
      return;
    }

    showToast("Chat deleted.", "success");
  } catch {
    // Rollback on network failure
    sidebarState.chats = prevChats;
    renderSidebar();
    showToast("Could not delete chat.", "error");
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
    _sessionCache.clear();
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

function openModal(defaultTab = "file") {
  el("modalBackdrop").classList.add("open");
  switchTab(defaultTab || "file");
}

function closeModal() {
  el("modalBackdrop").classList.remove("open");
  el("ytInput").value = "";
  el("fileInput").value = "";
}

function switchTab(tab) {
  _activeTab = tab;
  const uploadModal = el("uploadModal");
  if (!uploadModal) return;
  uploadModal.querySelectorAll(".modal-tab").forEach((b) =>
    b.classList.toggle("active", b.dataset.tab === tab)
  );
  uploadModal.querySelectorAll(".tab-panel").forEach((p) =>
    p.classList.toggle("active", p.id === `tab-${tab}`)
  );
  const btn = document.getElementById("submitUpload");
  if (btn) {
    btn.textContent = tab === "yt" ? "Add Video" : "Browse Files";
  }
}

async function handleSubmitUpload() {
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
 * queue/progress/retry text.
 */
function pollJobStatus(jobId, localId, { onSuccess, defaultErrorMessage }) {
  const POLL_INTERVAL_MS = 2500;

  const poll = async () => {
    const job = await apiFetch(`${API.upload}/status/${encodeURIComponent(jobId)}`);

    if (!job || job.status === undefined) {
      // Network hiccup polling — try again rather than giving up on one miss
      setTimeout(poll, POLL_INTERVAL_MS);
      return;
    }

    if (job.status === "processing" || job.status === "queued") {
      const src = workspaceState.sources.find((s) => s.localId === localId);
      if (src) {
        src.status = job.status;
        src.processingMessage = job.message || (job.status === "queued" ? "Waiting in queue..." : "Processing...");
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
  const isLarge = Boolean(file && file.size > 10 * 1024 * 1024);

  if (isLarge) {
    const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
    showToast(
      `Large document detected (${sizeMb} MB). Parsing chunks & generating embeddings may take 1-2 minutes. Please wait...`,
      "info",
      6500
    );
  }

  // 1. Immediately render processing chip above textarea
  workspaceState.sources.push({
    localId,
    name: file.name,
    collection: "",
    type: "doc",
    status: "processing",
    isNew: true,
    active: true,
    processingMessage: isLarge ? "Processing large document (>10MB)..." : "Uploading...",
  });
  setUploadBusy(true);
  renderComposerChips();
  renderSourcesPanel();

  try {
    const form = new FormData();
    form.append("file", file);
    if (workspaceState.sessionId) {
      form.append("session_id", workspaceState.sessionId);
    }

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
          src.embedding_model = result.embedding_model;
          src.vector_dim = result.vector_dim;
          src.needs_reindex = false;
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
    name: "YouTube URL (with Caption only)",
    collection: "",
    type: "yt",
    status: "processing",
    isNew: true,
    active: true,
    processingMessage: "Fetching captions...",
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
            : "YouTube URL (with Caption only)";
          finalSrc.collection = result.collection_name;
          finalSrc.status = "ready";
          finalSrc.isNew = true;
          finalSrc.embedding_model = result.embedding_model;
          finalSrc.vector_dim = result.vector_dim;
          finalSrc.needs_reindex = false;
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

// ═══════════════════════════════════════════════════════════════════
//  WEBSOCKET & BACKGROUND STREAM RESILIENCE MANAGER
// ═══════════════════════════════════════════════════════════════════

let _wsFailCount = 0;

function connectWebSocket() {
  if (_chatWs && (_chatWs.readyState === WebSocket.OPEN || _chatWs.readyState === WebSocket.CONNECTING)) {
    return;
  }
  const apiKey = getActiveApiKey();

  try {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    const wsUrl = apiKey
      ? `${proto}//${host}/ws/chat?api_key=${encodeURIComponent(apiKey)}`
      : `${proto}//${host}/ws/chat`;

    _chatWs = new WebSocket(wsUrl);

    _chatWs.onopen = () => {
      console.log("[DocuVortex] ✅ WebSocket connected to /ws/chat");
      _wsFailCount = 0;
      if (_wsReconnectTimer) {
        clearTimeout(_wsReconnectTimer);
        _wsReconnectTimer = null;
      }
      if (_wsPingInterval) clearInterval(_wsPingInterval);
      _wsPingInterval = setInterval(() => {
        if (_chatWs && _chatWs.readyState === WebSocket.OPEN) {
          _chatWs.send(JSON.stringify({ action: "ping" }));
        }
      }, 25000);

      // Re-subscribe to any active background streams
      Object.keys(activeStreams).forEach((sid) => {
        if (!activeStreams[sid].isCompleted) {
          _chatWs.send(JSON.stringify({ action: "subscribe", session_id: sid }));
        }
      });
    };

    _chatWs.onmessage = (e) => {
      try {
        const eventData = JSON.parse(e.data);
        handleIncomingStreamEvent(eventData);
      } catch (err) {
        console.warn("[DocuVortex] Failed to parse WebSocket message:", err, e.data);
      }
    };

    _chatWs.onclose = (event) => {
      if (_wsPingInterval) clearInterval(_wsPingInterval);
      _chatWs = null;
      _wsFailCount++;
      if (event.code !== 1000 && event.code !== 4401 && _wsFailCount <= 3 && !_wsReconnectTimer) {
        _wsReconnectTimer = setTimeout(() => {
          _wsReconnectTimer = null;
          connectWebSocket();
        }, 5000);
      }
    };

    _chatWs.onerror = (err) => {
      console.warn("[DocuVortex] WebSocket error:", err);
    };
  } catch (err) {
    console.warn("[DocuVortex] WebSocket initialization failed, using HTTP SSE fallback:", err);
  }
}

function attachActiveStreamUI(sessionId) {
  const container = el("messages");
  if (!container) return;

  const existing = container.querySelector("#activeStreamRow");
  if (existing) existing.remove();

  const stream = activeStreams[sessionId];
  if (!stream) return;

  const row = document.createElement("div");
  row.className = "msg-row ai";
  row.id = "activeStreamRow";
  row.dataset.sessionId = sessionId;
  row.innerHTML = `
    <div class="msg-avatar">AI</div>
    <div class="msg-body">
      <div class="tool-status-badge ${stream.statusStage || "classifying"}" id="currentToolBadge">
        <span class="tool-spinner"></span>
        <span class="tool-label">${escHtml(stream.statusMessage || "Analyzing query...")}</span>
      </div>
      <div class="msg-bubble" style="display:none;"></div>
    </div>`;
  container.appendChild(row);
  container.scrollTop = container.scrollHeight;
}

function reAttachActiveStreamUI(sessionId) {
  const container = el("messages");
  if (!container) return;

  const existing = container.querySelector("#activeStreamRow");
  if (existing) existing.remove();

  const stream = activeStreams[sessionId];
  if (!stream || stream.isCompleted) return;

  if (stream.isClarification && stream.clarificationData) {
    renderClarificationUI(sessionId, stream.clarificationData);
    return;
  }

  const row = document.createElement("div");
  row.className = "msg-row ai";
  row.id = "activeStreamRow";
  row.dataset.sessionId = sessionId;

  let badgeHtml = "";
  let bubbleStyle = "display:none;";
  let bubbleContent = "";

  if (stream.streamedText) {
    bubbleStyle = "display:block;";
    const full = stream.citations ? `${stream.streamedText}\n\n${stream.citations}` : stream.streamedText;
    bubbleContent = formatContent(full);
  } else if (stream.activeToolRunning) {
    badgeHtml = `
      <div class="tool-status-badge ${stream.activeToolBadgeType || "tool"}" id="currentToolBadge" style="display:inline-flex;">
        <span class="tool-spinner"></span>
        <span class="tool-label">${escHtml(stream.activeToolLabel || "Using tool...")}</span>
      </div>`;
  } else {
    badgeHtml = `
      <div class="tool-status-badge ${stream.statusStage || "classifying"}" id="currentToolBadge" style="display:inline-flex;">
        <span class="tool-spinner"></span>
        <span class="tool-label">${escHtml(stream.statusMessage || "Analyzing query...")}</span>
      </div>`;
  }

  row.innerHTML = `
    <div class="msg-avatar">AI</div>
    <div class="msg-body">
      ${badgeHtml}
      <div class="msg-bubble" style="${bubbleStyle}">${bubbleContent}</div>
    </div>`;
  container.appendChild(row);
  container.scrollTop = container.scrollHeight;
}

function updateActiveStreamStatusUI(sessionId) {
  if (workspaceState.sessionId !== sessionId) return;
  const stream = activeStreams[sessionId];
  if (!stream) return;
  // If tokens have already started streaming, do not show or re-open the status badge
  if (stream.streamedText && stream.streamedText.length > 0) return;
  const row = el("activeStreamRow");
  if (!row) return;
  const badge = row.querySelector("#currentToolBadge");
  if (badge) {
    badge.className = `tool-status-badge ${stream.statusStage || "classifying"}`;
    badge.innerHTML = `<span class="tool-spinner"></span><span class="tool-label">${escHtml(stream.statusMessage || "Analyzing query...")}</span>`;
    badge.style.display = "inline-flex";
  }
  const container = el("messages");
  if (container) container.scrollTop = container.scrollHeight;
}

function updateActiveStreamToolUI(sessionId) {
  if (workspaceState.sessionId !== sessionId) return;
  const stream = activeStreams[sessionId];
  if (!stream) return;
  let row = el("activeStreamRow");
  if (!row) {
    attachActiveStreamUI(sessionId);
    row = el("activeStreamRow");
  }
  if (!row) return;

  let badge = row.querySelector("#currentToolBadge");
  if (!badge) {
    badge = document.createElement("div");
    badge.id = "currentToolBadge";
    row.querySelector(".msg-body").prepend(badge);
  }
  badge.className = `tool-status-badge ${stream.activeToolBadgeType || "tool"}`;
  badge.innerHTML = `<span class="tool-spinner"></span><span class="tool-label">${escHtml(stream.activeToolLabel || "Using tool...")}</span>`;
  badge.style.display = "inline-flex";

  const container = el("messages");
  if (container) container.scrollTop = container.scrollHeight;
}

function updateActiveStreamToolEndUI(sessionId) {
  if (workspaceState.sessionId !== sessionId) return;
  const stream = activeStreams[sessionId];
  if (!stream) return;
  const row = el("activeStreamRow");
  if (!row) return;
  const badge = row.querySelector("#currentToolBadge");
  if (badge && stream.activeToolLabel) {
    badge.className = `tool-status-badge ${stream.activeToolBadgeType || "tool"}`;
    badge.innerHTML = `<span class="tool-spinner"></span><span class="tool-label">${escHtml(stream.activeToolLabel)} • Structuring...</span>`;
    badge.style.display = "inline-flex";
  }
}

function updateActiveStreamTokenUI(sessionId) {
  if (workspaceState.sessionId !== sessionId) return;
  const stream = activeStreams[sessionId];
  if (!stream) return;
  let row = el("activeStreamRow");
  if (!row) {
    attachActiveStreamUI(sessionId);
    row = el("activeStreamRow");
  }
  if (!row) return;

  const badge = row.querySelector("#currentToolBadge");
  if (badge) {
    badge.style.display = "none";
    badge.remove();
  }

  const bubbleEl = row.querySelector(".msg-bubble");
  if (bubbleEl) {
    bubbleEl.style.display = "block";
    bubbleEl.innerHTML = formatContent(stream.streamedText);
  }
  const container = el("messages");
  if (container) container.scrollTop = container.scrollHeight;
}

function updateActiveStreamCitationsUI(sessionId) {
  if (workspaceState.sessionId !== sessionId) return;
  const stream = activeStreams[sessionId];
  if (!stream || !stream.citations) return;
  const row = el("activeStreamRow");
  if (!row) return;
  const bubbleEl = row.querySelector(".msg-bubble");
  if (bubbleEl && stream.streamedText && stream.streamedText !== FALLBACK_ANSWER_TEXT) {
    const cleanText = stream.streamedText.trim();
    const hasSource = cleanText.includes(stream.citations.trim()) || /(?:sources?|citations?)\s*:\s*\n/i.test(cleanText);
    const full = hasSource ? cleanText : `${cleanText}\n\n${stream.citations}`;
    bubbleEl.innerHTML = formatContent(full);
  }
}

function finalizeActiveStreamUI(sessionId) {
  if (workspaceState.sessionId !== sessionId) return;
  const stream = activeStreams[sessionId];
  if (!stream) return;
  const row = el("activeStreamRow");
  if (!row) return;

  row.removeAttribute("id");
  const badge = row.querySelector("#currentToolBadge");
  if (badge) badge.remove();

  const bubbleEl = row.querySelector(".msg-bubble");
  if (bubbleEl && stream.finalAnswer && !stream.isClarification) {
    bubbleEl.style.display = "block";
    const cleanFinal = stream.finalAnswer.trim();
    const hasSource = /(?:sources?|citations?)\s*:\s*\n/i.test(cleanFinal) || (stream.citations && cleanFinal.includes(stream.citations.trim()));
    const full = (!hasSource && stream.citations) ? `${cleanFinal}\n\n${stream.citations}` : cleanFinal;
    bubbleEl.innerHTML = formatContent(full);
  }
}

function handleActiveStreamErrorUI(sessionId, errData) {
  if (workspaceState.sessionId !== sessionId) return;
  const stream = activeStreams[sessionId];
  const row = el("activeStreamRow");
  if (row && (!stream || !stream.streamedText)) {
    row.remove();
  }
  if (errData?.error_code === "embedding_unavailable") {
    showToast("The embedding model is unavailable. You can switch the embedding model in Settings.", "error");
    return;
  }
  if (
    errData?.error_code === "vectorstore_unavailable" ||
    (errData?.message && errData.message.toLowerCase().includes("vector database"))
  ) {
    showToast(
      errData.message || "Vector database service (Qdrant Cloud) is temporarily unavailable. Please verify your connection or try again.",
      "error",
      6000
    );
    return;
  }
  if (
    errData?.error_code === "collection_needs_reindex" ||
    (errData?.message && (errData.message.toLowerCase().includes("indexed using") || errData.message.toLowerCase().includes("indexed with")))
  ) {
    showToast(
      errData.message || "This document was indexed with a different embedding model. Please switch the embedding model in Settings or re-upload the document.",
      "error"
    );
    restoreSession(sessionId).then(() => {
      renderComposerChips();
      renderSourcesPanel();
    });
    return;
  }
  if (errData?.error_code === "collection_not_found") {
    const missing = Array.isArray(errData.missing_collections) ? errData.missing_collections : [];
    workspaceState.sources = workspaceState.sources.filter((s) => !missing.includes(s.collection));
    renderComposerChips();
    renderSourcesPanel();
    loadSessions();
    showToast("One or more sources were removed. Please upload again if needed.", "error");
    return;
  }
  if (errData?.error_code === "knowledge_base_empty") {
    return;
  }
  showToast(errData?.message || "Could not complete that request.", "error");
}

function renderClarificationUI(sessionId, data) {
  if (workspaceState.sessionId !== sessionId) return;
  const container = el("messages");
  if (!container) return;

  let messageRow = el("activeStreamRow");
  if (!messageRow) {
    messageRow = document.createElement("div");
    messageRow.className = "msg-row ai";
    messageRow.id = "activeStreamRow";
    messageRow.dataset.sessionId = sessionId;
    container.appendChild(messageRow);
  }

  const optionsHtml = (data.options || [])
    .map(
      (opt) =>
        `<button type="button" class="clarify-option" data-query="${escHtml(opt)}">${escHtml(opt)}</button>`
    )
    .join("");

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

  const bubble = messageRow.querySelector(".clarify-bubble");
  const customInput = bubble.querySelector(".clarify-input");
  const customSend = bubble.querySelector(".clarify-send");
  const optionBtns = bubble.querySelectorAll(".clarify-option");

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

  customInput.addEventListener("input", () => {
    const hasText = customInput.value.trim().length > 0;
    customSend.disabled = !hasText;
    optionBtns.forEach((b) => {
      b.disabled = hasText;
      b.classList.toggle("clarify-disabled", hasText);
    });
  });

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
}

function handleIncomingStreamEvent(event) {
  const sid = event.session_id;
  if (!sid) {
    if (event.type === "pong") return;
    if (event.type === "error") {
      showToast(event.message || "An error occurred", "error");
    }
    return;
  }

  let stream = activeStreams[sid];
  if (!stream) {
    stream = {
      sessionId: sid,
      query: "",
      statusStage: "classifying",
      statusMessage: "Analyzing query...",
      inFlight: true,
      lastActivity: Date.now(),
      activeToolRunning: false,
      activeToolLabel: "",
      activeToolBadgeType: "tool",
      streamedText: "",
      citations: "",
      isClarification: false,
      clarificationData: null,
      isCompleted: false,
      finalAnswer: "",
      error: null,
    };
    activeStreams[sid] = stream;
  }
  stream.lastActivity = Date.now();
  stream.inFlight = true;

  const isCurrentSession = (workspaceState.sessionId === sid);

  if (event.type === "status") {
    stream.statusStage = event.stage || "classifying";
    stream.statusMessage = event.message || "Analyzing query...";
    if (isCurrentSession && !stream.activeToolRunning && (!stream.streamedText || stream.streamedText.length === 0)) {
      updateActiveStreamStatusUI(sid);
    }
  } else if (event.type === "tool_status") {
    stream.activeToolRunning = true;
    stream.activeToolLabel = event.message || `🔧 Using ${event.tool || "tool"}...`;
    stream.activeToolBadgeType = event.badge_type || "tool";
    if (isCurrentSession) {
      updateActiveStreamToolUI(sid);
    }
  } else if (event.type === "tool_end") {
    stream.activeToolRunning = false;
    if (isCurrentSession) {
      updateActiveStreamToolEndUI(sid);
    }
  } else if (event.type === "token") {
    stream.activeToolRunning = false;
    stream.streamedText += event.content || "";
    if (isCurrentSession) {
      updateActiveStreamTokenUI(sid);
    }
  } else if (event.type === "citations") {
    stream.citations = event.citations || "";
    if (isCurrentSession) {
      updateActiveStreamCitationsUI(sid);
    }
  } else if (event.type === "clarification") {
    stream.isClarification = true;
    stream.clarificationData = event;
    stream.activeToolRunning = false;
    if (isCurrentSession) {
      renderClarificationUI(sid, event);
    }
  } else if (event.type === "done") {
    stream.isCompleted = true;
    stream.inFlight = false;
    stream.finalAnswer = event.final_answer || stream.streamedText;
    stream.citations = event.citations || stream.citations;
    updateSidebarGeneratingBadges();
    _sessionCache.delete(sid);

    if (isCurrentSession) {
      finalizeActiveStreamUI(sid);
      conversationState.isLoading = false;
      updateSendButton();
      loadSessions().catch(() => {});
    } else {
      // In background: refresh sessions list so title/preview update
      loadSessions().catch(() => {});
    }
  } else if (event.type === "error") {
    stream.isCompleted = true;
    stream.inFlight = false;
    stream.error = event;
    updateSidebarGeneratingBadges();

    if (isCurrentSession) {
      handleActiveStreamErrorUI(sid, event);
      conversationState.isLoading = false;
      updateSendButton();
    }
  }
}

async function sendMessage() {
  const input = el("queryInput");
  const query = input.value.trim();
  if (!query) return;
  if (hasProcessing()) {
    return showToast(
      "Please wait until all sources finish processing.",
      "error"
    );
  }
  if (conversationState.isLoading) return;

  // Check if any sources are uploaded/ready and not expired
  const readySources = workspaceState.sources.filter(
    (s) => s.status === "ready" && s.collection && !s.expired
  );

  // Determine scoped collections
  let scopedCollectionNames = [];

  if (readySources.length > 0) {
    const activeSources = readySources.filter((s) => s.active !== false && !s.needs_reindex);
    if (!activeSources.length) {
      const mismatched = readySources.filter((s) => s.needs_reindex);
      if (mismatched.length > 0) {
        const firstM = mismatched[0];
        const mModel = firstM.embedding_model || "a different embedding model";
        showToast(
          `The uploaded document '${firstM.name}' was indexed using ${mModel}. Please switch back to ${mModel} in Settings or re-upload the document.`,
          "error"
        );
        return;
      }
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

  // Collect newly-added chips
  const newSources = workspaceState.sources.filter(
    (s) => s.isNew && s.status === "ready" && s.collection
  );
  const messageAttachments = newSources.map((s) => ({
    name: s.name,
    collection: s.collection,
    type: s.type,
  }));

  // Reset isNew BEFORE rendering so composer chips disappear
  workspaceState.sources.forEach((s) => {
    s.isNew = false;
  });

  hideWelcome();
  appendMessage(query, "user", messageAttachments);

  input.value = "";
  input.style.height = "auto";
  input.style.overflowY = "hidden";
  renderComposerChips();
  renderSourcesPanel();
  updateSendButton();

  const sid = workspaceState.sessionId;
  activeStreams[sid] = {
    sessionId: sid,
    query,
    statusStage: "classifying",
    statusMessage: "Analyzing query...",
    inFlight: true,
    lastActivity: Date.now(),
    activeToolRunning: false,
    activeToolLabel: "",
    activeToolBadgeType: "tool",
    streamedText: "",
    citations: "",
    isClarification: false,
    clarificationData: null,
    isCompleted: false,
    finalAnswer: "",
    error: null,
  };

  attachActiveStreamUI(sid);
  updateSidebarGeneratingBadges();
  conversationState.isLoading = true;
  updateSendButton();

  const payload = {
    query,
    session_id: sid,
    message_attachments: messageAttachments.length ? messageAttachments : null,
    collection_names: scopedCollectionNames,
  };

  // 1. Try sending over WebSocket if open
  if (_chatWs && _chatWs.readyState === WebSocket.OPEN) {
    try {
      _chatWs.send(JSON.stringify({ action: "query", ...payload }));
      return;
    } catch (wsErr) {
      console.warn("[DocuVortex] WS send error, falling back to SSE:", wsErr);
    }
  }

  // 2. Fallback to SSE streamQueryAnswer
  try {
    await streamQueryAnswer(
      API.query,
      payload,
      {
        onStatus: (event) => handleIncomingStreamEvent({ ...event, session_id: sid }),
        onToolStatus: (event) => handleIncomingStreamEvent({ ...event, session_id: sid }),
        onToolStart: (event) => handleIncomingStreamEvent({ ...event, type: "tool_status", session_id: sid }),
        onToolEnd: (event) => handleIncomingStreamEvent({ type: "tool_end", session_id: sid }),
        onToken: (tok) => handleIncomingStreamEvent({ type: "token", content: tok, session_id: sid }),
        onCitations: (cites) => handleIncomingStreamEvent({ type: "citations", citations: cites, session_id: sid }),
        onClarification: (data) => handleIncomingStreamEvent({ ...data, type: "clarification", session_id: sid }),
        onDone: (data) => handleIncomingStreamEvent({ ...data, type: "done", session_id: sid }),
        onError: (err) => handleIncomingStreamEvent({ ...err, type: "error", session_id: sid }),
      }
    );
  } catch (err) {
    handleIncomingStreamEvent({ type: "error", error_code: "server_error", message: "Server is down. Please try again.", session_id: sid });
  } finally {
    if (workspaceState.sessionId === sid) {
      conversationState.isLoading = false;
      updateSendButton();
    }
  }
}


// ═══════════════════════════════════════════════════════════════════
//  RENDER — COMPOSER CHIPS
//
//  Shows pending new workspace sources (isNew === true) above the
//  textarea. When the user sends a query, chips attach to that user
//  message bubble in the thread and disappear from the composer.
//  Subsequent queries stay grounded via workspaceState.sources and
//  can be viewed/toggled in the [Sources] popover.
// ═══════════════════════════════════════════════════════════════════

function renderComposerChips() {
  const row = el("chipsRow");
  if (!row) return;
  const pendingSources = workspaceState.sources.filter((s) => s.isNew && !s.expired);

  if (!pendingSources.length) {
    row.hidden = true;
    row.innerHTML = "";
    return;
  }

  row.hidden = false;
  row.innerHTML = pendingSources
    .map((src) => {
      const processing = src.status === "processing" || src.status === "queued";
      const needsReindex = Boolean(src.needs_reindex);
      const typeLabel = src.type === "yt" ? "YT" : "DOC";
      const removeBtn = processing
        ? `<span class="chip-indicator processing" aria-label="Processing"></span>`
        : `<button class="chip-remove" data-collection="${escHtml(src.collection)}" data-local-id="${escHtml(src.localId)}" title="Remove source">
             <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3.5">
               <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
             </svg>
           </button>`;

      const displayName = cleanSourceName(src.name);
      const reindexBadge = needsReindex
        ? `<span class="chip-reindex-badge" title="Indexed with ${escHtml(src.embedding_model || 'another model')}. Switch model in Settings or re-upload.">Needs re-index</span>`
        : "";
      const chipTitle = processing
        ? escHtml(src.processingMessage || "Processing...")
        : needsReindex
        ? `Indexed with ${escHtml(src.embedding_model || 'another model')}. Switch model in Settings or re-upload to query.`
        : "";

      return `
        <div class="chip ${processing ? "processing" : ""} ${needsReindex ? "needs-reindex" : ""}" data-local-id="${escHtml(src.localId)}"
             title="${chipTitle}">
          <span class="chip-type">${typeLabel}</span>
          <span class="chip-name" title="${escHtml(displayName)}">${escHtml(displayName)}</span>
          ${reindexBadge}
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
  const mobileCount = el("mobileSourcesCount");
  const list = el("sourcesPopoverList");

  if (!readySources.length) {
    if (countBtn) countBtn.hidden = true;
    if (mobileCount) {
      mobileCount.textContent = "";
      mobileCount.style.display = "none";
    }
    if (list) {
      list.innerHTML = '<div style="padding: 12px 8px; font-size: 0.82rem; color: var(--text-muted, #777); text-align: center;">No sources loaded in workspace yet.</div>';
    }
    renderMobileActiveSourcesBar();
    return;
  }
  if (countBtn) countBtn.hidden = false;

  const activeCount = readySources.filter((s) => s.active !== false && !s.needs_reindex && !s.expired).length;
  countLabel.textContent = `${activeCount}/${readySources.length}`;
  if (mobileCount) {
    mobileCount.textContent = `${activeCount}/${readySources.length}`;
    mobileCount.style.display = "inline-flex";
  }

  list.innerHTML = readySources
    .map((src) => {
      const isExpired = Boolean(src.expired);
      const inactive = src.active === false || Boolean(src.needs_reindex) || isExpired;
      const needsReindex = Boolean(src.needs_reindex);
      const typeLabel = src.type === "yt" ? "YT" : "DOC";
      const displayName = cleanSourceName(src.name);
      let stateLabel = "Included";
      let stateStyle = "";
      if (isExpired) {
        stateLabel = "Expired (Re-upload)";
        stateStyle = "color: #b95d09; font-weight: 700;";
      } else if (needsReindex) {
        stateLabel = "Needs re-index";
        stateStyle = "color: #b95d09; font-weight: 700;";
      } else if (inactive) {
        stateLabel = "Excluded";
      }

      return `
        <div class="source-toggle-item ${inactive ? "inactive" : ""} ${needsReindex ? "needs-reindex" : ""} ${isExpired ? "expired" : ""}" data-local-id="${escHtml(src.localId)}">
          <span class="chip-type">${typeLabel}</span>
          <span class="source-toggle-name" title="${escHtml(displayName)}">${escHtml(displayName)}</span>
          <span class="source-toggle-state" style="${stateStyle}">${stateLabel}</span>
        </div>`;
    })
    .join("");

  list.querySelectorAll(".source-toggle-item").forEach((itemEl) => {
    itemEl.addEventListener("click", () => {
      const src = workspaceState.sources.find(
        (s) => s.localId === itemEl.dataset.localId
      );
      if (!src) return;
      if (src.expired) {
        showToast(
          "This document is older than 7 days (or no longer in vector storage). Please re-upload it to search.",
          "info",
          4800
        );
        return;
      }
      if (src.needs_reindex) {
        const modelName = src.embedding_model || "another embedding model";
        showToast(
          `The uploaded document '${src.name}' was indexed using ${modelName}. Please switch back to ${modelName} in Settings or re-upload the document to search.`,
          "error"
        );
        return;
      }
      src.active = src.active === false ? true : false;
      renderSourcesPanel();
    });
  });

  renderMobileActiveSourcesBar();
}

function renderMobileActiveSourcesBar() {
  const bar = el("mobileActiveSourcesBar");
  if (!bar) return;
  const readySources = workspaceState.sources.filter(
    (s) => s.status === "ready" && s.collection && !s.expired
  );
  const activeSources = readySources.filter(
    (s) => s.active !== false && !s.needs_reindex
  );
  if (!readySources.length) {
    bar.hidden = true;
    bar.innerHTML = "";
    return;
  }
  bar.hidden = false;
  bar.innerHTML = `
    <div class="mobile-sources-pill" id="mobileSourcesPill">
      <span class="mobile-sources-pill-dot ${activeSources.length ? 'active' : 'inactive'}"></span>
      <span class="mobile-sources-pill-text">${activeSources.length} of ${readySources.length} sources active</span>
      <button type="button" class="mobile-sources-pill-btn" id="mobilePillToggleBtn">${activeSources.length === readySources.length ? 'Filter' : 'Change'}</button>
    </div>
  `;
  const toggleBtn = el("mobilePillToggleBtn");
  if (toggleBtn) {
    toggleBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      e.preventDefault();
      toggleSourcesPopover();
    });
  }
}

function toggleSourcesPopover() {
  const popover = el("sourcesPopover");
  if (!popover) return;
  const willOpen = !popover.classList.contains("open");
  popover.classList.toggle("open");
  if (willOpen) {
    const attachMenu = el("mobileAttachMenu");
    if (attachMenu) attachMenu.classList.remove("open");
    renderSourcesPanel();
  }
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

  let safeAttachments = attachments;
  if (typeof safeAttachments === "string") {
    try {
      safeAttachments = JSON.parse(safeAttachments);
    } catch {
      safeAttachments = [];
    }
  }
  if (!Array.isArray(safeAttachments)) {
    safeAttachments = [];
  }

  // Only user messages can carry attachment chips; AI messages never do.
  const chipsHtml =
    role === "user" && safeAttachments.length
      ? `<div class="message-attachments">${buildReadOnlyChips(safeAttachments)}</div>`
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
    container.innerHTML = `<div class="empty-sidebar">${q ? "No matching chats found." : "No chats yet. Start a new conversation."}</div>`;
    return;
  }

  const nowDate = new Date();
  const todayMidnight = new Date(nowDate.getFullYear(), nowDate.getMonth(), nowDate.getDate()).getTime();
  const yesterdayMidnight = todayMidnight - 86_400_000;
  const prev7DaysMidnight = todayMidnight - 7 * 86_400_000;

  const groups = { Today: [], Yesterday: [], "Previous 7 Days": [], Older: [] };

  chats.forEach((c) => {
    const ms = new Date(c.last_active || 0).getTime();
    if (ms >= todayMidnight) {
      groups.Today.push(c);
    } else if (ms >= yesterdayMidnight) {
      groups.Yesterday.push(c);
    } else if (ms >= prev7DaysMidnight) {
      groups["Previous 7 Days"].push(c);
    } else {
      groups.Older.push(c);
    }
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
          <span class="chat-item-time">${escHtml(formatChatTime(c.last_active))}</span>
          <button
            class="chat-item-del"
            data-del="${escHtml(c.session_id)}"
            title="Delete chat"
            aria-label="Delete chat"
            ${deleting ? "disabled" : ""}
          >
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
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
      <p>Upload a PDF, DOCX file, or YouTube URL (with Caption only), then ask grounded questions without losing your document context.</p>
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
  updateSendButton();
}

function hasProcessing() {
  return workspaceState.sources.some(
    (s) => s.status === "processing" || s.status === "queued"
  );
}

function removeLocalSource(localId) {
  workspaceState.sources = workspaceState.sources.filter(
    (s) => s.localId !== localId
  );
}

function updateSendButton() {
  const btn = el("sendBtn");
  const input = el("queryInput");
  if (!btn || !input) return;
  const query = input.value.trim();
  const canSend =
    Boolean(query) &&
    !hasProcessing() &&
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

function showToast(message, type = "success", durationMs = 3800) {
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
  setTimeout(() => toast.classList.add("toast-fade"), durationMs);
  setTimeout(() => toast.remove(), durationMs + 600);
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
  let inCitationBlock = false;

  const closeList = () => {
    if (listType) {
      parts.push(listType === "ol" ? "</ol>" : "</ul>");
      listType = null;
    }
  };

  const closeCitationBlock = () => {
    if (inCitationBlock) {
      parts.push("</div></div>");
      inCitationBlock = false;
    }
  };

  lines.forEach((rawLine) => {
    const line = rawLine.trim();
    if (!line) {
      closeList();
      return;
    }

    // Source: citation heading
    if (/^(?:#{1,6}\s*)?sources?\s*:?$/i.test(line)) {
      closeList();
      closeCitationBlock();
      inCitationBlock = true;
      parts.push(`
        <div class="citation-block">
          <div class="citation-header">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
              <polyline points="14 2 14 8 20 8"/>
            </svg>
            <span>Sources &amp; Citations</span>
          </div>
          <div class="citation-list">`);
      return;
    }

    // Inside citation block: render each entry as a structured pill
    if (inCitationBlock) {
      // If a new heading starts, exit citation block
      if (/^#{1,6}\s+/.test(line)) {
        closeCitationBlock();
      } else {
        const clean = line.replace(/^[-*•\d\.\)]+\s*/, "").trim();
        if (clean) {
          const isYt = /youtube|timestamp|video/i.test(clean);
          const icon = isYt ? "🎥" : "📄";
          const pillClass = isYt ? "citation-pill video" : "citation-pill pdf";
          const splitParts = clean.split(/\s*[-–—]\s*/);
          const title = splitParts[0] || clean;
          const loc = splitParts.slice(1).join(" • ");
          const locBadge = loc ? `<span class="citation-badge-loc">${escHtml(loc)}</span>` : "";

          parts.push(`
            <span class="${pillClass}">
              <span class="citation-pill-icon">${icon}</span>
              <span class="citation-pill-title">${escHtml(title)}</span>
              ${locBadge}
            </span>`);
          return;
        }
      }
    }

    // Markdown headings (#, ##, ###, ####)
    const headingMatch = line.match(/^(#{1,6})\s+(.*)$/);
    if (headingMatch) {
      closeList();
      closeCitationBlock();
      const level = Math.min(Math.max(headingMatch[1].length, 2), 4);
      const headingText = escHtml(headingMatch[2]);
      parts.push(`<h${level} class="msg-heading">${applyInline(headingText)}</h${level}>`);
      return;
    }

    // Bullet list items (- , * , • )
    if (/^[-*•]\s+/.test(line)) {
      if (listType !== "ul") {
        closeList();
        closeCitationBlock();
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
        closeCitationBlock();
        parts.push("<ol>");
        listType = "ol";
      }
      const itemText = escHtml(line.replace(/^\d+[\.\)]\s+/, ""));
      parts.push(`<li>${applyInline(itemText)}</li>`);
      return;
    }

    // Regular paragraph
    closeList();
    closeCitationBlock();
    parts.push(`<p>${applyInline(escHtml(line))}</p>`);
  });

  closeList();
  closeCitationBlock();
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

function getActiveApiKey() {
  let key = (localStorage.getItem(API_KEY_STORAGE) || "").trim();
  if (!key || key === "guest" || !key.startsWith("dk_live_")) {
    try {
      const params = new URLSearchParams(window.location.search);
      key = (params.get("api_key") || sessionStorage.getItem("staged_api_key") || "").trim();
    } catch (_) {}
  }
  return (key && key.startsWith("dk_live_")) ? key : "";
}

/**
 * Unified fetch wrapper. Always returns parsed JSON or null on network error.
 * @param {string} url
 * @param {RequestInit} [options]
 * @returns {Promise<any|null>}
 */
async function apiFetch(url, options = {}) {
  try {
    const apiKey = getActiveApiKey();
    const headers = { ...(options.headers || {}) };
    if (apiKey) {
      headers["X-API-Key"] = apiKey;
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
      showToast("Session expired or authentication required. Redirecting to sign in...", "warn");
      setTimeout(() => {
        window.location.replace("/?action=login");
      }, 400);
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

function formatChatTime(iso) {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return "";

    const dateStr = d.toLocaleDateString("en-US", {
      month: "short",
      day: "2-digit",
    });
    const timeStr = d.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
    return `${dateStr}, ${timeStr}`;
  } catch {
    return "";
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
  const newHeight = Math.min(this.scrollHeight, 180);
  this.style.height = `${newHeight}px`;
  if (this.scrollHeight > 180) {
    this.style.overflowY = "auto";
  } else {
    this.style.overflowY = "hidden";
  }
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
    const apiKey = getActiveApiKey();
    const headers = {
      "Content-Type": "application/json",
    };
    if (apiKey) {
      headers["X-API-Key"] = apiKey;
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

// ═══════════════════════════════════════════════════════════════════
//  SETTINGS MODAL (Embedding Model Selector & API Key Management)
// ═══════════════════════════════════════════════════════════════════

async function openSettingsModal() {
  const backdrop = el("settingsModalBackdrop");
  if (!backdrop) return;
  backdrop.classList.add("open");

  try {
    const data = await apiFetch(API.settingsEmbedding);
    if (data?.providers) {
      const active = data.active_provider || "azure";
      const radioAzure = el("radioAzure");
      const radioBge = el("radioBge");
      if (radioAzure) radioAzure.checked = (active === "azure");
      if (radioBge) {
        radioBge.checked = (active === "bge");
        const bgeAvailable = data.providers.bge?.available !== false;
        radioBge.disabled = !bgeAvailable;
        const labelBge = el("labelBge");
        if (labelBge) {
          labelBge.style.opacity = bgeAvailable ? "1" : "0.5";
          labelBge.style.cursor = bgeAvailable ? "pointer" : "not-allowed";
        }
        const bgeDesc = el("bgeDesc");
        if (bgeDesc && !bgeAvailable) {
          bgeDesc.textContent = "Unavailable in current build (Torch/sentence-transformers not present).";
        }
      }
    }
  } catch (err) {
    console.error("Could not fetch embedding settings:", err);
  }
}

function closeSettingsModal() {
  const backdrop = el("settingsModalBackdrop");
  if (backdrop) backdrop.classList.remove("open");
}

async function saveSettingsModal() {
  const radioAzure = el("radioAzure");
  const radioBge = el("radioBge");
  const selectedProvider = radioBge && radioBge.checked ? "bge" : "azure";

  try {
    const res = await apiFetch(API.settingsEmbedding, {
      method: "POST",
      body: JSON.stringify({ provider: selectedProvider }),
    });
    if (res?.success) {
      showToast(res.message || `Settings updated. Provider: ${selectedProvider}.`, "success");
      closeSettingsModal();
      // Reload current session details to refresh 'needs_reindex' flags on active chips/sources
      const sessId = workspaceState.sessionId || localStorage.getItem(SESSION_KEY) || "";
      if (sessId) {
        await restoreSession(sessId);
      }
      renderComposerChips();
      renderSourcesPanel();
      updateSendButton();
    } else {
      showToast(res?.detail || res?.message || "Failed to update settings.", "error");
    }
  } catch (err) {
    showToast("Error saving embedding settings.", "error");
  }
}
