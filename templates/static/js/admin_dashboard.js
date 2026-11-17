"use strict";

document.addEventListener("DOMContentLoaded", () => {
  const tabMyBots = document.getElementById("tabMyBots");
  const tabLeads = document.getElementById("tabLeads");
  const panelMyBots = document.getElementById("panelMyBots");
  const panelLeads = document.getElementById("panelLeads");
  const myBotsGrid = document.getElementById("myBotsGrid");
  const leadsTableBody = document.getElementById("leadsTableBody");
  const adminSignOutBtn = document.getElementById("adminSignOutBtn");

  // Edit Modal Elements
  const editModalBackdrop = document.getElementById("editBotModalBackdrop");
  const editBotForm = document.getElementById("editBotForm");
  const closeEditModalBtn = document.getElementById("closeEditModalBtn");
  const cancelEditModalBtn = document.getElementById("cancelEditModalBtn");
  const editBotId = document.getElementById("editBotId");
  const editPersonaName = document.getElementById("editPersonaName");
  const editBizName = document.getElementById("editBizName");
  const editWebsiteUrl = document.getElementById("editWebsiteUrl");
  const editSummary = document.getElementById("editSummary");
  const editSystemPrompt = document.getElementById("editSystemPrompt");
  const editBotError = document.getElementById("editBotError");

  // Delete Modal Elements
  const deleteModalBackdrop = document.getElementById("deleteBotModalBackdrop");
  const closeDeleteModalBtn = document.getElementById("closeDeleteModalBtn");
  const cancelDeleteModalBtn = document.getElementById("cancelDeleteModalBtn");
  const confirmDeleteBotBtn = document.getElementById("confirmDeleteBotBtn");
  const deleteTargetPersonaName = document.getElementById("deleteTargetPersonaName");
  const deleteTargetBizName = document.getElementById("deleteTargetBizName");
  const deleteBotError = document.getElementById("deleteBotError");
  let pendingDeleteBotId = null;

  let adminUser = null;
  let adminBots = [];

  // Tab Switching
  if (tabMyBots && tabLeads) {
    tabMyBots.addEventListener("click", () => {
      tabMyBots.classList.add("active");
      tabLeads.classList.remove("active");
      panelMyBots.style.display = "block";
      panelLeads.style.display = "none";
    });

    tabLeads.addEventListener("click", () => {
      tabLeads.classList.add("active");
      tabMyBots.classList.remove("active");
      panelLeads.style.display = "block";
      panelMyBots.style.display = "none";
      loadAllAdminLeads();
    });
  }

  // Authentication check & parallel dashboard startup
  async function initDashboard() {
    try {
      // Pre-hydrate admin user and assistants from session cache for instant 0ms perception
      try {
        const cachedRaw = localStorage.getItem("docuvortex.user");
        if (cachedRaw) {
          const cachedUser = JSON.parse(cachedRaw);
          if (cachedUser && cachedUser.role === "admin") {
            adminUser = cachedUser;
          }
        }
        const cachedBotsRaw = sessionStorage.getItem("docuvortex.admin_bots");
        if (cachedBotsRaw) {
          const cachedBots = JSON.parse(cachedBotsRaw);
          if (Array.isArray(cachedBots) && cachedBots.length > 0) {
            renderAdminBotsList(cachedBots);
          }
        }
      } catch (e) {}

      // Parallelize auth verification and bot loading
      const [authRes, botsRes] = await Promise.all([
        fetch("/api/auth/me", { credentials: "include" }),
        fetch("/api/business-bot/admin/bots", { credentials: "include" })
      ]);

      if (!authRes.ok) throw new Error("Unauthorized");
      const authData = await authRes.json();
      if (!authData.authenticated || !authData.user || authData.user.role !== "admin") {
        localStorage.removeItem("docuvortex.user");
        sessionStorage.removeItem("docuvortex.admin_bots");
        window.location.replace("/?action=login");
        return;
      }
      adminUser = authData.user;
      localStorage.setItem("docuvortex.user", JSON.stringify(authData.user));

      if (botsRes.ok) {
        const botsData = await botsRes.json();
        const botsList = Array.isArray(botsData) ? botsData : (botsData.bots || []);
        sessionStorage.setItem("docuvortex.admin_bots", JSON.stringify(botsList));
        renderAdminBotsList(botsList);
      } else {
        await loadAdminBots();
      }

      // Check if URL has ?edit=bot_id
      const urlParams = new URLSearchParams(window.location.search);
      const editId = urlParams.get("edit");
      if (editId) {
        openEditModalForBot(editId);
      }
    } catch (e) {
      localStorage.removeItem("docuvortex.user");
      sessionStorage.removeItem("docuvortex.admin_bots");
      window.location.replace("/?action=login");
    }
  }

  // Sign out
  if (adminSignOutBtn) {
    adminSignOutBtn.addEventListener("click", async () => {
      try {
        await fetch("/api/auth/logout", { method: "POST", credentials: "include" });
      } catch (e) {}
      try {
        sessionStorage.setItem("dv_logged_out", "1");
        localStorage.removeItem("docuvortex.apiKey");
        localStorage.removeItem("docuvortex.user");
        sessionStorage.removeItem("docuvortex.apiKey");
        sessionStorage.removeItem("docuvortex.user");
        sessionStorage.removeItem("docuvortex.admin_bots");
      } catch (e) {}
      window.location.replace("/");
    });
  }

  function renderAdminBotsList(bots) {
    adminBots = bots || [];

    if (!adminBots || adminBots.length === 0) {
      myBotsGrid.innerHTML = `
        <div style="grid-column: 1 / -1; text-align: center; color: var(--forest-charcoal-muted, #677369); padding: 50px; background: #FFFFFF; border-radius: 16px; border: 1px dashed var(--border-warm, #E5E1D8); box-shadow: 0 4px 18px rgba(41,53,45,0.04);">
          <div style="font-size: 2.2rem; margin-bottom: 10px;">🤖</div>
          <h3 style="color: var(--forest-charcoal, #29352D); font-family: 'Lora', serif; font-size: 1.3rem; margin: 0 0 8px;">No Assistants Created Yet</h3>
          <p style="margin: 0 0 20px; font-size: 0.92rem; max-width: 480px; margin-left: auto; margin-right: auto; line-height: 1.5;">Create your first custom AI business assistant using our live Master Architect Wizard to answer inquiries and capture qualified leads.</p>
          <a href="/builder" class="btn-terracotta" style="display: inline-flex;">Launch Assistant Builder &rarr;</a>
        </div>`;
      return;
    }

    let html = "";
    adminBots.forEach((bot) => {
      const botId = String(bot.bot_id || bot.id);
      const persona = bot.persona_name || "Assistant";
      const bizName = bot.business_name || "Business";
      const summary = bot.business_summary || bot.summary || "Custom AI Assistant";
      const initial = persona.slice(0, 1).toUpperCase();
      const escapedPersona = escapeHtml(persona).replace(/'/g, "\\'");
      const escapedBiz = escapeHtml(bizName).replace(/'/g, "\\'");

      html += `
        <div class="dash-card" id="botCard_${botId}">
          <div>
            <div class="dash-card-header">
              <div class="dash-card-header-left">
                <div class="dash-bot-avatar">${initial}</div>
                <div>
                  <h3 class="dash-card-title">${escapeHtml(persona)}</h3>
                  <span class="dash-card-biz">${escapeHtml(bizName)}</span>
                </div>
              </div>
              <button type="button" class="btn-card-delete" onclick="confirmDeleteBot('${botId}', '${escapedPersona}', '${escapedBiz}')" title="Delete Assistant" aria-label="Delete Assistant ${escapeHtml(persona)}">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                  <polyline points="3 6 5 6 21 6"></polyline>
                  <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                  <line x1="10" y1="11" x2="10" y2="17"></line>
                  <line x1="14" y1="11" x2="14" y2="17"></line>
                </svg>
              </button>
            </div>
            <p class="dash-card-summary">${escapeHtml(summary)}</p>
          </div>
          <div class="dash-card-actions">
            <a href="/bot/${botId}" target="_blank" class="btn-terracotta" style="font-size: 0.82rem; padding: 7px 14px; flex: 1; text-align: center;">
              Chat Live &rarr;
            </a>
            <button type="button" class="btn-forest-ghost" onclick="openEditModalForBot('${botId}')" style="font-size: 0.82rem; padding: 7px 14px; cursor: pointer;">
              Edit
            </button>
            <button type="button" class="btn-lead-pill" onclick="viewBotLeads('${botId}')" style="font-size: 0.82rem; padding: 7px 14px; cursor: pointer;">
              Leads
            </button>
          </div>
        </div>`;
    });

    myBotsGrid.innerHTML = html;
  }

  // Load Admin's Bots
  async function loadAdminBots() {
    try {
      const res = await fetch("/api/business-bot/admin/bots", { credentials: "include" });
      if (!res.ok) throw new Error("Failed to load your assistants");
      const data = await res.json();
      const botsList = Array.isArray(data) ? data : (data.bots || []);
      sessionStorage.setItem("docuvortex.admin_bots", JSON.stringify(botsList));
      renderAdminBotsList(botsList);
    } catch (e) {
      myBotsGrid.innerHTML = `<div style="grid-column: 1 / -1; text-align: center; color: var(--status-red, #C6534E); padding: 36px;">Error loading assistants: ${escapeHtml(e.message)}</div>`;
    }
  }

  // Edit Modal Handlers - Normalized String comparison
  window.openEditModalForBot = function (botId) {
    const normalizedId = String(botId);
    const bot = adminBots.find((b) => String(b.bot_id || b.id) === normalizedId);
    if (!bot) return;

    editBotId.value = normalizedId;
    editPersonaName.value = bot.persona_name || "";
    editBizName.value = bot.business_name || "";
    editWebsiteUrl.value = bot.website_url || "";
    editSummary.value = bot.business_summary || bot.summary || "";
    editSystemPrompt.value = bot.dynamic_prompt || bot.system_prompt || "";
    editBotError.style.display = "none";

    editModalBackdrop.classList.add("open");
  };

  function closeEditModal() {
    editModalBackdrop.classList.remove("open");
  }

  if (closeEditModalBtn) closeEditModalBtn.addEventListener("click", closeEditModal);
  if (cancelEditModalBtn) cancelEditModalBtn.addEventListener("click", closeEditModal);

  // Delete Modal Handlers
  window.confirmDeleteBot = function (botId, personaName, bizName) {
    pendingDeleteBotId = String(botId);
    if (deleteTargetPersonaName) deleteTargetPersonaName.textContent = personaName || "Assistant";
    if (deleteTargetBizName) deleteTargetBizName.textContent = bizName || "Business";
    if (deleteBotError) deleteBotError.style.display = "none";
    if (confirmDeleteBotBtn) {
      confirmDeleteBotBtn.disabled = false;
      confirmDeleteBotBtn.textContent = "Delete Assistant";
    }
    if (deleteModalBackdrop) deleteModalBackdrop.classList.add("open");
  };

  function closeDeleteModal() {
    if (deleteModalBackdrop) deleteModalBackdrop.classList.remove("open");
    pendingDeleteBotId = null;
  }

  if (closeDeleteModalBtn) closeDeleteModalBtn.addEventListener("click", closeDeleteModal);
  if (cancelDeleteModalBtn) cancelDeleteModalBtn.addEventListener("click", closeDeleteModal);

  if (confirmDeleteBotBtn) {
    confirmDeleteBotBtn.addEventListener("click", async () => {
      if (!pendingDeleteBotId) return;
      const deletedId = pendingDeleteBotId;
      const cardEl = document.getElementById("botCard_" + deletedId);

      // 1. Instantly close modal
      closeDeleteModal();

      // 2. Instantly remove card optimistically with smooth exit transition
      if (cardEl) {
        cardEl.style.transition = "opacity 0.2s ease, transform 0.2s ease";
        cardEl.style.opacity = "0";
        cardEl.style.transform = "scale(0.95)";
        setTimeout(() => {
          cardEl.remove();
          adminBots = adminBots.filter((b) => String(b.bot_id || b.id) !== deletedId);
          sessionStorage.setItem("docuvortex.admin_bots", JSON.stringify(adminBots));
          if (adminBots.length === 0) {
            renderAdminBotsList([]);
          }
        }, 210);
      } else {
        adminBots = adminBots.filter((b) => String(b.bot_id || b.id) !== deletedId);
        sessionStorage.setItem("docuvortex.admin_bots", JSON.stringify(adminBots));
        if (adminBots.length === 0) {
          renderAdminBotsList([]);
        }
      }

      // 3. Instantly notify user
      showToast("Assistant deleted successfully.");

      // 4. Background server deletion
      try {
        const res = await fetch(`/api/business-bot/admin/bots/${deletedId}`, {
          method: "DELETE",
          credentials: "include",
        });
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(data.detail || "Server failed to delete assistant");
        }
      } catch (err) {
        // Rollback on network/server error
        showToast("Delete failed: " + err.message, true);
        await loadAdminBots();
      }
    });
  }

  // Escape key dismisses modals
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (editModalBackdrop && editModalBackdrop.classList.contains("open")) closeEditModal();
      if (deleteModalBackdrop && deleteModalBackdrop.classList.contains("open")) closeDeleteModal();
    }
  });

  if (editBotForm) {
    editBotForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      editBotError.style.display = "none";
      const submitBtn = editBotForm.querySelector("button[type='submit']");
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.textContent = "Saving...";
      }

      const botId = editBotId.value;
      const payload = {
        persona_name: editPersonaName.value.trim(),
        business_name: editBizName.value.trim(),
        website_url: editWebsiteUrl.value.trim() || null,
        business_summary: editSummary.value.trim() || null,
        dynamic_prompt: editSystemPrompt.value.trim(),
      };

      try {
        const res = await fetch(`/api/business-bot/admin/bots/${botId}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
          credentials: "include",
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Failed to update assistant configuration");

        closeEditModal();
        await loadAdminBots();
      } catch (err) {
        editBotError.textContent = err.message;
        editBotError.style.display = "block";
      } finally {
        if (submitBtn) {
          submitBtn.disabled = false;
          submitBtn.textContent = "Save Changes";
        }
      }
    });
  }

  // Leads View
  window.viewBotLeads = async function (botId) {
    tabLeads.click();
    await loadLeadsForBot(botId);
  };

  async function loadAllAdminLeads() {
    if (!adminBots.length) {
      await loadAdminBots();
    }
    if (!adminBots.length) {
      leadsTableBody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--forest-charcoal-muted, #677369); padding: 36px;">No assistants deployed yet.</td></tr>`;
      return;
    }

    leadsTableBody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--forest-charcoal-muted, #677369); padding: 36px;">Loading customer inquiries...</td></tr>`;

    let allLeads = [];
    for (const b of adminBots) {
      const bId = String(b.bot_id || b.id);
      try {
        const res = await fetch(`/api/business-bot/admin/leads/${bId}`, { credentials: "include" });
        if (res.ok) {
          const data = await res.json();
          const leads = Array.isArray(data) ? data : (data.leads || []);
          leads.forEach((l) => {
            l.bot_persona = b.persona_name || b.business_name;
            allLeads.push(l);
          });
        }
      } catch (e) {}
    }

    renderLeadsRows(allLeads);
  }

  async function loadLeadsForBot(botId) {
    const normalizedId = String(botId);
    leadsTableBody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--forest-charcoal-muted, #677369); padding: 36px;">Loading inquiries for selected assistant...</td></tr>`;
    try {
      const res = await fetch(`/api/business-bot/admin/leads/${normalizedId}`, { credentials: "include" });
      if (!res.ok) throw new Error("Could not fetch leads");
      const data = await res.json();
      const leads = Array.isArray(data) ? data : (data.leads || []);
      const bot = adminBots.find((b) => String(b.bot_id || b.id) === normalizedId);
      leads.forEach((l) => {
        l.bot_persona = bot ? bot.persona_name : "Assistant";
      });
      renderLeadsRows(leads);
    } catch (e) {
      leadsTableBody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--status-red, #C6534E); padding: 36px;">Failed to load inquiries: ${escapeHtml(e.message)}</td></tr>`;
    }
  }

  function formatIndianDateTime(dateStr) {
    if (!dateStr) return "Recent";
    const d = new Date(dateStr);
    if (isNaN(d.getTime())) return String(dateStr);
    const day = String(d.getDate()).padStart(2, "0");
    const month = String(d.getMonth() + 1).padStart(2, "0");
    const year = d.getFullYear();
    let hours = d.getHours();
    const minutes = String(d.getMinutes()).padStart(2, "0");
    const ampm = hours >= 12 ? "PM" : "AM";
    hours = hours % 12;
    hours = hours ? hours : 12;
    const formattedHours = String(hours).padStart(2, "0");
    return `${day}/${month}/${year}, ${formattedHours}:${minutes} ${ampm}`;
  }

  function renderLeadsRows(leads) {
    if (!leads || leads.length === 0) {
      leadsTableBody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--forest-charcoal-muted, #677369); padding: 36px;">No customer leads or inquiries captured yet.</td></tr>`;
      return;
    }

    leadsTableBody.innerHTML = leads
      .map((l) => {
        const dt = formatIndianDateTime(l.created_at);
        let contactHtml = "";
        const email = l.customer_email ? String(l.customer_email).trim() : "";
        const phone = l.customer_phone ? String(l.customer_phone).trim() : "";
        if (email || phone) {
          const parts = [];
          if (email) {
            parts.push(`<a href="mailto:${escapeHtml(email)}" style="color: var(--terracotta, #BC704F); text-decoration: underline; font-weight: 500;">${escapeHtml(email)}</a>`);
          }
          if (phone) {
            parts.push(`<a href="tel:${escapeHtml(phone)}" style="color: var(--forest-charcoal, #29352D); text-decoration: none; font-weight: 500;">📞 ${escapeHtml(phone)}</a>`);
          }
          contactHtml = parts.join("<br/>");
        } else if (l.contact_info && l.contact_info !== "Not provided") {
          contactHtml = escapeHtml(l.contact_info);
        } else {
          contactHtml = `<span style="color: var(--forest-charcoal-muted, #677369); font-style: italic;">Not provided</span>`;
        }

        return `
        <tr>
          <td style="white-space: nowrap; font-size: 0.82rem; color: var(--forest-charcoal-muted, #677369);">${escapeHtml(dt)}</td>
          <td><strong style="color: var(--terracotta, #BC704F);">${escapeHtml(l.bot_persona || "Assistant")}</strong></td>
          <td style="font-weight: 600;">${escapeHtml(l.customer_name || l.customer_username || "Customer")}</td>
          <td>${contactHtml}</td>
          <td>${escapeHtml(l.inquiry_details || l.message_excerpt || "Booking inquiry")}</td>
        </tr>`;
      })
      .join("");
  }

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function showToast(message, isError = false) {
    let toast = document.getElementById("adminToast");
    if (!toast) {
      toast = document.createElement("div");
      toast.id = "adminToast";
      toast.style.position = "fixed";
      toast.style.bottom = "24px";
      toast.style.right = "24px";
      toast.style.padding = "12px 20px";
      toast.style.borderRadius = "10px";
      toast.style.color = "#FFFFFF";
      toast.style.fontWeight = "600";
      toast.style.fontSize = "0.88rem";
      toast.style.boxShadow = "0 6px 20px rgba(41, 53, 45, 0.2)";
      toast.style.zIndex = "9999";
      toast.style.transition = "opacity 0.3s ease, transform 0.3s ease";
      document.body.appendChild(toast);
    }
    toast.style.background = isError ? "var(--status-red, #C6534E)" : "var(--forest-charcoal, #29352D)";
    toast.textContent = message;
    toast.style.opacity = "1";
    toast.style.transform = "translateY(0)";
    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(8px)";
    }, 3500);
  }

  initDashboard();
});
