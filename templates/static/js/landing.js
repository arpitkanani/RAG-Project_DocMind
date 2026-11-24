"use strict";

document.addEventListener("DOMContentLoaded", () => {
  // ══════════════════════ PERFORMANCE TELEMETRY (Component 0) ══════════════════════
  const _perfUrlParams = new URLSearchParams(window.location.search);
  const isPerf = _perfUrlParams.get("perf") === "1" || sessionStorage.getItem("dv_perf") === "1";
  if (_perfUrlParams.get("perf") === "1") {
    sessionStorage.setItem("dv_perf", "1");
  }

  // Intercept fetch calls when perf is enabled to log each call without logging sensitive bodies
  const _origFetch = window.fetch;
  if (isPerf) {
    window.fetch = async function (...args) {
      const t0 = performance.now();
      const url = typeof args[0] === "string" ? args[0] : (args[0]?.url || "unknown");
      const method = (args[1]?.method || "GET").toUpperCase();
      try {
        const res = await _origFetch.apply(this, args);
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

  function measureUiPaint(callback) {
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        callback();
      });
    });
  }

  function logPerfUi(action, tClick, tSent, tHeaders, tParsed, tUi, tRedirect = null) {
    if (!isPerf) return;
    const sentMs = Math.max(0, tSent - tClick).toFixed(0);
    const serverMs = Math.max(0, tHeaders - tSent).toFixed(0);
    const parseMs = Math.max(0, tParsed - tHeaders).toFixed(0);
    const uiMs = Math.max(0, tUi - tParsed).toFixed(0);
    let line = `[perf-ui] ${action} click->sent=${sentMs}ms server=${serverMs}ms parse=${parseMs}ms ui_update=${uiMs}ms`;
    if (tRedirect !== null) {
      const redirMs = Math.max(0, tRedirect - tClick).toFixed(0);
      line += ` click->redirect=${redirMs}ms`;
    } else {
      const totalMs = Math.max(0, tUi - tClick).toFixed(0);
      line += ` total=${totalMs}ms`;
    }
    console.log(line);
  }

  // Modal Elements
  const authBackdrop = document.getElementById("authModalBackdrop");
  const closeAuthModalBtn = document.getElementById("closeAuthModal");

  // Nav buttons
  const navSignInBtn = document.getElementById("navSignInBtn");
  const navAdminRegisterBtn = document.getElementById("navAdminRegisterBtn");
  const navSignOutBtn = document.getElementById("navSignOutBtn");
  const navGuestGroup = document.getElementById("navGuestGroup");
  const navUserGroup = document.getElementById("navUserGroup");
  const navUserBadge = document.getElementById("navUserBadge");
  const navAdminDashboardLink = document.getElementById("navAdminDashboardLink");
  const navExploreRagLink = document.getElementById("navExploreRagLink");

  // Mobile nav buttons & groups
  const mobileNavSignInBtn = document.getElementById("mobileNavSignInBtn");
  const mobileNavAdminRegisterBtn = document.getElementById("mobileNavAdminRegisterBtn");
  const mobileNavSignOutBtn = document.getElementById("mobileNavSignOutBtn");
  const mobileNavGuestGroup = document.getElementById("mobileNavGuestGroup");
  const mobileNavUserGroup = document.getElementById("mobileNavUserGroup");
  const mobileNavUserBadge = document.getElementById("mobileNavUserBadge");
  const mobileNavAdminDashboardLink = document.getElementById("mobileNavAdminDashboardLink");
  const mobileNavExploreRagLink = document.getElementById("mobileNavExploreRagLink");

  // Tab buttons & Panels
  const tabSignInBtn = document.getElementById("tabSignInBtn");
  const tabAdminSignUpBtn = document.getElementById("tabAdminSignUpBtn");

  const authPanelSignIn = document.getElementById("authPanelSignIn");
  const authPanelAdminSignUp = document.getElementById("authPanelAdminSignUp");

  // Sub-toggle for Sign In
  const subToggleUserLogin = document.getElementById("subToggleUserLogin");
  const subToggleAdminLogin = document.getElementById("subToggleAdminLogin");
  const userLoginForm = document.getElementById("userLoginForm");
  const adminLoginForm = document.getElementById("adminLoginForm");
  const adminRegisterForm = document.getElementById("adminRegisterForm");

  // Error elements
  const loginUserError = document.getElementById("loginUserError");
  const loginAdminError = document.getElementById("loginAdminError");
  const regAdminError = document.getElementById("regAdminError");

  // Marketplace Grid
  const botsMarketplaceGrid = document.getElementById("botsMarketplaceGrid");

  let currentUser = null;

  // Debounce guard: collapses rapid-fire calls (cross-tab storage events + auth transitions)
  // into a single fetch, preventing 5-20x redundant /api/business-bot/public/list hits.
  let _marketplaceDebounceTimer = null;
  function debouncedLoadMarketplaceBots(delayMs = 200) {
    clearTimeout(_marketplaceDebounceTimer);
    _marketplaceDebounceTimer = setTimeout(() => loadMarketplaceBots(), delayMs);
  }

  // ══════════════════════ TOAST NOTIFICATIONS ══════════════════════
  function showToast(message, type = "success", durationMs = 3800) {
    let stack = document.getElementById("toastStack");
    if (!stack) {
      stack = document.createElement("div");
      stack.id = "toastStack";
      stack.className = "toast-stack";
      document.body.appendChild(stack);
    }
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    
    let icon = "✓";
    if (type === "error") icon = "✕";
    if (type === "warn") icon = "⚠️";
    if (type === "info") icon = "ℹ️";
    
    toast.innerHTML = `<span style="font-weight: 800; font-size: 1rem;">${icon}</span> <span>${escapeHtml(message)}</span>`;
    stack.appendChild(toast);

    setTimeout(() => {
      toast.classList.add("toast-fade");
    }, durationMs);
    setTimeout(() => {
      toast.remove();
    }, durationMs + 500);
  }

  // ══════════════════════ MODAL NAVIGATION ══════════════════════
  function switchTab(tab) {
    [tabSignInBtn, tabAdminSignUpBtn].forEach((btn) => btn?.classList.remove("active"));
    [authPanelSignIn, authPanelAdminSignUp].forEach((panel) => {
      if (panel) panel.style.display = "none";
    });

    // Clear error messages
    [loginUserError, loginAdminError, regAdminError].forEach((err) => {
      if (err) {
        err.style.display = "none";
        err.textContent = "";
      }
    });

    if (tab === "admin-signup" || tab === "admin_register") {
      tabAdminSignUpBtn?.classList.add("active");
      if (authPanelAdminSignUp) authPanelAdminSignUp.style.display = "block";
    } else {
      tabSignInBtn?.classList.add("active");
      if (authPanelSignIn) authPanelSignIn.style.display = "block";
    }
  }

  function openAuthModal(defaultTab = "signin") {
    if (!authBackdrop) return;
    switchTab(defaultTab);
    authBackdrop.classList.add("open");
  }

  function closeAuthModal() {
    if (!authBackdrop) return;
    authBackdrop.classList.remove("open");
  }

  // Trigger bindings
  window.__openAuthModal = openAuthModal;
  if (navSignInBtn) navSignInBtn.addEventListener("click", () => openAuthModal("signin"));
  if (navAdminRegisterBtn) navAdminRegisterBtn.addEventListener("click", () => openAuthModal("admin-signup"));
  const expCardAdminRegisterBtn = document.getElementById("expCardAdminRegisterBtn");
  if (expCardAdminRegisterBtn) expCardAdminRegisterBtn.addEventListener("click", () => openAuthModal("admin-signup"));
  const bridgeAdminRegisterBtn = document.getElementById("bridgeAdminRegisterBtn");
  if (bridgeAdminRegisterBtn) bridgeAdminRegisterBtn.addEventListener("click", () => openAuthModal("admin-signup"));

  // Helper for pending post-login redirection
  function handlePendingPostLoginRedirect(defaultPath = null) {
    let pending = null;
    try {
      pending = sessionStorage.getItem("pending_redirect");
      if (pending) {
        sessionStorage.removeItem("pending_redirect");
      }
    } catch (e) {}

    const targetUrl = pending || defaultPath;
    if (targetUrl) {
      setTimeout(() => {
        window.location.href = targetUrl;
      }, 250);
      return true;
    }
    return false;
  }

  // Guard all landing page action links for unauthenticated visitors
  const heroExploreRagBtn = document.getElementById("heroExploreRagBtn");
  if (heroExploreRagBtn) {
    heroExploreRagBtn.addEventListener("click", (e) => {
      if (!currentUser) {
        e.preventDefault();
        try {
          sessionStorage.setItem("pending_redirect", "/app");
        } catch (err) {}
        showToast("Please sign in to access your personal RAG workspace.", "info");
        openAuthModal("signin");
      }
    });
  }

  const expCardOpenPersonalBtn = document.getElementById("expCardOpenPersonalBtn");
  if (expCardOpenPersonalBtn) {
    expCardOpenPersonalBtn.addEventListener("click", (e) => {
      if (!currentUser) {
        e.preventDefault();
        try {
          sessionStorage.setItem("pending_redirect", "/app");
        } catch (err) {}
        showToast("Please sign in to access your personal RAG workspace.", "info");
        openAuthModal("signin");
      }
    });
  }

  const heroCreateBotBtn = document.getElementById("heroCreateBotBtn");
  if (heroCreateBotBtn) {
    heroCreateBotBtn.addEventListener("click", (e) => {
      if (!currentUser) {
        e.preventDefault();
        try {
          sessionStorage.setItem("pending_redirect", "/builder");
        } catch (err) {}
        showToast("Please sign in or create a business account to build an assistant.", "info");
        openAuthModal("admin-signup");
        return;
      }

      if (currentUser.role !== "admin") {
        e.preventDefault();
        try {
          sessionStorage.setItem("pending_redirect", "/builder");
        } catch (err) {}
        showToast("Building custom assistants requires an Enterprise Business account. Please register your business account.", "info", 4000);
        openAuthModal("admin-signup");
        return;
      }
    });
  }



  if (closeAuthModalBtn) closeAuthModalBtn.addEventListener("click", closeAuthModal);
  const cancelSignInUserBtn = document.getElementById("cancelSignInUserBtn");
  const cancelSignInAdminBtn = document.getElementById("cancelSignInAdminBtn");
  const cancelAdminRegisterBtn = document.getElementById("cancelAdminRegisterBtn");
  [cancelSignInUserBtn, cancelSignInAdminBtn, cancelAdminRegisterBtn].forEach((btn) => {
    if (btn) btn.addEventListener("click", closeAuthModal);
  });

  const linkToAdminSignUpFromUser = document.getElementById("linkToAdminSignUpFromUser");
  const linkToAdminSignUpFromAdmin = document.getElementById("linkToAdminSignUpFromAdmin");
  [linkToAdminSignUpFromUser, linkToAdminSignUpFromAdmin].forEach((btn) => {
    if (btn) btn.addEventListener("click", () => switchTab("admin-signup"));
  });

  const linkToSignInFromAdminSignUp = document.getElementById("linkToSignInFromAdminSignUp");
  if (linkToSignInFromAdminSignUp) {
    linkToSignInFromAdminSignUp.addEventListener("click", () => {
      switchTab("signin");
      if (subToggleAdminLogin) subToggleAdminLogin.click();
    });
  }

  if (authBackdrop) {
    authBackdrop.addEventListener("click", (e) => {
      if (e.target === authBackdrop) closeAuthModal();
    });
  }

  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && authBackdrop && authBackdrop.classList.contains("open")) {
      closeAuthModal();
    }
  });

  if (tabSignInBtn) tabSignInBtn.addEventListener("click", () => switchTab("signin"));
  if (tabAdminSignUpBtn) tabAdminSignUpBtn.addEventListener("click", () => switchTab("admin-signup"));

  // Mobile navigation drawer toggle
  const mobileNavToggle = document.getElementById("mobileNavToggle");
  const navMenu = document.getElementById("navMenu") || document.querySelector(".nav-menu");

  function closeMobileNav() {
    if (navMenu) navMenu.classList.remove("open");
    if (mobileNavToggle) {
      mobileNavToggle.classList.remove("open");
      mobileNavToggle.setAttribute("aria-expanded", "false");
    }
  }

  if (mobileNavToggle && navMenu) {
    mobileNavToggle.addEventListener("click", () => {
      const isOpen = navMenu.classList.toggle("open");
      mobileNavToggle.classList.toggle("open", isOpen);
      mobileNavToggle.setAttribute("aria-expanded", String(isOpen));
    });

    // Auto-close on link click
    navMenu.querySelectorAll(".nav-menu-link").forEach((link) => {
      link.addEventListener("click", () => closeMobileNav());
    });

    // Auto-close on click outside
    document.addEventListener("click", (e) => {
      if (
        navMenu.classList.contains("open") &&
        !navMenu.contains(e.target) &&
        !mobileNavToggle.contains(e.target)
      ) {
        closeMobileNav();
      }
    });

    // Auto-close on Escape key
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && navMenu.classList.contains("open")) {
        closeMobileNav();
      }
    });
  }

  // Mobile Auth Actions wiring
  if (mobileNavSignInBtn) {
    mobileNavSignInBtn.addEventListener("click", () => {
      closeMobileNav();
      openAuthModal("signin");
    });
  }
  if (mobileNavAdminRegisterBtn) {
    mobileNavAdminRegisterBtn.addEventListener("click", () => {
      closeMobileNav();
      openAuthModal("admin-signup");
    });
  }
  if (mobileNavSignOutBtn) {
    mobileNavSignOutBtn.addEventListener("click", () => {
      closeMobileNav();
      signOut();
    });
  }

  // ══════════════════════ NAVBAR SCROLL-SPY ACTIVE INDICATOR ══════════════════════
  const navSections = [
    document.getElementById("overview"),
    document.getElementById("botShowcaseSection"),
    document.getElementById("howItWorksSection"),
    document.getElementById("evaluationSection"),
    document.getElementById("techSection"),
  ].filter(Boolean);

  const navLinksMap = new Map();
  document.querySelectorAll(".nav-menu-link").forEach((link) => {
    const hash = link.getAttribute("href");
    if (hash && hash.startsWith("#")) {
      navLinksMap.set(hash.slice(1), link);
    }
    link.addEventListener("click", () => {
      document.querySelectorAll(".nav-menu-link").forEach((l) => l.classList.remove("active"));
      link.classList.add("active");
    });
  });

  // ══════════════════════ FLOATING BACK-TO-TOP BUTTON ══════════════════════
  const backToTopBtn = document.getElementById("backToTopBtn");
  if (backToTopBtn) {
    backToTopBtn.addEventListener("click", () => {
      const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      window.scrollTo({
        top: 0,
        behavior: prefersReducedMotion ? "auto" : "smooth"
      });
    });
  }

  let _scrollSpyScheduled = false;
  function updateActiveNavLinkOnScroll() {
    if (_scrollSpyScheduled) return;
    _scrollSpyScheduled = true;
    requestAnimationFrame(() => {
      _scrollSpyScheduled = false;
      const scrollPos = window.scrollY + 140;
      let currentId = navSections[0]?.id || "overview";

      for (const sec of navSections) {
        const top = sec.offsetTop;
        const height = sec.offsetHeight;
        if (scrollPos >= top && scrollPos < top + height) {
          currentId = sec.id;
          break;
        } else if (scrollPos >= top) {
          currentId = sec.id;
        }
      }

      navLinksMap.forEach((link, id) => {
        if (id === currentId) {
          link.classList.add("active");
        } else {
          link.classList.remove("active");
        }
      });

      // Update floating back-to-top visibility
      if (backToTopBtn) {
        if (window.scrollY > 300) {
          backToTopBtn.classList.add("visible");
        } else {
          backToTopBtn.classList.remove("visible");
        }
      }
    });
  }

  window.addEventListener("scroll", updateActiveNavLinkOnScroll, { passive: true });
  updateActiveNavLinkOnScroll();

  // Sub-toggle for Sign In (Customer vs Admin)
  if (subToggleUserLogin && subToggleAdminLogin) {
    subToggleUserLogin.addEventListener("click", () => {
      subToggleUserLogin.classList.add("active");
      subToggleAdminLogin.classList.remove("active");
      userLoginForm.style.display = "block";
      adminLoginForm.style.display = "none";
    });

    subToggleAdminLogin.addEventListener("click", () => {
      subToggleAdminLogin.classList.add("active");
      subToggleUserLogin.classList.remove("active");
      adminLoginForm.style.display = "block";
      userLoginForm.style.display = "none";
    });
  }

  // Check URL params for direct actions (deferred so checkAuthAndResume runs first)
  const urlParams = new URLSearchParams(window.location.search);
  const actionParam = urlParams.get("action");
  const roleParam = urlParams.get("role");
  if (actionParam) {
    setTimeout(() => {
      if (!currentUser) {
        if (actionParam === "login") {
          openAuthModal("signin");
        } else if (actionParam === "register_admin") {
          openAuthModal("admin-signup");
        } else if (actionParam === "create_key") {
          openAuthModal("user-signup");
        }
      } else {
        // If logged in as normal user and redirected from an admin-guarded route
        if (currentUser.role !== "admin" && (actionParam === "register_admin" || roleParam === "admin")) {
          showToast("Building custom assistants requires an Enterprise Business account. Please register your business account.", "info", 4000);
          openAuthModal("admin-signup");
        }
        if (window.location.search) {
          window.history.replaceState({}, document.title, window.location.pathname);
        }
      }
    }, 120);
  }

  // ══════════════════════ FORM SUBMISSIONS ══════════════════════

  // 1. Customer / Standard User Login (Unified auto-provisioning & returning user)
  if (userLoginForm) {
    userLoginForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const tClick = performance.now();
      let tSent = tClick, tHeaders = tClick, tParsed = tClick;
      loginUserError.style.display = "none";
      const submitBtn = userLoginForm.querySelector("button[type='submit']");
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.textContent = "Signing In...";
      }

      const username = document.getElementById("loginUsername").value.trim();
      const password = document.getElementById("loginUserPassword").value;

      try {
        tSent = performance.now();
        const res = await fetch("/api/auth/login-customer", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username, password }),
        });
        tHeaders = performance.now();
        const data = await res.json();
        tParsed = performance.now();
        if (!res.ok) {
          throw new Error(data.detail || data.message || "Failed to sign in.");
        }

        sessionStorage.removeItem("dv_logged_out");
        if (data.api_key) {
          localStorage.setItem("docuvortex.apiKey", data.api_key);
        }
        if (data.user) {
          localStorage.setItem("docuvortex.user", JSON.stringify(data.user));
          applyNavbarAuthState(data.user);
        }
        closeAuthModal();
        if (window.location.search) {
          window.history.replaceState({}, document.title, window.location.pathname);
        }
        if (data.is_new) {
          showToast(`Welcome! Your account has been created.`, "success");
        } else {
          showToast(`Welcome back, ${data.user?.username || username}! Previous chats restored.`, "success");
        }
        if (!handlePendingPostLoginRedirect(null)) {
          await checkAuthAndResume();
        }
        measureUiPaint(() => {
          logPerfUi("user_login", tClick, tSent, tHeaders, tParsed, performance.now());
        });
      } catch (err) {
        loginUserError.textContent = err.message;
        loginUserError.style.display = "block";
        showToast(err.message, "error");
        measureUiPaint(() => {
          logPerfUi("user_login [error]", tClick, tSent, tHeaders, tParsed, performance.now());
        });
      } finally {
        if (submitBtn) {
          submitBtn.disabled = false;
          submitBtn.textContent = "Sign In / Continue";
        }
      }
    });
  }

  // 2. Admin Login
  if (adminLoginForm) {
    adminLoginForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const tClick = performance.now();
      let tSent = tClick, tHeaders = tClick, tParsed = tClick;
      loginAdminError.style.display = "none";
      const submitBtn = adminLoginForm.querySelector("button[type='submit']");
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.textContent = "Signing In...";
      }

      const email = document.getElementById("loginAdminEmail").value.trim();
      const password = document.getElementById("loginAdminPassword").value;

      try {
        tSent = performance.now();
        const res = await fetch("/api/auth/login-admin", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password }),
        });
        tHeaders = performance.now();
        const data = await res.json();
        tParsed = performance.now();
        if (!res.ok) {
          throw new Error(data.detail || data.message || "Invalid admin credentials.");
        }

        sessionStorage.removeItem("dv_logged_out");
        if (data.api_key) {
          localStorage.setItem("docuvortex.apiKey", data.api_key);
        }
        if (data.user) {
          localStorage.setItem("docuvortex.user", JSON.stringify(data.user));
          applyNavbarAuthState(data.user);
        }
        closeAuthModal();
        if (window.location.search) {
          window.history.replaceState({}, document.title, window.location.pathname);
        }

        if (handlePendingPostLoginRedirect(null)) {
          return;
        }

        // Check if admin has already created business bots
        let botCount = 0;
        try {
          const botsRes = await fetch("/api/business-bot/admin/bots", { credentials: "include" });
          if (botsRes.ok) {
            const botsData = await botsRes.json();
            botCount = botsData?.bots ? botsData.bots.length : 0;
          }
        } catch (e) {}

        if (botCount === 0) {
          sessionStorage.setItem("dv_perf_click_ts", Date.now().toString());
          sessionStorage.setItem("dv_perf_action", "admin_login_redirect_builder");
          showToast("Welcome Admin! Opening Bot Builder to create your first business bot...", "info", 2000);
          measureUiPaint(() => {
            logPerfUi("admin_login", tClick, tSent, tHeaders, tParsed, performance.now(), performance.now());
          });
          setTimeout(() => {
            window.location.href = "/builder";
          }, 150);
        } else {
          showToast(`Welcome back, Admin ${data.user?.username || email.split("@")[0]}!`, "success");
          await checkAuthAndResume();
          measureUiPaint(() => {
            logPerfUi("admin_login", tClick, tSent, tHeaders, tParsed, performance.now());
          });
        }
      } catch (err) {
        loginAdminError.textContent = err.message;
        loginAdminError.style.display = "block";
        showToast(err.message, "error");
        measureUiPaint(() => {
          logPerfUi("admin_login [error]", tClick, tSent, tHeaders, tParsed, performance.now());
        });
      } finally {
        if (submitBtn) {
          submitBtn.disabled = false;
          submitBtn.textContent = "Sign In as Admin";
        }
      }
    });
  }

  // 3. Admin Register (With unique email/username check & 409 handling)
  if (adminRegisterForm) {
    adminRegisterForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const tClick = performance.now();
      let tSent = tClick, tHeaders = tClick, tParsed = tClick;
      regAdminError.style.display = "none";
      const submitBtn = adminRegisterForm.querySelector("button[type='submit']");
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.textContent = "Registering...";
      }

      const username = document.getElementById("regAdminUsername").value.trim();
      const email = document.getElementById("regAdminEmail").value.trim();
      const password = document.getElementById("regAdminPassword").value;

      try {
        tSent = performance.now();
        const res = await fetch("/api/auth/register-admin", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username, email, password }),
        });
        tHeaders = performance.now();
        const data = await res.json();
        tParsed = performance.now();
        if (!res.ok) {
          const errMsg = data.detail || data.message || "Failed to register business admin.";
          if (res.status === 409 || errMsg.toLowerCase().includes("already exists")) {
            showToast("An account with that username or email already exists. Please use Admin Login to access your account.", "warn", 5000);
            openAuthModal("signin");
            if (subToggleAdminLogin) subToggleAdminLogin.click();
            const adminEmailInput = document.getElementById("loginAdminEmail");
            if (adminEmailInput) adminEmailInput.value = email;
            const adminPassInput = document.getElementById("loginAdminPassword");
            if (adminPassInput) adminPassInput.focus();
            return;
          }
          throw new Error(errMsg);
        }

        sessionStorage.removeItem("dv_logged_out");
        if (data.api_key) {
          localStorage.setItem("docuvortex.apiKey", data.api_key);
        }
        if (data.user) {
          localStorage.setItem("docuvortex.user", JSON.stringify(data.user));
          applyNavbarAuthState(data.user);
        }
        closeAuthModal();
        if (window.location.search) {
          window.history.replaceState({}, document.title, window.location.pathname);
        }

        sessionStorage.setItem("dv_perf_click_ts", Date.now().toString());
        sessionStorage.setItem("dv_perf_action", "admin_register_redirect_builder");
        showToast("Admin registered! Opening Bot Builder Wizard...", "success", 1800);
        measureUiPaint(() => {
          logPerfUi("admin_register", tClick, tSent, tHeaders, tParsed, performance.now(), performance.now());
        });
        if (!handlePendingPostLoginRedirect("/builder")) {
          setTimeout(() => {
            window.location.href = "/builder";
          }, 150);
        }
      } catch (err) {
        regAdminError.textContent = err.message;
        regAdminError.style.display = "block";
        showToast(err.message, "error");
        measureUiPaint(() => {
          logPerfUi("admin_register [error]", tClick, tSent, tHeaders, tParsed, performance.now());
        });
      } finally {
        if (submitBtn) {
          submitBtn.disabled = false;
          submitBtn.textContent = "Register Business Admin";
        }
      }
    });
  }

  // Helper: Centralized navbar auth rendering
  function applyNavbarAuthState(user) {
    if (user && (user.username || user.email)) {
      currentUser = user;
      try {
        localStorage.setItem("docuvortex.user", JSON.stringify(user));
      } catch (e) {}

      if (navGuestGroup) {
        navGuestGroup.classList.remove("auth-visible");
        navGuestGroup.classList.add("auth-hidden");
        navGuestGroup.style.removeProperty("display");
        navGuestGroup.style.setProperty("display", "none", "important");
      }
      if (navUserGroup) {
        navUserGroup.classList.remove("auth-hidden");
        navUserGroup.classList.add("auth-visible");
        navUserGroup.style.removeProperty("display");
        if (window.innerWidth > 960) {
          navUserGroup.style.setProperty("display", "flex");
        } else {
          navUserGroup.style.setProperty("display", "none", "important");
        }
      }

      if (navUserBadge) {
        const roleLabel = user.role === "admin" ? "Admin" : "User";
        const displayName = user.username || (user.email ? user.email.split("@")[0] : "User");
        navUserBadge.textContent = `${roleLabel}: ${displayName}`;
      }

      const expCreateBtn = document.getElementById("expCardCreateAssistantBtn");
      const expRegBtn = document.getElementById("expCardAdminRegisterBtn");
      const bridgeCreateBtn = document.getElementById("bridgeCreateAssistantBtn");
      const bridgeRegBtn = document.getElementById("bridgeAdminRegisterBtn");

      if (user.role === "admin") {
        if (navAdminDashboardLink) {
          navAdminDashboardLink.style.removeProperty("display");
          navAdminDashboardLink.style.setProperty("display", "inline-flex", "important");
          navAdminDashboardLink.className = "nav-link-primary";
        }
        if (navExploreRagLink) {
          navExploreRagLink.style.removeProperty("display");
          navExploreRagLink.style.setProperty("display", "none", "important");
        }
        if (expCreateBtn) {
          expCreateBtn.style.removeProperty("display");
          expCreateBtn.style.setProperty("display", "inline-flex", "important");
        }
        if (expRegBtn) {
          expRegBtn.style.removeProperty("display");
          expRegBtn.style.setProperty("display", "none", "important");
        }
        if (bridgeCreateBtn) {
          bridgeCreateBtn.style.removeProperty("display");
          bridgeCreateBtn.style.setProperty("display", "inline-flex", "important");
        }
        if (bridgeRegBtn) {
          bridgeRegBtn.style.removeProperty("display");
          bridgeRegBtn.style.setProperty("display", "none", "important");
        }
      } else {
        if (navAdminDashboardLink) {
          navAdminDashboardLink.style.removeProperty("display");
          navAdminDashboardLink.style.setProperty("display", "none", "important");
        }
        if (navExploreRagLink) {
          navExploreRagLink.style.removeProperty("display");
          navExploreRagLink.style.setProperty("display", "inline-flex", "important");
          navExploreRagLink.className = "nav-link-primary";
        }
        if (expCreateBtn) {
          expCreateBtn.style.removeProperty("display");
          expCreateBtn.style.setProperty("display", "none", "important");
        }
        if (expRegBtn) {
          expRegBtn.style.removeProperty("display");
          expRegBtn.style.setProperty("display", "inline-flex", "important");
        }
        if (bridgeCreateBtn) {
          bridgeCreateBtn.style.removeProperty("display");
          bridgeCreateBtn.style.setProperty("display", "none", "important");
        }
        if (bridgeRegBtn) {
          bridgeRegBtn.style.removeProperty("display");
          bridgeRegBtn.style.setProperty("display", "inline-flex", "important");
        }
      }

      // Sync mobile navigation auth elements
      if (mobileNavGuestGroup) mobileNavGuestGroup.style.display = "none";
      if (mobileNavUserGroup) mobileNavUserGroup.style.display = "flex";
      if (mobileNavUserBadge) {
        const roleLabel = user.role === "admin" ? "Admin" : "User";
        const displayName = user.username || (user.email ? user.email.split("@")[0] : "User");
        mobileNavUserBadge.textContent = `${roleLabel}: ${displayName}`;
      }
      if (user.role === "admin") {
        if (mobileNavAdminDashboardLink) mobileNavAdminDashboardLink.style.display = "inline-flex";
        if (mobileNavExploreRagLink) mobileNavExploreRagLink.style.display = "none";
      } else {
        if (mobileNavAdminDashboardLink) mobileNavAdminDashboardLink.style.display = "none";
        if (mobileNavExploreRagLink) mobileNavExploreRagLink.style.display = "inline-flex";
      }
    } else {
      currentUser = null;

      // Experience Card B & How It Works: Guest mode shows "Create Business Account →"
      const expCreateBtn = document.getElementById("expCardCreateAssistantBtn");
      const expRegBtn = document.getElementById("expCardAdminRegisterBtn");
      if (expCreateBtn) {
        expCreateBtn.style.removeProperty("display");
        expCreateBtn.style.setProperty("display", "none", "important");
      }
      if (expRegBtn) {
        expRegBtn.style.removeProperty("display");
        expRegBtn.style.setProperty("display", "inline-flex", "important");
      }
      const bridgeCreateBtn = document.getElementById("bridgeCreateAssistantBtn");
      const bridgeRegBtn = document.getElementById("bridgeAdminRegisterBtn");
      if (bridgeCreateBtn) {
        bridgeCreateBtn.style.removeProperty("display");
        bridgeCreateBtn.style.setProperty("display", "none", "important");
      }
      if (bridgeRegBtn) {
        bridgeRegBtn.style.removeProperty("display");
        bridgeRegBtn.style.setProperty("display", "inline-flex", "important");
      }

      // 1. Remove early static style injected by document.write if present
      const earlyStyle = document.getElementById("earlyNavAuthStyle");
      if (earlyStyle) {
        earlyStyle.remove();
      }

      // 2. Clear user badge text immediately
      if (navUserBadge) {
        navUserBadge.textContent = "";
      }

      // 3. Immediately hide user group and show guest group with both CSS class and !important override
      if (navUserGroup) {
        navUserGroup.classList.remove("auth-visible");
        navUserGroup.classList.add("auth-hidden");
        navUserGroup.style.removeProperty("display");
        navUserGroup.style.setProperty("display", "none", "important");
      }
      if (navGuestGroup) {
        navGuestGroup.classList.remove("auth-hidden");
        navGuestGroup.classList.add("auth-visible");
        navGuestGroup.style.removeProperty("display");
        if (window.innerWidth > 960) {
          navGuestGroup.style.setProperty("display", "flex");
        } else {
          navGuestGroup.style.setProperty("display", "none", "important");
        }
      }

      // 4. Hide auth-dependent navigation links
      if (navAdminDashboardLink) {
        navAdminDashboardLink.style.removeProperty("display");
        navAdminDashboardLink.style.setProperty("display", "none", "important");
      }
      if (navExploreRagLink) {
        navExploreRagLink.style.removeProperty("display");
        navExploreRagLink.style.setProperty("display", "none", "important");
      }

      // Sync mobile navigation auth elements for guest mode
      if (mobileNavUserBadge) mobileNavUserBadge.textContent = "";
      if (mobileNavUserGroup) mobileNavUserGroup.style.display = "none";
      if (mobileNavGuestGroup) mobileNavGuestGroup.style.display = "flex";
      if (mobileNavAdminDashboardLink) mobileNavAdminDashboardLink.style.display = "none";
      if (mobileNavExploreRagLink) mobileNavExploreRagLink.style.display = "inline-flex";

      // 5. Instantly remove all bot edit buttons
      document.querySelectorAll(".bot-edit-btn").forEach((el) => el.remove());
    }
  }

  // ══════════════════════ SIGN-OUT (centralised) ══════════════════════
  // Single source of truth for all sign-out logic.
  // Order: (1) set explicit logged-out session flag
  //        (2) clear module & storage state immediately
  //        (3) reset navbar and DOM synchronously
  //        (4) await backend cookie deletion
  //        (5) re-render bot grid in guest mode
  //        (6) defensive post-render sweep removing any lingering edit buttons
  //        (7) show toast
  async function signOut() {
    const tClick = performance.now();

    // 1. Mark explicit sign-out in session to prevent background auto-resume from restoring user
    sessionStorage.setItem("dv_logged_out", "1");
    currentUser = null;

    // 2. Wipe all client-side auth storage immediately
    try {
      localStorage.removeItem("docuvortex.apiKey");
      localStorage.removeItem("docuvortex.user");
      localStorage.removeItem("docuvortex.activeSessionId");
      sessionStorage.removeItem("docuvortex.apiKey");
      sessionStorage.removeItem("docuvortex.user");
      sessionStorage.removeItem("docuvortex.activeSessionId");
      sessionStorage.removeItem("pending_redirect");
      sessionStorage.removeItem("docuvortex.public_bots");
    } catch (e) {}

    // 3. Immediate synchronous UI purge: guest navbar, remove all .bot-edit-btn, hide admin links
    applyNavbarAuthState(null);

    // 4. Await backend logout so the cookie is definitively invalidated on the server BEFORE any other calls
    try {
      await fetch("/api/auth/logout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
      });
    } catch (e) {
      console.warn("Backend logout request note:", e);
    }

    // 5. Re-render marketplace bots cleanly in guest mode (currentUser is null)
    await loadMarketplaceBots();

    // 6. Defensive post-render sweep: ensure zero .bot-edit-btn elements remain anywhere in DOM
    document.querySelectorAll(".bot-edit-btn").forEach((el) => el.remove());

    // 7. Toast confirmation
    showToast("Signed out successfully.", "info");

    if (isPerf) {
      console.log(`[perf-ui] signout total=${(performance.now() - tClick).toFixed(0)}ms`);
    }
  }

  // Sign-out button binding
  if (navSignOutBtn) {
    navSignOutBtn.addEventListener("click", signOut);
  }

  // ══════════════════════ SESSION AUTO-RESUME ══════════════════════
  async function checkAuthAndResume() {
    // If user explicitly signed out, do not auto-resume
    if (sessionStorage.getItem("dv_logged_out") === "1") {
      applyNavbarAuthState(null);
      await loadMarketplaceBots();
      return;
    }

    try {
      const existingKey = (localStorage.getItem("docuvortex.apiKey") || "").trim();
      const headers = {};
      if (existingKey && existingKey.startsWith("dk_live_")) {
        headers["X-API-Key"] = existingKey;
      }
      const res = await fetch("/api/auth/me", {
        headers: headers,
        credentials: "include",
      });
      if (res.ok) {
        const data = await res.json();
        if (data.authenticated && data.user) {
          if (data.api_key) {
            try {
              localStorage.setItem("docuvortex.apiKey", data.api_key);
            } catch (e) {}
          }
          applyNavbarAuthState(data.user);

          // Scrub any action= query parameter from URL so browser Back never re-triggers modal
          if (window.location.search && (window.location.search.includes("action=") || window.location.search.includes("login"))) {
            closeAuthModal();
            window.history.replaceState({}, document.title, window.location.pathname);
          }
        } else {
          try {
            localStorage.removeItem("docuvortex.user");
          } catch (e) {}
          applyNavbarAuthState(null);
        }
      } else {
        try {
          localStorage.removeItem("docuvortex.user");
        } catch (e) {}
        applyNavbarAuthState(null);
      }
    } catch (e) {
      console.debug("Auth resume check note:", e);
    }

    // Always fetch bot showcase
    await debouncedLoadMarketplaceBots();
  }

  // Cross-tab synchronization: keep navbar in sync if another tab logs in or logs out
  window.addEventListener("storage", (event) => {
    if (event.key === "docuvortex.user" || event.key === "docuvortex.apiKey") {
      if (sessionStorage.getItem("dv_logged_out") === "1") {
        return;
      }
      const userRaw = localStorage.getItem("docuvortex.user");
      if (!userRaw) {
        applyNavbarAuthState(null);
        debouncedLoadMarketplaceBots();
      } else {
        try {
          const u = JSON.parse(userRaw);
          applyNavbarAuthState(u);
          debouncedLoadMarketplaceBots();
        } catch (e) {
          applyNavbarAuthState(null);
          debouncedLoadMarketplaceBots();
        }
      }
    }
  });

  // ══════════════════════ BOT MARKETPLACE SHOWCASE ══════════════════════
  function renderMarketplaceBots(bots) {
    if (!botsMarketplaceGrid) return;

    if (!bots || bots.length === 0) {
      botsMarketplaceGrid.innerHTML = `
        <div style="grid-column: 1 / -1; text-align: center; color: var(--text-muted); padding: 40px; background: var(--surface-white); border-radius: 16px; border: 1px dashed var(--border-warm);">
          <div style="font-size: 2rem; margin-bottom: 8px;">🤖</div>
          <h4 style="margin: 0 0 6px; color: var(--forest-charcoal); font-family: var(--font-heading); font-size: 1.2rem;">No Deployed Assistants Yet</h4>
          <p style="margin: 0 0 16px; font-size: 0.88rem;">Register as a Business Admin to build and deploy your first custom AI assistant.</p>
          <a href="/builder" class="nav-link-primary" style="display: inline-block;">Build Your Business Bot →</a>
        </div>`;
      return;
    }

    let userOwnsAnyBot = false;
    let html = "";
    bots.forEach((bot) => {
      const botId = String(bot.bot_id || bot.id);
      const initial = (bot.persona_name || bot.business_name || "B").slice(0, 1).toUpperCase();
      // Strict ownership: edit button appears ONLY for the creator of this specific bot
      const isAdminOwner = Boolean(currentUser && currentUser.id && String(currentUser.id) === String(bot.admin_id));
      if (isAdminOwner) userOwnsAnyBot = true;

      html += `
        <div class="bot-card" data-bot-id="${escapeHtml(botId)}">
          <div>
            <div class="bot-card-top">
              <div class="bot-avatar">${initial}</div>
              <div class="bot-meta">
                <h3 class="bot-persona">${escapeHtml(bot.persona_name || "Business Assistant")}</h3>
                <div class="bot-biz-name">${escapeHtml(bot.business_name || "Company Bot")}</div>
              </div>
            </div>
            <p class="bot-summary">${escapeHtml(bot.summary || bot.business_summary || "Ready to assist customers with product details, pricing, and scheduling.")}</p>
          </div>
          <div class="bot-card-footer">
            <a href="/bot/${escapeHtml(botId)}" class="bot-chat-btn" data-bot-id="${escapeHtml(botId)}" target="_blank">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
              </svg>
              Start Conversation →
            </a>
            ${
              isAdminOwner
                ? `<a href="/admin-dashboard?edit=${escapeHtml(botId)}" class="bot-edit-btn" data-bot-id="${escapeHtml(botId)}" title="Edit your business bot">
                     <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                       <path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/>
                     </svg>
                     Edit
                   </a>`
                : ""
            }
          </div>
        </div>`;
    });

    botsMarketplaceGrid.innerHTML = html;

    // If user owns any bots or is role admin, enable the Admin Dashboard link
    const showAdminActions = Boolean(currentUser && (currentUser.role === "admin" || userOwnsAnyBot));
    if (navAdminDashboardLink) {
      navAdminDashboardLink.style.removeProperty("display");
      navAdminDashboardLink.style.setProperty("display", showAdminActions ? "inline-flex" : "none", "important");
    }
  }

  async function loadMarketplaceBots() {
    if (!botsMarketplaceGrid) return;

    try {
      const res = await fetch("/api/business-bot/public/list");
      if (!res.ok) throw new Error("Could not load bots");
      const bots = await res.json();
      try {
        sessionStorage.setItem("docuvortex.public_bots", JSON.stringify(bots));
      } catch (e) {}
      renderMarketplaceBots(bots);
    } catch (e) {
      console.error("Marketplace fetch error:", e);
      // Only display error placeholder if grid has no cached cards
      if (!botsMarketplaceGrid.hasChildNodes() || botsMarketplaceGrid.children.length === 0) {
        botsMarketplaceGrid.innerHTML = `
          <div style="grid-column: 1 / -1; text-align: center; color: var(--muted, #5e657c); padding: 30px;">
            Failed to load business bots. Please refresh the page.
          </div>`;
      }
    }
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

  // Delegated click handler on Marketplace Grid for Start Conversation & Edit guards
  if (botsMarketplaceGrid) {
    botsMarketplaceGrid.addEventListener("click", (e) => {
      // 1. Guard Start Conversation for unauthenticated users
      const chatBtn = e.target.closest(".bot-chat-btn");
      if (chatBtn) {
        if (!currentUser) {
          e.preventDefault();
          const botId = chatBtn.getAttribute("data-bot-id") || "";
          if (botId) {
            try {
              sessionStorage.setItem("pending_redirect", `/bot/${botId}`);
            } catch (err) {}
          }
          showToast("Please sign in or register to start a conversation with this assistant.", "info");
          openAuthModal("signin");
          return;
        }
      }

      // 2. Defensive Guard on Edit Button — fires if a stale edit button is
      //    clicked after sign-out (before the grid has re-rendered without it).
      const editBtn = e.target.closest(".bot-edit-btn");
      if (editBtn) {
        if (!currentUser || currentUser.role !== "admin") {
          e.preventDefault();
          editBtn.remove(); // Remove the stale button immediately
          showToast("Session ended. Please sign in as Admin to edit assistants.", "warn");
          // Open modal on the Admin Sign-In sub-tab directly
          openAuthModal("signin");
          if (subToggleAdminLogin) subToggleAdminLogin.click();
          return;
        }
      }
    });
  }

  // Initialize: Pre-hydrate public bots marketplace from session cache for instant 0ms perception
  try {
    const cachedPublicBotsRaw = sessionStorage.getItem("docuvortex.public_bots");
    if (cachedPublicBotsRaw) {
      const cachedPublicBots = JSON.parse(cachedPublicBotsRaw);
      if (Array.isArray(cachedPublicBots) && cachedPublicBots.length > 0) {
        renderMarketplaceBots(cachedPublicBots);
      }
    }
  } catch (e) {}

  checkAuthAndResume();
});
