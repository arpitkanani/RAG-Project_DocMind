"use strict";

document.addEventListener("DOMContentLoaded", () => {
  // Modal Elements
  const backdrop = document.getElementById("createKeyModalBackdrop");
  const modal = document.getElementById("createKeyModal");
  const closeBtn = document.getElementById("closeCreateKeyModal");
  const cancelBtn = document.getElementById("cancelCreateKeyBtn");
  const navBtn = document.getElementById("navCreateKeyBtn");
  const heroBtn = document.getElementById("heroCreateKeyBtn");

  // Step 1: Input Screen
  const stepInput = document.getElementById("modalStepInput");
  const usernameInput = document.getElementById("usernameInput");
  const submitBtn = document.getElementById("submitCreateKeyBtn");
  const submitBtnText = document.getElementById("submitBtnText");
  const inputError = document.getElementById("modalInputError");

  // Step 1b: Collision Alert Screen
  const stepExists = document.getElementById("modalStepExists");
  const existingUsernameLabel = document.getElementById("existingUsernameLabel");
  const pickDifferentBtn = document.getElementById("pickDifferentUserBtn");
  const rotateKeyConfirmBtn = document.getElementById("rotateKeyConfirmBtn");

  // Step 2: Success Screen
  const stepSuccess = document.getElementById("modalStepSuccess");
  const generatedKeyText = document.getElementById("generatedKeyText");
  const accountRestoredBadge = document.getElementById("accountRestoredBadge");
  const copyKeyBtn = document.getElementById("copyKeyBtn");
  const copyBtnIcon = document.getElementById("copyBtnIcon");
  const copyBtnLabel = document.getElementById("copyBtnLabel");
  const openWorkspaceBtn = document.getElementById("openWorkspaceWithKeyBtn");

  let pendingUsername = "";
  let pendingLastSessionId = null;
  let currentGeneratedKey = "";
  let copyTimeout = null;

  function openModal() {
    if (!backdrop) return;
    // Reset to step 1
    stepInput.style.display = "block";
    stepExists.style.display = "none";
    stepSuccess.style.display = "none";
    if (inputError) {
      inputError.style.display = "none";
      inputError.textContent = "";
    }
    if (usernameInput) {
      usernameInput.value = "";
    }
    pendingUsername = "";
    currentGeneratedKey = "";

    backdrop.classList.add("open");
    setTimeout(() => {
      if (usernameInput) usernameInput.focus();
    }, 80);
  }

  function closeModal() {
    if (!backdrop) return;
    backdrop.classList.remove("open");
  }

  // Open modal triggers
  if (navBtn) navBtn.addEventListener("click", openModal);
  if (heroBtn) heroBtn.addEventListener("click", openModal);

  // Close modal triggers
  if (closeBtn) closeBtn.addEventListener("click", closeModal);
  if (cancelBtn) cancelBtn.addEventListener("click", closeModal);
  if (backdrop) {
    backdrop.addEventListener("click", (e) => {
      if (e.target === backdrop) closeModal();
    });
  }

  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && backdrop && backdrop.classList.contains("open")) {
      closeModal();
    }
  });

  // Check URL query parameters for direct action (e.g. redirected from /app)
  const urlParams = new URLSearchParams(window.location.search);
  if (urlParams.get("action") === "create_key") {
    openModal();
    if (window.history.replaceState) {
      window.history.replaceState({}, document.title, window.location.pathname);
    }
  }

  // API Call: Create or Rotate Key
  async function requestApiKey(username, forceRotate) {
    const trimmed = (username || "").trim();
    if (!forceRotate && (!trimmed || trimmed.length < 3)) {
      if (inputError) {
        inputError.textContent = !trimmed
          ? "Please enter a username to generate or access your workspace key."
          : "Username must be at least 3 characters long.";
        inputError.style.display = "block";
      }
      if (usernameInput) usernameInput.focus();
      return;
    }

    if (inputError) inputError.style.display = "none";
    if (submitBtn) {
      submitBtn.disabled = true;
      if (submitBtnText) submitBtnText.textContent = "Generating...";
    }
    if (rotateKeyConfirmBtn) {
      rotateKeyConfirmBtn.disabled = true;
      rotateKeyConfirmBtn.textContent = "Rotating API Key...";
    }

    try {
      const res = await fetch("/api/auth/create-key", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: trimmed,
          force_rotate: Boolean(forceRotate),
        }),
      });

      const data = await res.json();
      if (!res.ok || !data.success) {
        throw new Error(data.detail || data.message || "Failed to generate key.");
      }

      // Check if user already exists and confirmation is needed
      if (data.exists) {
        pendingUsername = data.username || username;
        pendingLastSessionId = data.last_session_id || null;
        if (existingUsernameLabel) {
          existingUsernameLabel.textContent = pendingUsername;
        }
        const descEl = document.getElementById("existingUserDesc");
        if (descEl) {
          if (data.session_count && data.session_count > 0) {
            descEl.textContent = `This account has ${data.session_count} existing chat session(s) and saved documents. Rotating your API key will generate a new active key and immediately resume your workspace right where you left off.`;
          } else {
            descEl.textContent = "This user account is already registered. You can rotate your API key to generate a fresh key and access your workspace, or choose a different username.";
          }
        }
        stepInput.style.display = "none";
        stepExists.style.display = "block";
        stepSuccess.style.display = "none";
        return;
      }

      // Key generated successfully
      currentGeneratedKey = data.api_key;
      if (data.last_session_id) {
        pendingLastSessionId = data.last_session_id;
      }
      if (generatedKeyText) {
        generatedKeyText.textContent = currentGeneratedKey;
      }
      if (accountRestoredBadge) {
        accountRestoredBadge.style.display = data.is_existing ? "inline-flex" : "none";
      }

      stepInput.style.display = "none";
      stepExists.style.display = "none";
      stepSuccess.style.display = "block";
    } catch (err) {
      if (inputError) {
        inputError.textContent = err.message || "Error generating API key. Please try again.";
        inputError.style.display = "block";
      }
    } finally {
      if (submitBtn) {
        submitBtn.disabled = false;
        if (submitBtnText) submitBtnText.textContent = "Generate Key";
      }
      if (rotateKeyConfirmBtn) {
        rotateKeyConfirmBtn.disabled = false;
        rotateKeyConfirmBtn.textContent = "Rotate API Key & Resume Workspace";
      }
    }
  }

  // Handle Step 1 Submit
  if (submitBtn) {
    submitBtn.addEventListener("click", () => {
      const val = usernameInput ? usernameInput.value : "";
      requestApiKey(val, false);
    });
  }

  if (usernameInput) {
    usernameInput.addEventListener("input", () => {
      if (inputError) inputError.style.display = "none";
    });
    usernameInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        requestApiKey(usernameInput.value, false);
      }
    });
  }

  // Handle Step 1b: Collision prompt options
  if (pickDifferentBtn) {
    pickDifferentBtn.addEventListener("click", () => {
      stepExists.style.display = "none";
      stepInput.style.display = "block";
      if (usernameInput) {
        usernameInput.focus();
        usernameInput.select();
      }
    });
  }

  if (rotateKeyConfirmBtn) {
    rotateKeyConfirmBtn.addEventListener("click", () => {
      requestApiKey(pendingUsername, true);
    });
  }

  // Copy Key to Clipboard
  if (copyKeyBtn) {
    copyKeyBtn.addEventListener("click", async () => {
      if (!currentGeneratedKey) return;
      let copied = false;
      try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(currentGeneratedKey);
          copied = true;
        }
      } catch (e) {
        console.debug("Clipboard API write failed, trying fallback:", e);
      }

      if (!copied) {
        // Fallback for older browsers or non-HTTPS contexts
        const tempTextArea = document.createElement("textarea");
        tempTextArea.value = currentGeneratedKey;
        tempTextArea.style.position = "fixed";
        tempTextArea.style.opacity = "0";
        document.body.appendChild(tempTextArea);
        tempTextArea.select();
        try {
          copied = document.execCommand("copy");
        } catch (err) {
          console.error("ExecCommand copy failed:", err);
        }
        document.body.removeChild(tempTextArea);
      }

      if (copied) {
        copyKeyBtn.classList.add("copied");
        if (copyBtnIcon) copyBtnIcon.textContent = "✓";
        if (copyBtnLabel) copyBtnLabel.textContent = "Copied!";

        if (copyTimeout) clearTimeout(copyTimeout);
        copyTimeout = setTimeout(() => {
          copyKeyBtn.classList.remove("copied");
          if (copyBtnIcon) copyBtnIcon.textContent = "📋";
          if (copyBtnLabel) copyBtnLabel.textContent = "Copy";
        }, 2200);
      }
    });
  }

  // Launch Workspace with Key
  if (openWorkspaceBtn) {
    openWorkspaceBtn.addEventListener("click", () => {
      if (currentGeneratedKey) {
        try {
          sessionStorage.setItem("staged_api_key", currentGeneratedKey);
          localStorage.setItem("docuvortex.apiKey", currentGeneratedKey);
          if (pendingLastSessionId) {
            localStorage.setItem("docuvortex.activeSessionId", pendingLastSessionId);
          } else {
            localStorage.removeItem("docuvortex.activeSessionId");
          }
        } catch (e) {}
        const sessionParam = pendingLastSessionId
          ? `&session_id=${encodeURIComponent(pendingLastSessionId)}`
          : "";
        window.location.href = `/app?api_key=${encodeURIComponent(currentGeneratedKey)}${sessionParam}`;
      } else {
        openModal();
      }
    });
  }
});
