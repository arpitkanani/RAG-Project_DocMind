"use strict";

document.addEventListener("DOMContentLoaded", () => {
  initCaptchaAuthFlow();
});

function initCaptchaAuthFlow() {
  const usernameInput = document.getElementById("authUsername");
  const emailInput = document.getElementById("authEmail");
  const passwordInput = document.getElementById("authPassword");
  const captchaQuestionEl = document.getElementById("captchaQuestion");
  const captchaTokenInput = document.getElementById("captchaToken");
  const captchaAnswerInput = document.getElementById("captchaAnswer");
  const btnRefreshCaptcha = document.getElementById("btnRefreshCaptcha");
  const btnSubmit = document.getElementById("btnSubmitAuth");
  const errorBox = document.getElementById("authError");

  function showError(msg) {
    if (!errorBox) return;
    errorBox.textContent = msg;
    errorBox.style.display = "block";
  }

  function hideError() {
    if (!errorBox) return;
    errorBox.textContent = "";
    errorBox.style.display = "none";
  }

  // Generate an instant local fallback challenge so the user NEVER sees "..."
  function setLocalFallbackCaptcha() {
    const num1 = Math.floor(Math.random() * 8) + 3; // 3..10
    const num2 = Math.floor(Math.random() * 8) + 2; // 2..9
    const ans = num1 + num2;
    if (captchaQuestionEl) {
      captchaQuestionEl.textContent = `${num1} + ${num2}`;
    }
    if (captchaTokenInput) {
      captchaTokenInput.value = `${ans}:fallback`;
    }
  }

  // Fetch & display fresh signed math captcha from the server
  async function loadCaptcha() {
    try {
      const res = await fetch("/api/auth/captcha");
      if (!res.ok) throw new Error("Server captcha unavailable");
      const data = await res.json();
      if (data && data.question && data.token) {
        if (captchaQuestionEl) captchaQuestionEl.textContent = data.question;
        if (captchaTokenInput) captchaTokenInput.value = data.token;
      }
      if (captchaAnswerInput) captchaAnswerInput.value = "";
    } catch (e) {
      console.warn("Using offline math challenge fallback:", e);
      setLocalFallbackCaptcha();
    }
  }

  // 2. Submit Auth Form
  async function handleAuthSubmit() {
    hideError();

    const username = usernameInput ? usernameInput.value.trim() : "";
    const email = emailInput ? emailInput.value.trim() : "";
    const password = passwordInput ? passwordInput.value.trim() : "";
    const captcha_answer = captchaAnswerInput ? captchaAnswerInput.value.trim() : "";
    const captcha_token = captchaTokenInput ? captchaTokenInput.value.trim() : "";

    if (!username) return showError("Please enter a username.");
    if (!email || !email.includes("@")) return showError("Please enter a valid email address.");
    if (!password) return showError("Please enter your password.");
    if (!captcha_answer) return showError("Please solve the math captcha.");

    if (btnSubmit) {
      btnSubmit.disabled = true;
      btnSubmit.textContent = "Signing In...";
    }

    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username,
          email,
          password,
          captcha_answer,
          captcha_token,
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || data.message || "Authentication failed. Please check your details.");
      }

      // Successful verification -> redirect to workspace
      window.location.href = data.redirect || "/app";
    } catch (err) {
      showError(err.message || "Failed to authenticate.");
      // Refresh captcha on failure
      loadCaptcha();
    } finally {
      if (btnSubmit) {
        btnSubmit.disabled = false;
        btnSubmit.innerHTML = "Sign In to Workspace &rarr;";
      }
    }
  }

  if (btnRefreshCaptcha) {
    btnRefreshCaptcha.addEventListener("click", () => {
      btnRefreshCaptcha.style.transform = "rotate(180deg)";
      setTimeout(() => {
        btnRefreshCaptcha.style.transform = "";
      }, 250);
      loadCaptcha();
    });
  }

  if (btnSubmit) {
    btnSubmit.addEventListener("click", handleAuthSubmit);
  }

  [usernameInput, emailInput, passwordInput, captchaAnswerInput].forEach((inp) => {
    if (inp) {
      inp.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          handleAuthSubmit();
        }
      });
    }
  });

  // Password visibility eye icon toggle
  const btnTogglePassword = document.getElementById("btnTogglePassword");
  if (btnTogglePassword && passwordInput) {
    const eyeOpen = btnTogglePassword.querySelector(".eye-open");
    const eyeClosed = btnTogglePassword.querySelector(".eye-closed");

    btnTogglePassword.addEventListener("click", () => {
      const isPwd = passwordInput.getAttribute("type") === "password";
      if (isPwd) {
        passwordInput.setAttribute("type", "text");
        if (eyeOpen) eyeOpen.style.display = "none";
        if (eyeClosed) eyeClosed.style.display = "block";
      } else {
        passwordInput.setAttribute("type", "password");
        if (eyeOpen) eyeOpen.style.display = "block";
        if (eyeClosed) eyeClosed.style.display = "none";
      }
    });
  }

  // Forgot Password Modal Flow
  const btnForgotPwd = document.getElementById("btnForgotPwd");
  const forgotBackdrop = document.getElementById("forgotModalBackdrop");
  const btnCloseForgot = document.getElementById("btnCloseForgot");
  const forgotEmailInput = document.getElementById("forgotEmail");
  const btnSubmitForgot = document.getElementById("btnSubmitForgot");
  const forgotErrorBox = document.getElementById("forgotError");
  const forgotFormBody = document.getElementById("forgotFormBody");
  const forgotResultBox = document.getElementById("forgotResultBox");
  const recoveredPasswordText = document.getElementById("recoveredPasswordText");
  const btnCopyRecoveredPwd = document.getElementById("btnCopyRecoveredPwd");
  const btnUseRecoveredPwd = document.getElementById("btnUseRecoveredPwd");

  function openForgotModal() {
    if (!forgotBackdrop) return;
    if (forgotErrorBox) {
      forgotErrorBox.style.display = "none";
      forgotErrorBox.textContent = "";
    }
    if (forgotFormBody) forgotFormBody.style.display = "block";
    if (forgotResultBox) forgotResultBox.style.display = "none";

    // Auto-populate from main email field if present
    if (forgotEmailInput && emailInput && emailInput.value.trim()) {
      forgotEmailInput.value = emailInput.value.trim();
    }
    forgotBackdrop.classList.add("open");
    setTimeout(() => {
      if (forgotEmailInput) forgotEmailInput.focus();
    }, 100);
  }

  function closeForgotModal() {
    if (forgotBackdrop) {
      forgotBackdrop.classList.remove("open");
    }
  }

  if (btnForgotPwd) {
    btnForgotPwd.addEventListener("click", openForgotModal);
  }
  if (btnCloseForgot) {
    btnCloseForgot.addEventListener("click", closeForgotModal);
  }
  if (forgotBackdrop) {
    forgotBackdrop.addEventListener("click", (e) => {
      if (e.target === forgotBackdrop) closeForgotModal();
    });
  }

  async function handleForgotSubmit() {
    const email = forgotEmailInput ? forgotEmailInput.value.trim() : "";
    if (!email || !email.includes("@")) {
      if (forgotErrorBox) {
        forgotErrorBox.textContent = "Please enter a valid email address.";
        forgotErrorBox.style.display = "block";
      }
      return;
    }

    if (forgotErrorBox) forgotErrorBox.style.display = "none";
    if (btnSubmitForgot) {
      btnSubmitForgot.disabled = true;
      btnSubmitForgot.textContent = "Retrieving...";
    }

    try {
      const res = await fetch("/api/auth/forgot-password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || data.message || "Failed to retrieve password.");
      }

      if (recoveredPasswordText) {
        recoveredPasswordText.textContent = data.password;
      }
      if (forgotFormBody) forgotFormBody.style.display = "none";
      if (forgotResultBox) forgotResultBox.style.display = "block";

      // Also set the main email input
      if (emailInput) {
        emailInput.value = email;
      }
    } catch (err) {
      if (forgotErrorBox) {
        forgotErrorBox.textContent = err.message || "Could not retrieve password.";
        forgotErrorBox.style.display = "block";
      }
    } finally {
      if (btnSubmitForgot) {
        btnSubmitForgot.disabled = false;
        btnSubmitForgot.textContent = "Show My Password \u2192";
      }
    }
  }

  if (btnSubmitForgot) {
    btnSubmitForgot.addEventListener("click", handleForgotSubmit);
  }
  if (forgotEmailInput) {
    forgotEmailInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        handleForgotSubmit();
      }
    });
  }

  if (btnCopyRecoveredPwd && recoveredPasswordText) {
    btnCopyRecoveredPwd.addEventListener("click", () => {
      navigator.clipboard.writeText(recoveredPasswordText.textContent.trim()).then(() => {
        btnCopyRecoveredPwd.textContent = "Copied!";
        setTimeout(() => {
          btnCopyRecoveredPwd.textContent = "Copy";
        }, 1500);
      });
    });
  }

  if (btnUseRecoveredPwd && recoveredPasswordText && passwordInput) {
    btnUseRecoveredPwd.addEventListener("click", () => {
      passwordInput.value = recoveredPasswordText.textContent.trim();
      closeForgotModal();
      passwordInput.focus();
    });
  }

  // Filter input to digits only for manual numerical input
  if (captchaAnswerInput) {
    captchaAnswerInput.addEventListener("input", (e) => {
      e.target.value = e.target.value.replace(/[^0-9]/g, "");
    });
  }

  // Render initial captcha immediately (0ms latency), then sync server HMAC token
  if (!captchaQuestionEl.textContent || captchaQuestionEl.textContent.trim() === "...") {
    setLocalFallbackCaptcha();
  }
  loadCaptcha();
}
