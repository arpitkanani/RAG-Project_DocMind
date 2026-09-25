"use strict";

document.addEventListener("DOMContentLoaded", () => {
  const modalBackdrop = document.getElementById("authModalBackdrop");
  const modalClose = document.getElementById("authModalClose");
  const loginBtns = document.querySelectorAll(".trigger-login-modal");

  function openLoginModal() {
    if (modalBackdrop) {
      modalBackdrop.classList.add("open");
    }
  }

  function closeLoginModal() {
    if (modalBackdrop) {
      modalBackdrop.classList.remove("open");
    }
    // Remember that user dismissed modal so we don't annoy them on every click
    sessionStorage.setItem("docuvortex.modalDismissed", "true");
  }

  loginBtns.forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      openLoginModal();
    });
  });

  if (modalClose) {
    modalClose.addEventListener("click", closeLoginModal);
  }

  if (modalBackdrop) {
    modalBackdrop.addEventListener("click", (e) => {
      if (e.target === modalBackdrop) {
        closeLoginModal();
      }
    });
  }

  // Check if user is already authenticated
  checkSessionState().then((isAuth) => {
    if (isAuth) {
      // User is logged in — direct them straight to workspace
      loginBtns.forEach((btn) => {
        btn.textContent = "Open Workspace →";
        btn.onclick = () => {
          window.location.href = "/app";
        };
      });
    } else {
      // Auto-show modal on first visit unless dismissed in this tab session
      const alreadyDismissed = sessionStorage.getItem("docuvortex.modalDismissed");
      if (!alreadyDismissed) {
        setTimeout(openLoginModal, 600);
      }
    }
  });
});

async function checkSessionState() {
  try {
    const res = await fetch("/api/auth/session-status");
    if (res.ok) {
      const data = await res.json();
      return Boolean(data.authenticated);
    }
  } catch {}
  return false;
}
