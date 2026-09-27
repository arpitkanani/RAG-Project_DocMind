"use strict";

document.addEventListener("DOMContentLoaded", () => {
  const loginBtns = document.querySelectorAll(".trigger-login-modal");

  loginBtns.forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      window.location.href = "/app";
    });
  });
});

