"use strict";

document.addEventListener("DOMContentLoaded", () => {
  const headerAvatar = document.getElementById("headerAvatar");
  const headerPersona = document.getElementById("headerPersona");
  const headerBizLink = document.getElementById("headerBizLink");
  const botGreetingAvatar = document.getElementById("botGreetingAvatar");
  const botGreetingBubble = document.getElementById("botGreetingBubble");
  const chatMessages = document.getElementById("chatMessages");
  const customerQueryInput = document.getElementById("customerQueryInput");
  const btnSendCustomerQuery = document.getElementById("btnSendCustomerQuery");

  // Extract bot_id from /bot/{bot_id}
  const pathParts = window.location.pathname.split("/").filter(Boolean);
  const botId = pathParts.length > 1 ? pathParts[1] : "";

  let chatWs = null;
  let currentBotRow = null;

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
    toast.innerHTML = `<span style="font-weight: 800;">${icon}</span> <span>${escapeHtml(message)}</span>`;
    stack.appendChild(toast);
    setTimeout(() => toast.classList.add("toast-fade"), durationMs);
    setTimeout(() => toast.remove(), durationMs + 500);
  }

  function dismissLoader() {
    const loader = document.getElementById("chatLoaderOverlay");
    if (loader) {
      loader.classList.add("fade-out");
      setTimeout(() => loader.remove(), 250);
    }
  }

  async function initChat() {
    if (!botId) {
      dismissLoader();
      const initialGreetingRow = document.getElementById("initialGreetingRow");
      if (initialGreetingRow) initialGreetingRow.style.display = "flex";
      if (botGreetingBubble) botGreetingBubble.textContent = "Error: Invalid bot link or bot not found.";
      return;
    }

    // Connect WebSocket immediately in parallel with metadata fetching
    connectWebSocket();

    // Safety fallback: if WebSocket response takes over 2.5s, dismiss loader and show greeting
    setTimeout(() => {
      const loader = document.getElementById("chatLoaderOverlay");
      if (loader) {
        dismissLoader();
        const initialGreetingRow = document.getElementById("initialGreetingRow");
        if (initialGreetingRow && !chatMessages.querySelector(".chat-row:not(#initialGreetingRow)")) {
          initialGreetingRow.style.display = "flex";
        }
      }
    }, 2500);

    try {
      const res = await fetch(`/api/business-bot/public/get/${botId}`);
      if (!res.ok) throw new Error("Bot not found");
      const bot = await res.json();

      const initial = (bot.persona_name || bot.business_name || "B").slice(0, 1).toUpperCase();
      if (headerAvatar) headerAvatar.textContent = initial;
      if (botGreetingAvatar) botGreetingAvatar.textContent = initial;

      document.title = `${bot.persona_name || "Assistant"} – ${bot.business_name || "DocuVortex"}`;

      if (headerPersona) {
        headerPersona.textContent = bot.persona_name || "Assistant";
      }

      const bizName = bot.business_name || "Official Company Bot";
      if (headerBizLink) {
        if (bot.website_url) {
          const cleanUrl = bot.website_url.startsWith("http") ? bot.website_url : `https://${bot.website_url}`;
          headerBizLink.href = cleanUrl;
          headerBizLink.title = `Visit ${cleanUrl}`;
          headerBizLink.innerHTML = `<span>${escapeHtml(bizName)}</span> <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="opacity: 0.75;"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6M15 3h6v6M10 14L21 3"/></svg>`;
          headerBizLink.classList.add("has-website");
        } else {
          headerBizLink.removeAttribute("href");
          headerBizLink.textContent = bizName;
          headerBizLink.classList.remove("has-website");
        }
      }

      if (botGreetingBubble && !botGreetingBubble.textContent.trim()) {
        botGreetingBubble.textContent = `Hello! I am ${bot.persona_name || "your assistant"} from ${bot.business_name || "our company"}. How can I assist you today? Feel free to ask about our offerings, pricing, or appointment booking.`;
      }
    } catch (e) {
      console.warn("Could not load bot metadata:", e);
      if (botGreetingBubble && !botGreetingBubble.textContent.trim()) {
        botGreetingBubble.textContent = "Hello! Welcome to our customer support. How can I help you today?";
      }
    }
  }

  let userIsScrollingUp = false;

  // Track if user manually scrolls away from bottom
  chatMessages.addEventListener("scroll", () => {
    const distanceFromBottom = chatMessages.scrollHeight - chatMessages.scrollTop - chatMessages.clientHeight;
    // Pause auto-scroll only if user scrolled up more than 80px from bottom
    userIsScrollingUp = distanceFromBottom > 80;
  }, { passive: true });

  function scrollToBottom(force = false, smooth = false) {
    if (force) {
      userIsScrollingUp = false;
    }
    if (!userIsScrollingUp || force) {
      if (smooth) {
        chatMessages.scrollTo({
          top: chatMessages.scrollHeight,
          behavior: "smooth",
        });
      } else {
        chatMessages.scrollTop = chatMessages.scrollHeight;
      }
    }
  }

  function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/business-bot/chat/${botId}`;

    chatWs = new WebSocket(wsUrl);

    chatWs.onopen = () => {
      console.log("Chat WS connected.");
    };

    chatWs.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === "resumed_history") {
          dismissLoader();
          // Remove initial greeting row from DOM permanently on resumed session
          const initialRow = document.getElementById("initialGreetingRow");
          if (initialRow) initialRow.remove();

          chatMessages.innerHTML = "";

          if (data.messages && data.messages.length > 0) {
            data.messages.forEach((msg) => {
              const row = document.createElement("div");
              const isUser = msg.role === "human";
              row.className = `chat-row ${isUser ? "user" : "bot"}`;
              const initial = isUser ? "You" : (headerAvatar?.textContent || "B").trim();
              row.innerHTML = `
                <div class="row-avatar">${escapeHtml(initial)}</div>
                <div class="chat-bubble">${formatMarkdown(msg.content)}</div>
              `;
              chatMessages.appendChild(row);
            });
            scrollToBottom(true);
          }
        } else if (data.type === "greeting") {
          dismissLoader();
          const initialGreetingRow = document.getElementById("initialGreetingRow");
          if (initialGreetingRow) {
            initialGreetingRow.style.display = "flex";
          }
          if (botGreetingBubble && data.content) {
            botGreetingBubble.innerHTML = formatMarkdown(data.content);
          }
          scrollToBottom(true);
          if (data.session_expired) {
            showToast("Previous conversation expired after 72 hours of inactivity. Starting a new session.", "warn", 5000);
          }
        } else if (data.type === "tool_status") {
          // Intentionally keep the three-dots thinking indicator active; do not render any tool status badge
        } else if (data.type === "token") {
          removeThinkingIndicator();
          appendBotToken(data.token);
        } else if (data.type === "lead_alert") {
          if (data.has_contact) {
            showLeadCapturedNotice(data);
            showToast("Inquiry recorded! Our team will be in touch shortly.", "success", 4000);
          }
        } else if (data.type === "done") {
          removeThinkingIndicator();
          if (data.latency) {
            console.info("[Bot Latency]", data.latency);
            if (window.location.search.includes("debug=1") && currentBotRow) {
              const latPill = document.createElement("div");
              latPill.className = "latency-debug-pill";
              latPill.style.cssText = "font-size: 0.72rem; color: #8F8C84; margin-top: 4px; margin-left: 48px; opacity: 0.85;";
              latPill.innerHTML = `⚡ ${data.latency.total_ms}ms (TTFT: ${data.latency.ttft_ms}ms · ${data.latency.tokens_per_sec} tok/s · RAG: ${data.latency.rag_ms}ms)`;
              currentBotRow.parentNode.insertBefore(latPill, currentBotRow.nextSibling);
            }
          }
          finalizeCurrentBotMessage();
        } else if (data.type === "error") {
          dismissLoader();
          removeThinkingIndicator();
          // Never display error messages inside the chat window
          if (currentBotRow && (!currentBotRawText || !currentBotRawText.trim())) {
            currentBotRow.remove();
          }
          finalizeCurrentBotMessage();
          showToast(data.message || "An issue occurred during chat session.", "error", 4000);
        }
      } catch (err) {
        console.error("WS parse error:", err);
      }
    };

    chatWs.onerror = (err) => {
      console.error("Chat WS error:", err);
      removeThinkingIndicator();
      finalizeCurrentBotMessage();
      showToast("WebSocket communication error. Please try again.", "error", 3500);
    };

    chatWs.onclose = () => {
      console.debug("Chat WS closed. Reconnecting in 3s...");
      removeThinkingIndicator();
      finalizeCurrentBotMessage();
      showToast("Connection lost. Reconnecting to assistant...", "warn", 3500);
      setTimeout(connectWebSocket, 3000);
    };
  }

  function showThinkingIndicator() {
    removeThinkingIndicator();
    const row = document.createElement("div");
    row.className = "chat-row bot";
    row.id = "thinkingIndicatorRow";
    const initial = (headerAvatar?.textContent || "B").trim();
    row.innerHTML = `
      <div class="row-avatar">${escapeHtml(initial)}</div>
      <div class="reasoning-bubble" title="Reasoning...">
        <div class="reasoning-dots">
          <span class="dot"></span>
          <span class="dot"></span>
          <span class="dot"></span>
        </div>
      </div>
    `;
    chatMessages.appendChild(row);
    scrollToBottom(true, true);
  }

  function removeThinkingIndicator() {
    const el = document.getElementById("thinkingIndicatorRow");
    if (el) el.remove();
  }

  function formatMarkdown(text) {
    if (!text) return "";
    // Normalize any raw HTML break variants into genuine newlines before escaping
    let s = String(text).replace(/<br\s*\/?>/gi, "\n");
    s = escapeHtml(s);

    // Defensive: replace any escaped break tags
    s = s.replace(/&lt;br\s*\/?&gt;/gi, "<br/>");

    // Headings
    s = s.replace(/^### (.*$)/gim, '<h4 style="margin: 12px 0 4px; color: #29352D; font-size: 0.98rem; font-weight: 700; font-family: \'Lora\', serif;">$1</h4>');
    s = s.replace(/^## (.*$)/gim, '<h3 style="margin: 14px 0 6px; color: #29352D; font-size: 1.08rem; font-weight: 800; font-family: \'Lora\', serif;">$1</h3>');

    // Bold & italic
    s = s.replace(/\*\*(.*?)\*\*/g, '<strong style="color: #29352D; font-weight: 700;">$1</strong>');
    s = s.replace(/\*(.*?)\*/g, '<em style="color: #5C5952;">$1</em>');

    // Inline code
    s = s.replace(/`([^`]+)`/g, '<code style="background: rgba(41, 53, 45, 0.08); padding: 2px 6px; border-radius: 4px; font-family: \'JetBrains Mono\', monospace; font-size: 0.88em;">$1</code>');

    // Numbered lists (e.g. "1. Item")
    s = s.replace(/^\s*(\d+)\.\s+(.*$)/gim, '<li style="margin-left: 20px; margin-bottom: 4px; color: #29352D;">$2</li>');

    // Bullet points (e.g. "- Item" or "• Item")
    s = s.replace(/^\s*[-•*]\s+(.*$)/gim, '<li style="margin-left: 20px; margin-bottom: 4px; color: #29352D;">$1</li>');

    // Wrap consecutive <li> into <ul>
    s = s.replace(/(<li[\s\S]*?<\/li>\s*)+/g, (match) => {
      return '<ul style="margin: 6px 0 10px; padding-left: 0;">' + match.trim() + '</ul>';
    });

    // Line breaks
    s = s.replace(/\n\n+/g, '<br/><br/>').replace(/\n/g, '<br/>');

    // Clean up unnecessary breaks directly flanking block elements
    s = s.replace(/(<br\s*\/?>)+(<\/?(h3|h4|ul|li))/gi, '$2');
    s = s.replace(/(<\/(h3|h4|ul|li)>)(<br\s*\/?>)+/gi, '$1');

    return s;
  }

  let currentBotRawText = "";

  function appendBotToken(token) {
    if (!currentBotRow) {
      currentBotRow = document.createElement("div");
      currentBotRow.className = "chat-row bot";
      const initial = (headerAvatar?.textContent || "B").trim();
      currentBotRow.innerHTML = `
        <div class="row-avatar">${escapeHtml(initial)}</div>
        <div class="chat-bubble"></div>
      `;
      chatMessages.appendChild(currentBotRow);
      currentBotRawText = "";
      scrollToBottom(true);
    }
    currentBotRawText += token;
    const bubble = currentBotRow.querySelector(".chat-bubble");
    bubble.innerHTML = formatMarkdown(currentBotRawText);
    scrollToBottom(false);
  }

  function showLeadCapturedNotice(lead) {
    const notice = document.createElement("div");
    notice.style.alignSelf = "flex-start";
    notice.style.marginLeft = "48px";
    notice.style.padding = "10px 16px";
    notice.style.background = "rgba(57, 130, 99, 0.12)";
    notice.style.border = "1px solid rgba(57, 130, 99, 0.35)";
    notice.style.borderRadius = "12px";
    notice.style.fontSize = "0.84rem";
    notice.style.color = "#29352D";
    notice.innerHTML = `
      <strong style="color: #29352D;">📅 Contact Details &amp; Request Received:</strong><br/>
      Our team has noted your details and will be in touch shortly.
    `;
    chatMessages.appendChild(notice);
    scrollToBottom(true);
  }

  function finalizeCurrentBotMessage() {
    currentBotRow = null;
    currentBotRawText = "";
    if (btnSendCustomerQuery) btnSendCustomerQuery.disabled = false;
    if (customerQueryInput) {
      customerQueryInput.disabled = false;
      customerQueryInput.focus();
    }
    scrollToBottom(true, true);
  }

  function sendQuery(queryText) {
    const text = (queryText || customerQueryInput.value).trim();
    if (!text || !chatWs || chatWs.readyState !== WebSocket.OPEN) return;

    // Permanently remove initial suggestion chips on first user message
    const initialSuggestions = document.getElementById("initialSuggestions");
    if (initialSuggestions) {
      initialSuggestions.remove();
    }

    // Append user message
    const userRow = document.createElement("div");
    userRow.className = "chat-row user";
    userRow.innerHTML = `
      <div class="row-avatar">You</div>
      <div class="chat-bubble">${escapeHtml(text)}</div>
    `;
    chatMessages.appendChild(userRow);
    scrollToBottom(true, true);

    // Reset input
    customerQueryInput.value = "";
    customerQueryInput.style.height = "auto";
    customerQueryInput.disabled = true;
    btnSendCustomerQuery.disabled = true;

    currentBotRow = null;
    showThinkingIndicator();
    chatWs.send(JSON.stringify({ query: text }));
  }

  if (btnSendCustomerQuery) {
    btnSendCustomerQuery.addEventListener("click", () => sendQuery());
  }

  if (customerQueryInput) {
    customerQueryInput.addEventListener("input", () => {
      customerQueryInput.style.height = "auto";
      const newHeight = Math.min(customerQueryInput.scrollHeight, 120);
      customerQueryInput.style.height = newHeight + "px";
      if (customerQueryInput.scrollHeight > 120) {
        customerQueryInput.style.overflowY = "auto";
      } else {
        customerQueryInput.style.overflowY = "hidden";
      }
    });

    customerQueryInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendQuery();
      }
    });
  }

  window.sendSuggestion = function (text) {
    sendQuery(text);
  };

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  initChat();
});
