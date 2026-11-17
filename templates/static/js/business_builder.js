"use strict";

document.addEventListener("DOMContentLoaded", () => {
  // ══════════════════════ PERFORMANCE TELEMETRY (Component 0) ══════════════════════
  const urlParams = new URLSearchParams(window.location.search);
  const isPerf = urlParams.get("perf") === "1" || sessionStorage.getItem("dv_perf") === "1";
  if (urlParams.get("perf") === "1") {
    sessionStorage.setItem("dv_perf", "1");
  }

  if (isPerf) {
    const _origFetch = window.fetch;
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

    const clickTs = sessionStorage.getItem("dv_perf_click_ts");
    if (clickTs) {
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          const elapsed = Date.now() - parseInt(clickTs, 10);
          const action = sessionStorage.getItem("dv_perf_action") || "admin register";
          console.log(`[perf-ui] ${action} | next page ready=${elapsed}ms since click`);
          sessionStorage.removeItem("dv_perf_click_ts");
          sessionStorage.removeItem("dv_perf_action");
        });
      });
    }
  }

  // Step Navigation Indicators
  const stepIndicators = [
    document.getElementById("stepIndicator1"),
    document.getElementById("stepIndicator2"),
    document.getElementById("stepIndicator3"),
    document.getElementById("stepIndicator4"),
  ];

  // Wizard Panels
  const panels = [
    document.getElementById("wizardStep1"),
    document.getElementById("wizardStep2"),
    document.getElementById("wizardStep3"),
    document.getElementById("wizardStep4"),
  ];

  // Step 1 Form Elements
  const bizNameInput = document.getElementById("bizName");
  const personaNameInput = document.getElementById("personaName");
  const websiteUrlInput = document.getElementById("websiteUrl");
  const websiteUrlsContainer = document.getElementById("websiteUrlsContainer");
  const btnAddUrlRow = document.getElementById("btnAddUrlRow");
  const aboutBusinessInput = document.getElementById("aboutBusiness");
  const btnScrapeUrl = document.getElementById("btnScrapeUrl");
  const scrapePreviewBox = document.getElementById("scrapePreviewBox");
  const pdfDropZone = document.getElementById("pdfDropZone");
  const pdfFileInput = document.getElementById("pdfFileInput");
  const stagedPdfList = document.getElementById("stagedPdfList");
  const step1Error = document.getElementById("step1Error");
  const btnGoToStep2 = document.getElementById("btnGoToStep2");

  // Step 2 Interview Elements
  const interviewTurnBadge = document.getElementById("interviewTurnBadge");
  const interviewChatContainer = document.getElementById("interviewChatContainer");
  const interviewInput = document.getElementById("interviewInput");
  const btnSendInterviewAnswer = document.getElementById("btnSendInterviewAnswer");
  const btnBackToStep1 = document.getElementById("btnBackToStep1");
  const btnGoToStep3 = document.getElementById("btnGoToStep3");

  // Step 3 Prompt Elements
  const generatedPromptArea = document.getElementById("generatedPromptArea");
  const btnBackToStep2 = document.getElementById("btnBackToStep2");
  const btnGoToStep4 = document.getElementById("btnGoToStep4");

  // Step 4 Simulator & Deploy Elements
  const simBotTitle = document.getElementById("simBotTitle");
  const simWebsiteLink = document.getElementById("simWebsiteLink");
  const simWebsiteText = document.getElementById("simWebsiteText");
  const simGreeting = document.getElementById("simGreeting");
  const simMessages = document.getElementById("simMessages");
  const simInput = document.getElementById("simInput");
  const btnSendSim = document.getElementById("btnSendSim");
  const botPublicSummary = document.getElementById("botPublicSummary");
  const deployStatusAlert = document.getElementById("deployStatusAlert");
  const btnBackToStep3 = document.getElementById("btnBackToStep3");
  const btnDeployBot = document.getElementById("btnDeployBot");

  // State
  let stagedFiles = [];
  let scrapedData = null;
  let extractedBrochuresText = "";
  let tempBotId = "bot_" + Math.random().toString(36).substring(2, 10);
  let interviewId = "interview_" + Math.random().toString(36).substring(2, 10);
  let interviewWs = null;
  let finalizedBlueprint = null;
  let currentTurn = 1;
  let simWs = null;

  // ══════════════════════ STEP NAVIGATION ══════════════════════
  function goToStep(stepIndex) {
    panels.forEach((p, idx) => {
      if (p) p.style.display = idx === stepIndex ? "block" : "none";
    });

    stepIndicators.forEach((ind, idx) => {
      if (!ind) return;
      ind.classList.remove("active", "completed");
      if (idx < stepIndex) ind.classList.add("completed");
      if (idx === stepIndex) ind.classList.add("active");
    });

    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  // ══════════════════════ STEP 1: ASSET STAGING ══════════════════════

  // PDF File Drop Handling
  if (pdfDropZone && pdfFileInput) {
    pdfDropZone.addEventListener("click", () => pdfFileInput.click());
    pdfDropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      pdfDropZone.style.borderColor = "var(--terracotta, #BC704F)";
    });
    pdfDropZone.addEventListener("dragleave", () => {
      pdfDropZone.style.borderColor = "rgba(188, 112, 79, 0.35)";
    });
    pdfDropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      pdfDropZone.style.borderColor = "rgba(188, 112, 79, 0.35)";
      handleFilesSelected(Array.from(e.dataTransfer.files || []));
    });

    pdfFileInput.addEventListener("change", (e) => {
      handleFilesSelected(Array.from(e.target.files || []));
      e.target.value = "";
    });
  }

  function handleFilesSelected(files) {
    step1Error.style.display = "none";
    for (const f of files) {
      if (!f.name.toLowerCase().endsWith(".pdf")) {
        step1Error.textContent = `File "${f.name}" is not a PDF. Only .pdf files are supported.`;
        step1Error.style.display = "block";
        return;
      }
      if (f.size > 4 * 1024 * 1024) {
        step1Error.textContent = `File "${f.name}" exceeds the 4 MB limit.`;
        step1Error.style.display = "block";
        return;
      }
      if (stagedFiles.length >= 3) {
        step1Error.textContent = "Maximum 3 brochure PDFs allowed.";
        step1Error.style.display = "block";
        return;
      }
      stagedFiles.push(f);
    }
    renderStagedFiles();
  }

  function renderStagedFiles() {
    if (!stagedPdfList) return;
    stagedPdfList.innerHTML = stagedFiles
      .map(
        (f, idx) => `
        <div class="file-badge">
          <span>📄 ${escapeHtml(f.name)} (${(f.size / 1024 / 1024).toFixed(2)} MB)</span>
          <button type="button" onclick="removeStagedFile(${idx})" style="background: none; border: none; color: #f87171; cursor: pointer; font-weight: 700;">✕</button>
        </div>`
      )
      .join("");
  }

  window.removeStagedFile = function (idx) {
    stagedFiles.splice(idx, 1);
    renderStagedFiles();
  };

  // Multi-URL Ingestion Helpers
  function collectEnteredUrls() {
    const inputs = document.querySelectorAll(".website-url-field");
    const urls = [];
    inputs.forEach((inp) => {
      const val = inp.value.trim();
      if (val) {
        val.split(/[\n,]+/).forEach((u) => {
          const clean = u.trim();
          if (clean && !urls.includes(clean)) urls.push(clean);
        });
      }
    });
    return urls;
  }

  if (btnAddUrlRow && websiteUrlsContainer) {
    btnAddUrlRow.addEventListener("click", () => {
      const currentRows = websiteUrlsContainer.querySelectorAll(".website-url-field");
      if (currentRows.length >= 5) {
        showToast("Maximum of 5 website pages allowed.", "info");
        return;
      }
      const row = document.createElement("div");
      row.className = "scrape-input-row extra-url-row";
      row.style.cssText = "display: flex; gap: 10px; align-items: center; margin-top: 4px;";
      row.innerHTML = `
        <input type="url" class="form-input website-url-field" placeholder="https://example.com/pricing" />
        <button type="button" class="btn-remove-url" style="background: none; border: 1px solid var(--border-warm, #E5E1D8); border-radius: 8px; color: var(--forest-charcoal-muted, #677369); cursor: pointer; padding: 6px 12px; font-size: 0.85rem; font-weight: 600; transition: all 0.2s ease;" title="Remove this page URL">✕ Remove</button>
      `;
      websiteUrlsContainer.appendChild(row);
      const removeBtn = row.querySelector(".btn-remove-url");
      removeBtn.addEventListener("click", () => {
        row.remove();
        if (websiteUrlsContainer.querySelectorAll(".website-url-field").length < 5 && btnAddUrlRow) {
          btnAddUrlRow.style.display = "inline-flex";
        }
      });
      removeBtn.addEventListener("mouseenter", () => {
        removeBtn.style.color = "var(--status-red, #C6534E)";
        removeBtn.style.borderColor = "var(--status-red, #C6534E)";
      });
      removeBtn.addEventListener("mouseleave", () => {
        removeBtn.style.color = "var(--forest-charcoal-muted, #677369)";
        removeBtn.style.borderColor = "var(--border-warm, #E5E1D8)";
      });
      if (websiteUrlsContainer.querySelectorAll(".website-url-field").length >= 5 && btnAddUrlRow) {
        btnAddUrlRow.style.display = "none";
      }
    });
  }

  // Website Scraper & Health Diagnostic Rendering
  if (btnScrapeUrl) {
    btnScrapeUrl.addEventListener("click", async () => {
      const urls = collectEnteredUrls();
      if (!urls.length) {
        step1Error.textContent = "Please enter at least one valid website URL to scrape.";
        step1Error.style.display = "block";
        return;
      }

      step1Error.style.display = "none";
      btnScrapeUrl.disabled = true;
      btnScrapeUrl.textContent = "Scraping...";
      scrapePreviewBox.style.display = "block";
      scrapePreviewBox.style.background = "var(--surface-soft, #FAF8F5)";
      scrapePreviewBox.style.border = "1px solid var(--border-warm, #E5E1D8)";
      scrapePreviewBox.innerHTML = `
        <div style="display: flex; align-items: center; gap: 8px; color: var(--forest-charcoal-muted, #677369);">
          <span class="chat-spinner" style="width: 14px; height: 14px; border-width: 2px;"></span>
          <span>Connecting to ${urls.length} page(s) and extracting content...</span>
        </div>`;

      try {
        const res = await fetch("/api/business-bot/admin/scrape-url", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ urls }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Failed to scrape URL(s)");

        scrapedData = data;
        renderScrapeDiagnosticBadge(data);
      } catch (err) {
        renderScrapeErrorBadge(err.message);
      } finally {
        btnScrapeUrl.disabled = false;
        btnScrapeUrl.textContent = "Scrape & Preview";
      }
    });
  }

  function renderScrapeDiagnosticBadge(data) {
    if (!scrapePreviewBox) return;
    scrapePreviewBox.style.display = "block";

    const status = data.overall_status || (data.char_count >= 500 ? "high_yield" : (data.char_count > 0 ? "low_content" : "error"));
    const charCount = Number(data.char_count || 0);
    const wordCount = Number(data.word_count || 0);
    const pages = Array.isArray(data.pages) ? data.pages : [];

    let badgeBg = "rgba(57, 130, 99, 0.08)";
    let badgeBorder = "1px solid rgba(57, 130, 99, 0.35)";
    let headerIcon = "✅";
    let headerTitle = `Ingestion Successful: ${charCount.toLocaleString()} chars (${wordCount.toLocaleString()} words)`;
    let headerColor = "var(--forest-charcoal, #29352D)";
    let guidanceHtml = "";

    if (status === "low_content") {
      badgeBg = "rgba(185, 134, 56, 0.09)";
      badgeBorder = "1px solid rgba(185, 134, 56, 0.4)";
      headerIcon = "⚠️";
      headerTitle = `Low Content Warning (${charCount.toLocaleString()} characters extracted)`;
      headerColor = "#92611A";
      guidanceHtml = `
        <div style="margin-top: 8px; padding: 10px 12px; background: rgba(255,255,255,0.7); border-radius: 8px; border: 1px dashed rgba(185, 134, 56, 0.4); font-size: 0.78rem; line-height: 1.45; color: var(--forest-charcoal, #29352D);">
          <strong>Why is this?</strong> This website appears to use client-side JavaScript rendering (Single Page Application / React / Vue) where text loads after execution.<br/>
          <strong>Safety Net Actions:</strong>
          <ul style="margin: 4px 0 0; padding-left: 18px;">
            <li>Paste your product, service, and pricing descriptions directly into <strong>'About Business'</strong> below.</li>
            <li>Or open the page in your browser, press <strong>Ctrl+P (Print &rarr; Save as PDF)</strong>, and drop the PDF into the Brochure Upload below.</li>
          </ul>
        </div>`;
    } else if (status === "blocked") {
      badgeBg = "rgba(198, 83, 78, 0.08)";
      badgeBorder = "1px solid rgba(198, 83, 78, 0.35)";
      headerIcon = "❌";
      headerTitle = "Website Ingestion Blocked (HTTP 403 / Anti-Bot Firewall)";
      headerColor = "var(--status-red, #C6534E)";
      guidanceHtml = `
        <div style="margin-top: 8px; padding: 10px 12px; background: rgba(255,255,255,0.7); border-radius: 8px; border: 1px dashed rgba(198, 83, 78, 0.35); font-size: 0.78rem; line-height: 1.45; color: var(--forest-charcoal, #29352D);">
          <strong>Firewall Restriction:</strong> Automated crawlers were denied access by the server's anti-bot protection (e.g. Cloudflare / WAF).<br/>
          <strong>No Problem!</strong> You don't need a live website scrape to create a high-precision bot. Simply copy and paste your service details or catalog text into <strong>'About Business'</strong> below, or upload your company brochures.
        </div>`;
    }

    let pagesBreakdown = "";
    if (pages.length > 0) {
      pagesBreakdown = `
        <div style="margin-top: 8px; display: flex; flex-direction: column; gap: 4px;">
          ${pages.map((p) => {
            const pageIcon = p.status === "success" ? "✓" : (p.status === "low_content" ? "⚠" : "✕");
            const pageColor = p.status === "success" ? "var(--forest-charcoal, #29352D)" : (p.status === "low_content" ? "#92611A" : "var(--status-red, #C6534E)");
            return `
              <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.76rem; background: rgba(255,255,255,0.6); padding: 4px 8px; border-radius: 6px; color: ${pageColor};">
                <span style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 75%;" title="${escapeHtml(p.url)}">
                  <strong>${pageIcon}</strong> ${escapeHtml(p.title || p.url)}
                </span>
                <span style="font-weight: 600; white-space: nowrap;">${p.char_count?.toLocaleString() || 0} chars</span>
              </div>`;
          }).join("")}
        </div>`;
    }

    let hintsHtml = "";
    if (data.pricing_hints && data.pricing_hints.length) {
      hintsHtml += `
        <div style="margin-top: 8px; font-size: 0.77rem;">
          <strong style="color: var(--terracotta, #BC704F);">Detected Pricing:</strong>
          <span style="color: var(--forest-charcoal, #29352D);">${escapeHtml(data.pricing_hints.slice(0, 5).join(" • "))}</span>
        </div>`;
    }
    if (data.contact_hints && data.contact_hints.length) {
      hintsHtml += `
        <div style="margin-top: 4px; font-size: 0.77rem;">
          <strong style="color: var(--terracotta, #BC704F);">Contacts:</strong>
          <span style="color: var(--forest-charcoal, #29352D);">${escapeHtml(data.contact_hints.slice(0, 4).join(" • "))}</span>
        </div>`;
    }

    let previewToggle = "";
    if (data.text && charCount > 0) {
      previewToggle = `
        <details style="margin-top: 10px; font-size: 0.76rem; cursor: pointer;">
          <summary style="font-weight: 600; color: var(--terracotta, #BC704F); outline: none;">Show Ingested Text Excerpt</summary>
          <pre style="margin-top: 6px; padding: 8px; background: rgba(0,0,0,0.03); border-radius: 6px; white-space: pre-wrap; max-height: 140px; overflow-y: auto; font-family: monospace; font-size: 0.72rem; color: var(--forest-charcoal, #29352D);">${escapeHtml(data.text.slice(0, 400))}...</pre>
        </details>`;
    }

    scrapePreviewBox.style.background = badgeBg;
    scrapePreviewBox.style.border = badgeBorder;
    scrapePreviewBox.innerHTML = `
      <div style="font-weight: 700; color: ${headerColor}; display: flex; align-items: center; gap: 6px; font-size: 0.85rem;">
        <span>${headerIcon}</span>
        <span>${escapeHtml(headerTitle)}</span>
      </div>
      ${guidanceHtml}
      ${pagesBreakdown}
      ${hintsHtml}
      ${previewToggle}
    `;
  }

  function renderScrapeErrorBadge(errorMsg) {
    if (!scrapePreviewBox) return;
    scrapePreviewBox.style.display = "block";
    scrapePreviewBox.style.background = "rgba(198, 83, 78, 0.08)";
    scrapePreviewBox.style.border = "1px solid rgba(198, 83, 78, 0.35)";
    scrapePreviewBox.innerHTML = `
      <div style="font-weight: 700; color: var(--status-red, #C6534E); display: flex; align-items: center; gap: 6px; font-size: 0.84rem;">
        <span>❌</span>
        <span>Scrape Error: ${escapeHtml(errorMsg)}</span>
      </div>
      <div style="margin-top: 6px; font-size: 0.78rem; color: var(--forest-charcoal, #29352D); line-height: 1.4;">
        Please double-check the URL format or use the <strong>'About Business'</strong> safety-net textarea below to provide product information directly.
      </div>
    `;
  }

  // Go to Step 2
  if (btnGoToStep2) {
    btnGoToStep2.addEventListener("click", async () => {
      const bizName = bizNameInput.value.trim();
      const personaName = personaNameInput.value.trim();

      if (!bizName || !personaName) {
        step1Error.textContent = "Please enter both Business Name and Bot Persona Name.";
        step1Error.style.display = "block";
        return;
      }

      if (stagedFiles.length === 0) {
        step1Error.textContent = "Please upload at least 1 brochure PDF (up to 3, max 4MB each).";
        step1Error.style.display = "block";
        return;
      }

      step1Error.style.display = "none";
      btnGoToStep2.disabled = true;
      btnGoToStep2.textContent = "Ingesting Brochures into Qdrant...";

      try {
        // If user entered URL(s) but didn't click "Scrape & Preview", auto-scrape them seamlessly before advancing
        if (!scrapedData) {
          const enteredUrls = collectEnteredUrls();
          if (enteredUrls.length > 0) {
            try {
              btnGoToStep2.textContent = "Extracting Website Knowledge...";
              const scrapeRes = await fetch("/api/business-bot/admin/scrape-url", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ urls: enteredUrls }),
              });
              if (scrapeRes.ok) {
                scrapedData = await scrapeRes.json();
              }
            } catch (e) {
              console.warn("Auto-scrape fallback error:", e);
            }
          }
        }

        btnGoToStep2.textContent = "Ingesting Brochures into Qdrant...";
        const formData = new FormData();
        formData.append("bot_id", tempBotId);
        stagedFiles.forEach((file) => formData.append("files", file));

        const res = await fetch("/api/business-bot/admin/upload-brochures", {
          method: "POST",
          body: formData,
        });
        const uploadResult = await res.json();
        if (!res.ok) throw new Error(uploadResult.detail || "Brochure ingestion failed.");

        extractedBrochuresText = uploadResult.extracted_brochures_text || "";
        goToStep(1); // Step 2 (0-indexed 1)
        startArchitectInterview();
      } catch (err) {
        step1Error.textContent = "Upload error: " + err.message;
        step1Error.style.display = "block";
      } finally {
        btnGoToStep2.disabled = false;
        btnGoToStep2.textContent = "Next: Start AI Architect Interview →";
      }
    });
  }

  // ══════════════════════ STEP 2: ARCHITECT INTERVIEW ══════════════════════
  function showArchitectThinking() {
    removeArchitectThinking();
    if (!interviewChatContainer) return;
    const bubble = document.createElement("div");
    bubble.className = "interview-msg architect";
    bubble.id = "architectThinkingBubble";
    bubble.innerHTML = `
      <div class="interview-avatar">AI</div>
      <div class="interview-bubble">
        <div class="typing-indicator">
          <span></span><span></span><span></span>
        </div>
      </div>
    `;
    interviewChatContainer.appendChild(bubble);
    interviewChatContainer.scrollTop = interviewChatContainer.scrollHeight;
  }

  function removeArchitectThinking() {
    const el = document.getElementById("architectThinkingBubble");
    if (el) el.remove();
  }

  function appendInterviewMessage(role, text) {
    removeArchitectThinking();
    if (!interviewChatContainer) return;
    const row = document.createElement("div");
    row.className = `interview-msg ${role}`;
    let formattedText = escapeHtml(text);
    if (role === "architect") {
      // Parse markdown bold headers (e.g. **Question 1: Target Audience** or **Insight & Strategy**)
      formattedText = formattedText.replace(/\*\*(.*?)\*\*/g, "<strong class=\"architect-header\">$1</strong>");
      // Format newlines to line breaks
      formattedText = formattedText.replace(/\n/g, "<br/>");
    }
    row.innerHTML = `
      <div class="interview-avatar">${role === "architect" ? "AI" : "You"}</div>
      <div class="interview-bubble">${formattedText}</div>
    `;
    interviewChatContainer.appendChild(row);
    interviewChatContainer.scrollTop = interviewChatContainer.scrollHeight;
  }

  function startArchitectInterview() {
    if (interviewWs && interviewWs.readyState === WebSocket.OPEN) return;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/business-bot/interview/${interviewId}`;

    interviewWs = new WebSocket(wsUrl);

    interviewWs.onopen = () => {
      interviewChatContainer.innerHTML = "";
      showArchitectThinking();

      let contextBlocks = [];
      if (scrapedData && scrapedData.text) {
        contextBlocks.push(`=== WEBSITE CONTENT (${scrapedData.url || "URL"}) ===\n${scrapedData.text.slice(0, 8000)}`);
      }
      if (extractedBrochuresText) {
        contextBlocks.push(`=== UPLOADED BROCHURES ===\n${extractedBrochuresText.slice(0, 10000)}`);
      }
      const aboutText = aboutBusinessInput ? aboutBusinessInput.value.trim() : "";
      if (aboutText) {
        contextBlocks.push(`=== ABOUT BUSINESS & PRODUCTS ===\n${aboutText}`);
      }

      const fullContext = contextBlocks.join("\n\n");

      interviewWs.send(
        JSON.stringify({
          type: "init",
          business_name: bizNameInput.value.trim(),
          persona_name: personaNameInput.value.trim(),
          extracted_context: fullContext,
        })
      );
    };

    interviewWs.onmessage = (event) => {
      try {
        removeArchitectThinking();
        const data = JSON.parse(event.data);

        if (data.stream_question) {
          appendInterviewMessage("architect", data.stream_question);
        }

        const covered = data.internal_analysis?.parameters_covered || (data.vector_covered ? [data.vector_covered] : []);
        if (Array.isArray(covered)) {
          covered.forEach((vec) => {
            const pill = document.getElementById(`vec_${vec}`);
            if (pill) {
              pill.classList.remove("active");
              pill.classList.add("done");
            }
          });
        }
        if (data.vector_states && typeof data.vector_states === "object") {
          Object.entries(data.vector_states).forEach(([vec, stateObj]) => {
            const state = typeof stateObj === "string" ? stateObj : (stateObj.state || "");
            const pill = document.getElementById(`vec_${vec}`);
            if (pill && (state.includes("VERIFIED") || state === "done")) {
              pill.classList.remove("active");
              pill.classList.add("done");
            }
          });
        }

        const nextVec = data.internal_analysis?.next_target_vector;
        if (nextVec) {
          const nextPill = document.getElementById(`vec_${nextVec}`);
          if (nextPill && !nextPill.classList.contains("done")) {
            nextPill.classList.add("active");
          }
        }

        const qCount = data.question_count || data.turn_count || data.turn;
        const isClarif = data.is_clarification;
        if (interviewTurnBadge && qCount) {
          currentTurn = qCount;
          if (isClarif) {
            interviewTurnBadge.textContent = `Clarification (Q${currentTurn} of 10)`;
          } else {
            interviewTurnBadge.textContent = `Question ${currentTurn} of 10`;
          }
        }

        if (data.event === "finalized_blueprint" || data.finalized_blueprint || data.type === "interview_complete" || data.interview_status === "COMPLETE") {
          finalizedBlueprint = data.finalized_blueprint || data.blueprint || data;
          appendInterviewMessage(
            "architect",
            "🎉 Excellent! Core business vectors gathered. Your custom bot blueprint is finalized. Click 'Next: Review & Synthesize Prompt' below to proceed!"
          );
          if (btnGoToStep3) {
            btnGoToStep3.disabled = false;
          }
          const btnFinishEarly = document.getElementById("btnFinishInterviewEarly");
          if (btnFinishEarly) {
            btnFinishEarly.style.display = "none";
          }
          if (interviewTurnBadge) {
            interviewTurnBadge.textContent = "Interview Completed";
            interviewTurnBadge.style.color = "var(--status-success, #398263)";
            interviewTurnBadge.style.background = "rgba(57, 130, 99, 0.12)";
            interviewTurnBadge.style.borderColor = "rgba(57, 130, 99, 0.35)";
          }
        }
      } catch (err) {
        console.error("Interview parse error:", err);
      }
    };

    interviewWs.onerror = (err) => {
      removeArchitectThinking();
      appendInterviewMessage("architect", "WebSocket connection error. Please refresh or check connection.");
    };

    interviewWs.onclose = () => {
      removeArchitectThinking();
      console.debug("Interview WS closed.");
    };
  }

  function sendInterviewAnswer() {
    const text = interviewInput.value.trim();
    if (!text || !interviewWs || interviewWs.readyState !== WebSocket.OPEN) return;

    appendInterviewMessage("user", text);
    showArchitectThinking();
    interviewWs.send(JSON.stringify({ text }));
    interviewInput.value = "";
  }

  const btnFinishEarly = document.getElementById("btnFinishInterviewEarly");
  if (btnFinishEarly) {
    btnFinishEarly.addEventListener("click", () => {
      if (!interviewWs || interviewWs.readyState !== WebSocket.OPEN) {
        if (finalizedBlueprint && btnGoToStep3) {
          btnGoToStep3.disabled = false;
          goToStep(2);
        }
        return;
      }
      showArchitectThinking();
      btnFinishEarly.disabled = true;
      btnFinishEarly.textContent = "Finalizing Blueprint...";
      interviewWs.send(JSON.stringify({ type: "finish" }));
    });
  }

  if (btnSendInterviewAnswer) {
    btnSendInterviewAnswer.addEventListener("click", sendInterviewAnswer);
  }
  if (interviewInput) {
    interviewInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        sendInterviewAnswer();
      }
    });
  }

  if (btnBackToStep1) btnBackToStep1.addEventListener("click", () => goToStep(0));

  // Go to Step 3 (Prompt Synthesis)
  if (btnGoToStep3) {
    btnGoToStep3.addEventListener("click", async () => {
      btnGoToStep3.disabled = true;
      btnGoToStep3.textContent = "Synthesizing Dynamic Master Prompt...";

      try {
        const payload = {
          interview_id: interviewId,
          business_name: bizNameInput.value.trim(),
          persona_name: personaNameInput.value.trim(),
          blueprint: finalizedBlueprint || {},
          website_content: scrapedData?.text || "",
        };

        const res = await fetch("/api/business-bot/admin/generate-prompt", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Failed to synthesize prompt");

        generatedPromptArea.value = data.prompt || "";
        goToStep(2); // Step 3
      } catch (err) {
        alert("Prompt synthesis error: " + err.message);
      } finally {
        btnGoToStep3.disabled = false;
        btnGoToStep3.textContent = "Next: Review & Synthesize Prompt →";
      }
    });
  }

  // ══════════════════════ STEP 3: PROMPT REVIEW ══════════════════════
  if (btnBackToStep2) btnBackToStep2.addEventListener("click", () => goToStep(1));

  function formatMarkdown(text) {
    if (!text) return "";
    let s = escapeHtml(text);
    // Headings
    s = s.replace(/^### (.*$)/gim, '<h4 style="margin: 8px 0 4px; color: var(--forest-charcoal, #252923); font-size: 0.95rem;">$1</h4>');
    s = s.replace(/^## (.*$)/gim, '<h3 style="margin: 10px 0 4px; color: var(--forest-charcoal, #252923); font-size: 1.05rem;">$1</h3>');
    // Bold & italic
    s = s.replace(/\*\*(.*?)\*\*/g, '<strong style="color: var(--forest-charcoal, #252923);">$1</strong>');
    s = s.replace(/\*(.*?)\*/g, '<em>$1</em>');
    // Bullet points
    s = s.replace(/^\s*[-•]\s+(.*$)/gim, '<li style="margin-left: 18px; margin-bottom: 3px;">$1</li>');
    // Wrap consecutive <li> into <ul>
    s = s.replace(/(<li[\s\S]*?<\/li>)+/g, '<ul style="margin: 4px 0 8px; padding-left: 0;">$&</ul>');
    // Line breaks
    s = s.replace(/\n\n/g, '<br/><br/>').replace(/\n/g, '<br/>');
    return s;
  }

  if (btnGoToStep4) {
    btnGoToStep4.addEventListener("click", async () => {
      if (!generatedPromptArea.value.trim()) {
        alert("System prompt cannot be empty.");
        return;
      }

      btnGoToStep4.disabled = true;
      btnGoToStep4.textContent = "Setting up Simulator...";

      try {
        const draftPayload = {
          bot_id: tempBotId,
          business_name: bizNameInput.value.trim(),
          persona_name: personaNameInput.value.trim(),
          dynamic_prompt: generatedPromptArea.value.trim(),
          system_prompt: generatedPromptArea.value.trim(),
          summary: botPublicSummary.value.trim() || `${personaNameInput.value.trim()} – Assistant for ${bizNameInput.value.trim()}`,
          website_url: collectEnteredUrls().join(", ") || websiteUrlInput.value.trim() || null,
        };

        await fetch("/api/business-bot/admin/save-draft", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(draftPayload),
        });
      } catch (e) {
        console.warn("Could not pre-save draft bot:", e);
      } finally {
        btnGoToStep4.disabled = false;
        btnGoToStep4.textContent = "Next: Test Drive & Deploy →";
      }

      simBotTitle.textContent = `${personaNameInput.value.trim()} Simulator`;
      simGreeting.textContent = `Hello! I am ${personaNameInput.value.trim()} from ${bizNameInput.value.trim()}. How can I assist you today?`;

      // Clickable website link in simulator
      const webUrl = websiteUrlInput.value.trim();
      if (webUrl && simWebsiteLink) {
        simWebsiteLink.href = webUrl.startsWith("http") ? webUrl : `https://${webUrl}`;
        simWebsiteLink.style.display = "inline-flex";
        if (simWebsiteText) {
          try {
            simWebsiteText.textContent = `Visit ${new URL(simWebsiteLink.href).hostname}`;
          } catch (e) {
            simWebsiteText.textContent = "Visit Official Website";
          }
        }
      }

      if (!botPublicSummary.value.trim()) {
        botPublicSummary.value = `${personaNameInput.value.trim()} – Official AI assistant for ${bizNameInput.value.trim()}, helping customers with products, pricing, and scheduling.`;
      }
      goToStep(3); // Step 4
      initSimulator();
    });
  }

  // ══════════════════════ STEP 4: SIMULATOR & DEPLOY ══════════════════════
  function initSimulator() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/business-bot/chat/${tempBotId}`;

    if (simWs) simWs.close();
    simWs = new WebSocket(wsUrl);

    simWs.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "token") {
          appendSimToken(data.token);
        } else if (data.type === "lead_alert") {
          appendSimLeadAlert(data);
        } else if (data.type === "done") {
          finalizeSimMessage();
        }
      } catch (err) {}
    };
  }

  let currentSimRow = null;
  let currentSimRawText = "";

  function appendSimToken(token) {
    if (!currentSimRow) {
      currentSimRow = document.createElement("div");
      currentSimRow.className = "interview-msg architect";
      currentSimRow.innerHTML = `
        <div class="interview-avatar">Bot</div>
        <div class="interview-bubble"></div>
      `;
      simMessages.appendChild(currentSimRow);
      currentSimRawText = "";
    }
    currentSimRawText += token;
    const bubble = currentSimRow.querySelector(".interview-bubble");
    bubble.innerHTML = formatMarkdown(currentSimRawText);
    simMessages.scrollTop = simMessages.scrollHeight;
  }

  function finalizeSimMessage() {
    currentSimRow = null;
    currentSimRawText = "";
    if (btnSendSim) btnSendSim.disabled = false;
    if (simInput) simInput.disabled = false;
  }

  function appendSimLeadAlert(lead) {
    const alertDiv = document.createElement("div");
    alertDiv.style.padding = "8px 12px";
    alertDiv.style.background = "rgba(57, 130, 99, 0.12)";
    alertDiv.style.border = "1px solid rgba(57, 130, 99, 0.3)";
    alertDiv.style.borderRadius = "8px";
    alertDiv.style.fontSize = "0.78rem";
    alertDiv.style.color = "var(--status-success, #398263)";
    alertDiv.style.fontWeight = "600";
    alertDiv.innerHTML = `✓ Lead Trigger Detected: Appointment inquiry logged for ${escapeHtml(lead.customer_name || "Customer")}`;
    simMessages.appendChild(alertDiv);
    simMessages.scrollTop = simMessages.scrollHeight;
  }

  function sendSimMessage() {
    const query = simInput.value.trim();
    if (!query || !simWs || simWs.readyState !== WebSocket.OPEN) return;

    const userRow = document.createElement("div");
    userRow.className = "interview-msg user";
    userRow.innerHTML = `
      <div class="interview-avatar">You</div>
      <div class="interview-bubble">${escapeHtml(query)}</div>
    `;
    simMessages.appendChild(userRow);
    simMessages.scrollTop = simMessages.scrollHeight;

    currentSimRow = null;
    simWs.send(JSON.stringify({ query }));
    simInput.value = "";
  }

  if (btnSendSim) btnSendSim.addEventListener("click", sendSimMessage);
  if (simInput) {
    simInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        sendSimMessage();
      }
    });
  }

  if (btnBackToStep3) btnBackToStep3.addEventListener("click", () => goToStep(2));

  // Deploy Bot Action
  if (btnDeployBot) {
    btnDeployBot.addEventListener("click", async () => {
      btnDeployBot.disabled = true;
      btnDeployBot.textContent = "Deploying to Marketplace...";
      deployStatusAlert.style.display = "none";

      try {
        const payload = {
          bot_id: tempBotId,
          business_name: bizNameInput.value.trim(),
          persona_name: personaNameInput.value.trim(),
          system_prompt: generatedPromptArea.value.trim(),
          summary: botPublicSummary.value.trim(),
          website_url: collectEnteredUrls().join(", ") || websiteUrlInput.value.trim() || null,
        };

        const res = await fetch("/api/business-bot/admin/deploy", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const result = await res.json();
        if (!res.ok) throw new Error(result.detail || "Deployment failed.");

        deployStatusAlert.style.display = "block";
        deployStatusAlert.style.background = "rgba(57, 130, 99, 0.12)";
        deployStatusAlert.style.border = "1px solid rgba(57, 130, 99, 0.3)";
        deployStatusAlert.style.color = "var(--status-success, #398263)";
        deployStatusAlert.innerHTML = `
          <strong>🎉 Success! Assistant Deployed Live!</strong><br/>
          Your assistant is now listed in the DocuVortex Assistant Directory.<br/>
          <div style="margin-top: 12px; display: flex; flex-wrap: wrap; gap: 10px;">
            <a href="/bot/${tempBotId}" target="_blank" class="nav-link-primary" style="font-size: 0.82rem; padding: 6px 14px;">Open Live Chat &rarr;</a>
            <a href="/" class="nav-link-ghost" style="font-size: 0.82rem; padding: 6px 14px;">&larr; Return to Directory</a>
            <a href="/admin-dashboard" class="nav-link-ghost" style="font-size: 0.82rem; padding: 6px 14px;">Go to Admin Dashboard &rarr;</a>
          </div>
        `;
      } catch (err) {
        deployStatusAlert.style.display = "block";
        deployStatusAlert.style.background = "rgba(198, 83, 78, 0.12)";
        deployStatusAlert.style.border = "1px solid rgba(198, 83, 78, 0.3)";
        deployStatusAlert.style.color = "var(--status-error, #C6534E)";
        deployStatusAlert.textContent = "Deployment failed: " + err.message;
        btnDeployBot.disabled = false;
        btnDeployBot.textContent = "🚀 Deploy Assistant to Directory";
      }
    });
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
});
