(() => {
  const body = document.body;
  const hint = document.createElement("div");
  hint.className = "presentation-hint";
  hint.setAttribute("role", "status");
  body.appendChild(hint);

  let hintTimer;
  function showHint(message) {
    hint.textContent = message;
    hint.classList.add("visible");
    window.clearTimeout(hintTimer);
    hintTimer = window.setTimeout(() => hint.classList.remove("visible"), 1800);
  }

  async function enterPresentation() {
    body.classList.add("presentation-mode");
    showHint("Presentation mode · Esc to exit");
    if (!document.fullscreenElement && document.documentElement.requestFullscreen) {
      try {
        await document.documentElement.requestFullscreen();
      } catch (_) {
        // The distraction-free layout still works when fullscreen is unavailable.
      }
    }
  }

  async function exitPresentation() {
    body.classList.remove("presentation-mode");
    if (document.fullscreenElement && document.exitFullscreen) {
      await document.exitFullscreen();
    }
  }

  document.addEventListener("keydown", (event) => {
    const target = event.target;
    const isTyping = target && (
      target.matches("input, textarea, select") || target.isContentEditable
    );
    if (isTyping) return;

    if (event.key.toLowerCase() === "p") {
      event.preventDefault();
      body.classList.contains("presentation-mode")
        ? exitPresentation()
        : enterPresentation();
    } else if (event.key === "Escape" && body.classList.contains("presentation-mode")) {
      exitPresentation();
    }
  });

  document.addEventListener("fullscreenchange", () => {
    if (!document.fullscreenElement) body.classList.remove("presentation-mode");
  });

  if (new URLSearchParams(window.location.search).get("present") === "1") {
    body.classList.add("presentation-mode");
    showHint("Presentation mode · press P or Esc to exit");
  }
})();
