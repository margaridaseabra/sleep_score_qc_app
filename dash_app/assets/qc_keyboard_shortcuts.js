(function () {
  function isTypingTarget(event) {
    const el = event.target;
    if (!el) return false;
    const tag = (el.tagName || "").toLowerCase();
    return (
      tag === "input" ||
      tag === "textarea" ||
      tag === "select" ||
      el.isContentEditable
    );
  }

  function findQcPlot() {
    const graph = document.getElementById("qc-graph");
    if (!graph) return null;
    return graph.querySelector(".js-plotly-plot");
  }

  function showToast(message) {
    let toast = document.getElementById("qc-keyboard-toast");

    if (!toast) {
      toast = document.createElement("div");
      toast.id = "qc-keyboard-toast";
      toast.style.position = "fixed";
      toast.style.right = "18px";
      toast.style.bottom = "18px";
      toast.style.zIndex = "9999";
      toast.style.background = "rgba(17, 24, 39, 0.92)";
      toast.style.color = "white";
      toast.style.padding = "10px 14px";
      toast.style.borderRadius = "10px";
      toast.style.fontFamily = "-apple-system, BlinkMacSystemFont, Segoe UI, Arial, sans-serif";
      toast.style.fontSize = "13px";
      toast.style.boxShadow = "0 8px 24px rgba(0,0,0,0.22)";
      toast.style.pointerEvents = "none";
      document.body.appendChild(toast);
    }

    toast.textContent = message;
    toast.style.opacity = "1";

    clearTimeout(window.__qcToastTimer);
    window.__qcToastTimer = setTimeout(function () {
      toast.style.opacity = "0";
    }, 1600);
  }

  function setPlotMode(mode) {
    const plot = findQcPlot();

    if (!plot || !window.Plotly) {
      showToast("QC plot not loaded yet");
      return;
    }

    const update = { dragmode: mode };

    if (mode === "select") {
      update.selectdirection = "h";
    }

    window.Plotly.relayout(plot, update);

    if (mode === "pan") showToast("QC mode: Pan");
    if (mode === "select") showToast("QC mode: Select window");
    if (mode === "zoom") showToast("QC mode: Zoom");
  }

  function clickButton(id, message) {
    const btn = document.getElementById(id);

    if (!btn) {
      showToast("Button not available: " + id);
      return;
    }

    btn.click();
    showToast(message);
  }

  document.addEventListener("keydown", function (event) {
    if (isTypingTarget(event)) return;

    const key = event.key.toLowerCase();

    // Mouse interaction modes
    if (key === "p") {
      event.preventDefault();
      setPlotMode("pan");
      return;
    }

    if (key === "s") {
      event.preventDefault();
      setPlotMode("select");
      return;
    }

    if (key === "z") {
      event.preventDefault();
      setPlotMode("zoom");
      return;
    }

    // Scoring shortcuts
    if (key === "1") {
      event.preventDefault();
      clickButton("score-wake", "Apply Wake to selected interval");
      return;
    }

    if (key === "2") {
      event.preventDefault();
      clickButton("score-nrem", "Apply NREM to selected interval");
      return;
    }

    if (key === "3") {
      event.preventDefault();
      clickButton("score-rem", "Apply REM to selected interval");
      return;
    }

    // A = automatic / Somnotate scoring
    if (key === "a") {
      event.preventDefault();
      clickButton("score-somnotate", "Apply automatic / Somnotate scoring");
      return;
    }

    if (key === "l") {
      event.preventDefault();
      clickButton("score-layer1", "Apply Layer 1 scoring");
      return;
    }

    if (key === "m") {
      event.preventDefault();
      clickButton("score-manual", "Apply Manual scoring");
      return;
    }
  });

  window.addEventListener("load", function () {
    setTimeout(function () {
      showToast("Shortcuts: P=pan, S=select, 1/2/3=Wake/NREM/REM, A=Somnotate");
    }, 900);
  });
})();
