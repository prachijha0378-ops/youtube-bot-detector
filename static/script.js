// ── Population stats for Z-score display ─────────────────
const FEAT_STATS = {
  "Channel Age (days)":         {mean:1050,  std:620},
  "Views / Subscriber":         {mean:28,    std:18},
  "Views / Video":              {mean:8500,  std:9200},
  "Videos / Day":               {mean:0.13,  std:0.14},
  "Monthly Views / Subscriber": {mean:0.055, std:0.06},
  "Monthly Upload Rate":        {mean:0.15,  std:0.18},
  "Est. Views (30 days)":       {mean:55000, std:80000},
  "Log Subscribers":            {mean:10.1,  std:2.1},
  "Log Total Views":            {mean:13.0,  std:2.3},
};

async function analyzeChannel() {
  const url  = document.getElementById("channelUrl").value.trim();
  const btn  = document.getElementById("analyzeBtn");
  const load = document.getElementById("loading");
  const res  = document.getElementById("result");
  const err  = document.getElementById("error");

  if (!url) {
    err.textContent = "Pehle YouTube channel URL dalo.";
    err.style.display = "block";
    return;
  }

  err.style.display  = "none";
  res.style.display  = "none";
  load.style.display = "block";
  btn.disabled = true;
  btn.textContent = "⏳ Analyzing...";

  document.getElementById("loadingTxt").textContent =
    "YouTube se real data fetch ho raha hai...";

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({url})
    });
    const data = await response.json();

    if (!response.ok)
      throw new Error(data.error || "Kuch galat ho gaya.");

    document.getElementById("loadingTxt").textContent =
      "Isolation Forest ML model analyze kar raha hai...";

    await new Promise(r => setTimeout(r, 400)); // brief pause for UX

    renderResult(data);

  } catch (e) {
    err.textContent = "❌ " + e.message;
    err.style.display = "block";
  } finally {
    load.style.display = "none";
    btn.disabled = false;
    btn.textContent = "🔍 Analyze";
  }
}

function renderResult(data) {
  const ch  = data.channel;
  const an  = data.analysis;
  const score = an.score;

  // Channel card
  const img = document.getElementById("thumbnail");
  img.src = ch.thumbnail || "";
  img.style.display = ch.thumbnail ? "block" : "none";
  document.getElementById("channelName").textContent = ch.name;
  document.getElementById("channelHandle").textContent = ch.handle || "";
  document.getElementById("channelId").textContent    = "ID: " + ch.id;

  // Stats
  document.getElementById("subscribers").textContent = fmt(ch.subscribers);
  document.getElementById("views").textContent       = fmt(ch.views);
  document.getElementById("videos").textContent      = fmt(ch.videos);

  // Verdict card
  const vBox   = document.getElementById("verdictBox");
  const isHigh = score >= 65;
  const isMid  = score >= 45 && score < 65;
  vBox.className = "verdict-card " + (isHigh ? "danger" : isMid ? "suspicious" : "safe");

  document.getElementById("score").textContent   = score;
  document.getElementById("verdict").textContent = an.verdict;

  // Risk bar (animate)
  const fill = document.getElementById("riskBarFill");
  fill.style.width = "0%";
  setTimeout(() => { fill.style.width = score + "%"; }, 60);

  // Feature table — z_scores now use same keys as features
  const featTable = document.getElementById("featTable");
  featTable.innerHTML = "";
  const features = an.features || {};
  const zScores  = an.z_scores || {};

  for (const [dispName, val] of Object.entries(features)) {
    const z  = zScores[dispName] !== undefined ? zScores[dispName] : 0;
    const za = Math.abs(z);
    const flagH  = za > 2.5;
    const flagM  = za > 1.5 && !flagH;
    const zCls   = za > 1.5 ? (z > 0 ? "hi" : "lo") : "ok";
    const rowCls = flagH ? "feat-row danger-flag" : flagM ? "feat-row flagged" : "feat-row";
    const nameCls= flagH ? "feat-name danger-flag" : flagM ? "feat-name flagged" : "feat-name";
    const flag   = flagH ? "🚨 " : flagM ? "⚑ " : "";
    const row = document.createElement("div");
    row.className = rowCls;
    row.innerHTML = `
      <span class="${nameCls}">${flag}${dispName}</span>
      <span class="feat-val">${fmtVal(val)}</span>
      <span class="feat-z ${zCls}">${z >= 0 ? "+" : ""}${Number(z).toFixed(2)}</span>
    `;
    featTable.appendChild(row);
  }

  // Signals
  const sigBox = document.getElementById("signals");
  sigBox.innerHTML = "";
  (an.signals || []).forEach(sig => {
    const div = document.createElement("div");
    const isDanger = sig.startsWith("🚨");
    const isWarn   = sig.startsWith("⚠️");
    const isOk     = sig.startsWith("✅");
    div.className = "signal " +
      (isDanger ? "danger-sig" : isWarn ? "warn-sig" : isOk ? "ok-sig" : "");
    div.textContent = sig;
    sigBox.appendChild(div);
  });

  // ML info
  const mlGrid = document.getElementById("mlGrid");
  mlGrid.innerHTML = "";
  const mlItems = [
    ["Algorithm",       "Isolation Forest"],
    ["Estimators",      "200 trees"],
    ["Contamination",   "5% threshold"],
    ["ML Prediction",   an.ml_label === -1 ? "🤖 Anomaly (-1)" : "✅ Normal (+1)"],
    ["Raw IF Score",    an.raw_if_score],
    ["Risk Score",      score + " / 100"],
  ];
  mlItems.forEach(([label, val]) => {
    const d = document.createElement("div");
    d.className = "ml-item";
    d.innerHTML = `<div class="ml-label">${label}</div><div class="ml-val">${val}</div>`;
    mlGrid.appendChild(d);
  });

  document.getElementById("result").style.display = "block";
  document.getElementById("result").scrollIntoView({behavior:"smooth", block:"start"});
}

function fmt(n) {
  n = Number(n || 0);
  if (n >= 1e9) return (n/1e9).toFixed(2) + "B";
  if (n >= 1e6) return (n/1e6).toFixed(2) + "M";
  if (n >= 1e3) return (n/1e3).toFixed(1) + "K";
  return n.toLocaleString();
}

function fmtVal(v) {
  v = Number(v);
  if (v >= 1e6) return (v/1e6).toFixed(2) + "M";
  if (v >= 1e3) return (v/1e3).toFixed(1) + "K";
  if (v < 1 && v > 0) return v.toFixed(5);
  return (+v.toFixed(2)).toString();
}

// Enter key support
document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("channelUrl").addEventListener("keydown", e => {
    if (e.key === "Enter") analyzeChannel();
  });
});
