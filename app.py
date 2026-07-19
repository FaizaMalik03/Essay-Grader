"""
app.py  -  Essay Grading Web Interface (Multi-Agent v2)
IT9204 | Student 202507410 Faiza Malik
Run: python app.py
Open: http://127.0.0.1:5000
"""

from flask import Flask, request, jsonify, Response
import os

app = Flask(__name__)

try:
    from agent_v2 import grade_essay
    AGENT_AVAILABLE = True
    print("Agent loaded (agent_v2.py)")
except Exception as e:
    AGENT_AVAILABLE = False
    print(f"Agent not available: {e}")


@app.route("/")
def index():
    return Response(HTML_PAGE, mimetype="text/html")


@app.route("/grade", methods=["POST"])
def grade():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"})

    user_message = (data.get("user_message") or data.get("essay_text") or "").strip()
    if not user_message:
        return jsonify({"error": "No message provided"})

    if not AGENT_AVAILABLE:
        return jsonify({"error": "Agent not loaded. Check agent_v2.py."})

    if not os.environ.get("GROQ_API_KEY"):
        return jsonify({"error": "GROQ_API_KEY not set."})

    try:
        result = grade_essay(user_message, verbose=True)
        result["word_count"] = result.get("word_count") or len(user_message.split())
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Essay Grading Assistant</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',sans-serif;background:#f0f4f8;min-height:100vh;padding:24px 16px}
.wrap{max-width:860px;margin:0 auto}
.hdr{background:linear-gradient(135deg,#1e3a5f 0%,#2d6a4f 100%);color:#fff;padding:28px;border-radius:14px;margin-bottom:22px;text-align:center}
.hdr h1{font-size:24px;font-weight:700;margin-bottom:6px}
.hdr p{font-size:13px;opacity:.8}
.tabs{display:flex;gap:8px;margin-bottom:18px;flex-wrap:wrap}
.tab{padding:9px 20px;border-radius:8px;border:2px solid #d1d5db;background:#fff;font-size:13px;font-weight:600;cursor:pointer;color:#555;transition:.2s}
.tab.active{border-color:#1e3a5f;background:#1e3a5f;color:#fff}
.tab:hover:not(.active){border-color:#1e3a5f;color:#1e3a5f}
.card{background:#fff;border-radius:12px;padding:24px;margin-bottom:18px;box-shadow:0 2px 10px rgba(0,0,0,.07)}
.card h2{font-size:16px;font-weight:700;color:#1e3a5f;margin-bottom:14px}
.hint{font-size:12px;color:#888;margin-bottom:10px;line-height:1.5}
textarea{width:100%;height:190px;border:2px solid #e0e0e0;border-radius:8px;padding:12px;font-size:14px;font-family:inherit;resize:vertical;outline:none;transition:.2s}
textarea:focus{border-color:#1e3a5f}
input[type=text]{width:100%;border:2px solid #e0e0e0;border-radius:8px;padding:10px 14px;font-size:14px;font-family:inherit;outline:none;transition:.2s}
input[type=text]:focus{border-color:#1e3a5f}
.btns{display:flex;gap:10px;margin-top:12px}
.btn-primary{padding:11px 28px;border:none;border-radius:8px;font-size:14px;font-weight:700;cursor:pointer;background:#1e3a5f;color:#fff;flex:1}
.btn-primary:hover{background:#16304f}
.btn-primary:disabled{opacity:.5;cursor:not-allowed}
.btn-clear{padding:11px 20px;border:2px solid #d1d5db;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer;background:#fff;color:#555}
.btn-clear:hover{background:#f5f5f5}
.loading{display:none;text-align:center;padding:28px;color:#555}
.spin{width:40px;height:40px;border:4px solid #e0e0e0;border-top:4px solid #1e3a5f;border-radius:50%;animation:spin .8s linear infinite;margin:0 auto 14px}
@keyframes spin{to{transform:rotate(360deg)}}
.err{background:#fef2f2;border:1px solid #fca5a5;border-radius:8px;padding:12px 16px;color:#dc2626;display:none;margin-bottom:14px;font-size:14px}
#result{display:none}
.score-card{background:linear-gradient(135deg,#1e3a5f,#2d6a4f);color:#fff;border-radius:12px;padding:22px 26px;display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}
.score-num{font-size:52px;font-weight:700;line-height:1}
.score-denom{font-size:12px;opacity:.75;margin-top:2px}
.score-raw{font-size:12px;opacity:.6}
.score-right{text-align:right}
.band-badge{display:inline-block;padding:5px 16px;border-radius:20px;font-size:13px;font-weight:700;margin-bottom:6px}
.model-lbl{font-size:11px;opacity:.7}
.fb{margin-bottom:12px;padding:14px 16px;background:#f8f9fa;border-radius:8px;border-left:4px solid #ccc}
.fb-title{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px}
.fb-body{font-size:14px;color:#444;line-height:1.65;white-space:pre-wrap}
.fb.summary{border-color:#6366f1}.fb.summary .fb-title{color:#6366f1}
.fb.strengths{border-color:#22c55e}.fb.strengths .fb-title{color:#22c55e}
.fb.improve{border-color:#ef4444}.fb.improve .fb-title{color:#ef4444}
.fb.nextsteps{border-color:#f97316}.fb.nextsteps .fb-title{color:#f97316}
.fb.advice{border-color:#8b5cf6}.fb.advice .fb-title{color:#8b5cf6}
.fb.stats{border-color:#0ea5e9}.fb.stats .fb-title{color:#0ea5e9}
.stats-grid{display:flex;gap:20px;flex-wrap:wrap;margin-bottom:12px}
.stat-pill{background:#e0f2fe;border-radius:8px;padding:8px 16px;font-size:13px}
.stat-pill strong{display:block;font-size:20px;color:#0369a1}
.stats-table{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}
.stats-table th,.stats-table td{border:1px solid #e5e7eb;padding:6px 10px;text-align:left}
.stats-table th{background:#f0f9ff;font-weight:600;color:#0369a1}
.criteria-list{list-style:none;padding:0}
.criteria-list li{padding:8px 12px;background:#f5f3ff;border-radius:6px;margin-bottom:6px;font-size:14px;color:#4c1d95;border-left:3px solid #8b5cf6}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}
.chip{background:#f0f4f8;border-radius:6px;padding:4px 10px;font-size:12px;color:#666}
.agent-badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;margin-bottom:14px;background:#e0f2fe;color:#0369a1}
</style>
</head>
<body>
<div class="wrap">

  <div class="hdr">
    <h1>AI Essay Grading Assistant</h1>
    <p>IT9204 Project | Student 202507410 Faiza Malik | Bahrain Polytechnic</p>
  </div>

  <div class="tabs">
    <button class="tab active" id="tab-grading">Full Grading</button>
    <button class="tab"        id="tab-scoring">Score Only</button>
    <button class="tab"        id="tab-advice" >Writing Advice</button>
    <button class="tab"        id="tab-stats"  >Statistics</button>
  </div>

  <div class="card">
    <h2 id="inputTitle">Grade My Essay</h2>
    <p class="hint" id="inputHint">Paste your essay below for a full score, feedback, strengths and improvements.</p>
    <textarea id="essayInput" placeholder="Paste your essay here..."></textarea>
    <input id="queryInput" type="text" placeholder="e.g. What makes a strong thesis?" style="display:none">
    <div class="btns">
      <button class="btn-primary" id="submitBtn">Grade Essay</button>
      <button class="btn-clear"   id="clearBtn">Clear</button>
    </div>
  </div>

  <div class="err" id="errBox"></div>

  <div class="loading" id="loading">
    <div class="spin"></div>
    <p id="loadingMsg">Grading your essay — please wait...</p>
  </div>

  <div class="card" id="result">
    <h2>Result</h2>
    <div class="agent-badge" id="agentBadge"></div>

    <div id="secScore" style="display:none">
      <div class="score-card">
        <div>
          <div class="score-num" id="sNum">-</div>
          <div class="score-denom">out of 10.0</div>
          <div class="score-raw" id="sRaw"></div>
        </div>
        <div class="score-right">
          <div class="band-badge" id="sBand">-</div>
          <div class="model-lbl"  id="sModel"></div>
        </div>
      </div>
    </div>

    <div class="fb summary"   id="secSummary"   style="display:none"><div class="fb-title">Summary</div><div class="fb-body" id="fSum"></div></div>
    <div class="fb strengths" id="secStrengths"  style="display:none"><div class="fb-title">Strengths</div><div class="fb-body" id="fStr"></div></div>
    <div class="fb improve"   id="secImprove"    style="display:none"><div class="fb-title">Improvements</div><div class="fb-body" id="fImp"></div></div>
    <div class="fb nextsteps" id="secNextSteps"  style="display:none"><div class="fb-title">Next Steps</div><div class="fb-body" id="fNxt"></div></div>

    <div id="secAdvice" style="display:none">
      <div class="fb advice"><div class="fb-title">Rubric Criteria</div><ul class="criteria-list" id="fCriteria"></ul></div>
      <div class="fb advice"><div class="fb-title">Writing Advice</div><div class="fb-body" id="fAdvice"></div></div>
    </div>

    <div id="secStats" style="display:none">
      <div class="fb stats">
        <div class="fb-title">Dataset Statistics</div>
        <div class="stats-grid" id="fStatsGrid"></div>
        <table class="stats-table"><thead><tr><th>Band</th><th>Count</th></tr></thead><tbody id="fStatsBands"></tbody></table>
        <p style="font-size:12px;color:#666;margin-top:8px" id="fStatsInterp"></p>
      </div>
    </div>

    <div class="chips">
      <div class="chip" id="cWords"></div>
      <div class="chip" id="cModel"></div>
    </div>
  </div>

</div>

<script>
// ── Config per mode ───────────────────────────────────────────────────────────
var MODES = {
  grading: {
    title:   "Grade My Essay",
    hint:    "Paste your essay for a full score, feedback, strengths and improvements.",
    btn:     "Grade Essay",
    input:   "essay",
    loading: "Grading your essay — please wait...",
    prefix:  "Please grade this essay fully with feedback:\n\n"
  },
  scoring: {
    title:   "Score My Essay",
    hint:    "Paste your essay to receive just the numeric score.",
    btn:     "Get Score",
    input:   "essay",
    loading: "Scoring your essay...",
    prefix:  "What score would this essay receive?\n\n"
  },
  advice: {
    title:   "Get Writing Advice",
    hint:    "Ask anything about essay writing, rubric criteria, or how to improve.",
    btn:     "Get Advice",
    input:   "query",
    loading: "Fetching rubric criteria...",
    prefix:  ""
  },
  stats: {
    title:   "Dataset Statistics",
    hint:    "Click the button to view score distributions from the ASAP dataset.",
    btn:     "Show Statistics",
    input:   "none",
    loading: "Loading statistics...",
    prefix:  "Show me the score distribution statistics for the dataset."
  }
};

var currentMode = "grading";

// ── Tab switching ─────────────────────────────────────────────────────────────
function setMode(mode) {
  currentMode = mode;
  var cfg = MODES[mode];

  // Update active tab highlight
  ["grading","scoring","advice","stats"].forEach(function(m) {
    var el = document.getElementById("tab-" + m);
    if (el) el.className = "tab" + (m === mode ? " active" : "");
  });

  // Update title, hint, button label
  document.getElementById("inputTitle").textContent = cfg.title;
  document.getElementById("inputHint").textContent  = cfg.hint;
  document.getElementById("submitBtn").textContent  = cfg.btn;

  // Show/hide inputs
  document.getElementById("essayInput").style.display = (cfg.input === "essay") ? "block" : "none";
  document.getElementById("queryInput").style.display = (cfg.input === "query") ? "block" : "none";

  // Hide result panel when switching
  document.getElementById("result").style.display = "none";
  hideErr();
}

// ── Submit ────────────────────────────────────────────────────────────────────
function submitQuery() {
  var cfg = MODES[currentMode];
  var userMessage = "";

  if (cfg.input === "essay") {
    var txt = document.getElementById("essayInput").value.trim();
    if (!txt) { showErr("Please paste an essay."); return; }
    if (txt.split(/\s+/).length < 10) { showErr("Essay is too short (minimum 10 words)."); return; }
    userMessage = cfg.prefix + txt;
  } else if (cfg.input === "query") {
    var q = document.getElementById("queryInput").value.trim();
    if (!q) { showErr("Please type a question."); return; }
    userMessage = cfg.prefix + q;
  } else {
    // stats mode — fixed message
    userMessage = cfg.prefix;
  }

  hideErr();
  document.getElementById("result").style.display  = "none";
  document.getElementById("loading").style.display = "block";
  document.getElementById("loadingMsg").textContent = cfg.loading;
  document.getElementById("submitBtn").disabled     = true;

  fetch("/grade", {
    method:  "POST",
    headers: {"Content-Type": "application/json"},
    body:    JSON.stringify({user_message: userMessage})
  })
  .then(function(r) { return r.json(); })
  .then(function(d) {
    document.getElementById("loading").style.display = "none";
    document.getElementById("submitBtn").disabled    = false;
    if (d.error) { showErr(d.error); return; }
    renderResult(d);
  })
  .catch(function(e) {
    document.getElementById("loading").style.display = "none";
    document.getElementById("submitBtn").disabled    = false;
    showErr("Network error: " + e);
  });
}

// ── Render — only show sections matching agent_type ───────────────────────────
var ALL_SECS = ["secScore","secSummary","secStrengths","secImprove","secNextSteps","secAdvice","secStats"];

function show(id) { document.getElementById(id).style.display = "block"; }
function hide(id) { document.getElementById(id).style.display = "none";  }

function renderResult(d) {
  // Hide everything first
  ALL_SECS.forEach(hide);

  var atype = d.agent_type || "grading";

  // Agent badge
  var labels = {scoring:"Scoring Agent", advice:"Advice Agent", stats:"Statistics Agent", grading:"Grading Agent"};
  document.getElementById("agentBadge").textContent = labels[atype] || atype;

  // Score card — scoring + grading
  if (atype === "scoring" || atype === "grading") {
    show("secScore");
    document.getElementById("sNum").textContent   = d.score != null ? d.score.toFixed(2) : "-";
    document.getElementById("sRaw").textContent   = d.raw_score ? "approx " + d.raw_score + " / 6" : "";
    document.getElementById("sModel").textContent = d.model_used || "";
    setBand(d.band || getBand(d.score));
  }

  // Grading sections
  if (atype === "grading") {
    if (d.summary)      { show("secSummary");   document.getElementById("fSum").textContent = d.summary; }
    if (d.strengths)    { show("secStrengths"); document.getElementById("fStr").textContent = d.strengths; }
    if (d.improvements) { show("secImprove");   document.getElementById("fImp").textContent = d.improvements; }
    if (d.next_steps && d.next_steps.length) {
      show("secNextSteps");
      document.getElementById("fNxt").textContent = d.next_steps.map(function(s,i){ return (i+1)+". "+s; }).join("\n");
    }
    if (d.stats) { fillStats(d.stats); show("secStats"); }
  }

  // Advice sections
  if (atype === "advice") {
    show("secAdvice");
    var ul = document.getElementById("fCriteria");
    ul.innerHTML = "";
    (d.rubric_criteria || []).forEach(function(c) {
      var li = document.createElement("li");
      li.textContent = c;
      ul.appendChild(li);
    });
    document.getElementById("fAdvice").textContent = d.advice_text || "";
  }

  // Stats section
  if (atype === "stats" && d.stats) {
    fillStats(d.stats);
    show("secStats");
  }

  // Chips
  document.getElementById("cWords").textContent = (d.word_count || "-") + " words";
  document.getElementById("cModel").textContent = "Model: " + (d.model_used || "Linear Regression");

  document.getElementById("result").style.display = "block";
  document.getElementById("result").scrollIntoView({behavior:"smooth"});
}

function fillStats(stats) {
  var grid = document.getElementById("fStatsGrid");
  grid.innerHTML = "";
  if (stats.total_essays != null) grid.innerHTML += "<div class='stat-pill'><strong>"+stats.total_essays+"</strong>Total Essays</div>";
  if (stats.mean_score   != null) grid.innerHTML += "<div class='stat-pill'><strong>"+stats.mean_score+"</strong>Mean Score</div>";
  if (stats.median_score != null) grid.innerHTML += "<div class='stat-pill'><strong>"+stats.median_score+"</strong>Median Score</div>";
  var tbody = document.getElementById("fStatsBands");
  tbody.innerHTML = "";
  var bands = stats.score_bands || {};
  Object.keys(bands).forEach(function(b) {
    tbody.innerHTML += "<tr><td>"+b+"</td><td>"+bands[b]+"</td></tr>";
  });
  document.getElementById("fStatsInterp").textContent = stats.interpretation || "";
}

// ── Helpers ───────────────────────────────────────────────────────────────────
var BAND_COLORS = {
  Emerging  : {bg:"#fee2e2", color:"#dc2626"},
  Beginning : {bg:"#fef3c7", color:"#d97706"},
  Developing: {bg:"#dbeafe", color:"#2563eb"},
  Proficient: {bg:"#d1fae5", color:"#059669"},
  Exemplary : {bg:"#ede9fe", color:"#7c3aed"}
};
function setBand(band) {
  var el = document.getElementById("sBand");
  el.textContent = band;
  var c = BAND_COLORS[band] || {bg:"#e5e7eb", color:"#374151"};
  el.style.background = c.bg;
  el.style.color      = c.color;
}
function getBand(s) {
  if (s <= 2) return "Emerging";
  if (s <= 4) return "Beginning";
  if (s <= 6) return "Developing";
  if (s <= 8) return "Proficient";
  return "Exemplary";
}
function showErr(msg) {
  var b = document.getElementById("errBox");
  b.textContent   = msg;
  b.style.display = "block";
}
function hideErr() { document.getElementById("errBox").style.display = "none"; }
function clearAll() {
  document.getElementById("essayInput").value      = "";
  document.getElementById("queryInput").value      = "";
  document.getElementById("result").style.display  = "none";
  hideErr();
}

// ── Wire up everything once DOM is ready ─────────────────────────────────────
document.getElementById("tab-grading").onclick = function() { setMode("grading"); };
document.getElementById("tab-scoring").onclick = function() { setMode("scoring"); };
document.getElementById("tab-advice" ).onclick = function() { setMode("advice");  };
document.getElementById("tab-stats"  ).onclick = function() { setMode("stats");   };
document.getElementById("submitBtn"  ).onclick = submitQuery;
document.getElementById("clearBtn"   ).onclick = clearAll;
</script>
</body>
</html>"""


if __name__ == "__main__":
    print("=" * 55)
    print("  Essay Grading Web Interface (Multi-Agent v2)")
    print("  IT9204 | Student 202507410 Faiza Malik")
    print("  Open: http://127.0.0.1:5000")
    print("  Requires: python main.py  (ML server on :8000)")
    print("=" * 55)
    app.run(debug=False, port=5000)
    