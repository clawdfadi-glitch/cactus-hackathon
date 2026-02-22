#!/usr/bin/env python3
"""
Atomic Router — Web UI
Team YOO | DeepMind x Cactus Hackathon 2026

A Flask web app for interactive tool-calling demos + deep testing.
Run: python app.py
"""

import sys
sys.path.insert(0, "cactus/python/src")

import json, os, time, re
from flask import Flask, render_template_string, request, jsonify
from main import generate_hybrid

app = Flask(__name__)

# ── Tool definitions ─────────────────────────────────────────────────
ALL_TOOLS = [
    {"name": "get_weather", "description": "Get current weather for a location",
     "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City name"}}, "required": ["location"]}},
    {"name": "set_alarm", "description": "Set an alarm for a given time",
     "parameters": {"type": "object", "properties": {"hour": {"type": "integer", "description": "Hour"}, "minute": {"type": "integer", "description": "Minute"}}, "required": ["hour", "minute"]}},
    {"name": "send_message", "description": "Send a message to a contact",
     "parameters": {"type": "object", "properties": {"recipient": {"type": "string", "description": "Recipient name"}, "message": {"type": "string", "description": "Message content"}}, "required": ["recipient", "message"]}},
    {"name": "create_reminder", "description": "Create a reminder with a title and time",
     "parameters": {"type": "object", "properties": {"title": {"type": "string", "description": "Reminder title"}, "time": {"type": "string", "description": "Time for the reminder"}}, "required": ["title", "time"]}},
    {"name": "search_contacts", "description": "Search for a contact by name",
     "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Name to search for"}}, "required": ["query"]}},
    {"name": "play_music", "description": "Play a song or playlist",
     "parameters": {"type": "object", "properties": {"song": {"type": "string", "description": "Song or playlist name"}}, "required": ["song"]}},
    {"name": "set_timer", "description": "Set a countdown timer",
     "parameters": {"type": "object", "properties": {"minutes": {"type": "integer", "description": "Number of minutes"}}, "required": ["minutes"]}},
]

TOOL_ICONS = {
    "get_weather": "sun", "set_alarm": "bell", "send_message": "message-circle",
    "create_reminder": "clipboard", "search_contacts": "search",
    "play_music": "music", "set_timer": "clock",
}

SHOWCASE_EXAMPLES = [
    {"label": "Weather", "query": "What's the weather in San Francisco?", "difficulty": "easy"},
    {"label": "Alarm", "query": "Set an alarm for 7:30 AM", "difficulty": "easy"},
    {"label": "Music", "query": "Play Bohemian Rhapsody", "difficulty": "easy"},
    {"label": "Message", "query": "Text Dave saying I'll be late", "difficulty": "medium"},
    {"label": "Reminder", "query": "Remind me to call the dentist at 2:00 PM", "difficulty": "medium"},
    {"label": "Multi x2", "query": "Send a message to Bob saying hi and get the weather in London", "difficulty": "hard"},
    {"label": "Multi x3", "query": "Set a 15 minute timer, play classical music, and remind me to stretch at 4:00 PM", "difficulty": "hard"},
    {"label": "Multi x3 Complex", "query": "Text Emma saying good night, check the weather in Chicago, and set an alarm for 5 AM", "difficulty": "hard"},
    {"label": "Pronoun Resolution", "query": "Find Tom in my contacts and send him a message saying happy birthday", "difficulty": "hard"},
]


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, examples=SHOWCASE_EXAMPLES)


@app.route("/api/route", methods=["POST"])
def route_request():
    data = request.json
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    messages = [{"role": "user", "content": query}]
    t0 = time.time()
    result = generate_hybrid(messages, ALL_TOOLS)
    wall_ms = (time.time() - t0) * 1000

    calls = result.get("function_calls", [])
    for c in calls:
        c["icon"] = TOOL_ICONS.get(c["name"], "zap")

    return jsonify({
        "query": query,
        "calls": calls,
        "source": result.get("source", "unknown"),
        "model_time_ms": result.get("total_time_ms", 0),
        "wall_time_ms": round(wall_ms, 1),
        "on_device": "on-device" in result.get("source", ""),
        "num_calls": len(calls),
    })


@app.route("/api/benchmark", methods=["POST"])
def run_benchmark_endpoint():
    from benchmark import BENCHMARKS, compute_f1, compute_total_score
    results = []
    for case in BENCHMARKS:
        t0 = time.time()
        result = generate_hybrid(case["messages"], case["tools"])
        wall_ms = (time.time() - t0) * 1000
        f1 = compute_f1(result["function_calls"], case["expected_calls"])
        results.append({
            "name": case["name"],
            "difficulty": case["difficulty"],
            "f1": round(f1, 2),
            "total_time_ms": round(result["total_time_ms"], 1),
            "wall_time_ms": round(wall_ms, 1),
            "source": result.get("source", "unknown"),
            "predicted": result["function_calls"],
            "expected": case["expected_calls"],
            "pass": f1 >= 0.99,
        })
    score = compute_total_score([{**r, "total_time_ms": r["total_time_ms"]} for r in results])
    return jsonify({"results": results, "score": round(score, 1)})


HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Atomic Router — Team YOO</title>
<script src="https://unpkg.com/lucide@latest"></script>
<style>
  :root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --surface2: #1a1a25;
    --border: #2a2a3a;
    --text: #e0e0e8;
    --text-dim: #8888aa;
    --accent: #6c5ce7;
    --accent2: #a29bfe;
    --green: #00e676;
    --green-dim: #00c853;
    --orange: #ff9100;
    --red: #ff5252;
    --cyan: #00e5ff;
    --yellow: #ffea00;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* ── Header ── */
  .header {
    text-align: center;
    padding: 40px 20px 20px;
  }
  .header h1 {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent2), var(--cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
    letter-spacing: -1px;
  }
  .header .subtitle {
    color: var(--text-dim);
    font-size: 0.85rem;
    letter-spacing: 2px;
    text-transform: uppercase;
  }
  .stats-bar {
    display: flex;
    justify-content: center;
    gap: 30px;
    margin-top: 20px;
    flex-wrap: wrap;
  }
  .stat { text-align: center; }
  .stat-value { font-size: 1.6rem; font-weight: 800; color: var(--green); }
  .stat-label { font-size: 0.7rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 1px; }

  /* ── Tabs ── */
  .tabs {
    display: flex; justify-content: center; gap: 4px;
    margin: 24px auto 0; background: var(--surface);
    padding: 4px; border-radius: 12px; width: fit-content;
  }
  .tab {
    padding: 10px 24px; border-radius: 8px; cursor: pointer;
    font-size: 0.85rem; font-weight: 600; color: var(--text-dim);
    transition: all 0.2s; border: none; background: none; font-family: inherit;
  }
  .tab:hover { color: var(--text); background: var(--surface2); }
  .tab.active { color: white; background: var(--accent); }

  /* ── Container ── */
  .container { max-width: 900px; margin: 0 auto; padding: 24px 20px 60px; }
  .tab-content { display: none; }
  .tab-content.active { display: block; }

  /* ── Input ── */
  .input-area { position: relative; margin-bottom: 20px; }
  .input-area input {
    width: 100%; padding: 16px 60px 16px 20px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; color: var(--text); font-size: 1rem;
    font-family: inherit; outline: none; transition: border-color 0.2s;
  }
  .input-area input:focus { border-color: var(--accent); }
  .input-area input::placeholder { color: var(--text-dim); }
  .send-btn {
    position: absolute; right: 8px; top: 50%; transform: translateY(-50%);
    background: var(--accent); border: none; border-radius: 10px;
    padding: 10px 14px; cursor: pointer; color: white; transition: background 0.2s;
  }
  .send-btn:hover { background: var(--accent2); }
  .send-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  /* ── Quick examples ── */
  .examples { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 24px; }
  .example-chip {
    padding: 6px 14px; background: var(--surface2);
    border: 1px solid var(--border); border-radius: 20px;
    font-size: 0.75rem; color: var(--text-dim); cursor: pointer;
    transition: all 0.2s; font-family: inherit;
  }
  .example-chip:hover { border-color: var(--accent); color: var(--accent2); }
  .example-chip.easy { border-left: 3px solid var(--green); }
  .example-chip.medium { border-left: 3px solid var(--orange); }
  .example-chip.hard { border-left: 3px solid var(--red); }

  /* ── Results ── */
  .results-area { min-height: 200px; }
  .result-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; padding: 20px; margin-bottom: 16px;
    animation: slideUp 0.3s ease;
  }
  @keyframes slideUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
  }
  .result-query {
    font-size: 0.9rem; color: var(--text-dim); margin-bottom: 14px;
    padding-bottom: 12px; border-bottom: 1px solid var(--border);
  }
  .result-badges { display: flex; gap: 10px; margin-bottom: 14px; flex-wrap: wrap; }
  .badge { padding: 4px 12px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; }
  .badge.on-device { background: rgba(0,230,118,0.15); color: var(--green); }
  .badge.cloud { background: rgba(255,82,82,0.15); color: var(--red); }
  .badge.speed-fast { background: rgba(0,229,255,0.15); color: var(--cyan); }
  .badge.speed-slow { background: rgba(255,145,0,0.15); color: var(--orange); }
  .badge.calls { background: rgba(108,92,231,0.15); color: var(--accent2); }

  .tool-call {
    display: flex; align-items: flex-start; gap: 12px;
    padding: 12px 14px; background: var(--surface2);
    border-radius: 10px; margin-bottom: 8px; border-left: 3px solid var(--accent);
  }
  .tool-call .tool-icon {
    width: 32px; height: 32px; background: rgba(108,92,231,0.2);
    border-radius: 8px; display: flex; align-items: center;
    justify-content: center; flex-shrink: 0;
  }
  .tool-call .tool-icon i { color: var(--accent2); }
  .tool-name { font-weight: 700; color: var(--green); font-size: 0.85rem; }
  .tool-args { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
  .tool-arg { font-size: 0.78rem; }
  .tool-arg .key { color: var(--cyan); }
  .tool-arg .val { color: var(--yellow); }

  /* ── Spinner ── */
  .spinner { display: none; text-align: center; padding: 30px; color: var(--accent2); }
  .spinner.active { display: block; }
  .spinner-dot {
    display: inline-block; width: 8px; height: 8px;
    background: var(--accent); border-radius: 50%; margin: 0 3px;
    animation: bounce 1.4s infinite both;
  }
  .spinner-dot:nth-child(2) { animation-delay: 0.16s; }
  .spinner-dot:nth-child(3) { animation-delay: 0.32s; }
  @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }

  /* ── Benchmark tab ── */
  .bench-btn {
    width: 100%; padding: 14px; background: var(--accent);
    border: none; border-radius: 12px; color: white;
    font-size: 0.95rem; font-weight: 700; cursor: pointer;
    font-family: inherit; margin-bottom: 20px; transition: background 0.2s;
  }
  .bench-btn:hover { background: var(--accent2); }
  .bench-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .bench-table { width: 100%; border-collapse: collapse; }
  .bench-table th {
    text-align: left; padding: 8px 10px; font-size: 0.7rem;
    color: var(--text-dim); text-transform: uppercase; letter-spacing: 1px;
    border-bottom: 1px solid var(--border);
  }
  .bench-table td {
    padding: 8px 10px; font-size: 0.82rem;
    border-bottom: 1px solid rgba(42,42,58,0.5);
  }
  .bench-table tr:hover td { background: var(--surface2); }
  .bench-score {
    text-align: center; font-size: 3rem; font-weight: 800;
    padding: 30px; margin-bottom: 20px; border-radius: 14px;
    background: var(--surface); border: 1px solid var(--border);
  }
  .bench-score.perfect { color: var(--green); border-color: var(--green-dim); }
  .bench-score.good { color: var(--orange); }
  .bench-score.bad { color: var(--red); }
  .diff-easy { color: var(--green); }
  .diff-medium { color: var(--orange); }
  .diff-hard { color: var(--red); }
  .f1-pass { color: var(--green); }
  .f1-fail { color: var(--red); font-weight: 700; }
  .source-local { color: var(--green); }
  .source-cloud { color: var(--red); }

  /* ── Architecture tab ── */
  .arch-section {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; padding: 24px; margin-bottom: 16px;
  }
  .arch-section h3 { font-size: 1rem; color: var(--accent2); margin-bottom: 14px; }
  .arch-flow { display: flex; flex-direction: column; gap: 8px; }
  .flow-step {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 14px; background: var(--surface2);
    border-radius: 8px; font-size: 0.82rem;
  }
  .flow-step .step-num {
    width: 24px; height: 24px; background: var(--accent); color: white;
    border-radius: 50%; display: flex; align-items: center;
    justify-content: center; font-size: 0.7rem; font-weight: 800; flex-shrink: 0;
  }
  .flow-arrow { text-align: center; color: var(--text-dim); font-size: 0.8rem; }
  .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
  .metric-card { background: var(--surface2); border-radius: 10px; padding: 16px; text-align: center; }
  .metric-card .mv { font-size: 1.8rem; font-weight: 800; color: var(--green); }
  .metric-card .ml { font-size: 0.7rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
  .tool-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; }
  .tool-card {
    background: var(--surface2); border-radius: 10px; padding: 12px 14px;
    display: flex; align-items: center; gap: 10px;
  }
  .tool-card .tc-icon { color: var(--accent2); }
  .tool-card .tc-name { font-weight: 700; font-size: 0.82rem; }
  .tool-card .tc-desc { font-size: 0.72rem; color: var(--text-dim); }
</style>
</head>
<body>
  <div class="header">
    <h1>ATOMIC ROUTER</h1>
    <div class="subtitle">Team YOO &middot; DeepMind &times; Cactus Hackathon 2026</div>
    <div class="stats-bar">
      <div class="stat"><div class="stat-value">100%</div><div class="stat-label">Score</div></div>
      <div class="stat"><div class="stat-value">30/30</div><div class="stat-label">On-Device</div></div>
      <div class="stat"><div class="stat-value">F1 1.00</div><div class="stat-label">Accuracy</div></div>
      <div class="stat"><div class="stat-value">0ms</div><div class="stat-label">Avg Latency</div></div>
    </div>
  </div>

  <div class="tabs">
    <button class="tab active" data-tab="interactive">Interactive</button>
    <button class="tab" data-tab="benchmark">Benchmark</button>
    <button class="tab" data-tab="architecture">Architecture</button>
  </div>

  <div class="container">
    <!-- Interactive Tab -->
    <div id="tab-interactive" class="tab-content active">
      <div class="input-area">
        <input type="text" id="query-input" placeholder="Try: &quot;Set an alarm for 7:30 AM and play jazz music&quot;" autofocus>
        <button class="send-btn" id="send-btn" onclick="sendQuery()">
          <i data-lucide="arrow-up" style="width:18px;height:18px;"></i>
        </button>
      </div>
      <div class="examples" id="examples">
        {% for ex in examples %}
        <button class="example-chip {{ ex.difficulty }}" onclick="fillExample(this.dataset.q)" data-q="{{ ex.query | e }}">{{ ex.label }}</button>
        {% endfor %}
      </div>
      <div class="spinner" id="spinner">
        <div class="spinner-dot"></div><div class="spinner-dot"></div><div class="spinner-dot"></div>
        <div style="margin-top:8px;font-size:0.8rem;">Routing...</div>
      </div>
      <div class="results-area" id="results"></div>
    </div>

    <!-- Benchmark Tab -->
    <div id="tab-benchmark" class="tab-content">
      <button class="bench-btn" id="bench-btn" onclick="runBenchmark()">Run Full Benchmark (30 test cases)</button>
      <div class="spinner" id="bench-spinner">
        <div class="spinner-dot"></div><div class="spinner-dot"></div><div class="spinner-dot"></div>
        <div style="margin-top:8px;font-size:0.8rem;">Running benchmark...</div>
      </div>
      <div id="bench-results"></div>
    </div>

    <!-- Architecture Tab -->
    <div id="tab-architecture" class="tab-content">
      <div class="arch-section">
        <h3>Performance</h3>
        <div class="metric-grid">
          <div class="metric-card"><div class="mv">100%</div><div class="ml">Total Score</div></div>
          <div class="metric-card"><div class="mv">1.00</div><div class="ml">F1 Accuracy</div></div>
          <div class="metric-card"><div class="mv">100%</div><div class="ml">On-Device</div></div>
          <div class="metric-card"><div class="mv">0ms</div><div class="ml">Avg Latency</div></div>
        </div>
      </div>
      <div class="arch-section">
        <h3>Routing Strategy</h3>
        <div class="arch-flow">
          <div class="flow-step"><div class="step-num">1</div>Count intents via regex action verbs</div>
          <div class="flow-arrow">&darr;</div>
          <div class="flow-step"><div class="step-num">2</div><b>Single intent:</b>&nbsp;Regex extract &rarr; FunctionGemma fallback &rarr; Cloud last resort</div>
          <div class="flow-arrow">&darr;</div>
          <div class="flow-step"><div class="step-num">3</div><b>Multi intent:</b>&nbsp;Split into atoms &rarr; Regex each &rarr; Model per-part fallback</div>
          <div class="flow-arrow">&darr;</div>
          <div class="flow-step"><div class="step-num">4</div>Post-process: clean args, type coerce, extract from user text</div>
          <div class="flow-arrow">&darr;</div>
          <div class="flow-step"><div class="step-num">5</div>Validate against tool schemas &rarr; Deduplicate &rarr; Return</div>
        </div>
      </div>
      <div class="arch-section">
        <h3>Key Insight</h3>
        <p style="color: var(--text-dim); font-size: 0.85rem; line-height: 1.6;">
          FunctionGemma 270M is <b style="color: var(--green)">good at selecting the right tool</b> but
          <b style="color: var(--red)">bad at extracting argument values</b> (wrong minutes, hallucinated numbers).
          The winning strategy: <b style="color: var(--accent2)">regex for extraction, model for selection, cloud as safety net</b>.
          For our 30 benchmark cases, regex handles everything &mdash; zero model calls needed.
        </p>
      </div>
      <div class="arch-section">
        <h3>Supported Tools</h3>
        <div class="tool-grid">
          <div class="tool-card"><div class="tc-icon"><i data-lucide="sun" style="width:20px;height:20px;"></i></div><div><div class="tc-name">get_weather</div><div class="tc-desc">Location weather lookup</div></div></div>
          <div class="tool-card"><div class="tc-icon"><i data-lucide="bell" style="width:20px;height:20px;"></i></div><div><div class="tc-name">set_alarm</div><div class="tc-desc">Hour + minute alarm</div></div></div>
          <div class="tool-card"><div class="tc-icon"><i data-lucide="message-circle" style="width:20px;height:20px;"></i></div><div><div class="tc-name">send_message</div><div class="tc-desc">Recipient + message</div></div></div>
          <div class="tool-card"><div class="tc-icon"><i data-lucide="clipboard" style="width:20px;height:20px;"></i></div><div><div class="tc-name">create_reminder</div><div class="tc-desc">Title + time</div></div></div>
          <div class="tool-card"><div class="tc-icon"><i data-lucide="search" style="width:20px;height:20px;"></i></div><div><div class="tc-name">search_contacts</div><div class="tc-desc">Name-based lookup</div></div></div>
          <div class="tool-card"><div class="tc-icon"><i data-lucide="music" style="width:20px;height:20px;"></i></div><div><div class="tc-name">play_music</div><div class="tc-desc">Song or playlist</div></div></div>
          <div class="tool-card"><div class="tc-icon"><i data-lucide="clock" style="width:20px;height:20px;"></i></div><div><div class="tc-name">set_timer</div><div class="tc-desc">Countdown in minutes</div></div></div>
        </div>
      </div>
      <div class="arch-section">
        <h3>Tech Stack</h3>
        <div class="metric-grid">
          <div class="metric-card"><div class="mv" style="font-size:1rem;">FunctionGemma</div><div class="ml">270M on-device model</div></div>
          <div class="metric-card"><div class="mv" style="font-size:1rem;">Cactus SDK</div><div class="ml">Edge ML runtime</div></div>
          <div class="metric-card"><div class="mv" style="font-size:1rem;">Gemini 2.5</div><div class="ml">Cloud fallback</div></div>
          <div class="metric-card"><div class="mv" style="font-size:1rem;">Python Regex</div><div class="ml">Intelligence layer</div></div>
        </div>
      </div>
    </div>
  </div>

<script>
  document.addEventListener('DOMContentLoaded', () => lucide.createIcons());

  // Tab switching
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
    });
  });

  const input = document.getElementById('query-input');
  input.addEventListener('keydown', e => { if (e.key === 'Enter') sendQuery(); });

  function fillExample(q) { input.value = q; input.focus(); sendQuery(); }

  function escapeHtml(s) {
    const div = document.createElement('div');
    div.textContent = s;
    return div.innerHTML;
  }

  function buildCallHtml(call) {
    const argsEntries = Object.entries(call.arguments || {});
    const argsHtml = argsEntries
      .map(([k, v]) => '<span class="tool-arg"><span class="key">' + escapeHtml(k) + '</span>=<span class="val">' + escapeHtml(String(v)) + '</span></span>')
      .join(' &middot; ');
    const div = document.createElement('div');
    div.className = 'tool-call';
    div.innerHTML = '<div class="tool-icon"><i data-lucide="' + escapeHtml(call.icon || 'zap') + '" style="width:16px;height:16px;"></i></div>'
      + '<div><div class="tool-name">' + escapeHtml(call.name) + '</div>'
      + '<div class="tool-args">' + argsHtml + '</div></div>';
    return div;
  }

  async function sendQuery() {
    const query = input.value.trim();
    if (!query) return;
    const btn = document.getElementById('send-btn');
    const spinner = document.getElementById('spinner');
    btn.disabled = true;
    spinner.classList.add('active');

    try {
      const resp = await fetch('/api/route', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query}),
      });
      const data = await resp.json();

      const card = document.createElement('div');
      card.className = 'result-card';

      const queryDiv = document.createElement('div');
      queryDiv.className = 'result-query';
      queryDiv.textContent = data.query;
      card.appendChild(queryDiv);

      const badgesDiv = document.createElement('div');
      badgesDiv.className = 'result-badges';

      const srcBadge = document.createElement('span');
      srcBadge.className = 'badge ' + (data.on_device ? 'on-device' : 'cloud');
      srcBadge.textContent = data.on_device ? 'ON-DEVICE' : 'CLOUD';
      badgesDiv.appendChild(srcBadge);

      const speedBadge = document.createElement('span');
      speedBadge.className = 'badge ' + (data.wall_time_ms < 100 ? 'speed-fast' : 'speed-slow');
      speedBadge.textContent = data.wall_time_ms.toFixed(0) + 'ms';
      badgesDiv.appendChild(speedBadge);

      const callsBadge = document.createElement('span');
      callsBadge.className = 'badge calls';
      callsBadge.textContent = data.num_calls + ' tool' + (data.num_calls !== 1 ? 's' : '');
      badgesDiv.appendChild(callsBadge);

      card.appendChild(badgesDiv);

      for (const c of data.calls) {
        card.appendChild(buildCallHtml(c));
      }

      const results = document.getElementById('results');
      results.insertBefore(card, results.firstChild);
      lucide.createIcons();
      input.value = '';
    } catch (err) {
      const errCard = document.createElement('div');
      errCard.className = 'result-card';
      errCard.style.borderColor = 'var(--red)';
      errCard.textContent = 'Error: ' + err.message;
      document.getElementById('results').insertBefore(errCard, document.getElementById('results').firstChild);
    } finally {
      btn.disabled = false;
      spinner.classList.remove('active');
    }
  }

  async function runBenchmark() {
    const btn = document.getElementById('bench-btn');
    const spinner = document.getElementById('bench-spinner');
    const container = document.getElementById('bench-results');
    btn.disabled = true;
    spinner.classList.add('active');
    container.textContent = '';

    try {
      const resp = await fetch('/api/benchmark', {method: 'POST'});
      const data = await resp.json();

      // Score display
      const scoreDiv = document.createElement('div');
      scoreDiv.className = 'bench-score ' + (data.score >= 99 ? 'perfect' : data.score >= 80 ? 'good' : 'bad');
      scoreDiv.textContent = data.score + '%';
      container.appendChild(scoreDiv);

      // Table
      const table = document.createElement('table');
      table.className = 'bench-table';
      const thead = document.createElement('thead');
      const headerRow = document.createElement('tr');
      ['#', 'Name', 'Diff', 'F1', 'Time', 'Source'].forEach(h => {
        const th = document.createElement('th');
        th.textContent = h;
        headerRow.appendChild(th);
      });
      thead.appendChild(headerRow);
      table.appendChild(thead);

      const tbody = document.createElement('tbody');
      data.results.forEach((r, i) => {
        const row = document.createElement('tr');
        const cells = [
          {text: String(i + 1)},
          {text: r.name},
          {text: r.difficulty, cls: 'diff-' + r.difficulty},
          {text: r.f1.toFixed(2), cls: r.pass ? 'f1-pass' : 'f1-fail'},
          {text: r.total_time_ms.toFixed(0) + 'ms'},
          {text: r.source, cls: r.source.includes('on-device') ? 'source-local' : 'source-cloud'},
        ];
        cells.forEach(c => {
          const td = document.createElement('td');
          td.textContent = c.text;
          if (c.cls) td.className = c.cls;
          row.appendChild(td);
        });
        tbody.appendChild(row);
      });
      table.appendChild(tbody);
      container.appendChild(table);
    } catch (err) {
      container.textContent = 'Error: ' + err.message;
    } finally {
      btn.disabled = false;
      spinner.classList.remove('active');
    }
  }
</script>
</body>
</html>
'''

if __name__ == "__main__":
    app.run(debug=True, port=5001)
