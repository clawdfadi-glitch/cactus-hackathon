# 🏗️ Hackathon Plan — Cactus x DeepMind Hybrid Router
*Last updated: 2026-02-21 12:05 PST*

## ⏰ Timeline
- **Now → 5:00 PM**: Build & iterate
- **5:30 PM**: Submissions due
- **6:00 PM**: Preliminary judging
- **7:00 PM**: Final demos (if we make top 10)

## 🎯 What We're Building
**"Atomic Router"** — a hybrid routing strategy that makes a tiny 270M model (FunctionGemma) punch above its weight by decomposing complex requests into simple ones it can handle.

### The Problem
- FunctionGemma (local): fast (~50ms), great at single tool calls, bad at multi-tool
- Gemini (cloud): smart, handles everything, but slow (~300-500ms network)
- **Goal**: maximize local execution while maintaining correctness

### The Strategy
```
User request → Is it multi-call? 
  YES → Split into atomic parts → FunctionGemma each → Validate → Cloud only for failures
  NO  → FunctionGemma once → Validate → Cloud fallback if bad
```

## 📊 Scoring (how we're ranked)
| Weight | Metric | What it means |
|--------|--------|---------------|
| 60% | F1 accuracy | Did we pick the right function + args? |
| 15% | Speed | Faster = better (baseline 500ms) |
| 25% | On-device ratio | More local = better |

Difficulty weights: Easy 20%, Medium 30%, Hard 50%

## ✅ Status

### Done
- [x] Repo created: https://github.com/clawdfadi-glitch/cactus-hackathon
- [x] Cactus cloned + built (Python bindings)
- [x] cmake, python 3.12 installed
- [x] Atomic Router v1 written in main.py
- [x] Gemini API key configured (existing)

### In Progress
- [ ] FunctionGemma model download (HF token provided)
- [ ] Cactus auth (need API key from https://cactuscompute.com/dashboard/api-keys)
- [ ] Run benchmark.py to get baseline score
- [ ] Iterate on strategy

### TODO
- [ ] Test & tune confidence threshold (currently 0.3)
- [ ] Improve multi-call detection heuristics
- [ ] Better atomic splitting (handle edge cases)
- [ ] Consider: prompt engineering for FunctionGemma
- [ ] Consider: parallel local+cloud for latency optimization
- [ ] Build end-to-end demo (qualitative judging rubric 2)
- [ ] Consider voice-to-action demo (qualitative rubric 3)
- [ ] Submit: `python submit.py --team "YourTeamName" --location "SF"`

## 🔧 Who's Doing What

### Kai (remote on Mac mini)
- Environment setup & builds
- Code implementation in main.py
- Running benchmarks & iterating
- Research & ideas

### Fadi (at hackathon in SF)
- Local testing on his machine
- Hackathon comms & networking
- Demo prep & presentation
- Strategic direction

## 💡 Ideas Backlog
1. **Atomic decomposition** ✅ (implemented v1)
2. **Schema validation layer** ✅ (implemented v1)
3. **Model handle reuse** ✅ (implemented — avoid re-init overhead)
4. **Prompt engineering** — optimize system prompt for FunctionGemma
5. **Confidence calibration** — map raw confidence to routing decisions
6. **Tool description optimization** — rewrite tool descriptions to help small model
7. **Two-pass verification** — run FunctionGemma twice with different prompts
8. **Selective cloud** — only cloud the specific failed sub-task, not everything
9. **Meta-layer orchestration** — Python logic as the "brain", model as the "hands"
10. **Voice-to-action** — use cactus_transcribe for bonus qualitative points

## 📝 Notes
- Final eval uses HIDDEN test cases (harder than benchmark.py)
- Can submit to leaderboard 1x per hour
- Top 10 on leaderboard → qualitative judging round
- Cactus API key needed for submissions

## 🔗 Links
- Hackathon: https://sf.aitinkerers.org/hackathons/h_DRGnrtIWaG8/
- Repo template: https://github.com/cactus-compute/functiongemma-hackathon
- Leaderboard: https://cactusevals.ngrok.app
- Cactus keys: https://cactuscompute.com/dashboard/api-keys
- GCP credits (SF): https://trygcp.dev/claim/cactus-x-gdm-hackathon-sf
