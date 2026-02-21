# 🏗️ Hackathon Plan — Cactus x DeepMind Hybrid Router
*Last updated: 2026-02-21 12:30 PST*

## ⏰ Timeline
- **Now → 5:00 PM**: Build & iterate
- **5:30 PM**: Submissions due
- **6:00 PM**: Preliminary judging
- **7:00 PM**: Final demos (if we make top 10)

---

## 🎯 What We're Building
**"Atomic Router"** — a hybrid routing strategy that makes a tiny 270M model (FunctionGemma) punch above its weight by decomposing complex requests into simple ones it can handle.

### The Problem
- FunctionGemma (local): fast (~50ms), great at single tool calls, bad at multi-tool
- Gemini (cloud): smart, handles everything, but slow (~1500-2000ms network)
- **Goal**: maximize local execution while maintaining correctness

### The Strategy
```
User request → Count intents (regex action verbs)
  MULTI → Split into atomic parts → FunctionGemma each → Validate → Cloud only for failures  
  SINGLE → FunctionGemma once → Validate → Cloud fallback if bad
  Always → Clean args (strip punctuation) → Deduplicate
```

---

## 📊 Scoring
| Weight | Metric | What it means |
|--------|--------|---------------|
| 60% | F1 accuracy | Correct function + args |
| 15% | Speed | Faster = better (baseline 500ms) |
| 25% | On-device ratio | More local = better |

Difficulty weights: Easy 20%, Medium 30%, Hard 50%

---

## 📈 Score History
| Version | Score | F1 | On-device | Notes |
|---------|-------|-----|-----------|-------|
| Baseline (template) | ~30% | low | 0% | Always cloud |
| v1 Atomic Router | 50.6% | 0.77 | 30% | First attempt |
| v2 + arg cleaning | **59.2%** | **0.91** | 30% | Fixed cloud F1 bugs |

### Current Failures (v2)
- `alarm_10am` (easy): FunctionGemma returns minute=9 instead of 0. Confident but wrong.
- `search_and_message` (hard): F1=0.67
- `alarm_and_reminder` (hard): F1=0.50  
- `timer_music_reminder` (hard): F1=0.40
- `search_message_weather` (hard): F1=0.80

### Biggest Score Levers
1. **Keep more on-device** (currently 30% → target 60%+) = biggest score impact
2. **Fix remaining F1 failures** in hard cases
3. **Speed** is already good for on-device (~200ms), cloud kills it (~1500ms)

---

## ✅ Status

### Done
- [x] Repo: https://github.com/clawdfadi-glitch/cactus-hackathon
- [x] Cactus built (Python bindings on Mac mini)
- [x] FunctionGemma model downloaded
- [x] Cactus API key configured
- [x] Gemini API key configured  
- [x] Atomic Router v2 implemented
- [x] Benchmark running: **59.2%**

### TODO — Priority Order
- [ ] **Post-process time args** (fix alarm minute=9 → 0 bug)
- [ ] **Lower confidence threshold** to keep more calls on-device
- [ ] **Improve hard case splitting** (3-intent requests)
- [ ] **Prompt engineering** for FunctionGemma (better system prompt)
- [ ] Build end-to-end demo app (qualitative judging)
- [ ] Voice-to-action demo using cactus_transcribe (bonus points)
- [ ] Submit to leaderboard: `python submit.py --team "TEAM_NAME" --location "SF"`

---

## 🖥️ Local Setup (for Fadi)

```bash
# 1. Clone the repo
git clone https://github.com/clawdfadi-glitch/cactus-hackathon
cd cactus-hackathon

# 2. Clone cactus runtime inside the repo
git clone https://github.com/cactus-compute/cactus

# 3. Setup cactus (needs python 3.12)
brew install python@3.12 cmake
cd cactus && source ./setup && cd ..

# 4. Build cactus python bindings
cactus build --python

# 5. Login to HuggingFace and download model
pip install huggingface_hub
huggingface-cli login --token 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
cactus download google/functiongemma-270m-it --reconvert

# 6. Auth cactus
cactus auth
# Enter: cactus_live_7f99a6c99157bc32adb78c898ddacf6b

# 7. Install Gemini SDK
pip install google-genai

# 8. Set Gemini API key
export GEMINI_API_KEY="your-gemini-key"

# 9. Run benchmark
python benchmark.py

# 10. Submit (1x per hour max)
python submit.py --team "TEAM_NAME" --location "SF"
```

**Note:** Steps 2-6 create ~1GB of files in `cactus/` dir (gitignored). The model weights land in `cactus/weights/functiongemma-270m-it/`.

---

## 🔧 Who's Doing What

### Kai (remote on Mac mini)
- ✅ Environment setup & builds
- ✅ Code implementation in main.py  
- Running benchmarks & iterating on score
- Research & ideas

### Fadi (at hackathon in SF)
- Clone repo + local setup (see above)
- Local testing & experimentation
- Demo prep & presentation strategy
- Strategic direction & hackathon comms
- Leaderboard submissions

---

## 💡 Ideas Backlog (ranked)
1. ✅ **Atomic decomposition** — split multi-call → single calls
2. ✅ **Schema validation** — verify function names + required args
3. ✅ **Arg cleaning** — strip punctuation from model outputs
4. 🔜 **Time/number post-processing** — parse "10 AM" → hour:10, minute:0
5. 🔜 **Confidence calibration** — find optimal threshold per tool type
6. **Tool description optimization** — rewrite descriptions to help small model
7. **Two-pass verification** — FunctionGemma twice with different prompts
8. **Parallel local+cloud** — fire both, take fastest valid result
9. **Meta-layer orchestration** — Python as brain, model as hands
10. **Voice-to-action** — cactus_transcribe for qualitative bonus

---

## 📝 Key Learnings
- FunctionGemma is excellent at: weather, music, timer (single tool, simple args)
- FunctionGemma struggles with: alarms (wrong minute values), messages (content extraction)
- Cloud (Gemini) adds trailing punctuation to string args → need cleaning
- `cactus_reset()` between calls is important for clean state
- Model reuse (global handle) saves ~100ms per call vs re-init

## 🔗 Links
- Hackathon: https://sf.aitinkerers.org/hackathons/h_DRGnrtIWaG8/
- Leaderboard: https://cactusevals.ngrok.app
- Cactus keys: https://cactuscompute.com/dashboard/api-keys
- GCP credits (SF): https://trygcp.dev/claim/cactus-x-gdm-hackathon-sf
