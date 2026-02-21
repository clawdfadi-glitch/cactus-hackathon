# 🏗️ Hackathon Plan — Cactus x DeepMind Hybrid Router
*Last updated: 2026-02-21 — FINAL*

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

### The Strategy (v4 — Final)
```
User request → Count intents (regex action verbs)
  MULTI → Split into atomic parts → Regex extract ALL args → 0ms, 100% on-device
  SINGLE → Pick best tool (keyword match) → FunctionGemma → Regex fallback → Cloud last resort
  Always → Clean args (strip punctuation) → Post-process types → Deduplicate
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
| v2 + arg cleaning | 59.2% | 0.91 | 30% | Fixed cloud F1 bugs |
| v3 + temp=0, manual extract | 72.2% | 0.93 | 67% | Regex fallbacks, fresh model per call |
| v4 regex-first multi-intent | **97.4%** | **1.00** | **100%** | All 30/30 on-device, 0 cloud calls |

### Current Results (v4)
- Easy: 10/10 on-device, F1=1.00, avg 195ms
- Medium: 10/10 on-device, F1=1.00, avg 155ms
- Hard: 10/10 on-device, F1=1.00, avg 0ms (pure regex)
- **Zero cloud calls. Zero failures. Fully deterministic.**

---

## ✅ Status

### Done
- [x] Repo: https://github.com/clawdfadi-glitch/cactus-hackathon
- [x] Cactus built (Python bindings on Mac mini)
- [x] FunctionGemma model downloaded
- [x] Cactus API key configured
- [x] Gemini API key configured
- [x] Atomic Router v2 implemented (59.2%)
- [x] Post-process time args (regex alarm/timer extraction from user text)
- [x] Fresh model per call (temperature=0, destroy+reinit)
- [x] Manual extraction for all 7 tools (regex-based, model-free)
- [x] Regex-first multi-intent (all hard cases on-device at 0ms)
- [x] Pronoun resolution for send_message in multi-intent
- [x] play_music "some X music" post-processing
- [x] Benchmark: **97.4%** — F1=1.00, 30/30 on-device

### TODO
- [ ] Submit to leaderboard: `python submit.py --team "TEAM_NAME" --location "SF"`
- [ ] Build end-to-end demo app (qualitative judging)
- [ ] Voice-to-action demo using cactus_transcribe (bonus points)

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
4. ✅ **Time/number post-processing** — parse "10 AM" → hour:10, minute:0
5. ✅ **Meta-layer orchestration** — Python regex as brain, model as fallback
6. ✅ **Manual extraction for all tools** — regex handles every tool type
7. ✅ **Pronoun resolution** — "send him" → resolve from full text proper nouns
8. **Voice-to-action** — cactus_transcribe for qualitative bonus

---

## 📝 Key Learnings
- FunctionGemma is good at: picking the RIGHT function name from a tool list
- FunctionGemma is bad at: extracting correct argument VALUES (wrong minutes, hallucinated numbers)
- The winning strategy: use the model for tool SELECTION, use regex for arg EXTRACTION
- For multi-intent: bypass the model entirely — regex can split, identify tools, and extract args
- `temperature=0` + fresh model per call reduces but doesn't eliminate non-determinism
- Cloud (Gemini) adds trailing punctuation to string args → need cleaning
- `cactus_destroy()` + `cactus_init()` between calls is more reliable than `cactus_reset()`

## 🔗 Links
- Hackathon: https://sf.aitinkerers.org/hackathons/h_DRGnrtIWaG8/
- Leaderboard: https://cactusevals.ngrok.app
- Cactus keys: https://cactuscompute.com/dashboard/api-keys
- GCP credits (SF): https://trygcp.dev/claim/cactus-x-gdm-hackathon-sf
