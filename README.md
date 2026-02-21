# Cactus x DeepMind Hackathon - Edge/Cloud Hybrid Routing

## What
Smart routing strategy for FunctionGemma (local) vs Gemini (cloud) tool calling.

## Setup
```bash
# 1. Clone cactus runtime into this dir
git clone https://github.com/cactus-compute/cactus

# 2. Build cactus
cd cactus && source ./setup && cd ..
cactus build --python
cactus download google/functiongemma-270m-it --reconvert

# 3. Auth
cactus auth  # enter your cactus API key
export GEMINI_API_KEY="your-key"

# 4. Install deps
pip install google-genai

# 5. Run benchmark
python benchmark.py
```

## Strategy
See main.py `generate_hybrid()` — that's the only function we modify.

