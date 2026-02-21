import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, re, time
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types


# ── Shared model handle (avoid re-init per call) ──────────────────────
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = cactus_init(functiongemma_path)
    return _model


def _cactus_call(messages, tools, force_tools=True):
    """Single FunctionGemma call — reuses model handle."""
    model = _get_model()

    cactus_tools = [{"type": "function", "function": t} for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
        tools=cactus_tools,
        force_tools=force_tools,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0}

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    return _cactus_call(messages, tools)


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]

    start_time = time.time()

    gemini_response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config=types.GenerateContentConfig(tools=gemini_tools),
    )

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


# ── Atomic Router: the hybrid strategy ────────────────────────────────

def _is_multi_call(text):
    """Heuristic: detect if user wants multiple actions.
    
    Looks for conjunctions joining action phrases.
    """
    # Patterns that strongly suggest multiple intents
    multi_patterns = [
        r'\band\b.*\b(set|send|text|check|get|play|find|look|remind|create)\b',
        r'\b(set|send|text|check|get|play|find|look|remind|create)\b.*\band\b.*\b(set|send|text|check|get|play|find|look|remind|create)\b',
    ]
    for p in multi_patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False


def _split_into_atomic(text):
    """Split a multi-intent request into atomic sub-requests.
    
    Uses simple conjunction splitting + cleanup.
    """
    # Split on ", and ", " and " but be smart about it
    # First try splitting on ", and "
    parts = re.split(r',?\s+and\s+', text, flags=re.IGNORECASE)
    
    # Clean up each part
    cleaned = []
    for part in parts:
        part = part.strip().rstrip('.')
        if len(part) > 5:  # skip fragments
            cleaned.append(part)
    
    return cleaned if len(cleaned) > 1 else [text]


def _validate_call(call, tools):
    """Check if a function call is valid against the tool definitions."""
    tool_names = {t["name"] for t in tools}
    
    if call.get("name") not in tool_names:
        return False
    
    # Find the matching tool
    tool = next(t for t in tools if t["name"] == call["name"])
    required = tool["parameters"].get("required", [])
    args = call.get("arguments", {})
    
    # Check required args are present
    for req in required:
        if req not in args:
            return False
        # Check for empty values
        if isinstance(args[req], str) and not args[req].strip():
            return False
    
    return True


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Atomic Router: decompose → local-first → validate → fallback.
    
    Strategy:
    1. Detect multi-call requests and split into atomic sub-requests
    2. Run FunctionGemma on each atomic request independently
    3. Validate results (correct function name, required args present)
    4. Only fall back to cloud for failed/invalid atomic calls
    5. Merge all results
    """
    user_text = ""
    for m in messages:
        if m["role"] == "user":
            user_text = m["content"]
    
    total_time = 0
    all_calls = []
    used_cloud = False
    
    if _is_multi_call(user_text):
        # ── MULTI-CALL PATH: Atomic decomposition ──
        atomic_parts = _split_into_atomic(user_text)
        
        for part in atomic_parts:
            sub_messages = [{"role": "user", "content": part}]
            local = _cactus_call(sub_messages, tools)
            total_time += local["total_time_ms"]
            
            # Validate local result
            valid_calls = [c for c in local["function_calls"] if _validate_call(c, tools)]
            
            if valid_calls and local["confidence"] >= 0.3:
                all_calls.extend(valid_calls)
            else:
                # Fallback to cloud for this atomic part only
                cloud = generate_cloud(sub_messages, tools)
                total_time += cloud["total_time_ms"]
                all_calls.extend(cloud["function_calls"])
                used_cloud = True
    else:
        # ── SINGLE-CALL PATH: Try local first ──
        local = _cactus_call(messages, tools)
        total_time += local["total_time_ms"]
        
        valid_calls = [c for c in local["function_calls"] if _validate_call(c, tools)]
        
        if valid_calls and local["confidence"] >= 0.3:
            all_calls = valid_calls
        else:
            # Fallback to cloud
            cloud = generate_cloud(messages, tools)
            total_time += cloud["total_time_ms"]
            all_calls = cloud["function_calls"]
            used_cloud = True
    
    # Deduplicate calls (same function + same args = keep one)
    seen = set()
    deduped = []
    for call in all_calls:
        key = (call["name"], json.dumps(call.get("arguments", {}), sort_keys=True))
        if key not in seen:
            seen.add(key)
            deduped.append(call)
    
    return {
        "function_calls": deduped,
        "total_time_ms": total_time,
        "source": "cloud (fallback)" if used_cloud else "on-device",
        "confidence": 1.0 if not used_cloud else 0.0,
    }


def print_result(label, result):
    """Pretty-print a generation result."""
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


############## Example usage ##############

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid Atomic Router", hybrid)
