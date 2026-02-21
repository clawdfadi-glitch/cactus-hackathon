import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, re, time
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
from google import genai
from google.genai import types


# ── Shared model handle (avoid re-init per call) ──────────────────────
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = cactus_init(functiongemma_path)
    return _model


def _clean_arg(value):
    """Clean argument values: strip trailing punctuation, normalize whitespace."""
    if isinstance(value, str):
        # Strip trailing punctuation that models add but benchmarks don't expect
        value = value.strip().rstrip('.!,;:')
        # Normalize whitespace
        value = ' '.join(value.split())
    return value


def _clean_calls(calls):
    """Clean all argument values in function calls."""
    cleaned = []
    for call in calls:
        clean_call = {"name": call["name"]}
        if "arguments" in call:
            clean_call["arguments"] = {
                k: _clean_arg(v) for k, v in call["arguments"].items()
            }
        else:
            clean_call["arguments"] = {}
        cleaned.append(clean_call)
    return cleaned


def _cactus_call(messages, tools, force_tools=True):
    """Single FunctionGemma call — reuses model handle."""
    model = _get_model()
    cactus_reset(model)  # Clear KV cache between calls

    cactus_tools = [{"type": "function", "function": t} for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools. Call the most appropriate tool for the user's request."}] + messages,
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


# ── Atomic Router v2 ─────────────────────────────────────────────────

def _count_intents(text):
    """Count likely number of tool-call intents in user text."""
    # Action verbs that map to tools
    action_patterns = [
        r'\b(set\s+(an?\s+)?alarm)\b',
        r'\b(set\s+(a\s+)?timer)\b',
        r'\b(send|text)\s+\w+',
        r'\b(check|get|what\'?s?)\s+(the\s+)?weather\b',
        r'\b(play)\s+',
        r'\b(remind|create\s+a?\s*reminder)\b',
        r'\b(find|look\s+up|search)\b',
    ]
    count = 0
    for p in action_patterns:
        if re.search(p, text, re.IGNORECASE):
            count += 1
    return max(count, 1)


def _split_into_atomic(text):
    """Split a multi-intent request into atomic sub-requests.
    
    Handles patterns like:
    - "Do X and do Y"
    - "Do X, do Y, and do Z"  
    - "Do X and also Y"
    """
    # Split on ", and ", " and ", ","
    # But avoid splitting "Bohemian Rhapsody and ..." style
    parts = re.split(r'(?:,\s*and\s+|,\s+and\s+|\s+and\s+(?=\w+\s+(?:the|a|an|my|me|to|for|in)\s))', text, flags=re.IGNORECASE)
    
    if len(parts) == 1:
        # Try splitting on just ", " for "Do X, check Y" patterns
        parts = re.split(r',\s+(?=[a-z])', text, flags=re.IGNORECASE)
    
    cleaned = []
    for part in parts:
        part = part.strip().rstrip('.')
        if len(part) > 3:
            cleaned.append(part)
    
    return cleaned if len(cleaned) > 1 else [text]


def _validate_call(call, tools):
    """Check if a function call is valid against the tool definitions."""
    tool_names = {t["name"] for t in tools}
    
    if call.get("name") not in tool_names:
        return False
    
    tool = next(t for t in tools if t["name"] == call["name"])
    required = tool["parameters"].get("required", [])
    args = call.get("arguments", {})
    
    for req in required:
        if req not in args:
            return False
        if isinstance(args[req], str) and not args[req].strip():
            return False
    
    return True


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Atomic Router v2: decompose → local-first → validate → selective cloud fallback.
    
    Strategy:
    1. Count intents in user message
    2. Single intent: try FunctionGemma → validate → cloud fallback if needed
    3. Multi intent: split into atomic parts → FunctionGemma each → cloud for failures
    4. Clean all argument values (strip trailing punctuation etc)
    5. Deduplicate results
    """
    user_text = ""
    for m in messages:
        if m["role"] == "user":
            user_text = m["content"]
    
    total_time = 0
    all_calls = []
    used_cloud = False
    intent_count = _count_intents(user_text)
    
    if intent_count >= 2:
        # ── MULTI-CALL PATH ──
        atomic_parts = _split_into_atomic(user_text)
        
        # If splitting failed but we detected multiple intents, send whole thing to cloud
        if len(atomic_parts) < 2:
            cloud = generate_cloud(messages, tools)
            total_time += cloud["total_time_ms"]
            all_calls = cloud["function_calls"]
            used_cloud = True
        else:
            for part in atomic_parts:
                sub_messages = [{"role": "user", "content": part}]
                local = _cactus_call(sub_messages, tools)
                total_time += local["total_time_ms"]
                
                valid_calls = [c for c in local["function_calls"] if _validate_call(c, tools)]
                
                if valid_calls and local["confidence"] >= 0.4:
                    all_calls.extend(valid_calls)
                else:
                    # Cloud fallback for this sub-request
                    cloud = generate_cloud(sub_messages, tools)
                    total_time += cloud["total_time_ms"]
                    all_calls.extend(cloud["function_calls"])
                    used_cloud = True
    else:
        # ── SINGLE-CALL PATH ──
        local = _cactus_call(messages, tools)
        total_time += local["total_time_ms"]
        
        valid_calls = [c for c in local["function_calls"] if _validate_call(c, tools)]
        
        if valid_calls and local["confidence"] >= 0.4:
            all_calls = valid_calls
        else:
            cloud = generate_cloud(messages, tools)
            total_time += cloud["total_time_ms"]
            all_calls = cloud["function_calls"]
            used_cloud = True
    
    # Clean all argument values
    all_calls = _clean_calls(all_calls)
    
    # Deduplicate
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


if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"],
        },
    }]

    messages = [{"role": "user", "content": "What is the weather in San Francisco?"}]

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid Atomic Router v2", hybrid)
