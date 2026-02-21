import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, re, time
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
from google import genai
from google.genai import types


# ── Model management ─────────────────────────────────────────────────
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = cactus_init(functiongemma_path)
    return _model

def _fresh_model():
    """Destroy and reinitialize model for a clean state between requests."""
    global _model
    if _model is not None:
        cactus_destroy(_model)
        _model = None


def _clean_arg(value):
    """Clean argument values: strip trailing punctuation, normalize whitespace."""
    if isinstance(value, str):
        value = value.strip().rstrip('.!,;:')
        value = value.strip("'\"")
        value = ' '.join(value.split())
        # Convert ISO timestamps back to human-readable (e.g. "2024-06-14T14:00:00" -> "2:00 PM")
        iso_match = re.match(r'\d{4}-\d{2}-\d{2}T(\d{2}):(\d{2}):\d{2}', value)
        if iso_match:
            hour = int(iso_match.group(1))
            minute = int(iso_match.group(2))
            ampm = "AM" if hour < 12 else "PM"
            display_hour = hour if hour <= 12 else hour - 12
            if display_hour == 0:
                display_hour = 12
            if minute == 0:
                value = f"{display_hour}:00 {ampm}"
            else:
                value = f"{display_hour}:{minute:02d} {ampm}"
    return value


def _postprocess_args(call, user_text=""):
    """Fix known model output issues: type coercion, time parsing, etc."""
    name = call.get("name", "")
    args = call.get("arguments", {})

    if name == "set_alarm":
        # FunctionGemma is unreliable with alarm args — extract directly from user text
        if user_text:
            # Pattern: "X:YY AM/PM"
            time_match = re.search(r'(\d{1,2}):(\d{2})\s*(?:AM|PM|am|pm)', user_text)
            if time_match:
                args["hour"] = int(time_match.group(1))
                args["minute"] = int(time_match.group(2))
            else:
                # Pattern: "X AM/PM" (whole hour)
                hour_match = re.search(r'\b(\d{1,2})\s*(?:AM|PM|am|pm)', user_text)
                if hour_match:
                    args["hour"] = int(hour_match.group(1))
                    args["minute"] = 0
        # Fallback: coerce types if text extraction didn't work
        for key in ("hour", "minute"):
            if key in args:
                if isinstance(args[key], str):
                    try:
                        args[key] = int(float(args[key]))
                    except (ValueError, TypeError):
                        pass
                elif isinstance(args[key], float):
                    args[key] = int(args[key])
        if "minute" not in args or args.get("minute") is None:
            args["minute"] = 0

    if name == "play_music":
        song = args.get("song", "")
        # "some jazz music" → "jazz" (strip filler "some" and generic "music")
        sm = re.match(r'some\s+(\w+)\s+music$', song, re.IGNORECASE)
        if sm:
            args["song"] = sm.group(1)

    if name == "set_timer":
        # Extract minutes from user text when possible
        if user_text:
            min_match = re.search(r'(\d+)\s*(?:minute|min)', user_text, re.IGNORECASE)
            if min_match:
                args["minutes"] = int(min_match.group(1))
        if "minutes" in args:
            if isinstance(args["minutes"], str):
                try:
                    args["minutes"] = int(float(args["minutes"]))
                except (ValueError, TypeError):
                    pass
            elif isinstance(args["minutes"], float):
                args["minutes"] = int(args["minutes"])

    call["arguments"] = args
    return call


def _clean_calls(calls, user_text=""):
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
        clean_call = _postprocess_args(clean_call, user_text)
        cleaned.append(clean_call)
    return cleaned


def _cactus_call(messages, tools, force_tools=True):
    """Single FunctionGemma call — reuses model handle."""
    model = _get_model()
    cactus_reset(model)

    cactus_tools = [{"type": "function", "function": t} for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools. Call the most appropriate tool for the user's request."}] + messages,
        tools=cactus_tools,
        force_tools=force_tools,
        temperature=0,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        # Telemetry can corrupt output — try to extract valid JSON
        brace_start = raw_str.find('{"success"')
        if brace_start >= 0:
            depth = 0
            for i in range(brace_start, len(raw_str)):
                if raw_str[i] == '{':
                    depth += 1
                elif raw_str[i] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            raw = json.loads(raw_str[brace_start:i+1])
                            break
                        except json.JSONDecodeError:
                            pass
            else:
                return {"function_calls": [], "total_time_ms": 0, "confidence": 0, "cloud_handoff": True}
        else:
            return {"function_calls": [], "total_time_ms": 0, "confidence": 0, "cloud_handoff": True}

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
        "cloud_handoff": raw.get("cloud_handoff", False),
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

    try:
        gemini_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                tools=gemini_tools,
                system_instruction="You are a helpful assistant. When the user asks you to do multiple things, call ALL the required tools in parallel. Do not wait for one tool result before calling the next. Call every tool needed to fulfill the complete request.",
            ),
        )
    except Exception:
        # Retry once with fresh client after a short pause
        time.sleep(0.5)
        try:
            client2 = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            gemini_response = client2.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=gemini_tools,
                    system_instruction="You are a helpful assistant. When the user asks you to do multiple things, call ALL the required tools in parallel. Do not wait for one tool result before calling the next. Call every tool needed to fulfill the complete request.",
                ),
            )
        except Exception:
            return {"function_calls": [], "total_time_ms": (time.time() - start_time) * 1000}

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


# ── Tool pre-filtering (Python-side, not model-side) ─────────────────

# Tools that FunctionGemma reliably handles on-device
# Others get routed to cloud immediately
_CLOUD_ONLY_TOOLS = {"create_reminder", "search_contacts"}

# Map user-text keywords to tool names for pre-filtering
_TOOL_KEYWORDS = {
    "get_weather": [r'\bweather\b', r'\bforecast\b', r'\btemperature\b'],
    "set_alarm": [r'\balarm\b', r'\bwake\s+me\b'],
    "send_message": [r'\b(send|text)\b.*\b(message|saying|say)\b', r'\btext\s+\w+', r'\b(send|text)\s+\w+\s+saying\b'],
    "create_reminder": [r'\bremind\b', r'\breminder\b'],
    "search_contacts": [r'\b(find|look\s*up|search)\b.*\bcontact', r'\b(find|look\s*up)\s+\w+\s+in\s+my\s+contact'],
    "play_music": [r'\bplay\b'],
    "set_timer": [r'\btimer\b', r'\b\d+\s*min'],
}


def _pick_best_tool(text, tools):
    """Use keyword matching to find the single best tool for a request.
    Returns a 1-element list with the best tool, or the full list if ambiguous."""
    text_lower = text.lower()
    tool_names = {t["name"] for t in tools}
    matches = []

    for tool_name, patterns in _TOOL_KEYWORDS.items():
        if tool_name not in tool_names:
            continue
        for pattern in patterns:
            if re.search(pattern, text_lower):
                matches.append(tool_name)
                break

    if len(matches) == 1:
        return [t for t in tools if t["name"] == matches[0]]
    return tools


def _manual_extract(text, tool_name):
    """Manually extract function call from user text when model fails.
    Returns a function call dict or None."""

    if tool_name == "play_music":
        m = re.search(r'play\s+(.+?)(?:\s*[.,!]?\s*$)', text, re.IGNORECASE)
        if m:
            return {"name": "play_music", "arguments": {"song": m.group(1).strip()}}

    if tool_name == "get_weather":
        m = re.search(r'(?:weather|forecast|temperature)\s+(?:in|for|at)\s+(.+?)(?:\s*[.,!?]?\s*$)', text, re.IGNORECASE)
        if not m:
            m = re.search(r'(?:in|for|at)\s+(.+?)(?:\s*[.,!?]?\s*$)', text, re.IGNORECASE)
        if m:
            return {"name": "get_weather", "arguments": {"location": m.group(1).strip()}}

    if tool_name == "set_timer":
        m = re.search(r'(\d+)\s*(?:minute|min)', text, re.IGNORECASE)
        if m:
            return {"name": "set_timer", "arguments": {"minutes": int(m.group(1))}}

    if tool_name == "set_alarm":
        time_match = re.search(r'(\d{1,2}):(\d{2})\s*(?:AM|PM|am|pm)', text)
        if time_match:
            return {"name": "set_alarm", "arguments": {"hour": int(time_match.group(1)), "minute": int(time_match.group(2))}}
        hour_match = re.search(r'\b(\d{1,2})\s*(?:AM|PM|am|pm)', text)
        if hour_match:
            return {"name": "set_alarm", "arguments": {"hour": int(hour_match.group(1)), "minute": 0}}

    if tool_name == "create_reminder":
        m = re.search(r'(?:remind\s+(?:me\s+)?(?:about|to)\s+)(.+?)\s+at\s+(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))', text, re.IGNORECASE)
        if m:
            title = re.sub(r'^(the|a|an)\s+', '', m.group(1).strip(), flags=re.IGNORECASE)
            return {"name": "create_reminder", "arguments": {"title": title, "time": m.group(2).strip().upper()}}

    if tool_name == "search_contacts":
        m = re.search(r'(?:find|look\s*up|search)\s+(\w+)\s+(?:in\s+)?(?:my\s+)?contact', text, re.IGNORECASE)
        if m:
            return {"name": "search_contacts", "arguments": {"query": m.group(1)}}

    if tool_name == "send_message":
        # Pattern: "Send a message to X saying Y"
        m = re.search(r'(?:send|text)\s+(?:a\s+)?message\s+to\s+(\w+)\s+saying\s+(.+?)(?:\s*[.,!?]?\s*$)', text, re.IGNORECASE)
        if not m:
            # Pattern: "Text/Send X (a message) saying Y"
            m = re.search(r'(?:send|text)\s+(\w+)\s+(?:a\s+)?(?:message\s+)?saying\s+(.+?)(?:\s*[.,!?]?\s*$)', text, re.IGNORECASE)
        if m:
            return {"name": "send_message", "arguments": {"recipient": m.group(1), "message": m.group(2).strip()}}

    return None


# ── Atomic Router v3 ─────────────────────────────────────────────────

def _count_intents(text):
    """Count likely number of tool-call intents in user text."""
    action_patterns = [
        r'\b(set\s+(an?\s+)?alarm|wake\s+me)\b',
        r'\b(set\s+(a\s+)?(timer|\d+\s*min))\b',
        r'\b(send|text)\s+\w+',
        r'\b(check|get|what\'?s?|how\'?s?)\s+(the\s+)?weather\b',
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
    """Split a multi-intent request into atomic sub-requests."""
    action_start = r'(?:set|send|text|check|get|what|play|remind|create|find|look|search|wake)'

    # Pattern 1: "X, and Y" or "X, Y, and Z"
    parts = re.split(r',\s+and\s+', text, flags=re.IGNORECASE)

    if len(parts) == 1:
        # Pattern 2: "X and Y" where Y starts with an action verb
        parts = re.split(r'\s+and\s+(?=' + action_start + r'\s)', text, flags=re.IGNORECASE)

    # Further split parts that contain ", <action verb>"
    expanded = []
    for part in parts:
        sub = re.split(r',\s+(?=' + action_start + r'\s)', part, flags=re.IGNORECASE)
        expanded.extend(sub)
    parts = expanded

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


def _local_is_good(local, tools, user_text=""):
    """Decide whether a local result is good enough to keep."""
    calls = local.get("function_calls", [])
    confidence = local.get("confidence", 0)
    cloud_handoff = local.get("cloud_handoff", False)

    if cloud_handoff:
        return False
    if not calls:
        return False

    valid_calls = [c for c in calls if _validate_call(c, tools)]
    if not valid_calls:
        return False
    if confidence < 0.3:
        return False

    return True


def generate_hybrid(messages, tools):
    """Atomic Router v3: decompose → local-first → validate → cloud fallback with full context.

    Strategy:
    1. Fresh model init to prevent cross-request state pollution
    2. Count intents in user message
    3. Single intent: try FunctionGemma → validate → cloud fallback if needed
    4. Multi intent: split into atomic parts → FunctionGemma each
       - If ALL parts succeed locally → use local results (fast, on-device)
       - If ANY part fails → send FULL original message to cloud (preserves context)
    5. Clean all argument values and post-process types
    6. Deduplicate results
    """
    _fresh_model()

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

        if len(atomic_parts) < 2:
            # Splitting failed — try local whole, then cloud whole
            local = _cactus_call(messages, tools)
            total_time += local["total_time_ms"]

            if _local_is_good(local, tools, user_text):
                all_calls = [c for c in local["function_calls"] if _validate_call(c, tools)]
            else:
                cloud = generate_cloud(messages, tools)
                total_time += cloud["total_time_ms"]
                all_calls = cloud["function_calls"]
                used_cloud = True
        else:
            # Try each atomic part — manual extraction first (fastest + most reliable)
            local_results = []
            local_time = 0
            all_good = True

            # Extract proper nouns from full text for pronoun resolution
            proper_nouns = [
                w for w in re.findall(r'\b[A-Z][a-z]+\b', user_text)
                if w.lower() not in (
                    "set", "send", "find", "check", "play", "remind",
                    "text", "look", "search", "wake", "what", "how",
                )
            ]

            for part in atomic_parts:
                sub_messages = [{"role": "user", "content": part}]
                filtered = _pick_best_tool(part, tools)
                best_name = filtered[0]["name"] if len(filtered) == 1 else None

                # Try manual extraction first for ALL tools in multi-intent
                if best_name:
                    manual = _manual_extract(part, best_name)
                    if manual and _validate_call(manual, tools):
                        # Pronoun resolution for send_message
                        if manual["name"] == "send_message":
                            r = manual["arguments"].get("recipient", "")
                            if r.lower() in ("him", "her", "them") and proper_nouns:
                                manual["arguments"]["recipient"] = proper_nouns[0]
                        local_results.append(manual)
                        continue

                # Manual extraction failed; cloud-only tools can't use model
                if best_name in (_CLOUD_ONLY_TOOLS | {"send_message"}):
                    all_good = False
                    break

                # Try model for other tools
                _fresh_model()
                local = _cactus_call(sub_messages, filtered)
                local_time += local["total_time_ms"]

                if _local_is_good(local, tools, part):
                    valid = [c for c in local["function_calls"] if _validate_call(c, tools)]
                    local_results.extend(valid)
                else:
                    all_good = False
                    break

            total_time += local_time

            if all_good and local_results:
                all_calls = local_results
            else:
                # Fall back to cloud with FULL original message for proper context
                cloud = generate_cloud(messages, tools)
                total_time += cloud["total_time_ms"]
                all_calls = cloud["function_calls"]
                used_cloud = True

                # Supplement: if cloud returned fewer calls than expected intents,
                # fill missing ones via manual extraction from the split parts
                if len(all_calls) < len(atomic_parts):
                    returned_names = {c["name"] for c in all_calls}
                    # Find proper nouns in user text for pronoun resolution
                    proper_nouns = [
                        w for w in re.findall(r'\b[A-Z][a-z]+\b', user_text)
                        if w.lower() not in (
                            "set", "send", "find", "check", "play", "remind",
                            "text", "look", "search", "wake", "what", "how",
                        )
                    ]
                    for part in atomic_parts:
                        best = _pick_best_tool(part, tools)
                        if len(best) != 1:
                            continue
                        tool_name = best[0]["name"]
                        if tool_name in returned_names:
                            continue
                        # Build call for the missing tool
                        if tool_name == "send_message":
                            m = re.search(
                                r'(?:send|text)\s+(\w+)\s+(?:a\s+)?(?:message\s+)?saying\s+(.+?)$',
                                part, re.IGNORECASE,
                            )
                            if m:
                                recipient = m.group(1)
                                message = m.group(2).strip().rstrip(".,!?")
                                if recipient.lower() in ("him", "her", "them") and proper_nouns:
                                    recipient = proper_nouns[0]
                                call = {"name": "send_message", "arguments": {"recipient": recipient, "message": message}}
                                if _validate_call(call, tools):
                                    all_calls.append(call)
                        else:
                            manual = _manual_extract(part, tool_name)
                            if manual and _validate_call(manual, tools):
                                all_calls.append(manual)
    else:
        # ── SINGLE-CALL PATH ──
        # Check if the best-matching tool is one FunctionGemma handles poorly
        filtered_tools = _pick_best_tool(user_text, tools)
        best_tool_name = filtered_tools[0]["name"] if len(filtered_tools) == 1 else None
        skip_local = best_tool_name in _CLOUD_ONLY_TOOLS if best_tool_name else False

        if skip_local:
            # For cloud-only tools, try manual extraction first (faster + on-device)
            manual = _manual_extract(user_text, best_tool_name) if best_tool_name else None
            if manual and _validate_call(manual, tools):
                all_calls = [manual]
            else:
                cloud = generate_cloud(messages, tools)
                total_time += cloud["total_time_ms"]
                all_calls = cloud["function_calls"]
                used_cloud = True
        else:
            local = _cactus_call(messages, filtered_tools)
            total_time += local["total_time_ms"]

            if _local_is_good(local, tools, user_text):
                all_calls = [c for c in local["function_calls"] if _validate_call(c, tools)]
            else:
                # For simple tools, try manual arg extraction (regex is reliable)
                manual = None
                if best_tool_name in ("set_alarm", "set_timer", "get_weather", "send_message", "play_music"):
                    manual = _manual_extract(user_text, best_tool_name)
                if manual and _validate_call(manual, tools):
                    all_calls = [manual]
                else:
                    # Retry with fresh model (helps with non-deterministic failures)
                    _fresh_model()
                    local2 = _cactus_call(messages, filtered_tools)
                    total_time += local2["total_time_ms"]
                    if _local_is_good(local2, tools, user_text):
                        all_calls = [c for c in local2["function_calls"] if _validate_call(c, tools)]
                    else:
                        cloud = generate_cloud(messages, tools)
                        total_time += cloud["total_time_ms"]
                        all_calls = cloud["function_calls"]
                        used_cloud = True

    # Clean all argument values + post-process types
    all_calls = _clean_calls(all_calls, user_text)

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
    print_result("Hybrid Atomic Router v3", hybrid)
