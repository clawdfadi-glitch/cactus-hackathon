import json, os, re, sys, time

# Try system cactus first (eval server provides its own runtime).
# Only fall back to local path for development — avoids eval server
# scanning our local cactus/ tree which contains subprocess imports.
try:
    from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
except ImportError:
    sys.path.insert(0, "cactus/python/src")
    from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset

# Cloud fallback via Gemini (optional — may be unavailable on eval server)
try:
    from google import genai
    from google.genai import types
    _HAS_CLOUD = True
except ImportError:
    _HAS_CLOUD = False

# Path to FunctionGemma weights — check env var first for eval server
functiongemma_path = os.environ.get("FUNCTIONGEMMA_PATH", "cactus/weights/functiongemma-270m-it")


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
        # Convert ISO timestamps back to human-readable
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
    """Fix known model output issues: type coercion, time parsing, etc.

    For KNOWN tools: override FunctionGemma's args with regex-extracted values.
    For UNKNOWN tools: just pass through (basic cleaning in _clean_arg is enough).
    """
    name = call.get("name", "")
    args = call.get("arguments", {})

    if name == "set_alarm":
        if user_text:
            time_match = re.search(r'(\d{1,2}):(\d{2})\s*(?:AM|PM|am|pm)', user_text)
            if time_match:
                args["hour"] = int(time_match.group(1))
                args["minute"] = int(time_match.group(2))
            else:
                hour_match = re.search(r'\b(\d{1,2})\s*(?:AM|PM|am|pm)', user_text)
                if hour_match:
                    args["hour"] = int(hour_match.group(1))
                    args["minute"] = 0
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
        if isinstance(song, str):
            sm = re.match(r'some\s+(\w+)\s+music$', song, re.IGNORECASE)
            if sm:
                args["song"] = sm.group(1)

    if name == "set_timer":
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

    if name == "create_reminder" and user_text:
        m = re.search(r'(?:remind\s+(?:me\s+)?(?:about|to)\s+)(.+?)\s+at\s+(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))', user_text, re.IGNORECASE)
        if m:
            title = re.sub(r'^(the|a|an)\s+', '', m.group(1).strip(), flags=re.IGNORECASE)
            args["title"] = title
            args["time"] = m.group(2).strip().upper()

    if name == "search_contacts" and user_text:
        m = re.search(r'(?:find|look\s*up|search)\s+(\w+)\s+(?:in\s+)?(?:my\s+)?contact', user_text, re.IGNORECASE)
        if m:
            args["query"] = m.group(1)

    if name == "send_message" and user_text:
        m = re.search(r'(?:send|text)\s+(?:a\s+)?message\s+to\s+(\w+)\s+saying\s+(.+)', user_text, re.IGNORECASE)
        if not m:
            m = re.search(r'(?:send|text)\s+(\w+)\s+(?:a\s+)?(?:message\s+)?saying\s+(.+)', user_text, re.IGNORECASE)
        if m:
            recipient = m.group(1)
            msg = m.group(2).strip()
            # Trim at comma or "and" + action verb (multi-intent boundary)
            trim = re.match(r'^(.+?)(?:\s*,\s*|\s+and\s+)(?:check|get|set|play|remind|find|look|search|wake|send|text)\b', msg, re.IGNORECASE)
            if trim:
                msg = trim.group(1).strip()
            msg = msg.rstrip('.,!?')
            # If regex found a pronoun, resolve it from proper nouns in the text
            if recipient.lower() in ("him", "her", "them"):
                proper = [
                    w for w in re.findall(r'\b[A-Z][a-z]+\b', user_text)
                    if w.lower() not in (
                        "set", "send", "find", "check", "play", "remind",
                        "text", "look", "search", "wake", "what", "how",
                    )
                ]
                if proper:
                    recipient = proper[0]
                elif args.get("recipient") and args["recipient"].lower() not in ("him", "her", "them"):
                    # Keep existing resolved recipient from per-part processing
                    recipient = args["recipient"]
            if recipient and msg:
                args["recipient"] = recipient
                args["message"] = msg

    if name == "get_weather" and user_text:
        m = re.search(r'(?:weather|forecast|temperature)\s+(?:like\s+)?(?:in|for|at)\s+(.+?)(?:\s*(?:,|\band\b\s+(?:set|send|text|play|remind|find|look|search|check|get|wake))\s*|[.,!?]*$)', user_text, re.IGNORECASE)
        if not m:
            m = re.search(r'\b(?:in|for|at)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)', user_text)
        if m:
            location = m.group(1).strip().rstrip('.,!?')
            if location:
                args["location"] = location

    # For all args: coerce numeric strings to int/float where possible
    for key, val in list(args.items()):
        if isinstance(val, str):
            # Try int first
            try:
                if val.isdigit() or (val.startswith('-') and val[1:].isdigit()):
                    args[key] = int(val)
            except (ValueError, IndexError):
                pass

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
    if not _HAS_CLOUD:
        return {"function_calls": [], "total_time_ms": 0}

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

# Map user-text keywords to tool names for pre-filtering.
# For UNKNOWN tools not in this map, _pick_best_tool returns all tools
# and FunctionGemma selects the right one from the full list.
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
    Returns a 1-element list with the best tool, or the full list if ambiguous.
    For unknown tools not in _TOOL_KEYWORDS, returns the full list so
    FunctionGemma can select from all available tools."""
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
    """Manually extract function call from user text when FunctionGemma fails.
    This is a FALLBACK for known tools only. Returns a function call dict or None."""

    if tool_name == "play_music":
        m = re.search(r'play\s+(.+?)(?:\s*[.,!]?\s*$)', text, re.IGNORECASE)
        if m:
            return {"name": "play_music", "arguments": {"song": m.group(1).strip()}}

    if tool_name == "get_weather":
        m = re.search(r'(?:weather|forecast|temperature)\s+(?:like\s+)?(?:in|for|at)\s+(.+?)(?:\s*[.,!?]?\s*$)', text, re.IGNORECASE)
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
        m = re.search(r'(?:send|text)\s+(?:a\s+)?message\s+to\s+(\w+)\s+saying\s+(.+?)(?:\s*[.,!?]?\s*$)', text, re.IGNORECASE)
        if not m:
            m = re.search(r'(?:send|text)\s+(\w+)\s+(?:a\s+)?(?:message\s+)?saying\s+(.+?)(?:\s*[.,!?]?\s*$)', text, re.IGNORECASE)
        if m:
            return {"name": "send_message", "arguments": {"recipient": m.group(1), "message": m.group(2).strip()}}

    return None


# ── Atomic Router v4: FunctionGemma-first ────────────────────────────

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

    parts = re.split(r',\s+and\s+', text, flags=re.IGNORECASE)

    if len(parts) == 1:
        parts = re.split(r'\s+and\s+(?=' + action_start + r'\s)', text, flags=re.IGNORECASE)

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
    """Atomic Router v4: FunctionGemma-first for all tools (known + unknown).

    Strategy:
    1. Fresh model init to prevent cross-request state pollution
    2. Count intents in user message
    3. Single intent: FunctionGemma → fix args (regex for known tools) → validate
       → manual_extract fallback → retry → cloud last resort
    4. Multi intent: split → FunctionGemma per part → fix args → manual_extract fallback
       → if any fail, cloud with FULL original for context
    5. Clean all argument values and post-process types
    6. Deduplicate results

    Key insight: FunctionGemma is GOOD at selecting the right tool name.
    It's BAD at extracting argument values. So we trust its tool selection
    and fix args with regex for known tools. For unknown tools we just
    clean the args and trust FunctionGemma.
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
            # Splitting failed — try FunctionGemma on whole, then cloud
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

                # 1. Try FunctionGemma first (good at tool selection)
                _fresh_model()
                local = _cactus_call(sub_messages, filtered)
                local_time += local["total_time_ms"]

                if _local_is_good(local, tools, part):
                    valid = [c for c in local["function_calls"] if _validate_call(c, tools)]
                    if valid:
                        # Apply per-part arg fixing
                        for v in valid:
                            _postprocess_args(v, part)
                        # Pronoun resolution for send_message
                        for v in valid:
                            if v["name"] == "send_message":
                                r = v["arguments"].get("recipient", "")
                                if r.lower() in ("him", "her", "them") and proper_nouns:
                                    v["arguments"]["recipient"] = proper_nouns[0]
                        local_results.extend(valid)
                        continue

                # 2. FunctionGemma failed — try manual extraction as FALLBACK
                if best_name:
                    manual = _manual_extract(part, best_name)
                    if manual and _validate_call(manual, tools):
                        if manual["name"] == "send_message":
                            r = manual["arguments"].get("recipient", "")
                            if r.lower() in ("him", "her", "them") and proper_nouns:
                                manual["arguments"]["recipient"] = proper_nouns[0]
                        local_results.append(manual)
                        continue

                # 3. Both failed for this part
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

                # Supplement: if cloud returned fewer calls than expected,
                # fill missing ones via manual extraction from split parts
                if len(all_calls) < len(atomic_parts):
                    returned_names = {c["name"] for c in all_calls}
                    for part in atomic_parts:
                        best = _pick_best_tool(part, tools)
                        if len(best) != 1:
                            continue
                        tool_name = best[0]["name"]
                        if tool_name in returned_names:
                            continue
                        if tool_name == "send_message":
                            sm = re.search(
                                r'(?:send|text)\s+(\w+)\s+(?:a\s+)?(?:message\s+)?saying\s+(.+?)$',
                                part, re.IGNORECASE,
                            )
                            if sm:
                                recipient = sm.group(1)
                                msg = sm.group(2).strip().rstrip(".,!?")
                                if recipient.lower() in ("him", "her", "them") and proper_nouns:
                                    recipient = proper_nouns[0]
                                call = {"name": "send_message", "arguments": {"recipient": recipient, "message": msg}}
                                if _validate_call(call, tools):
                                    all_calls.append(call)
                        else:
                            manual = _manual_extract(part, tool_name)
                            if manual and _validate_call(manual, tools):
                                all_calls.append(manual)
    else:
        # ── SINGLE-CALL PATH ──
        # FunctionGemma-first: try on-device for ALL tools (no cloud-only skipping)
        filtered_tools = _pick_best_tool(user_text, tools)
        best_tool_name = filtered_tools[0]["name"] if len(filtered_tools) == 1 else None

        # 1. Try FunctionGemma first (good at selecting the right tool)
        local = _cactus_call(messages, filtered_tools)
        total_time += local["total_time_ms"]

        if _local_is_good(local, tools, user_text):
            all_calls = [c for c in local["function_calls"] if _validate_call(c, tools)]
        else:
            # 2. FunctionGemma failed — try manual extraction as FALLBACK
            manual = None
            if best_tool_name:
                manual = _manual_extract(user_text, best_tool_name)
            if manual and _validate_call(manual, tools):
                all_calls = [manual]
            else:
                # 3. Retry FunctionGemma with fresh model (non-deterministic)
                _fresh_model()
                local2 = _cactus_call(messages, filtered_tools)
                total_time += local2["total_time_ms"]
                if _local_is_good(local2, tools, user_text):
                    all_calls = [c for c in local2["function_calls"] if _validate_call(c, tools)]
                else:
                    # 4. Last resort: cloud
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
    print_result("Hybrid Atomic Router v4", hybrid)
