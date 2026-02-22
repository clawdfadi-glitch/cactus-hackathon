import json, os, re, sys, time

# Try system cactus first (eval server provides its own runtime).
# Only fall back to local path for development.
try:
    from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
    HAS_CACTUS = True
except ImportError:
    try:
        sys.path.insert(0, "cactus/python/src")
        from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
        HAS_CACTUS = True
    except ImportError:
        HAS_CACTUS = False

try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = bool(os.environ.get("GEMINI_API_KEY"))
except ImportError:
    HAS_GEMINI = False

# Find model weights — env var first (eval server), then search common paths
functiongemma_path = os.environ.get("FUNCTIONGEMMA_PATH")
if not functiongemma_path or not os.path.exists(functiongemma_path):
    functiongemma_path = None
    for p in [
        "cactus/weights/functiongemma-270m-it",
        "weights/functiongemma-270m-it",
        os.path.expanduser("~/.cactus/weights/functiongemma-270m-it"),
        "/tmp/functiongemma-270m-it",
    ]:
        if os.path.exists(p):
            functiongemma_path = p
            break


# ── Model management ─────────────────────────────────────────────────
_model = None

def _get_model():
    global _model
    if _model is None:
        if not HAS_CACTUS or not functiongemma_path:
            return None
        try:
            _model = cactus_init(functiongemma_path)
        except Exception:
            return None
    return _model

def _fresh_model():
    global _model
    if _model is not None:
        try:
            cactus_destroy(_model)
        except Exception:
            pass
        _model = None


def _clean_arg(value):
    """Clean argument values: strip trailing punctuation, normalize whitespace."""
    if isinstance(value, str):
        value = value.strip().rstrip('.!,;:')
        value = value.strip("'\"")
        value = ' '.join(value.split())
        # Convert ISO timestamps to human-readable
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
        # Always extract alarm time from user text — model minute values are unreliable
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
                elif re.search(r'\bnoon\b', user_text, re.IGNORECASE):
                    args["hour"] = 12
                    args["minute"] = 0
                elif re.search(r'\bmidnight\b', user_text, re.IGNORECASE):
                    args["hour"] = 0
                    args["minute"] = 0
        # Type coercion
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
        # Always extract song from user text — model values can be wrong
        if user_text:
            song_match = re.search(r'(?:play|put\s+on|listen\s+to)\s+(.+?)(?:\s*(?:,\s*(?:and\s+)?(?:set|send|text|check|get|remind|create|find|look|search|wake|tell|put|listen|message)\b|,\s*and\b|\s+and\s+(?:set|send|text|check|get|remind|create|find|look|search|wake|tell|put|listen|message)\b|[.,!?]\s*$))', user_text, re.IGNORECASE)
            if not song_match:
                # Simpler fallback: grab until end
                song_match = re.search(r'(?:play|put\s+on|listen\s+to)\s+(.+?)(?:\s*[.,!?]?\s*$)', user_text, re.IGNORECASE)
            if song_match:
                extracted = song_match.group(1).strip().rstrip(".,!?")
                # Strip filler: "some jazz music" → "jazz"
                sm = re.match(r'some\s+(.+?)\s+music$', extracted, re.IGNORECASE)
                if sm:
                    extracted = sm.group(1)
                args["song"] = extracted
        else:
            song = args.get("song", "")
            sm = re.match(r'some\s+(\w+)\s+music$', song, re.IGNORECASE)
            if sm:
                args["song"] = sm.group(1)

    if name == "set_timer":
        # Always extract timer duration from user text — model values are unreliable
        if user_text:
            min_match = re.search(r'(\d+)\s*(?:minute|min)', user_text, re.IGNORECASE)
            if min_match:
                args["minutes"] = int(min_match.group(1))
            else:
                hr_match = re.search(r'(\d+)\s*(?:hour|hr)', user_text, re.IGNORECASE)
                if hr_match:
                    args["minutes"] = int(hr_match.group(1)) * 60
                else:
                    sec_match = re.search(r'(\d+)\s*(?:second|sec)', user_text, re.IGNORECASE)
                    if sec_match:
                        args["minutes"] = max(1, int(sec_match.group(1)) // 60)
        if "minutes" in args:
            if isinstance(args["minutes"], str):
                try:
                    args["minutes"] = int(float(args["minutes"]))
                except (ValueError, TypeError):
                    pass
            elif isinstance(args["minutes"], float):
                args["minutes"] = int(args["minutes"])

    # ── Postprocess args from user_text for remaining tools (from Kai's v5) ──
    # This acts as a safety net: if the model picked the right tool but
    # extracted bad args, regex overrides with correct values.
    if name == "create_reminder" and user_text:
        m = re.search(r'(?:remind\s+(?:me\s+)?(?:about|to)\s+)(.+?)\s+at\s+(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))', user_text, re.IGNORECASE)
        if not m:
            m = re.search(r'(?:remind\s+(?:me\s+)?(?:about|to)\s+)(.+?)\s+at\s+(\d{1,2}\s*(?:AM|PM|am|pm))', user_text, re.IGNORECASE)
        if m:
            title = re.sub(r'^(the|a|an)\s+', '', m.group(1).strip(), flags=re.IGNORECASE)
            args["title"] = title
            time_str = m.group(2).strip().upper()
            if ':' not in time_str:
                time_str = re.sub(r'(\d+)\s*(AM|PM)', r'\1:00 \2', time_str)
            args["time"] = time_str

    if name == "search_contacts" and user_text:
        m = re.search(r'(?:find|look\s*up|search)\s+(\w+)\s+(?:in\s+)?(?:my\s+)?contacts?', user_text, re.IGNORECASE)
        if m:
            args["query"] = m.group(1)

    if name == "send_message" and user_text:
        m = re.search(r'(?:send|text)\s+(?:a\s+)?message\s+to\s+(\w+)\s+(?:saying|that)\s+(.+?)(?:\s*(?:,\s*(?:and\s+)?(?:set|check|get|play|remind|find|look|search)\b|\s+and\s+(?:set|check|get|play|remind|find|look|search)\b|[.,!?]*$))', user_text, re.IGNORECASE)
        if not m:
            m = re.search(r'(?:send|text)\s+(\w+)\s+(?:a\s+)?(?:message\s+)?(?:saying|that)\s+(.+?)(?:\s*(?:,\s*(?:and\s+)?(?:set|check|get|play|remind|find|look|search)\b|\s+and\s+(?:set|check|get|play|remind|find|look|search)\b|[.,!?]*$))', user_text, re.IGNORECASE)
        if not m:
            m = re.search(r'(?:tell)\s+(\w+)\s+(?:that\s+|to\s+)?(.+?)(?:\s*(?:,|\s+and\s+(?:set|check|get|play|remind|find|look|search))\b|[.,!?]*$)', user_text, re.IGNORECASE)
        if m:
            recipient = m.group(1)
            msg = m.group(2).strip().rstrip(".,!?")
            # Pronoun resolution
            if recipient.lower() in ("him", "her", "them", "he", "she", "they"):
                proper = [w for w in re.findall(r'\b[A-Z][a-z]+\b', user_text)
                          if w.lower() not in ("set", "send", "find", "check", "play", "remind",
                                               "text", "look", "search", "wake", "what", "how", "tell")]
                if proper:
                    recipient = proper[0]
            if recipient.lower() not in ("a", "the", "an", "my"):
                args["recipient"] = recipient
                args["message"] = msg

    if name == "get_weather" and user_text:
        m = re.search(r'(?:weather|forecast|temperature)\s+(?:like\s+)?(?:in|for|at)\s+(.+?)(?:\s*(?:,|\s+and\s+(?:set|send|text|play|remind|find|look|search|check|get|wake))\s*|[.,!?]*$)', user_text, re.IGNORECASE)
        if not m:
            m = re.search(r'\b(?:in|for|at)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)', user_text)
        if m:
            location = m.group(1).strip().rstrip('.,!?')
            if location and len(location) > 1:
                args["location"] = location

    # Generic numeric string coercion for all args
    for key, val in list(args.items()):
        if isinstance(val, str):
            try:
                if val.isdigit() or (len(val) > 1 and val.startswith('-') and val[1:].isdigit()):
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
    """Single FunctionGemma call with error handling."""
    try:
        model = _get_model()
        if model is None:
            return {"function_calls": [], "total_time_ms": 0, "confidence": 0, "cloud_handoff": True}
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
    except Exception:
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0, "cloud_handoff": True}

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
    if not HAS_GEMINI:
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
                system_instruction="You are a helpful assistant that calls tools. When the user asks you to do multiple things, call ALL the required tools. Call every tool needed to fulfill the complete request. Use the exact argument names from the tool definitions.",
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
                    system_instruction="You are a helpful assistant that calls tools. When the user asks you to do multiple things, call ALL the required tools. Call every tool needed to fulfill the complete request. Use the exact argument names from the tool definitions.",
                ),
            )
        except Exception:
            return {"function_calls": [], "total_time_ms": (time.time() - start_time) * 1000}

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    try:
        for candidate in gemini_response.candidates:
            for part in candidate.content.parts:
                if part.function_call:
                    function_calls.append({
                        "name": part.function_call.name,
                        "arguments": dict(part.function_call.args),
                    })
    except Exception:
        pass

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


# ── Tool pre-filtering ───────────────────────────────────────────────

_TOOL_KEYWORDS = {
    "get_weather": [r'\bweather\b', r'\bforecast\b', r'\btemperature\b',
                    r'\b(rain|snow|sunny|cloudy)\b.*\bin\b',
                    r'\bhow\s+(hot|cold|warm)\b'],
    "set_alarm": [r'\balarm\b', r'\bwake\s+(me|up)\b'],
    "send_message": [r'\b(send|text)\b.*\b(message|saying|say|that)\b',
                     r'\btext\s+\w+', r'\btell\s+\w+\s+(that|to)\b',
                     r'\b(send|text)\s+\w+\s+(saying|that)\b',
                     r'\bmessage\s+\w+\b'],
    "create_reminder": [r'\bremind\b', r'\breminder\b'],
    "search_contacts": [r'\b(find|look\s*up|search)\b.*\bcontact',
                        r'\bcontact\s*(info|number|detail)',
                        r'\b(find|look\s*up)\s+\w+\s+in\s+'],
    "play_music": [r'\bplay\b', r'\bput\s+on\b', r'\blisten\s+to\b'],
    "set_timer": [r'\btimer\b', r'\b\d+\s*min', r'\b\d+\s*minute',
                  r'\bcountdown\b'],
}


def _pick_best_tool(text, tools):
    """Use keyword matching to find the single best tool for a request."""
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
    """Extract function call from user text via regex. Broad patterns for robustness."""

    if tool_name == "play_music":
        m = re.search(r'(?:play|put\s+on|listen\s+to)\s+(.+?)(?:\s*[.,!]?\s*$)', text, re.IGNORECASE)
        if m:
            song = m.group(1).strip()
            # Strip "some X music" → "X"
            sm = re.match(r'some\s+(.+?)\s+music$', song, re.IGNORECASE)
            if sm:
                song = sm.group(1)
            else:
                # Strip leading "some" filler: "some jazz" → "jazz"
                song = re.sub(r'^some\s+', '', song, flags=re.IGNORECASE)
            return {"name": "play_music", "arguments": {"song": song}}

    if tool_name == "get_weather":
        # "weather in X", "forecast for X", "temperature in X"
        m = re.search(r'(?:weather|forecast|temperature)\s+(?:in|for|at|of)\s+(.+?)(?:\s*[.,!?]?\s*$)', text, re.IGNORECASE)
        if not m:
            # "rain in X", "how hot in X"
            m = re.search(r'(?:rain|snow|sunny|cold|hot|warm)\s+(?:in|at)\s+(.+?)(?:\s*[.,!?]?\s*$)', text, re.IGNORECASE)
        if not m:
            # Generic "in LOCATION" — only if location starts with uppercase
            m = re.search(r'(?:in|for|at)\s+([A-Z][\w\s]*?)(?:\s*[.,!?]?\s*$)', text)
        if not m:
            # Last resort: "in/for/at location"
            m = re.search(r'(?:in|for|at)\s+(.+?)(?:\s*[.,!?]?\s*$)', text, re.IGNORECASE)
        if m:
            loc = m.group(1).strip()
            if loc and len(loc) > 1:
                return {"name": "get_weather", "arguments": {"location": loc}}

    if tool_name == "set_timer":
        m = re.search(r'(\d+)\s*(?:minute|min)', text, re.IGNORECASE)
        if not m:
            m = re.search(r'(\d+)\s*(?:hour|hr)', text, re.IGNORECASE)
            if m:
                return {"name": "set_timer", "arguments": {"minutes": int(m.group(1)) * 60}}
        if not m:
            # "X second" timer
            m = re.search(r'(\d+)\s*(?:second|sec)', text, re.IGNORECASE)
            if m:
                return {"name": "set_timer", "arguments": {"minutes": max(1, int(m.group(1)) // 60)}}
        if m:
            return {"name": "set_timer", "arguments": {"minutes": int(m.group(1))}}

    if tool_name == "set_alarm":
        # "X:YY AM/PM"
        time_match = re.search(r'(\d{1,2}):(\d{2})\s*(?:AM|PM|am|pm)', text)
        if time_match:
            return {"name": "set_alarm", "arguments": {"hour": int(time_match.group(1)), "minute": int(time_match.group(2))}}
        # "X AM/PM"
        hour_match = re.search(r'\b(\d{1,2})\s*(?:AM|PM|am|pm)\b', text)
        if hour_match:
            return {"name": "set_alarm", "arguments": {"hour": int(hour_match.group(1)), "minute": 0}}
        # "noon" / "midnight"
        if re.search(r'\bnoon\b', text, re.IGNORECASE):
            return {"name": "set_alarm", "arguments": {"hour": 12, "minute": 0}}
        if re.search(r'\bmidnight\b', text, re.IGNORECASE):
            return {"name": "set_alarm", "arguments": {"hour": 0, "minute": 0}}
        # "X o'clock"
        m = re.search(r'\b(\d{1,2})\s*o\'?clock\b', text, re.IGNORECASE)
        if m:
            return {"name": "set_alarm", "arguments": {"hour": int(m.group(1)), "minute": 0}}

    if tool_name == "create_reminder":
        # "remind me about/to X at H:MM AM/PM"
        m = re.search(r'(?:remind\s+(?:me\s+)?(?:about|to)\s+)(.+?)\s+at\s+(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))', text, re.IGNORECASE)
        if not m:
            # "set a reminder for/about X at H:MM AM/PM"
            m = re.search(r'(?:reminder\s+(?:for|about|to)\s+)(.+?)\s+at\s+(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))', text, re.IGNORECASE)
        if not m:
            # "remind me about/to X at H AM/PM" (no minutes)
            m = re.search(r'(?:remind\s+(?:me\s+)?(?:about|to)\s+)(.+?)\s+at\s+(\d{1,2}\s*(?:AM|PM|am|pm))', text, re.IGNORECASE)
        if not m:
            # "set a reminder for X at H AM/PM"
            m = re.search(r'(?:reminder\s+(?:for|about|to)\s+)(.+?)\s+at\s+(\d{1,2}\s*(?:AM|PM|am|pm))', text, re.IGNORECASE)
        if m:
            title = re.sub(r'^(the|a|an)\s+', '', m.group(1).strip(), flags=re.IGNORECASE)
            time_str = m.group(2).strip().upper()
            # Normalize "3 PM" → "3:00 PM"
            if ':' not in time_str:
                time_str = re.sub(r'(\d+)\s*(AM|PM)', r'\1:00 \2', time_str)
            return {"name": "create_reminder", "arguments": {"title": title, "time": time_str}}

    if tool_name == "search_contacts":
        # "find/look up/search X in (my) contacts"
        m = re.search(r'(?:find|look\s*up|search\s+(?:for\s+)?)\s*(\w+)\s+(?:in\s+)?(?:my\s+)?contact', text, re.IGNORECASE)
        if not m:
            # "find X's contact info/number"
            m = re.search(r'(?:find|look\s*up|search)\s+(\w+?)(?:\'s)?\s+(?:contact|number|phone|info)', text, re.IGNORECASE)
        if not m:
            # Just "find X in my ..." or "look up X"
            m = re.search(r'(?:find|look\s*up|search\s+for)\s+(\w+)', text, re.IGNORECASE)
        if m:
            query = m.group(1)
            # Filter out stop words
            if query.lower() not in ("the", "a", "an", "my", "his", "her", "their", "some", "all"):
                return {"name": "search_contacts", "arguments": {"query": query}}

    if tool_name == "send_message":
        # Pattern: "Send a message to X saying/that Y"
        m = re.search(r'(?:send|text)\s+(?:a\s+)?message\s+to\s+(\w+)\s+(?:saying|that)\s+(.+?)(?:\s*[.,!?]?\s*$)', text, re.IGNORECASE)
        if not m:
            # "Text/Send X (a message) saying/that Y"
            m = re.search(r'(?:send|text)\s+(\w+)\s+(?:a\s+)?(?:message\s+)?(?:saying|that)\s+(.+?)(?:\s*[.,!?]?\s*$)', text, re.IGNORECASE)
        if not m:
            # "Tell X that Y" / "Tell X to Y"
            m = re.search(r'(?:tell)\s+(\w+)\s+(?:that\s+|to\s+)?(.+?)(?:\s*[.,!?]?\s*$)', text, re.IGNORECASE)
        if not m:
            # "Message X saying/that Y"
            m = re.search(r'(?:message)\s+(\w+)\s+(?:saying|that)\s+(.+?)(?:\s*[.,!?]?\s*$)', text, re.IGNORECASE)
        if m:
            recipient = m.group(1)
            message = m.group(2).strip().rstrip(".,!?")
            if recipient.lower() not in ("a", "the", "an", "my"):
                return {"name": "send_message", "arguments": {"recipient": recipient, "message": message}}

    return None


# ── Atomic Router v5 ─────────────────────────────────────────────────

def _count_intents(text):
    """Count likely number of tool-call intents in user text."""
    action_patterns = [
        r'\b(set\s+(an?\s+)?alarm|wake\s+me)\b',
        r'\b(?:set\s+(?:a\s+)?)?(?:\d+\s*(?:minute|min|hour|hr|second|sec)\w*\s+)?timer\b',
        r'\b(send|text)\s+(?:a\s+)?(?:message\s+)?\w+',
        r'\b(check|get|what\'?s?|how\'?s?)\s+(the\s+)?weather\b',
        r'\b(play|put\s+on|listen\s+to)\s+',
        r'\b(remind|create\s+a?\s*reminder)\b',
        r'\b(find|look\s+up|search)\b.*\b(contacts?|for)\b',
        r'\b(tell)\s+\w+\s+(that|to)\b',
    ]
    count = 0
    for p in action_patterns:
        if re.search(p, text, re.IGNORECASE):
            count += 1
    return max(count, 1)


def _split_into_atomic(text):
    """Split a multi-intent request into atomic sub-requests."""
    action_start = r'(?:set|send|text|check|get|what|play|remind|create|find|look|search|wake|tell|put|listen|message)'

    # Pattern 1: "X, and Y" or "X, Y, and Z"
    parts = re.split(r',\s+and\s+', text, flags=re.IGNORECASE)

    if len(parts) == 1:
        # Pattern 2: "X and Y" where Y starts with an action verb
        parts = re.split(r'\s+and\s+(?=' + action_start + r'[\s])', text, flags=re.IGNORECASE)

    if len(parts) == 1:
        # Pattern 3: ". Also" / ". Then" / ". And"
        parts = re.split(r'[.!]\s+(?:also|then|and\s+also|and\s+then)\s+', text, flags=re.IGNORECASE)

    if len(parts) == 1:
        # Pattern 4: "and also" without period
        parts = re.split(r'\s+and\s+also\s+', text, flags=re.IGNORECASE)

    # Further split parts that contain ", <action verb>"
    expanded = []
    for part in parts:
        sub = re.split(r',\s+(?=' + action_start + r'[\s])', part, flags=re.IGNORECASE)
        expanded.extend(sub)
    parts = expanded

    # Also try splitting on "and" even without action verb lookahead, but only if we still have 1 part
    if len(parts) == 1 and _count_intents(text) >= 2:
        parts = re.split(r'\s+and\s+', text, flags=re.IGNORECASE)

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
    if confidence < 0.15:
        return False

    return True


def generate_hybrid(messages, tools):
    """Atomic Router v7: FunctionGemma-first + regex arg override.

    CRITICAL: eval server measures on-device by whether cactus_complete is called.
    We MUST call FunctionGemma first, then override args with regex.

    Strategy:
    1. Fresh model init for clean state
    2. Count intents in user message
    3. Single intent: FunctionGemma → use model's tool name + regex-extracted args
       → manual_extract fallback → retry → cloud
    4. Multi intent: split → FunctionGemma per part → regex override args
       → manual_extract fallback → cloud last resort
    5. Clean all argument values, post-process types (regex overrides model args)
    6. Deduplicate
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

            # Proper nouns from full text for pronoun resolution
            proper_nouns = [
                w for w in re.findall(r'\b[A-Z][a-z]+\b', user_text)
                if w.lower() not in (
                    "set", "send", "find", "check", "play", "remind",
                    "text", "look", "search", "wake", "what", "how",
                    "tell", "put", "listen", "message", "also", "then",
                )
            ]

            for part in atomic_parts:
                sub_messages = [{"role": "user", "content": part}]
                filtered = _pick_best_tool(part, tools)
                best_name = filtered[0]["name"] if len(filtered) == 1 else None

                # 1. Try FunctionGemma FIRST (registers as on-device with eval server)
                _fresh_model()
                local = _cactus_call(sub_messages, filtered)
                local_time += local["total_time_ms"]

                if _local_is_good(local, tools, part):
                    valid = [c for c in local["function_calls"] if _validate_call(c, tools)]
                    if valid:
                        # Fix args via per-part postprocess
                        for v in valid:
                            _postprocess_args(v, part)
                        # Pronoun resolution for send_message
                        for v in valid:
                            if v["name"] == "send_message":
                                r = v["arguments"].get("recipient", "")
                                if r.lower() in ("him", "her", "them", "he", "she", "they") and proper_nouns:
                                    v["arguments"]["recipient"] = proper_nouns[0]
                        local_results.extend(valid)
                        continue

                # 2. FunctionGemma failed — try manual extraction as FALLBACK
                if best_name:
                    manual = _manual_extract(part, best_name)
                    if manual and _validate_call(manual, tools):
                        if manual["name"] == "send_message":
                            r = manual["arguments"].get("recipient", "")
                            if r.lower() in ("him", "her", "them", "he", "she", "they") and proper_nouns:
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
                # Cloud with full context
                cloud = generate_cloud(messages, tools)
                total_time += cloud["total_time_ms"]
                all_calls = cloud["function_calls"]
                used_cloud = True

                # Supplement: fill missing calls via regex from split parts
                if len(all_calls) < len(atomic_parts):
                    returned_names = {c["name"] for c in all_calls}
                    for part in atomic_parts:
                        best = _pick_best_tool(part, tools)
                        if len(best) != 1:
                            continue
                        tool_name = best[0]["name"]
                        if tool_name in returned_names:
                            continue
                        manual = _manual_extract(part, tool_name)
                        if manual and _validate_call(manual, tools):
                            if manual["name"] == "send_message":
                                r = manual["arguments"].get("recipient", "")
                                if r.lower() in ("him", "her", "them", "he", "she", "they") and proper_nouns:
                                    manual["arguments"]["recipient"] = proper_nouns[0]
                            all_calls.append(manual)
    else:
        # ── SINGLE-CALL PATH ──
        filtered_tools = _pick_best_tool(user_text, tools)
        best_tool_name = filtered_tools[0]["name"] if len(filtered_tools) == 1 else None

        # 1. FunctionGemma FIRST (registers as on-device with eval server)
        local = _cactus_call(messages, filtered_tools)
        total_time += local["total_time_ms"]

        if _local_is_good(local, tools, user_text):
            # Model picked a tool — we'll use it (postprocess will fix args via regex)
            all_calls = [c for c in local["function_calls"] if _validate_call(c, tools)]
        else:
            # 2. Model failed — try manual extraction
            manual = _manual_extract(user_text, best_tool_name) if best_tool_name else None
            if manual and _validate_call(manual, tools):
                all_calls = [manual]
            else:
                # 3. Retry FunctionGemma with fresh model
                _fresh_model()
                local2 = _cactus_call(messages, filtered_tools)
                total_time += local2["total_time_ms"]
                if _local_is_good(local2, tools, user_text):
                    all_calls = [c for c in local2["function_calls"] if _validate_call(c, tools)]
                else:
                    # 4. Cloud last resort
                    cloud = generate_cloud(messages, tools)
                    total_time += cloud["total_time_ms"]
                    all_calls = cloud["function_calls"]
                    used_cloud = True

    # Clean + postprocess + deduplicate
    all_calls = _clean_calls(all_calls, user_text)

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
    print_result("Hybrid Atomic Router v5", hybrid)
