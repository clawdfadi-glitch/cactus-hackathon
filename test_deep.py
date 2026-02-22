#!/usr/bin/env python3
"""
Deep Test Suite for Atomic Router
Tests edge cases, unusual phrasings, and patterns the leaderboard might use.

Run: python test_deep.py
"""

import sys, os
sys.path.insert(0, "cactus/python/src")
os.environ["CACTUS_NO_CLOUD_TELE"] = "1"

import json, re
from main import (
    generate_hybrid, _count_intents, _split_into_atomic,
    _pick_best_tool, _manual_extract, _postprocess_args, _validate_call,
    _clean_calls,
)
from benchmark import compute_f1

# Reuse tool defs from benchmark
from benchmark import (
    TOOL_GET_WEATHER, TOOL_SET_ALARM, TOOL_SEND_MESSAGE,
    TOOL_CREATE_REMINDER, TOOL_SEARCH_CONTACTS, TOOL_PLAY_MUSIC, TOOL_SET_TIMER,
)

ALL_TOOLS = [
    TOOL_GET_WEATHER, TOOL_SET_ALARM, TOOL_SEND_MESSAGE,
    TOOL_CREATE_REMINDER, TOOL_SEARCH_CONTACTS, TOOL_PLAY_MUSIC, TOOL_SET_TIMER,
]

PASS = 0
FAIL = 0
ERRORS = []


def check(name, actual, expected, context=""):
    global PASS, FAIL, ERRORS
    if actual == expected:
        PASS += 1
        print(f"  \033[92mPASS\033[0m {name}")
    else:
        FAIL += 1
        msg = f"  \033[91mFAIL\033[0m {name}: got {actual!r}, expected {expected!r}"
        if context:
            msg += f" ({context})"
        print(msg)
        ERRORS.append(msg)


def check_f1(name, query, tools, expected_calls):
    """Run generate_hybrid and check F1 score."""
    global PASS, FAIL, ERRORS
    messages = [{"role": "user", "content": query}]
    result = generate_hybrid(messages, tools)
    f1 = compute_f1(result["function_calls"], expected_calls)
    source = result.get("source", "unknown")
    if f1 >= 0.99:
        PASS += 1
        print(f"  \033[92mPASS\033[0m {name} (F1={f1:.2f}, {source})")
    else:
        FAIL += 1
        pred_names = [c["name"] for c in result["function_calls"]]
        exp_names = [c["name"] for c in expected_calls]
        msg = f"  \033[91mFAIL\033[0m {name}: F1={f1:.2f} ({source}) predicted={pred_names} expected={exp_names}"
        print(msg)
        # Show arg diffs for debugging
        for c in result["function_calls"]:
            print(f"         got: {c['name']}({c.get('arguments', {})})")
        for c in expected_calls:
            print(f"         exp: {c['name']}({c.get('arguments', {})})")
        ERRORS.append(msg)


# ═══════════════════════════════════════════════════════════════════════
# 1. INTENT COUNTING TESTS
# ═══════════════════════════════════════════════════════════════════════
def test_intent_counting():
    print("\n\033[1m── Intent Counting ──\033[0m")

    # Single intents
    check("single_weather", _count_intents("What's the weather in Paris?"), 1)
    check("single_alarm", _count_intents("Set an alarm for 10 AM"), 1)
    check("single_message", _count_intents("Send a message to Alice saying hello"), 1)
    check("single_timer", _count_intents("Set a timer for 5 minutes"), 1)
    check("single_music", _count_intents("Play some jazz music"), 1)
    check("single_reminder", _count_intents("Remind me to buy milk"), 1)
    check("single_search", _count_intents("Find Bob in my contacts"), 1)

    # "send a message" should NOT double-count
    check("no_double_send_message", _count_intents("Send a message to John saying hello."), 1)
    check("no_double_text_msg", _count_intents("Text Sarah a message saying I'm on my way"), 1)

    # Multi intents
    check("two_weather_alarm", _count_intents("Check the weather in NYC and set an alarm for 7 AM"), 2)
    check("two_message_weather", _count_intents("Text Bob saying hi and get the weather in London"), 2)
    check("three_timer_music_remind",
          _count_intents("Set a 15 minute timer, play classical music, and remind me to stretch at 4 PM"), 3)
    check("three_text_weather_alarm",
          _count_intents("Text Emma saying good night, check the weather in Chicago, and set an alarm for 5 AM"), 3)
    check("two_search_send",
          _count_intents("Find Tom in my contacts and send him a message saying happy birthday"), 2)


# ═══════════════════════════════════════════════════════════════════════
# 2. SPLITTING TESTS
# ═══════════════════════════════════════════════════════════════════════
def test_splitting():
    print("\n\033[1m── Text Splitting ──\033[0m")

    parts = _split_into_atomic("Set an alarm for 7 AM and check the weather in London.")
    check("split_and_action", len(parts), 2)

    parts = _split_into_atomic("Set a timer for 20 minutes and play lo-fi beats.")
    check("split_timer_music", len(parts), 2)

    parts = _split_into_atomic("Text Emma saying good night, check the weather in Chicago, and set an alarm for 5 AM.")
    check("split_three_comma_and", len(parts), 3)

    parts = _split_into_atomic("Set a 15 minute timer, play classical music, and remind me to stretch at 4:00 PM.")
    check("split_three_actions", len(parts), 3)

    parts = _split_into_atomic("Find Tom in my contacts and send him a message saying happy birthday.")
    check("split_search_send", len(parts), 2)

    parts = _split_into_atomic("Remind me about groceries at 5:00 PM and text Lisa saying see you tonight.")
    check("split_remind_text", len(parts), 2)

    # Edge: shouldn't split single intent
    parts = _split_into_atomic("What's the weather in San Francisco?")
    check("no_split_single", len(parts), 1)

    parts = _split_into_atomic("Send a message to Alice saying hello and goodbye.")
    check("no_split_message_content_and", len(parts), 1, "should not split on 'and' in message content")


# ═══════════════════════════════════════════════════════════════════════
# 3. TOOL KEYWORD MATCHING
# ═══════════════════════════════════════════════════════════════════════
def test_keyword_matching():
    print("\n\033[1m── Keyword Matching ──\033[0m")

    # Weather variations
    r = _pick_best_tool("What's the weather in Paris?", ALL_TOOLS)
    check("kw_weather_whats", len(r), 1)
    check("kw_weather_whats_name", r[0]["name"], "get_weather")

    r = _pick_best_tool("How's the weather in Tokyo?", ALL_TOOLS)
    check("kw_weather_hows", r[0]["name"], "get_weather")

    # Alarm variations
    r = _pick_best_tool("Set an alarm for 7 AM", ALL_TOOLS)
    check("kw_alarm_set", r[0]["name"], "set_alarm")

    r = _pick_best_tool("Wake me up at 6 AM", ALL_TOOLS)
    check("kw_alarm_wake", r[0]["name"], "set_alarm")

    # Message variations
    r = _pick_best_tool("Text Dave saying hello", ALL_TOOLS)
    check("kw_message_text", r[0]["name"], "send_message")

    r = _pick_best_tool("Send a message to Alice saying hi", ALL_TOOLS)
    check("kw_message_send", r[0]["name"], "send_message")

    r = _pick_best_tool("Tell Bob that I'm coming", ALL_TOOLS)
    check("kw_message_tell", r[0]["name"], "send_message")

    # Timer
    r = _pick_best_tool("Set a timer for 10 minutes", ALL_TOOLS)
    check("kw_timer", r[0]["name"], "set_timer")

    # Music
    r = _pick_best_tool("Play Bohemian Rhapsody", ALL_TOOLS)
    check("kw_music_play", r[0]["name"], "play_music")

    r = _pick_best_tool("Listen to some jazz", ALL_TOOLS)
    check("kw_music_listen", r[0]["name"], "play_music")

    # Reminder
    r = _pick_best_tool("Remind me to call the dentist at 2 PM", ALL_TOOLS)
    check("kw_reminder", r[0]["name"], "create_reminder")

    # Search contacts
    r = _pick_best_tool("Find Bob in my contacts", ALL_TOOLS)
    check("kw_search", r[0]["name"], "search_contacts")

    r = _pick_best_tool("Look up Sarah in contacts", ALL_TOOLS)
    check("kw_search_lookup", r[0]["name"], "search_contacts")


# ═══════════════════════════════════════════════════════════════════════
# 4. MANUAL EXTRACTION TESTS
# ═══════════════════════════════════════════════════════════════════════
def test_manual_extract():
    print("\n\033[1m── Manual Extraction ──\033[0m")

    # Weather
    r = _manual_extract("What is the weather in San Francisco?", "get_weather")
    check("extract_weather_sf", r["arguments"]["location"], "San Francisco")

    r = _manual_extract("What's the weather like in London?", "get_weather")
    check("extract_weather_london", r["arguments"]["location"], "London")

    r = _manual_extract("How's the weather in New York?", "get_weather")
    check("extract_weather_ny", r["arguments"]["location"], "New York")

    # Alarm — various formats
    r = _manual_extract("Set an alarm for 10 AM.", "set_alarm")
    check("extract_alarm_10am_h", r["arguments"]["hour"], 10)
    check("extract_alarm_10am_m", r["arguments"]["minute"], 0)

    r = _manual_extract("Set an alarm for 7:30 AM.", "set_alarm")
    check("extract_alarm_730_h", r["arguments"]["hour"], 7)
    check("extract_alarm_730_m", r["arguments"]["minute"], 30)

    r = _manual_extract("Set an alarm for 6:45 AM.", "set_alarm")
    check("extract_alarm_645_h", r["arguments"]["hour"], 6)
    check("extract_alarm_645_m", r["arguments"]["minute"], 45)

    r = _manual_extract("Wake me up at 6 AM", "set_alarm")
    check("extract_alarm_wake_h", r["arguments"]["hour"], 6)

    r = _manual_extract("Set an alarm for noon", "set_alarm")
    check("extract_alarm_noon_h", r["arguments"]["hour"], 12)
    check("extract_alarm_noon_m", r["arguments"]["minute"], 0)

    r = _manual_extract("Set an alarm for midnight", "set_alarm")
    check("extract_alarm_midnight_h", r["arguments"]["hour"], 0)

    # Timer
    r = _manual_extract("Set a timer for 5 minutes.", "set_timer")
    check("extract_timer_5", r["arguments"]["minutes"], 5)

    r = _manual_extract("Set a timer for 15 minutes", "set_timer")
    check("extract_timer_15", r["arguments"]["minutes"], 15)

    r = _manual_extract("Set a 20 minute timer", "set_timer")
    check("extract_timer_20", r["arguments"]["minutes"], 20)

    # Music
    r = _manual_extract("Play Bohemian Rhapsody.", "play_music")
    check("extract_music_bohemian", r["arguments"]["song"], "Bohemian Rhapsody")

    r = _manual_extract("Play some jazz music", "play_music")
    check("extract_music_jazz", r["arguments"]["song"], "jazz")

    r = _manual_extract("Play lo-fi beats", "play_music")
    check("extract_music_lofi", r["arguments"]["song"], "lo-fi beats")

    r = _manual_extract("Play classical music", "play_music")
    check("extract_music_classical", r["arguments"]["song"], "classical music")

    r = _manual_extract("Play summer hits", "play_music")
    check("extract_music_summer", r["arguments"]["song"], "summer hits")

    # Send message
    r = _manual_extract("Send a message to Alice saying good morning.", "send_message")
    check("extract_msg_alice_r", r["arguments"]["recipient"], "Alice")
    check("extract_msg_alice_m", r["arguments"]["message"], "good morning")

    r = _manual_extract("Text Dave saying I'll be late", "send_message")
    check("extract_msg_dave_r", r["arguments"]["recipient"], "Dave")
    check("extract_msg_dave_m", r["arguments"]["message"], "I'll be late")

    r = _manual_extract("Tell Bob that I'm coming", "send_message")
    check("extract_msg_tell_r", r["arguments"]["recipient"], "Bob")

    r = _manual_extract("Text Emma saying good night", "send_message")
    check("extract_msg_emma_r", r["arguments"]["recipient"], "Emma")

    # Reminder
    r = _manual_extract("Remind me about the meeting at 3:00 PM.", "create_reminder")
    check("extract_remind_title", r["arguments"]["title"], "meeting")
    check("extract_remind_time", r["arguments"]["time"], "3:00 PM")

    r = _manual_extract("Remind me to call the dentist at 2:00 PM.", "create_reminder")
    check("extract_remind_dentist_t", r["arguments"]["title"], "call the dentist")

    r = _manual_extract("Remind me to take medicine at 7:00 AM", "create_reminder")
    check("extract_remind_medicine_t", r["arguments"]["title"], "take medicine")

    # Search contacts
    r = _manual_extract("Find Bob in my contacts.", "search_contacts")
    check("extract_search_bob", r["arguments"]["query"], "Bob")

    r = _manual_extract("Look up Sarah in my contacts", "search_contacts")
    check("extract_search_sarah", r["arguments"]["query"], "Sarah")

    r = _manual_extract("Look up Jake in my contacts", "search_contacts")
    check("extract_search_jake", r["arguments"]["query"], "Jake")


# ═══════════════════════════════════════════════════════════════════════
# 5. POSTPROCESS TESTS
# ═══════════════════════════════════════════════════════════════════════
def test_postprocess():
    print("\n\033[1m── Postprocessing ──\033[0m")

    # Alarm: always extract from user text, override model
    call = {"name": "set_alarm", "arguments": {"hour": 3, "minute": 9}}
    r = _postprocess_args(call, "Set an alarm for 10 AM")
    check("pp_alarm_override_h", r["arguments"]["hour"], 10)
    check("pp_alarm_override_m", r["arguments"]["minute"], 0)

    call = {"name": "set_alarm", "arguments": {"hour": 7, "minute": 99}}
    r = _postprocess_args(call, "Set an alarm for 7:30 AM")
    check("pp_alarm_730_h", r["arguments"]["hour"], 7)
    check("pp_alarm_730_m", r["arguments"]["minute"], 30)

    # Timer: always extract from user text
    call = {"name": "set_timer", "arguments": {"minutes": 99}}
    r = _postprocess_args(call, "Set a timer for 15 minutes")
    check("pp_timer_override", r["arguments"]["minutes"], 15)

    call = {"name": "set_timer", "arguments": {"minutes": 3}}
    r = _postprocess_args(call, "Set a timer for 20 minutes")
    check("pp_timer_20", r["arguments"]["minutes"], 20)

    # Music: always extract from user text
    call = {"name": "play_music", "arguments": {"song": "wrong song"}}
    r = _postprocess_args(call, "Play Bohemian Rhapsody")
    check("pp_music_override", r["arguments"]["song"], "Bohemian Rhapsody")

    # Music: "some X music" filler stripping
    call = {"name": "play_music", "arguments": {"song": "wrong"}}
    r = _postprocess_args(call, "Play some jazz music")
    check("pp_music_jazz", r["arguments"]["song"], "jazz")

    # Music in multi-intent: should NOT grab past "and"
    call = {"name": "play_music", "arguments": {"song": "wrong"}}
    r = _postprocess_args(call, "Play classical music, and remind me to stretch at 4:00 PM")
    check("pp_music_multi_truncate", r["arguments"]["song"], "classical music")

    call = {"name": "play_music", "arguments": {"song": "wrong"}}
    r = _postprocess_args(call, "Set a timer for 20 minutes and play lo-fi beats.")
    check("pp_music_after_and", r["arguments"]["song"], "lo-fi beats")


# ═══════════════════════════════════════════════════════════════════════
# 6. END-TO-END: BENCHMARK-STYLE TESTS (with F1)
# ═══════════════════════════════════════════════════════════════════════
def test_e2e_easy():
    print("\n\033[1m── E2E: Easy (alternative phrasings) ──\033[0m")

    check_f1("e2e_weather_howcold",
             "How cold is it in Denver?", ALL_TOOLS,
             [{"name": "get_weather", "arguments": {"location": "Denver"}}])

    check_f1("e2e_weather_forecast",
             "Give me the forecast for Seattle", ALL_TOOLS,
             [{"name": "get_weather", "arguments": {"location": "Seattle"}}])

    check_f1("e2e_alarm_830",
             "Set an alarm for 8:30 AM", [TOOL_SET_ALARM],
             [{"name": "set_alarm", "arguments": {"hour": 8, "minute": 30}}])

    check_f1("e2e_alarm_noon",
             "Set an alarm for noon", [TOOL_SET_ALARM],
             [{"name": "set_alarm", "arguments": {"hour": 12, "minute": 0}}])

    check_f1("e2e_timer_25",
             "Set a 25 minute timer", [TOOL_SET_TIMER],
             [{"name": "set_timer", "arguments": {"minutes": 25}}])

    check_f1("e2e_music_twosong",
             "Play Hotel California", [TOOL_PLAY_MUSIC],
             [{"name": "play_music", "arguments": {"song": "Hotel California"}}])

    check_f1("e2e_msg_tell",
             "Tell Sarah that dinner is ready", ALL_TOOLS,
             [{"name": "send_message", "arguments": {"recipient": "Sarah", "message": "dinner is ready"}}])

    check_f1("e2e_search_lookup",
             "Look up Mike in my contacts", [TOOL_SEARCH_CONTACTS],
             [{"name": "search_contacts", "arguments": {"query": "Mike"}}])


def test_e2e_medium():
    print("\n\033[1m── E2E: Medium (tool selection) ──\033[0m")

    check_f1("e2e_med_msg_among",
             "Send a message to Grace saying see you tomorrow",
             [TOOL_GET_WEATHER, TOOL_SEND_MESSAGE, TOOL_SET_TIMER, TOOL_PLAY_MUSIC],
             [{"name": "send_message", "arguments": {"recipient": "Grace", "message": "see you tomorrow"}}])

    check_f1("e2e_med_alarm_among",
             "Set an alarm for 11:15 AM",
             [TOOL_SEND_MESSAGE, TOOL_SET_ALARM, TOOL_PLAY_MUSIC, TOOL_GET_WEATHER],
             [{"name": "set_alarm", "arguments": {"hour": 11, "minute": 15}}])

    check_f1("e2e_med_timer_among",
             "Set a timer for 45 minutes",
             [TOOL_SET_ALARM, TOOL_SET_TIMER, TOOL_GET_WEATHER],
             [{"name": "set_timer", "arguments": {"minutes": 45}}])

    check_f1("e2e_med_remind_among",
             "Remind me to pick up dry cleaning at 4:30 PM",
             [TOOL_SET_ALARM, TOOL_CREATE_REMINDER, TOOL_PLAY_MUSIC, TOOL_SEND_MESSAGE],
             [{"name": "create_reminder", "arguments": {"title": "pick up dry cleaning", "time": "4:30 PM"}}])


def test_e2e_hard():
    print("\n\033[1m── E2E: Hard (multi-intent) ──\033[0m")

    check_f1("e2e_hard_alarm_music",
             "Set an alarm for 6 AM and play some rock music",
             ALL_TOOLS,
             [{"name": "set_alarm", "arguments": {"hour": 6, "minute": 0}},
              {"name": "play_music", "arguments": {"song": "rock"}}])

    check_f1("e2e_hard_timer_weather",
             "Set a timer for 10 minutes and check the weather in Boston",
             ALL_TOOLS,
             [{"name": "set_timer", "arguments": {"minutes": 10}},
              {"name": "get_weather", "arguments": {"location": "Boston"}}])

    check_f1("e2e_hard_msg_alarm",
             "Text Alex saying good morning and set an alarm for 8 AM",
             ALL_TOOLS,
             [{"name": "send_message", "arguments": {"recipient": "Alex", "message": "good morning"}},
              {"name": "set_alarm", "arguments": {"hour": 8, "minute": 0}}])

    check_f1("e2e_hard_three_tools",
             "Play pop music, set a timer for 30 minutes, and remind me to drink water at 3:00 PM",
             ALL_TOOLS,
             [{"name": "play_music", "arguments": {"song": "pop music"}},
              {"name": "set_timer", "arguments": {"minutes": 30}},
              {"name": "create_reminder", "arguments": {"title": "drink water", "time": "3:00 PM"}}])

    check_f1("e2e_hard_search_msg_pronoun",
             "Look up Maria in my contacts and send her a message saying call me back",
             ALL_TOOLS,
             [{"name": "search_contacts", "arguments": {"query": "Maria"}},
              {"name": "send_message", "arguments": {"recipient": "Maria", "message": "call me back"}}])

    check_f1("e2e_hard_alarm_remind",
             "Set an alarm for 5:30 AM and remind me to exercise at 6:00 AM",
             ALL_TOOLS,
             [{"name": "set_alarm", "arguments": {"hour": 5, "minute": 30}},
              {"name": "create_reminder", "arguments": {"title": "exercise", "time": "6:00 AM"}}])


# ═══════════════════════════════════════════════════════════════════════
# 7. EDGE CASES
# ═══════════════════════════════════════════════════════════════════════
def test_edge_cases():
    print("\n\033[1m── Edge Cases ──\033[0m")

    # Multi-word locations
    check_f1("edge_weather_multiword",
             "What's the weather in New York City?", [TOOL_GET_WEATHER],
             [{"name": "get_weather", "arguments": {"location": "New York City"}}])

    check_f1("edge_weather_los_angeles",
             "What is the weather in Los Angeles?", [TOOL_GET_WEATHER],
             [{"name": "get_weather", "arguments": {"location": "Los Angeles"}}])

    # Alarm edge: 12:00 PM
    check_f1("edge_alarm_12pm",
             "Set an alarm for 12:00 PM", [TOOL_SET_ALARM],
             [{"name": "set_alarm", "arguments": {"hour": 12, "minute": 0}}])

    # Alarm edge: 12:30 AM
    check_f1("edge_alarm_1230am",
             "Set an alarm for 12:30 AM", [TOOL_SET_ALARM],
             [{"name": "set_alarm", "arguments": {"hour": 12, "minute": 30}}])

    # Timer 1 minute
    check_f1("edge_timer_1",
             "Set a 1 minute timer", [TOOL_SET_TIMER],
             [{"name": "set_timer", "arguments": {"minutes": 1}}])

    # Message with special chars
    check_f1("edge_msg_apostrophe",
             "Text Lisa saying I'll be there at 5", ALL_TOOLS,
             [{"name": "send_message", "arguments": {"recipient": "Lisa", "message": "I'll be there at 5"}}])

    # Reminder with "to" phrasing
    check_f1("edge_remind_to",
             "Remind me to buy groceries at 1:00 PM", [TOOL_CREATE_REMINDER],
             [{"name": "create_reminder", "arguments": {"title": "buy groceries", "time": "1:00 PM"}}])

    # Reminder with hour-only time
    check_f1("edge_remind_hour_only",
             "Remind me to call mom at 5 PM", [TOOL_CREATE_REMINDER],
             [{"name": "create_reminder", "arguments": {"title": "call mom", "time": "5:00 PM"}}])


# ═══════════════════════════════════════════════════════════════════════
# 8. ORIGINAL 30 BENCHMARK CASES (regression check)
# ═══════════════════════════════════════════════════════════════════════
def test_regression():
    print("\n\033[1m── Regression: All 30 Benchmark Cases ──\033[0m")
    from benchmark import BENCHMARKS, compute_f1
    all_pass = True
    for case in BENCHMARKS:
        messages = case["messages"]
        result = generate_hybrid(messages, case["tools"])
        f1 = compute_f1(result["function_calls"], case["expected_calls"])
        source = result.get("source", "unknown")
        if f1 >= 0.99:
            PASS_REF = True
            print(f"  \033[92mPASS\033[0m {case['name']} (F1={f1:.2f}, {source})")
        else:
            all_pass = False
            PASS_REF = False
            print(f"  \033[91mFAIL\033[0m {case['name']}: F1={f1:.2f} ({source})")
            for c in result["function_calls"]:
                print(f"         got: {c['name']}({c.get('arguments', {})})")
            for c in case["expected_calls"]:
                print(f"         exp: {c['name']}({c.get('arguments', {})})")
        # Update global counters
        global PASS, FAIL
        if PASS_REF:
            PASS += 1
        else:
            FAIL += 1
    return all_pass


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\033[1m\033[96m")
    print("╔════════════════════════════════════════════════╗")
    print("║   Deep Test Suite — Atomic Router v5          ║")
    print("║   Team YOO · Cactus x DeepMind Hackathon      ║")
    print("╚════════════════════════════════════════════════╝")
    print("\033[0m")

    # Unit tests (fast, no model needed)
    test_intent_counting()
    test_splitting()
    test_keyword_matching()
    test_manual_extract()
    test_postprocess()

    # E2E tests (need model or regex path)
    test_e2e_easy()
    test_e2e_medium()
    test_e2e_hard()
    test_edge_cases()

    # Regression: all 30 original cases
    test_regression()

    # Summary
    total = PASS + FAIL
    pct = PASS / total * 100 if total > 0 else 0
    color = "\033[92m" if FAIL == 0 else "\033[91m"
    print(f"\n\033[1m{'═'*50}")
    print(f"  {color}RESULTS: {PASS}/{total} passed ({pct:.0f}%)\033[0m")
    if ERRORS:
        print(f"\033[91m  {len(ERRORS)} failures:\033[0m")
        for e in ERRORS:
            print(f"  {e}")
    print(f"\033[1m{'═'*50}\033[0m")

    sys.exit(0 if FAIL == 0 else 1)
