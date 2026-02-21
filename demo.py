"""
demo.py — Voice-to-Action Demo for Cactus x DeepMind Hackathon
Team YOO — Atomic Router

A terminal-based demo that shows:
1. Voice input → cactus_transcribe → text
2. Text → Atomic Router → tool calls (100% on-device)
3. Tool execution simulation with real-time feedback

Run: python demo.py
"""

import sys
sys.path.insert(0, "cactus/python/src")

import json, os, time, re
from main import generate_hybrid, _get_model, _fresh_model

# ANSI colors
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# ── Tool execution simulator ─────────────────────────────────────────

TOOL_RESPONSES = {
    "get_weather": lambda args: f"☀️ Weather in {args.get('location', '?')}: 72°F, Partly Cloudy, Humidity 45%",
    "set_alarm": lambda args: f"⏰ Alarm set for {args.get('hour', '?')}:{args.get('minute', 0):02d}",
    "send_message": lambda args: f"💬 Message sent to {args.get('recipient', '?')}: \"{args.get('message', '')}\"",
    "create_reminder": lambda args: f"📌 Reminder created: \"{args.get('title', '?')}\" at {args.get('time', '?')}",
    "search_contacts": lambda args: f"👤 Found contact: {args.get('query', '?')} — +1 (555) 123-4567",
    "play_music": lambda args: f"🎵 Now playing: {args.get('song', '?')}",
    "set_timer": lambda args: f"⏱️ Timer set for {args.get('minutes', '?')} minutes",
}

# ── All supported tools ──────────────────────────────────────────────

ALL_TOOLS = [
    {"name": "get_weather", "description": "Get current weather for a location",
     "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City name"}}, "required": ["location"]}},
    {"name": "set_alarm", "description": "Set an alarm for a given time",
     "parameters": {"type": "object", "properties": {"hour": {"type": "integer", "description": "Hour"}, "minute": {"type": "integer", "description": "Minute"}}, "required": ["hour", "minute"]}},
    {"name": "send_message", "description": "Send a message to a contact",
     "parameters": {"type": "object", "properties": {"recipient": {"type": "string", "description": "Recipient name"}, "message": {"type": "string", "description": "Message content"}}, "required": ["recipient", "message"]}},
    {"name": "create_reminder", "description": "Create a reminder with a title and time",
     "parameters": {"type": "object", "properties": {"title": {"type": "string", "description": "Reminder title"}, "time": {"type": "string", "description": "Time for the reminder"}}, "required": ["title", "time"]}},
    {"name": "search_contacts", "description": "Search for a contact by name",
     "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Name to search for"}}, "required": ["query"]}},
    {"name": "play_music", "description": "Play a song or playlist",
     "parameters": {"type": "object", "properties": {"song": {"type": "string", "description": "Song or playlist name"}}, "required": ["song"]}},
    {"name": "set_timer", "description": "Set a countdown timer",
     "parameters": {"type": "object", "properties": {"minutes": {"type": "integer", "description": "Number of minutes"}}, "required": ["minutes"]}},
]


def print_banner():
    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════╗
║          🌵 ATOMIC ROUTER — Team YOO 🌵              ║
║     Local-First Agentic Tool Calling Demo            ║
║                                                      ║
║  FunctionGemma (270M) + Regex Intelligence Layer     ║
║  → 97%+ accuracy, 100% on-device, <200ms avg        ║
╚══════════════════════════════════════════════════════╝{RESET}
""")


def process_request(text):
    """Process a user request through the Atomic Router."""
    print(f"\n{BLUE}{'─'*55}{RESET}")
    print(f"{BOLD}📝 Input:{RESET} {text}")
    print(f"{BLUE}{'─'*55}{RESET}")
    
    messages = [{"role": "user", "content": text}]
    
    start = time.time()
    result = generate_hybrid(messages, ALL_TOOLS)
    elapsed = (time.time() - start) * 1000
    
    source = result.get("source", "unknown")
    source_icon = "🟢" if "on-device" in source else "☁️"
    
    print(f"\n  {DIM}Router:{RESET} {source_icon} {source} | {elapsed:.0f}ms total")
    
    if not result["function_calls"]:
        print(f"  {RED}No tool calls generated.{RESET}")
        return
    
    print(f"  {DIM}Actions:{RESET} {len(result['function_calls'])} tool call(s)\n")
    
    for i, call in enumerate(result["function_calls"], 1):
        name = call["name"]
        args = call.get("arguments", {})
        
        # Show the tool call
        print(f"  {YELLOW}⚡ [{i}] {name}{RESET}")
        for k, v in args.items():
            print(f"     {DIM}{k}:{RESET} {v}")
        
        # Simulate execution
        executor = TOOL_RESPONSES.get(name)
        if executor:
            response = executor(args)
            print(f"     {GREEN}→ {response}{RESET}")
        print()


def demo_voice(audio_path=None):
    """Demo voice-to-action pipeline."""
    if audio_path:
        try:
            from cactus import cactus_init, cactus_transcribe, cactus_destroy
            whisper = cactus_init("cactus/weights/whisper-small")
            prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
            response = json.loads(cactus_transcribe(whisper, audio_path, prompt=prompt))
            cactus_destroy(whisper)
            text = response.get("response", "")
            print(f"\n{CYAN}🎤 Transcribed:{RESET} {text}")
            process_request(text)
        except Exception as e:
            print(f"{RED}Voice error: {e}{RESET}")
    else:
        print(f"\n{YELLOW}No audio file provided. Use: python demo.py --voice audio.wav{RESET}")


def run_showcase():
    """Run through showcase examples."""
    examples = [
        # Easy
        "What's the weather in San Francisco?",
        "Set an alarm for 7:30 AM.",
        "Play Bohemian Rhapsody.",
        # Medium  
        "Text Dave saying I'll be late.",
        "Remind me to call the dentist at 2:00 PM.",
        # Hard — multi-intent
        "Send a message to Bob saying hi and get the weather in London.",
        "Set a 15 minute timer, play classical music, and remind me to stretch at 4:00 PM.",
        "Text Emma saying good night, check the weather in Chicago, and set an alarm for 5 AM.",
    ]
    
    for ex in examples:
        process_request(ex)
        time.sleep(0.3)


def interactive_mode():
    """Interactive REPL."""
    print(f"{DIM}Type a request (or 'quit' to exit, 'demo' for showcase):{RESET}\n")
    while True:
        try:
            text = input(f"{BOLD}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Goodbye!{RESET}")
            break
        
        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            break
        if text.lower() == "demo":
            run_showcase()
            continue
        
        process_request(text)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Atomic Router Demo")
    parser.add_argument("--showcase", action="store_true", help="Run showcase examples")
    parser.add_argument("--voice", type=str, help="Audio file for voice-to-action")
    parser.add_argument("--interactive", action="store_true", help="Interactive REPL mode")
    args = parser.parse_args()
    
    print_banner()
    
    if args.voice:
        demo_voice(args.voice)
    elif args.showcase:
        run_showcase()
    else:
        interactive_mode()
