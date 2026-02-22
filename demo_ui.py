#!/usr/bin/env python3
"""
🌵 Cactus Hybrid Router — Interactive Demo UI
Team YOO | DeepMind x Cactus Hackathon 2026

A rich terminal UI for demonstrating the hybrid routing strategy.
Shows real-time routing decisions, timing, and tool call results.
"""

import json, os, sys, time, random

# Add cactus to path
sys.path.insert(0, "cactus/python/src")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.columns import Columns
from rich.markdown import Markdown
from rich import box
from rich.prompt import Prompt
from rich.rule import Rule
from rich.align import Align
from rich.style import Style
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

# ── Showcase presets ──────────────────────────────────────────────────
SHOWCASE = [
    # Easy — single tool
    ("☀️  Weather", "What's the weather like in Tokyo?"),
    ("⏰ Alarm", "Set an alarm for 7:30 AM"),
    ("🎵 Music", "Play some jazz music"),
    ("⏱️  Timer", "Set a timer for 10 minutes"),
    # Medium — tricky args
    ("💬 Message", "Send a message to Sarah saying I'll be late"),
    ("🔍 Contacts", "Find John in my contacts"),
    ("📝 Reminder", "Remind me about the meeting at 3:00 PM"),
    # Hard — multi-intent
    ("🔥 Multi x2", "Set an alarm for 6 AM and play some rock music"),
    ("🔥 Multi x3", "Text Mom saying happy birthday, set a timer for 30 minutes, and check the weather in Chicago"),
    ("💀 Multi x4", "Set an alarm for 7 AM, send a message to Dave saying good morning, play classical music, and what's the weather in Boston"),
]

TOOLS_DEF = [
    {"name": "set_alarm", "description": "Set an alarm", "parameters": {"type": "object", "properties": {"hour": {"type": "integer"}, "minute": {"type": "integer"}}, "required": ["hour", "minute"]}},
    {"name": "set_timer", "description": "Set a countdown timer", "parameters": {"type": "object", "properties": {"minutes": {"type": "integer"}}, "required": ["minutes"]}},
    {"name": "play_music", "description": "Play a song or genre", "parameters": {"type": "object", "properties": {"song": {"type": "string"}}, "required": ["song"]}},
    {"name": "get_weather", "description": "Get current weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}},
    {"name": "send_message", "description": "Send a text message", "parameters": {"type": "object", "properties": {"recipient": {"type": "string"}, "message": {"type": "string"}}, "required": ["recipient", "message"]}},
    {"name": "create_reminder", "description": "Create a reminder", "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "time": {"type": "string"}}, "required": ["title", "time"]}},
    {"name": "search_contacts", "description": "Search contacts", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
]

TOOL_EMOJI = {
    "set_alarm": "⏰", "set_timer": "⏱️", "play_music": "🎵",
    "get_weather": "☀️", "send_message": "💬", "create_reminder": "📝",
    "search_contacts": "🔍",
}


def format_call(call):
    """Format a function call as a pretty string."""
    name = call.get("name", "unknown")
    args = call.get("arguments", {})
    emoji = TOOL_EMOJI.get(name, "🔧")
    args_str = ", ".join(f"[cyan]{k}[/]=[yellow]{v}[/]" for k, v in args.items())
    return f"{emoji} [bold green]{name}[/]({args_str})"


def render_result(query, result, elapsed_wall):
    """Render a result panel with routing info."""
    calls = result.get("function_calls", [])
    total_ms = result.get("total_time_ms", 0)
    on_device = result.get("on_device_ratio", 0)
    used_cloud = result.get("used_cloud", False)

    # Build call list
    call_lines = []
    for c in calls:
        call_lines.append(format_call(c))

    if not call_lines:
        call_lines = ["[dim red]No function calls extracted[/]"]

    # Routing badge
    if on_device >= 1.0:
        route_badge = "[bold green]🟢 100% ON-DEVICE[/]"
    elif on_device > 0:
        route_badge = f"[bold yellow]🟡 {on_device*100:.0f}% on-device[/]"
    else:
        route_badge = "[bold red]🔴 CLOUD[/]"

    # Speed badge
    if total_ms < 100:
        speed_badge = f"[bold green]⚡ {total_ms:.0f}ms[/]"
    elif total_ms < 500:
        speed_badge = f"[bold yellow]🏃 {total_ms:.0f}ms[/]"
    else:
        speed_badge = f"[bold red]🐌 {total_ms:.0f}ms[/]"

    # Intent count
    n = len(calls)
    intent_badge = f"[bold]{n} tool{'s' if n != 1 else ''}[/]"

    header = f"{route_badge}  {speed_badge}  {intent_badge}"

    content = "\n".join([
        f"[bold white]Query:[/] [italic]{query}[/]",
        "",
        header,
        "",
        *call_lines,
    ])

    panel = Panel(
        content,
        title="[bold blue]🌵 Routing Result[/]",
        border_style="blue",
        padding=(1, 2),
    )
    console.print(panel)


def run_showcase():
    """Run all showcase examples in sequence."""
    from main import generate_hybrid

    console.print(Rule("[bold magenta]🎬 SHOWCASE MODE[/]"))
    console.print("[dim]Running all preset examples...[/]\n")

    stats = {"total": 0, "on_device": 0, "cloud": 0, "total_ms": 0, "total_calls": 0}

    for label, query in SHOWCASE:
        console.print(f"[bold cyan]━━━ {label} ━━━[/]")
        messages = [{"role": "user", "content": query}]

        with console.status("[bold green]Routing...", spinner="dots"):
            t0 = time.time()
            result = generate_hybrid(messages, TOOLS_DEF)
            elapsed = (time.time() - t0) * 1000

        render_result(query, result, elapsed)

        stats["total"] += 1
        stats["total_ms"] += result.get("total_time_ms", 0)
        stats["total_calls"] += len(result.get("function_calls", []))
        if result.get("on_device_ratio", 0) >= 1.0:
            stats["on_device"] += 1
        if result.get("used_cloud", False):
            stats["cloud"] += 1

        console.print()

    # Summary
    summary = Table(title="📊 Showcase Summary", box=box.ROUNDED, border_style="magenta")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right")
    summary.add_row("Total Queries", str(stats["total"]))
    summary.add_row("Total Tool Calls", str(stats["total_calls"]))
    summary.add_row("On-Device", f"[green]{stats['on_device']}/{stats['total']}[/]")
    summary.add_row("Cloud Fallbacks", f"[{'red' if stats['cloud'] else 'green'}]{stats['cloud']}[/]")
    summary.add_row("Avg Latency", f"{stats['total_ms']/max(stats['total'],1):.0f}ms")
    console.print(summary)


def run_interactive():
    """Interactive mode — type queries and see routing results."""
    from main import generate_hybrid

    console.print(Rule("[bold green]💬 INTERACTIVE MODE[/]"))
    console.print("[dim]Type a query (or 'quit' to exit, 'showcase' to run presets)[/]\n")

    while True:
        try:
            query = Prompt.ask("[bold cyan]🗣️  You[/]")
        except (KeyboardInterrupt, EOFError):
            break

        if not query.strip():
            continue
        if query.strip().lower() in ("quit", "exit", "q"):
            break
        if query.strip().lower() == "showcase":
            run_showcase()
            continue

        messages = [{"role": "user", "content": query}]

        with console.status("[bold green]🌵 Routing...", spinner="dots"):
            t0 = time.time()
            result = generate_hybrid(messages, TOOLS_DEF)
            elapsed = (time.time() - t0) * 1000

        render_result(query, result, elapsed)
        console.print()

    console.print("\n[bold green]👋 Bye![/]")


def run_benchmark_visual():
    """Run benchmark.py test cases with visual output."""
    from main import generate_hybrid

    console.print(Rule("[bold yellow]🧪 BENCHMARK MODE[/]"))

    # Load benchmark cases
    try:
        with open("benchmark.py") as f:
            src = f.read()
        # Extract test cases from benchmark.py
        import ast
        tree = ast.parse(src)
        # Find the test_cases list
        test_cases = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "test_cases":
                        test_cases = ast.literal_eval(node.value)
                        break
        if not test_cases:
            console.print("[red]Could not parse test cases from benchmark.py[/]")
            return
    except Exception as e:
        console.print(f"[red]Error loading benchmark: {e}[/]")
        return

    passed = 0
    failed = 0
    total_time = 0

    for i, tc in enumerate(test_cases):
        query = tc["input"]
        expected = tc["expected_calls"]
        difficulty = tc.get("difficulty", "medium")
        diff_color = {"easy": "green", "medium": "yellow", "hard": "red"}.get(difficulty, "white")

        messages = [{"role": "user", "content": query}]
        t0 = time.time()
        result = generate_hybrid(messages, TOOLS_DEF)
        elapsed = (time.time() - t0) * 1000
        total_time += elapsed

        actual_calls = result.get("function_calls", [])
        # Simple name-based check
        expected_names = sorted([c["name"] for c in expected])
        actual_names = sorted([c.get("name", "") for c in actual_calls])
        match = expected_names == actual_names

        status = "[green]✅[/]" if match else "[red]❌[/]"
        if match:
            passed += 1
        else:
            failed += 1

        on_device = result.get("on_device_ratio", 0)
        route = "🟢" if on_device >= 1.0 else ("🟡" if on_device > 0 else "🔴")

        console.print(f"  {status} [{diff_color}]{difficulty:6}[/] {route} {elapsed:6.0f}ms │ {query[:60]}")
        if not match:
            console.print(f"       [dim]expected: {expected_names}[/]")
            console.print(f"       [dim]got:      {actual_names}[/]")

    console.print()
    pct = passed / max(passed + failed, 1) * 100
    console.print(f"  [bold]Result: {passed}/{passed+failed} passed ({pct:.0f}%) | Avg: {total_time/max(passed+failed,1):.0f}ms[/]")


def main():
    # Banner
    banner = """
[bold blue]
  ╔═══════════════════════════════════════════════════╗
  ║  🌵  CACTUS HYBRID ROUTER  ·  Team YOO           ║
  ║  DeepMind x Cactus Compute Hackathon 2026         ║
  ╚═══════════════════════════════════════════════════╝
[/]"""
    console.print(banner)

    # Architecture overview
    arch = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    arch.add_column(style="bold cyan")
    arch.add_column()
    arch.add_row("Strategy", "Atomic Router v5 — FunctionGemma-first")
    arch.add_row("On-Device", "FunctionGemma 270M (tool selection + args)")
    arch.add_row("Fallback", "Regex extraction → FunctionGemma retry → Gemini 2.5 Flash")
    arch.add_row("Multi-Intent", "NLP split → route each atom independently")
    console.print(Panel(arch, title="[bold]Architecture[/]", border_style="dim"))

    # Menu
    console.print()
    console.print("[bold]Choose a mode:[/]")
    console.print("  [cyan]1[/] · 💬 Interactive — type queries freely")
    console.print("  [cyan]2[/] · 🎬 Showcase — run preset examples")
    console.print("  [cyan]3[/] · 🧪 Benchmark — run all 30 test cases")
    console.print()

    choice = Prompt.ask("[bold]Mode", choices=["1", "2", "3"], default="1")

    if choice == "1":
        run_interactive()
    elif choice == "2":
        run_showcase()
    elif choice == "3":
        run_benchmark_visual()


if __name__ == "__main__":
    main()
