"""Terminal progress for agent runs: Claude's reasoning summaries, tool calls and results."""
import json
import time
from typing import Any, Callable

from claude_agent_sdk import AssistantMessage, ResultMessage, UserMessage


def console_printer(quiet: bool = False, width: int = 300) -> Callable[[Any], None]:
    start = time.monotonic()

    def show(message: Any) -> None:
        if quiet:
            return
        t = f"[{time.monotonic() - start:6.1f}s]"
        if isinstance(message, AssistantMessage):
            for block in message.content:
                kind = type(block).__name__
                if kind == "ThinkingBlock" and block.thinking.strip():
                    print(f"{t} thinks: {block.thinking.strip()[:width]}")
                elif kind == "TextBlock" and block.text.strip():
                    print(f"{t} says:   {block.text.strip()[:width]}")
                elif kind == "ToolUseBlock":
                    print(f"{t} calls:  {block.name.split('__')[-1]} {json.dumps(block.input)[:160]}")
        elif isinstance(message, UserMessage) and isinstance(message.content, list):
            for block in message.content:
                if type(block).__name__ == "ToolResultBlock":
                    content = block.content if isinstance(block.content, list) else [
                        {"type": "text", "text": str(block.content)}]
                    texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                    first = texts[0].splitlines()[0] if texts and texts[0] else "(image)"
                    images = sum(c.get("type") == "image" for c in content)
                    print(f"{t} result: [{'x' if block.is_error else 'ok'}] {first[:150]}"
                          + (f" +{images} image" if images else ""))
        elif isinstance(message, ResultMessage):
            print(f"{t} finished: {message.subtype}, {message.num_turns} turns, "
                  f"CLI cost estimate ${message.total_cost_usd or 0:.4f}")
    return show
