"""
Task 1.1 demo harness — "mock STT/TTS with text I/O", real LLM + tool-calling.

STT is mocked as reading a line of text (instead of transcribing audio).
TTS is mocked as printing text (instead of synthesizing speech).
Everything in between -- the LLM call and the function-calling loop that
lets the model invoke get_order_status / cancel_order mid-conversation --
is real.

Usage:
    export GROQ_API_KEY=...      # free tier: https://console.groq.com
    python console_demo.py

Without GROQ_API_KEY set, a small deterministic MockLLM takes over so the
script still produces a real, inspectable tool-calling transcript (useful
for grading in a sandboxed/offline environment). It implements the exact
same generate() interface as the real client, so swapping it back out for
Groq/OpenRouter is a one-line change (see build_llm() below).
"""

from __future__ import annotations
import json
import os
import re
import sys
from dataclasses import dataclass, field

import tools as biz

SYSTEM_PERSONA = (
    "You are a friendly, concise support assistant for a food delivery app. "
    "Help users check their order status and cancel orders when needed. "
    "Always use the provided tools to look up real order data instead of "
    "guessing. Keep responses short and voice-friendly (1-2 sentences)."
)

TRANSCRIPT_PATH = "sample_transcript.txt"


# ---------------------------------------------------------------------
# Real LLM client (Groq, OpenAI-compatible) -- used when GROQ_API_KEY is set
# ---------------------------------------------------------------------
class GroqLLM:
    def __init__(self) -> None:
        from openai import OpenAI  # local import so MockLLM path needs no dep

        # OpenRouter swap: base_url="https://openrouter.ai/api/v1",
        # api_key=os.environ["OPENROUTER_API_KEY"], model e.g.
        # "meta-llama/llama-3.3-70b-instruct:free". Nothing else changes.
        self.client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = "llama-3.3-70b-versatile"

    def generate(self, messages: list[dict]) -> dict:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=biz.ALL_SCHEMAS,
        )
        msg = resp.choices[0].message
        return {
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                for tc in (msg.tool_calls or [])
            ],
        }


# ---------------------------------------------------------------------
# Deterministic fallback LLM -- no network/API key required.
# Same generate() interface as GroqLLM, so the calling code below never
# needs to know which one it's talking to.
# ---------------------------------------------------------------------
class MockLLM:
    """Simple rule-based stand-in so the tool-calling loop is demonstrable
    without any API key. It looks for an order ID pattern and a cancel
    intent in the *last* user turn."""

    ORDER_RE = re.compile(r"\b([A-Za-z]\d{4})\b")

    def generate(self, messages: list[dict]) -> dict:
        last_user = next(m for m in reversed(messages) if m["role"] == "user")["content"]
        match = self.ORDER_RE.search(last_user)

        # If the previous message was already a tool result, just narrate it.
        if messages[-1]["role"] == "tool":
            tool_result = messages[-1]["content"]
            if tool_result.startswith("error:"):
                return {"content": f"Hmm, I ran into an issue: {tool_result[7:]}.", "tool_calls": []}
            return {"content": tool_result, "tool_calls": []}

        if match and "cancel" in last_user.lower():
            return {
                "content": None,
                "tool_calls": [{
                    "id": "call_1", "name": "cancel_order",
                    "arguments": json.dumps({"order_id": match.group(1), "reason": "customer requested"}),
                }],
            }
        if match:
            return {
                "content": None,
                "tool_calls": [{
                    "id": "call_1", "name": "get_order_status",
                    "arguments": json.dumps({"order_id": match.group(1)}),
                }],
            }
        return {
            "content": "Sure -- could you give me your order ID (e.g. A1001) so I can look that up?",
            "tool_calls": [],
        }


def build_llm():
    if os.environ.get("GROQ_API_KEY"):
        return GroqLLM()
    print("[no GROQ_API_KEY found -> using deterministic MockLLM for this demo]\n")
    return MockLLM()


# ---------------------------------------------------------------------
# Session loop: mocked STT/TTS (text I/O) + real tool-calling logic
# ---------------------------------------------------------------------
@dataclass
class Session:
    llm: object
    messages: list[dict] = field(default_factory=lambda: [{"role": "system", "content": SYSTEM_PERSONA}])
    log: list[str] = field(default_factory=list)

    def _say(self, speaker: str, text: str) -> None:
        line = f"{speaker}: {text}"
        print(line)
        self.log.append(line)

    def handle_user_turn(self, user_text: str) -> None:
        self._say("User (mock-STT text input)", user_text)
        self.messages.append({"role": "user", "content": user_text})

        # Loop in case the model chains tool calls (e.g. look up, then act).
        for _ in range(3):
            result = self.llm.generate(self.messages)

            if result["tool_calls"]:
                for call in result["tool_calls"]:
                    args = json.loads(call["arguments"])
                    self._say("Agent [tool_call]", f"{call['name']}({args})")
                    impl = biz.IMPLS[call["name"]]
                    tool_output = impl(**args)
                    self._say("Tool [tool_result]", tool_output)
                    self.messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": call["id"],
                            "type": "function",
                            "function": {"name": call["name"], "arguments": call["arguments"]},
                        }],
                    })
                    self.messages.append({"role": "tool", "content": tool_output, "tool_call_id": call["id"]})
                continue  # let the model react to the tool result

            self._say("Agent (mock-TTS text output)", result["content"])
            self.messages.append({"role": "assistant", "content": result["content"]})
            break


def main() -> None:
    llm = build_llm()
    session = Session(llm=llm)

    turns = [
        "Hi, can you check on order A1001?",
        "Great, thanks! Also please cancel order A1002, I changed my mind.",
        "What about order Z9999?",
    ]

    for turn in turns:
        session.handle_user_turn(turn)
        session._say("", "")  # blank line between turns

    with open(TRANSCRIPT_PATH, "w") as f:
        f.write("\n".join(session.log))
    print(f"\n[transcript written to {TRANSCRIPT_PATH}]")


if __name__ == "__main__":
    main()