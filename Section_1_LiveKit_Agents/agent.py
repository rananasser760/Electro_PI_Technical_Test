"""
Task 1.1 — minimal LiveKit voice agent, real SDK shape.

Run this against an actual LiveKit server + provider keys:
    export LIVEKIT_URL=...
    export LIVEKIT_API_KEY=...
    export LIVEKIT_API_SECRET=...
    export GROQ_API_KEY=...          # free LLM provider (OpenAI-compatible)
    export DEEPGRAM_API_KEY=...      # optional real STT
    export CARTESIA_API_KEY=...      # optional real TTS
    python agent.py dev

If you don't have STT/TTS keys, this file also shows exactly where to
drop in the mock text-based STT/TTS from `console_demo.py` instead
(see the comment in `entrypoint`) -- the pipeline wiring doesn't change,
only the two constructor arguments do. That decoupling is the point
of Task 1.2.
"""

from __future__ import annotations
import logging
import os

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli, function_tool
from livekit.plugins import openai, silero, deepgram, cartesia

import tools as biz

logger = logging.getLogger("food-delivery-agent")

SYSTEM_PERSONA = (
    "You are a friendly, concise support assistant for a food delivery app. "
    "Help users check their order status and cancel orders when needed. "
    "Always use the provided tools to look up real order data instead of "
    "guessing. Keep responses short and voice-friendly (1-2 sentences)."
)


class SupportAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PERSONA)

    # --- Tool 1 (required by the assignment) ---------------------------
    @function_tool
    async def get_order_status(self, order_id: str) -> str:
        """Look up the current status and ETA of a food delivery order.

        Args:
            order_id: The customer's order ID, e.g. "A1001".
        """
        result = biz.get_order_status_impl(order_id)
        logger.info("get_order_status(%s) -> %s", order_id, result)
        return result

    # --- Tool 2 (shows how to add a second tool safely) -----------------
    @function_tool
    async def cancel_order(self, order_id: str, reason: str) -> str:
        """Cancel a food delivery order if it hasn't shipped yet.

        Args:
            order_id: The customer's order ID, e.g. "A1002".
            reason: Why the customer wants to cancel.
        """
        result = biz.cancel_order_impl(order_id, reason)
        logger.info("cancel_order(%s, %s) -> %s", order_id, reason, result)
        return result


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    session = AgentSession(
        # STT: swap deepgram.STT() for any other provider, or for the
        # text-based MockSTT shown in console_demo.py, with zero changes
        # anywhere else in this file (see Task 1.2 write-up in README).
        stt=deepgram.STT(model="nova-3"),
        # LLM: Groq's OpenAI-compatible endpoint (free tier). Swapping to
        # OpenRouter is a one-line base_url/model change -- see README.
        llm=openai.LLM(
            model="llama-3.3-70b-versatile",
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_API_KEY"],
        ),
        # TTS: swap cartesia.TTS() the same way as STT above.
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
    )

    await session.start(agent=SupportAgent(), room=ctx.room)

    # Greet the caller so there's an initial turn to react to.
    await session.generate_reply(
        instructions="Greet the caller and ask how you can help with their order."
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))