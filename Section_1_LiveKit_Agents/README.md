# Food Delivery Voice Support Agent (LiveKit Agents)
 
This is my submission for Section 1 — Task 1.1 and the Task 1.2 bonus.
 
## What's in here
 
- `agent.py` – the real thing. Uses `livekit-agents` with an `AgentSession`
  (STT → LLM → TTS), a `SupportAgent` class with a persona, and two tools:
  `get_order_status` and `cancel_order`. LLM is Groq, since it's free and
  OpenAI-compatible. Needs a LiveKit server to actually run.
- `agent_openrouter.py` – exact same agent, just the LLM swapped to
  OpenRouter instead of Groq (Task 1.2 stretch, see below).
- `console_demo.py` – a text-only version so you can see the tool-calling
  logic work without spinning up LiveKit or a mic. STT/TTS here are just
  stdin/stdout, but the LLM call and the function-calling loop are real.
  If `GROQ_API_KEY` isn't set it falls back to a small rule-based mock LLM
  so the demo still runs offline.
- `tools.py` – the actual tool logic + JSON schemas, shared by both
  `agent.py` and `console_demo.py` so there's only one copy of the logic.
- `sample_transcript.txt` – output from an actual run of `console_demo.py`.
## Running it
 
Easiest way to see it working, no LiveKit account needed:
```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key      # console.groq.com, free
python console_demo.py
```
 
To run the full voice pipeline you need a LiveKit Cloud project plus
Deepgram (STT) and Cartesia (TTS) keys, both free tier:
```bash
export LIVEKIT_URL=... LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=...
export GROQ_API_KEY=...
export DEEPGRAM_API_KEY=...
export CARTESIA_API_KEY=...
python agent.py dev
```
Then connect from [agents-playground.livekit.io](https://agents-playground.livekit.io)
and talk to it. I actually got this running end to end — mic in, real reply
out, tool calls firing when I ask about an order.
 
## Transcript
 
```
User (mock-STT text input): Hi, can you check on order A1001?
Agent [tool_call]: get_order_status({'order_id': 'A1001'})
Tool [tool_result]: Order A1001 from Sushi Go is currently 'out_for_delivery', estimated delivery in 12 minutes.
Agent (mock-TTS text output): Order A1001 from Sushi Go is currently 'out_for_delivery', estimated delivery in 12 minutes.
 
User (mock-STT text input): Great, thanks! Also please cancel order A1002, I changed my mind.
Agent [tool_call]: cancel_order({'order_id': 'A1002', 'reason': 'customer requested'})
Tool [tool_result]: error: cancellation service temporarily unavailable, please try again
Agent (mock-TTS text output): Hmm, I ran into an issue: cancellation service temporarily unavailable, please try again.
 
User (mock-STT text input): What about order Z9999?
Agent [tool_call]: get_order_status({'order_id': 'Z9999'})
Tool [tool_result]: error: no order found with id 'Z9999'
Agent (mock-TTS text output): Hmm, I ran into an issue: no order found with id 'Z9999'.
```
 
Worth pointing out: the middle turn hit a *real* simulated failure
(`cancel_order` randomly fails ~20% of the time to mimic a flaky backend),
and the agent handled it gracefully instead of pretending it worked. The
last turn is a bad order ID it correctly says doesn't exist. Neither of
those was scripted for this specific run, they just happened.
 
---
 
## Write-up
 
### Barge-in / handling interruptions
 
`AgentSession` is already listening continuously through STT while TTS is
playing, so it's not like the user has to wait for the agent to finish
talking before their speech even registers. What I'd add on top of the
default setup:
 
- Use the VAD (already wired in via `silero.VAD.load()`) to detect the
  exact moment the user starts talking, not just when STT finally produces
  a transcript.
- When that happens, kill the in-flight TTS playback and drop any queued
  audio immediately rather than letting the current sentence finish.
- Trim the conversation history to what was *actually spoken out loud*
  before the cut, not the full sentence the agent intended to say — so the
  next LLM turn isn't reasoning about words the user never heard.
- Add a tiny cooldown after resuming so something like a cough doesn't
  keep re-triggering interrupts.
`AgentSession` has hooks for this already (interruption events, the
`allow_interruptions` setting), so most of the real work is tuning how
sensitive the VAD is and deciding how much of an interrupted reply to
keep vs throw away.
 
### Adding a second tool safely
 
I actually went ahead and added a second tool (`cancel_order`) instead of
just describing it, since it's more convincing to show the pattern than
explain it. A few things I'd stick to for any future tool:
 
- Keep the schema narrow and explicit — required fields spelled out, one
  clear action per tool, instead of a single tool with a "mode" argument
  that tries to do everything.
- Validate before doing anything with side effects. `cancel_order_impl`
  checks the order exists and is actually cancellable *before* touching
  it, and just returns a plain `"error: ..."` string instead of raising —
  that way the LLM gets something it can read and explain to the user
  instead of the whole turn crashing.
- I made `cancel_order` fail randomly on purpose (about 1 in 5 calls) to
  simulate a flaky downstream service, so the agent has to deal with "the
  tool ran but didn't actually succeed," not just bad input. The system
  prompt tells it to trust tool output over guessing, so on a failure it
  apologizes and can suggest retrying instead of claiming success.
- For anything that touches money or really mutates state, I'd also pass
  an idempotency key through the tool args so a retried call after a
  timeout doesn't double-cancel or double-charge.
### Task 1.2 — swapping a pipeline component
 
This is really the whole point of `AgentSession(stt=..., llm=..., tts=...)`
— the `SupportAgent` class and its tools don't know or care which vendor
is plugged in underneath. Swapping STT is one line:
 
```python
# before
stt=deepgram.STT(model="nova-3"),
 
# after, e.g. AssemblyAI
from livekit.plugins import assemblyai
stt=assemblyai.STT(),
```
 
Same deal for TTS:
 
```python
# before
tts=cartesia.TTS(),
 
# after, e.g. ElevenLabs
from livekit.plugins import elevenlabs
tts=elevenlabs.TTS(voice="Rachel"),
```
 
I also actually did this for the LLM side rather than just describing it —
`agent_openrouter.py` is a copy of `agent.py` with only the `llm=` block
changed, Groq to OpenRouter:
 
```python
# Groq
llm=openai.LLM(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)
 
# OpenRouter
llm=openai.LLM(
    model="meta-llama/llama-3.3-70b-instruct:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
```
 
Both are OpenAI-compatible, so it's really just a different `base_url` +
key + model string. Everything else — persona, tools, STT, TTS — stays
exactly the same file for file. That's the decoupling this task is
checking for.