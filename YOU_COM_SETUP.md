# You.com OpenAI-Compatible Chat Proxy — Setup Guide

The Flux LoRA Bridge now includes a fully functional You.com chat proxy (`/v1/chat/completions`, `/v1/models`) that exposes Your.com Custom Agents as an OpenAI-compatible LLM backend for SillyTavern.

## Prerequisites

1. **You.com Account** with a paid subscription (Custom Agents require paid plan)
2. **API Key** from https://you.com/platform
3. **Custom Agent** created at https://you.com/agents

## Step 1: Create a Custom Agent on You.com

1. Go to https://you.com/agents
2. Click **Create Agent**
3. Fill in name: e.g. "SillyTavern RP Agent"
4. **Model selection: IMPORTANT** — Pick a specific model like:
   - `Claude 3.5 Sonnet`
   - `GPT-4o`
   - `Gemini Pro`
   - **DO NOT use "Auto"** (auto-switching breaks consistency)
5. **Instructions (optional):** Leave empty or very minimal. The bridge will pass SillyTavern's system prompt through.
6. **Web Search:** Disable (toggle off) for roleplay scenarios
7. Click **Save**
8. Copy the **Agent UUID** from the URL:
   ```
   https://you.com/agents?chatMode=user_mode_<YOUR_UUID_HERE>
   ```
   Copy just the UUID part.

## Step 2: Get Your You.com API Key

1. Go to https://you.com/platform
2. Navigate to **API Keys** or **Integrations**
3. Create or copy your API key
4. Keep it safe — don't commit to repos

## Step 3: Configure the Bridge

### Option A: Single Key (Simplest)

Edit `.env` in the bridge directory:

```bash
YOU_COM_API_KEY=ydc-sk-your-actual-key-here
YOU_COM_DEFAULT_AGENT=a67dd509-a4b2-4115-b43d-2bf897d39022
```

Restart the bridge:
```bash
python flux_lora_bridge.py
```

### Option B: Multiple Keys (for key rotation / higher throughput)

If you have multiple You.com accounts:

```bash
YOU_COM_API_KEYS=key1,key2,key3
YOU_COM_DEFAULT_AGENT=agent-uuid-1
```

The bridge will rotate through keys, auto-failing keys that return 401/403.

### Option C: Multiple Agents (Different Models)

Map model names to different agents:

```bash
YOU_COM_AGENT_MAP=claude-sonnet=uuid-1,gpt-4o=uuid-2,gemini=uuid-3
```

Then SillyTavern's model dropdown will show `claude-sonnet`, `gpt-4o`, `gemini`.

## Step 4: Configure SillyTavern

### In SillyTavern UI:

1. Open **Settings** (⚙️)
2. Go to **API Connections** or **LLM**
3. Select **API Type:** `Custom (OpenAI-compatible)` or `Chat Completions`
4. Fill in:
   - **Custom Endpoint:**
     - Local: `http://localhost:7861/v1`
     - Remote: `https://bridge.midnighttavern.online/v1`
   - **API Key:** (can be anything, e.g., `sk-dummy`, the bridge uses its own keys)
   - **Model:** Select from the dropdown (populated from `/v1/models`)
     - If you set `YOU_COM_DEFAULT_AGENT` only: shows "default"
     - If you set `YOU_COM_AGENT_MAP`: shows all mapped model names
5. Set **Context Size:** Match the underlying model
   - Claude Sonnet: ~4000-6000 tokens recommended
   - GPT-4o: ~8000 tokens
   - Gemini Pro: ~8000 tokens
6. Click **Test** to verify connection

### Example Configuration Screenshot (text):

```
API Type:          Custom (OpenAI-compatible)
Endpoint:          http://localhost:7861/v1
API Key:           (anything - bridge uses YOU_COM_API_KEY)
Model:             claude-sonnet    [dropdown ▼]
Context Size:      6000
Temperature:       1.0
```

## Step 5: Test the Proxy

### From Command Line:

**Non-streaming test:**
```bash
curl -X POST http://localhost:7861/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "stream": false,
    "messages": [
      {"role": "user", "content": "Say hello in 3 words."}
    ]
  }'
```

**Streaming test:**
```bash
curl -X POST http://localhost:7861/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Say hello in 3 words."}
    ]
  }' \
  -N  # disable buffering to see stream chunks
```

**Get available models:**
```bash
curl http://localhost:7861/v1/models
```

### In SillyTavern:

1. Open a chat
2. Send a message — the bridge should respond via the selected agent
3. Check logs (bridge console) for `[YouProxy]` messages

## Monitoring & Debugging

### Logs

The bridge logs all You.com proxy activity with the `[YouProxy]` prefix:

```
[YouProxy] Request: model=claude-sonnet agent=a67dd509-... stream=true input_len=156 hash=abc12345
[YouProxy] Response: key=...key1 tokens_approx=45
```

### Common Issues

| Issue | Fix |
|-------|-----|
| `401 Unauthorized` | Check API key in `.env`, verify it's active on you.com/platform |
| `Missing agent_id` | Set `YOU_COM_DEFAULT_AGENT` or `YOU_COM_AGENT_MAP` in `.env` |
| Response is `[Error: Could not parse...]` | The agent returned an unexpected format; check agent settings on you.com |
| Streaming ends abruptly | You.com timeout; try a simpler prompt or reduce context |
| Keys rotating frequently | Some keys may be rate-limited; check your.com/usage |

### Enable Verbose Token Logging

For debugging, set in `.env`:
```bash
DEBUG_STREAM_TAP=true
```

Then logs will show every streamed token:
```
[YouProxy TOKEN] Hello
[YouProxy TOKEN]  world
```

## Architecture

```
SillyTavern (Port 8000)
  │
  ├─ User sends message
  │  └→ Browser calls https://bridge.midnighttavern.online/v1/chat/completions
  │
Browser (AutoImageGen plugin)
  │
  └─→ Bridge (Port 7861)
       ├─ /v1/chat/completions (You.com proxy)
       │   ├─ Receives: OpenAI-format request
       │   ├─ Converts to: You.com agent format
       │   ├─ Calls: https://api.you.com/v1/agents/runs
       │   └─ Returns: OpenAI-format response (streaming or non-streaming)
       │
       ├─ /v1/models (dynamic model list)
       │   └─ Returns: Models from YOU_COM_AGENT_MAP
       │
       ├─ /sdapi/v1/txt2img (image generation — unchanged)
       │
       └─ [Other image providers: Runware, FAL, etc.]
```

## Limitations

- **Web Search:** If enabled in the agent, response will include web results (may make output longer)
- **Temperature/Max Tokens:** Configured in agent UI only, not via API parameters
- **Rate Limiting:** You.com applies per-minute and per-day limits; use multiple keys for higher throughput
- **Context Length:** Limited by the underlying model (Claude Sonnet ~200k, GPT-4o ~128k, etc.)

## Multi-Character Conversations

When using the bridge for multi-character roleplay:

1. Each character sends messages as `{"role": "user", "content": "[Character Name]\n...message..."}`
2. Responses come back as `{"role": "assistant", "content": "Response text"}`
3. The bridge maintains conversation history via SillyTavern's message context

The agent sees the full conversation history, so it can track all character interactions.

## Spending Tracking

Each API call to You.com costs credits. Monitor at:

1. **You.com Dashboard:** https://you.com/platform → Billing/Usage
2. **Bridge Logs:** Each response logs approximate token count
   - Rough cost: (input_tokens + output_tokens) × per_token_rate

For cost optimization:
- Enable prompt summarization (if using image generation too): reduces input tokens
- Use shorter system prompts
- Disable web search in agent config
- Consider rotating between agents (different model tiers cost differently)

## Support

If the proxy isn't working:

1. Check `.env` has `YOU_COM_API_KEY` and `YOU_COM_DEFAULT_AGENT` set
2. Verify agent exists at https://you.com/agents
3. Check bridge logs for `[YouProxy]` errors
4. Try the curl test above to isolate ST vs bridge issues
5. Confirm bridge is reachable at its URL (local or remote)

---

**Done!** Messages sent in SillyTavern will now route through the You.com proxy via the bridge.
