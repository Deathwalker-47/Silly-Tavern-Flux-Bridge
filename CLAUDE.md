# CLAUDE.md — Flux LoRA Bridge for SillyTavern

## Project Overview

FastAPI bridge (v3.0.0) between SillyTavern and multiple Flux image generation providers.
Exposes an AUTOMATIC1111-compatible API (`/sdapi/v1/txt2img`) so SillyTavern can request
image generation without knowing about the underlying providers.

**Live at:** `https://midnighttavern.online` (SillyTavern) + `https://bridge.midnighttavern.online` (bridge, nginx proxied)

## Architecture

```
Browser (SillyTavern UI)
  └→ AutoImageGen plugin (JS, runs in browser)
       └→ POST https://bridge.midnighttavern.online/sdapi/v1/txt2img
            └→ flux_lora_bridge.py (FastAPI, port 7861)
                 ├→ DeepSeek V3 Summarizer (strips RP narration → visual prompt, 1-3s)
                 ├→ LoRA Keyword Matcher (master_lora_dict.json)
                 └→ Provider Fallback Chain:
                      1. Runware (PRIMARY) — ~10s, JPEG, 12 LoRAs max
                      2. HF ZeroGPU (Gradio) — ~22s, WEBP, 10 LoRAs max
                      3. Wavespeed (queue-based, needs polling) — ~8s, JPEG, 4 LoRAs max
                      4. FAL (queue-based, needs polling) — ~5s, JPEG, 3 LoRAs max
                      5. Together AI (SDK v2, image_loras param) — ~8s, JPEG, 2 LoRAs max
                      6. Pixel Dojo (aspect_ratio based, 1 LoRA) — ~9s, PNG
```

## Key Files

| File | Purpose |
|------|---------|
| `flux_lora_bridge.py` | Main server — all providers, LoRA manager, summarizer, API |
| `master_lora_dict.json` | LoRA definitions: keywords, URLs, weights, prepend/append prompts |
| `runware_lora_mapping.json` | Auto-generated cache: HF URL → Runware AIR ID |
| `silly-tavern-pluggin/index.js` | Browser-side auto-image plugin for SillyTavern |
| `silly-tavern-pluggin/settings.json` | Plugin defaults |
| `tests/test_provider_response_parsing.py` | Unit tests for provider response parsing |
| `test_providers_live.py` | Live integration test (hits real APIs, gitignored) |
| `.env` | API keys (gitignored, never commit) |
| `env.example` | Template for .env |

## Running

```bash
# Install deps
pip install -r requirements.txt

# Create .env from env.example, fill in keys
cp env.example .env

# Start bridge
python flux_lora_bridge.py

# Run unit tests
python -m pytest tests/ -v

# Run live provider tests (requires .env with real keys)
python test_providers_live.py
```

## Infrastructure

- **Server:** SillyTavern on port 8000, bridge on port 7861
- **Nginx:** 443 → 8000 (SillyTavern), bridge.midnighttavern.online → 7861
- **Critical:** The AutoImageGen plugin runs in the BROWSER. It must call the bridge
  via a publicly reachable URL (not localhost). The plugin reads the SD Web UI URL from
  SillyTavern's built-in Image Generation settings (`extensionSettings.sd.auto_url`).

## Provider-Specific Notes

### Runware (PRIMARY)
- Synchronous API, returns image URL immediately
- Supports custom LoRA upload via `modelUpload` task
- LoRAs resolved through `runware_lora_mapping.json` cache
- AIR format: `deathwalker:{hash}@1`

### HF ZeroGPU
- Uses `gradio_client` v2.x (`token=` not `hf_token=`)
- Space API endpoint: `/run_lora`
- Required params: `prompt`, `image_url` (empty for txt2img), `lora_strings_json` (JSON string)
- Also requires empty R2 params: `account_id`, `access_key`, `secret_key`, `bucket`
- Returns tuple: `(image_data_dict, result_json_str)`

### Wavespeed
- **Queue-based API** — initial POST returns job ID + polling URL
- Must poll `data.urls.get` until `status != "processing"`
- Outputs in `data.outputs[]`

### FAL
- **Queue-based API** — initial POST returns `IN_QUEUE` + `response_url`/`status_url`
- Must poll `response_url` until images appear
- Response: `{"images": [{"url": "..."}]}`

### Together AI
- SDK v2: uses `image_loras=[{"path": url, "scale": float}]` (NOT `loras=json_string`)
- Model `FLUX.1-dev-lora` requires at least 1 LoRA

### Pixel Dojo
- Endpoint: `https://pixeldojo.ai/api/v1/flux` (NOT `api.pixeldojo.ai`)
- Uses `aspect_ratio` ("1:1", "16:9", etc.) instead of width/height
- Does NOT support: negative_prompt, num_inference_steps, guidance_scale
- Single LoRA only: `lora_weights` (URL) + `lora_scale` (float)
- Response: `{"images": [url_or_dict, ...]}`

## Testing API Keys (Controlled/Test-Only)

Scoped test keys for development and CI testing are stored in `.env` (gitignored).
Copy from `env.example` and fill in the values. The required env vars are:

```
RUNWARE_API_KEY=...
RUNWARE_ENDPOINT=https://api.runware.ai/v1
RUNWARE_MODEL=runware:101@1
HF_SPACE_NAME=anujithc/flux-dev-lora-private
HF_TOKEN=...
WAVESPEED_API_KEY=...
FAL_API_KEY=...
TOGETHER_API_KEY=...
PIXELDOJO_API_KEY=...
```

> **Note:** Actual test key values are stored in the project's Claude Code auto-memory
> (`/root/.claude/projects/.../memory/test-keys.md`) to avoid GitHub push protection blocks.
> They are also available on the server's `.env` file.

## Known Issues & Gotchas

1. **gradio_client version matters:** v2.x uses `token=`, v1.x used `hf_token=`. Pin `>=2.0.0`.
2. **Together SDK v2 breaking change:** `loras` param renamed to `image_loras`, type changed from JSON string to list of dicts.
3. **Wavespeed + FAL are async:** They return queued jobs. Code must poll for results.
4. **Plugin runs in browser:** Any URL it calls must be reachable from the user's browser, NOT from the server. `localhost:7861` will never work for remote users.
5. **No auth on bridge:** Anyone who discovers `bridge.midnighttavern.online` can generate images on your API credits.
6. **Multi-character LoRA bleed:** When multiple character LoRAs are active in a single generation, features bleed between characters (e.g., hair color, clothing, face shape mixing). This is a fundamental limitation of how Flux (and SD-based models generally) handle multiple LoRAs — each LoRA adjusts the same weight space, and with 2+ character LoRAs the model can't cleanly separate which tokens should be influenced by which LoRA. Current mitigations attempted: prompt ordering, keyword isolation, weight tuning. None fully solve it. Potential approaches worth investigating:
   - **Merged/composite LoRAs** — train a single LoRA on scenes with multiple characters together, so the model learns their co-existence
   - **Regional prompting / attention masking** — assign each LoRA's influence to a spatial region of the image (some ComfyUI nodes support this)
   - **Two-pass inpainting** — generate one character, then inpaint the second into the scene with only that character's LoRA active
   - **IP-Adapter / reference image conditioning** — use reference images instead of (or alongside) LoRAs for character identity

## Current Challenges

### Multi-Character LoRA Bleed (Critical)
The biggest unresolved problem. When a scene has multiple characters, their LoRA features bleed into each other. All prompt-level mitigations have been exhausted — this likely requires a fundamentally different approach (composite LoRAs, regional attention, or multi-pass generation). This is the main bottleneck preventing production-quality multi-character scenes.

## Roadmap

### Phase 1: Current — Flux LoRA Bridge (this project)
- Image generation bridge with 6 provider fallback
- Character LoRA auto-injection from `master_lora_dict.json`
- DeepSeek V3 prompt summarization
- SillyTavern AutoImageGen plugin integration

### Phase 2: Next — Wan 2.2 Video Generation Bridge
- Same architecture pattern as Flux bridge but for video generation
- Target: Wan 2.2 model with character + style LoRAs
- Expose video-gen API that SillyTavern (or a new plugin) can call
- Key challenges: much longer generation times, higher compute cost, LoRA ecosystem for video is less mature
- If multi-character LoRA bleed is solved for Flux, the same approach may transfer to Wan 2.2

## Quick Wins / Future Work

### High Priority
- [ ] **Add auth to bridge** — simple API key or token check on `/sdapi/v1/txt2img` to prevent unauthorized usage
- [ ] **Add rate limiting** — per-IP or per-token limit (e.g., 10 images/minute) to prevent cost runaway
- [ ] **Add spend tracking** — log provider + cost per request, alert when daily spend exceeds threshold

### Medium Priority
- [ ] **Plugin: read steps/cfg/size from SillyTavern settings** instead of hardcoding in generateImage()
- [ ] **Plugin: add loading indicator** — show a spinner/placeholder while image generates
- [ ] **Plugin: retry on failure** — if bridge returns error, retry once after 5s
- [ ] **Cache DeepSeek summaries** — same prompt text = same summary, skip the 1-3s API call
- [ ] **Split flux_lora_bridge.py** — extract providers into separate modules (currently 1700+ lines)

### Low Priority / Nice-to-Have
- [ ] **Multi-tenant support** — per-user API keys, usage metering, billing dashboard
- [ ] **User LoRA management** — let users upload/configure their own character LoRAs via web UI
- [ ] **Provider health dashboard** — `/status` endpoint showing real-time provider availability + latency
- [ ] **WebSocket support** — stream generation progress to the plugin instead of waiting for full response
- [ ] **Image caching** — cache generated images by prompt hash to avoid regenerating identical requests
- [ ] **Plugin queue** — if multiple messages arrive quickly, queue generations instead of dropping them
