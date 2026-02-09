# SillyTavern Integration Guide (Flux LoRA Bridge)

## Overview

This bridge presents an AUTOMATIC1111-compatible API (`/sdapi/v1/txt2img`) so
SillyTavern can use Flux LoRA generation through multiple providers with
automatic fallback.

### What it does
- Matches LoRAs by keywords from `master_lora_dict.json`
- Optionally summarizes long prompts using DeepSeek V3 via Together
- Tries providers in this order:
  1. Runware
  2. HF ZeroGPU Space
  3. Wavespeed
  4. FAL
  5. Together
  6. Pixel Dojo

---

## Prerequisites

- Python 3.10+
- SillyTavern running
- At least one valid provider credential

---

## Installation

```bash
cd /workspace/Silly-Tavern-Flux-Bridge
pip install -r requirements.txt
cp env.example .env
```

Edit `.env` and set at least one provider key.

### Minimal recommended `.env`

```bash
BRIDGE_PORT=7861
RUNWARE_API_KEY=your_runware_key
TOGETHER_API_KEY=your_together_key
ENABLE_SUMMARIZATION=true
```

> If you do not use Runware, set credentials for one of the fallback providers
> (`HF_SPACE_NAME`, `WAVESPEED_API_KEY`, `FAL_API_KEY`, etc.).

---

## Start the Bridge

```bash
python flux_lora_bridge.py
```

Health check:

```bash
curl http://localhost:7861/status
```

---

## Configure SillyTavern

1. Open **Extensions → Image Generation**
2. Set **Source** = `Stable Diffusion`
3. Set **SD WebUI URL** = `http://localhost:7861`
4. Enable image generation

No plugin-specific endpoint is required for baseline compatibility.

---

## API Compatibility

### A1111-compatible endpoints
- `GET /sdapi/v1/options`
- `GET /sdapi/v1/sd-models`
- `GET /sdapi/v1/samplers`
- `POST /sdapi/v1/txt2img`

### Operational endpoints
- `GET /`
- `GET /status`
- `POST /reset`

### Optional OpenAI-style proxy endpoints
- `POST /v1/chat/completions` (You.com agents backend)
- `GET /v1/models`

To use chat proxy, set:
- `YOU_COM_API_KEY`
- `YOU_COM_DEFAULT_AGENT` (or provide `agent_id` in request payload)

---

## Testing

### 1) Service status

```bash
curl http://localhost:7861/status
```

### 2) A1111 txt2img smoke test

```bash
curl -X POST http://localhost:7861/sdapi/v1/txt2img \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "portrait of a woman, cinematic lighting",
    "negative_prompt": "blurry, low quality",
    "steps": 20,
    "cfg_scale": 3.5,
    "width": 1024,
    "height": 1024
  }'
```

If successful, response includes a base64 image under `images[0]`.

---

## LoRA Matching Notes

- LoRA definitions and keywords come from `master_lora_dict.json`.
- Permanent LoRAs are auto-included if configured in dict `config.permanent_loras`.
- Role caps are applied before per-provider LoRA limits.
- Provider-specific max LoRAs are then enforced.

---

## Security & Secret Management

- **No hardcoded live secrets should exist in source.**
- Store credentials only in `.env` or secret manager.
- Add `.env` to your global/local ignore list if needed.
- Rotate any credential previously exposed in commit history.

---

## Common Issues

### Bridge starts, but generation fails
- Check at least one provider credential is valid.
- Ensure host can reach provider APIs.

### HF provider not working
- Verify `gradio_client` is installed.
- Set `HF_SPACE_NAME` correctly.
- If space is private, set `HF_TOKEN`.

### Chat proxy returns auth errors
- Verify `YOU_COM_API_KEY` and selected/default agent ID.

### SillyTavern can’t connect
- Confirm ST uses `http://localhost:7861`
- Ensure bridge is running and port is open

---

## File Map

- `flux_lora_bridge.py` – main server and provider clients
- `master_lora_dict.json` – LoRA config and keyword map
- `runware_lora_mapping.json` – dynamic URL→Runware ID cache
- `env.example` – safe environment template
- `QUICK_REFERENCE.md` – concise operational checklist
