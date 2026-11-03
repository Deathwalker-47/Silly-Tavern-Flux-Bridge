# SillyTavern Integration Guide (Flux LoRA Bridge)

## Overview

SillyTavern talks to this bridge through Media Studio, the extension in
`silly-tavern-pluggin/`. The AUTOMATIC1111 `/sdapi/v1/txt2img` endpoint that
SillyTavern's own Image Generation used is retired and answers 410: it took no
token, so anyone who found a public bridge could spend provider credits.

### What it does
- Media Studio sends chat photos and videos on Runware, NanoGPT, Sogni,
  Replicate or PixelDojo. See "MEDIA STUDIO" in README.md
- LoRAs and their keywords come from `master_lora_dict.json`. Private Hugging
  Face files work with `HF_TOKEN` set
- `/edit` changes a photo from a plain-English instruction (FLUX Kontext)

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
HELPER_API_KEY=your_nanogpt_key      # helper LLM: summaries, media director, voice tags
ENABLE_SUMMARIZATION=true
```

> If you do not use Runware, set `REPLICATE_API_TOKEN` for the Replicate fallback.
> Video keys (`NANOGPT_API_KEY`, `SOGNI_API_KEY`, ...) are listed in `env.example`.

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

Install Media Studio as described under "MEDIA STUDIO" in README.md, then fill
in the bridge address and the bridge token (`EDIT_API_TOKEN`) in its settings.

Don't point SillyTavern's own Image Generation at the bridge: it needs the
retired `/sdapi/v1/txt2img`.

---

## API Compatibility

### A1111-compatible endpoints
- `GET /sdapi/v1/options`
- `GET /sdapi/v1/sd-models`
- `GET /sdapi/v1/samplers`

### Retired (answer 410)
- `POST /sdapi/v1/txt2img`
- `POST /v1/images/generations`

### Operational endpoints
- `GET /`
- `GET /status`
- `POST /reset`

### Stub OpenAI-style endpoints (no backend configured)
- `POST /v1/chat/completions` (returns 501 — placeholder for future proxy)

---

## Testing

### 1) Service status

```bash
curl http://localhost:7861/status
```

### 2) Without spending credits

Run `dev/mock_providers.py` and `dev/run_local.py` (see README.md), then make
an image from Media Studio.

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

### SillyTavern can’t connect
- Confirm Media Studio has the bridge address and the bridge token
- Ensure bridge is running and port is open

Example:

## File Map

- `flux_lora_bridge.py` – main server and provider clients
- `master_lora_dict.json` – LoRA config and keyword map
- `runware_lora_mapping.json` – dynamic URL→Runware ID cache
- `env.example` – safe environment template
- `QUICK_REFERENCE.md` – concise operational checklist
