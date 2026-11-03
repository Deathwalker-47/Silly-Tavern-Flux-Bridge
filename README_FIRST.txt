================================================================================
FLUX LORA BRIDGE - START HERE
================================================================================

This repository provides a FastAPI bridge for SillyTavern. Its extension, Media
Studio, sends chat photos and videos through several providers, with Flux LoRAs
from `master_lora_dict.json`.

The AUTOMATIC1111 `/sdapi/v1/txt2img` and OpenAI `/v1/images/generations`
endpoints are retired and answer 410. They took no token, so anyone who found a
public bridge could spend provider credits and use every LoRA.

It also includes:
- Keyword-based LoRA matching from `master_lora_dict.json`
- Optional prompt summarization via the helper LLM (NanoGPT; Together fallback)
- Media Studio: chat photos and videos for SillyTavern (see README.md)
- A stub OpenAI-compatible endpoint (`/v1/chat/completions`) for future proxy use

--------------------------------------------------------------------------------
QUICK SETUP
--------------------------------------------------------------------------------

1) Install dependencies
   pip install -r requirements.txt

2) Configure environment
   cp env.example .env
   # edit .env and add at least one provider key

3) Start bridge
   python flux_lora_bridge.py

4) Verify
   curl http://localhost:7861/status

--------------------------------------------------------------------------------
SILLYTAVERN SETTINGS
--------------------------------------------------------------------------------

Install Media Studio (see README.md). SillyTavern's own Image Generation
extension can't use the bridge any more, because it needed the retired
`/sdapi/v1/txt2img`.

--------------------------------------------------------------------------------
IMPORTANT SECURITY NOTE
--------------------------------------------------------------------------------

- This repo intentionally does NOT ship hardcoded live API keys.
- Keep real credentials in `.env` only.
- Do not paste secrets into docs, screenshots, or commits.

--------------------------------------------------------------------------------
API ENDPOINTS
--------------------------------------------------------------------------------

- GET  /                         health summary
- GET  /status                   provider + LoRA status
- POST /reset                    lightweight reset message endpoint
- POST /v1/chat/completions      (stub, returns 501 — no proxy configured)
- POST /sdapi/v1/txt2img, POST /v1/images/generations   retired, answer 410
- Media Studio's routes are listed in README.md

--------------------------------------------------------------------------------
FILES
--------------------------------------------------------------------------------

- flux_lora_bridge.py
- master_lora_dict.json
- requirements.txt
- env.example
- SILLYTAVERN_INTEGRATION.md
- QUICK_REFERENCE.md

--------------------------------------------------------------------------------
NEXT
--------------------------------------------------------------------------------

Read `SILLYTAVERN_INTEGRATION.md` for full setup and troubleshooting.
