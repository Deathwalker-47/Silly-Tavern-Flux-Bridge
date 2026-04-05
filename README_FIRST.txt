================================================================================
FLUX LORA BRIDGE - START HERE
================================================================================

This repository provides a FastAPI bridge that exposes an AUTOMATIC1111-style
`/sdapi/v1/txt2img` endpoint for SillyTavern and routes image generation to
multiple Flux LoRA providers with fallback.

Current provider order in code:
1) Runware (primary)
2) Wavespeed
3) FAL
4) Together

It also includes:
- Keyword-based LoRA matching from `master_lora_dict.json`
- Optional DeepSeek V3 prompt summarization via Together API
- Stub OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/models`) for future proxy use

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

In SillyTavern:
- Extensions -> Image Generation
- Source: Stable Diffusion
- SD WebUI URL: http://localhost:7861
- Enable image generation

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
- POST /sdapi/v1/txt2img         A1111-compatible txt2img
- POST /v1/chat/completions      (stub, returns 501 — no proxy configured)
- GET  /v1/models                (stub, returns empty list)

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
