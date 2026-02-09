================================================================================
FLUX LORA BRIDGE - START HERE
================================================================================

This repository provides a FastAPI bridge that exposes an AUTOMATIC1111-style
`/sdapi/v1/txt2img` endpoint for SillyTavern and routes image generation to
multiple Flux LoRA providers with fallback.

Current provider order in code:
1) Runware (primary)
2) HF ZeroGPU Space (gradio)
3) Wavespeed
4) FAL
5) Together
6) Pixel Dojo

It also includes:
- Keyword-based LoRA matching from `master_lora_dict.json`
- Optional DeepSeek V3 prompt summarization via Together API
- OpenAI-compatible chat proxy endpoints (`/v1/chat/completions`, `/v1/models`)

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
- POST /v1/chat/completions      OpenAI-compatible proxy to You.com agents
- GET  /v1/models                OpenAI-compatible models list

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

--------------------------------------------------------------------------------
SILLYTAVERN PLUGIN (AUTO IMAGE UNIVERSAL)
--------------------------------------------------------------------------------

A ready-to-use plugin is included under `silly-tavern-pluggin/` in this repo.
To install it into SillyTavern, copy the files into:

`public/scripts/extensions/auto-image-universal/`

Example:

```bash
mkdir -p /path/to/SillyTavern/public/scripts/extensions/auto-image-universal
cp -r silly-tavern-pluggin/* /path/to/SillyTavern/public/scripts/extensions/auto-image-universal/
```

Important:
- Set your OpenRouter key via runtime config (do NOT hardcode keys in plugin files).
- The plugin uses bridge endpoint `http://localhost:7861/sdapi/v1/txt2img` by default.
- Configure runtime overrides by setting `window.AUTO_IMAGE_UNIVERSAL_CONFIG` or
  `localStorage.autoImageUniversalConfig` in the browser.
