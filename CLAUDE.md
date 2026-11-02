# CLAUDE.md — Project Instructions for Claude Code

## Project Overview

Silly-Tavern-Flux-Bridge is a **Python/FastAPI** bridge that exposes an AUTOMATIC1111-compatible `/sdapi/v1/txt2img` endpoint. It routes image generation requests through a cascading chain of Flux LoRA providers (Runware → Replicate by default; Wavespeed, FAL and Together AI are deprecated but kept) with automatic fallback. Includes a SillyTavern plugin (JavaScript) in `silly-tavern-pluggin/`.

## Quick Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python flux_lora_bridge.py

# Run tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_provider_response_parsing.py -v
```

## Project Structure

- `flux_lora_bridge.py` — Main application (Config, txt2img providers, LoRA matching, core API routes)
- `media/` — Media Studio's bridge side (see below)
- `master_lora_dict.json` — LoRA definitions and keyword mappings
- `lora_dict.py` — dictionary format (one file per architecture under `models`) and its migration
- `requirements.txt` — Python dependencies (FastAPI, httpx, Pillow, etc.)
- `env.example` — Environment variable template (copy to `.env`)
- `tests/` — Unit tests (Python unittest via pytest)
- `silly-tavern-pluggin/` — SillyTavern browser plugin (JavaScript)
  - `index.js` — Settings, auto media, panels, slash commands
  - `src/` — `jobs.js` (submit, poll, resume, attach), `media-panel.js`, `groups.js`, `profile.js`, `context.js`, `assets.js`, `store.js`, `bridge.js`, `telemetry.js`
  - `manifest.json` / `settings.html` / `style.css`

## Architecture

### Provider Chain (fallback order)
Set by `IMAGE_PROVIDER_ORDER` (default `runware,replicate`):
1. **Runware** (primary, up to 12 LoRAs)
2. **Replicate** (`lucataco/flux-dev-multi-lora`, up to MAXLORAS_REPLICATE)
Deprecated, off by default: **Wavespeed** (4), **FAL** (3), **Together AI** (2).

### Media Studio (`media/` + `silly-tavern-pluggin/`)
- `catalog.py`: `IMAGE_CATALOG` / `VIDEO_CATALOG`, provider → family → options (first = default). A model a provider lacks is just absent. No SFW-only model anywhere in the chat pipeline
- `providers/`: one adapter per provider (`base.py` is the interface): Runware, NanoGPT, Sogni (official SDK), Replicate, PixelDojo, Mock
- `jobs.py`: `JobRunner`, jobs in SQLite (`data/media.db`), resumed after a restart, one semaphore per provider
- `pipelines.py`: image, Best Option, the three video modes, keyframes
- `director.py` + `guides/`: picks people, aspect and references, writes the prompt in each model's format. `minor_block` stops sexual content involving minors, the only thing the bridge blocks
- `references.py`, `assets.py`, `loras.py`: slot picking, the reference window and EXIF stripping, LoRA stacks
- `telemetry.py`, `insights.py`: every step recorded, keys scrubbed; `dev/insights.py` reads it
- `helper.py`: `HelperLLM`, NanoGPT chat for the director, prompt summaries and TavernVoice voice tags
- Safety checkers are disabled wherever a provider offers the switch

### Key Classes (all in `flux_lora_bridge.py`)
- `Config` — Loads settings from environment variables
- `DeepSeekSummarizer` — Optional prompt summarization (now through the helper LLM)
- `LoRAManager` — Keyword-based LoRA matching and role-cap filtering
- `ProviderClient` (abstract) — Base class for all providers
- Concrete providers: `RunwareClient`, `WavespeedClient`, `FALClient`, `TogetherAIClient`
- `ProviderState` — Manages provider selection and fallback ordering
- `MaskGenerator` — Generates masks for multi-character inpainting
- `MultiCharPipeline` — Multi-character inpainting pipeline with layout templates
- `Txt2ImgRequest` / `Txt2ImgResponse` — API request/response models

### Data Flow
```
Request → LoRA keyword matching → Optional summarization → Role-cap filtering →
Provider-specific pruning → Prompt enhancement → Provider loop (try until success) →
Return base64 image
```

## Coding Conventions

- **Python**: No linter/formatter configured. Follow existing code style (type hints on function signatures, f-strings, logging with emoji prefixes per provider).
- **JavaScript (plugin)**: Vanilla JS, no build step. Uses MutationObserver pattern.
- **Environment**: Secrets via `.env`, or `data/keys.json` when saved from settings (`media/keys.py`; write-only, needs EDIT_API_TOKEN). Never in SillyTavern settings, except TavernVoice's voice keys (xAI, ElevenLabs, HeyGen), which the browser must use. Never hardcode API keys.
- **Logging**: Use Python `logging` module. Include provider-specific emoji prefixes (e.g., `🌐 [Runware]`, `🤖 [DeepSeek]`). Redact long prompts and sensitive content.
- **Error handling**: Each provider call should be wrapped in try/except with fallback to next provider.

## Important Notes

- `flux_lora_bridge.py` holds the txt2img path and core routes; Media Studio lives in `media/`.
- LoRA matching is case-insensitive keyword matching against `master_lora_dict.json`.
- Role-based LoRA caps: character=6, nsfw=4, expression=2, general=2, misc=1.
- Image generation timeout: 120s (provider calls: 90s).
- CORS is open (all origins) — intentional for local SillyTavern use.
- Tests use stubs/mocks for FastAPI, Pydantic, and httpx (no live API calls in CI). `tests/media_http_check.py` runs real FastAPI in a subprocess. `from media import helper` is the instance, not the module: tests use `importlib.import_module("media.helper")`.
- Local run without spending credits: `dev/mock_providers.py` + `dev/run_local.py`.
