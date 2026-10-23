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

- `flux_lora_bridge.py` — Main application (all Python logic: Config, providers, LoRA matching, API routes)
- `master_lora_dict.json` — LoRA definitions and keyword mappings
- `requirements.txt` — Python dependencies (FastAPI, httpx, Pillow, etc.)
- `env.example` — Environment variable template (copy to `.env`)
- `tests/` — Unit tests (Python unittest via pytest)
- `silly-tavern-pluggin/` — SillyTavern browser plugin (JavaScript)
  - `index.js` — Plugin logic
  - `manifest.json` / `settings.json` / `style.css`

## Architecture

### Provider Chain (fallback order)
Set by `IMAGE_PROVIDER_ORDER` (default `runware,replicate`):
1. **Runware** (primary, up to 12 LoRAs)
2. **Replicate** (`lucataco/flux-dev-multi-lora`, up to MAXLORAS_REPLICATE)
Deprecated, off by default: **Wavespeed** (4), **FAL** (3), **Together AI** (2).

### Media Studio (`media_studio.py` + `silly-tavern-pluggin/`)
- `VIDEO_CATALOG`: provider → family → variants (first = default). A model a provider lacks is just absent
- Video providers: `RunwareVideo`, `NanoGPTVideo`, `SogniVideo` (official SDK), `ReplicateVideo`, `MockVideo`
- `VideoJobs` + in-memory `VideoStore`: async jobs, finished clips served at `/videos/{id}`
- `HelperLLM`: NanoGPT chat (Together fallback) for the media director, prompt summaries and TavernVoice voice tags
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
- **Environment**: All secrets via `.env` file only. Never hardcode API keys.
- **Logging**: Use Python `logging` module. Include provider-specific emoji prefixes (e.g., `🌐 [Runware]`, `🤖 [DeepSeek]`). Redact long prompts and sensitive content.
- **Error handling**: Each provider call should be wrapped in try/except with fallback to next provider.

## Important Notes

- The main app is a single file (`flux_lora_bridge.py`, ~2,077 lines). All core logic lives here.
- LoRA matching is case-insensitive keyword matching against `master_lora_dict.json`.
- Role-based LoRA caps: character=6, nsfw=4, expression=2, general=2, misc=1.
- Image generation timeout: 120s (provider calls: 90s).
- CORS is open (all origins) — intentional for local SillyTavern use.
- Tests use stubs/mocks for FastAPI, Pydantic, and httpx (no live API calls in CI).
