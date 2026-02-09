# CLAUDE.md — Project Instructions for Claude Code

## Project Overview

Silly-Tavern-Flux-Bridge is a **Python/FastAPI** bridge that exposes an AUTOMATIC1111-compatible `/sdapi/v1/txt2img` endpoint. It routes image generation requests through a cascading chain of Flux LoRA providers (Runware → HF ZeroGPU → Wavespeed → FAL → Together AI → Pixel Dojo) with automatic fallback. Includes a SillyTavern plugin (JavaScript) in `silly-tavern-pluggin/`.

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
1. **Runware** (primary, up to 12 LoRAs)
2. **HF ZeroGPU Space** (Gradio client, up to 10 LoRAs)
3. **Wavespeed** (up to 4 LoRAs)
4. **FAL** (up to 3 LoRAs)
5. **Together AI** (up to 2 LoRAs)
6. **Pixel Dojo** (1 LoRA)

### Key Classes (all in `flux_lora_bridge.py`)
- `Config` — Loads settings from environment variables
- `DeepSeekSummarizer` — Optional prompt summarization via Together AI
- `LoRAManager` — Keyword-based LoRA matching and role-cap filtering
- `ProviderClient` (abstract) — Base class for all providers
- Concrete providers: `RunwareClient`, `HFZeroGPUClient`, `WavespeedClient`, `FALClient`, `TogetherAIClient`, `PixelDojoClient`

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

- The main app is a single file (`flux_lora_bridge.py`, ~1,873 lines). All core logic lives here.
- LoRA matching is case-insensitive keyword matching against `master_lora_dict.json`.
- Role-based LoRA caps: character=6, nsfw=4, expression=2, general=2, misc=1.
- Image generation timeout: 120s (provider calls: 90s).
- CORS is open (all origins) — intentional for local SillyTavern use.
- Tests use stubs/mocks for FastAPI, Pydantic, and httpx (no live API calls in CI).
