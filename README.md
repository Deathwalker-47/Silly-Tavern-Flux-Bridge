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
- FLUX.1 Kontext instruction-based image editing (`POST /edit`, browser UI at `/edit-ui`)
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
- POST /edit                     FLUX Kontext instruction image edit
- GET  /edit-ui                  browser UI for /edit (works on a phone)
- POST /sdapi/v1/img2img         A1111-shaped alias for /edit
- GET  /images/{id}              serves generated images by id
- POST /v1/images/generations    OpenAI-compatible image generation
- POST /v1/chat/completions      (stub, returns 501 — no proxy configured)
- GET  /v1/models                (stub, returns empty list)

--------------------------------------------------------------------------------
IMAGE EDITING (FLUX KONTEXT)
--------------------------------------------------------------------------------

`POST /edit` edits an existing image from a plain-English instruction. It runs on
Runware's FLUX.1 Kontext and is a separate path from txt2img: no LoRA matching, no
DeepSeek summarization, no mask. The instruction is sent verbatim, because Kontext
follows it literally.

Request fields (`image` and `instruction` are required):

```jsonc
{
  "image": "data:image/jpeg;base64,...",  // data URI, bare base64, http(s) URL, or Runware image UUID
  "instruction": "make the sky sunset orange, keep everything else identical",
  "reference_image": null,                 // optional 2nd image to take a face/style/object from
  "model": null,                           // optional Kontext AIR id override
  "steps": null,                           // dev model only
  "cfg_scale": null,                       // dev model only
  "width": null,                           // optional; snapped to the nearest supported size
  "height": null
}
```

The response matches the txt2img shape, so existing clients can reuse it:

```json
{"images": ["<base64>"], "image_urls": ["https://.../images/<id>.jpg"],
 "parameters": {...}, "info": "{...}"}
```

Example:

```bash
curl -X POST http://localhost:7861/edit \
  -H "Content-Type: application/json" \
  -d '{"image":"https://example.com/photo.jpg",
       "instruction":"make the sky sunset orange, keep everything else identical"}' \
  | jq '.image_urls'
```

Notes:
- Kontext renders a fixed set of sizes. The bridge reads the input's aspect ratio and
  snaps to the closest one, so portrait photos stay portrait.
- Inputs are EXIF-rotated, flattened to RGB and downscaled to `KONTEXT_MAX_EDGE`
  before being uploaded, so phone photos work without any client-side preparation.
- Face replacement works best when the instruction says what to preserve, e.g.
  *"Replace the face of the woman on the right with the person in the second reference
  image. Keep her hairstyle, clothing, pose, lighting and background exactly the same."*
  If identity drift is too high, try `KONTEXT_MODEL=bfl:4@1` (Kontext [max]).
- Every call spends Runware credits.

For a browser (including a phone), open `http://localhost:7861/edit-ui`.

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
