================================================================================
FLUX LORA BRIDGE - START HERE
================================================================================

This repository provides a FastAPI bridge that exposes an AUTOMATIC1111-style
`/sdapi/v1/txt2img` endpoint for SillyTavern and routes image generation to
multiple Flux LoRA providers with fallback.

Image provider order (IMAGE_PROVIDER_ORDER in .env):
1) Runware (primary)
2) Replicate (Flux multi-LoRA backup)
Wavespeed, FAL and Together are deprecated: their code is kept but they are not
in the default order.

It also includes:
- Media Studio: a SillyTavern extension (in `silly-tavern-pluggin/`) where
  characters send photos and videos in chat. Video runs on Runware, NanoGPT,
  Sogni or Replicate; a helper LLM decides when to send what
- A helper LLM (NanoGPT, OpenAI-compatible) for the media director, prompt
  summarization and TavernVoice's voice tags
- Keyword-based LoRA matching from `master_lora_dict.json`
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
- POST /edit/immich              edit an Immich photo and save the result back
- GET  /edit-ui                  browser UI for /edit (works on a phone)
- POST /sdapi/v1/img2img         A1111-shaped alias for /edit
- GET  /immich/status            whether the bridge can reach Immich
- GET  /immich/albums            list albums
- GET  /immich/albums/{id}       images in an album
- GET  /immich/assets            recent images
- GET  /immich/assets/{id}/thumbnail   thumbnail proxy
- GET  /immich/assets/{id}/original    original-file proxy
- GET  /images/{id}              serves generated images by id
- GET  /media/capabilities       Media Studio menus: providers, models, helper
- POST /media/image              image for a chat message (LoRA or avatar mode)
- POST /media/lora-status        which characters have a LoRA
- POST /video/generate           start a video job (returns job_id)
- GET  /video/jobs/{id}          job progress; video_path when done
- GET  /videos/{id}              serves finished videos (MP4, streams with Range)
- POST /helper/decide            media director: should the character send something?
- POST /helper/voice-tags        Grok speech tags for TavernVoice
- GET  /helper/models            helper model list (for the model menus)
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
AUTH FOR EDIT ENDPOINTS
--------------------------------------------------------------------------------

`/edit`, `/edit/immich` and `/sdapi/v1/img2img` spend Runware credits on every
call. Set `EDIT_API_TOKEN` in `.env` whenever the bridge is reachable from the
internet, and send it as either header:

```
Authorization: Bearer <token>
X-Edit-Token: <token>
```

With no token configured the endpoints stay open and a warning is logged at startup.

--------------------------------------------------------------------------------
IMMICH INTEGRATION
--------------------------------------------------------------------------------

Set `IMMICH_BASE_URL` and `IMMICH_API_KEY` (Immich → Account Settings → API Keys)
and the bridge can read photos out of an Immich library and file the edits back:

```bash
curl -X POST http://localhost:7861/edit/immich \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $EDIT_API_TOKEN" \
  -d '{"asset_id":"<immich-asset-id>",
       "instruction":"remove the car, keep everything else identical"}'
```

The bridge pulls the original, runs Kontext, uploads the result as a **new** asset
and adds it to the `IMMICH_EDIT_ALBUM` album (`AI Edits` by default, created if it
doesn't exist). The original is never modified. Pass `"save": false` to get the
edited image back without writing anything to Immich.

The response is the `/edit` shape plus an `immich` block:

```json
{"images": ["<base64>"], "image_urls": ["..."],
 "immich": {"asset_id": "...", "album_id": "...", "album_name": "AI Edits"}}
```

Notes:
- The Immich API key stays on the bridge. Browser clients talk only to the bridge,
  which also sidesteps Immich's CORS rules.
- Immich is optional and the integration is failure-tolerant: if the bridge cannot
  reach Immich, the `/immich/*` endpoints answer `503` with the reason and
  `/immich/status` reports `reachable: false`, so a client that can reach Immich
  itself can fall back to posting image bytes to `/edit` directly.
- If an edit succeeds but the upload afterwards fails, the edited image is still
  returned in the response rather than discarded.

--------------------------------------------------------------------------------
FILES
--------------------------------------------------------------------------------

- flux_lora_bridge.py        the FastAPI app
- media_studio.py            video providers and catalog, video jobs, helper LLM
- silly-tavern-pluggin/      Media Studio, the SillyTavern extension
- dev/                       local mocks, for testing without spending credits
- tests/                     unit tests (python -m pytest tests)
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
MEDIA STUDIO (SILLYTAVERN EXTENSION)
--------------------------------------------------------------------------------

Characters send photos and short videos in chat. The helper LLM decides when to
send something (or images come every turn), every message gets a video button,
and each character has a gallery with up to 5 avatar images. Characters without
a LoRA get images from their avatars (Kontext) or plain txt2img.

Install: copy `silly-tavern-pluggin/` into your SillyTavern user's extension
folder (any folder name works), then reload SillyTavern:

```bash
cp -r silly-tavern-pluggin /path/to/SillyTavern/data/default-user/extensions/MediaStudio
```

To update, `git pull` here and copy it again.

Then open Extensions -> Media Studio (Flux Bridge):
- Bridge URL: the bridge's address as your browser sees it (empty = the SD
  WebUI URL from Image Generation)
- Bridge token: the EDIT_API_TOKEN value, if set
- Video: provider, model, variant, duration and resolution. "Sogni queue"
  (Sogni only) picks fast (billed) or relaxed (free, can queue for an hour or
  more); empty uses SOGNI_NETWORK
- Helper LLM: the media director's model (empty = HELPER_MODEL)

Keys live only in the bridge's .env, never in the extension. Missing keys just
hide that provider:

  Runware    RUNWARE_API_KEY      LTX-2.5 Fast, Wan 2.7, Seedance 1.5 / 2.0, MiniMax H3
  NanoGPT    NANOGPT_API_KEY      LTX-2.3, Wan 2.2 / 2.7, Seedance 1.5 / 2.0, MiniMax H3
                                  ("Spicy" variants first)
  Sogni      SOGNI_API_KEY        LTX-2.3 10Eros, Wan 2.2, Seedance 2.0, MiniMax H3
                                  (H3 gets the VBVR LoRA, SOGNI_H3_LORA)
  Replicate  REPLICATE_API_TOKEN  LTX-2.3 Fast, Wan 2.2 / 2.7, Seedance 1.5 / 2.0, MiniMax H3
  Helper     HELPER_API_KEY       any NanoGPT chat model (HELPER_MODEL)

Safety checkers are switched off wherever a provider has a switch. Runware no
longer offers LTX-2.3: every LTX-2.3 job failed there with a bare
"inferenceError" (Oct 2026). Use NanoGPT, Sogni or Replicate for LTX-2.3.

TavernVoice (https://github.com/Deathwalker-47/TavernVoice) uses the same
bridge for its "voice director" (POST /helper/voice-tags).

--------------------------------------------------------------------------------
BRIDGE ON A REMOTE SERVER (SILLYTAVERN OVER HTTPS)
--------------------------------------------------------------------------------

When SillyTavern is served over HTTPS (on a VPS, for example), the browser
blocks calls from that page to a plain http:// bridge. Give the bridge its own
HTTPS address, for example with Caddy next to SillyTavern:

  bridge.example.com {
      reverse_proxy localhost:7861
  }

Then:
- Set EDIT_API_TOKEN. Endpoints that spend credits (images, video, helper,
  edits) require it, and anyone can reach a public bridge. Paste the same
  token into Media Studio and TavernVoice.
- Set PUBLIC_BASE_URL to that HTTPS address, so image URLs resolve.
- Videos stay on the bridge for VIDEO_STORE_TTL seconds. Media Studio
  downloads each finished video and saves it in SillyTavern's own files, so
  chat playback on a phone or PC is served by SillyTavern.
- Keep it running as a service (systemd, pm2 or screen):
    pip install -r requirements.txt     # includes sogni-client
    python flux_lora_bridge.py

--------------------------------------------------------------------------------
TESTING WITHOUT SPENDING CREDITS
--------------------------------------------------------------------------------

  python dev/mock_providers.py     # terminal 1: fake Runware and helper LLM
  python dev/run_local.py          # terminal 2: bridge on :7861 using the mocks

The "mock" video provider returns a bundled 2 s clip. Unit tests:
  python -m pytest tests
