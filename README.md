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
- Media Studio: a SillyTavern extension (in `silly-tavern-pluggin/`) where
  characters send photos and videos in chat, on Runware, NanoGPT, Sogni,
  Replicate or PixelDojo. A director LLM decides what to send and writes the
  prompt for each model
- A helper LLM (NanoGPT, OpenAI-compatible) for the director, prompt
  summarization and TavernVoice's voice tags
- Keyword-based LoRA matching from `master_lora_dict.json`
- FLUX.1 Kontext instruction-based image editing (`POST /edit`, browser UI at `/edit-ui`)
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

Install Media Studio (see MEDIA STUDIO below). SillyTavern's own Image
Generation extension can't use the bridge any more, because it needed the
retired `/sdapi/v1/txt2img`.

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
- GET  /media/capabilities       Media Studio menus: providers, models, modes, helper
- GET  /media/balances           provider balances
- GET  /media/keys               where each key comes from, last 4 at most
- PUT  /media/keys/{provider}    save a key and check it (needs EDIT_API_TOKEN)
- DELETE /media/keys/{provider}  remove a saved key, back to .env
- POST /assets                   upload an image (window check, metadata stripped)
- POST /media/lora-status        which characters have a LoRA
- POST /media/jobs               start an image or video job (returns job_id)
- GET  /media/jobs               a chat's open jobs
- GET  /media/jobs/{id}          job progress
- POST /media/jobs/{id}/cancel   stop a job and its remote task
- POST /media/jobs/{id}/ack      the extension saved the file
- GET  /media/files/{id}         serves finished files (streams with Range)
- GET  /lora-files/{link}/{name} a private LoRA for a provider (signed link)
- GET  /media/generations        gallery rows, by person and kind
- PATCH /media/generations/{id}  rating and comment
- GET  /media/generations/{id}/trace   every pipeline step of one generation
- POST /media/routes/test        run your prompt on a route at its lowest setting
- PUT  /media/routes/{provider}/{model}   pass or fail a route
- GET  /media/insights           the pipeline report (/export for raw data)
- POST /director/decide          should the character send something?
- POST /director/next-speaker    group chats: who acts next
- POST /helper/voice-tags        Grok speech tags for TavernVoice
- GET  /helper/models            helper model list (for the model menus)
- POST /v1/chat/completions      (stub, returns 501 — no proxy configured)
- POST /sdapi/v1/txt2img, POST /v1/images/generations   retired, answer 410

--------------------------------------------------------------------------------
IMAGE EDITING (FLUX KONTEXT)
--------------------------------------------------------------------------------

`POST /edit` edits an existing image from a plain-English instruction. It runs on
Runware's FLUX.1 Kontext, with no LoRAs, no prompt summarization and no mask. The instruction is sent verbatim, because Kontext
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

Every `/immich/*` route needs the token too, because they read your photo library.

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
- media/                     Media Studio's bridge side: catalog, director and its
                             guides, providers, jobs, pipelines, telemetry, routes
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

Characters send photos and short videos in chat. The director (the helper LLM)
decides when to send something, what's in the shot and how to write the prompt
for the chosen model. Jobs run on the bridge, so they survive a page reload, a
locked phone or a bridge restart, and the result lands on its message when you
come back.

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
- Images and Video: provider, model and, for video, Reference, First frame or
  First + last frame. Each mode says what it's for
- Auto media: off, the director decides, or an image every turn, with cooldowns
  and daily limits
- Test this route: runs your own prompt on the selected model at the lowest
  setting. Pass keeps the route. Fail hides it until you restore it
- Pipeline report: what worked, what's slow, what it cost and what to change

Each character and persona has a Media panel:
- References: five tagged slots (Face 1, Face 2, Full body, Portrait, Naked).
  Nothing goes to a provider until the confirmation box is ticked
- Appearance: age (required, 18 or over) and a description that wins over
  anything in the chat
- Gallery: every generation with its settings, cost, a 1-5 rating, a comment
  and the full pipeline steps

Group chats use native SillyTavern groups in Manual reply mode. Open a group and
press the clapperboard button: the director picks who acts until it's your turn,
one normal card can be the group's Setting card (world, rules, side characters),
and each round makes at most one media decision. A round ends when it's your
turn, at 5 messages or the number of characters, when you press Stop, or when
you start typing.

Add keys under API keys in the Media Studio settings, or in the bridge's .env.
A key saved in settings replaces the one in .env. It's stored in
`data/keys.json` on the bridge and never sent back, and saving needs
EDIT_API_TOKEN set on the bridge. A provider with no key disappears from every
picker. TavernVoice's keys (xAI, ElevenLabs, HeyGen) don't go here: they stay
in SillyTavern, because TavernVoice calls those providers from the browser.

  Runware    RUNWARE_API_KEY      Flux Dev + LoRA, Juggernaut Z + LoRA
  NanoGPT    NANOGPT_API_KEY      Seedream 4.5, Qwen Edit, The Best Option, video,
                                  and the director (HELPER_MODEL, DIRECTOR_MODEL)
  Sogni      SOGNI_API_KEY        Z-Image, Qwen Edit, MiniMax H3 Reference, video
  Replicate  REPLICATE_API_TOKEN  Flux Dev + LoRA (backup)
  PixelDojo  PIXELDOJO_API_KEY    Seedream 4.5, The Best Option, video (billed in
                                  credits, the most expensive route)
  SpicyAPI   SPICYAPI_API_KEY     Flux and Z-Image + LoRA, Qwen Edit, Seedream 4.5,
                                  The Best Option, every video family (no filter)
  Venice     VENICE_API_KEY       Qwen Edit, Seedream 5 Pro, Wan 3.0, MiniMax H3,
                                  Wan 2.2; Seedance and LTX wait on a test

Private LoRAs: when the LoRA files sit in a private Hugging Face repo, add a
Hugging Face token (HF_TOKEN) the same way. Read access to that repo is enough.
Runware and Replicate download LoRAs by link and can't send a token, so the
bridge gives them a signed link of its own (/lora-files/...) that streams the
file. The link doesn't expire, because Runware keeps it and downloads the file
again later; only the bridge can make one, and changing LORA_LINK_SECRET retires
them all. Providers have to reach that link, so it needs PUBLIC_BASE_URL set to
the bridge's public HTTPS address, and it doesn't work on localhost.

Safety checkers are switched off wherever a provider has a switch. The bridge
blocks one thing itself: sexual content involving minors. Everyone in a shot
needs an age of 18 or over.

Everything the pipeline does is recorded in the bridge's own database
(`data/media.db`): each step, what it cost, how long it took and how you rated
the result. Read it from the Pipeline report button, or:

```bash
python dev/insights.py --days 30          # routes, ratings, slow steps, suggestions
python dev/insights.py --trace <job id>   # every step of one generation
python dev/insights.py --export out.jsonl
```

TavernVoice (https://github.com/Deathwalker-47/TavernVoice) uses the same
bridge for its "voice director" (POST /helper/voice-tags) and to find the people
on a story card (POST /helper/cast), giving each a voice in the language they
speak from any provider TavernVoice has a key for.

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
- Set PUBLIC_BASE_URL to that HTTPS address, so image URLs and private LoRA
  links resolve.
- Media Studio saves each finished file in SillyTavern's own files, so chat
  playback on a phone or PC is served by SillyTavern. The bridge deletes its
  copy MEDIA_FILE_TTL_DAYS after that, or MEDIA_UNSAVED_TTL_DAYS if it was
  never saved. Keep `data/` on the server: it holds the job and rating history.
- Keep it running as a service (systemd, pm2 or screen):
    pip install -r requirements.txt     # includes sogni-client
    python flux_lora_bridge.py

--------------------------------------------------------------------------------
TESTING WITHOUT SPENDING CREDITS
--------------------------------------------------------------------------------

  python dev/mock_providers.py     # terminal 1: fake providers and helper LLM
  python dev/run_local.py          # terminal 2: bridge on :7861 using the mocks

The "mock" provider returns local images and a bundled 2 s clip. Unit tests:
  python -m pytest tests
