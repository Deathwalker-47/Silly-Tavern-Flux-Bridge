# Flux LoRA Bridge - Quick Reference

## 1) Install + Run

```bash
pip install -r requirements.txt
cp env.example .env
# edit .env with provider keys
python flux_lora_bridge.py
```

Default port is `7861` unless `BRIDGE_PORT` is set.

## 2) SillyTavern

- **Source**: Stable Diffusion
- **SD WebUI URL**: `http://localhost:7861`
- Enable image generation

## 3) Core Endpoints

```bash
# health
curl http://localhost:7861/

# status
curl http://localhost:7861/status

# txt2img test
curl -X POST http://localhost:7861/sdapi/v1/txt2img \
  -H "Content-Type: application/json" \
  -d '{"prompt":"portrait photo","steps":20,"width":1024,"height":1024}'
```

## 4) Provider Order (fallback chain)

1. Runware
2. HF ZeroGPU Space
3. Wavespeed
4. FAL
5. Together
6. Pixel Dojo

## 5) Environment Variables You’ll Usually Set

- `RUNWARE_API_KEY`
- `HF_SPACE_NAME` (+ `HF_TOKEN` if private)
- `WAVESPEED_API_KEY`
- `FAL_API_KEY`
- `TOGETHER_API_KEY`
- `PIXELDOJO_API_KEY`
- `ENABLE_SUMMARIZATION`

For full list, see `env.example`.

## 6) Security Checklist

- Keep secrets in `.env` only.
- Never commit `.env`.
- Rotate any key that was ever exposed in source history.

## 7) Fast Troubleshooting

### Bridge not reachable
```bash
curl http://localhost:7861/status
```
If it fails, restart bridge and verify port.

### “All providers failed”
- Check `.env` has at least one valid provider key.
- Confirm outbound network access from host.

### No LoRAs applied
- Confirm trigger keywords exist in `master_lora_dict.json`.
- Check bridge logs for matched LoRA IDs.

### Chat proxy failing (`/v1/chat/completions`)
- Set `YOU_COM_API_KEY`.
- Set `YOU_COM_DEFAULT_AGENT` or pass `agent_id` in request.


## 8) Install Included SillyTavern Plugin

Plugin source in this repo: `silly-tavern-pluggin/`

Copy into your SillyTavern repo:

```bash
mkdir -p /path/to/SillyTavern/public/scripts/extensions/auto-image-universal
cp -r silly-tavern-pluggin/* /path/to/SillyTavern/public/scripts/extensions/auto-image-universal/
```

Then reload SillyTavern.

If using OpenRouter prompt generation, set config at runtime (no hardcoded key):

```js
localStorage.setItem('autoImageUniversalConfig', JSON.stringify({
  OPENROUTER_API_KEY: 'your_openrouter_key'
}));
```
