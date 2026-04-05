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
2. Wavespeed
3. FAL
4. Together

## 5) Environment Variables You’ll Usually Set

- `RUNWARE_API_KEY`
- `WAVESPEED_API_KEY`
- `FAL_API_KEY`
- `TOGETHER_API_KEY`
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

### Chat proxy (`/v1/chat/completions`)
- Currently returns 501 (no proxy backend configured).
- Implement a proxy backend in `flux_lora_bridge.py` to enable.
