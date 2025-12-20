# 🚀 QUICK REFERENCE CARD

## Installation (3 Commands)

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with API keys, then:
python flux_lora_bridge.py
```

## SillyTavern Setup (4 Clicks)

1. **Extensions** → **Image Generation**
2. **Source**: `Stable Diffusion`
3. **SD WebUI URL**: `http://localhost:7860`
4. **Enable Image Generation**: ✓

Done! 🎉

## File Checklist

```
flux_lora_bridge/
├── flux_lora_bridge.py        ← Main bridge
├── master_lora_dict.json      ← Your LoRA database (47 LoRAs)
├── requirements.txt           ← Python dependencies
├── .env                       ← Your API keys (create from .env.example)
├── .env.example               ← Template
└── SILLYTAVERN_INTEGRATION.md ← Full guide
```

## Provider Priority (Daily Reset)

```
1. HF ZeroGPU    → FREE, unlimited LoRAs
2. Together AI   → $0.02/image, 10 LoRAs max
3. Wavespeed     → $0.015/image, 4 LoRAs max

Resets to #1 at midnight automatically
```

## Essential Commands

```bash
# Start bridge
python flux_lora_bridge.py

# Check status
curl http://localhost:7860/status

# Reset to primary provider
curl -X POST http://localhost:7860/reset

# Test generation
curl http://localhost:7860/status
```

## Keyword Cheat Sheet

### Characters
- `nimya` `nimya33` → Main character
- `altnimya33` → Alternative
- `shraddha` `sara` `hansika` → Other actresses

### NSFW
- `nsfw` `naked` `nude` → General
- `nsfw3` → Strong nude
- `nsfw2` → Artistic
- `nsfw_photo` → Photorealistic
- `blowjob` `bj` → Oral POV
- `cowgirl` `riding` → Cowgirl POV
- `from behind` `doggy` → Doggy style
- `showing pussy` → Pussy exposed
- `showing ass` → Ass exposed

### Expressions
- `lip biting` → Biting lip
- `blushing` → Red cheeks
- `licking lips` → Tongue out
- `orgasm face` `ahegao` → Pleasure
- `sad` `angry` `happy` → Emotions
- `seductive facial expression` → Seductive

## Troubleshooting (30-Second Fixes)

### "Cannot connect"
```bash
# Is bridge running?
curl http://localhost:7860/status

# Restart bridge
python flux_lora_bridge.py
```

### "All providers failed"
```bash
# Check .env has credentials
cat .env

# Reset and retry
curl -X POST http://localhost:7860/reset
```

### "No auto-generation"
- SillyTavern → Extensions → Image Generation
- ✓ Enable Image Generation
- ✓ Send Requests Automatically

### "Wrong LoRAs"
- Check logs: Bridge shows matched LoRAs
- Use specific keywords (nsfw3, altnimya33, etc.)
- Lower rank = higher priority

## Bridge Logs Explained

```
INFO: 🎨 Generation:
INFO:    Provider: together           ← Current provider
INFO:    Original: nimya smiling...   ← Your prompt
INFO:    Enhanced: nimya33, a beau... ← With LoRA triggers
INFO:    LoRAs: 4                     ← Number of LoRAs used
INFO:       - Shakkar 24 Nimya (1.0)  ← LoRAs applied
INFO:       - Realism LoRA (0.4)
INFO:       - Imagination (0.5)
INFO:       - Indian Style Face (0.4)
INFO: ✅ Success with together        ← Generation succeeded
```

## URL Reference

| Service | URL | Purpose |
|---------|-----|---------|
| Bridge | http://localhost:7860 | Main API |
| Status | http://localhost:7860/status | Check status |
| Reset | http://localhost:7860/reset | Reset provider |
| SillyTavern | http://localhost:8000 | Your ST instance |
| Together AI | https://api.together.xyz | Get API key |
| Wavespeed | https://wavespeed.ai | Get API key |
| HuggingFace | https://huggingface.co | Create Space |

## Cost Calculator

| Provider | Cost/Image | 100 images | 1000 images |
|----------|------------|------------|-------------|
| HF ZeroGPU | FREE | $0 | $0 |
| Together AI | $0.02 | $2 | $20 |
| Wavespeed | $0.015 | $1.50 | $15 |

*Together AI: $25 free credits = 1,250 free images*

## Emergency Fixes

**Bridge crashed?**
```bash
python flux_lora_bridge.py
```

**Can't connect?**
```bash
# Check firewall
sudo ufw allow 7860  # Linux
# Windows: Allow Python in Windows Firewall
```

**Providers failing?**
```bash
# Test provider APIs
echo $HF_ZEROGPU_ENDPOINT
echo $TOGETHER_API_KEY
echo $WAVESPEED_API_KEY
```

**Wrong images?**
```bash
# Check LoRA keywords
curl http://localhost:7860/status
# Shows total LoRAs loaded (should be 47)
```

## Support Files

- **Full Documentation**: SILLYTAVERN_INTEGRATION.md
- **LoRA Database**: master_lora_dict.json (edit to add/modify LoRAs)
- **Bridge Code**: flux_lora_bridge.py
- **Dependencies**: requirements.txt

## One-Line Tests

```bash
# Test 1: Bridge alive?
curl http://localhost:7860

# Test 2: Can generate?
curl -X POST http://localhost:7860/sdapi/v1/txt2img   -H "Content-Type: application/json"   -d '{"prompt": "test", "steps": 20}' | wc -c

# Test 3: LoRAs loaded?
curl http://localhost:7860/status | grep total_loras

# Test 4: Provider working?
curl http://localhost:7860/status | grep current_provider
```

## Performance Tips

1. **Lower steps** for faster generation: 20-30 steps
2. **Use Together AI** when HF is slow (queued)
3. **Reduce resolution** temporarily: 512x512 or 768x768
4. **Fewer LoRAs** = faster (but less control)
5. **Warm up HF Space** if using: Visit URL first to wake it

## Daily Workflow

**Morning:**
- Bridge auto-resets to HF ZeroGPU (free)
- Check status: `curl http://localhost:7860/status`

**During Use:**
- Bridge runs continuously in background
- Auto-falls back if provider fails
- Logs show real-time activity

**Evening:**
- Check which provider was used (for costs)
- Optional: Manual reset to free tier

**Night:**
- Leave bridge running (auto-resets at midnight)
- Or stop: Ctrl+C in bridge terminal

## Advanced: Run in Background

**Linux/Mac (tmux):**
```bash
tmux new -s flux
python flux_lora_bridge.py
# Detach: Ctrl+B then D
# Reattach: tmux attach -t flux
```

**Linux (systemd service):**
```bash
# See SILLYTAVERN_INTEGRATION.md for setup
sudo systemctl start flux-bridge
sudo systemctl status flux-bridge
```

**Windows:**
- Run in separate PowerShell window
- Or use Task Scheduler to run on startup

---

📖 **Full Guide**: See SILLYTAVERN_INTEGRATION.md
🐛 **Issues**: Check bridge logs and SillyTavern console (F12)
🎨 **Have fun generating!**
