# SillyTavern Integration Guide

## 📋 Prerequisites

Before starting, ensure you have:
- [x] Python 3.10 or higher installed
- [x] SillyTavern installed and running
- [x] At least ONE provider API key/endpoint configured

## 🚀 Quick Start

### Step 1: Install Bridge

```bash
# Navigate to bridge directory
cd /path/to/flux_lora_bridge

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Providers

```bash
# Copy environment template
cp .env.example .env

# Edit with your favorite editor
nano .env   # or: vim .env, code .env, etc.
```

**Add at least ONE provider:**

#### Option A: HuggingFace ZeroGPU (FREE) ⭐ RECOMMENDED

1. Create a HuggingFace account at https://huggingface.co
2. Create a new Space with:
   - SDK: Gradio
   - Hardware: ZeroGPU (FREE tier)
3. Deploy a Flux + LoRA inference endpoint
4. Copy your Space URL (e.g., `https://your-username-flux-space.hf.space`)
5. Add to `.env`:
   ```bash
   HF_ZEROGPU_ENDPOINT=https://your-username-flux-space.hf.space/generate
   ```

#### Option B: Together AI ($0.02/image) ⭐ RECOMMENDED BACKUP

1. Sign up at https://api.together.xyz
2. Go to Settings → API Keys
3. Create a new API key
4. You get $25 free credits (1,250 images!)
5. Add to `.env`:
   ```bash
   TOGETHER_API_KEY=your_together_api_key_here
   ```

#### Option C: Wavespeed ($0.015/image)

1. Sign up at https://wavespeed.ai
2. Go to Dashboard → API Keys
3. Create API key and add credits
4. Add to `.env`:
   ```bash
   WAVESPEED_API_KEY=your_wavespeed_api_key_here
   ```

### Step 3: Verify master_lora_dict.json

Ensure `master_lora_dict.json` is in the same directory as `flux_lora_bridge.py`.

```bash
# Verify JSON is valid
python -c "import json; json.load(open('master_lora_dict.json')); print('✅ JSON valid')"
```

### Step 4: Start the Bridge

```bash
# Start the bridge
python flux_lora_bridge.py
```

You should see:
```
INFO: 🚀 Flux LoRA Bridge starting...
INFO: ✅ Loaded 47 LoRAs
INFO: ✅ Provider initialized: hf_zerogpu
INFO: Uvicorn running on http://0.0.0.0:7860
```

**Keep this terminal window open!** The bridge needs to run continuously.

### Step 5: Test the Bridge

Open a new terminal and test:

```bash
# Test status
curl http://localhost:7860/status

# Should return:
# {
#   "status": "running",
#   "current_provider": "hf_zerogpu",
#   "total_loras": 47
# }
```

## 🎨 SillyTavern Configuration

### Method 1: Using the UI (Recommended)

1. **Open SillyTavern** in your browser (usually `http://localhost:8000`)

2. **Open Extensions Panel**
   - Click the **Extensions** icon (puzzle piece) in the top-right
   - Or press the extensions hotkey

3. **Navigate to Image Generation**
   - In the Extensions menu, click **Image Generation**

4. **Configure Settings**

   **API Settings:**
   - **Source**: Select `Stable Diffusion`
   - **SD WebUI URL**: Enter `http://localhost:7860`
   - Click **Connect** or **Test**
   - You should see "✅ Connected" or similar

   **Generation Settings:**
   - **Prompt Prefix**: Leave empty or customize (optional)
   - **Prompt**: Default is usually `{{char}}, {{user}}'s message`
   - **Negative Prompt**: Can leave empty (bridge adds defaults)
   - **Width**: 1024 (recommended for Flux)
   - **Height**: 1024 (recommended for Flux)
   - **Steps**: 20-40 (default 40)
   - **CFG Scale**: 3.5 (Flux optimal)
   - **Sampler**: Any (ignored by bridge, but won't cause errors)

   **Auto-Generation:**
   - ☑️ **Enable Image Generation**: Check this
   - ☑️ **Send Requests Automatically**: Check this for auto-gen on messages

5. **Save Settings**
   - Click **Save** or settings auto-save

### Method 2: Using config.yaml (Advanced)

If you prefer editing the config file directly:

```yaml
# In SillyTavern/public/settings.yaml or user config

sd:
  source: 'stable-diffusion'
  url: 'http://localhost:7860'

  # Generation parameters
  width: 1024
  height: 1024
  steps: 40
  cfg_scale: 3.5

  # Prompts
  prompt: '{{char}}, {{user}}'s message'
  negative_prompt: ''

  # Auto-generation
  enabled: true
  interactive_mode: true
```

## ✅ Verify Integration

### Test 1: Manual Generation

1. In SillyTavern, open a character chat
2. In the chat input, type: `/imagine nimya33 woman in a red dress, portrait`
3. Press Enter
4. Check bridge logs - you should see:
   ```
   INFO: 🎨 Generation:
   INFO:    Provider: hf_zerogpu
   INFO:    LoRAs: 4
   INFO: ✅ Success with hf_zerogpu
   ```
5. Image should appear in SillyTavern

### Test 2: Automatic Generation

1. Chat with your character
2. Type a normal message: "Show me what you're wearing today"
3. If auto-generation is enabled, an image should generate automatically
4. Check bridge logs for generation activity

### Test 3: LoRA Injection

Try these prompts to test keyword-based LoRA injection:

**Character keyword:**
```
nimya33 smiling portrait
```
Should inject: shakkar_24_nimya + permanent LoRAs

**Expression keyword:**
```
nimya lip biting seductive look
```
Should inject: lip_biting LoRA

**NSFW keyword:**
```
nimya nude on bed
```
Should inject: nsfw_master LoRA

**Alternative character:**
```
shraddha in traditional dress
```
Should inject: shraddha LoRA instead of nimya

## 🔧 Troubleshooting

### Issue: "Cannot connect to SD WebUI"

**Symptoms:**
- SillyTavern shows connection error
- Red "X" or "Disconnected" status

**Solutions:**

1. **Check bridge is running:**
   ```bash
   curl http://localhost:7860/status
   ```
   If this fails, bridge isn't running or crashed.

2. **Check URL is correct:**
   - Must be: `http://localhost:7860`
   - NOT: `http://localhost:7860/` (trailing slash can cause issues)
   - NOT: `https://localhost:7860` (no HTTPS)

3. **Check firewall:**
   ```bash
   # Linux: Allow port 7860
   sudo ufw allow 7860

   # Windows: Check Windows Firewall
   # Add rule to allow Python on port 7860
   ```

4. **Check logs:**
   Look at the bridge terminal for errors

### Issue: "All providers failed"

**Symptoms:**
- Generation starts but fails
- Bridge logs show "❌ Failed with [provider]"

**Solutions:**

1. **Verify at least one provider is configured:**
   ```bash
   cat .env
   # Should show at least ONE provider with credentials
   ```

2. **Test provider directly:**
   ```bash
   # HF ZeroGPU
   curl $HF_ZEROGPU_ENDPOINT

   # Together AI
   curl https://api.together.xyz/v1/models      -H "Authorization: Bearer $TOGETHER_API_KEY"
   ```

3. **Check provider status:**
   - HF Space: Visit Space URL, ensure it's running (not sleeping)
   - Together AI: Check account has credits
   - Wavespeed: Check API key is valid

4. **Manual reset and retry:**
   ```bash
   curl -X POST http://localhost:7860/reset
   ```

### Issue: "Images not generating automatically"

**Symptoms:**
- Manual `/imagine` works
- But auto-generation doesn't trigger

**Solutions:**

1. **Check auto-generation is enabled:**
   - SillyTavern → Extensions → Image Generation
   - Verify "Enable Image Generation" is checked
   - Verify "Send Requests Automatically" is checked

2. **Check prompt template:**
   - Must include variables like `{{char}}` or `{{user}}`
   - Default: `{{char}}, {{user}}'s message`

3. **Check character settings:**
   - Some characters can have image generation disabled
   - Check character JSON: `extensions.imageGen.disabled`

### Issue: "LoRAs not injecting"

**Symptoms:**
- Generation works but wrong LoRAs are used
- Check bridge logs: "LoRAs: 3" but expected more

**Solutions:**

1. **Check keywords match:**
   - LoRA keywords are case-insensitive
   - Must appear in prompt or negative prompt
   - Example: "nimya" will match shakkar_24_nimya

2. **Check LoRA rank:**
   - Lower rank = higher priority
   - If max LoRAs reached, lower-ranked LoRAs win

3. **Check provider limits:**
   - HF ZeroGPU: Unlimited
   - Together AI: Max 10 LoRAs
   - Wavespeed: Max 4 LoRAs (auto-trimmed by rank)

4. **View matched LoRAs in logs:**
   ```
   INFO:    LoRAs: 4
   INFO:       - Shakkar 24 Nimya (Main Character) (1.0)
   INFO:       - Realism LoRA (0.4)
   ...
   ```

### Issue: "Images are low quality"

**Solutions:**

1. **Increase steps:**
   - SillyTavern → Image Generation → Steps: 40-50

2. **Adjust CFG scale:**
   - Flux optimal: 3.5
   - For more adherence: 4.0-5.0
   - For more creativity: 2.5-3.0

3. **Use better prompt:**
   - The bridge auto-adds quality prompts
   - But you can be more specific in your request

4. **Check provider:**
   - HF ZeroGPU: Free but may be slower/queued
   - Together AI / Wavespeed: Faster, more consistent

## 📊 Monitoring

### View Real-Time Logs

In the bridge terminal, you'll see:
```
INFO: 🎨 Generation:
INFO:    Provider: together
INFO:    Original: nimya woman smiling...
INFO:    Enhanced: nimya33, a beautiful south indian women...
INFO:    LoRAs: 4
INFO:       - Shakkar 24 Nimya (Main Character) (1.0)
INFO:       - Realism LoRA (0.4)
INFO:       - Imagination (Anatomy Fix) (0.5)
INFO:       - Indian Style Face & Skin (0.4)
INFO: ✅ Success with together
```

### Check Provider Status

```bash
curl http://localhost:7860/status | jq
```

Output:
```json
{
  "status": "running",
  "current_provider": "together",
  "fallback_triggered": false,
  "max_loras": 10,
  "last_reset_date": "2025-12-19",
  "total_loras": 47
}
```

### Manual Provider Reset

```bash
# Reset to primary (HF ZeroGPU)
curl -X POST http://localhost:7860/reset
```

## 🎯 Usage Tips

### Effective Prompting

**For character consistency:**
```
nimya33 [your scene description]
```
The bridge auto-adds character details.

**For expressions:**
```
nimya lip biting, seductive look
nimya blushing, shy expression  
nimya orgasm face, pleasure expression
```

**For NSFW:**
```
nimya nude on bed (general NSFW)
nimya nsfw3 (strong nude focus)
nimya nsfw2 (artistic nude)
nimya nsfw_photo (photorealistic nude)
```

**For other characters:**
```
shraddha in traditional dress
sara ali beach photoshoot
sai pallavi dancing
```

### Keyword Reference

**Character Keywords:**
- `nimya`, `nimya33` → Main character (shakkar_24)
- `altnimya33` → Alternative (shakkar_42 epoch 8)
- `nimya_astria` → Astria-trained version
- `shraddha`, `sara`, `hansika`, etc. → Other actresses

**NSFW Keywords:**
- `nsfw`, `naked`, `nude` → General NSFW (nsfw_master)
- `nsfw3` → Strong nude focus (nsfw_master_3)
- `nsfw2` → Artistic style (nsfw_master_2)
- `nsfw_photo` → Photorealistic (nsfw_master_photorealism)
- `blowjob`, `bj` → POV blowjob
- `cowgirl`, `riding` → POV cowgirl
- `from behind`, `doggy` → Sex from behind
- `blowbang` → Multiple men oral
- `showing pussy`, `spread legs` → Pussy show
- `showing ass` → Ass exposed

**Expression Keywords:**
- `lip biting` → Lip biting
- `shh`, `finger on lips` → Shushing gesture
- `biting finger` → Finger biting
- `blushing`, `red cheeks` → Blushing
- `licking lips` → Licking lips
- `orgasm face`, `ahegao` → Orgasm expression
- `sad`, `angry`, `happy` → Basic emotions
- `aroused facial expression` → Aroused
- `seductive facial expression` → Seductive

### Cost Management

**Free Tier (HF ZeroGPU):**
- Primary provider
- Unlimited generations
- May have queue times
- Resets to this every midnight

**Paid Backup (Together AI):**
- $0.02 per image
- $25 free credits = 1,250 images
- Only used if HF fails
- Fast, reliable

**Last Resort (Wavespeed):**
- $0.015 per image
- Only 4 LoRAs max
- Only used if both above fail
- Cheapest paid option

**Monitor costs:**
- Check bridge logs for which provider is used
- Check provider dashboards for usage
- Use `/reset` endpoint to force back to free tier

## 🔄 Updating

### Update LoRA Dictionary

```bash
# Edit master_lora_dict.json
nano master_lora_dict.json

# Restart bridge
# Ctrl+C to stop, then:
python flux_lora_bridge.py
```

### Update Bridge Code

```bash
# Download new version
# Replace flux_lora_bridge.py

# Restart bridge
python flux_lora_bridge.py
```

## 🚀 Production Deployment

### Run as Service (Linux)

Create `/etc/systemd/system/flux-bridge.service`:

```ini
[Unit]
Description=Flux LoRA Bridge
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/flux_lora_bridge
Environment="PATH=/usr/bin:/usr/local/bin"
EnvironmentFile=/path/to/flux_lora_bridge/.env
ExecStart=/usr/bin/python3 flux_lora_bridge.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable flux-bridge
sudo systemctl start flux-bridge
sudo systemctl status flux-bridge

# View logs
sudo journalctl -u flux-bridge -f
```

### Run in tmux (Quick Method)

```bash
# Start session
tmux new -s flux-bridge

# Run bridge
python flux_lora_bridge.py

# Detach: Press Ctrl+B, then D
# Reattach later: tmux attach -t flux-bridge
```

## 📞 Support

If you encounter issues:

1. Check bridge logs (terminal output)
2. Check SillyTavern console (F12 in browser)
3. Verify provider status (`curl http://localhost:7860/status`)
4. Test provider APIs directly
5. Check this guide's troubleshooting section

---

**🎉 Enjoy your AI-powered image generation with dynamic LoRA injection!**
