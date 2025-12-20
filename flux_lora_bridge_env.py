#!/usr/bin/env python3
"""
Flux LoRA Bridge for SillyTavern
================================
AUTOMATIC1111-compatible API with multi-provider fallback

Provider Hierarchy (Daily Reset):
1. HF ZeroGPU (Free, unlimited LoRAs) - via Gradio Client
2. Together AI ($0.02/image, 10 LoRAs max) - via Together SDK
3. Wavespeed ($0.015/image, 4 LoRAs max) - via REST API

Features:
- Full AUTOMATIC1111 API compatibility
- Keyword-based LoRA injection
- Prompt deduplication
- Daily midnight reset
- Reads config from environment variables (bashrc)
"""

import json
import re
import logging
import base64
from datetime import datetime
from typing import List, Dict, Tuple
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import httpx
from PIL import Image
from io import BytesIO

# Import provider-specific clients
try:
    from gradio_client import Client as GradioClient
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    logging.warning("⚠️  gradio_client not installed - HF ZeroGPU will not work")

try:
    from together import Together
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False
    logging.warning("⚠️  together not installed - Together AI will not work")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#===========================================================================================
# CONFIGURATION - Reads from environment variables set in bashrc
#===========================================================================================

class Config:
    """Bridge configuration - reads from environment variables"""
    HOST = "0.0.0.0"
    PORT = int(os.getenv("BRIDGE_PORT", "7861"))

    # LoRA limits per provider
    MAX_LORAS_DEFAULT = 10
    MAX_LORAS_TOGETHER = 10
    MAX_LORAS_WAVESPEED = 4

    # Paths
    LORA_DICT_PATH = os.getenv("LORA_DICT_PATH", "master_lora_dict.json")

    # Provider credentials from environment
    HF_SPACE_NAME = os.getenv("HF_SPACE_NAME", "")
    HF_TOKEN = os.getenv("HF_TOKEN", "")

    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")

    WAVESPEED_API_KEY = os.getenv("WAVESPEED_API_KEY", "")
    WAVESPEED_ENDPOINT = "https://api.wavespeed.ai/api/v3/wavespeed-ai/flux-dev-lora"

    @classmethod
    def print_config(cls):
        """Print configuration status"""
        logger.info("=" * 80)
        logger.info("BRIDGE CONFIGURATION")
        logger.info("=" * 80)
        logger.info(f"Port: {cls.PORT}")
        logger.info(f"LoRA Dictionary: {cls.LORA_DICT_PATH}")
        logger.info("")
        logger.info("PROVIDER STATUS:")

        # HF ZeroGPU
        if cls.HF_SPACE_NAME:
            status = "✅ CONFIGURED" if GRADIO_AVAILABLE else "⚠️  Missing gradio_client"
            logger.info(f"  HF ZeroGPU: {status}")
            logger.info(f"    Space: {cls.HF_SPACE_NAME}")
            logger.info(f"    Token: {'✓ Set' if cls.HF_TOKEN else '✗ Not set (public space)'}")
        else:
            logger.info(f"  HF ZeroGPU: ✗ NOT CONFIGURED (HF_SPACE_NAME not set)")

        # Together AI
        if cls.TOGETHER_API_KEY:
            status = "✅ CONFIGURED" if TOGETHER_AVAILABLE else "⚠️  Missing together package"
            logger.info(f"  Together AI: {status}")
            logger.info(f"    API Key: {cls.TOGETHER_API_KEY[:20]}...")
        else:
            logger.info(f"  Together AI: ✗ NOT CONFIGURED (TOGETHER_API_KEY not set)")

        # Wavespeed
        if cls.WAVESPEED_API_KEY:
            logger.info(f"  Wavespeed: ✅ CONFIGURED")
            logger.info(f"    API Key: {cls.WAVESPEED_API_KEY[:20]}...")
        else:
            logger.info(f"  Wavespeed: ✗ NOT CONFIGURED (WAVESPEED_API_KEY not set)")

        logger.info("=" * 80)

#===========================================================================================
# PROVIDER STATE
#===========================================================================================

class ProviderState:
    """Manages provider selection and daily reset"""

    def __init__(self):
        self.providers = ["hf_zerogpu", "together", "wavespeed"]
        self.current_provider = "hf_zerogpu"
        self.last_reset_date = datetime.now().date()
        self.fallback_triggered = False
        logger.info(f"✅ Provider initialized: {self.current_provider}")

    def check_daily_reset(self):
        """Check if it's a new day and reset to primary"""
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            logger.info(f"🔄 Daily reset: {self.last_reset_date} → {current_date}")
            self.reset_to_primary()
            self.last_reset_date = current_date
            return True
        return False

    def reset_to_primary(self):
        """Reset to primary provider"""
        self.current_provider = "hf_zerogpu"
        self.fallback_triggered = False
        logger.info("✅ Reset to primary: HF ZeroGPU")

    def fallback_to_next(self):
        """Fallback to next provider"""
        current_index = self.providers.index(self.current_provider)
        if current_index < len(self.providers) - 1:
            self.current_provider = self.providers[current_index + 1]
            self.fallback_triggered = True
            logger.warning(f"⚠️  Falling back to: {self.current_provider}")
            return True
        else:
            logger.error("❌ All providers exhausted")
            return False

    def get_max_loras(self) -> int:
        """Get max LoRAs for current provider"""
        if self.current_provider == "wavespeed":
            return Config.MAX_LORAS_WAVESPEED
        elif self.current_provider == "together":
            return Config.MAX_LORAS_TOGETHER
        return Config.MAX_LORAS_DEFAULT

#===========================================================================================
# LORA MANAGER
#===========================================================================================

class LoRAManager:
    """Manages LoRA dictionary and keyword injection"""

    def __init__(self, dict_path: str):
        self.dict_path = dict_path
        self.lora_dict = self._load_dict()
        logger.info(f"✅ Loaded {len(self.lora_dict['loras'])} LoRAs")

    def _load_dict(self) -> Dict:
        """Load master LoRA dictionary"""
        with open(self.dict_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_permanent_loras(self) -> List[str]:
        """Get permanent LoRA IDs"""
        return self.lora_dict.get("config", {}).get("permanent_loras", [])

    def get_default_negative_prompt(self) -> str:
        """Get default negative prompt"""
        return self.lora_dict.get("config", {}).get("default_negative_prompt", "")

    def match_loras_by_keywords(self, prompt: str, negative_prompt: str = "") -> List[Dict]:
        """Match LoRAs based on keywords"""
        matched = []
        prompt_lower = prompt.lower()
        negative_lower = negative_prompt.lower()
        combined_text = f"{prompt_lower} {negative_lower}"

        for lora_id, lora_data in self.lora_dict["loras"].items():
            # Check if permanent
            if lora_data.get("permanent", False):
                matched.append({
                    "id": lora_id,
                    "data": lora_data,
                    "reason": "permanent"
                })
                continue

            # Check keywords
            keywords = lora_data.get("keywords", [])
            for keyword in keywords:
                if keyword.lower() in combined_text:
                    matched.append({
                        "id": lora_id,
                        "data": lora_data,
                        "reason": f"keyword:{keyword}"
                    })
                    break

        # Sort by rank (lower = higher priority)
        matched.sort(key=lambda x: x["data"]["rank"])
        return matched

    def build_lora_list(self, matched_loras: List[Dict], max_loras: int) -> List[Dict]:
        """Build final LoRA list"""
        lora_list = []
        seen_ids = set()

        for item in matched_loras:
            lora_id = item["id"]
            if lora_id in seen_ids:
                continue

            lora_data = item["data"]
            lora_list.append({
                "url": lora_data["url"],
                "weight": lora_data["weight"],
                "name": lora_data["name"],
                "id": lora_id
            })

            seen_ids.add(lora_id)

            if len(lora_list) >= max_loras:
                break

        return lora_list

    def build_enhanced_prompt(self, original_prompt: str, matched_loras: List[Dict]) -> Tuple[str, str]:
        """Build enhanced prompt with LoRA metadata"""
        prepend_parts = []
        append_parts = []
        negative_parts = [self.get_default_negative_prompt()]

        for item in matched_loras:
            lora_data = item["data"]

            prepend = lora_data.get("prepend_prompt", "").strip()
            if prepend:
                prepend_parts.append(prepend)

            append = lora_data.get("append_prompt", "").strip()
            if append:
                append_parts.append(append)

            negative = lora_data.get("negative_prompt", "").strip()
            if negative:
                negative_parts.append(negative)

        # Combine
        full_prompt = " ".join(filter(None, prepend_parts + [original_prompt] + append_parts))
        full_negative = ", ".join(filter(None, negative_parts))

        # Deduplicate
        full_prompt = self._deduplicate_prompt(full_prompt)
        full_negative = self._deduplicate_prompt(full_negative)

        return full_prompt, full_negative

    def _deduplicate_prompt(self, prompt: str) -> str:
        """Remove duplicate words while preserving order"""
        parts = re.split(r'(,|\.)', prompt)
        seen = set()
        deduped = []

        for part in parts:
            part_clean = part.strip().lower()
            if part_clean and part_clean not in seen:
                seen.add(part_clean)
                deduped.append(part.strip())
            elif not part_clean:
                deduped.append(part)

        return " ".join(deduped).strip()

#===========================================================================================
# PROVIDER CLIENTS
#===========================================================================================

class ProviderClient:
    """Base provider client"""
    async def generate(self, prompt: str, negative_prompt: str, loras: List[Dict], params: Dict) -> bytes:
        raise NotImplementedError

class HFZeroGPUClient(ProviderClient):
    """HuggingFace ZeroGPU client using Gradio"""

    def __init__(self):
        self.client = None
        if GRADIO_AVAILABLE and Config.HF_SPACE_NAME:
            try:
                if Config.HF_TOKEN:
                    self.client = GradioClient(Config.HF_SPACE_NAME, hf_token=Config.HF_TOKEN)
                else:
                    self.client = GradioClient(Config.HF_SPACE_NAME)
                logger.info(f"✅ Connected to HF Space: {Config.HF_SPACE_NAME}")
            except Exception as e:
                logger.error(f"❌ Failed to connect to HF Space: {e}")

    async def generate(self, prompt: str, negative_prompt: str, loras: List[Dict], params: Dict) -> bytes:
        if not self.client:
            raise ValueError("HF ZeroGPU client not initialized")

        # Build LoRA config JSON for HF Space
        lora_config = json.dumps([
            {
                "repo": lora["url"].split("huggingface.co/")[1].split("/resolve")[0] if "huggingface.co" in lora["url"] else lora["url"],
                "weights": lora["url"].split("/")[-1],
                "adapter_name": lora["id"],
                "adapter_weight": lora["weight"]
            }
            for lora in loras
        ])

        # Call the HF Space API
        result = self.client.predict(
            prompt=prompt,
            image_url="",  # No image-to-image
            lora_strings_json=lora_config,
            image_strength=0.75,
            cfg_scale=params["cfg_scale"],
            steps=params["steps"],
            randomize_seed=True,
            seed=params["seed"] if params["seed"] >= 0 else 0,
            width=params["width"],
            height=params["height"],
            upload_to_r2=False,
            account_id="",
            access_key="",
            secret_key="",
            bucket="",
            api_name="/run_lora"
        )

        # Result is tuple (image_path, json_result)
        if isinstance(result, tuple):
            image_path = result[0]
        else:
            image_path = result

        # Read image file
        with open(image_path, 'rb') as f:
            return f.read()

class TogetherAIClient(ProviderClient):
    """Together AI client using official SDK"""

    def __init__(self):
        self.client = None
        if TOGETHER_AVAILABLE and Config.TOGETHER_API_KEY:
            self.client = Together(api_key=Config.TOGETHER_API_KEY)
            logger.info("✅ Together AI client initialized")

    async def generate(self, prompt: str, negative_prompt: str, loras: List[Dict], params: Dict) -> bytes:
        if not self.client:
            raise ValueError("Together AI client not initialized")

        # Together AI format for image_loras
        image_loras = [
            {
                "path": lora["url"],
                "scale": lora["weight"]
            }
            for lora in loras
        ]

        # Generate image
        response = self.client.images.generate(
            prompt=f"{prompt}. {negative_prompt}" if negative_prompt else prompt,
            model="black-forest-labs/FLUX.1-dev-lora",
            width=params["width"],
            height=params["height"],
            steps=params["steps"],
            n=1,
            image_loras=image_loras
        )

        # Download image from URL
        image_url = response.data[0].url
        async with httpx.AsyncClient() as http_client:
            img_response = await http_client.get(image_url)
            return img_response.content

class WavespeedClient(ProviderClient):
    """Wavespeed client via REST API"""

    async def generate(self, prompt: str, negative_prompt: str, loras: List[Dict], params: Dict) -> bytes:
        if not Config.WAVESPEED_API_KEY:
            raise ValueError("WAVESPEED_API_KEY not configured")

        headers = {
            "Authorization": f"Bearer {Config.WAVESPEED_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "prompt": f"{prompt}. Negative: {negative_prompt}" if negative_prompt else prompt,
            "loras": [{"path": l["url"], "scale": l["weight"]} for l in loras],
            "size": f"{params['width']}x{params['height']}",
            "num_inference_steps": params["steps"],
            "guidance_scale": params["cfg_scale"],
            "enable_sync_mode": True,
            "output_format": "jpeg"
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(Config.WAVESPEED_ENDPOINT, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()

            # Get image URL from response
            if "data" in result and "outputs" in result["data"]:
                image_url = result["data"]["outputs"][0]
                img_response = await client.get(image_url)
                return img_response.content
            else:
                raise ValueError(f"Unexpected response format: {result}")

#===========================================================================================
# FASTAPI APP
#===========================================================================================

app = FastAPI(title="Flux LoRA Bridge", version="1.0.0")

# Initialize
provider_state = ProviderState()
lora_manager = LoRAManager(Config.LORA_DICT_PATH)

# Initialize clients
clients = {
    "hf_zerogpu": HFZeroGPUClient(),
    "together": TogetherAIClient(),
    "wavespeed": WavespeedClient()
}

#===========================================================================================
# API MODELS (AUTOMATIC1111 Compatible)
#===========================================================================================

class Txt2ImgRequest(BaseModel):
    """A1111-compatible txt2img request"""
    prompt: str = Field(default="", description="Prompt")
    negative_prompt: str = Field(default="", description="Negative prompt")
    steps: int = Field(default=40, description="Sampling steps")
    cfg_scale: float = Field(default=3.5, description="CFG scale")
    width: int = Field(default=1024, description="Image width")
    height: int = Field(default=1024, description="Image height")
    seed: int = Field(default=-1, description="Seed (-1 for random)")
    batch_size: int = Field(default=1, description="Batch size")
    n_iter: int = Field(default=1, description="Number of iterations")

    # A1111 compatibility fields (ignored but accepted)
    sampler_name: str = Field(default="Euler a")
    sampler_index: str = Field(default="Euler a")
    enable_hr: bool = Field(default=False)
    denoising_strength: float = Field(default=0.7)
    restore_faces: bool = Field(default=False)
    tiling: bool = Field(default=False)
    override_settings: dict = Field(default_factory=dict)
    override_settings_restore_afterwards: bool = Field(default=True)

class Txt2ImgResponse(BaseModel):
    """A1111-compatible txt2img response"""
    images: List[str] = Field(description="Base64-encoded images")
    parameters: dict = Field(description="Generation parameters")
    info: str = Field(description="Generation info (JSON string)")

#===========================================================================================
# ENDPOINTS
#===========================================================================================

@app.on_event("startup")
async def startup_event():
    """Startup tasks"""
    logger.info("🚀 Flux LoRA Bridge starting...")
    Config.print_config()
    provider_state.check_daily_reset()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Flux LoRA Bridge",
        "version": "1.0.0",
        "status": "running",
        "provider": provider_state.current_provider,
        "compatibility": "AUTOMATIC1111 API"
    }

@app.get("/sdapi/v1/options")
async def get_options():
    """A1111 compatibility: Get options"""
    return {}

@app.get("/sdapi/v1/sd-models")
async def get_models():
    """A1111 compatibility: List models"""
    return [{"title": "flux-dev", "model_name": "flux-dev", "hash": "flux1dev"}]

@app.get("/sdapi/v1/samplers")
async def get_samplers():
    """A1111 compatibility: List samplers"""
    return [{"name": "Euler a", "aliases": ["euler_a"]}]

@app.get("/status")
async def get_status():
    """Get bridge status"""
    return {
        "status": "running",
        "current_provider": provider_state.current_provider,
        "fallback_triggered": provider_state.fallback_triggered,
        "max_loras": provider_state.get_max_loras(),
        "last_reset_date": str(provider_state.last_reset_date),
        "total_loras": len(lora_manager.lora_dict["loras"])
    }

@app.post("/reset")
async def manual_reset():
    """Manually reset to primary provider"""
    provider_state.reset_to_primary()
    return {"message": "Reset to primary", "provider": provider_state.current_provider}

@app.post("/sdapi/v1/txt2img")
async def txt2img(request: Txt2ImgRequest):
    """AUTOMATIC1111-compatible txt2img endpoint"""
    # Check daily reset
    provider_state.check_daily_reset()

    # Match LoRAs
    matched_loras = lora_manager.match_loras_by_keywords(request.prompt, request.negative_prompt)
    max_loras = provider_state.get_max_loras()
    lora_list = lora_manager.build_lora_list(matched_loras, max_loras)

    # Build enhanced prompt
    full_prompt, full_negative = lora_manager.build_enhanced_prompt(
        request.prompt, matched_loras[:max_loras]
    )

    logger.info(f"🎨 Generation:")
    logger.info(f"   Provider: {provider_state.current_provider}")
    logger.info(f"   Original: {request.prompt[:80]}...")
    logger.info(f"   Enhanced: {full_prompt[:80]}...")
    logger.info(f"   LoRAs: {len(lora_list)}")
    for lora in lora_list[:5]:  # Show first 5
        logger.info(f"      - {lora['name']} ({lora['weight']})")

    # Generation params
    params = {
        "steps": request.steps,
        "cfg_scale": request.cfg_scale,
        "width": request.width,
        "height": request.height,
        "seed": request.seed
    }

    # Try generation with fallback
    max_retries = len(provider_state.providers)
    for attempt in range(max_retries):
        try:
            client = clients[provider_state.current_provider]
            image_bytes = await client.generate(full_prompt, full_negative, lora_list, params)

            logger.info(f"✅ Success with {provider_state.current_provider}")

            # Convert to base64 (A1111 format)
            base64_image = base64.b64encode(image_bytes).decode('utf-8')

            # Build response
            info_dict = {
                "prompt": full_prompt,
                "negative_prompt": full_negative,
                "steps": request.steps,
                "cfg_scale": request.cfg_scale,
                "width": request.width,
                "height": request.height,
                "seed": request.seed,
                "provider": provider_state.current_provider,
                "loras_used": len(lora_list)
            }

            return Txt2ImgResponse(
                images=[base64_image],
                parameters=info_dict,
                info=json.dumps(info_dict)
            )

        except Exception as e:
            logger.error(f"❌ Failed with {provider_state.current_provider}: {e}")

            if attempt < max_retries - 1:
                if provider_state.fallback_to_next():
                    max_loras = provider_state.get_max_loras()
                    lora_list = lora_manager.build_lora_list(matched_loras, max_loras)
                    logger.info(f"🔄 Retry with {provider_state.current_provider}")
                else:
                    break
            else:
                break

    # All failed
    raise HTTPException(status_code=500, detail="All providers failed")

#===========================================================================================
# MAIN
#===========================================================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        log_level="info"
    )
