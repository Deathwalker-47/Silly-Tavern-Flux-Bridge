"""Every image and video route the bridge may use, with each model's own limits.

A route that isn't listed here can't be selected or reached by a fallback, which is
how SFW-only and low-quality models stay out of the chat pipeline. Every provider and
model has different limits, so nothing here is shared between models.

Provider -> family -> options. The first option in a family is its default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from . import db

PORTRAIT = ("9:16", "3:4")
LANDSCAPE = ("16:9", "4:3")
ALL_SHAPES = PORTRAIT + LANDSCAPE
IMAGE_FLOOR_PIXELS = 1_000_000   # 1K: nothing under about 1 megapixel is usable
LONGEST_SECONDS = 15

IMAGE_MODES = ("lora", "reference", "text")
VIDEO_MODES = ("reference", "first", "first_last")


class Route(str, Enum):
    ALLOWED = "allowed"
    CHECK = "check"
    BANNED = "banned"


class RouteNotAllowed(RuntimeError):
    pass


@dataclass
class ModelOption:
    id: str
    label: str
    kind: str                       # "image" | "video"
    family: str                     # "seedream-4.5", "wan-3.0" ...
    provider: str
    modes: tuple                    # image: ("lora",), ("reference",) or ("text",); video: VIDEO_MODES
    max_refs: int = 0
    ref_field: str = ""             # the provider's own field name for this model
    lora_arch: Optional[str] = None  # "flux" | "zimage"
    resolutions: tuple = ()
    lowest: str = ""
    seconds: tuple = ()             # every duration this model accepts
    max_prompt: Optional[int] = None
    aspects: tuple = ("9:16", "16:9")  # what the director may choose; empty = the first frame sets the shape
    ref_token: str = "image {n}"
    negative_prompt: bool = False
    nsfw: str = "allowed"           # "allowed" | "check" | "banned"
    tested: bool = False
    hidden: bool = False            # a step inside another option, not a picker entry
    hint: str = ""
    usd: Optional[float] = None     # per image
    usd_per_second: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def people_cap(self) -> int:
        return 6 if self.max_refs >= 30 else 3

    def longest(self, cap: int = LONGEST_SECONDS) -> int:
        return max(s for s in self.seconds if s <= cap)

    def limits_text(self) -> str:
        aspects = ", ".join(self.aspects) or "none, the first frame sets the shape"
        return f"Allowed aspect ratios: {aspects}. Prompt limit: {self.max_prompt or 'none'}."

    def cost(self, seconds: Optional[float] = None) -> Optional[float]:
        if self.usd is not None:
            return self.usd
        if self.usd_per_second is not None and seconds:
            return round(self.usd_per_second * seconds, 4)
        return None

    def to_public(self) -> Dict[str, Any]:
        return {
            "id": self.id, "label": self.label, "modes": list(self.modes), "max_refs": self.max_refs,
            "lora_arch": self.lora_arch, "resolution": self.lowest, "seconds": self.longest() if self.seconds else None,
            "aspects": list(self.aspects), "hint": self.hint, "audio_switch": bool(self.extra.get("audio_switch")),
            "status": route_status(self).value,
        }


def pick_aspect(option: ModelOption, wanted: str) -> str:
    """The director's choice held to what this model offers: portrait 9:16, 3:4, then landscape."""
    if not option.aspects or wanted in option.aspects:
        return wanted
    first, second = (LANDSCAPE, PORTRAIT) if wanted in LANDSCAPE else (PORTRAIT, LANDSCAPE)
    return next(a for a in first + second if a in option.aspects)


def _span(low: int, high: int) -> tuple:
    return tuple(range(low, high + 1))


def _i(provider: str, family: str, id: str, label: str, modes: tuple, **kw) -> ModelOption:
    known = {k: kw.pop(k) for k in list(kw) if k in ModelOption.__dataclass_fields__}
    return ModelOption(id=id, label=label, kind="image", family=family, provider=provider, modes=modes,
                       extra=kw, **known)


def _v(provider: str, family: str, id: str, label: str, modes: tuple, res: tuple, seconds: tuple, **kw) -> ModelOption:
    known = {k: kw.pop(k) for k in list(kw) if k in ModelOption.__dataclass_fields__}
    known.setdefault("aspects", ())
    known.setdefault("lowest", res[0])
    return ModelOption(id=id, label=label, kind="video", family=family, provider=provider, modes=modes,
                       resolutions=res, seconds=seconds, extra=kw, **known)


REF = ("reference",)
FRAMES = ("first", "first_last")
FIRST = ("first",)
ALL = VIDEO_MODES

FLUX_SIZES = {"9:16": (768, 1344), "16:9": (1344, 768)}
SEEDREAM_45_SIZES = {"9:16": (2304, 4096), "16:9": (4096, 2304), "3:4": (2304, 3072), "4:3": (3072, 2304)}
SEEDANCE_TOKEN = "Image{n}"
H3_TOKEN = "<Picture {n}>"

IMAGE_FAMILIES = {
    "flux-lora": "Flux Dev + LoRA",
    "zimage": "Z-Image + LoRA",
    "qwen-edit": "Qwen Edit",
    "seedream-4.5": "Seedream 4.5",
    "seedream-5-pro": "Seedream 5 Pro",
    "seedream-5.5": "Seedream 5.5",
    "best": "The Best Option",
}

VIDEO_FAMILIES = {
    "wan-3.0": "Wan 3.0",
    "seedance-2": "Seedance 2",
    "seedance-2.5": "Seedance 2.5",
    "minimax-h3": "MiniMax H3",
    "ltx-2.3": "LTX-2.3",
    "ltx-2.5": "LTX-2.5",
    "wan-2.2": "Wan 2.2",
}

PROVIDER_LABELS = {
    "runware": "Runware",
    "nanogpt": "NanoGPT",
    "sogni": "Sogni",
    "replicate": "Replicate",
    "pixeldojo": "PixelDojo (expensive)",
    "venice": "Venice",
    "spicyapi": "SpicyAPI",
    "mock": "Mock (local test files)",
    "huggingface": "Hugging Face (private LoRAs)",
}

IMAGE_CATALOG: Dict[str, Dict[str, List[ModelOption]]] = {
    "runware": {
        "flux-lora": [_i("runware", "flux-lora", "runware:101@1", "Flux Dev", ("lora",), ref_field="lora", lora_arch="flux",
                         lowest="768x1344", usd=0.0037, hint="$0.004", tested=True, sizes=FLUX_SIZES, steps=28, cfg=3.5)],
        "zimage": [_i("runware", "zimage", "rundiffusion:200@100", "Juggernaut Z", ("lora",), ref_field="lora", lora_arch="zimage",
                      lowest="768x1344", usd=0.0129, hint="$0.013", tested=True, sizes=FLUX_SIZES, steps=20)],
    },
    "replicate": {
        "flux-lora": [_i("replicate", "flux-lora", "lucataco/flux-dev-multi-lora", "Flux Dev", ("lora",), ref_field="hf_loras",
                         lora_arch="flux", lowest="768x1344", usd=0.02, hint="~$0.02", tested=True, sizes=FLUX_SIZES, steps=28, cfg=3.5)],
    },
    "nanogpt": {
        "qwen-edit": [_i("nanogpt", "qwen-edit", "qwen-edit-uncensored", "Qwen Edit Uncensored", REF, tested=True, max_refs=3, ref_field="input_references",
                         max_prompt=1500, aspects=("9:16", "3:4", "16:9"), lowest="1k", usd=0.04, hint="$0.04, 3 refs",
                         resolution="1k", send_aspect=True, needs_ref=True)],
        "seedream-4.5": [_i("nanogpt", "seedream-4.5", "seedream-v4.5", "Seedream 4.5", REF, tested=True, max_refs=10, ref_field="input_references",
                            aspects=ALL_SHAPES, lowest="2304x4096", usd=0.04, hint="$0.04, 10 refs", sizes=SEEDREAM_45_SIZES)],
        "seedream-5-pro": [_i("nanogpt", "seedream-5-pro", "bytedance/seedream-v5.0-pro/edit", "Seedream 5 Pro", REF, tested=True, max_refs=10,
                              ref_field="input_references", aspects=ALL_SHAPES, lowest="1k", usd=0.09,
                              hint="$0.09 in the scene's shape, $0.045 at 1k in the references' shape (Media Studio setting). 10 refs",
                              hidden=True, resolution="1k", send_aspect=True, cheaper_without_aspect=True, needs_ref=True)],
        "best": [_i("nanogpt", "best", "best", "The Best Option", REF, tested=True, max_refs=10, aspects=ALL_SHAPES, lowest="1k",
                    usd=0.13, hint="$0.13: Seedream 5 Pro scene ($0.09), then Seedream 4.5 face pass ($0.04). $0.085 with the 1k setting")],
    },
    "sogni": {
        "zimage": [_i("sogni", "zimage", "z_image_turbo_bf16", "Z-Image Turbo", ("text",), tested=True, lowest="768x1344", hint="Spark, no LoRA", sizes=FLUX_SIZES)],
        "qwen-edit": [_i("sogni", "qwen-edit", "qwen_image_edit_2511_fp8_lightning", "Qwen Edit 2511", REF, tested=True, max_refs=3, ref_field="contextImages",
                         lowest="768x1344", hint="Spark, 3 refs", sizes=FLUX_SIZES, needs_ref=True)],
        "seedream-5-pro": [_i("sogni", "seedream-5-pro", "seedream-5-0-pro-uncensored", "Seedream 5 Pro Uncensored", REF, max_refs=10,
                              ref_field="contextImages", lowest="768x1344", hint="Spark, 10 refs", hidden=True, sizes=FLUX_SIZES)],
    },
    "pixeldojo": {
        "seedream-4.5": [_i("pixeldojo", "seedream-4.5", "seedream-4", "Seedream 4.5", REF, tested=True, max_refs=14, ref_field="image_urls",
                            aspects=ALL_SHAPES, lowest="portrait_9_16", hint="1.25 cr, 14 refs", build="seedream_4")],
        "seedream-5-pro": [_i("pixeldojo", "seedream-5-pro", "seedream-5", "Seedream 5", REF, tested=True, max_refs=10, ref_field="image_urls",
                              lowest="768x1344", hint="1.5 cr at 1K, 10 refs", hidden=True, build="seedream_5", sizes=FLUX_SIZES)],
        "best": [_i("pixeldojo", "best", "best", "The Best Option", REF, tested=True, max_refs=10, aspects=ALL_SHAPES, lowest="1k",
                    hint="2.75 cr: Seedream 5 scene (1.5 cr), then Seedream 4.5 face pass (1.25 cr)")],
    },
    # SpicyAPI: one model id per task, listed in tasks. LoRAs go by signed link, three at most.
    "spicyapi": {
        "flux-lora": [_i("spicyapi", "flux-lora", "black-forest-labs/flux-1-dev-lora", "Flux Dev", ("lora",), ref_field="loras", lora_arch="flux",
                         lowest="1MP", usd=0.018, hint="$0.018, 3 LoRAs at most", build="lora", steps=28, cfg=3.5,
                         tasks={"lora": "black-forest-labs/flux-1-dev-lora/text-to-image"})],
        "zimage": [
            _i("spicyapi", "zimage", "alibaba/z-image-turbo-lora", "Z-Image Turbo", ("lora",), ref_field="loras", lora_arch="zimage",
               lowest="1MP", usd=0.012, hint="$0.012, 3 LoRAs at most", build="lora", tasks={"lora": "alibaba/z-image-turbo-lora/text-to-image"}),
            _i("spicyapi", "zimage", "alibaba/z-image-spicy", "Z-Image Spicy", ("text",), lowest="864x1536", usd=0.0124, hint="$0.012, no LoRA",
               build="sized_text", sizes={"9:16": (864, 1536), "16:9": (1536, 864)}, tasks={"text": "alibaba/z-image-spicy/text-to-image"}),
        ],
        "qwen-edit": [
            _i("spicyapi", "qwen-edit", "alibaba/qwen-image-edit-2509", "Qwen Edit 2509", REF, max_refs=3, ref_field="image_urls", ref_token="Picture {n}",
               aspects=ALL_SHAPES, lowest="1MP", usd=0.03, hint="$0.03, 3 refs", build="qwen_edit", needs_ref=True,
               tasks={"reference": "alibaba/qwen-image-edit-2509-lora/edit"}),
            _i("spicyapi", "qwen-edit", "alibaba/qwen-image-spicy-edit", "Qwen Edit Spicy", REF, max_refs=1, ref_field="image_url", aspects=(),
               lowest="source", usd=0.038, hint="$0.038, 1 ref, keeps its shape", build="qwen_spicy", needs_ref=True,
               tasks={"reference": "alibaba/qwen-image-spicy-edit/edit"}),
        ],
        "seedream-4.5": [_i("spicyapi", "seedream-4.5", "bytedance/seedream-4.5", "Seedream 4.5", REF, max_refs=14, ref_field="image_urls",
                            ref_token="Image {n}", aspects=ALL_SHAPES, lowest="2k", usd=0.04, hint="$0.04, 14 refs", build="seedream", resolution="2k",
                            tasks={"reference": "bytedance/seedream-4.5/edit", "text": "bytedance/seedream-4.5/text-to-image"})],
        "seedream-5-pro": [_i("spicyapi", "seedream-5-pro", "bytedance/seedream-5.0-pro", "Seedream 5 Pro", REF, max_refs=10, ref_field="image_urls",
                              ref_token="Image {n}", aspects=ALL_SHAPES, lowest="1k", usd=0.036, hint="$0.036 at 1K, 10 refs",
                              build="seedream", resolution="1k",
                              tasks={"reference": "bytedance/seedream-5.0-pro/edit", "text": "bytedance/seedream-5.0-pro/text-to-image"})],
        # 1.5k costs the same as 1k here. The Best Option's scene step uses this one.
        "seedream-5.5": [_i("spicyapi", "seedream-5.5", "bytedance/seedream-5.5", "Seedream 5.5", REF, max_refs=10, ref_field="image_urls",
                            ref_token="Image {n}", aspects=ALL_SHAPES, lowest="1.5k", usd=0.02, hint="$0.02 at 1.5K, 10 refs",
                            build="seedream", resolution="1.5k", output_format="jpeg",
                            tasks={"reference": "bytedance/seedream-5.5/edit", "text": "bytedance/seedream-5.5/text-to-image"})],
        "best": [_i("spicyapi", "best", "best", "The Best Option", REF, max_refs=10, aspects=ALL_SHAPES, lowest="1.5k", usd=0.06,
                    hint="$0.06: Seedream 5.5 scene ($0.02), then Seedream 4.5 face pass ($0.04)")],
    },
    # Venice has no Seedream 4.5, so its Seedream 5 Pro stands alone and there's no Best Option here.
    "venice": {
        "qwen-edit": [_i("venice", "qwen-edit", "qwen-edit-uncensored", "Qwen Edit Uncensored", REF, max_refs=3, ref_field="images", max_prompt=1500,
                         aspects=ALL_SHAPES, lowest="1MP", usd=0.04, hint="$0.04, 3 refs", edit="qwen-edit-uncensored", needs_ref=True)],
        "seedream-5-pro": [_i("venice", "seedream-5-pro", "seedream-v5-pro", "Seedream 5 Pro", REF, max_refs=6, ref_field="images",
                              aspects=ALL_SHAPES, lowest="1K", usd=0.06, hint="$0.06 at 1K, 6 refs, each after the first $0.0035 more",
                              edit="seedream-v5-pro-edit", generate="seedream-v5-pro", resolution="1K")],
        "seedream-5.5": [_i("venice", "seedream-5.5", "seedream-v5-5", "Seedream 5.5", REF, max_refs=10, ref_field="images",
                            aspects=ALL_SHAPES, lowest="1K", usd=0.0255, hint="$0.026 at 1K, 10 refs, each $0.01 more",
                            edit="seedream-v5-5-edit", generate="seedream-v5-5", resolution="1K")],
        # No Z-Image: Venice's tops out at 1280 px a side (under 1 MP at 9:16), takes no LoRA and isn't tagged uncensored.
    },
}

VIDEO_CATALOG: Dict[str, Dict[str, List[ModelOption]]] = {
    "nanogpt": {
        "wan-3.0": [_v("nanogpt", "wan-3.0", "alibaba/wan-3.0/image-to-video-spicy", "Wan 3.0 Spicy", FIRST, ("480p", "720p", "1080p"), _span(2, 30), tested=True,
                       usd_per_second=0.05, hint="$0.05/s, 15 s clip $0.75", params=("resolution", "duration", "seed"))],
        "seedance-2": [
            _v("nanogpt", "seedance-2", "bytedance/seedance-2.0-fast/image-to-video-spicy", "Seedance 2 Fast Spicy", FRAMES,
               ("480p", "720p", "1080p"), _span(4, 15), usd_per_second=0.10, hint="$0.10/s, 15 s clip $1.50. Can undress unprompted", audio_switch=True,
               params=("resolution", "duration", "aspect_ratio", "generateAudio", "last_image", "seed")),
            _v("nanogpt", "seedance-2", "bytedance/seedance-2.0/image-to-video-spicy", "Seedance 2 Spicy", FRAMES,
               ("480p", "720p", "1080p"), _span(4, 15), usd_per_second=0.12, hint="$0.12/s, 15 s clip $1.80", audio_switch=True,
               params=("resolution", "duration", "aspect_ratio", "generateAudio", "last_image", "seed")),
            # Frames only: NanoGPT takes references as public links, and private photos never get one.
            _v("nanogpt", "seedance-2", "bytedance-seedance-2-0-fast", "Seedance 2 Fast", FRAMES, ("720p", "1080p"), (5, 10, 15), tested=True,
               aspects=ALL_SHAPES, usd_per_second=0.12,
               hint="$0.12/s, 15 s clip $1.80", params=("resolution", "duration", "aspect_ratio", "last_image")),
            _v("nanogpt", "seedance-2", "bytedance-seedance-2-0-mini", "Seedance 2 Mini", FRAMES, ("720p", "1080p"), _span(5, 12),
               aspects=ALL_SHAPES, usd_per_second=0.12,
               hint="$0.12/s, 12 s clip $1.44", audio_switch=True,
               params=("resolution", "duration", "aspect_ratio", "generateAudio", "last_image")),
        ],
        "seedance-2.5": [_v("nanogpt", "seedance-2.5", "bytedance/seedance-2.5-spicy", "Seedance 2.5 Spicy", FIRST,
                            ("480p", "720p", "1080p", "4K"), _span(4, 15), tested=True, usd_per_second=0.18, hint="$0.18/s, 15 s clip $2.70",
                            params=("resolution", "duration"))],
        "minimax-h3": [_v("nanogpt", "minimax-h3", "minimax-h3/image-to-video-spicy", "MiniMax H3 Spicy", FRAMES, ("480p", "768p"), _span(3, 15), tested=True,
                          usd_per_second=0.04, hint="$0.04/s, 15 s clip $0.60", params=("resolution", "duration", "last_image", "seed"))],
        "ltx-2.3": [_v("nanogpt", "ltx-2.3", "ltx-2.3-spicy/image-to-video", "LTX-2.3 Spicy", FIRST, ("480p", "720p", "1080p"), _span(3, 20), tested=True,
                       max_prompt=3000, usd_per_second=0.02, hint="$0.02/s", params=("resolution", "duration", "seed"))],
        "wan-2.2": [_v("nanogpt", "wan-2.2", "wan-video-22", "Wan 2.2", FIRST, ("480p", "720p"), (5, 8), usd_per_second=0.04,
                       hint="$0.04/s, 8 s longest", params=("resolution", "duration"))],
    },
    "sogni": {
        "minimax-h3": [
            _v("sogni", "minimax-h3", "minimax-h3-fastvideo-omni-int8_r2v_turbo", "MiniMax H3 Reference, FastH3", REF, ("768p",), _span(6, 15),
               max_refs=9, ref_field="contextImages", ref_token=H3_TOKEN, aspects=("9:16", "16:9"), usd_per_second=0.03, tested=True,
               hint="$0.03/s, 15 s clip $0.45", h3_lora=True, sizes={"9:16": (768, 1344), "16:9": (1344, 768)}),
            _v("sogni", "minimax-h3", "minimax-h3-ref2va-fp8_r2v_turbo", "MiniMax H3 Reference, Turbo", REF, ("544p",), _span(6, 15),
               max_refs=9, ref_field="contextImages", ref_token=H3_TOKEN, aspects=("9:16", "16:9"), usd_per_second=0.03, tested=True,
               hint="$0.03/s, 15 s clip $0.45", h3_lora=True, sizes={"9:16": (544, 960), "16:9": (960, 544)}),
            _v("sogni", "minimax-h3", "minimax-h3-ref2va-fp8_r2v_balanced", "MiniMax H3 Reference, Balanced", REF, ("544p",), _span(6, 15),
               max_refs=9, ref_field="contextImages", ref_token=H3_TOKEN, aspects=("9:16", "16:9"), usd_per_second=0.05, tested=True,
               hint="15 s clip $1.07", h3_lora=True, sizes={"9:16": (544, 960), "16:9": (960, 544)}),
            _v("sogni", "minimax-h3", "minimax-h3-ref2va-fp8_r2v", "MiniMax H3 Reference, Standard", REF, ("544p",), _span(6, 15),
               max_refs=9, ref_field="contextImages", ref_token=H3_TOKEN, aspects=("9:16", "16:9"), usd_per_second=0.08, tested=True,
               hint="15 s clip $1.53", h3_lora=True, sizes={"9:16": (544, 960), "16:9": (960, 544)}),
            _v("sogni", "minimax-h3", "minimax-h3-fastvideo-int8_i2v_turbo", "MiniMax H3 FastH3", FRAMES, ("default",), _span(6, 15),
               hint="Spark, fastest", h3_lora=True, last_model="minimax-h3-fastvideo-int8_flf2v_turbo"),
            _v("sogni", "minimax-h3", "minimax-h3-fl2va-fp8_i2v_turbo", "MiniMax H3 LightX2V Turbo", FRAMES, ("default",), _span(6, 15),
               hint="Spark, fast", h3_lora=True, last_model="minimax-h3-fl2va-fp8_flf2v_turbo"),
            _v("sogni", "minimax-h3", "minimax-h3-fl2va-fp8_i2v", "MiniMax H3 Standard", FRAMES, ("default",), _span(6, 15),
               hint="Spark, best quality, slow", h3_lora=True, last_model="minimax-h3-fl2va-fp8_flf2v"),
        ],
        "wan-3.0": [_v("sogni", "wan-3.0", "wan3.0-spicy-video", "Wan 3 Uncensored", ALL, ("480p", "720p", "1080p"), _span(2, 30), tested=True, max_refs=10,
                       ref_field="referenceImageUrls", ref_token="@Image {n}", aspects=ALL_SHAPES, usd_per_second=0.08,
                       hint="$0.08/s, 15 s clip $1.20", url_refs=True, sends_ratio=True, fps=30, tall={"480p": (480, 854)})],
        "seedance-2": [_v("sogni", "seedance-2", "seedance-2-0-mini-uncensored", "Seedance 2 Mini Uncensored", ALL, ("480p", "720p"), _span(4, 15), tested=True, max_refs=9,
                          ref_field="referenceImageUrls", ref_token=SEEDANCE_TOKEN, usd_per_second=0.084, hint="$0.084/s, 15 s clip $1.26",
                          url_refs=True, fps=24, tall={"480p": (480, 854)}, consent=True)],
        "seedance-2.5": [_v("sogni", "seedance-2.5", "seedance-2-5-uncensored", "Seedance 2.5 Uncensored", ALL, ("480p", "720p", "1080p"), _span(4, 30), tested=True, max_refs=30,
                            ref_field="referenceImageUrls", ref_token="@image{n}", usd_per_second=0.154, hint="$0.154/s, 15 s clip $2.31",
                            url_refs=True, seedance_task=True, fps=24, tall={"480p": (480, 854)}, consent=True)],
        "ltx-2.3": [
            _v("sogni", "ltx-2.3", "ltx23-22b-10eros-v1.4-fp8mixed_i2v", "LTX-2.3 10Eros", FIRST, ("480p", "720p", "1080p"), _span(2, 20),
               hint="Spark", sizing="wh", min_side=640, size_step=64),
            _v("sogni", "ltx-2.3", "ltx23-22b-fp8_i2v_distilled", "LTX-2.3 Distilled", FIRST, ("480p", "720p", "1080p"), _span(2, 20),
               hint="Spark, fast", sizing="wh", min_side=640, size_step=64),
        ],
        "ltx-2.5": [_v("sogni", "ltx-2.5", "ltx25-22b-int8_i2v_distilled", "LTX-2.5", FIRST, ("480p", "720p", "1080p"), _span(2, 20),
                       hint="Spark", sizing="wh", min_side=640, size_step=64)],
        "wan-2.2": [
            _v("sogni", "wan-2.2", "wan_v2.2-14b-fp8_i2v_lightx2v", "Wan 2.2 LightX2V", FIRST, ("480p", "720p"), _span(2, 8), hint="Spark, fast", sizing="wh", min_side=480, size_step=32),
            _v("sogni", "wan-2.2", "wan_v2.2-14b-fp8_i2v", "Wan 2.2", FIRST, ("480p", "720p"), _span(2, 8), hint="Spark, full quality", sizing="wh", min_side=480, size_step=32),
        ],
    },
    "pixeldojo": {
        "minimax-h3": [
            _v("pixeldojo", "minimax-h3", "minimax-h3-turbo", "MiniMax H3 Turbo", ALL, ("480p", "720p"), _span(5, 15), max_refs=9,
               ref_field="reference_images", ref_token=H3_TOKEN, aspects=("9:16", "3:4", "16:9", "4:3"), hint="0.75 cr/s, 15 s clip 11.25 cr",
               tested=True, build="h3_turbo"),
            _v("pixeldojo", "minimax-h3", "minimax-h3", "MiniMax H3", ALL, ("768p", "2k"), _span(5, 15), tested=True, max_refs=9, ref_field="reference_images",
               ref_token=H3_TOKEN, aspects=("9:16", "3:4", "16:9", "4:3"), hint="2.5 cr/s, 15 s clip 41.5 cr; first frame bills 4 cr/s", build="h3"),
        ],
        "wan-3.0": [_v("pixeldojo", "wan-3.0", "wan-3-0-video", "Wan 3.0", ALL, ("480P", "720P", "1080P", "2K", "4K"), _span(2, 30), tested=True, max_refs=10,
                       ref_field="reference_image_urls", ref_token="@Image {n}", aspects=ALL_SHAPES, hint="1.2 cr/s at 480P", audio_switch=True,
                       build="wan_3")],
        "seedance-2.5": [_v("pixeldojo", "seedance-2.5", "seedance-2-5", "Seedance 2.5", ALL, ("480p", "720p", "1080p"), _span(4, 30), tested=True, max_refs=30,
                            ref_field="reference_images", ref_token="@Image{n}", aspects=ALL_SHAPES, hint="3 cr/s, 15 s clip 45 cr", audio_switch=True,
                            build="seedance_25")],
        "seedance-2": [_v("pixeldojo", "seedance-2", "seedance-2-reference", "Seedance 2 Reference", REF, ("480p", "720p", "1080p", "4k"), _span(4, 15), tested=True,
                          max_refs=9, ref_field="reference_images", ref_token=SEEDANCE_TOKEN, aspects=ALL_SHAPES, hint="2 cr/s, 15 s clip 30 cr", audio_switch=True,
                          build="seedance_2_reference")],
    },
    # SpicyAPI: no content filter on any of these. The plain models are its "uncensored" ones; the
    # Spicy ones are tuned for adult scenes and take frames only. Prices are at the lowest resolution.
    "spicyapi": {
        "wan-3.0": [
            _v("spicyapi", "wan-3.0", "alibaba/wan-3.0", "Wan 3.0", ALL, ("480p", "720p", "1080p"), _span(2, 30), max_refs=10,
               ref_field="reference_image_urls", ref_token="Image {n}", aspects=ALL_SHAPES, usd_per_second=0.045, hint="$0.045/s, 15 s clip $0.68",
               audio_switch=True, i2v_aspect="adaptive",
               tasks={"reference": "alibaba/wan-3.0/reference-to-video", "first": "alibaba/wan-3.0/image-to-video"}),
            _v("spicyapi", "wan-3.0", "alibaba/wan-3.0-spicy", "Wan 3.0 Spicy", FRAMES, ("480p", "720p", "1080p"), _span(2, 30),
               usd_per_second=0.057, hint="$0.057/s, 15 s clip $0.86", audio_switch=True, i2v_aspect="frame",
               tasks={"first": "alibaba/wan-3.0-spicy/image-to-video"}),
        ],
        "seedance-2": [
            _v("spicyapi", "seedance-2", "bytedance/seedance-2.0-mini", "Seedance 2 Mini", ALL, ("480p", "720p"), _span(4, 15), max_refs=9,
               ref_field="reference_image_urls", ref_token="@Image{n}", aspects=ALL_SHAPES, usd_per_second=0.0246, hint="$0.025/s, 15 s clip $0.37",
               audio_switch=True,
               tasks={"reference": "bytedance/seedance-2.0-mini/reference-to-video", "first": "bytedance/seedance-2.0-mini/image-to-video"}),
            _v("spicyapi", "seedance-2", "bytedance/seedance-2.0-mini-spicy", "Seedance 2 Mini Spicy", FRAMES, ("480p", "720p"), _span(4, 15),
               usd_per_second=0.0387, hint="$0.039/s, 15 s clip $0.58", audio_switch=True,
               tasks={"first": "bytedance/seedance-2.0-mini-spicy/image-to-video"}),
            _v("spicyapi", "seedance-2", "bytedance/seedance-2.0-fast", "Seedance 2 Fast", ALL, ("480p", "720p"), _span(4, 15), max_refs=9,
               ref_field="reference_image_urls", ref_token="@Image{n}", aspects=ALL_SHAPES, usd_per_second=0.045, hint="$0.045/s, 15 s clip $0.67",
               audio_switch=True,
               tasks={"reference": "bytedance/seedance-2.0-fast/reference-to-video", "first": "bytedance/seedance-2.0-fast/image-to-video"}),
            _v("spicyapi", "seedance-2", "bytedance/seedance-2.0-fast-spicy", "Seedance 2 Fast Spicy", FRAMES, ("480p", "720p"), _span(4, 15),
               usd_per_second=0.081, hint="$0.081/s, 15 s clip $1.22", audio_switch=True,
               tasks={"first": "bytedance/seedance-2.0-fast-spicy/image-to-video"}),
            _v("spicyapi", "seedance-2", "bytedance/seedance-2.0", "Seedance 2", ALL, ("480p", "720p", "1080p", "4k"), _span(4, 15), max_refs=9,
               ref_field="reference_image_urls", ref_token="@Image{n}", aspects=ALL_SHAPES, usd_per_second=0.07, hint="$0.07/s, 15 s clip $1.05",
               audio_switch=True,
               tasks={"reference": "bytedance/seedance-2.0/reference-to-video", "first": "bytedance/seedance-2.0/image-to-video"}),
            _v("spicyapi", "seedance-2", "bytedance/seedance-2.0-spicy", "Seedance 2 Spicy", FRAMES, ("480p", "720p", "1080p", "4k"), _span(4, 15),
               max_prompt=5000, usd_per_second=0.114, hint="$0.114/s, 15 s clip $1.71", audio_switch=True,
               tasks={"first": "bytedance/seedance-2.0-spicy/image-to-video"}),
        ],
        "seedance-2.5": [
            _v("spicyapi", "seedance-2.5", "bytedance/seedance-2.5", "Seedance 2.5 Uncensored", ALL, ("480p", "720p", "1080p"), _span(4, 30), max_refs=30,
               ref_field="reference_image_urls", ref_token="@Image{n}", aspects=ALL_SHAPES, usd_per_second=0.1234,
               hint="$0.123/s, 15 s clip $1.85, 30 s $3.70", audio_switch=True,
               tasks={"reference": "bytedance/seedance-2.5/reference-to-video", "first": "bytedance/seedance-2.5/image-to-video"}),
            _v("spicyapi", "seedance-2.5", "bytedance/seedance-2.5-spicy", "Seedance 2.5 Spicy", FRAMES, ("480p", "720p", "1080p", "4k"), _span(4, 30),
               max_prompt=5000, usd_per_second=0.216, hint="$0.216/s, 15 s clip $3.24", audio_switch=True,
               tasks={"first": "bytedance/seedance-2.5-spicy/image-to-video"}),
        ],
        "minimax-h3": [
            _v("spicyapi", "minimax-h3", "minimax/h3", "MiniMax H3", ALL, ("480p", "768p", "2k"), _span(4, 15), max_refs=9,
               ref_field="reference_image_urls", ref_token=H3_TOKEN, aspects=ALL_SHAPES, max_prompt=7000, usd_per_second=0.025,
               hint="$0.025/s, 15 s clip $0.38", no_seed=True,
               tasks={"reference": "minimax/h3/reference-to-video", "first": "minimax/h3/image-to-video"}),
            _v("spicyapi", "minimax-h3", "minimax/h3-x-turbo", "MiniMax H3 X Turbo", ALL, ("480p", "720p", "1080p"), _span(2, 15), max_refs=9,
               ref_field="reference_image_urls", ref_token="Image {n}", aspects=ALL_SHAPES, max_prompt=7000, usd_per_second=0.015,
               hint="$0.015/s, 15 s clip $0.23", no_seed=True,
               tasks={"reference": "minimax/h3-x-turbo/reference-to-video", "first": "minimax/h3-x-turbo/image-to-video"}),
            _v("spicyapi", "minimax-h3", "minimax/h3-spicy", "MiniMax H3 Spicy", FRAMES, ("480p", "540p", "768p", "1080p"), _span(3, 15),
               usd_per_second=0.038, hint="$0.038/s, 15 s clip $0.57", tasks={"first": "minimax/h3-spicy/image-to-video"}),
        ],
        "ltx-2.3": [_v("spicyapi", "ltx-2.3", "lightricks/ltx-2.3-spicy", "LTX-2.3 Spicy", FIRST, ("480p", "720p", "1080p"), _span(3, 20),
                       usd_per_second=0.019, hint="$0.019/s", tasks={"first": "lightricks/ltx-2.3-spicy/image-to-video"})],
        "ltx-2.5": [_v("spicyapi", "ltx-2.5", "lightricks/ltx-2.5", "LTX-2.5", FRAMES, ("720p", "1080p", "2k", "4k"), _span(5, 20),
                       usd_per_second=0.09, hint="$0.09/s at 720p, its lowest", tasks={"first": "lightricks/ltx-2.5/image-to-video"})],
        "wan-2.2": [_v("spicyapi", "wan-2.2", "alibaba/wan-2.2-spicy", "Wan 2.2 Spicy", FRAMES, ("480p", "720p"), (5, 8),
                       usd_per_second=0.019, hint="$0.019/s, 8 s longest", tasks={"first": "alibaba/wan-2.2-spicy/image-to-video"})],
    },
    # Venice: Wan, MiniMax H3 and Wan 2.2 are tagged uncensored. Seedance and LTX aren't, and Seedance
    # has a face-consent rule, so those wait on a test. End frames aren't documented, so first frame only.
    "venice": {
        "wan-3.0": [_v("venice", "wan-3.0", "wan-3-0", "Wan 3.0 Uncensored", ("reference", "first"), ("480p", "720p", "1080p"), (2, 5, 10, 15, 20, 25, 30),
                       max_refs=10, ref_field="reference_image_urls", ref_token="@Image {n}", aspects=ALL_SHAPES, usd_per_second=0.0653,
                       hint="$0.065/s, 15 s clip $0.98", audio_switch=True, i2v_aspect="adaptive",
                       tasks={"reference": "wan-3-0-reference-to-video", "first": "wan-3-0-image-to-video"})],
        "minimax-h3": [
            _v("venice", "minimax-h3", "minimax-h3-max", "MiniMax H3 Max", ("reference", "first"), ("480P", "768P", "1080P"), _span(5, 15), max_refs=9,
               ref_field="reference_image_urls", ref_token=H3_TOKEN, aspects=ALL_SHAPES, max_prompt=10000, usd_per_second=0.06,
               hint="15 s clip $0.90 from references, $0.63 from a first frame",
               tasks={"reference": "minimax-h3-max-reference-to-video", "first": "minimax-h3-max-image-to-video"}),
            _v("venice", "minimax-h3", "minimax-h3-max-turbo", "MiniMax H3 Max Turbo", FIRST, ("480P", "768P", "1080P"), _span(5, 15),
               max_prompt=10000, usd_per_second=0.0207, hint="$0.021/s, 15 s clip $0.31", tasks={"first": "minimax-h3-max-turbo-image-to-video"}),
            _v("venice", "minimax-h3", "minimax-h3", "MiniMax H3", ("reference", "first"), ("768P", "2K"), _span(5, 15), max_refs=9,
               ref_field="reference_image_urls", ref_token=H3_TOKEN, aspects=ALL_SHAPES, max_prompt=7000, usd_per_second=0.10,
               hint="$0.10/s, 15 s clip $1.50", tasks={"reference": "minimax-h3-reference-to-video", "first": "minimax-h3-image-to-video"}),
        ],
        "seedance-2": [
            _v("venice", "seedance-2", "seedance-2-0-mini", "Seedance 2 Mini", ("reference", "first"), ("480p", "720p"), _span(4, 15), max_refs=9,
               ref_field="reference_image_urls", ref_token=SEEDANCE_TOKEN, aspects=ALL_SHAPES, usd_per_second=0.044, nsfw="check",
               hint="$0.044/s, 15 s clip $0.66. Not tagged uncensored", audio_switch=True,
               tasks={"reference": "seedance-2-0-mini-reference-to-video-basic", "first": "seedance-2-0-mini-image-to-video-basic"}),
            _v("venice", "seedance-2", "seedance-2-0-fast", "Seedance 2 Fast", ("reference", "first"), ("480p", "720p"), _span(4, 15), max_refs=9,
               ref_field="reference_image_urls", ref_token=SEEDANCE_TOKEN, aspects=ALL_SHAPES, usd_per_second=0.07, nsfw="check",
               hint="$0.07/s, 15 s clip $1.05. Not tagged uncensored", audio_switch=True,
               tasks={"reference": "seedance-2-0-fast-reference-to-video-basic", "first": "seedance-2-0-fast-image-to-video-basic"}),
            _v("venice", "seedance-2", "seedance-2-0", "Seedance 2", ("reference", "first"), ("480p", "720p", "1080p", "4k"), _span(4, 15), max_refs=9,
               ref_field="reference_image_urls", ref_token=SEEDANCE_TOKEN, aspects=ALL_SHAPES, usd_per_second=0.088, nsfw="check",
               hint="$0.088/s, 15 s clip $1.32. Not tagged uncensored", audio_switch=True,
               tasks={"reference": "seedance-2-0-reference-to-video-basic", "first": "seedance-2-0-image-to-video-basic"}),
        ],
        "seedance-2.5": [_v("venice", "seedance-2.5", "seedance-2-5", "Seedance 2.5", ("reference", "first"), ("480p", "720p", "1080p", "4k"), _span(4, 30),
                            max_refs=30, ref_field="reference_image_urls", ref_token="@Image{n}", aspects=ALL_SHAPES, usd_per_second=0.1287, nsfw="check",
                            hint="$0.129/s, 15 s clip $1.93. Not tagged uncensored", audio_switch=True,
                            tasks={"reference": "seedance-2-5-reference-to-video-basic", "first": "seedance-2-5-image-to-video-basic"})],
        "ltx-2.3": [_v("venice", "ltx-2.3", "ltx-2-v2-3-fast", "LTX-2.3 Fast", FIRST, ("1080p", "1440p", "2160p"), (6, 8, 10, 12, 14, 16, 18, 20),
                       usd_per_second=0.066, nsfw="check", hint="$0.066/s at 1080p, its lowest. Not tagged uncensored", audio_switch=True,
                       i2v_aspect="frame", tasks={"first": "ltx-2-v2-3-fast-image-to-video"})],
        "ltx-2.5": [_v("venice", "ltx-2.5", "ltx-2-5-fast", "LTX-2.5 Fast", FIRST, ("720p", "1080p", "1440p", "2160p"), (6, 8, 10, 12, 14, 16, 18, 20),
                       usd_per_second=0.099, nsfw="check", hint="$0.099/s at 720p. Not tagged uncensored", audio_switch=True,
                       i2v_aspect="auto", tasks={"first": "ltx-2-5-fast-image-to-video"})],
        "wan-2.2": [_v("venice", "wan-2.2", "wan-2-2-enhanced", "Wan 2.2 Enhanced", FIRST, ("480p", "720p"), (5, 8), usd_per_second=0.04,
                       hint="$0.04/s, 8 s clip $0.32", tasks={"first": "wan-2-2-enhanced-image-to-video"})],
    },
}

MOCK_IMAGES = {family: [_i("mock", family, "mock-image", "Mock image (no API call)", REF, max_refs=10, lowest="768x1344",
                           aspects=ALL_SHAPES, sizes=SEEDREAM_45_SIZES, hint="free, instant")]
               for family in ("seedream-4.5", "best")}
MOCK_IMAGES["flux-lora"] = [_i("mock", "flux-lora", "mock-flux", "Mock Flux (no API call)", ("lora",), lora_arch="flux", lowest="768x1344",
                               sizes=FLUX_SIZES, hint="free, instant")]
MOCK_IMAGES["seedream-5-pro"] = [_i("mock", "seedream-5-pro", "mock-image-5", "Mock scene (no API call)", REF, max_refs=10, lowest="768x1344",
                                    aspects=ALL_SHAPES, sizes=SEEDREAM_45_SIZES, hidden=True)]
MOCK_VIDEOS = {"minimax-h3": [_v("mock", "minimax-h3", "mock-video", "Mock clip (no API call)", ALL, ("480p",), (2, 5, 15), max_refs=9,
                                 ref_token=H3_TOKEN, aspects=("9:16", "16:9"), hint="free, a bundled test clip")]}


def tables(kind: str) -> Dict[str, Dict[str, List[ModelOption]]]:
    from .config import MediaConfig

    table = dict(IMAGE_CATALOG if kind == "image" else VIDEO_CATALOG)
    if MediaConfig.VIDEO_MOCK_PROVIDER:
        table["mock"] = MOCK_IMAGES if kind == "image" else MOCK_VIDEOS
    return table


def options(kind: str) -> List[ModelOption]:
    return [o for families in tables(kind).values() for variants in families.values() for o in variants]


def find(kind: str, provider: str, family: str, variant: str = "") -> Optional[ModelOption]:
    variants = tables(kind).get(provider, {}).get(family) or []
    if not variant:
        return variants[0] if variants else None
    return next((o for o in variants if o.id == variant), None)


def by_id(kind: str, provider: str, model_id: str) -> Optional[ModelOption]:
    """The option a stored job ran on."""
    return next((o for o in options(kind) if o.provider == provider and o.id == model_id), None)


def route_status(option: ModelOption) -> Route:
    """Your test result wins over the catalog default."""
    return Route(db.route_override(option.provider, option.id) or option.nsfw)


def require_route(option: ModelOption) -> ModelOption:
    status = route_status(option)
    if status is not Route.ALLOWED:
        raise RouteNotAllowed(f"{option.label} on {PROVIDER_LABELS.get(option.provider, option.provider)} is {status.value}")
    return option


def _pick(kind: str, provider: str, family: str, variant: str) -> ModelOption:
    option = find(kind, provider, family, variant)
    if option:
        return require_route(option)
    labels = IMAGE_FAMILIES if kind == "image" else VIDEO_FAMILIES
    who = PROVIDER_LABELS.get(provider, provider)
    raise ValueError(f"{who} has no {labels.get(family, family)} model" + (f" '{variant}'" if variant else ""))


def image(provider: str, family: str, variant: str = "") -> ModelOption:
    """A blocked route can't be reached here, so a keyframe or fallback never gets one."""
    return _pick("image", provider, family, variant)


def video(provider: str, family: str, variant: str = "") -> ModelOption:
    return _pick("video", provider, family, variant)
