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
    "best": "The Best Option",
}

VIDEO_FAMILIES = {
    "wan-3.0": "Wan 3.0",
    "seedance-2": "Seedance 2",
    "seedance-2.5": "Seedance 2.5",
    "seedance-1.5": "Seedance 1.5",
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
    "mock": "Mock (local test files)",
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
        "qwen-edit": [_i("nanogpt", "qwen-edit", "qwen-edit-uncensored", "Qwen Edit Uncensored", REF, max_refs=3, ref_field="input_references",
                         max_prompt=1500, aspects=("9:16", "3:4", "16:9"), lowest="1k", usd=0.04, hint="$0.04, 3 refs",
                         resolution="1k", send_aspect=True, needs_ref=True)],
        "seedream-4.5": [_i("nanogpt", "seedream-4.5", "seedream-v4.5", "Seedream 4.5", REF, max_refs=10, ref_field="input_references",
                            aspects=ALL_SHAPES, lowest="2304x4096", usd=0.04, hint="$0.04, 10 refs", sizes=SEEDREAM_45_SIZES)],
        "seedream-5-pro": [_i("nanogpt", "seedream-5-pro", "bytedance/seedream-v5.0-pro/edit", "Seedream 5 Pro", REF, max_refs=10,
                              ref_field="input_references", aspects=ALL_SHAPES, lowest="1k", usd=0.045, hint="$0.045, 10 refs",
                              hidden=True, resolution="1k", send_aspect=True, needs_ref=True)],
        "best": [_i("nanogpt", "best", "best", "The Best Option", REF, max_refs=10, aspects=ALL_SHAPES, lowest="1k",
                    usd=0.085, hint="$0.085: Seedream 5 Pro scene, then Seedream 4.5 face pass")],
    },
    "sogni": {
        "zimage": [_i("sogni", "zimage", "z_image_bf16", "Z-Image", ("text",), lowest="768x1344", hint="Spark, no LoRA", sizes=FLUX_SIZES)],
        "qwen-edit": [_i("sogni", "qwen-edit", "qwen_image_edit_2511_fp8_lightning", "Qwen Edit 2511", REF, max_refs=3, ref_field="contextImages",
                         lowest="768x1344", hint="Spark, 3 refs", sizes=FLUX_SIZES, needs_ref=True)],
        "seedream-5-pro": [_i("sogni", "seedream-5-pro", "seedream-5-0-pro-uncensored", "Seedream 5 Pro Uncensored", REF, max_refs=10,
                              ref_field="contextImages", lowest="768x1344", hint="Spark, 10 refs", hidden=True, sizes=FLUX_SIZES)],
    },
    "pixeldojo": {
        "seedream-4.5": [_i("pixeldojo", "seedream-4.5", "seedream-4", "Seedream 4.5", REF, max_refs=14, ref_field="image_urls",
                            aspects=ALL_SHAPES, lowest="portrait_9_16", hint="1 cr, 14 refs", build="seedream_4")],
        "seedream-5-pro": [_i("pixeldojo", "seedream-5-pro", "seedream-5", "Seedream 5", REF, max_refs=10, ref_field="image_urls",
                              lowest="768x1344", hint="1.25 cr, 10 refs", hidden=True, build="seedream_5", sizes=FLUX_SIZES)],
        "best": [_i("pixeldojo", "best", "best", "The Best Option", REF, max_refs=10, aspects=ALL_SHAPES, lowest="1k",
                    hint="2.25 cr: Seedream 5 scene, then Seedream 4.5 face pass")],
    },
}

VIDEO_CATALOG: Dict[str, Dict[str, List[ModelOption]]] = {
    "nanogpt": {
        "wan-3.0": [_v("nanogpt", "wan-3.0", "alibaba/wan-3.0/image-to-video-spicy", "Wan 3.0 Spicy", FIRST, ("480p", "720p", "1080p"), _span(2, 30),
                       usd_per_second=0.05, hint="$0.05/s, 15 s clip $0.75", params=("resolution", "duration", "seed"))],
        "seedance-2": [
            _v("nanogpt", "seedance-2", "bytedance/seedance-2.0-fast/image-to-video-spicy", "Seedance 2 Fast Spicy", FRAMES,
               ("480p", "720p", "1080p"), _span(4, 15), usd_per_second=0.10, hint="$0.10/s, 15 s clip $1.50", audio_switch=True,
               params=("resolution", "duration", "aspect_ratio", "generateAudio", "last_image", "seed")),
            _v("nanogpt", "seedance-2", "bytedance/seedance-2.0/image-to-video-spicy", "Seedance 2 Spicy", FRAMES,
               ("480p", "720p", "1080p"), _span(4, 15), usd_per_second=0.12, hint="$0.12/s, 15 s clip $1.80", audio_switch=True,
               params=("resolution", "duration", "aspect_ratio", "generateAudio", "last_image", "seed")),
            _v("nanogpt", "seedance-2", "bytedance-seedance-2-0-fast", "Seedance 2 Fast", ALL, ("720p", "1080p"), (5, 10, 15),
               max_refs=9, ref_field="reference_images", ref_token=SEEDANCE_TOKEN, aspects=ALL_SHAPES, usd_per_second=0.12,
               hint="$0.12/s, 15 s clip $1.80", params=("resolution", "duration", "aspect_ratio", "last_image")),
            _v("nanogpt", "seedance-2", "bytedance-seedance-2-0-mini", "Seedance 2 Mini", ALL, ("720p", "1080p"), _span(5, 12),
               max_refs=9, ref_field="reference_images", ref_token=SEEDANCE_TOKEN, aspects=ALL_SHAPES, usd_per_second=0.12,
               hint="$0.12/s, 12 s clip $1.44", audio_switch=True,
               params=("resolution", "duration", "aspect_ratio", "generateAudio", "last_image")),
        ],
        "seedance-2.5": [_v("nanogpt", "seedance-2.5", "bytedance/seedance-2.5-spicy", "Seedance 2.5 Spicy", FIRST,
                            ("480p", "720p", "1080p", "4K"), _span(4, 15), usd_per_second=0.18, hint="$0.18/s, 15 s clip $2.70",
                            params=("resolution", "duration"))],
        "seedance-1.5": [_v("nanogpt", "seedance-1.5", "bytedance-seedance-v1.5-pro", "Seedance 1.5 Pro", FIRST, ("480p", "720p", "1080p"),
                            _span(4, 12), usd_per_second=0.024, hint="$0.024/s, audio doubles it", audio_switch=True,
                            params=("resolution", "duration", "aspect_ratio", "generateAudio"))],
        "minimax-h3": [_v("nanogpt", "minimax-h3", "minimax-h3/image-to-video-spicy", "MiniMax H3 Spicy", FRAMES, ("480p", "768p"), _span(3, 15),
                          usd_per_second=0.04, hint="$0.04/s, 15 s clip $0.60", params=("resolution", "duration", "last_image", "seed"))],
        "ltx-2.3": [_v("nanogpt", "ltx-2.3", "ltx-2.3-spicy/image-to-video", "LTX-2.3 Spicy", FIRST, ("480p", "720p", "1080p"), _span(3, 20),
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
        "wan-3.0": [_v("sogni", "wan-3.0", "wan3.0-spicy-video", "Wan 3 Uncensored", ALL, ("480p", "720p", "1080p"), _span(2, 30), max_refs=10,
                       ref_field="referenceImageUrls", ref_token="@Image {n}", aspects=ALL_SHAPES, usd_per_second=0.08,
                       hint="$0.08/s, 15 s clip $1.20", url_refs=True, sends_ratio=True, fps=30, tall={"480p": (480, 854)})],
        "seedance-2": [_v("sogni", "seedance-2", "seedance-2-0-mini-uncensored", "Seedance 2 Mini Uncensored", ALL, ("480p", "720p"), _span(4, 15), max_refs=9,
                          ref_field="referenceImageUrls", ref_token=SEEDANCE_TOKEN, usd_per_second=0.084, hint="$0.084/s, 15 s clip $1.26",
                          url_refs=True, fps=24, tall={"480p": (480, 854)}, consent=True)],
        "seedance-2.5": [_v("sogni", "seedance-2.5", "seedance-2-5-uncensored", "Seedance 2.5 Uncensored", ALL, ("480p", "720p", "1080p"), _span(4, 30), max_refs=30,
                            ref_field="referenceImageUrls", ref_token="@image{n}", usd_per_second=0.154, hint="$0.154/s, 15 s clip $2.31",
                            url_refs=True, seedance_task=True, fps=24, tall={"480p": (480, 854)}, consent=True)],
        "ltx-2.3": [
            _v("sogni", "ltx-2.3", "ltx23-22b-10eros-v1.4-fp8mixed_i2v", "LTX-2.3 10Eros", FIRST, ("480p", "720p", "1080p"), _span(2, 20),
               hint="Spark", sizing="wh"),
            _v("sogni", "ltx-2.3", "ltx23-22b-fp8_i2v_distilled", "LTX-2.3 Distilled", FIRST, ("480p", "720p", "1080p"), _span(2, 20),
               hint="Spark, fast", sizing="wh"),
        ],
        "ltx-2.5": [_v("sogni", "ltx-2.5", "ltx25-22b-int8_i2v_distilled", "LTX-2.5", FIRST, ("480p", "720p", "1080p"), _span(2, 20),
                       hint="Spark", sizing="wh")],
        "wan-2.2": [
            _v("sogni", "wan-2.2", "wan_v2.2-14b-fp8_i2v_lightx2v", "Wan 2.2 LightX2V", FIRST, ("480p", "720p"), _span(2, 8), hint="Spark, fast", sizing="wh"),
            _v("sogni", "wan-2.2", "wan_v2.2-14b-fp8_i2v", "Wan 2.2", FIRST, ("480p", "720p"), _span(2, 8), hint="Spark, full quality", sizing="wh"),
        ],
    },
    "pixeldojo": {
        "minimax-h3": [
            _v("pixeldojo", "minimax-h3", "minimax-h3-turbo", "MiniMax H3 Turbo", ALL, ("480p", "720p"), _span(5, 15), max_refs=9,
               ref_field="reference_images", ref_token=H3_TOKEN, aspects=("9:16", "3:4", "16:9", "4:3"), hint="0.75 cr/s, 15 s clip 11.25 cr",
               tested=True, build="h3_turbo"),
            _v("pixeldojo", "minimax-h3", "minimax-h3", "MiniMax H3", ALL, ("768p", "2k"), _span(5, 15), max_refs=9, ref_field="reference_images",
               ref_token=H3_TOKEN, aspects=("9:16", "3:4", "16:9", "4:3"), hint="2.5 cr/s, 15 s clip 41.5 cr; first frame bills 4 cr/s", build="h3"),
        ],
        "wan-3.0": [_v("pixeldojo", "wan-3.0", "wan-3-0-video", "Wan 3.0", ALL, ("480P", "720P", "1080P", "2K", "4K"), _span(2, 30), max_refs=10,
                       ref_field="reference_image_urls", ref_token="@Image {n}", aspects=ALL_SHAPES, hint="1.5 cr/s at 480P", audio_switch=True,
                       build="wan_3")],
        "seedance-2.5": [_v("pixeldojo", "seedance-2.5", "seedance-2-5", "Seedance 2.5", ALL, ("480p", "720p", "1080p"), _span(4, 30), max_refs=30,
                            ref_field="reference_images", ref_token="@Image{n}", aspects=ALL_SHAPES, hint="3 cr/s, 15 s clip 45 cr", audio_switch=True,
                            build="seedance_25")],
        "seedance-2": [_v("pixeldojo", "seedance-2", "seedance-2-reference", "Seedance 2 Reference", REF, ("480p", "720p", "1080p", "4k"), _span(4, 15),
                          max_refs=9, ref_field="reference_images", ref_token=SEEDANCE_TOKEN, aspects=ALL_SHAPES, hint="fast tier", audio_switch=True,
                          build="seedance_2_reference")],
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
