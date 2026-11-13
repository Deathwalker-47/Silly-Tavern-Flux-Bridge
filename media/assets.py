"""Reference images and frames: the size window, clean storage and provider fitting.

Uploads are rotated upright, stripped of EXIF (GPS included) and stored as JPEG, so a
phone photo reaches no provider with its location attached.
"""

from __future__ import annotations

import base64
import hashlib
import os
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageOps

from . import db
from .config import MediaConfig

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    register_heif_opener = None

MIN_BYTES = 20 * 1024
MAX_BYTES = 10 * 1024 * 1024
MIN_SIDE = 512
MAX_SIDE = 8192
MAX_PIXELS = 50_000_000
MIN_SHAPE = 2 / 5                # shortest side divided by longest side
FORMATS = {"JPEG": "JPEG", "PNG": "PNG", "WEBP": "WebP", "HEIF": "HEIC"}
STORED_EDGE = 2048
STORED_QUALITY = 90
THUMB_EDGE, THUMB_QUALITY = 256, 78
LABEL_CHARS = 80

WINDOW = {
    "min_bytes": MIN_BYTES, "max_bytes": MAX_BYTES, "min_side": MIN_SIDE, "max_side": MAX_SIDE,
    "max_pixels": MAX_PIXELS, "min_shape": MIN_SHAPE, "formats": list(FORMATS.values()),
}


class AssetRejected(ValueError):
    """The upload is outside the reference window; the message names the problem."""


@dataclass
class Asset:
    id: str
    path: Path
    mime: str = "image/jpeg"
    bytes: int = 0

    def read(self) -> bytes:
        return self.path.read_bytes()


def asset_dir() -> Path:
    path = MediaConfig.DATA_DIR / "assets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def asset_id(data: bytes) -> str:
    """Hash of the uploaded bytes, so the browser can check before sending the file."""
    return hashlib.sha256(data).hexdigest()[:32]


def _open(data: bytes) -> Image.Image:
    try:
        return Image.open(BytesIO(data))
    except Exception:
        raise AssetRejected("This file isn't a readable image.")


def _size_text(count: int) -> str:
    return f"{count / (1024 * 1024):.1f} MB" if count >= 1024 * 1024 else f"{count // 1024} KB"


def check_window(data: bytes) -> Tuple[int, int]:
    """Width and height if the image is inside the window; otherwise says exactly why not."""
    if len(data) < MIN_BYTES:
        raise AssetRejected(f"Image is {_size_text(len(data))}. The minimum is 20 KB.")
    if len(data) > MAX_BYTES:
        raise AssetRejected(f"Image is {_size_text(len(data))}. The maximum is 10 MB.")
    image = _open(data)
    if image.format not in FORMATS:
        raise AssetRejected(f"{image.format or 'This format'} isn't accepted. Use JPEG, PNG, WebP or HEIC.")
    width, height = image.size
    short, long = min(width, height), max(width, height)
    if short < MIN_SIDE:
        raise AssetRejected(f"Image is {width}×{height}. The shortest side must be at least {MIN_SIDE} px.")
    if long > MAX_SIDE:
        raise AssetRejected(f"Image is {width}×{height}. The longest side can be at most {MAX_SIDE} px.")
    if width * height > MAX_PIXELS:
        raise AssetRejected(f"Image is {width * height / 1_000_000:.0f} megapixels. The maximum is 50.")
    if short / long < MIN_SHAPE:
        raise AssetRejected(f"Image is {width}×{height}, which is too narrow. The shape must be between 2:5 and 5:2.")
    return width, height


def clean_jpeg(data: bytes, edge: int = STORED_EDGE, quality: int = STORED_QUALITY) -> bytes:
    """Upright JPEG with no metadata at all: it's copied onto a new image, so nothing can follow."""
    with _open(data) as source:
        upright = ImageOps.exif_transpose(source).convert("RGBA")
    flat = Image.new("RGB", upright.size, (255, 255, 255))
    flat.paste(upright, mask=upright.getchannel("A"))
    flat.thumbnail((edge, edge), Image.LANCZOS)
    buf = BytesIO()
    flat.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def _write(path: Path, data: bytes) -> None:
    temp = path.with_suffix(f".tmp{os.getpid()}")
    temp.write_bytes(data)
    temp.replace(path)


def store(data: bytes, purpose: str = "reference", label: str = "") -> Tuple["Asset", bool]:
    """Saves an image once. Returns the asset and whether it already existed.

    References must fit the window, and land in the library under their label. Frames
    (keyframes, message images) skip both, because fit_refs handles them before any
    provider sees them.
    """
    if purpose == "reference":
        check_window(data)
    label = " ".join(str(label or "").split())[:LABEL_CHARS] if purpose == "reference" else ""
    key = asset_id(data)
    row = db.asset(key)
    if row and Path(row["path"]).exists():
        if purpose == "reference":
            db.reuse_asset(key, label)
        return Asset(key, Path(row["path"]), row["mime"], row["bytes"]), True
    cleaned = clean_jpeg(data)
    path = asset_dir() / f"{key}.jpg"
    _write(path, cleaned)
    db.put_asset(key, str(path), "image/jpeg", len(cleaned), purpose, label)
    return Asset(key, path, "image/jpeg", len(cleaned)), False


def _thumb_path(key: str) -> Path:
    return asset_dir() / "thumbs" / f"{key}.jpg"


def thumbnail(path: Path, key: str) -> bytes:
    """A small JPEG for the library grid, made once and kept next to the image."""
    thumb = _thumb_path(key)
    if thumb.exists():
        return thumb.read_bytes()
    data = clean_jpeg(path.read_bytes(), THUMB_EDGE, THUMB_QUALITY)
    thumb.parent.mkdir(parents=True, exist_ok=True)
    _write(thumb, data)
    return data


def library(limit: int = 48, offset: int = 0) -> Dict[str, Any]:
    """One page of the shared image library: every reference uploaded for anyone, newest first,
    each with a thumbnail inline so the grid needs one request and no open file route."""
    rows = db.library(limit + 1, offset)
    items = []
    for row in rows[:limit]:
        path = Path(row["path"])
        if not path.exists():
            continue
        try:
            with Image.open(path) as image:
                width, height = image.size
            thumb = thumbnail(path, row["id"])
        except Exception:
            continue
        items.append({"id": row["id"], "label": row.get("label") or "", "created_at": row["created_at"],
                      "width": width, "height": height,
                      "thumb": "data:image/jpeg;base64," + base64.b64encode(thumb).decode("ascii")})
    return {"assets": items, "more": len(rows) > limit}


def hide(key: str) -> bool:
    """Takes an image out of the library. The file stays, so any slot still using it keeps working."""
    if not db.asset(key):
        return False
    db.hide_asset(key)
    return True


SHAPES = ("9:16", "3:4", "1:1", "4:3", "16:9")


def dimensions(key: str) -> Optional[Tuple[int, int]]:
    """Width and height of a stored image, read from its header. None when it isn't on the bridge."""
    row = db.asset(key)
    if not row or not Path(row["path"]).exists():
        return None
    try:
        with Image.open(row["path"]) as image:
            return image.size
    except Exception:
        return None


def closest_shape(width: int, height: int, options: Sequence[str] = SHAPES) -> str:
    """The named shape nearest to these dimensions. Names only: no image is ever cropped or padded to match it."""
    target = width / max(1, height)
    return min(options, key=lambda o: abs(int(o.split(":")[0]) / int(o.split(":")[1]) - target))


def shape_text(key: str) -> Optional[str]:
    size = dimensions(key)
    return f"{size[0]}x{size[1]}, closest {closest_shape(*size)}" if size else None


def exists(key: str) -> bool:
    row = db.asset(key)
    return bool(row and Path(row["path"]).exists())


def load(key: str) -> Asset:
    row = db.asset(key)
    if not row or not Path(row["path"]).exists():
        raise KeyError(f"Reference {key} is no longer on the bridge. Upload it again.")
    return Asset(key, Path(row["path"]), row["mime"], row["bytes"])


def to_jpeg(asset: Asset, edge: int, quality: int) -> bytes:
    return clean_jpeg(asset.read(), edge, quality)


def fit_refs(assets: Sequence[Asset], max_total: int) -> List[bytes]:
    """JPEGs small enough for the provider's request limit. NanoGPT uses 4_400_000."""
    for edge, quality in ((1536, 85), (1280, 82), (1024, 80), (768, 78)):
        images = [to_jpeg(a, edge, quality) for a in assets]
        if sum(len(i) * 4 // 3 for i in images) <= max_total:
            return images
    raise RuntimeError("References are too large for this provider")


def sweep_frames(before: Optional[int] = None) -> int:
    """Keyframes and message images go after the unsaved-file window. References stay."""
    cutoff = before if before is not None else int(time.time()) - MediaConfig.MEDIA_UNSAVED_TTL_DAYS * 86400
    count = 0
    for row in db.stale_frames(cutoff):
        Path(row["path"]).unlink(missing_ok=True)
        _thumb_path(row["id"]).unlink(missing_ok=True)
        db.delete_asset(row["id"])
        count += 1
    return count
