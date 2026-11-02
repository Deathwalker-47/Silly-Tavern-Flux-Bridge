"""The LoRA dictionary as the media pipeline sees it.

flux_lora_bridge.py owns the dictionary and Runware's upload mapping; it hands both over
at start-up through `use`, so this package never imports the bridge.

Each LoRA is read for one architecture, so a model only ever gets LoRAs built for it,
and an entry that's empty for that architecture is skipped.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

MAX_CONCEPTS = 3

_manager: Any = None
_resolve: Optional[Callable[..., List[Dict[str, Any]]]] = None


def use(manager: Any, resolve_runware_loras: Callable[..., List[Dict[str, Any]]]) -> None:
    global _manager, _resolve
    _manager, _resolve = manager, resolve_runware_loras


def resolve_runware(items: List[Dict[str, Any]], key: str, arch: str) -> List[Dict[str, Any]]:
    return _resolve(items, key, arch)


def character_lora(name: str, sd_prompt: str, arch: str = "flux") -> Optional[Dict[str, Any]]:
    """The one character LoRA for this person, never several. Best rank wins."""
    if not _manager or not (f"{name}{sd_prompt}").strip():
        return None
    matched = _manager.match_loras_by_keywords(f"{name} {sd_prompt}", "", arch)
    for item in matched:
        if item.get("reason") != "permanent" and item["data"].get("category") == "character":
            return {"id": item["id"], **item["data"]}
    return None


def concepts(arch: str = "flux") -> List[Dict[str, Any]]:
    """Everything the director may add to a scene: every LoRA that isn't a character."""
    if not _manager or not arch:
        return []
    from lora_dict import flat_entry

    out = []
    permanent = set(_manager.get_permanent_loras())
    for key, entry in _manager.loradict.get("loras", {}).items():
        data = flat_entry(entry, arch)
        if data and key not in permanent and data.get("category") != "character":
            out.append({"key": key, "keywords": data.get("keywords") or [], "category": data.get("category", "")})
    return out


def lora_stack(concept_keys: List[str], arch: str, name: str, sd_prompt: str) -> List[Dict[str, Any]]:
    """One character LoRA, up to 3 concept LoRAs the director chose, plus permanent ones."""
    if not _manager or not arch:
        return []
    from lora_dict import flat_entry

    entries = _manager.loradict.get("loras", {})
    stack: List[Dict[str, Any]] = []
    character = character_lora(name, sd_prompt, arch)
    if character:
        stack.append(character)
    keys = [k for k in concept_keys if k in entries][:MAX_CONCEPTS] + list(_manager.get_permanent_loras())
    for key in keys:
        data = flat_entry(entries.get(key) or {}, arch)
        if data and not any(item["id"] == key for item in stack):
            stack.append({"id": key, **data})
    return stack


def compose(prompt: str, stack: List[Dict[str, Any]], negative: str = "") -> tuple:
    """The prompt with each LoRA's trigger added, and the negative prompt."""
    before = [item.get("prepend_prompt", "").strip() for item in stack]
    after = [item.get("append_prompt", "").strip() for item in stack]
    default = _manager.get_default_negative_prompt() if _manager else ""
    negatives = [default, negative] + [item.get("negative_prompt", "").strip() for item in stack]
    return " ".join(filter(None, before + [prompt] + after)), ", ".join(filter(None, negatives))
