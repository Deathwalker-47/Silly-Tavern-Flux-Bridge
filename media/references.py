"""Who gets which reference, and how the prompt names them.

The director writes {{ref:Maya:face_1}} and never a number. Code picks the images, then
swaps each placeholder for the model's own numbered token, so the numbers always match
the order the images are sent in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_ORDER = ["face_1", "full_body", "naked", "face_2", "portrait"]
PLACEHOLDER = re.compile(r"\{\{\s*ref\s*:\s*([^:{}]+?)\s*:\s*([^{}]+?)\s*\}\}", re.I)


@dataclass
class Person:
    key: str
    name: str
    age: Optional[int] = None
    appearance: str = ""
    refs: Dict[str, str] = field(default_factory=dict)   # slot -> asset id
    sd_prompt: str = ""
    role: str = "other"

    @classmethod
    def from_context(cls, raw: Dict[str, Any], role: str = "other") -> "Person":
        age = raw.get("age")
        return cls(key=str(raw.get("key") or raw.get("name") or ""), name=str(raw.get("name") or ""),
                   age=int(age) if str(age or "").strip().isdigit() else None,
                   appearance=str(raw.get("appearance") or ""),
                   refs={k: v for k, v in (raw.get("refs") or {}).items() if v},
                   sd_prompt=str(raw.get("sd_prompt") or ""), role=role)


@dataclass
class Ref:
    person: Person
    slot: str
    asset: str

    def public(self) -> Dict[str, str]:
        return {"asset": self.asset, "person": self.person.key, "slot": self.slot}


def pick_references(people: Sequence[Person], order: Sequence[str], max_refs: int) -> List[Ref]:
    """Everyone gets a face before anyone gets a second image. Main character first."""
    order = [s for s in order if s in DEFAULT_ORDER] or list(DEFAULT_ORDER)
    order += [s for s in DEFAULT_ORDER if s not in order]
    queues = [[Ref(p, s, p.refs[s]) for s in order if p.refs.get(s)] for p in people]
    chosen: List[Ref] = []
    while len(chosen) < max_refs and any(queues):
        for queue in queues:
            if queue and len(chosen) < max_refs:
                chosen.append(queue.pop(0))
    return chosen


def render_tokens(text: str, option: Any, refs: Sequence[Ref], offset: int = 0) -> str:
    """Placeholders become this model's token. A slot that didn't fit uses that person's first
    reference; someone with no reference at all is simply named."""
    def token(index: int) -> str:
        return option.ref_token.format(n=index + 1 + offset)

    def swap(match: "re.Match[str]") -> str:
        name, slot = match.group(1).lower(), match.group(2).lower()
        hits = [i for i, r in enumerate(refs) if r.person.name.lower() == name]
        exact = [i for i in hits if refs[i].slot == slot]
        if exact or hits:
            return token((exact or hits)[0])
        return match.group(1)
    return PLACEHOLDER.sub(swap, text or "")
