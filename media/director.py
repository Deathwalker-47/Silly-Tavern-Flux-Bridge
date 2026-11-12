"""The director: reads the scene, decides who and what is in the shot, writes the prompt.

Runs on an uncensored model. It refuses one thing only, sexual content involving minors,
and that rule is enforced twice: by the model and then in code, so it doesn't depend on
an uncensored model behaving.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from . import assets, loras, telemetry
from .config import MediaConfig
from .helper import extract_json, helper
from .references import DEFAULT_ORDER, Person

GUIDES = Path(__file__).parent / "guides"
GUIDE_FOR = {
    "flux-lora": "flux-dev", "zimage": "zimage", "qwen-edit": "qwen-edit", "seedream-4.5": "seedream-4.5",
    "seedream-5-pro": "seedream-5", "best": "seedream-5", "wan-3.0": "wan-3.0", "wan-2.2": "wan-3.0",
    "seedance-2": "seedance-2","seedance-2.5": "seedance-2.5",
    "minimax-h3": "minimax-h3", "ltx-2.3": "ltx", "ltx-2.5": "ltx",
}

DIRECTOR_RULES = """You are the media director for a roleplay chat. You turn the latest moment into one image or one video.
Reply with one JSON object only.

1. Read the whole reply and the history. Describe what is visible now, not what was said about it.
2. Pick the main character. Include only people who are physically present in this moment, and every person with an entry who is.
3. If "force_pov" is true, describe only the main character, set "pov" to true, and put only the user's hands, arms, legs or similar in "user_visible". No other person may appear.
4. Otherwise choose POV or third person from the scene. Include at most "people_cap" people, main character first.
5. Use each person's Appearance text. It wins over anything else, including the chat.
6. Order the reference slots for this scene in "ref_order": face_1, face_2, full_body, portrait, naked. Never drop one. Code fits as many as the model allows.
7. Choose "aspect_ratio" from the model's allowed list, and let the scene decide it: tall subjects and phone-style framing take 9:16, wide scenes, landscapes and groups 16:9. The references never limit this choice. They are sent exactly as they are and are never cropped, padded or stretched, whatever shape the picture has. Each person's "reference_shapes" gives their real shapes; use it only to break a tie between shapes that suit the scene equally. If a shape isn't allowed use the nearest allowed one, portrait before landscape. Leave it empty when the list says the first frame sets the shape. Give the reason in "aspect_reason".
8. Write in the target model's format below. Refer to a reference image only as {{ref:Name:slot}}, for example {{ref:Maya:face_1}}, where Name is the person's "name" exactly as given, even when the story calls them something else. Never write an image number.
9. Set "blocked" to true only if the scene sexualizes anyone under 18 or anyone described as a child, and say why in "block_reason". Never soften, refuse or reword anything else.
10. Set "adult" to true when the scene shows nudity or sex. It only chooses LoRAs and never changes the prompt.
11. "concept_loras": up to 3 keys from the list that truly fit what is visible. Leave it empty if none fit.
12. "caption": a short line the character might text along with it, or empty.
13. If "user_request" is present, the user asked for exactly that. Follow it.
14. If "call" is present, the user is on a phone call with the person named in "call.with", and they are in different places. That person is the main character, even on a card that names several people, and they are the one taking the photo or video: a selfie, a mirror shot, or what they point the camera at. The user is not in the shot and "pov" is false. Describe that person from their Appearance text when they have an entry in "characters", otherwise from the story.
15. An entry with "side_of" is a side character inside that story card, with their own Appearance and references. The story card's own entry stays the default: use it for any shot that doesn't feature one of its side characters. When a side character is physically present in this moment, or is the one on the call, include them by their own key. Every reference image of each person you include is sent to the model, and nobody else's, so include the story card's own entry alongside them only when the story shows its own character in this moment, not just because the scene is set in their place or their world. Never describe a side character with the story card's Appearance, or the story card's character with a side character's."""

PLAN_FIELDS = {
    "blocked": {"type": "boolean"}, "block_reason": {"type": "string"}, "main_character": {"type": "string"},
    "people": {"type": "array", "items": {"type": "object", "properties": {"key": {"type": "string"}, "role": {"type": "string"}},
                                          "required": ["key", "role"], "additionalProperties": False}},
    "pov": {"type": "boolean"}, "adult": {"type": "boolean"}, "user_visible": {"type": "string"},
    "aspect_ratio": {"type": "string"}, "aspect_reason": {"type": "string"},
    "ref_order": {"type": "array", "items": {"type": "string"}}, "concept_loras": {"type": "array", "items": {"type": "string"}},
    "prompt": {"type": "string"}, "negative_prompt": {"type": "string"}, "last_frame_prompt": {"type": "string"},
    "motion_prompt": {"type": "string"}, "caption": {"type": "string"},
}
PLAN_SCHEMA = {"type": "json_schema", "json_schema": {"name": "plan", "strict": True, "schema": {
    "type": "object", "properties": PLAN_FIELDS, "required": list(PLAN_FIELDS), "additionalProperties": False}}}


@dataclass
class Plan:
    prompt: str = ""
    blocked: bool = False
    block_reason: str = ""
    main_character: str = ""
    people: List[Dict[str, str]] = field(default_factory=list)
    pov: bool = False
    adult: bool = False
    user_visible: str = ""
    aspect_ratio: str = ""
    aspect_reason: str = ""
    ref_order: List[str] = field(default_factory=lambda: list(DEFAULT_ORDER))
    concept_loras: List[str] = field(default_factory=list)
    negative_prompt: str = ""
    last_frame_prompt: str = ""
    motion_prompt: str = ""
    caption: str = ""

    @classmethod
    def parse(cls, text: str) -> "Plan":
        data = extract_json(text)
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and v is not None}
        plan = cls(**known)
        if not isinstance(plan.people, list) or not isinstance(plan.ref_order, list):
            raise ValueError("people and ref_order must be lists")
        plan.people = [p if isinstance(p, dict) else {"key": str(p), "role": "other"} for p in plan.people]
        plan.ref_order = plan.ref_order or list(DEFAULT_ORDER)
        if not plan.blocked and not (plan.prompt.strip() or plan.motion_prompt.strip()):
            raise ValueError("the plan has no prompt")
        return plan

    def public(self) -> Dict[str, Any]:
        return asdict(self)


class DirectorInput:
    """The context the extension sent (never cut), plus what only the bridge knows."""

    def __init__(self, raw: Optional[Dict[str, Any]] = None):
        self.raw = dict(raw or {})
        user = self.raw.get("user") or {}
        self.user = Person.from_context(user, "user") if user else None
        self.characters = [Person.from_context(c, "character") for c in self.raw.get("characters") or []]
        self.force_pov = False

    @property
    def everyone(self) -> List[Person]:
        return self.characters + ([self.user] if self.user else [])

    def people_by_key(self, keys: Iterable[str]) -> List[Person]:
        by_key = {p.key: p for p in self.everyone}
        by_name = {p.name.lower(): p for p in self.everyone}
        chosen: List[Person] = []
        for key in keys:
            person = by_key.get(key) or by_name.get(str(key).lower())
            if person and person not in chosen:
                chosen.append(person)
        return chosen

    def public(self, option: Any) -> Dict[str, Any]:
        out = {k: v for k, v in self.raw.items() if k not in ("user", "characters")}
        arch = option.lora_arch or "flux"

        def person(p: Person) -> Dict[str, Any]:
            lora = loras.character_lora(p.name, p.sd_prompt, arch)
            shapes = {slot: text for slot, key in p.refs.items() if (text := assets.shape_text(key))}
            entry = {"key": p.key, "name": p.name, "age": p.age, "appearance": p.appearance, "slots": list(p.refs),
                     "reference_shapes": shapes, "has_lora": bool(lora), "lora_trigger": (lora or {}).get("prepend_prompt", "")}
            if p.side_of:
                entry["side_of"] = p.side_of
            return entry

        out["characters"] = [person(p) for p in self.characters]
        if self.user:
            out["user"] = person(self.user)
        out["force_pov"] = self.force_pov
        out["people_cap"] = 1 if option.lora_arch else option.people_cap
        out["concept_loras"] = [{"key": c["key"], "keywords": c["keywords"]} for c in loras.concepts(option.lora_arch)]
        return out


def load_guide(family: str) -> str:
    name = GUIDE_FOR.get(family, "seedream-4.5")
    text = (GUIDES / f"{name}.md").read_text(encoding="utf-8")
    return text.split("---", 2)[-1].strip() if text.startswith("---") else text.strip()


def mode_rules(mode: str, option: Any) -> str:
    if mode == "lora":
        return ("Mode: LoRA image. Write the still in `prompt`. Don't describe the character's identity beyond what the "
                "Appearance text says, since the LoRA carries the face. No {{ref:...}} placeholders.")
    if mode == "text":
        return "Mode: text image. Write the still in `prompt` with a full physical description. No {{ref:...}} placeholders."
    if option.kind == "image":
        return (f"Mode: reference image. The model receives up to {option.max_refs} reference images. "
                "Write the still in `prompt`, naming what each reference carries with {{ref:Name:slot}}.")
    if mode == "reference":
        return (f"Mode: reference video. No still is made first; the model builds the clip from up to {option.max_refs} "
                "reference images. Write the whole clip in `motion_prompt` using {{ref:Name:slot}}, and also write a still "
                "of the opening moment in `prompt`, in case no reference exists yet. Leave `last_frame_prompt` empty.")
    if mode == "first_last":
        return ("Mode: first and last frame. `prompt` is the opening still, `last_frame_prompt` is the closing still, "
                "and `motion_prompt` is what happens between them. Make one clear change between the two stills. "
                "Each still is an image for another model, so describe it fully, using each person's name and {{ref:Name:slot}}.")
    return ("Mode: first frame. `prompt` is the opening still, an image for another model, so describe it fully using each "
            "person's name and {{ref:Name:slot}}. `motion_prompt` is what moves from that still onward. "
            "Don't use {{ref:...}} in `motion_prompt`. Leave `last_frame_prompt` empty.")


def director_messages(ctx: DirectorInput, option: Any, mode: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": DIRECTOR_RULES},
        {"role": "system", "content": f"Target model: {option.label}\n{option.limits_text()}\n{load_guide(option.family)}"},
        {"role": "system", "content": mode_rules(mode, option)},
        {"role": "user", "content": json.dumps(ctx.public(option), ensure_ascii=False)},
    ]


async def make_plan(ctx: DirectorInput, option: Any, mode: str) -> Plan:
    """One retry on a failed or malformed reply, then the job fails. There's no fallback to a raw
    prompt, so a bad director result can't get past the minor guard.

    The timeout covers the reasoning room as well as the plan: GLM writes about 55 tokens a second."""
    messages = director_messages(ctx, option, mode)
    last_error: Optional[Exception] = None
    for attempt in (1, 2):
        try:
            with telemetry.step("director", "plan", attempt=attempt, family=option.family, mode=mode, provider=option.provider) as facts:
                raw = await helper.chat(messages, model=MediaConfig.DIRECTOR_MODEL, response_format=PLAN_SCHEMA,
                                        max_tokens=3000, temperature=0.5, timeout=240, purpose="plan")
                facts["raw"] = {"reply": raw}
                plan = Plan.parse(raw)
                facts.update(blocked=plan.blocked, adult=plan.adult, pov=plan.pov, people=len(plan.people),
                             aspect=plan.aspect_ratio, prompt_chars=len(plan.prompt), motion_chars=len(plan.motion_prompt),
                             concepts=plan.concept_loras, over_limit=bool(option.max_prompt and len(plan.motion_prompt or plan.prompt) > option.max_prompt))
            return plan
        except Exception as e:
            last_error = e
    raise RuntimeError(f"The director failed: {last_error}")


# ---- minor guard ----

MINOR_WORDS = re.compile(
    r"\b(child|children|kid|kids|minor|underage|schoolgirl|schoolboy|preteen|pre-teen|teen|teens|teenage|teenager"
    r"|toddler|infant|loli|shota)\b", re.I)
YEARS_OLD = re.compile(r"\b(\d{1,2})[\s-]*(?:years?|yrs?|yo)[\s-]*old\b|\b(\d{1,2})\s*yo\b", re.I)


def minor_block(plan: Plan, people: Sequence[Person]) -> Optional[str]:
    """Reason to refuse, or None. Naked references are always in play, so any hit stops the job."""
    if plan.blocked:
        return plan.block_reason or "The director flagged a minor"
    missing = [p.name for p in people if p.age is None or p.age < 18]
    if missing:
        return f"{', '.join(missing)} needs an age of 18 or over. Add it in the Media panel's Appearance tab."
    text = " ".join([plan.prompt, plan.motion_prompt, plan.last_frame_prompt])
    if MINOR_WORDS.search(text):
        return "The prompt mentions a minor"
    if any(int(a or b) < 18 for a, b in YEARS_OLD.findall(text)):
        return "The prompt gives an age under 18"
    return None


# ---- send media or not ----

DECIDE_SYSTEM = """You are the media director for a roleplay chat between {user} and {char}.
After {char}'s latest message, decide whether the scene also gets a photo or a short video right now.

Media is an event in the story, not a running illustration of it. Most messages get none. Send one only when this moment gives a reason the last one didn't:
- someone asks for, takes, sends or shows a photo, selfie or video. That includes {user} photographing or filming {char}: show the picture that was taken.
- {char} says they'll send something, or deliberately shows {user} an outfit, a look, their body, a place or an object.
- the scene turns over: a new place, a new outfit, someone arriving, a jump in time or weather, something revealed. Send it once, for the change itself, not again while it lasts.
The first two are reasons on their own. The third counts only when the change lands in this message, not when it carried over from earlier ones.

Everything else is "none", however well it is written: a gesture, an expression, a feeling, a scene already shown, a place already pictured, or a moment that merely could be photographed. If you find yourself weighing it up, answer "none". Another moment will come, and media sent without a reason is worse than media missed.

- image: the normal choice when something is sent.
- video: when movement is the point (dancing, a gesture being performed, something unfolding) or a video was asked for or is being filmed. Videos cost far more, so choose image unless the motion is the reason.
{context}{limits}
Reply with only a JSON object: {{"action": "none|image|video", "reason": "<10 words>"}}"""


def _ago(count: Any) -> str:
    try:
        n = int(count)
    except (TypeError, ValueError):
        return "never in this chat"
    return "on the latest message" if n == 0 else f"{n} {'message' if n == 1 else 'messages'} ago"


async def decide(body: Dict[str, Any]) -> Dict[str, Any]:
    allow_image, allow_video = bool(body.get("allow_image", True)), bool(body.get("allow_video", True))
    limits = ""
    if not allow_image or not allow_video:
        allowed = [a for a, ok in (("image", allow_image), ("video", allow_video)) if ok]
        limits = f"\nRight now only these are allowed: {', '.join(allowed) or 'none'}. Anything else must be \"none\".\n"
    char, user = body.get("call_with") or body.get("char") or "Character", body.get("user") or "User"
    context = ""
    if "since_image" in body or "since_video" in body:
        context += (f"\nLast photo: {_ago(body.get('since_image'))}. Last video: {_ago(body.get('since_video'))}."
                    " The more recent it is, the stronger the reason has to be, and never send the same shot twice.\n")
    if body.get("call"):
        context += (f"\n{user} and {char} are on a live voice call: {char}'s lines are spoken, not texted. People still send photos and"
                    f" videos while on the phone, so the same reasons apply, and anything {char} says they're sending should arrive.\n")
    system = DECIDE_SYSTEM.format(user=user, char=char, limits=limits, context=context)
    transcript = []
    for m in body.get("history") or []:
        transcript.append(f"{m.get('name') or ('User' if m.get('is_user') else char)}: {m.get('text') or ''}")
    if body.get("reply"):
        transcript.append(f"{char}: {body['reply']}")
    prompt = "\n".join(transcript)
    if body.get("summary"):
        prompt = f"Summary so far: {body['summary']}\n\n{prompt}"
    reply = await helper.chat([{"role": "system", "content": system}, {"role": "user", "content": f"{prompt}\n\nDecide now."}],
                              model=body.get("model") or MediaConfig.DIRECTOR_MODEL, max_tokens=450, temperature=0.3, timeout=120,
                              purpose="decide")
    decision = extract_json(reply)
    action = str(decision.get("action") or "none").lower()
    if action not in ("none", "image", "video") or (action == "image" and not allow_image) or (action == "video" and not allow_video):
        action = "none"
    result = {"action": action, "reason": str(decision.get("reason") or "")[:200]}
    telemetry.record("director", "decide", ok=True, chat_id=body.get("chat_id"), allow_image=allow_image, allow_video=allow_video,
                     call=bool(body.get("call")), **result)
    return result


# ---- group turns ----

NEXT_SPEAKER_SYSTEM = """You run the turn order in a group roleplay chat. Decide who acts next.

Choose from the members. The Setting card speaks for side characters and the world, so choose its name when one of them should act.
Everyone able to act should, including reacting to each other, until the user's input is really needed.
Choose "user" when the user was asked something, when the user's decision is needed, or when nobody else would naturally act now.
Don't pick someone just because they haven't spoken.

Reply with only a JSON object:
{"speaker": "<member name or user>", "beat": "<one light nudge, not a script, under 15 words>", "long": false}
Set "long" to true only when this moment deserves a fuller reply. Most replies are short."""


async def next_speaker(body: Dict[str, Any]) -> Dict[str, Any]:
    members = body.get("members") or []
    names = {str(m.get("name")).lower(): m.get("name") for m in members}
    reply = await helper.chat([{"role": "system", "content": NEXT_SPEAKER_SYSTEM},
                               {"role": "user", "content": json.dumps(body, ensure_ascii=False)}],
                              model=body.get("model") or MediaConfig.DIRECTOR_MODEL, max_tokens=300, temperature=0.5, purpose="next_speaker")
    decision = extract_json(reply)
    speaker = str(decision.get("speaker") or "user")
    resolved = names.get(speaker.lower(), "user")
    result = {"speaker": resolved, "beat": str(decision.get("beat") or "")[:300], "long": bool(decision.get("long"))}
    telemetry.record("group", "next_speaker", ok=True, chat_id=body.get("chat_id"), members=len(members),
                     spoken=len(body.get("spoken_this_turn") or []), asked_for=speaker, **result)
    return result
