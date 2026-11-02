"""The bridge's text model: voice tags, summaries and the chat client the director uses.

OpenAI-compatible, NanoGPT by default. Every call is recorded with its purpose, timing
and token use, so slow or failing prompts show up in the pipeline report.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from . import telemetry
from .config import MediaConfig
from .providers.base import error_text

logger = logging.getLogger("media.helper")

INLINE_TAGS = ["pause", "long-pause", "hum-tune", "laugh", "chuckle", "giggle", "cry",
               "tsk", "tongue-click", "lip-smack", "breath", "inhale", "exhale", "sigh"]
WRAPPING_TAGS = ["soft", "whisper", "loud", "build-intensity", "decrease-intensity",
                 "higher-pitch", "lower-pitch", "slow", "fast", "sing-song", "singing", "emphasis"]
SUBTLE_INLINE = ["pause", "laugh", "chuckle", "sigh", "breath"]
SUBTLE_WRAPPING = ["whisper", "soft"]

_TAG_RE = re.compile(r"\[[^\]\n]{1,30}\]|</?[a-zA-Z][\w-]{0,30}>")
_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


def strip_tags(text: str) -> str:
    return _TAG_RE.sub(" ", text or "")


def sanitize_speech_tags(text: str, inline: List[str], wrapping: List[str]) -> str:
    """Normalize LLM tag output to what Grok understands.

    <laugh> becomes [laugh] (small models mix up the two syntaxes), [whisper] is
    dropped (wrapping tags need a span), unknown tags are removed, and wrapping tags
    left open are closed at the end.
    """
    inline_set, wrap_set = set(inline), set(wrapping)
    stack: List[str] = []

    def norm(name: str) -> str:
        return name.strip().lower().replace(" ", "-").replace("_", "-")

    def repl(m: re.Match) -> str:
        token = m.group(0)
        if token.startswith("["):
            name = norm(token[1:-1])
            return f"[{name}]" if name in inline_set else ""
        closing = token.startswith("</")
        name = norm(token[2:-1] if closing else token[1:-1])
        if name in inline_set:
            return "" if closing else f"[{name}]"
        if name not in wrap_set:
            return ""
        if not closing:
            stack.append(name)
            return f"<{name}>"
        if name in stack:
            stack.remove(name)
            return f"</{name}>"
        return ""

    out = _TAG_RE.sub(repl, text or "")
    out += "".join(f"</{n}>" for n in reversed(stack))
    return re.sub(r"\s{2,}", " ", out).strip()


def same_words(a: str, b: str) -> bool:
    """True when two texts have identical words in order, ignoring speech tags and punctuation."""
    norm = lambda s: [w.lower() for w in _WORD_RE.findall(strip_tags(s))]  # noqa: E731
    return norm(a) == norm(b)


def extract_json(text: str) -> Dict[str, Any]:
    """First JSON object in an LLM reply (tolerates code fences and chatter)."""
    if not text:
        raise ValueError("empty reply")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        return json.loads(fenced.group(1))
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in reply: {text[:120]}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError(f"unterminated JSON object in reply: {text[:120]}")


VOICE_SYSTEM = """You add expressive speech tags to one line of dialogue that a text-to-speech voice will speak as {char}.
Rules:
- Return exactly the same words in the same order. Never add, drop or change a word. Only insert tags.
- Inline tags go where the sound happens: {inline}
- Wrapping tags go around a whole phrase, with an opening and a matching closing tag: {wrapping}
- Use 0 to 2 tags. A neutral line needs none.
- Match the mood from the context.
Output only the tagged line, nothing else."""


def text_model_allowed(model: str) -> bool:
    """No SFW-only model anywhere in the chat pipeline: only uncensored text models run."""
    named = (MediaConfig.DIRECTOR_MODEL, MediaConfig.HELPER_MODEL, MediaConfig.SUMMARIZER_MODEL)
    return model in named or "uncensored" in model.lower() or model.startswith("mock/")


# GLM always reasons before it answers, and that reasoning spends max_tokens first.
REASONING_MODELS = ("z-ai/glm",)
REASONING_MIN_TOKENS = 800


class HelperLLM:
    """OpenAI-compatible chat client used for decisions, voice tags and summaries."""

    def __init__(self):
        self._models_cache: Tuple[float, List[Dict]] = (0.0, [])

    def backend(self, model: Optional[str] = None) -> Tuple[str, str, str, str]:
        """(url, key, model, label)."""
        if MediaConfig.HELPER_API_KEY:
            return MediaConfig.HELPER_API_URL, MediaConfig.HELPER_API_KEY, model or MediaConfig.HELPER_MODEL, "helper"
        return "", "", "", "none"

    @property
    def configured(self) -> bool:
        return self.backend()[3] != "none"

    async def chat(self, messages: List[Dict[str, str]], *, model: Optional[str] = None, max_tokens: int = 500,
                   temperature: float = 0.4, timeout: float = 45, response_format: Optional[Dict] = None,
                   purpose: str = "chat") -> str:
        url, key, model_id, label = self.backend(model)
        if not key:
            raise RuntimeError("No helper LLM configured (set NANOGPT_API_KEY or HELPER_API_KEY)")
        if not text_model_allowed(model_id):
            raise RuntimeError(f"{model_id} isn't an uncensored model, so it can't run in the chat pipeline")
        payload: Dict[str, Any] = {"model": model_id, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        if model_id.startswith(REASONING_MODELS):
            payload.update(reasoning_effort="low", max_tokens=max(max_tokens, REASONING_MIN_TOKENS))
        if response_format:
            payload["response_format"] = response_format
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        with telemetry.step("llm", purpose, model=model_id, max_tokens=payload["max_tokens"],
                            input_chars=sum(len(str(m.get("content") or "")) for m in messages)) as facts:
            facts["raw"] = {"messages": messages}
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code >= 300:
                raise RuntimeError(f"helper LLM ({label}/{model_id}) {resp.status_code}: {error_text(resp)}")
            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            content = message.get("content") or ""
            if isinstance(content, list):  # some gateways return content parts
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            content = content.strip()
            usage = data.get("usage") or {}
            facts.update(output_chars=len(content), finish_reason=choice.get("finish_reason"), usage=usage or None,
                         empty=not content)
            facts["raw"]["reply"] = content
        return content

    async def voice_tags(self, text: str, *, char: str = "", tag_level: str = "full", context: str = "",
                         model: Optional[str] = None) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text or tag_level == "none":
            return {"text": text, "changed": False}
        inline = SUBTLE_INLINE if tag_level == "subtle" else INLINE_TAGS
        wrapping = SUBTLE_WRAPPING if tag_level == "subtle" else WRAPPING_TAGS
        system = VOICE_SYSTEM.format(
            char=char or "the character",
            inline=" ".join(f"[{t}]" for t in inline),
            wrapping=" ".join(f"<{t}>...</{t}>" for t in wrapping),
        )
        user_msg = (f"Context (earlier lines, do not output): {context[-1200:]}\n\n" if context else "") + f"Line: {text}"
        reply = await self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
            model=model, max_tokens=max(60, len(text) // 2 + 60), temperature=0.3, timeout=20, purpose="voice_tags",
        )
        reply = reply.strip().strip('"').strip()
        if reply.lower().startswith("line:"):
            reply = reply[5:].strip()
        if not reply or not same_words(text, reply):
            telemetry.record("voice", "tags", ok=False, tag_level=tag_level, chars=len(text), rejected=bool(reply))
            return {"text": text, "changed": False, "rejected": bool(reply)}
        reply = sanitize_speech_tags(reply, inline, wrapping)
        telemetry.record("voice", "tags", ok=True, tag_level=tag_level, chars=len(text), changed=reply != text,
                         tags=len(_TAG_RE.findall(reply)))
        return {"text": reply, "changed": reply != text}

    async def models(self, force: bool = False) -> List[Dict[str, Any]]:
        """Text models from NanoGPT's public catalog (cached for an hour)."""
        cached_at, cached = self._models_cache
        if cached and not force and time.time() - cached_at < 3600:
            return cached
        url, key, _model, label = self.backend()
        out: List[Dict[str, Any]] = []
        try:
            if "nano-gpt.com" in url or label == "none":
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.get("https://nano-gpt.com/api/v1/models", params={"detailed": "true"})
                items = (resp.json() or {}).get("data") or []
            else:
                base = url.rsplit("/chat/completions", 1)[0]
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"})
                items = (resp.json() or {}).get("data") or []
            for m in items:
                outputs = ((m.get("architecture") or {}).get("output_modalities")) or ["text"]
                if "text" not in outputs:
                    continue
                pricing = m.get("pricing") or {}
                out.append({
                    "id": m.get("id"),
                    "name": m.get("name") or m.get("id"),
                    "input_price": _float(pricing.get("prompt", pricing.get("input"))),
                    "output_price": _float(pricing.get("completion", pricing.get("output"))),
                    "context": m.get("context_length") or 0,
                })
            out.sort(key=lambda m: ((m["input_price"] or 0) + (m["output_price"] or 0), m["id"] or ""))
        except Exception as e:
            logger.warning(f"🧠 [Helper] model list failed: {e}")
        if out:
            self._models_cache = (time.time(), out)
        return out


def _float(value) -> Optional[float]:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


helper = HelperLLM()
