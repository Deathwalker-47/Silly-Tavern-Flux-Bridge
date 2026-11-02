#!/usr/bin/env python3
"""
Local stand-ins for the paid services, for testing the bridge without spending credits.
Pair with dev/run_local.py.

    python dev/mock_providers.py          # http://localhost:8788

Runware (POST /runware):
  imageUpload      -> a fake imageUUID
  imageInference   -> a generated PNG (imageBase64Data) labelled with the prompt
NanoGPT (/nanogpt): /v1/images, /generate-video, /video/status, /check-balance
PixelDojo (/pixeldojo): /upload, /models/{id}/run, /jobs/{id}, /credits
Helper LLM (POST /v1/chat/completions, OpenAI-compatible):
  director plan     -> a plan JSON for the first character in the context
  decide            -> {"action", "reason"} ("video"/"pic"/"photo"/"selfie" in the last lines)
  next speaker      -> the first member who hasn't spoken, then "user" ("keep talking" never yields)
  voice tags prompt -> the same line with one [laugh] inserted
  summarizer prompt -> a short visual prompt

Video renders take about 4 seconds, so progress and cancel can be watched.
"""

import base64
import json
import re
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8788
CLIP = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mock_clip.mp4"
RENDER_SECONDS = 4
VIDEO_TASKS = {}


def log(msg):
    print(time.strftime("[%H:%M:%S]"), msg, flush=True)


def fake_png(label: str) -> bytes:
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (512, 768), (40 + hash(label) % 160, 60, 110))
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(re.findall(r".{1,40}(?:\s|$)", label[:400])):
        draw.text((16, 16 + i * 16), line.strip(), fill=(255, 255, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def runware(tasks):
    out = []
    for task in tasks:
        kind, tid = task.get("taskType"), task.get("taskUUID") or str(uuid.uuid4())
        if kind == "imageUpload":
            out.append({"taskType": kind, "taskUUID": tid, "imageUUID": str(uuid.uuid4())})
            log("Runware imageUpload")
        elif kind == "imageInference":
            prompt = task.get("positivePrompt", "")
            log(f"Runware imageInference model={task.get('model')} loras={len(task.get('lora') or [])}: {prompt[:90]}")
            out.append({"taskType": kind, "taskUUID": tid, "cost": 0.0037,
                        "imageBase64Data": base64.b64encode(fake_png(prompt)).decode()})
        elif kind == "accountManagement":
            out.append({"taskType": kind, "taskUUID": tid, "balance": 12.5})
        else:
            return {"errors": [{"message": f"mock does not support {kind}"}]}
    return {"data": out}


def mock_plan(ctx):
    """A plan the way the director writes one, for whoever is in the context."""
    people = ctx.get("characters") or []
    main = people[0] if people else {"key": "", "name": "the character", "slots": []}
    slot = (main.get("slots") or ["face_1"])[0]
    ref = f"{{{{ref:{main['name']}:{slot}}}}}" if main.get("slots") else main["name"]
    request = ctx.get("user_request") or "smiling at the camera in a warm room"
    return {
        "blocked": False, "block_reason": "", "main_character": main.get("key", ""),
        "people": [{"key": p["key"], "role": "main" if i == 0 else "other"} for i, p in enumerate(people[:ctx.get("people_cap", 3)])],
        "pov": bool(ctx.get("force_pov")), "adult": False, "user_visible": "", "aspect_ratio": "9:16",
        "aspect_reason": "phone framing", "ref_order": ["face_1", "full_body", "portrait", "face_2", "naked"], "concept_loras": [],
        "prompt": f"{ref}, {request}", "negative_prompt": "", "last_frame_prompt": f"{ref}, looking out of the window",
        "motion_prompt": f"{ref} turns slowly and smiles", "caption": "here you go",
    }


def chat(body):
    messages = body.get("messages") or []
    system = "\n".join(m["content"] for m in messages if m.get("role") == "system")
    user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")

    if "Target model:" in system:
        reply = json.dumps(mock_plan(json.loads(user)))
        log(f"Director plan: {reply[:120]}")
    elif "turn order" in system:
        data = json.loads(user)
        spoken = set(data.get("spoken_this_turn") or [])
        names = [m.get("name") for m in data.get("members") or [] if not m.get("setting_card")]
        speaker = next((n for n in names if n not in spoken), "user")
        if "keep talking" in json.dumps(data.get("history")).lower():     # for hitting the round limit
            speaker = names[0]
        time.sleep(0.8)                                                   # a window to type or press Stop
        reply = json.dumps({"speaker": speaker, "beat": "reacts to the last line", "long": False})
        log(f"Next speaker -> {speaker}")
    elif "media director" in system:
        last = "\n".join(user.lower().splitlines()[-4:])
        action = "video" if "video" in last else "image" if re.search(r"pic|photo|selfie|picture", last) else "none"
        reply = json.dumps({"action": action, "reason": "mock decision"})
        log(f"Director decide -> {action}")
    elif "expressive speech tags" in system:
        line = user.split("Line:", 1)[-1].strip()
        reply = re.sub(r"([.!?])\s", r"\1 [laugh] ", line, count=1) if re.search(r"[.!?]\s", line) else f"[laugh] {line}"
        log(f"Helper voice tags: {reply[:80]}")
    elif "visual prompt engineer" in system:
        text = user.split("\n\n", 1)[-1]
        reply = "photo of " + " ".join(text.split()[:30])
        log("Helper summarizer")
    else:                       # SillyTavern's own reply, for chats on this mock
        reply = f"*smiles and answers* {str(user)[:120]}"
        if "slowly" in str(user):
            time.sleep(4)       # a reply long enough to press Stop on
    return {"id": "chatcmpl-mock", "object": "chat.completion",
            "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}]}


def nanogpt(path, body):
    if path == "/v1/images":
        log(f"NanoGPT image model={body.get('model')} resolution={body.get('resolution')} refs={len(body.get('input_references') or [])}")
        return {"cost": 0.04, "data": [{"b64_json": base64.b64encode(fake_png(body.get("prompt", ""))).decode()}]}
    if path == "/generate-video":
        run_id = uuid.uuid4().hex
        VIDEO_TASKS[run_id] = time.time()
        log(f"NanoGPT video model={body.get('model')} duration={body.get('duration')} first={bool(body.get('imageDataUrl'))} last={bool(body.get('last_image'))} refs={len(body.get('reference_images') or [])}")
        return {"runId": run_id, "cost": 0.6}
    if path == "/check-balance":
        return {"usd_balance": 25.0}
    return None


def pixeldojo_job(job_id):
    started = VIDEO_TASKS.get(job_id)
    if started is None:
        return None
    if time.time() - started < RENDER_SECONDS:
        return {"jobId": job_id, "status": "processing"}
    return {"jobId": job_id, "status": "completed", "creditCost": 11.25, "output": {"video": f"http://localhost:{PORT}/clip.mp4"}}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, payload=None, raw=None, ctype="application/json"):
        data = raw if raw is not None else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/clip.mp4":
            return self._send(200, raw=CLIP.read_bytes(), ctype="video/mp4")
        if url.path == "/image.png":
            return self._send(200, raw=fake_png("mock image"), ctype="image/png")
        if url.path == "/nanogpt/video/status":
            run_id = parse_qs(url.query).get("requestId", [""])[0]
            started = VIDEO_TASKS.get(run_id, 0)
            if time.time() - started < RENDER_SECONDS:
                return self._send(200, {"data": {"status": "IN_PROGRESS"}})
            return self._send(200, {"data": {"status": "COMPLETED", "output": {"video": {"url": f"http://localhost:{PORT}/clip.mp4"}}}})
        if url.path.startswith("/pixeldojo/jobs/"):
            job = pixeldojo_job(url.path.rsplit("/", 1)[-1])
            return self._send(200 if job else 404, job or {"error": "not found"})
        if url.path == "/pixeldojo/credits":
            return self._send(200, {"credits": 60})
        if url.path.startswith("/v1/models"):
            return self._send(200, {"data": [
                {"id": "mock/fast-helper", "name": "Mock Fast Helper", "pricing": {"prompt": 0.01, "completion": 0.02}, "context_length": 32000},
                {"id": "mock/smart-helper", "name": "Mock Smart Helper", "pricing": {"prompt": 0.2, "completion": 0.4}, "context_length": 128000},
            ]})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.path == "/pixeldojo/upload":             # multipart, not JSON
            log(f"PixelDojo upload {len(raw)} bytes")
            return self._send(200, {"url": f"http://localhost:{PORT}/image.png?{uuid.uuid4().hex}", "expiresInHours": 24})
        body = json.loads(raw or b"null")
        if self.path.startswith("/runware"):
            return self._send(200, runware(body))
        if self.path.startswith("/v1/chat/completions"):
            reply = chat(body)
            if not body.get("stream"):
                return self._send(200, reply)
            text = reply["choices"][0]["message"]["content"]
            chunk = {"id": reply["id"], "object": "chat.completion.chunk",
                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}
            return self._send(200, raw=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode(), ctype="text/event-stream")
        if self.path.startswith("/nanogpt"):
            reply = nanogpt(self.path[len("/nanogpt"):], body)
            return self._send(200 if reply else 404, reply or {"error": "not found"})
        if self.path.startswith("/pixeldojo/models/"):
            job_id = uuid.uuid4().hex
            VIDEO_TASKS[job_id] = time.time()
            log(f"PixelDojo run {self.path}: {sorted(body)}")
            return self._send(200, {"jobId": job_id, "status": "pending", "creditCost": 11.25, "creditsRemaining": 48.75})
        self._send(404, {"error": "not found"})


if __name__ == "__main__":
    log(f"Mock providers + helper LLM on http://localhost:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
