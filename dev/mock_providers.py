#!/usr/bin/env python3
"""
Local stand-ins for Runware and the helper LLM, for testing the bridge without
spending credits. Pair with dev/run_local.py.

    python dev/mock_providers.py          # http://localhost:8788

Runware (POST /runware):
  imageUpload      -> a fake imageUUID
  imageInference   -> a generated PNG (imageBase64Data) labelled with the prompt
  videoInference   -> queued; getResponse reports "processing" for ~4 s, then a videoURL
Helper LLM (POST /v1/chat/completions, OpenAI-compatible):
  media director prompt  -> a JSON decision ("video"/"pic"/"photo"/"selfie" in the last user line)
  voice tags prompt      -> the same line with one [laugh] inserted
  summarizer prompt      -> a short visual prompt
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

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8788
CLIP = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mock_clip.mp4"
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
            log(f"Runware imageInference model={task.get('model')} loras={len(task.get('lora') or [])} refs={bool(task.get('inputs') or task.get('referenceImages'))}: {prompt[:90]}")
            out.append({"taskType": kind, "taskUUID": tid,
                        "imageBase64Data": base64.b64encode(fake_png(prompt)).decode()})
        elif kind == "videoInference":
            VIDEO_TASKS[tid] = time.time()
            frames = (task.get("inputs") or {}).get("frameImages") or []
            log(f"Runware videoInference model={task.get('model')} duration={task.get('duration')} "
                f"res={task.get('resolution') or (task.get('width'), task.get('height'))} frames={len(frames)} "
                f"safety={task.get('safety')}: {task.get('positivePrompt', '')[:80]}")
            out.append({"taskType": kind, "taskUUID": tid})
        elif kind == "getResponse":
            started = VIDEO_TASKS.get(tid)
            if started is None:
                return {"errors": [{"message": f"unknown task {tid}"}]}
            if time.time() - started < 4:
                out.append({"taskType": "videoInference", "taskUUID": tid, "status": "processing"})
            else:
                out.append({"taskType": "videoInference", "taskUUID": tid, "status": "success",
                            "videoURL": f"http://localhost:{PORT}/clip.mp4", "cost": 0})
        else:
            return {"errors": [{"message": f"mock does not support {kind}"}]}
    return {"data": out}


def chat(body):
    messages = body.get("messages") or []
    system = next((m["content"] for m in messages if m.get("role") == "system"), "")
    user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")

    if "media director" in system:
        names = re.search(r"between (.+?) and (.+?)\.", system)
        user_name, char = (names.group(1), names.group(2)) if names else ("User", "the character")
        user_lines = [l for l in user.splitlines() if l.startswith(f"{user_name}:")]
        last = user_lines[-1].lower() if user_lines else ""
        action = "video" if "video" in last else "image" if re.search(r"pic|photo|selfie|picture", last) else "none"
        reply = json.dumps({
            "action": action, "reason": "mock decision",
            "image_prompt": f"{char}, mirror selfie in a cozy bedroom, soft lamp light" if action != "none" else "",
            "motion_prompt": f"{char} waves and smiles at the camera" if action == "video" else "",
            "caption": "here you go 😘" if action != "none" else "",
        })
        log(f"Helper director -> {action}")
    elif "expressive speech tags" in system:
        line = user.split("Line:", 1)[-1].strip()
        reply = re.sub(r"([.!?])\s", r"\1 [laugh] ", line, count=1) if re.search(r"[.!?]\s", line) else f"[laugh] {line}"
        log(f"Helper voice tags: {reply[:80]}")
    elif "visual prompt engineer" in system:
        text = user.split("\n\n", 1)[-1]
        reply = "photo of " + " ".join(text.split()[:30])
        log("Helper summarizer")
    else:
        reply = "Mock helper reply."
    return {"id": "chatcmpl-mock", "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}]}


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
        if self.path.startswith("/clip.mp4"):
            return self._send(200, raw=CLIP.read_bytes(), ctype="video/mp4")
        if self.path.startswith("/v1/models"):
            return self._send(200, {"data": [
                {"id": "mock/fast-helper", "name": "Mock Fast Helper", "pricing": {"prompt": 0.01, "completion": 0.02}, "context_length": 32000},
                {"id": "mock/smart-helper", "name": "Mock Smart Helper", "pricing": {"prompt": 0.2, "completion": 0.4}, "context_length": 128000},
            ]})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"null")
        if self.path.startswith("/runware"):
            return self._send(200, runware(body))
        if self.path.startswith("/v1/chat/completions"):
            return self._send(200, chat(body))
        self._send(404, {"error": "not found"})


if __name__ == "__main__":
    log(f"Mock Runware + helper LLM on http://localhost:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
