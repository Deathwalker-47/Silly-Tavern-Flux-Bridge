"""Drives the real bridge app over real FastAPI routing, with the mock provider.

The other tests stub FastAPI, so they can't see a route that was never wired, a bad path
pattern or a token check that isn't applied. This runs in its own process
(tests/test_media_http.py) with the real packages. Prints PASS and exits 0, or fails.
"""
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time

folder = tempfile.mkdtemp()
os.environ.update({
    "DATA_DIR": folder, "VIDEO_MOCK_PROVIDER": "true", "MOCK_RENDER_SECONDS": "0.05", "EDIT_API_TOKEN": "test-token",
    "HELPER_API_KEY": "fake", "DIRECTOR_MODEL": "mock/fast", "NANOGPT_API_KEY": "", "RUNWARE_API_KEY": "", "REPLICATE_API_TOKEN": "",
    "SOGNI_API_KEY": "", "PIXELDOJO_API_KEY": "", "LORA_DICT_PATH": os.path.join(folder, "none.json"),
})
Path(folder, "none.json").write_text(json.dumps({"version": 3, "loras": {}}), encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.chdir(ROOT)

from fastapi.testclient import TestClient  # noqa: E402
import httpx  # noqa: E402

import flux_lora_bridge as bridge  # noqa: E402
from media import director, lora_files  # noqa: E402
from media.config import MediaConfig  # noqa: E402
from media.providers import PROVIDERS  # noqa: E402
from media.routes import public_generation  # noqa: E402
from media_support import noisy_jpeg, small_jpeg  # noqa: E402

AUTH = {"Authorization": "Bearer test-token"}
PLAN = json.dumps({"prompt": "{{ref:Maya:face_1}} by the window", "motion_prompt": "{{ref:Maya:face_1}} turns", "main_character": "c:maya",
                   "people": [{"key": "c:maya", "role": "main"}], "aspect_ratio": "9:16", "caption": "hello"})


async def fake_chat(messages, **kw):
    purpose = kw.get("purpose")
    if purpose == "decide":
        return '{"action": "image", "reason": "asked"}'
    if purpose == "next_speaker":
        return '{"speaker": "Maya", "beat": "smiles"}'
    return PLAN


director.helper.chat = fake_chat


HF_TOKEN = "hf_test_token"
LORA = bytes(range(256)) * 64


def fake_hugging_face(request):
    """Hugging Face wants the token and redirects to its CDN, which must never see it."""
    if request.url.host == "huggingface.co":
        allowed = request.headers.get("authorization") == f"Bearer {HF_TOKEN}"
        if request.url.path == "/api/whoami-v2":
            return httpx.Response(200 if allowed else 401, json={"name": "me"})
        if not allowed:
            return httpx.Response(401)
        if "missing" in request.url.path:
            return httpx.Response(404)
        return httpx.Response(302, headers={"Location": "https://cdn.test/blob?sig=1"})
    assert "authorization" not in request.headers, "the Hugging Face token reached the CDN"
    byte_range = request.headers.get("range")
    if byte_range:
        start, end = (int(n) for n in byte_range.removeprefix("bytes=").split("-"))
        return httpx.Response(206, content=LORA[start:end + 1],
                              headers={"Content-Range": f"bytes {start}-{end}/{len(LORA)}", "Accept-Ranges": "bytes"})
    return httpx.Response(200, content=LORA, headers={"Accept-Ranges": "bytes"})


def check_lora_links(client):
    """Private LoRAs: a random link streams the file from Hugging Face, ranges included, no bridge token."""
    lora_files._transport = httpx.MockTransport(fake_hugging_face)
    MediaConfig.PUBLIC_BASE_URL = "https://bridge.test"
    saved = client.put("/media/keys/huggingface", json={"key": HF_TOKEN}, headers=AUTH).json()
    assert saved["saved"] and not saved["error"], saved
    row = next(k for k in client.get("/media/keys", headers=AUTH).json()["keys"] if k["id"] == "huggingface")
    assert (row["label"], row["source"]) == ("Hugging Face (private LoRAs)", "settings"), row

    path = lora_files.provider_url("https://huggingface.co/u/r/resolve/main/Maya%20(v2).safetensors").removeprefix("https://bridge.test")
    assert path.startswith("/lora-files/") and path.endswith("/Maya_v2_.safetensors"), path
    whole = client.get(path)
    assert whole.status_code == 200 and whole.content == LORA, whole.status_code
    part = client.get(path, headers={"Range": "bytes=10-19"})
    assert part.status_code == 206 and part.content == LORA[10:20], part.status_code
    assert part.headers["content-range"] == f"bytes 10-19/{len(LORA)}", part.headers
    head = client.head(path)
    assert head.status_code == 200 and head.headers["content-length"] == str(len(LORA)), head.headers
    assert client.get("/lora-files/made-up/x.safetensors").status_code == 404
    missing = lora_files.provider_url("https://huggingface.co/u/r/resolve/main/missing.safetensors")
    assert client.get(missing.removeprefix("https://bridge.test")).status_code == 404

    refused = client.put("/media/keys/huggingface", json={"key": "hf_wrong_token"}, headers=AUTH).json()
    assert refused["error"] == "Hugging Face refused the token (HTTP 401)", refused
    assert client.get(path).status_code == 502, "a refused token is the bridge's problem, not a missing file"
    assert client.delete("/media/keys/huggingface", headers=AUTH).status_code == 200
    assert lora_files.provider_url("https://huggingface.co/u/r/resolve/main/a.safetensors").startswith("https://huggingface.co/")


def wait(client, job_id):
    for _ in range(400):
        job = client.get(f"/media/jobs/{job_id}", headers=AUTH).json()
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"job stuck: {job}")


def main():
    with TestClient(bridge.app) as client:
        # the token guards everything except files
        assert client.get("/media/capabilities").status_code == 401
        assert client.get("/media/insights").status_code == 401

        caps = client.get("/media/capabilities", headers=AUTH).json()
        mock_image = next(p for p in caps["image"]["providers"] if p["id"] == "mock")
        assert mock_image["configured"] and mock_image["families"]
        assert [f for f in mock_image["families"] if f["id"] == "seedream-5-pro"] == [], "hidden steps stay out of the pickers"
        assert caps["reference_window"]["max_bytes"] == 10 * 1024 * 1024
        assert list(caps["modes"]) == ["reference", "first", "first_last"]

        # references: the window, and uploading once
        photo = noisy_jpeg()
        upload = client.post("/assets", content=photo, headers=AUTH)
        assert upload.status_code == 200, upload.text
        asset = upload.json()["asset_id"]
        assert client.post("/assets", content=photo, headers=AUTH).json()["existed"] is True
        assert client.get(f"/assets/{asset}/exists", headers=AUTH).json() == {"exists": True}
        refused = client.post("/assets", content=small_jpeg(), headers=AUTH)
        assert refused.status_code == 422 and "20 KB" in refused.json()["detail"]
        assert client.post("/assets?purpose=frame", content=small_jpeg(), headers=AUTH).status_code == 200
        assert client.post("/assets?purpose=nope", content=photo, headers=AUTH).status_code == 400

        # the director's two quick questions
        assert client.post("/director/decide", json={"reply": "x"}, headers=AUTH).json()["action"] == "image"
        speaker = client.post("/director/next-speaker", json={"members": [{"name": "Maya"}]}, headers=AUTH).json()
        assert speaker["speaker"] == "Maya"

        # a whole job
        person = {"key": "c:maya", "name": "Maya", "age": 24, "refs": {"face_1": asset}}
        body = {"kind": "video", "selection": {"provider": "mock", "family": "minimax-h3", "mode": "reference"}, "chat_id": "chat", "message_id": 3,
                "message_key": "k1", "folder": "Maya", "context": {"characters": [person], "reply": "hi"}}
        started = client.post("/media/jobs", json=body, headers=AUTH)
        assert started.status_code == 200, started.text
        job = wait(client, started.json()["job_id"])
        assert job["status"] == "done", job
        assert job["caption"] == "hello"
        assert client.get(job["file_url"]).status_code == 200, "files work with no token, for <video> tags"
        assert client.get("/media/files/nothing").status_code == 404
        listed = client.get("/media/jobs", params={"chat_id": "chat"}, headers=AUTH).json()["jobs"]
        assert [j["id"] for j in listed] == [job["id"]]
        assert client.post(f"/media/jobs/{job['id']}/ack", json={"st_url": "/user/images/Maya/x.mp4"}, headers=AUTH).json() == {"ok": True}
        assert client.get("/media/jobs", params={"chat_id": "chat"}, headers=AUTH).json()["jobs"] == []

        # gallery, rating, trace
        gallery = client.get("/media/generations", params={"person": "c:maya"}, headers=AUTH).json()
        assert gallery["unrated"] == 1 and gallery["generations"][0]["url"] == "/user/images/Maya/x.mp4"
        assert client.patch(f"/media/generations/{job['id']}", json={"rating": 9}, headers=AUTH).status_code == 400
        assert client.patch(f"/media/generations/{job['id']}", json={"rating": 5, "comment": "great"}, headers=AUTH).json() == {"ok": True}
        assert client.get("/media/generations", params={"person": "c:maya"}, headers=AUTH).json()["unrated"] == 0
        trace = client.get(f"/media/generations/{job['id']}/trace", headers=AUTH).json()
        assert any(s["stage"] == "provider" and s["event"] == "render" for s in trace["steps"])

        # bad requests are clear refusals, not 500s
        bad = client.post("/media/jobs", json={"kind": "image", "selection": {"provider": "mock", "family": "wan-3.0"}}, headers=AUTH)
        assert bad.status_code == 400 and "has no" in bad.json()["detail"]
        gone = client.post("/media/jobs", json={"kind": "video", "selection": {"provider": "mock", "family": "minimax-h3", "mode": "first"},
                                                "message_image_asset": "0" * 32}, headers=AUTH)
        assert gone.status_code == 400 and "Upload it again" in gone.json()["detail"], gone.text

        # test this route, then fail it: it leaves the pickers, model ids with slashes included
        assert client.post("/media/routes/test", json={"prompt": " "}, headers=AUTH).status_code == 400
        test = client.post("/media/routes/test", json={"kind": "image", "prompt": "a chair", "selection": {"provider": "mock", "family": "best"}}, headers=AUTH)
        assert wait(client, test.json()["job_id"])["status"] == "done"
        assert client.put("/media/routes/nanogpt/alibaba/wan-3.0/image-to-video-spicy", json={"status": "banned"}, headers=AUTH).json() == {"ok": True}
        caps = client.get("/media/capabilities", headers=AUTH).json()
        nano = next(p for p in caps["video"]["providers"] if p["id"] == "nanogpt")
        assert "wan-3.0" not in [f["id"] for f in nano["families"]]
        assert [r["model"] for r in caps["failed_routes"]] == ["alibaba/wan-3.0/image-to-video-spicy"]

        # restore puts it back
        client.put("/media/routes/nanogpt/alibaba/wan-3.0/image-to-video-spicy", json={"status": "allowed"}, headers=AUTH)
        caps = client.get("/media/capabilities", headers=AUTH).json()
        nano = next(p for p in caps["video"]["providers"] if p["id"] == "nanogpt")
        assert caps["failed_routes"] == [] and "wan-3.0" in [f["id"] for f in nano["families"]]

        # a gallery row counts every step of its generation
        row = {"id": "g", "kind": "video", "created_at": 1, "cost_credits": 11.25, "steps_usd": 0.04, "steps_credits": None}
        assert (public_generation(row)["cost_usd"], public_generation(row)["cost_credits"]) == (0.04, 11.25)
        assert public_generation({"id": "g", "kind": "image", "created_at": 1})["cost_usd"] is None

        # the record reads back
        assert client.post("/media/events", json={"events": [{"event": "attached", "generation_id": job["id"], "ms": 12}]}, headers=AUTH).json() == {"ok": True}
        report = client.get("/media/insights", params={"days": 1}, headers=AUTH).json()
        assert report["totals"]["done"] >= 1 and report["routes"] and report["client"]["events"]["attached"] == 1
        lines = client.get("/media/insights/export", params={"days": 1}, headers=AUTH).text.splitlines()
        assert json.loads(lines[0])["type"] == "generation"
        assert client.get("/media/balances", headers=AUTH).json() == {"balances": []}

        # cancel: allowed while the job is still queued, refused once the mock provider has it,
        # since a provider with no cancel API renders on and bills whatever the browser does.
        # Which of the two happens is a race against the render, so both are checked.
        slow = bridge.app  # the app object is kept alive by the client; the job below finishes or cancels inside this block
        started = client.post("/media/jobs", json={"kind": "image", "selection": {"provider": "mock", "family": "seedream-4.5"}, "prompt": "x"}, headers=AUTH)
        job_id = started.json()["job_id"]
        assert "can_cancel" in client.get(f"/media/jobs/{job_id}", headers=AUTH).json(), "the browser is told whether to offer it"
        stop = client.post(f"/media/jobs/{job_id}/cancel", headers=AUTH)
        if stop.status_code == 409:
            assert "no way to stop" in stop.json()["detail"], stop.json()
        else:
            assert stop.json()["status"] in ("cancelled", "done"), stop.json()
        assert slow

        # keys: nobody gets one back, and no real provider sees a fake one
        secret = "fake-test-key-9876"

        async def fake_balance():
            return {"amount": 5.0, "unit": "USD"}

        PROVIDERS["nanogpt"].balance = fake_balance
        assert client.get("/media/keys").status_code == 401
        assert client.put("/media/keys/nanogpt", json={"key": secret}).status_code == 401
        assert client.put("/media/keys/nothing", json={"key": secret}, headers=AUTH).status_code == 404
        assert client.put("/media/keys/nanogpt", json={"key": "has space in it"}, headers=AUTH).status_code == 400
        saved = client.put("/media/keys/nanogpt", json={"key": secret}, headers=AUTH)
        assert saved.status_code == 200 and saved.json()["balance"] == 5.0, saved.text
        listed = client.get("/media/keys", headers=AUTH)
        row = next(k for k in listed.json()["keys"] if k["id"] == "nanogpt")
        assert (row["source"], row["hint"], listed.json()["writable"]) == ("settings", "9876", True), listed.text
        caps = client.get("/media/capabilities", headers=AUTH)
        assert next(p for p in caps.json()["image"]["providers"] if p["id"] == "nanogpt")["configured"]
        for reply in (saved, listed, caps):
            assert secret not in reply.text
        events = sqlite3.connect(Path(folder, "media.db")).execute("SELECT * FROM events").fetchall()
        assert any("key_saved" in str(e) for e in events) and secret not in str(events)
        bridge.Config.EDIT_API_TOKEN = ""
        assert client.put("/media/keys/nanogpt", json={"key": secret}).status_code == 403
        assert client.delete("/media/keys/nanogpt").status_code == 403
        closed = client.get("/media/keys").json()
        assert not closed["writable"] and all(k["hint"] == "" for k in closed["keys"]), closed
        bridge.Config.EDIT_API_TOKEN = "test-token"
        assert client.delete("/media/keys/nanogpt", headers=AUTH).status_code == 200
        assert next(k for k in client.get("/media/keys", headers=AUTH).json()["keys"] if k["id"] == "nanogpt")["source"] == "none"

        # retired: they took no token, so anyone could spend credits and use every LoRA
        assert client.post("/sdapi/v1/txt2img", json={"prompt": "x"}).status_code == 410
        assert client.post("/v1/images/generations", json={"prompt": "x"}).status_code == 410
        assert client.get("/v1/models").status_code == 404

        # Immich is a photo library: token only
        for path in ("/immich/status", "/immich/albums", "/immich/albums/a1", "/immich/assets",
                     "/immich/assets/a1/thumbnail", "/immich/assets/a1/original"):
            assert client.get(path).status_code == 401, path
        assert client.get("/immich/status", headers=AUTH).json()["configured"] is False

        check_lora_links(client)

    print("PASS")


if __name__ == "__main__":
    main()
