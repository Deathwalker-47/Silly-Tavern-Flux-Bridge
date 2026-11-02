"""The job runner. Jobs are rows in `generations` run as asyncio tasks.

A row outlives everything around it: the browser can reload, the phone can lock and the
bridge can restart. Poll-based jobs carry on from provider_task_id. Socket and direct
jobs can't, so they fail as interrupted, and Sogni renders are cancelled so they stop billing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import assets, catalog, db, pipelines, telemetry
from .catalog import PROVIDER_LABELS, Route
from .config import MediaConfig
from .director import DirectorInput, Plan
from .providers import PROVIDERS, provider_configured
from .providers.base import RenderRequest

logger = logging.getLogger("media.jobs")

SWEEP_SECONDS = 6 * 3600
TERMINAL = ("done", "failed", "cancelled")


class Job:
    """One running generation: its row id, the request and helpers for the pipelines."""

    def __init__(self, job_id: str, body: Dict[str, Any], option: Any, runner: "JobRunner"):
        self.id = job_id
        self.body = body
        self.option = option
        self.runner = runner
        self.kind = body["kind"]
        self.selection = body.get("selection") or {}
        self.image_selection = body.get("image_selection") or {}
        self.mode = self.selection.get("mode") or option.modes[0]
        self.network = body.get("sogni_network") or ""
        self.device = (body.get("context") or {}).get("device") or "portrait"
        self.aspect_setting = body.get("aspect") or "director"
        self.ctx = DirectorInput(body.get("context"))
        self.direct = bool(body.get("prompt") or body.get("motion_prompt"))
        self.test = bool(body.get("test"))
        self.keyframe_id: Optional[str] = None
        self.message_image = self._asset(body.get("message_image_asset"))
        self.last_image = self._asset(body.get("last_image_asset"))

    @staticmethod
    def _asset(asset_id: Optional[str]) -> Optional[assets.Asset]:
        return assets.load(asset_id) if asset_id else None

    def direct_plan(self) -> Plan:
        """Prompts written by hand: route tests, avatars and slash commands with an exact prompt."""
        body = self.body
        return Plan(prompt=str(body.get("prompt") or body.get("motion_prompt") or ""), motion_prompt=str(body.get("motion_prompt") or ""),
                    last_frame_prompt=str(body.get("last_frame_prompt") or ""), negative_prompt=str(body.get("negative_prompt") or ""),
                    caption=str(body.get("caption") or ""), aspect_ratio=str(body.get("aspect_ratio") or ""))

    def direct_refs(self) -> List[assets.Asset]:
        return [assets.load(a) for a in self.body.get("refs") or []] if self.direct else []

    def lora_person(self) -> Optional[Dict[str, str]]:
        """Panel images for a character: a hand-written prompt that still uses its LoRA."""
        return self.body.get("lora_for") if self.direct else None

    def seconds_for(self, option: Any) -> int:
        wanted = self.body.get("seconds")
        if wanted:
            return min(option.seconds, key=lambda s: abs(s - float(wanted)))
        return min(option.seconds) if self.test else option.longest()

    def audio_for(self, option: Any) -> bool:
        return bool(self.body.get("audio")) and bool(option.extra.get("audio_switch"))

    def report(self, status: str, message: str, row_id: Optional[str] = None) -> None:
        if row_id and row_id != self.id:
            db.update_generation(row_id, status=status, message=message)
        db.update_generation(self.id, status=status, message=message)
        telemetry.record("job", status, message=message, row=row_id or self.id)
        logger.info(f"🎬 [Job {self.id[:8]}] {status}: {message}")

    def slot(self, provider: str) -> asyncio.Semaphore:
        return self.runner.slot(provider)


class JobRunner:
    def __init__(self) -> None:
        self.tasks: Dict[str, "asyncio.Task[None]"] = {}
        self.children: Dict[str, List[str]] = {}
        self.cancelled: set = set()
        self._slots: Dict[str, asyncio.Semaphore] = {}
        self._sweeper: Optional["asyncio.Task[None]"] = None

    def slot(self, provider: str) -> asyncio.Semaphore:
        if provider not in self._slots:
            self._slots[provider] = asyncio.Semaphore(MediaConfig.PROVIDER_CONCURRENCY)
        return self._slots[provider]

    def _option(self, kind: str, selection: Dict[str, Any], test: bool) -> Any:
        provider, family, variant = selection.get("provider") or "", selection.get("family") or "", selection.get("variant") or ""
        if test:                    # Test this route works on any route, including a failed one
            option = catalog.find(kind, provider, family, variant)
            if not option:
                raise ValueError(f"Unknown route {provider}/{family}/{variant}")
            return option
        return catalog.image(provider, family, variant) if kind == "image" else catalog.video(provider, family, variant)

    def create(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Starts a job, or returns the open one if this message already has one of this kind."""
        kind = body.get("kind")
        if kind not in ("image", "video"):
            raise ValueError("kind must be image or video")
        selection = body.get("selection") or {}
        test = bool(body.get("test"))
        option = self._option(kind, selection, test)
        if not provider_configured(option.provider):
            raise ValueError(f"{PROVIDER_LABELS.get(option.provider, option.provider)} is not configured on the bridge")
        mode = selection.get("mode") or option.modes[0]
        if mode not in option.modes:
            raise ValueError(f"{option.label} has no {mode} mode")
        key = body.get("message_key")
        if key:
            running = db.open_for_message(key, kind)
            if running:
                telemetry.record("job", "duplicate_blocked", generation_id=running["id"], message_key=key)
                return running
        job_id = uuid.uuid4().hex
        db.insert_generation(id=job_id, kind=kind, status="queued", message="Queued", chat_id=body.get("chat_id"),
                             message_id=body.get("message_id"), message_key=key, group_id=body.get("group_id"),
                             folder=body.get("folder"), provider=option.provider, model_id=option.id, family=option.family, mode=mode)
        job = Job(job_id, body, option, self)
        self._spawn(job.id, self._run(job))
        return db.generation(job_id)

    def child(self, job: Job, role: str) -> str:
        row = uuid.uuid4().hex
        db.insert_generation(id=row, kind="image", status="queued", message_id=job.body.get("message_id"),
                             group_id=job.body.get("group_id"), folder=job.body.get("folder"), parent_id=job.id,
                             comment=None, message=role)
        self.children.setdefault(job.id, []).append(row)
        return row

    def _spawn(self, job_id: str, coroutine: Any) -> None:
        task = asyncio.get_running_loop().create_task(coroutine)
        self.tasks[job_id] = task
        task.add_done_callback(lambda _t: self.tasks.pop(job_id, None))

    def _end_children(self, job_id: str, status: str, error: str = "") -> None:
        for row in self.children.pop(job_id, []):
            child = db.generation(row)
            if child and child["status"] not in TERMINAL:
                db.update_generation(row, status=status, error=error or None, finished_at=int(time.time()))

    def _fail(self, job_id: str, error: str, event: str = "failed") -> None:
        db.update_generation(job_id, status="failed", message="Failed", error=error, finished_at=int(time.time()))
        self._end_children(job_id, "failed", error)
        telemetry.record("job", event, ok=False, generation_id=job_id, error=error[:500])
        logger.error(f"🎬 [Job {job_id[:8]}] {event}: {error}")

    async def _run(self, job: Job) -> None:
        telemetry.current_generation.set(job.id)
        telemetry.current_chat.set(job.body.get("chat_id"))
        ctx = job.body.get("context") or {}
        telemetry.record("job", "created", kind=job.kind, provider=job.option.provider, model=job.option.id, family=job.option.family,
                         mode=job.mode, origin=job.body.get("origin") or "api", direct=job.direct, test=job.test,
                         characters=len(ctx.get("characters") or []), history=len(ctx.get("history") or []),
                         reply_chars=len(ctx.get("reply") or ""), group=bool(job.body.get("group_id")), device=job.device)
        try:
            await (pipelines.run_image if job.kind == "image" else pipelines.run_video)(job)
            row = db.generation(job.id) or {}
            telemetry.record("job", "done", ok=True, seconds=int(time.time()) - (row.get("created_at") or 0),
                             provider=row.get("provider"), model=row.get("model_id"), usd=row.get("cost_usd"), credits=row.get("cost_credits"))
        except asyncio.CancelledError:
            # A user cancel already wrote its row. On shutdown the row stays open, so recovery finds it.
            if job.id in self.cancelled:
                self._end_children(job.id, "cancelled")
            raise
        except pipelines.Blocked as e:
            self._fail(job.id, str(e), "blocked")
        except Exception as e:
            self._fail(job.id, str(e) or e.__class__.__name__)

    def cancel(self, job_id: str) -> Optional[Dict[str, Any]]:
        row = db.generation(job_id)
        if not row or row["status"] in TERMINAL:
            return row
        db.update_generation(job_id, status="cancelled", message="Cancelled", finished_at=int(time.time()))
        telemetry.record("job", "cancelled", generation_id=job_id, stage_at_cancel=row["status"], provider=row["provider"])
        self.cancelled.add(job_id)
        task = self.tasks.get(job_id)
        if task:
            task.cancel()
        else:
            self._end_children(job_id, "cancelled")
        provider = PROVIDERS.get(row["provider"] or "")
        if provider and row["provider_task_id"]:
            asyncio.get_running_loop().create_task(self._stop_remote(provider, row["provider_task_id"]))
        return db.generation(job_id)

    async def _stop_remote(self, provider: Any, task_id: str) -> None:
        try:
            await provider.cancel(task_id)
        except Exception as e:
            logger.warning(f"🎬 could not cancel {provider.name} task {task_id}: {e}")

    # ---- restart ----

    async def startup(self) -> None:
        db.connect()
        for row in db.open_generations():
            self._recover(row)
        self.sweep()
        self._sweeper = asyncio.get_running_loop().create_task(self._sweep_loop())

    async def shutdown(self) -> None:
        """Stops the tasks and leaves their rows open, so the next start picks them up."""
        if self._sweeper:
            self._sweeper.cancel()
        for task in list(self.tasks.values()):
            task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)

    def _recover(self, row: Dict[str, Any]) -> None:
        provider = PROVIDERS.get(row.get("provider") or "")
        option = catalog.by_id(row["kind"], row["provider"], row["model_id"]) if provider else None
        task_id = row.get("provider_task_id")
        resumable = row["status"] == "rendering" and task_id and provider and provider.resumable and option
        if resumable:
            telemetry.record("job", "resumed", generation_id=row["id"], provider=row["provider"], task=task_id)
            logger.info(f"🎬 [Job {row['id'][:8]}] resuming {row['provider']} task {task_id}")
            self._spawn(row["id"], self._resume(row, provider, option))
            return
        self._fail(row["id"], "Interrupted by a bridge restart", "interrupted")
        if task_id and provider and not provider.resumable:
            asyncio.get_running_loop().create_task(self._stop_remote(provider, task_id))

    async def _resume(self, row: Dict[str, Any], provider: Any, option: Any) -> None:
        telemetry.current_generation.set(row["id"])
        telemetry.current_chat.set(row.get("chat_id"))
        req = RenderRequest(option=option, mode=row.get("mode") or option.modes[0], prompt=row.get("prompt") or "",
                            aspect_ratio=row.get("aspect_ratio") or "9:16", seconds=int(row.get("seconds") or 5),
                            resolution=row.get("resolution") or option.lowest, seed=row.get("seed"))
        try:
            result = await pipelines.collect(provider, row["provider_task_id"], req, row["created_at"])
            pipelines.finish_row(row["id"], result, req, message=f"Done in {time.time() - row['created_at']:.0f}s")
            telemetry.record("job", "done", ok=True, resumed=True, seconds=int(time.time()) - row["created_at"],
                             provider=row["provider"], model=row["model_id"])
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._fail(row["id"], str(e) or e.__class__.__name__)

    # ---- retention ----

    def sweep(self, now: Optional[int] = None) -> int:
        """Files go 7 days after they're saved, or after 30 if they never were. Rows stay,
        so ratings keep their history."""
        now = now or int(time.time())
        saved = now - MediaConfig.MEDIA_FILE_TTL_DAYS * 86400
        unsaved = now - MediaConfig.MEDIA_UNSAVED_TTL_DAYS * 86400
        count = 0
        for row in db.expired_files(saved, unsaved):
            Path(row["file_path"]).unlink(missing_ok=True)
            db.update_generation(row["id"], file_path=None)
            count += 1
        assets.sweep_frames(unsaved)
        telemetry.prune()
        return count

    async def _sweep_loop(self) -> None:
        while True:
            await asyncio.sleep(SWEEP_SECONDS)
            try:
                self.sweep()
            except Exception as e:
                logger.warning(f"🎬 file sweep failed: {e}")


def public_job(row: Dict[str, Any]) -> Dict[str, Any]:
    plan = json.loads(row["plan_json"]) if row.get("plan_json") else {}
    parent = row.get("parent_id")
    return {
        "id": row["id"], "kind": row["kind"], "status": row["status"], "message": row.get("message"), "error": row.get("error"),
        "elapsed": (row.get("finished_at") or int(time.time())) - row["created_at"],
        "file_url": f"/media/files/{row['id']}" if row["status"] == "done" and row.get("file_path") else None,
        "keyframe_url": f"/media/files/{parent}" if parent and row["kind"] == "video" else None,
        "caption": plan.get("caption") or "", "prompt": row.get("prompt") or "", "chat_id": row.get("chat_id"),
        "message_id": row.get("message_id"), "message_key": row.get("message_key"), "folder": row.get("folder"),
        "mode": row.get("mode"), "provider": row.get("provider"), "st_url": row.get("st_url"),
    }


runner = JobRunner()
