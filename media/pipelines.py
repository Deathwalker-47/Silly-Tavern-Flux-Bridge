"""From a job to a finished file: plan, guard, references, keyframes, render, store.

Image and video always use the provider, model and mode that were selected. The only
automatic switch is a Flux job for a character with no LoRA, which moves to Seedream 4.5
with references.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import httpx

from . import assets, catalog, db, director, loras, telemetry
from .catalog import ALL_SHAPES, PROVIDER_LABELS, Route
from .config import MediaConfig
from .director import Plan
from .providers import PROVIDERS, provider_configured
from .providers.base import RenderRequest, RenderResult
from .references import Ref, pick_references, render_tokens

logger = logging.getLogger("media.pipelines")

EXTENSIONS = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp", "video/mp4": "mp4", "video/webm": "webm"}


class Blocked(RuntimeError):
    """The minor guard refused the job. No provider was called."""


def new_seed() -> int:
    return random.randint(0, 2_147_483_647)


def media_dir() -> Path:
    path = MediaConfig.DATA_DIR / "media"
    path.mkdir(parents=True, exist_ok=True)
    return path


def image_route(family: str, providers: Sequence[str]) -> catalog.ModelOption:
    """The first of these providers that offers the family, has a key and an allowed route."""
    for name in dict.fromkeys(providers):
        option = catalog.find("image", name, family)
        if option and provider_configured(name) and catalog.route_status(option) is Route.ALLOWED:
            return option
    raise ValueError(f"No provider is available for {catalog.IMAGE_FAMILIES.get(family, family)}")


def people_in_plan(plan: Plan, ctx: director.DirectorInput, option: Any) -> List[Any]:
    """The people the director chose, main character first, cut to this model's limit."""
    chosen = ctx.people_by_key([plan.main_character] + [p.get("key") for p in plan.people])
    if not plan.main_character:
        chosen = ctx.people_by_key(p.get("key") for p in plan.people)
    return chosen[:1] if option.lora_arch else chosen[:option.people_cap]


def guard(plan: Plan, people: Sequence[Any]) -> None:
    reason = director.minor_block(plan, people)
    telemetry.record("guard", "check", ok=reason is None, reason=reason, people=len(people))
    if reason:
        raise Blocked(reason)


async def plan_shot(job: Any, option: Any, mode: str) -> Tuple[Plan, List[Any]]:
    if job.direct:
        plan = job.direct_plan()
        guard(plan, [])
        return plan, []
    job.report("planning", "The director is planning the shot")
    plan = await director.make_plan(job.ctx, option, mode)
    people = people_in_plan(plan, job.ctx, option)
    db.update_generation(job.id, plan_json=json.dumps(plan.public(), ensure_ascii=False))
    guard(plan, people)
    db.add_people(job.id, [{"key": p.key, "role": p.role} for p in people])
    return plan, people


def choose_aspect(job: Any, option: Any, plan: Plan) -> str:
    """The director decides unless a fixed ratio is set. If it gave none, the screen decides."""
    screen = "16:9" if job.device == "landscape" else "9:16"
    wanted = job.aspect_setting if job.aspect_setting in ALL_SHAPES else plan.aspect_ratio
    wanted = wanted if wanted in ALL_SHAPES else screen
    aspect = catalog.pick_aspect(option, wanted)
    telemetry.record("pipeline", "aspect", wanted=plan.aspect_ratio, used=aspect, allowed=list(option.aspects), screen=job.device)
    return aspect


def to_assets(refs: Sequence[Ref]) -> List[assets.Asset]:
    return [assets.load(r.asset) for r in refs]


def timeout_for(req: RenderRequest) -> int:
    if req.option.provider == "sogni" and (req.network or MediaConfig.SOGNI_NETWORK) == "relaxed":
        return MediaConfig.SOGNI_RELAXED_TIMEOUT
    return MediaConfig.VIDEO_JOB_TIMEOUT if req.option.kind == "video" else MediaConfig.IMAGE_JOB_TIMEOUT


def trim_prompt(req: RenderRequest) -> None:
    """The director writes within the limit. This is the last stop before a provider's 400."""
    limit = req.option.max_prompt
    if limit and len(req.prompt) > limit:
        telemetry.record("pipeline", "prompt_trimmed", chars=len(req.prompt), limit=limit)
        req.prompt = req.prompt[:limit]


async def collect(provider: Any, task_id: str, req: RenderRequest, created_at: float) -> RenderResult:
    """Polls until the render ends. A network blip never ends the wait, only the deadline does."""
    deadline = created_at + timeout_for(req)
    polls = 0
    with telemetry.step("provider", "render", provider=provider.name, model=req.option.id, mode=req.mode, task=task_id) as facts:
        while True:
            polls += 1
            try:
                result = await provider.poll(task_id, req)
            except httpx.HTTPError as e:
                telemetry.record("provider", "poll_error", provider=provider.name, error=str(e)[:200])
                result = None
            if result:
                facts.update(polls=polls, bytes=len(result.data), mime=result.mime)
                return result
            if time.time() > deadline:
                facts["polls"] = polls
                raise TimeoutError(f"{provider.name} render timed out")
            await asyncio.sleep(provider.poll_seconds)


async def render(job: Any, req: RenderRequest, row_id: Optional[str] = None) -> RenderResult:
    row_id = row_id or job.id
    provider = PROVIDERS[req.option.provider]
    trim_prompt(req)
    fields = dict(provider=req.option.provider, model_id=req.option.id, family=req.option.family, mode=req.mode,
                  prompt=req.prompt, negative_prompt=req.negative_prompt, aspect_ratio=req.aspect_ratio,
                  resolution=req.resolution or req.option.lowest, seconds=req.seconds if req.option.kind == "video" else None)
    db.update_generation(row_id, **fields)
    telemetry.record("pipeline", "prompt", row=row_id, family=req.option.family, chars=len(req.prompt), refs=len(req.refs),
                     loras=[l.get("id") for l in req.loras], raw={"prompt": req.prompt, "negative": req.negative_prompt})
    async with job.slot(provider.name):
        label = PROVIDER_LABELS.get(provider.name, provider.name).split(" (")[0]
        queue = ", relaxed queue: can take a while" if req.network == "relaxed" else ""
        job.report("rendering", f"Rendering on {label} ({req.option.label}{queue})", row_id=row_id)
        with telemetry.step("provider", "submit", provider=provider.name, model=req.option.id, mode=req.mode, refs=len(req.refs),
                            seconds=req.seconds if req.option.kind == "video" else None, aspect=req.aspect_ratio) as facts:
            task_id = await provider.submit(req)
            facts.update(task=task_id, credits=req.cost_credits, usd=req.cost_usd, raw={"params": req.params})
        db.update_generation(row_id, provider_task_id=task_id, seed=req.params.get("seed"),
                             params_json=json.dumps(telemetry.scrub(req.params), ensure_ascii=False, default=str),
                             cost_usd=req.cost_usd, cost_credits=req.cost_credits)
        return await collect(provider, task_id, req, time.time())


def save_file(row_id: str, result: RenderResult) -> Path:
    if len(result.data) < 100:
        raise RuntimeError("The provider returned an empty file")
    path = media_dir() / f"{row_id}.{EXTENSIONS.get(result.mime, 'bin')}"
    temp = path.with_suffix(".part")
    temp.write_bytes(result.data)
    temp.replace(path)
    return path


def finish_row(row_id: str, result: RenderResult, req: RenderRequest, *, message: str = "Done", **fields: Any) -> None:
    row = db.generation(row_id) or {}
    path = save_file(row_id, result)
    usd = next((c for c in (result.cost_usd, row.get("cost_usd"), req.cost_usd) if c is not None), None)
    if usd is None and row.get("cost_credits") is None and result.cost_credits is None:
        usd = req.option.cost(req.seconds)
    credits = result.cost_credits if result.cost_credits is not None else row.get("cost_credits")
    seed = result.seed if result.seed is not None else row.get("seed")
    db.update_generation(row_id, status="done", message=message, file_path=str(path), mime=result.mime, seed=seed,
                         cost_usd=usd, cost_credits=credits, finished_at=int(time.time()), **fields)
    width, height = None, None
    if result.mime.startswith("image/"):
        from .providers.base import image_size
        width, height = image_size(result.data)
    telemetry.record("pipeline", "stored", row=row_id, bytes=len(result.data), mime=result.mime, width=width, height=height,
                     usd=usd, credits=credits, seed=seed)


def note_inputs(row_id: str, refs: Sequence[Ref] = (), stack: Sequence[dict] = ()) -> None:
    """Stored before the render, so a job that resumes after a restart keeps them."""
    db.update_generation(row_id, refs_json=json.dumps([r.public() for r in refs]),
                         loras_json=json.dumps([{"id": l.get("id"), "weight": l.get("weight")} for l in stack]))


def finish(job: Any, req: RenderRequest, result: RenderResult, parent: Optional[str] = None) -> None:
    started = (db.generation(job.id) or {}).get("created_at") or time.time()
    finish_row(job.id, result, req, message=f"Done in {time.time() - started:.0f}s", parent_id=parent)


def child_row(job: Any, role: str) -> str:
    """Keyframes and Best Option step 1 get their own generation, outside the chat's open jobs."""
    return job.runner.child(job, role)


# ---- images ----

def reference_people(option: Any, plan: Plan, people: Sequence[Any], job: Any) -> List[Ref]:
    if "reference" not in option.modes:
        return []
    refs = pick_references(people, plan.ref_order, option.max_refs)
    telemetry.record("refs", "picked", family=option.family, cap=option.max_refs, sent=len(refs),
                     available=sum(len(p.refs) for p in people), slots=[f"{r.person.name}:{r.slot}" for r in refs])
    return refs


async def run_image(job: Any) -> None:
    option = job.option
    job.ctx.force_pov = bool(option.lora_arch)
    mode = option.modes[0]
    plan, people = await plan_shot(job, option, mode)

    if option.family == "best":
        return await run_best(job, plan, people, option)

    no_lora = option.lora_arch == "flux" and not job.direct and (
        not people or not loras.character_lora(people[0].name, people[0].sd_prompt, "flux"))
    if no_lora:
        job.report("planning", "No LoRA for this character, using Seedream 4.5 with references")
        telemetry.record("pipeline", "no_lora_switch", character=people[0].name if people else "")
        option = image_route("seedream-4.5", (option.provider, "nanogpt", "pixeldojo"))
        job.ctx.force_pov = False
        mode = option.modes[0]
        plan, people = await plan_shot(job, option, mode)      # re-plan for the new model's format

    refs = reference_people(option, plan, people, job)
    extra_assets = job.direct_refs()
    stack: List[dict] = []
    prompt = render_tokens(plan.prompt, option, refs)
    negative = plan.negative_prompt
    subject = (people[0].name, people[0].sd_prompt) if people else None
    if job.lora_person():
        subject = (job.lora_person().get("name") or "", job.lora_person().get("sd_prompt") or "")
    if option.lora_arch and subject:
        stack = loras.lora_stack(plan.concept_loras, option.lora_arch, *subject)
        telemetry.record("loras", "stack", arch=option.lora_arch, character=subject[0], used=[l["id"] for l in stack],
                         concepts_asked=plan.concept_loras)
        prompt, negative = loras.compose(prompt, stack, negative)
    req = RenderRequest(option=option, mode=mode, prompt=prompt, negative_prompt=negative, refs=to_assets(refs) + extra_assets,
                        loras=stack, aspect_ratio=choose_aspect(job, option, plan), seed=new_seed(), network=job.network)
    note_inputs(job.id, refs, stack)
    result = await render(job, req)
    finish(job, req, result)


def face_edit_prompt(faces: Sequence[Ref], option: Any) -> str:
    """Neither Seedream model has a true face swap, so step 2 is an edit that keeps everything else."""
    lines = [f"{option.ref_token.format(n=1)} is the finished scene."]
    for index, ref in enumerate(faces, 2):
        lines.append(f"{option.ref_token.format(n=index)} shows {ref.person.name}'s face.")
    lines.append("Give each person in the scene their matching face exactly. Keep the pose, body, clothing, "
                 "expression, lighting, framing and background of the scene unchanged.")
    return " ".join(lines)


async def run_best(job: Any, plan: Plan, people: Sequence[Any], option: Any) -> None:
    scene = image_route("seedream-5-pro", (option.provider,))
    finish_model = image_route("seedream-4.5", (option.provider,))
    aspect = choose_aspect(job, finish_model, plan)

    refs = pick_references(people, plan.ref_order, scene.max_refs)
    step_one = RenderRequest(option=scene, mode="reference", prompt=render_tokens(plan.prompt, scene, refs), refs=to_assets(refs),
                             aspect_ratio=catalog.pick_aspect(scene, aspect), seed=new_seed(), network=job.network)
    row = child_row(job, "best_scene")
    first = await render(job, step_one, row)
    finish_row(row, first, step_one)
    base, _existing = assets.store(first.data, "frame")

    faces = pick_references(people, ["face_1", "face_2"], finish_model.max_refs - 1)
    final = RenderRequest(option=finish_model, mode="reference", prompt=face_edit_prompt(faces, finish_model),
                          refs=[base, *to_assets(faces)], aspect_ratio=aspect, seed=new_seed(), network=job.network)
    note_inputs(job.id, [*refs, *faces])
    result = await render(job, final)
    finish(job, final, result, parent=row)


# ---- video ----

def keyframe_route(job: Any, people: Sequence[Any]) -> Tuple[catalog.ModelOption, bool]:
    chosen = job.image_selection
    active = chosen.get("provider", "")
    flux = chosen.get("family") == "flux-lora" and len(people) == 1 and \
        loras.character_lora(people[0].name, people[0].sd_prompt, "flux")
    if flux:
        return image_route("flux-lora", (active, "runware", "replicate")), True
    return image_route("seedream-4.5", (active, "nanogpt", "pixeldojo")), False


async def make_keyframe(job: Any, plan: Plan, people: Sequence[Any], prompt: str, extra: Sequence[assets.Asset] = (),
                        shape: Optional[str] = None, role: str = "keyframe") -> assets.Asset:
    option, flux = keyframe_route(job, people)
    job.report("keyframe", "Generating the last frame" if extra else "Generating the first frame")
    aspect = catalog.pick_aspect(option, shape or choose_aspect(job, option, plan))
    stack: List[dict] = []
    refs: List[Ref] = []
    negative = ""
    if flux:
        stack = loras.lora_stack(plan.concept_loras, "flux", people[0].name, people[0].sd_prompt)
        text, negative = loras.compose(render_tokens(prompt, option, []), stack)
    else:
        refs = pick_references(people, plan.ref_order, option.max_refs - len(extra))
        text = render_tokens(prompt, option, refs, offset=len(extra))
        if extra:
            text = f"{option.ref_token.format(n=1)} is the opening frame: same place, people and light. {text}"
    req = RenderRequest(option=option, mode=option.modes[0], prompt=text, negative_prompt=negative,
                        refs=[*extra, *to_assets(refs)], loras=stack, aspect_ratio=aspect, seed=new_seed(), network=job.network)
    row = child_row(job, role)
    note_inputs(row, refs, stack)
    result = await render(job, req, row)
    finish_row(row, result, req)
    asset, _existing = assets.store(result.data, "frame")
    job.keyframe_id = job.keyframe_id or row
    db.update_generation(job.id, parent_id=job.keyframe_id)
    return asset


async def run_video(job: Any) -> None:
    option = job.option
    plan, people = await plan_shot(job, option, job.mode)
    aspect = choose_aspect(job, option, plan)
    seconds = job.seconds_for(option)
    common = dict(option=option, mode=job.mode, negative_prompt=plan.negative_prompt, aspect_ratio=aspect, seconds=seconds,
                  resolution=option.lowest, seed=new_seed(), audio=job.audio_for(option), adult=plan.adult, network=job.network)
    motion = plan.motion_prompt or plan.prompt
    refs: List[Ref] = []

    if job.mode == "reference":
        refs = reference_people(option, plan, people, job)
        images = to_assets(refs) + job.direct_refs()
        prompt = render_tokens(motion, option, refs)
        if not images:                                 # no slots filled: this message's image, or a new still
            images = [job.message_image or await make_keyframe(job, plan, people, plan.prompt)]
            prompt = f"{option.ref_token.format(n=1)} sets the people and the place. {prompt}"
        req = RenderRequest(prompt=prompt, refs=images, **common)
    else:
        first = job.message_image or await make_keyframe(job, plan, people, plan.prompt)
        last = None
        if job.mode == "first_last":
            last = job.last_image or await make_keyframe(job, plan, people, plan.last_frame_prompt or motion, extra=[first],
                                                         shape=aspect, role="last_frame")
        req = RenderRequest(prompt=render_tokens(motion, option, []), first=first, last=last, **common)
    note_inputs(job.id, refs)
    result = await render(job, req)
    finish(job, req, result, parent=job.keyframe_id)
