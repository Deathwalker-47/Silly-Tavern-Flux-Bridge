// Submit, poll, resume after a reload, save into SillyTavern, acknowledge.
//
// The bridge owns the job, so the browser can reload or the phone can lock: the result
// attaches to its message when you come back. Jobs find their message through
// msg.extra.fm_key, because message numbers shift when earlier messages are deleted.

import { bridge, bridgeBlob, BridgeError } from './bridge.js';
import { attachMedia, blobToBase64, saveFile } from './store.js';
import { track } from './telemetry.js';

const ctx = () => SillyTavern.getContext();
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const REMEMBER_KEY = 'fm_open_jobs';
const ENDED = ['failed', 'cancelled'];
const EXTENSIONS = { 'image/jpeg': 'jpg', 'image/png': 'png', 'image/webp': 'webp', 'video/mp4': 'mp4', 'video/webm': 'webm' };
const esc = (s) => $('<span>').text(String(s ?? '')).html();

const watching = new Set();
let hooks = { counted() {}, note() {} };

/** counted(kind) bumps the daily usage, note(job) tells the character what it sent. */
export function configureJobs(next) {
    hooks = { ...hooks, ...next };
}

// ---- which jobs this browser started, so a failure while you were away still reaches you ----

function remembered() {
    try {
        return JSON.parse(localStorage.getItem(REMEMBER_KEY) || '{}');
    } catch {
        return {};
    }
}

function setRemembered(jobs) {
    try {
        localStorage.setItem(REMEMBER_KEY, JSON.stringify(jobs));
    } catch { /* storage blocked */ }
}

function remember(jobId, chatId, kind) {
    setRemembered({ ...remembered(), [jobId]: { chatId, kind } });
}

function forget(jobId) {
    const jobs = remembered();
    delete jobs[jobId];
    setRemembered(jobs);
}

// ---- messages ----

/**
 * A random id stored on the message, saved straight away so it survives a reload.
 * SillyTavern copies extra onto a new swipe, so the key remembers which swipe it was made on.
 */
export async function messageKey(message) {
    if (!message.extra || typeof message.extra !== 'object') message.extra = {};
    const swipe = message.swipe_id ?? 0;
    if (!message.extra.fm_key || message.extra.fm_swipe !== swipe) {
        message.extra.fm_key = Math.random().toString(36).slice(2, 10);
        message.extra.fm_swipe = swipe;
        await ctx().saveChat();
    }
    return message.extra.fm_key;
}

export function messageIndexByKey(key) {
    return key ? ctx().chat.findIndex(m => m?.extra?.fm_key === key && (m.swipe_id ?? 0) === (m.extra.fm_swipe ?? 0)) : -1;
}

// ---- status line ----

function statusLine(job) {
    const id = `fm-status-${job.id}`;
    const place = () => {
        let el = document.getElementById(id);
        if (el) return el;
        const index = messageIndexByKey(job.message_key);
        const block = document.querySelector(`#chat .mes[mesid="${index}"] .mes_block`);
        if (!block) return null;
        el = document.createElement('div');
        el.id = id;
        el.className = 'fm-status';
        el.innerHTML = '<span class="fm-status-text"></span> <span class="fm-cancel menu_button_icon" title="Cancel"><i class="fa-solid fa-xmark"></i> Cancel</span>';
        el.querySelector('.fm-cancel').addEventListener('click', (event) => {
            event.currentTarget.textContent = 'Cancelling…';
            cancelJob(job.id);
        });
        block.appendChild(el);
        return el;
    };
    return {
        set(state) {
            const el = place();
            if (!el) return;
            const icon = state.kind === 'video' ? 'fa-film' : 'fa-image';
            el.querySelector('.fm-status-text').innerHTML = `<i class="fa-solid ${icon} fa-beat-fade"></i> ${esc(state.message || state.status)} · ${state.elapsed ?? 0}s`;
        },
        remove: () => document.getElementById(id)?.remove(),
    };
}

export async function cancelJob(jobId) {
    try {
        await bridge(`/media/jobs/${jobId}/cancel`, { method: 'POST', timeout: 15000 });
    } catch (err) {
        toastr.error(err?.message || String(err), 'Media Studio');
    }
}

// ---- saving ----

async function ack(job, attached, reason, started) {
    forget(job.id);
    const url = attached?.url;
    await bridge(`/media/jobs/${job.id}/ack`, {
        method: 'POST',
        body: { st_url: url, attached: !!attached?.onMessage, reason, wait_seconds: started ? Math.round((Date.now() - started) / 1000) : null },
        timeout: 15000,
    }).catch(() => {});
}

async function storeFile(path, kind, folder, prefix) {
    const blob = await bridgeBlob(path);
    const ext = EXTENSIONS[blob.type] || (kind === 'video' ? 'mp4' : 'jpg');
    return saveFile(await blobToBase64(blob), ext, folder, `${prefix}_${Date.now()}`);
}

/**
 * Message found: attach, then ack. Chat changed: leave the job open, and resumeOpenJobs
 * attaches it when you return. Message gone: keep the file in the gallery with a toast.
 */
export async function saveResult(job, started) {
    const c = ctx();
    if (job.st_url) return;                                 // another tab got there first
    if (c.getCurrentChatId() !== job.chat_id) return;
    const index = messageIndexByKey(job.message_key);
    const folder = job.folder || 'MediaStudio';
    const url = await storeFile(job.file_url, job.kind, folder, job.kind === 'video' ? 'video' : 'img');
    hooks.counted(job.kind === 'video' ? 'videos' : 'images');
    if (index < 0) {
        toastr.info(`The message is gone, so this ${job.kind} went to ${folder}'s gallery.`, 'Media Studio');
        track('attach_missed', { generation_id: job.id, kind: job.kind });
        await ack(job, { url, onMessage: false }, 'message_gone', started);
        return;
    }
    const title = job.caption || job.prompt;
    if (job.keyframe_url) {
        const still = await storeFile(job.keyframe_url, 'image', folder, 'frame');
        await attachMedia(index, { url: still, type: 'image', title, origin: job.origin, job: `${job.id}:frame` });
    }
    await attachMedia(index, { url, type: job.kind, title, origin: job.origin, job: job.id });
    if (job.prompt) {
        c.chatMetadata.fm_last_prompt = job.prompt;
        c.saveMetadataDebounced?.();
    }
    hooks.note(job);
    await ack(job, { url, onMessage: true }, 'attached', started);
}

// ---- watching a job ----

/** Polls until the job ends or you leave the chat. Network blips never end the wait. */
async function watch(first) {
    if (watching.has(first.id)) return;
    watching.add(first.id);
    const chatId = first.chat_id;
    const started = Date.now();
    const status = statusLine(first);
    status.set(first);
    let hidden = 0;
    try {
        for (let job = first; ;) {
            if (job.status === 'done') return await saveResult({ ...job, origin: first.origin }, started);
            if (ENDED.includes(job.status)) {
                forget(job.id);
                if (job.status === 'failed') {
                    toastr.error(job.error || 'Generation failed', `Media Studio: ${job.kind} failed`);
                    track('job_failed_seen', { generation_id: job.id, kind: job.kind });
                }
                return;
            }
            await sleep(document.hidden ? 8000 : 3000);
            if (document.hidden) hidden++;
            if (ctx().getCurrentChatId() !== chatId) return;   // it attaches when you come back
            const next = await bridge(`/media/jobs/${job.id}`, { timeout: 20000 }).catch((err) => {
                if (err instanceof BridgeError && err.status === 404) return { ...job, status: 'failed', error: 'The bridge no longer has this job' };
                return null;
            });
            if (!next) continue;
            job = next;
            status.set(job);
        }
    } catch (err) {
        console.error('[MediaStudio]', err);
        toastr.error(err?.message || String(err), 'Media Studio');
        track('save_failed', { generation_id: first.id, error: String(err?.message || err).slice(0, 200), hidden_polls: hidden });
    } finally {
        status.remove();
        watching.delete(first.id);
    }
}

/** Starts a job for a message. Returns when it has been saved. */
export async function runJob(body, messageId) {
    const c = ctx();
    const chatId = c.getCurrentChatId();
    body.chat_id = chatId;
    body.message_id = messageId;
    body.message_key = await messageKey(c.chat[messageId]);
    const { job_id } = await bridge('/media/jobs', { method: 'POST', body, timeout: 60000 });
    remember(job_id, chatId, body.kind);
    const job = await bridge(`/media/jobs/${job_id}`, { timeout: 20000 });
    await watch({ ...job, origin: body.origin });
}

/** A job with no message (panel images, route tests). Resolves with the finished job. */
export async function runDetached(body, path = '/media/jobs') {
    const { job_id } = await bridge(path, { method: 'POST', body, timeout: 60000 });
    for (;;) {
        await sleep(2500);
        const job = await bridge(`/media/jobs/${job_id}`, { timeout: 20000 }).catch(() => null);
        if (!job) continue;
        if (job.status === 'done') return job;
        if (ENDED.includes(job.status)) throw new Error(job.error || job.status);
    }
}

/** On page load and chat change: pick up running jobs, save finished ones, report failures. */
export async function resumeOpenJobs() {
    const chatId = ctx().getCurrentChatId();
    if (!chatId) return;
    let jobs;
    try {
        ({ jobs } = await bridge(`/media/jobs?chat_id=${encodeURIComponent(chatId)}&open=1`, { timeout: 20000 }));
    } catch {
        return;
    }
    if (jobs.length) track('resumed', { jobs: jobs.length });
    for (const job of jobs) watch(job);
    const seen = new Set(jobs.map(j => j.id));
    for (const [id, info] of Object.entries(remembered())) {
        if (info.chatId !== chatId || seen.has(id)) continue;
        const job = await bridge(`/media/jobs/${id}`, { timeout: 20000 }).catch(() => null);
        forget(id);
        if (job?.status === 'failed') toastr.error(job.error || 'Generation failed', `Media Studio: ${job.kind} failed while you were away`);
    }
}
