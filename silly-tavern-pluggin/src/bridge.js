// Client for the Flux LoRA Bridge (images, video jobs, helper LLM).

const ctx = () => SillyTavern.getContext();
const SETTINGS_KEY = 'flux_media';

let capabilities = null;

/** Bridge base URL: our setting, else SillyTavern's Image Generation "SD WebUI URL", else localhost. */
export function bridgeBase() {
    const own = ctx().extensionSettings?.[SETTINGS_KEY]?.bridgeUrl?.trim();
    const sd = ctx().extensionSettings?.sd?.auto_url?.trim();
    const raw = own || sd || 'http://localhost:7861';
    return raw.replace(/\/sdapi\/v1\/txt2img\/?$/, '').replace(/\/+$/, '');
}

export class BridgeError extends Error {
    constructor(message, status = 0) {
        super(message);
        this.status = status;
    }
}

/**
 * fetch() against the bridge with the optional edit token and readable errors.
 * @returns {Promise<any>} parsed JSON
 */
export async function bridge(path, { method = 'GET', body, timeout = 240000 } = {}) {
    const token = ctx().extensionSettings?.[SETTINGS_KEY]?.bridgeToken?.trim();
    const headers = { 'Content-Type': 'application/json' };
    if (token) headers['X-Edit-Token'] = token;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    let res;
    try {
        res = await fetch(`${bridgeBase()}${path}`, {
            method,
            headers,
            body: body === undefined ? undefined : JSON.stringify(body),
            signal: controller.signal,
        });
    } catch (err) {
        if (err?.name === 'AbortError') throw new BridgeError(`Bridge timed out on ${path}`);
        throw new BridgeError(`Cannot reach the bridge at ${bridgeBase()} (${err?.message ?? err})`);
    } finally {
        clearTimeout(timer);
    }
    let data = null;
    try { data = await res.json(); } catch { /* non-JSON */ }
    if (!res.ok) {
        const detail = data?.detail ?? data?.error?.message ?? data?.error ?? res.statusText;
        throw new BridgeError(`Bridge ${res.status}: ${typeof detail === 'string' ? detail : JSON.stringify(detail)}`, res.status);
    }
    return data;
}

/** Cached /media/capabilities (providers, video catalog, helper status). */
export async function getCapabilities(force = false) {
    if (!capabilities || force) capabilities = await bridge('/media/capabilities', { timeout: 15000 });
    return capabilities;
}

export function cachedCapabilities() {
    return capabilities;
}

/** Downloads a bridge-hosted file (e.g. /videos/<id>.mp4) as a Blob. */
export async function bridgeBlob(path) {
    const res = await fetch(`${bridgeBase()}${path}`);
    if (!res.ok) throw new BridgeError(`Could not download ${path} (${res.status})`, res.status);
    return await res.blob();
}
