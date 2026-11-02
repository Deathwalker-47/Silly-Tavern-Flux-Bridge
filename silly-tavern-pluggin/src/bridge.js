// Client for the Flux LoRA Bridge (jobs, assets, director).

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
    constructor(message, status = 0, detail = '') {
        super(message);
        this.status = status;
        this.detail = detail;           // the bridge's own words, without the status prefix
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
    const timer = timeout ? setTimeout(() => controller.abort(), timeout) : null;  // 0 = wait as long as it takes
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
        const text = typeof detail === 'string' ? detail : JSON.stringify(detail);
        throw new BridgeError(`Bridge ${res.status}: ${text}`, res.status, text);
    }
    return data;
}

/** Cached /media/capabilities (image and video catalogs, modes, reference window, helper status). */
export async function getCapabilities(force = false) {
    if (!capabilities || force) capabilities = await bridge('/media/capabilities', { timeout: 15000 });
    return capabilities;
}

export function cachedCapabilities() {
    return capabilities;
}

/** Sends a raw file body. Reference uploads go through here. */
export async function bridgeUpload(path, blob) {
    const token = ctx().extensionSettings?.[SETTINGS_KEY]?.bridgeToken?.trim();
    const headers = { 'Content-Type': blob.type || 'application/octet-stream' };
    if (token) headers['X-Edit-Token'] = token;
    let res;
    try {
        res = await fetch(`${bridgeBase()}${path}`, { method: 'POST', headers, body: blob });
    } catch (err) {
        throw new BridgeError(`Cannot reach the bridge at ${bridgeBase()} (${err?.message ?? err})`);
    }
    const data = await res.json().catch(() => null);
    if (!res.ok) {
        const text = data?.detail ?? res.statusText;
        throw new BridgeError(`Bridge ${res.status}: ${text}`, res.status, text);
    }
    return data;
}

/** Downloads a bridge-hosted file (/media/files/<id>) as a Blob. */
export async function bridgeBlob(path) {
    const res = await fetch(`${bridgeBase()}${path}`);
    if (!res.ok) throw new BridgeError(`Could not download ${path} (${res.status})`, res.status);
    return await res.blob();
}
