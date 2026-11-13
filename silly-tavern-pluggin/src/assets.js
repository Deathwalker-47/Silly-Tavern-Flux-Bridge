// Images on their way to the bridge: the size window, then uploaded once.

import { bridge, bridgeBlob, bridgeUpload, cachedCapabilities } from './bridge.js';

const byUrl = new Map();

async function hashOf(blob) {
    if (!crypto.subtle) return null;       // plain http on a LAN address: the bridge dedupes instead
    const digest = await crypto.subtle.digest('SHA-256', await blob.arrayBuffer());
    return [...new Uint8Array(digest)].map(b => b.toString(16).padStart(2, '0')).join('').slice(0, 32);
}

const sizeText = (bytes) => (bytes >= 1048576 ? `${(bytes / 1048576).toFixed(1)} MB` : `${Math.floor(bytes / 1024)} KB`);

/** Why this file can't be a reference, or '' if it can. The bridge checks again on upload. */
export async function windowProblem(file) {
    const w = cachedCapabilities()?.reference_window;
    if (!w) return '';
    if (file.size < w.min_bytes) return `Image is ${sizeText(file.size)}. The minimum is 20 KB.`;
    if (file.size > w.max_bytes) return `Image is ${sizeText(file.size)}. The maximum is 10 MB.`;
    let bitmap;
    try {
        bitmap = await createImageBitmap(file);
    } catch {
        return '';                         // HEIC doesn't decode here; the bridge reads it
    }
    const { width, height } = bitmap;
    bitmap.close();
    const short = Math.min(width, height);
    const long = Math.max(width, height);
    if (short < w.min_side) return `Image is ${width}×${height}. The shortest side must be at least ${w.min_side} px.`;
    if (long > w.max_side) return `Image is ${width}×${height}. The longest side can be at most ${w.max_side} px.`;
    if (width * height > w.max_pixels) return `Image is ${Math.round(width * height / 1e6)} megapixels. The maximum is 50.`;
    if (short / long < w.min_shape) return `Image is ${width}×${height}, which is too narrow. The shape must be between 2:5 and 5:2.`;
    return '';
}

/**
 * Stores an image on the bridge, once, and returns its asset id. A reference joins the
 * shared library under its label, such as "Maya, Face 1".
 */
export async function uploadAsset(blob, purpose = 'reference', label = '') {
    const hash = await hashOf(blob);
    if (hash && purpose === 'frame') {
        const { exists } = await bridge(`/assets/${hash}/exists`, { timeout: 15000 }).catch(() => ({ exists: false }));
        if (exists) return hash;
    }
    const named = label ? `&label=${encodeURIComponent(label)}` : '';
    const { asset_id } = await bridgeUpload(`/assets?purpose=${purpose}${named}`, blob);
    return asset_id;
}

// ---- the shared library: every reference uploaded for anyone, to pick again without uploading ----

/** One page, newest first: [{ id, label, created_at, width, height, thumb }] and whether more follow. */
export const libraryPage = (offset = 0, limit = 48) => bridge(`/assets?limit=${limit}&offset=${offset}`, { timeout: 30000 });

/** The full stored image, to save into a slot's folder. */
export const libraryImage = (id) => bridgeBlob(`/assets/${encodeURIComponent(id)}/file`, { auth: true });

/** Out of the library only. Slots already using it keep it. */
export const removeFromLibrary = (id) => bridge(`/assets/${encodeURIComponent(id)}`, { method: 'DELETE', timeout: 15000 });

/** A SillyTavern file as a bridge asset. Message images and keyframes are frames, so they skip the window. */
export async function urlToAsset(url, purpose = 'frame') {
    const key = `${purpose}:${url}`;
    if (byUrl.has(key)) return byUrl.get(key);
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Could not read ${url} (${res.status})`);
    const id = await uploadAsset(await res.blob(), purpose);
    byUrl.set(key, id);
    return id;
}
