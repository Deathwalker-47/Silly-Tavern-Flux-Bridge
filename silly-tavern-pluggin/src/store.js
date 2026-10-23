// SillyTavern-side storage: media files in the character's image folder, media on chat messages.

const ctx = () => SillyTavern.getContext();

export const MEDIA_LIST_IMAGE = 0b001;
export const MEDIA_LIST_VIDEO = 0b010;
const VIDEO_EXT = /\.(mp4|webm|mov|mkv|avi|wmv|flv|3gp|mpg)$/i;

export function isVideoUrl(url) {
    return VIDEO_EXT.test(String(url || '').split('?')[0]);
}

export function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result).split(',')[1] || '');
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(blob);
    });
}

export async function urlToDataUri(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Could not read ${url} (${res.status})`);
    const blob = await res.blob();
    return `data:${blob.type || 'image/png'};base64,${await blobToBase64(blob)}`;
}

/** Sniffs a base64 image's format for the upload extension. */
export function imageFormat(base64) {
    const head = atob(base64.slice(0, 24));
    if (head.startsWith('\x89PNG')) return 'png';
    if (head.startsWith('\xff\xd8')) return 'jpg';
    if (head.startsWith('RIFF') && head.slice(8, 12) === 'WEBP') return 'webp';
    return 'png';
}

/**
 * Saves base64 media into user/images/<folder>/ and returns its URL path.
 * Files there also show up in SillyTavern's built-in Gallery.
 */
export async function saveFile(base64, format, folder, name) {
    const res = await fetch('/api/images/upload', {
        method: 'POST',
        headers: ctx().getRequestHeaders(),
        body: JSON.stringify({ image: base64, format, ch_name: folder, filename: String(name).replace(/\./g, '_') }),
    });
    if (!res.ok) throw new Error(`SillyTavern rejected the upload (${res.status})`);
    const { path } = await res.json();
    return path.startsWith('/') ? path : `/${path}`;
}

/** Images and videos in a character's folder, newest first. */
export async function listFolder(folder) {
    const res = await fetch('/api/images/list', {
        method: 'POST',
        headers: ctx().getRequestHeaders(),
        body: JSON.stringify({ folder, type: MEDIA_LIST_IMAGE | MEDIA_LIST_VIDEO, sortField: 'date', sortOrder: 'desc' }),
    });
    if (!res.ok) return [];
    const files = await res.json();
    return (files || []).map((name) => {
        const url = `/user/images/${encodeURIComponent(folder)}/${encodeURIComponent(name)}`;
        return { name, url, type: isVideoUrl(name) ? 'video' : 'image' };
    });
}

export async function deleteFile(url) {
    const path = decodeURIComponent(url.replace(/^\//, ''));
    const res = await fetch('/api/images/delete', {
        method: 'POST',
        headers: ctx().getRequestHeaders(),
        body: JSON.stringify({ path }),
    });
    if (!res.ok) throw new Error(`Delete failed (${res.status})`);
}

/**
 * Adds an image or video to a chat message's media gallery, the same way
 * SillyTavern's own image generation does, then saves the chat.
 */
export async function attachMedia(messageId, { url, type, title, origin }) {
    const c = ctx();
    const message = c.chat[messageId];
    if (!message) throw new Error('Message no longer exists');
    if (!message.extra || typeof message.extra !== 'object') message.extra = {};
    if (!Array.isArray(message.extra.media)) message.extra.media = [];
    if (!message.extra.media.length && !message.extra.media_display) message.extra.media_display = 'gallery';

    message.extra.inline_image = !(message.extra.media.length && !message.extra.inline_image);
    message.extra.media.push({
        url,
        type,
        title: String(title || '').slice(0, 500),
        source: 'generated',
        flux_media: origin || 'manual',
    });
    message.extra.media_index = message.extra.media.length - 1;

    const element = $(`#chat .mes[mesid="${messageId}"]`);
    if (element.length) c.appendMediaToMessage(message, element, 'keep');
    await c.saveChat();
}

/** Media items already on a message (images and videos). */
export function messageMedia(message) {
    return Array.isArray(message?.extra?.media) ? message.extra.media.filter(m => m?.url) : [];
}

/** How many character messages ago media of this type was last sent (Infinity if never). */
export function messagesSinceMedia(type) {
    const chat = ctx().chat;
    let count = 0;
    for (let i = chat.length - 1; i >= 0; i--) {
        const m = chat[i];
        if (m.is_user || m.is_system) continue;
        if (messageMedia(m).some(x => x.type === type && x.source === 'generated')) return count;
        count++;
    }
    return Infinity;
}
