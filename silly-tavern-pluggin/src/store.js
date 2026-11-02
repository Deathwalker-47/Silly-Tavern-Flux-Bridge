// SillyTavern-side storage: media files in the character's image folder, media on chat messages.

const ctx = () => SillyTavern.getContext();

export function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result).split(',')[1] || '');
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(blob);
    });
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

/**
 * Adds an image or video to a chat message's media gallery, the same way
 * SillyTavern's own image generation does, then saves the chat.
 */
export async function attachMedia(messageId, { url, type, title, origin, job }) {
    const c = ctx();
    const message = c.chat[messageId];
    if (!message) throw new Error('Message no longer exists');
    if (!message.extra || typeof message.extra !== 'object') message.extra = {};
    if (!Array.isArray(message.extra.media)) message.extra.media = [];
    if (job && message.extra.media.some(m => m.fm_job === job)) return;
    if (!message.extra.media.length && !message.extra.media_display) message.extra.media_display = 'gallery';

    message.extra.inline_image = !(message.extra.media.length && !message.extra.inline_image);
    message.extra.media.push({
        url,
        type,
        title: String(title || '').slice(0, 500),
        source: 'generated',
        flux_media: origin || 'manual',
        fm_job: job,
    });
    message.extra.media_index = message.extra.media.length - 1;

    const element = $(`#chat .mes[mesid="${messageId}"]`);
    if (element.length) c.appendMediaToMessage(message, element, 'keep');
    await c.saveChat();
    // A voice call's window covers the chat, so it shows new media itself.
    document.dispatchEvent(new CustomEvent('mediastudio:attached', { detail: { chatId: c.getCurrentChatId(), messageId, url, type, title } }));
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
        if (m.extra?.type !== 'narrator') count++;      // a /sys line, like a call log, isn't a character message
    }
    return Infinity;
}
