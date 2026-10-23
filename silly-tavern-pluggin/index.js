// Media Studio for the Flux LoRA Bridge.
// Characters send photos and videos in chat: decided by the bridge's helper LLM (or
// every turn), rendered through the bridge, saved into the message and the
// character's gallery. Also: per-character avatars for characters without a LoRA.

import { bridge, bridgeBase, bridgeBlob, getCapabilities, cachedCapabilities } from './src/bridge.js';
import { attachMedia, blobToBase64, imageFormat, messageMedia, messagesSinceMedia, saveFile, urlToDataUri } from './src/store.js';
import { AVATAR_SLOTS, openCharacterMedia } from './src/gallery.js';

const ctx = () => SillyTavern.getContext();
const KEY = 'flux_media';          // extension_settings key and character card field
const NOTE_KEY = 'FLUX_MEDIA_NOTE';
const TAG = '[MediaStudio]';

const DEFAULTS = Object.freeze({
    bridgeUrl: '',
    bridgeToken: '',
    imageMode: 'auto',          // auto | lora | avatar
    imageSize: 'portrait',      // portrait | square | landscape
    steps: 40,
    cfg: 3.5,
    autoMedia: 'director',      // off | director | every_turn
    allowAutoImages: true,
    allowAutoVideos: true,
    imageCooldown: 3,
    videoCooldown: 8,
    dailyImageCap: 60,
    dailyVideoCap: 10,
    helperModel: '',
    notifyCharacter: true,
    videoProvider: 'runware',
    videoFamily: 'ltx-2.3',
    videoVariant: '',
    videoDuration: 5,
    videoResolution: '',
    videoAudio: false,
    useTwoImages: true,
    debug: false,
});
const CHAR_DEFAULTS = Object.freeze({ avatars: [], imageMode: '', auto: true, avatarPrompt: '' });
const SIZES = { portrait: [832, 1216], square: [1024, 1024], landscape: [1216, 832] };
const SKIP_TYPES = new Set(['first_message', 'swipe', 'continue', 'impersonate', 'quiet']);

function settings() {
    const all = ctx().extensionSettings;
    all[KEY] ??= {};
    const s = all[KEY];
    for (const [k, v] of Object.entries(DEFAULTS)) if (s[k] === undefined) s[k] = v;
    s.usage ??= { date: '', images: 0, videos: 0 };
    return s;
}
const save = () => ctx().saveSettingsDebounced();
const log = (...a) => settings().debug && console.log(TAG, ...a);

/** Carry the old plugin's bridge URL over once. */
function migrateLegacyConfig() {
    const s = settings();
    if (s.bridgeUrl) return;
    try {
        const legacy = JSON.parse(localStorage.getItem('autoImageUniversalConfig') || '{}');
        if (legacy.BRIDGE_URL && !ctx().extensionSettings?.sd?.auto_url) {
            s.bridgeUrl = String(legacy.BRIDGE_URL).replace(/\/sdapi\/v1\/txt2img\/?$/, '');
            save();
        }
    } catch { /* storage blocked */ }
}

// ---------------------------------------------------------------------------
// Characters
// ---------------------------------------------------------------------------

let cardWrites = Promise.resolve();

async function getCharMedia(chid) {
    const c = ctx();
    if (chid === undefined || chid === null || !c.characters[chid]) return { ...CHAR_DEFAULTS };
    await c.unshallowCharacter?.(chid); // lazy-loaded cards don't carry extension fields
    return { ...CHAR_DEFAULTS, ...(c.characters[chid]?.data?.extensions?.[KEY] ?? {}) };
}

/** Queued read-modify-write so quick edits don't overwrite each other. */
function setCharMedia(chid, patch) {
    const run = cardWrites.then(async () => {
        const next = { ...(await getCharMedia(chid)), ...patch };
        await ctx().writeExtensionField(chid, KEY, next);
        return next;
    });
    cardWrites = run.catch(() => {});
    return run;
}

/** The character who wrote a message (group-aware). */
function characterOf(message) {
    const c = ctx();
    if (!c.groupId) return { chid: c.characterId, char: c.characters[c.characterId] };
    const idx = c.characters.findIndex(ch => ch.avatar === message?.original_avatar || ch.name === message?.name);
    return { chid: idx >= 0 ? idx : undefined, char: c.characters[idx] };
}

const sdPrompt = (char) => char?.data?.extensions?.sd_character_prompt || char?.data?.extensions?.sd_api_prompt || '';

function visibleCharacters() {
    const c = ctx();
    const names = new Set();
    // Narrator notes (/sys, e.g. TavernVoice's call markers) are not characters.
    for (const m of c.chat.slice(-10)) if (!m.is_user && !m.is_system && m.extra?.type !== 'narrator' && m.name) names.add(m.name);
    if (c.groupId) {
        const group = c.groups.find(g => g.id === c.groupId);
        for (const avatar of group?.members ?? []) {
            const ch = c.characters.find(x => x.avatar === avatar);
            if (ch?.name) names.add(ch.name);
        }
    } else if (c.name2) {
        names.add(c.name2);
    }
    const prompts = {};
    for (const name of names) {
        const ch = c.characters.find(x => x.name === name);
        if (sdPrompt(ch)) prompts[name] = sdPrompt(ch);
    }
    return { names: [...names], prompts };
}

/** Raw scene text for the bridge; its summarizer turns this into a visual prompt. */
function narrativeFor(messageId) {
    const c = ctx();
    const msg = c.chat[messageId];
    const { char } = characterOf(msg);
    const { names } = visibleCharacters();
    const recent = c.chat.slice(Math.max(0, messageId - 3), messageId)
        .filter(m => !m.is_system).map(m => m.mes).join(' ').slice(-500);
    return [sdPrompt(char), names.length ? `Characters present: ${names.join(', ')}` : '', msg?.mes || '',
        recent ? `Recent context: ${recent}` : ''].filter(Boolean).join('\n\n');
}

function chatTail(uptoId, count = 10) {
    const c = ctx();
    return c.chat.slice(Math.max(0, uptoId + 1 - count), uptoId + 1)
        .filter(m => !m.is_system && m.extra?.type !== 'narrator')
        .map(m => ({ name: m.name, is_user: !!m.is_user, text: String(m.mes || '').slice(0, 1500) }));
}

async function avatarsFor(cfg) {
    const urls = (cfg.avatars || []).filter(Boolean).slice(0, 2);
    const out = [];
    for (const url of urls) {
        try { out.push(await urlToDataUri(url)); } catch (err) { console.warn(TAG, 'avatar unreadable', url, err); }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Usage limits and the note for the main LLM
// ---------------------------------------------------------------------------

function usage() {
    const s = settings();
    const today = new Date().toISOString().slice(0, 10);
    if (s.usage.date !== today) s.usage = { date: today, images: 0, videos: 0 };
    return s.usage;
}

function bumpUsage(kind) {
    usage()[kind]++;
    save();
}

let noteArmed = false;

function noteForCharacter(char, kind, text) {
    if (!settings().notifyCharacter) return;
    const c = ctx();
    const note = `[${char || c.name2} sent ${c.name1} a ${kind}${text ? `: ${String(text).slice(0, 200)}` : ''}]`;
    c.setExtensionPrompt(NOTE_KEY, note, 1, 1, false, 0); // in-chat, depth 1, system role
    noteArmed = false;
}

// ---------------------------------------------------------------------------
// Status line under a message while media renders
// ---------------------------------------------------------------------------

function statusLine(messageId, text) {
    const id = `fm-status-${messageId}-${Math.random().toString(36).slice(2, 7)}`;
    const place = () => {
        let el = document.getElementById(id);
        if (!el) {
            const block = document.querySelector(`#chat .mes[mesid="${messageId}"] .mes_block`);
            if (!block) return null;
            el = document.createElement('div');
            el.id = id;
            el.className = 'fm-status';
            block.appendChild(el);
        }
        return el;
    };
    const set = (t) => { const el = place(); if (el) el.innerHTML = t; };
    set(text);
    return { set, remove: () => document.getElementById(id)?.remove() };
}

// ---------------------------------------------------------------------------
// Image and video generation
// ---------------------------------------------------------------------------

async function imageRequest(messageId, prompt) {
    const s = settings();
    const c = ctx();
    const msg = c.chat[messageId];
    const { chid, char } = characterOf(msg);
    const cfg = await getCharMedia(chid);
    const { names, prompts } = visibleCharacters();
    const [width, height] = SIZES[s.imageSize] || SIZES.portrait;
    return {
        body: {
            prompt: prompt || narrativeFor(messageId),
            character_name: char?.name || c.name2,
            character_prompts: prompts,
            visible_characters: names,
            mode: cfg.imageMode || s.imageMode,
            avatars: await avatarsFor(cfg),
            width, height, steps: Number(s.steps), cfg_scale: Number(s.cfg),
        },
        char,
    };
}

const busy = new Set(); // "image:12" / "video:12" so a message isn't generated twice at once

async function generateImage(messageId, { prompt = '', caption = '', origin = 'manual' } = {}) {
    const key = `image:${messageId}`;
    if (busy.has(key)) return;
    busy.add(key);
    const chatId = ctx().getCurrentChatId();
    const status = statusLine(messageId, '<i class="fa-solid fa-spinner fa-spin"></i> Generating image…');
    try {
        const { body, char } = await imageRequest(messageId, prompt);
        const res = await bridge('/media/image', { method: 'POST', body });
        const b64 = res.images?.[0];
        if (!b64) throw new Error('The bridge returned no image');
        const url = await saveFile(b64, imageFormat(b64), char?.name || 'MediaStudio', `img_${Date.now()}`);
        bumpUsage('images');
        if (ctx().getCurrentChatId() !== chatId) {
            toastr.info('Chat changed; the image was saved to the character gallery.', 'Media Studio');
            return;
        }
        await attachMedia(messageId, { url, type: 'image', title: caption || res.parameters?.prompt || body.prompt, origin });
        noteForCharacter(char?.name, 'photo', caption || prompt);
        log('image', res.mode, url);
    } catch (err) {
        console.error(TAG, err);
        toastr.error(err?.message || String(err), 'Media Studio: image failed');
    } finally {
        status.remove();
        busy.delete(key);
    }
}

async function generateVideo(messageId, { motion = '', imagePrompt = '', caption = '', origin = 'manual' } = {}) {
    const key = `video:${messageId}`;
    if (busy.has(key)) return;
    busy.add(key);
    const s = settings();
    const c = ctx();
    const chatId = c.getCurrentChatId();
    const msg = c.chat[messageId];
    const { char } = characterOf(msg);
    const folder = char?.name || 'MediaStudio';
    const status = statusLine(messageId, '<i class="fa-solid fa-film"></i> Preparing video…');
    try {
        const images = messageMedia(msg).filter(m => m.type === 'image');
        const payload = {
            provider: s.videoProvider,
            family: s.videoFamily,
            variant: s.videoVariant || undefined,
            prompt: motion,
            duration: Number(s.videoDuration) || 5,
            resolution: s.videoResolution || undefined,
            audio: !!s.videoAudio,
        };
        if (images.length) {
            // Animate the message's own image; with 2+ images, travel from the first to the last.
            const two = s.useTwoImages && images.length > 1;
            payload.image = await urlToDataUri(images[two ? 0 : images.length - 1].url);
            if (two) payload.last_image = await urlToDataUri(images[images.length - 1].url);
        } else {
            payload.keyframe = (await imageRequest(messageId, imagePrompt)).body;
        }
        const job = await bridge('/video/generate', { method: 'POST', body: payload });
        const started = Date.now();
        let state = job;
        while (!['done', 'failed'].includes(state.status)) {
            await new Promise(r => setTimeout(r, 2500));
            state = await bridge(`/video/jobs/${job.job_id}`, { timeout: 20000 });
            const secs = Math.round((Date.now() - started) / 1000);
            status.set(`<i class="fa-solid fa-film fa-beat-fade"></i> ${state.message || state.status} · ${secs}s`);
            if (Date.now() - started > 25 * 60 * 1000) throw new Error('Video took longer than 25 minutes');
        }
        if (state.status === 'failed') throw new Error(state.error || 'Video generation failed');

        const sameChat = () => ctx().getCurrentChatId() === chatId;
        if (state.keyframe_b64) {
            // The bridge made the first frame; keep it too, so the chat has the still.
            const kurl = await saveFile(state.keyframe_b64, imageFormat(state.keyframe_b64), folder, `frame_${Date.now()}`);
            if (sameChat()) await attachMedia(messageId, { url: kurl, type: 'image', title: imagePrompt || motion, origin });
        }
        const blob = await bridgeBlob(state.video_path);
        const vurl = await saveFile(await blobToBase64(blob), 'mp4', folder, `video_${Date.now()}`);
        bumpUsage('videos');
        if (!sameChat()) {
            toastr.info('Chat changed; the video was saved to the character gallery.', 'Media Studio');
            return;
        }
        await attachMedia(messageId, { url: vurl, type: 'video', title: caption || motion, origin });
        noteForCharacter(char?.name, 'short video', caption || motion);
        log('video', state);
    } catch (err) {
        console.error(TAG, err);
        toastr.error(err?.message || String(err), 'Media Studio: video failed');
    } finally {
        status.remove();
        busy.delete(key);
    }
}

/** Motion + still prompts for a manual video: from the helper when available. */
async function promptsForVideo(messageId) {
    const s = settings();
    const c = ctx();
    const msg = c.chat[messageId];
    try {
        const d = await bridge('/helper/decide', {
            method: 'POST',
            body: { messages: chatTail(messageId), char: characterOf(msg).char?.name || c.name2, user: c.name1, force: 'video', model: s.helperModel || undefined },
        });
        return { motion: d.motion_prompt, imagePrompt: d.image_prompt, caption: d.caption };
    } catch (err) {
        log('helper unavailable for video prompts', err);
        return { motion: '', imagePrompt: '', caption: '' };
    }
}

// ---------------------------------------------------------------------------
// Auto media after each character message
// ---------------------------------------------------------------------------

let warnedHelper = false;

async function onMessageReceived(messageId, type) {
    if (noteArmed) {
        ctx().setExtensionPrompt(NOTE_KEY, '', 1, 1);
        noteArmed = false;
    }
    const s = settings();
    if (s.autoMedia === 'off' || SKIP_TYPES.has(type)) return;
    const c = ctx();
    const msg = c.chat[messageId];
    if (!msg || msg.is_user || msg.is_system || msg.extra?.type === 'narrator' || String(msg.mes || '').length < 20) return;
    const { chid, char } = characterOf(msg);
    const cfg = await getCharMedia(chid);
    if (cfg.auto === false) return;
    const u = usage();

    if (s.autoMedia === 'every_turn') {
        if (u.images >= Number(s.dailyImageCap)) return;
        generateImage(messageId, { origin: 'auto' });
        return;
    }

    // Helper decides. Cooldowns and caps are enforced here, not left to the LLM.
    const caps = cachedCapabilities();
    const videoReady = !!caps?.video?.providers?.find(p => p.id === s.videoProvider)?.configured;
    const allowImage = s.allowAutoImages && messagesSinceMedia('image') >= Number(s.imageCooldown) && u.images < Number(s.dailyImageCap);
    const allowVideo = s.allowAutoVideos && videoReady && messagesSinceMedia('video') >= Number(s.videoCooldown) && u.videos < Number(s.dailyVideoCap);
    if (!allowImage && !allowVideo) return;
    try {
        const d = await bridge('/helper/decide', {
            method: 'POST',
            body: {
                messages: chatTail(messageId), char: char?.name || c.name2, user: c.name1,
                allow_image: allowImage, allow_video: allowVideo, model: s.helperModel || undefined,
            },
        });
        log('director', d);
        if (d.action === 'image') generateImage(messageId, { prompt: d.image_prompt, caption: d.caption, origin: 'auto' });
        else if (d.action === 'video') generateVideo(messageId, { motion: d.motion_prompt, imagePrompt: d.image_prompt, caption: d.caption, origin: 'auto' });
    } catch (err) {
        console.warn(TAG, 'director failed', err);
        if (!warnedHelper) {
            warnedHelper = true;
            toastr.warning(`Helper LLM unavailable, so auto media is paused for now: ${err?.message || err}`, 'Media Studio');
        }
    }
}

// ---------------------------------------------------------------------------
// Settings panel
// ---------------------------------------------------------------------------

const esc = (s) => $('<span>').text(String(s ?? '')).html();

function videoSelection() {
    const s = settings();
    const caps = cachedCapabilities();
    const provider = caps?.video?.providers?.find(p => p.id === s.videoProvider);
    const family = provider?.families?.find(f => f.id === s.videoFamily);
    const variant = family?.variants?.find(v => v.id === s.videoVariant) || family?.variants?.[0];
    return { caps, provider, family, variant };
}

function renderVideoPickers() {
    const s = settings();
    const { caps, provider, family, variant } = videoSelection();
    if (!caps) return;
    $('#fm_video_provider').html(caps.video.providers.map(p =>
        `<option value="${p.id}" ${p.id === s.videoProvider ? 'selected' : ''} ${p.configured ? '' : 'disabled'}>${esc(p.label)}${p.configured ? '' : ' (no key on bridge)'}</option>`).join(''));
    $('#fm_video_family').html((provider?.families || []).map(f =>
        `<option value="${f.id}" ${f.id === s.videoFamily ? 'selected' : ''}>${esc(f.label)}</option>`).join('')
        || '<option value="">No models</option>');
    $('#fm_video_variant').html((family?.variants || []).map(v =>
        `<option value="${esc(v.id)}" ${v.id === variant?.id ? 'selected' : ''}>${esc(v.label)}${v.hint ? ` (${esc(v.hint)})` : ''}</option>`).join(''));
    $('#fm_video_resolution').html(['<option value="">Model default</option>', ...(variant?.resolutions || []).map(r =>
        `<option value="${esc(r)}" ${r === s.videoResolution ? 'selected' : ''}>${esc(r)}</option>`)].join(''));
    if (variant) {
        const durs = variant.durations;
        const hint = variant.duration_mode === 'options' ? `allowed: ${durs.join(', ')} s` : `${durs[0]}–${durs[1]} s`;
        $('#fm_video_duration_hint').text(`${hint}${variant.last_frame ? ' · supports first+last frame' : ''}`);
    }
}

async function refreshBridgeStatus({ quiet = true } = {}) {
    const $s = $('#fm_status');
    try {
        const caps = await getCapabilities(true);
        const configuredVideo = caps.video.providers.filter(p => p.configured).map(p => p.label);
        $s.removeClass('fm-bad').addClass('fm-ok').text(`Connected · video: ${configuredVideo.join(', ') || 'none'} · helper: ${caps.helper.configured ? caps.helper.default_model : 'not set'}`);
        // Keep the chosen provider valid.
        const s = settings();
        if (!caps.video.providers.find(p => p.id === s.videoProvider)?.configured) {
            const first = caps.video.providers.find(p => p.configured && p.id !== 'mock') || caps.video.providers.find(p => p.configured);
            if (first) { s.videoProvider = first.id; s.videoFamily = first.families[0]?.id || ''; s.videoVariant = ''; save(); }
        }
        renderVideoPickers();
        if (!quiet) toastr.success('Bridge connected.', 'Media Studio');
    } catch (err) {
        $s.removeClass('fm-ok').addClass('fm-bad').text(err?.message || 'Not connected');
        if (!quiet) toastr.error(err?.message || String(err), 'Media Studio');
    }
}

async function loadHelperModels(refresh = false) {
    const s = settings();
    const $sel = $('#fm_helper_model');
    try {
        const data = await bridge(`/helper/models${refresh ? '?refresh=true' : ''}`, { timeout: 40000 });
        const filter = String($('#fm_helper_filter').val() || '').toLowerCase();
        const models = data.models.filter(m => !filter || m.id.toLowerCase().includes(filter) || String(m.name).toLowerCase().includes(filter));
        const price = (m) => (m.input_price != null ? ` · $${m.input_price}/$${m.output_price} per M` : '');
        $sel.html([`<option value="">Bridge default (${esc(data.default_model)})</option>`,
            ...models.slice(0, 400).map(m => `<option value="${esc(m.id)}" ${m.id === s.helperModel ? 'selected' : ''}>${esc(m.id)}${price(m)}</option>`)].join(''));
        if (s.helperModel && !models.some(m => m.id === s.helperModel)) $sel.append(`<option value="${esc(s.helperModel)}" selected>${esc(s.helperModel)}</option>`);
    } catch (err) {
        $sel.html(`<option value="${esc(s.helperModel)}">${esc(s.helperModel || 'Bridge default')}</option>`);
    }
}

function bindSettings() {
    const s = settings();
    const $root = $('#flux_media_settings');
    $root.find('[data-fm]').each((_, el) => {
        const k = el.dataset.fm;
        if (el.type === 'checkbox') el.checked = !!s[k];
        else el.value = s[k];
    });
    $('#fm_bridge_url').attr('placeholder', bridgeBase());

    $root.on('input change', '[data-fm]', (e) => {
        const el = e.target;
        const k = el.dataset.fm;
        let v = el.type === 'checkbox' ? el.checked : el.value;
        if (el.type === 'number' || el.type === 'range') v = Number(v);
        if (s[k] === v) return;
        s[k] = v;
        save();
        if (k === 'bridgeUrl' || k === 'bridgeToken') return;
        if (k === 'videoProvider') { s.videoFamily = videoSelection().provider?.families?.[0]?.id || ''; s.videoVariant = ''; s.videoResolution = ''; }
        if (k === 'videoFamily') { s.videoVariant = ''; s.videoResolution = ''; }
        if (k === 'videoVariant') { s.videoResolution = ''; }
        if (k.startsWith('video')) renderVideoPickers();
    });
    $('#fm_bridge_url, #fm_bridge_token').on('change', async () => {
        await refreshBridgeStatus({ quiet: false });
        loadHelperModels();
    });
    $('#fm_test').on('click', () => refreshBridgeStatus({ quiet: false }));
    $('#fm_helper_refresh').on('click', () => loadHelperModels(true));
    $('#fm_helper_filter').on('input', () => loadHelperModels(false));
    $('#fm_helper_test').on('click', async () => {
        const $out = $('#fm_helper_out').text('Asking the helper…');
        try {
            const r = await bridge('/helper/voice-tags', { method: 'POST', body: { text: 'Wait, you actually came all this way just to see me?', char: 'Test', model: s.helperModel || undefined } });
            $out.text(r.error ? `Error: ${r.error}` : `Helper says: ${r.text}`);
        } catch (err) {
            $out.text(`Error: ${err?.message || err}`);
        }
    });
    $('#fm_usage').text(`Today: ${usage().images} images, ${usage().videos} videos`);
}

// ---------------------------------------------------------------------------
// Buttons: per message and on the character card
// ---------------------------------------------------------------------------

const MES_BUTTONS = '<div title="Media Studio: generate image" class="mes_button fm_mes_image fa-solid fa-image"></div>'
    + '<div title="Media Studio: generate video" class="mes_button fm_mes_video fa-solid fa-film"></div>';

function addMessageButtons() {
    $('#message_template .mes_buttons .extraMesButtons').prepend(MES_BUTTONS);
    $('#chat .mes .extraMesButtons').each(function () {
        if (!$(this).find('.fm_mes_image').length) $(this).prepend(MES_BUTTONS);
    });
    $(document).on('click', '.fm_mes_image', function () {
        generateImage(Number($(this).closest('.mes').attr('mesid')), { origin: 'manual' });
    });
    $(document).on('click', '.fm_mes_video', async function () {
        const id = Number($(this).closest('.mes').attr('mesid'));
        const prompts = await promptsForVideo(id);
        generateVideo(id, { ...prompts, origin: 'manual' });
    });
}

function refreshCharButton() {
    const c = ctx();
    const editing = c.characterId !== undefined && c.characterId !== null && c.menuType !== 'create';
    $('#fm_char_button').toggleClass('displayNone', !editing);
}

const galleryDeps = {
    getCharMedia,
    setCharMedia,
    async loraStatus(chid) {
        const ch = ctx().characters[chid];
        try {
            const r = await bridge('/media/lora-status', { method: 'POST', body: { names: [ch.name], character_prompts: { [ch.name]: sdPrompt(ch) } }, timeout: 15000 });
            return !!r.characters?.[ch.name];
        } catch {
            return null;
        }
    },
    async generateAvatar(chid, prompt) {
        const s = settings();
        const ch = ctx().characters[chid];
        const res = await bridge('/media/image', {
            method: 'POST',
            body: { prompt, character_name: ch.name, character_prompts: { [ch.name]: sdPrompt(ch) }, visible_characters: [ch.name],
                mode: 'lora', avatars: [], width: 832, height: 1216, steps: Number(s.steps), cfg_scale: Number(s.cfg) },
        });
        const b64 = res.images?.[0];
        if (!b64) throw new Error('The bridge returned no image');
        return await saveFile(b64, imageFormat(b64), ch.name, `avatar_${Date.now()}`);
    },
};

// ---------------------------------------------------------------------------
// Slash commands
// ---------------------------------------------------------------------------

function lastCharacterMessageId() {
    const chat = ctx().chat;
    for (let i = chat.length - 1; i >= 0; i--) if (!chat[i].is_user && !chat[i].is_system) return i;
    return -1;
}

function registerSlashCommands() {
    const { SlashCommandParser, SlashCommand, SlashCommandArgument, ARGUMENT_TYPE } = ctx();
    if (!SlashCommandParser?.addCommandObject) return;
    SlashCommandParser.addCommandObject(SlashCommand.fromProps({
        name: 'fmimage',
        callback: async (_a, prompt) => { const id = lastCharacterMessageId(); if (id >= 0) await generateImage(id, { prompt: String(prompt || ''), origin: 'manual' }); return ''; },
        unnamedArgumentList: [SlashCommandArgument.fromProps({ description: 'optional scene prompt', typeList: [ARGUMENT_TYPE.STRING] })],
        helpString: 'Media Studio: generate an image for the last character message.',
    }));
    SlashCommandParser.addCommandObject(SlashCommand.fromProps({
        name: 'fmvideo',
        callback: async (_a, motion) => {
            const id = lastCharacterMessageId();
            if (id < 0) return '';
            const prompts = motion ? { motion: String(motion) } : await promptsForVideo(id);
            await generateVideo(id, { ...prompts, origin: 'manual' });
            return '';
        },
        unnamedArgumentList: [SlashCommandArgument.fromProps({ description: 'optional motion prompt', typeList: [ARGUMENT_TYPE.STRING] })],
        helpString: 'Media Studio: generate a video for the last character message (uses its image as the first frame).',
    }));
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

jQuery(async () => {
    const c = ctx();
    const et = c.eventTypes;
    settings();
    migrateLegacyConfig();

    const html = await (await fetch(new URL('./settings.html', import.meta.url))).text();
    $('#extensions_settings2').append(html);
    bindSettings();

    addMessageButtons();
    $('.form_create_bottom_buttons_block').first().append('<div id="fm_char_button" class="menu_button fa-solid fa-images displayNone" title="Media gallery & avatars"></div>');
    $('#fm_char_button').on('click', () => openCharacterMedia(ctx().characterId, galleryDeps));
    refreshCharButton();

    c.eventSource.on(et.MESSAGE_RECEIVED, (id, type) => onMessageReceived(id, type));
    c.eventSource.on(et.MESSAGE_SENT, () => { noteArmed = true; });
    c.eventSource.on(et.CHAT_CHANGED, () => {
        ctx().setExtensionPrompt(NOTE_KEY, '', 1, 1);
        noteArmed = false;
        refreshCharButton();
    });
    c.eventSource.on(et.CHARACTER_EDITOR_OPENED, refreshCharButton);

    registerSlashCommands();
    await refreshBridgeStatus();
    loadHelperModels();

    globalThis.MediaStudio = { settings, generateImage, generateVideo, onMessageReceived, getCharMedia, setCharMedia, AVATAR_SLOTS };
    console.log(TAG, 'loaded');
});
