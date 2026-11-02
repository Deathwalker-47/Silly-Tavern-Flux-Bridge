// Media Studio for the Flux LoRA Bridge.
// Characters send photos and videos in chat. The bridge's director decides what's in the
// shot and writes the prompt; jobs run on the bridge, so they survive a reload or a locked
// phone. Character and persona panels hold references, Appearance and the rated gallery.

import { bridge, bridgeBase, cachedCapabilities, getCapabilities } from './src/bridge.js';
import { urlToAsset } from './src/assets.js';
import { characterOf, currentGroup, directorContext, history, sdPrompt } from './src/context.js';
import { groupProfile, initGroups, loopActive, setGroupProfile } from './src/groups.js';
import { configureJobs, resumeOpenJobs, runDetached, runJob } from './src/jobs.js';
import { openMediaPanel } from './src/media-panel.js';
import { charKey, getCharProfile, getPersonaProfile, personaId, personaName, setCharProfile, setPersonaProfile } from './src/profile.js';
import { messageMedia, messagesSinceMedia } from './src/store.js';
import { track } from './src/telemetry.js';

const ctx = () => SillyTavern.getContext();
const KEY = 'flux_media';          // extension_settings key and character card field
const NOTE_KEY = 'FLUX_MEDIA_NOTE';
const TAG = '[MediaStudio]';
const LOW_DIRECTOR_BALANCE = 1;    // USD. The director runs on NanoGPT: at zero, all media stops.

const DEFAULTS = Object.freeze({
    bridgeUrl: '',
    bridgeToken: '',
    autoMedia: 'director',          // off | director | every_turn
    allowAutoImages: true,
    allowAutoVideos: true,
    imageCooldown: 3,
    videoCooldown: 8,
    dailyImageCap: 60,              // 0 = unlimited
    dailyVideoCap: 10,              // 0 = unlimited
    image: { provider: 'runware', family: 'flux-lora', variant: '' },
    video: { provider: 'sogni', family: 'minimax-h3', variant: '', mode: 'reference' },
    sogniNetwork: '',
    videoAudio: false,              // only for models with an audio switch
    aspect: 'director',             // director | 9:16 | 3:4 | 16:9
    notifyCharacter: true,
    personas: {},
    groups: {},
    debug: false,
});
const RETIRED = ['imageMode', 'imageSize', 'steps', 'cfg', 'helperModel', 'videoProvider', 'videoFamily', 'videoVariant',
    'videoDuration', 'videoResolution', 'useTwoImages', 'directorModel'];
const SKIP_TYPES = new Set(['first_message', 'swipe', 'continue', 'impersonate', 'quiet']);

function settings() {
    const all = ctx().extensionSettings;
    all[KEY] ??= {};
    const s = all[KEY];
    for (const [k, v] of Object.entries(DEFAULTS)) if (s[k] === undefined) s[k] = structuredClone(v);
    for (const k of RETIRED) delete s[k];
    s.usage ??= { date: '', images: 0, videos: 0 };
    return s;
}
const save = () => ctx().saveSettingsDebounced();
const log = (...a) => settings().debug && console.log(TAG, ...a);
const esc = (s) => $('<span>').text(String(s ?? '')).html();

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
// Usage limits and the note for the main LLM
// ---------------------------------------------------------------------------

function usage() {
    const s = settings();
    const today = new Date().toISOString().slice(0, 10);
    if (s.usage.date !== today) s.usage = { date: today, images: 0, videos: 0 };
    return s.usage;
}

const underCap = (used, cap) => Number(cap) === 0 || used < Number(cap);

let noteArmed = false;

function noteForCharacter(job) {
    if (!settings().notifyCharacter) return;
    const c = ctx();
    const kind = job.kind === 'video' ? 'short video' : 'photo';
    const text = job.caption || job.prompt;
    const note = `[${job.folder || c.name2} sent ${c.name1} a ${kind}${text ? `: ${String(text).slice(0, 200)}` : ''}]`;
    c.setExtensionPrompt(NOTE_KEY, note, 1, 1, false, 0); // in-chat, depth 1, system role
    noteArmed = false;
}

// ---------------------------------------------------------------------------
// Generating
// ---------------------------------------------------------------------------

const warnedAge = new Set();

/** A character with no age can't be generated. One toast, with a way to fix it. */
async function ageMissing(chid, char, origin) {
    if (chid === undefined || !char) return false;
    const profile = await getCharProfile(chid);
    if (Number(profile.appearance?.age) >= 18) return false;
    track('auto_skip', { reason: 'no_age', origin });
    if (origin === 'manual' || !warnedAge.has(char.avatar)) {
        warnedAge.add(char.avatar);
        toastr.warning(`${char.name} needs an age of 18 or over before media can be generated. Tap to add it.`, 'Media Studio', {
            timeOut: 12000, onclick: () => openCharacterPanel(chid, 'appearance'),
        });
    }
    return true;
}

async function jobBody(kind, messageId, origin, request) {
    const s = settings();
    const c = ctx();
    const msg = c.chat[messageId];
    const { char } = characterOf(msg);
    const context = await directorContext(messageId, kind, groupProfile());
    if (request) context.user_request = request;
    const body = {
        kind, selection: s[kind], image_selection: s.image, context, aspect: s.aspect, origin,
        folder: char?.name || 'MediaStudio', group_id: c.groupId || null,
        sogni_network: s.sogniNetwork || undefined, audio: !!s.videoAudio,
    };
    if (kind === 'video') {
        const images = messageMedia(msg).filter(m => m.type === 'image');
        const twoFrames = s.video.mode === 'first_last';
        if (images.length) body.message_image_asset = await urlToAsset(images[twoFrames ? 0 : images.length - 1].url);
        if (twoFrames && images.length > 1) body.last_image_asset = await urlToAsset(images[images.length - 1].url);
    }
    return body;
}

const busy = new Set(); // "image:12" / "video:12" so a message isn't sent to the bridge twice at once

async function generate(kind, messageId, { origin = 'manual', request = '' } = {}) {
    const key = `${kind}:${messageId}`;
    if (busy.has(key)) return;
    busy.add(key);
    try {
        const { chid, char } = characterOf(ctx().chat[messageId]);
        if (await ageMissing(chid, char, origin)) return;
        await runJob(await jobBody(kind, messageId, origin, request), messageId);
    } catch (err) {
        console.error(TAG, err);
        toastr.error(err?.message || String(err), `Media Studio: ${kind} failed`);
    } finally {
        busy.delete(key);
    }
}

// ---------------------------------------------------------------------------
// Auto media after each character message, or once per group round
// ---------------------------------------------------------------------------

let warnedDirector = false;

async function autoMedia(messageId, mode) {
    const s = settings();
    const c = ctx();
    const msg = c.chat[messageId];
    if (mode === 'off' || !msg) return;
    const { chid, char } = characterOf(msg);
    const profile = await getCharProfile(chid);
    if (profile.auto === false) return track('auto_skip', { reason: 'character_off' });
    if (await ageMissing(chid, char, 'auto')) return;
    const u = usage();

    if (mode === 'every_turn') {
        if (!underCap(u.images, s.dailyImageCap)) return track('auto_skip', { reason: 'image_cap' });
        return generate('image', messageId, { origin: 'auto' });
    }

    // Cooldowns and caps are enforced here, not left to the LLM.
    const providers = cachedCapabilities()?.video?.providers ?? [];
    const videoReady = !!providers.find(p => p.id === s.video.provider)?.configured;
    const allowImage = s.allowAutoImages && messagesSinceMedia('image') >= Number(s.imageCooldown) && underCap(u.images, s.dailyImageCap);
    const allowVideo = s.allowAutoVideos && videoReady && messagesSinceMedia('video') >= Number(s.videoCooldown) && underCap(u.videos, s.dailyVideoCap);
    if (!allowImage && !allowVideo) return track('auto_skip', { reason: 'cooldown_or_cap' });
    try {
        const d = await bridge('/director/decide', {
            method: 'POST',
            body: {
                chat_id: c.getCurrentChatId(), reply: String(msg.mes || ''), history: history(messageId), char: char?.name || c.name2, user: c.name1,
                allow_image: allowImage, allow_video: allowVideo,
            },
        });
        log('director', d);
        if (d.action === 'image' || d.action === 'video') generate(d.action, messageId, { origin: 'auto' });
    } catch (err) {
        console.warn(TAG, 'director failed', err);
        if (!warnedDirector) {
            warnedDirector = true;
            toastr.warning(`The director is unavailable, so auto media is paused for now: ${err?.message || err}`, 'Media Studio');
        }
    }
}

function onMessageReceived(messageId, type) {
    if (noteArmed) {
        ctx().setExtensionPrompt(NOTE_KEY, '', 1, 1);
        noteArmed = false;
    }
    const s = settings();
    if (s.autoMedia === 'off' || SKIP_TYPES.has(type) || loopActive()) return;
    const msg = ctx().chat[messageId];
    if (!msg || msg.is_user || msg.is_system || msg.extra?.type === 'narrator' || String(msg.mes || '').length < 20) return;
    autoMedia(messageId, s.autoMedia);
}

function lastCharacterMessageId() {
    const chat = ctx().chat;
    for (let i = chat.length - 1; i >= 0; i--) if (!chat[i].is_user && !chat[i].is_system) return i;
    return -1;
}

async function onRoundEnd(profile) {
    const id = lastCharacterMessageId();
    if (id >= 0 && settings().autoMedia !== 'off') await autoMedia(id, profile.media_mode);
}

// ---------------------------------------------------------------------------
// Settings: the image and video pickers
// ---------------------------------------------------------------------------

const routeInfo = (kind) => {
    const s = settings();
    const caps = cachedCapabilities()?.[kind];
    const provider = caps?.providers?.find(p => p.id === s[kind].provider);
    const family = provider?.families?.find(f => f.id === s[kind].family);
    const variant = family?.variants?.find(v => v.id === s[kind].variant) || family?.variants?.[0];
    return { caps, provider, family, variant };
};

/** A saved route the bridge dropped falls back to the default, then to anything configured. */
function keepRouteValid(kind) {
    const s = settings();
    const { caps, provider, family, variant } = routeInfo(kind);
    if (!caps) return;
    if (!provider?.configured || !family || !variant) {
        const fallback = DEFAULTS[kind];
        const usable = (p, f) => p?.configured && p.families.find(x => x.id === f);
        const home = caps.providers.find(p => p.id === fallback.provider);
        const first = caps.providers.find(p => p.configured && p.id !== 'mock' && p.families.length) || caps.providers.find(p => p.configured && p.families.length);
        const chosen = usable(home, fallback.family) ? home : first;
        if (!chosen) return;
        const familyId = chosen.families.find(f => f.id === fallback.family)?.id || chosen.families[0].id;
        s[kind] = { ...s[kind], provider: chosen.id, family: familyId, variant: '' };
    }
    if (s[kind].variant && !routeInfo(kind).family.variants.some(v => v.id === s[kind].variant)) s[kind].variant = '';
    if (kind === 'video') {
        const modes = routeInfo('video').variant.modes;
        if (!modes.includes(s.video.mode)) s.video.mode = modes[0];
    }
    save();
}

function renderPicker(kind) {
    const s = settings();
    const { caps, provider, family, variant } = routeInfo(kind);
    if (!caps) return;
    $(`#fm_${kind}_provider`).html(caps.providers.filter(p => p.families.length).map(p =>
        `<option value="${p.id}" ${p.id === s[kind].provider ? 'selected' : ''} ${p.configured ? '' : 'disabled'}>${esc(p.label)}${p.configured ? '' : ' (no key on bridge)'}</option>`).join(''));
    $(`#fm_${kind}_family`).html((provider?.families || []).map(f =>
        `<option value="${esc(f.id)}" ${f.id === s[kind].family ? 'selected' : ''}>${esc(f.label)}</option>`).join('') || '<option value="">No models</option>');
    $(`#fm_${kind}_variant`).html((family?.variants || []).map(v =>
        `<option value="${esc(v.id)}" ${v.id === variant?.id ? 'selected' : ''}>${esc(v.label)}</option>`).join(''));
    $(`#fm_${kind}_variant_row`).toggleClass('displayNone', (family?.variants?.length || 0) < 2);
    $(`#fm_${kind}_hint`).text(variant?.hint || '');
    if (kind === 'video') {
        const modes = cachedCapabilities().modes;
        $('#fm_video_modes').html((variant?.modes || []).map(m =>
            `<label class="checkbox_label"><input type="radio" name="fm_video_mode" value="${m}" ${m === s.video.mode ? 'checked' : ''}> ${esc(modes[m].label)}</label>`).join(''));
        $('#fm_video_mode_text').text(modes[s.video.mode]?.text || '');
        $('#fm_sogni_network_row').toggleClass('displayNone', s.video.provider !== 'sogni');
        $('#fm_sogni_network option[value=""]').text(`Bridge default (${cachedCapabilities().video.sogni_network || 'fast'})`);
        $('#fm_video_audio_row').toggleClass('displayNone', !variant?.audio_switch);
    }
}

function renderPickers() {
    keepRouteValid('image');
    keepRouteValid('video');
    renderPicker('image');
    renderPicker('video');
}

async function refreshBridgeStatus({ quiet = true } = {}) {
    const $s = $('#fm_status');
    try {
        const caps = await getCapabilities(true);
        const ready = caps.video.providers.filter(p => p.configured).map(p => p.label.split(' (')[0]);
        $s.removeClass('fm-bad').addClass('fm-ok').text(`Connected · ${ready.join(', ') || 'no providers'} · director: ${caps.helper.configured ? caps.helper.default_model : 'not set'}`);
        renderPickers();
        refreshBalances();
        if (!quiet) toastr.success('Bridge connected.', 'Media Studio');
    } catch (err) {
        $s.removeClass('fm-ok').addClass('fm-bad').text(err?.message || 'Not connected');
        if (!quiet) toastr.error(err?.message || String(err), 'Media Studio');
    }
}

async function refreshBalances() {
    try {
        const { balances } = await bridge('/media/balances', { timeout: 30000 });
        const text = balances.filter(b => b.amount != null).map((b) => {
            const name = b.label.split(' (')[0];
            return b.unit === 'USD' ? `${name} $${b.amount.toFixed(2)}` : `${name} ${Math.round(b.amount)} ${b.unit}`;
        });
        $('#fm_balances').text(text.join(' · '));
        const director = balances.find(b => b.id === 'nanogpt');
        if (director?.amount != null && director.amount < LOW_DIRECTOR_BALANCE) {
            toastr.warning(`NanoGPT is down to $${director.amount.toFixed(2)}. The director runs there, so at zero all media stops.`, 'Media Studio');
        }
    } catch (err) {
        log('balances', err);
    }
}

// ---------------------------------------------------------------------------
// Settings: test a route, read the pipeline report
// ---------------------------------------------------------------------------

async function testRoute() {
    const kind = $('#fm_test_kind').val();
    const prompt = String($('#fm_test_prompt').val() || '').trim();
    const s = settings();
    const route = s[kind];
    const $out = $('#fm_test_out').text('Running at the lowest setting…');
    $('#fm_test_verdict').addClass('displayNone');
    try {
        const job = await runDetached({ kind, selection: route, prompt, motion_prompt: kind === 'video' ? prompt : '', image_selection: s.image, audio: false }, '/media/routes/test');
        const url = `${bridgeBase()}${job.file_url}`;
        $out.html(kind === 'video' ? `<video src="${esc(url)}" controls class="fm-test-media"></video>` : `<img src="${esc(url)}" class="fm-test-media" alt="">`);
        const { variant } = routeInfo(kind);
        $('#fm_test_verdict').removeClass('displayNone').data('route', { provider: route.provider, model: variant.id });
    } catch (err) {
        $out.text(err?.detail || err?.message || String(err));
    }
}

async function saveVerdict(status) {
    const { provider, model } = $('#fm_test_verdict').data('route');
    await bridge(`/media/routes/${provider}/${model}`, { method: 'PUT', body: { status } });
    $('#fm_test_verdict').addClass('displayNone');
    toastr.success(status === 'allowed' ? 'Route passed.' : 'Route failed and hidden from the picker.', 'Media Studio');
    refreshBridgeStatus();
}

const money = (r) => [r.total_usd ? `$${r.total_usd.toFixed(2)}` : '', r.total_credits ? `${r.total_credits} cr` : ''].filter(Boolean).join(' + ') || '-';

async function openReport() {
    const c = ctx();
    let r;
    try {
        r = await bridge('/media/insights?days=30', { timeout: 30000 });
    } catch (err) {
        return toastr.error(err?.message || String(err), 'Media Studio');
    }
    const routes = r.routes.map(x => `<tr><td>${esc(x.model)}<br><small>${esc(x.provider)} · ${esc(x.mode || '')}</small></td><td>${x.done}/${x.runs}</td><td>${x.median_seconds ?? '-'}s</td><td>${money(x)}</td><td>${x.rating ?? '-'}</td></tr>`).join('');
    const notes = r.suggestions.map(n => `<li>${esc(n)}</li>`).join('') || '<li>Not enough runs yet. Rate your generations and this fills in.</li>';
    const tables = Object.entries(r.tables).map(([name, rows]) => rows.length ? `<h4>By ${esc(name)}</h4><table class="fm-report"><tr><th></th><th>Runs</th><th>Rated</th><th>Rating</th></tr>${rows.slice(0, 8).map(x => `<tr><td>${esc(x.value)}</td><td>${x.runs}</td><td>${x.rated}</td><td>${x.rating ?? '-'}</td></tr>`).join('')}</table>` : '').join('');
    const slow = r.steps.filter(x => x.median_ms).sort((a, b) => b.median_ms - a.median_ms).slice(0, 8)
        .map(x => `<tr><td>${esc(x.stage)}/${esc(x.event)}</td><td>${x.count}</td><td>${(x.median_ms / 1000).toFixed(1)}s</td><td>${x.failure_rate == null ? '-' : `${Math.round(x.failure_rate * 100)}%`}</td></tr>`).join('');
    const $root = $(`<div class="fm-popup">
        <h3>Pipeline report: last 30 days</h3>
        <p>${r.totals.done} of ${r.totals.generations} generations finished · $${r.totals.usd.toFixed(2)} + ${r.totals.credits} cr · average rating ${r.totals.rating ?? '-'}</p>
        <h4>What to look at</h4><ul>${notes}</ul>
        <h4>Routes</h4><table class="fm-report"><tr><th>Model</th><th>Done</th><th>Median</th><th>Spent</th><th>Rating</th></tr>${routes}</table>
        ${tables}
        <h4>Slowest steps</h4><table class="fm-report"><tr><th>Step</th><th>Count</th><th>Median</th><th>Failed</th></tr>${slow}</table>
    </div>`);
    await c.callGenericPopup($root, c.POPUP_TYPE.TEXT, '', { okButton: 'Close', wide: true, large: true, allowVerticalScrolling: true, leftAlign: true });
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
    });
    for (const kind of ['image', 'video']) {
        const change = (field) => function () {
            s[kind][field] = this.value;
            if (field === 'provider') { s[kind].family = routeInfo(kind).provider?.families?.[0]?.id || ''; s[kind].variant = ''; }
            if (field === 'family') s[kind].variant = '';
            keepRouteValid(kind);
            renderPicker(kind);
        };
        $(`#fm_${kind}_provider`).on('change', change('provider'));
        $(`#fm_${kind}_family`).on('change', change('family'));
        $(`#fm_${kind}_variant`).on('change', change('variant'));
    }
    $root.on('change', 'input[name="fm_video_mode"]', function () {
        s.video.mode = this.value;
        save();
        renderPicker('video');
    });
    $('#fm_bridge_url, #fm_bridge_token').on('change', () => refreshBridgeStatus({ quiet: false }));
    $('#fm_test').on('click', () => refreshBridgeStatus({ quiet: false }));
    $('#fm_test_run').on('click', testRoute);
    $('#fm_test_pass').on('click', () => saveVerdict('allowed'));
    $('#fm_test_fail').on('click', () => saveVerdict('banned'));
    $('#fm_report').on('click', openReport);
    $('#fm_usage').text(`Today: ${usage().images} images, ${usage().videos} videos`);
}

// ---------------------------------------------------------------------------
// Panels and buttons
// ---------------------------------------------------------------------------

function openCharacterPanel(chid, tab) {
    const char = ctx().characters[chid];
    if (!char) return;
    return openMediaPanel({
        name: char.name, key: charKey(char), folder: char.name, sdPrompt: sdPrompt(char), tab,
        load: () => getCharProfile(chid), save: patch => setCharProfile(chid, patch),
    });
}

async function openPersonaPanel() {
    const id = await personaId();
    const name = personaName(id);
    return openMediaPanel({
        name, key: `persona:${id}`, folder: name,
        load: async () => getPersonaProfile(id), save: async patch => setPersonaProfile(id, patch),
    });
}

async function openGroupPanel() {
    const c = ctx();
    const group = currentGroup();
    if (!group) return toastr.info('Open a group chat first.', 'Media Studio');
    const profile = groupProfile(group.id);
    const options = group.members.map((avatar) => {
        const ch = c.characters.find(x => x.avatar === avatar);
        return ch ? `<option value="${esc(avatar)}" ${avatar === profile.setting_avatar ? 'selected' : ''}>${esc(ch.name)}</option>` : '';
    }).join('');
    const $root = $(`<div class="fm-popup">
        <h3><i class="fa-solid fa-users"></i> ${esc(group.name)}</h3>
        <label class="checkbox_label"><input type="checkbox" class="fm-g-loop" ${profile.loop ? 'checked' : ''}> The director runs this group's turns (sets Manual reply mode)</label>
        <label>Setting card <small>(the world, rules and side characters; stays muted)</small>
            <select class="text_pole fm-g-setting"><option value="">None</option>${options}</select></label>
        <label>Replies <select class="text_pole fm-g-style">
            <option value="short" ${profile.reply_style === 'short' ? 'selected' : ''}>Short by default</option>
            <option value="long" ${profile.reply_style === 'long' ? 'selected' : ''}>Long</option></select></label>
        <label>Media after each round <select class="text_pole fm-g-media">
            <option value="director" ${profile.media_mode === 'director' ? 'selected' : ''}>The director decides</option>
            <option value="every_turn" ${profile.media_mode === 'every_turn' ? 'selected' : ''}>Every round: an image</option>
            <option value="off" ${profile.media_mode === 'off' ? 'selected' : ''}>Off (message buttons only)</option></select></label>
        <small class="fm-hint">Everyone able to act does, then it's your turn. A round stops at 5 messages, or the number of characters if that's higher. Typing ends the round.</small>
    </div>`);
    const update = (patch) => setGroupProfile(group.id, patch).catch(err => toastr.error(err?.message || String(err), 'Media Studio'));
    $root.on('change', '.fm-g-loop', function () { update({ loop: this.checked }); });
    $root.on('change', '.fm-g-setting', function () { update({ setting_avatar: this.value }); });
    $root.on('change', '.fm-g-style', function () { update({ reply_style: this.value }); });
    $root.on('change', '.fm-g-media', function () { update({ media_mode: this.value }); });
    await c.callGenericPopup($root, c.POPUP_TYPE.TEXT, '', { okButton: 'Close', wide: false, leftAlign: true });
}

const MES_BUTTONS = '<div title="Media Studio: generate image" class="mes_button fm_mes_image fa-solid fa-image"></div>'
    + '<div title="Media Studio: generate video" class="mes_button fm_mes_video fa-solid fa-film"></div>';

function addMessageButtons() {
    $('#message_template .mes_buttons .extraMesButtons').prepend(MES_BUTTONS);
    $('#chat .mes .extraMesButtons').each(function () {
        if (!$(this).find('.fm_mes_image').length) $(this).prepend(MES_BUTTONS);
    });
    $(document).on('click', '.fm_mes_image', function () { generate('image', Number($(this).closest('.mes').attr('mesid'))); });
    $(document).on('click', '.fm_mes_video', function () { generate('video', Number($(this).closest('.mes').attr('mesid'))); });
}

async function refreshCharButton() {
    const c = ctx();
    const editing = c.characterId !== undefined && c.characterId !== null && c.menuType !== 'create';
    const $button = $('#fm_char_button').toggleClass('displayNone', !editing);
    if (!editing || !c.characters[c.characterId]) return;
    try {
        const key = encodeURIComponent(charKey(c.characters[c.characterId]));
        const { unrated } = await bridge(`/media/generations?person=${key}&limit=1`, { timeout: 10000 });
        $button.toggleClass('fm-has-unrated', unrated > 0).attr('title', unrated ? `Media: ${unrated} unrated` : 'Media: references, appearance, gallery');
    } catch { /* the dot is a convenience */ }
}

function registerSlashCommands() {
    const { SlashCommandParser, SlashCommand, SlashCommandArgument, ARGUMENT_TYPE } = ctx();
    if (!SlashCommandParser?.addCommandObject) return;
    const command = (name, kind, help) => SlashCommandParser.addCommandObject(SlashCommand.fromProps({
        name,
        callback: async (_a, request) => {
            const id = lastCharacterMessageId();
            if (id >= 0) await generate(kind, id, { request: String(request || '') });
            return '';
        },
        unnamedArgumentList: [SlashCommandArgument.fromProps({ description: 'optional request for the director', typeList: [ARGUMENT_TYPE.STRING] })],
        helpString: help,
    }));
    command('fmimage', 'image', 'Media Studio: generate an image for the last character message.');
    command('fmvideo', 'video', 'Media Studio: generate a video for the last character message.');
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
    $('.form_create_bottom_buttons_block').first().append('<div id="fm_char_button" class="menu_button fa-solid fa-images displayNone" title="Media: references, appearance, gallery"></div>');
    $('#fm_char_button').on('click', () => openCharacterPanel(ctx().characterId));
    $('.persona_controls_buttons_block').first().append('<div id="fm_persona_button" class="menu_button fa-solid fa-images" title="Media: references, appearance, gallery"></div>');
    $('#fm_persona_button').on('click', openPersonaPanel);
    $('#rm_group_buttons').append('<div id="fm_group_button" class="menu_button fa-solid fa-clapperboard" title="Media Studio: director and Setting card"></div>');
    $('#fm_group_button').on('click', openGroupPanel);
    refreshCharButton();

    configureJobs({
        counted(kind) { usage()[kind]++; save(); },
        note: noteForCharacter,
    });
    initGroups({ onRoundEnd });

    c.eventSource.on(et.MESSAGE_RECEIVED, onMessageReceived);
    c.eventSource.on(et.MESSAGE_SENT, () => { noteArmed = true; });
    c.eventSource.on(et.MESSAGE_SWIPED, () => track('swiped'));
    c.eventSource.on(et.MESSAGE_DELETED, () => track('message_deleted'));
    c.eventSource.on(et.CHAT_CHANGED, () => {
        ctx().setExtensionPrompt(NOTE_KEY, '', 1, 1);
        noteArmed = false;
        refreshCharButton();
        resumeOpenJobs();
    });
    c.eventSource.on(et.CHARACTER_EDITOR_OPENED, refreshCharButton);

    registerSlashCommands();
    await refreshBridgeStatus();
    resumeOpenJobs();

    globalThis.MediaStudio = { settings, generate, onMessageReceived, resumeOpenJobs };
    console.log(TAG, 'loaded');
});
