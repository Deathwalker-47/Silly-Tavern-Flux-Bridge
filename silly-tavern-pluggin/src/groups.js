// Group chats on native SillyTavern groups. The group runs in Manual reply mode, so
// SillyTavern posts your message and stops; the director then picks who acts until it's
// your turn. One normal card is the group's Setting card: it holds the world and the side
// characters, stays muted so SillyTavern never picks it, and /trigger can still choose it.

import { bridge } from './bridge.js';
import { currentGroup, history, settingCard, settingCardText } from './context.js';
import { track } from './telemetry.js';

const ctx = () => SillyTavern.getContext();
const KEY = 'flux_media';
const PROMPT_KEY = 'fm_group_turn';
const MANUAL = 2;                        // group_activation_strategy.MANUAL

const turn = { running: false, stop: false, pending: false };
let afterRound = async () => {};

const DEFAULT_PROFILE = Object.freeze({ setting_avatar: '', media_mode: 'director', reply_style: 'short', loop: false });

export function groupProfile(id = ctx().groupId) {
    if (!id) return null;
    const store = (ctx().extensionSettings[KEY].groups ??= {});
    return { ...DEFAULT_PROFILE, ...(store[id] || {}) };
}

async function saveGroup(group) {
    const c = ctx();
    const res = await fetch('/api/groups/edit', { method: 'POST', headers: c.getRequestHeaders(), body: JSON.stringify(group) });
    if (!res.ok) throw new Error(`SillyTavern rejected the group update (${res.status})`);
}

/** Saving the profile also puts the group in Manual mode and mutes the Setting card. */
export async function setGroupProfile(id, patch) {
    const c = ctx();
    const store = (c.extensionSettings[KEY].groups ??= {});
    const next = { ...groupProfile(id), ...patch };
    store[id] = next;
    c.saveSettingsDebounced();
    const group = c.groups.find(g => g.id === id);
    if (!group) return next;
    const muted = new Set(group.disabled_members || []);
    if (patch.setting_avatar !== undefined) {
        const previous = groupProfile(id).setting_avatar;
        if (previous && previous !== next.setting_avatar) muted.delete(previous);
    }
    if (next.setting_avatar) muted.add(next.setting_avatar);
    group.disabled_members = [...muted];
    if (next.loop) group.activation_strategy = MANUAL;
    await saveGroup(group);
    return next;
}

export const loopActive = () => {
    const group = currentGroup();
    return !!group && groupProfile(group.id)?.loop;
};

function presentMembers(profile) {
    const c = ctx();
    return (currentGroup()?.members ?? []).filter(a => a !== profile.setting_avatar)
        .map(a => c.characters.find(ch => ch.avatar === a)).filter(Boolean);
}

function membersForDirector(profile) {
    const c = ctx();
    const lastSpoke = (name) => {
        for (let i = c.chat.length - 1, back = 0; i >= 0; i--, back++) if (c.chat[i].name === name) return back;
        return null;
    };
    const entry = (ch) => ({ name: ch.name, brief: String(ch.description || '').slice(0, 300), last_spoke: lastSpoke(ch.name) });
    const members = presentMembers(profile).map(entry);
    const card = settingCard(profile);
    if (card) members.push({ ...entry(card), setting_card: true });
    return members;
}

function turnContext(profile, spoken) {
    const c = ctx();
    return {
        chat_id: c.getCurrentChatId(),
        history: history(c.chat.length),
        summary: [...c.chat].reverse().find(m => m.extra?.memory)?.extra.memory || '',
        setting: settingCardText(profile),
        members: membersForDirector(profile),
        spoken_this_turn: spoken,
        user_name: c.name1,
    };
}

function setTurnPrompt(next, profile) {
    const lines = [
        settingCardText(profile),
        `Present: ${presentMembers(profile).map(m => m.name).join(', ')}.`,
        next.long || profile.reply_style === 'long' ? 'This moment deserves a fuller reply.' : 'Keep this reply short: 1-3 sentences or a brief action.',
        next.beat ? `Direction: ${next.beat}` : '',
    ];
    ctx().setExtensionPrompt(PROMPT_KEY, lines.filter(Boolean).join('\n'), 1, 1);
}

const clearTurnPrompt = () => ctx().setExtensionPrompt(PROMPT_KEY, '', 1, 1);

/** /trigger clears the input box, so typing ends the round before it can erase anything. */
const userIsTyping = () => String($('#send_textarea').val() || '').trim().length > 0;

const quote = (s) => `"${String(s).replace(/"/g, '\\"')}"`;

export async function runGroupTurn() {
    if (turn.running) return;
    const profile = groupProfile();
    if (!profile?.loop) return;
    Object.assign(turn, { running: true, stop: false });
    const limit = Math.max(5, presentMembers(profile).length);
    const spoken = [];
    let stoppedBy = 'limit';
    try {
        while (spoken.length < limit) {
            if (turn.stop) { stoppedBy = 'stop_button'; break; }
            if (userIsTyping()) { stoppedBy = 'typing'; break; }
            const next = await bridge('/director/next-speaker', { method: 'POST', body: turnContext(profile, spoken), timeout: 60000 });
            if (next.speaker === 'user') { stoppedBy = 'user_turn'; break; }
            setTurnPrompt(next, profile);
            await ctx().executeSlashCommandsWithOptions(`/trigger await=true ${quote(next.speaker)}`);
            clearTurnPrompt();
            spoken.push(next.speaker);
        }
    } catch (err) {
        stoppedBy = 'error';
        console.error('[MediaStudio]', err);
        toastr.warning(`The director couldn't pick a speaker: ${err?.message || err}`, 'Media Studio');
    } finally {
        clearTurnPrompt();
        turn.running = false;
        track('group_round', { messages: spoken.length, stopped_by: stoppedBy, limit });
        if (spoken.length) await afterRound(profile);     // one media decision for the whole round
    }
}

export function initGroups({ onRoundEnd }) {
    const c = ctx();
    afterRound = onRoundEnd;
    const et = c.eventTypes;
    c.eventSource.on(et.GENERATION_STOPPED, () => { turn.stop = true; });
    c.eventSource.on(et.MESSAGE_SENT, () => { if (loopActive()) turn.pending = true; });
    // In Manual mode the wrapper finishes right after posting your message.
    c.eventSource.on(et.GROUP_WRAPPER_FINISHED, () => {
        if (!turn.pending) return;
        turn.pending = false;
        setTimeout(runGroupTurn, 0);
    });
}
