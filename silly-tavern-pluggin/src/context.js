// Everything the director reads, raw. There are no length limits: reply is never cut.

import { charKey, ensureAssets, getCharProfile, getPersonaProfile, personaId, personaName, personForContext, setCharProfile, setPersonaProfile } from './profile.js';

const ctx = () => SillyTavern.getContext();

export const sdPrompt = (char) => char?.data?.extensions?.sd_character_prompt || char?.data?.extensions?.sd_api_prompt || '';

/** The character who wrote a message (group-aware). */
export function characterOf(message) {
    const c = ctx();
    if (!c.groupId) return { chid: c.characterId, char: c.characters[c.characterId] };
    const idx = c.characters.findIndex(ch => ch.avatar === message?.original_avatar || ch.name === message?.name);
    return { chid: idx >= 0 ? idx : undefined, char: c.characters[idx] };
}

export function currentGroup() {
    const c = ctx();
    return c.groupId ? c.groups.find(g => g.id === c.groupId) : null;
}

/** The newest summary from SillyTavern's Summarize extension at or before this message. */
function latestSummary(chat, messageId) {
    for (let i = messageId; i >= 0; i--) if (chat[i]?.extra?.memory) return chat[i].extra.memory;
    return '';
}

export function settingCard(groupProfile) {
    const avatar = groupProfile?.setting_avatar;
    return avatar ? ctx().characters.find(ch => ch.avatar === avatar) : null;
}

export function settingCardText(groupProfile) {
    const card = settingCard(groupProfile);
    return card ? [card.description, card.scenario].filter(Boolean).join('\n\n') : '';
}

async function profileEntry(chid) {
    const c = ctx();
    const char = c.characters[chid];
    let profile = await getCharProfile(chid);
    profile = await ensureAssets(profile, patch => setCharProfile(chid, patch)).catch(() => profile);
    return personForContext(profile, charKey(char), char.name, sdPrompt(char));
}

/** Who can be in the shot: the speaker, then the group's other members. The Setting card isn't a person. */
async function charactersInScene(messageId, groupProfile) {
    const c = ctx();
    const { chid, char } = characterOf(c.chat[messageId]);
    const ids = [];
    if (chid !== undefined && char) ids.push(Number(chid));
    const group = currentGroup();
    for (const avatar of group?.members ?? []) {
        if (avatar === groupProfile?.setting_avatar) continue;
        const idx = c.characters.findIndex(ch => ch.avatar === avatar);
        if (idx >= 0 && !ids.includes(idx)) ids.push(idx);
    }
    return Promise.all(ids.map(profileEntry));
}

export async function personaForContext() {
    const id = await personaId();
    let profile = getPersonaProfile(id);
    profile = await ensureAssets(profile, async patch => setPersonaProfile(id, patch)).catch(() => profile);
    return personForContext(profile, `persona:${id}`, personaName(id));
}

export function history(messageId, count = 10) {
    return ctx().chat.slice(Math.max(0, messageId - count), messageId)
        .filter(m => !m.is_system && m.extra?.type !== 'narrator')
        .map(m => ({ name: m.name, is_user: !!m.is_user, text: String(m.mes || '') }));
}

export async function directorContext(messageId, kind, groupProfile) {
    const c = ctx();
    const msg = c.chat[messageId];
    return {
        kind,
        reply: String(msg?.mes || ''),
        history: history(messageId),
        summary: latestSummary(c.chat, messageId),
        setting: settingCardText(groupProfile),
        user: await personaForContext(),
        characters: await charactersInScene(messageId, groupProfile),
        device: window.innerHeight >= window.innerWidth ? 'portrait' : 'landscape',
        last_prompt: c.chatMetadata?.fm_last_prompt || '',
    };
}
