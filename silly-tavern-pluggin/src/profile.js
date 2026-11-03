// Character and persona media profiles: five tagged reference slots plus Appearance.
// Characters keep theirs on the card (data.extensions.flux_media), so it travels with
// exports. SillyTavern personas have no extension field, so theirs live in our settings.
// A story card can also hold side characters, each a profile of their own under `side`.

import { urlToAsset } from './assets.js';

const ctx = () => SillyTavern.getContext();
const KEY = 'flux_media';

export const SLOTS = [
    { id: 'face_1', label: 'Face 1', required: true },
    { id: 'face_2', label: 'Face 2' },
    { id: 'full_body', label: 'Full body' },
    { id: 'portrait', label: 'Portrait' },
    { id: 'naked', label: 'Naked (optional)' },
];

const blank = () => ({
    schema: 2,
    auto: true,
    appearance: { age: '', text: '' },
    refs: Object.fromEntries(SLOTS.map(s => [s.id, null])),
    fictional_confirmed: false,
});

/** Schema 1 had avatars[0..4] and avatarPrompt. Assets are filled in by ensureAssets. */
export function migrateProfile(raw) {
    if (raw?.schema === 2) return { ...blank(), ...raw, appearance: { ...blank().appearance, ...raw.appearance }, refs: { ...blank().refs, ...raw.refs } };
    const next = blank();
    if (!raw) return next;
    if (raw.auto === false) next.auto = false;
    next.appearance.text = raw.avatarPrompt || '';
    SLOTS.forEach((slot, i) => {
        const url = raw.avatars?.[i];
        if (url) next.refs[slot.id] = { url, asset: null };
    });
    next.migrated = SLOTS.some((_, i) => raw.avatars?.[i]);
    return next;
}

// ---- characters ----

let cardWrites = Promise.resolve();

export async function getCharProfile(chid) {
    const c = ctx();
    if (chid === undefined || chid === null || !c.characters[chid]) return blank();
    await c.unshallowCharacter?.(chid);     // lazy-loaded cards don't carry extension fields
    return migrateProfile(c.characters[chid]?.data?.extensions?.[KEY]);
}

/** Queued read-modify-write so quick edits don't overwrite each other. A patch can be a function of the current profile. */
export function setCharProfile(chid, patch) {
    const run = cardWrites.then(async () => {
        const current = await getCharProfile(chid);
        const next = { ...current, ...(typeof patch === 'function' ? patch(current) : patch) };
        await ctx().writeExtensionField(chid, KEY, next);
        return next;
    });
    cardWrites = run.catch(() => {});
    return run;
}

export const charKey = (char) => `char:${char?.avatar || char?.name}`;

// ---- side characters on a story card ----
// The card's own references stay the default. Each side character keeps a stable id, so a
// rename keeps their gallery, which the bridge files under this key and the card's.

export const sideKey = (char, id) => `${charKey(char)}#side:${id}`;

export const sideList = (profile) => (profile?.side || []).filter(p => p?.id);

const sideProfile = (person) => ({ ...migrateProfile(person), id: person.id, name: person.name || '' });

export async function getSideProfile(chid, id) {
    const person = sideList(await getCharProfile(chid)).find(p => p.id === id);
    return sideProfile(person || { ...blank(), id });
}

export async function setSideProfile(chid, id, patch) {
    const apply = p => ({ ...p, ...(typeof patch === 'function' ? patch(sideProfile(p)) : patch) });
    const next = await setCharProfile(chid, current => ({ side: sideList(current).map(p => (p.id === id ? apply(p) : p)) }));
    return sideProfile(sideList(next).find(p => p.id === id) || { ...blank(), id });
}

export async function addSidePerson(chid, name, fields = {}) {
    const id = Math.random().toString(36).slice(2, 10);
    await setCharProfile(chid, current => ({ side: [...sideList(current), { ...blank(), id, name, ...fields }] }));
    return id;
}

export const removeSidePerson = (chid, id) => setCharProfile(chid, current => ({ side: sideList(current).filter(p => p.id !== id) }));

// ---- personas ----

let personaModule = null;

/** The active persona's avatar id. It isn't on getContext, so it comes from personas.js. */
export async function personaId() {
    personaModule ??= await import('/scripts/personas.js').catch(() => ({}));
    return personaModule.user_avatar || ctx().powerUserSettings?.default_persona || ctx().name1 || 'user';
}

export const personaName = (id) => ctx().powerUserSettings?.personas?.[id] || ctx().name1 || 'User';

export function getPersonaProfile(id) {
    return migrateProfile(ctx().extensionSettings?.[KEY]?.personas?.[id]);
}

export function setPersonaProfile(id, patch) {
    const c = ctx();
    const store = (c.extensionSettings[KEY].personas ??= {});
    const current = getPersonaProfile(id);
    store[id] = { ...current, ...(typeof patch === 'function' ? patch(current) : patch) };
    c.saveSettingsDebounced();
    return store[id];
}

// ---- both ----

/** Migrated slots have a file but no asset yet. Any outside the window are marked for replacement and aren't sent. */
export async function ensureAssets(profile, save) {
    let changed = false;
    const refs = { ...profile.refs };
    for (const slot of SLOTS) {
        const ref = refs[slot.id];
        if (!ref?.url || ref.asset || ref.rejected) continue;
        try {
            refs[slot.id] = { ...ref, asset: await urlToAsset(ref.url, 'reference') };
        } catch (err) {
            if (err?.status !== 422) throw err;
            refs[slot.id] = { ...ref, rejected: err.detail || err.message };
        }
        changed = true;
    }
    if (!changed) return profile;
    const next = { ...profile, refs };
    await save({ refs });
    return next;
}

/** A person as the director sees them. References go only with the one-time confirmation. */
export function personForContext(profile, key, name, sdPrompt = '') {
    const refs = {};
    if (profile.fictional_confirmed) {
        for (const slot of SLOTS) {
            const ref = profile.refs[slot.id];
            if (ref?.asset && !ref.rejected) refs[slot.id] = ref.asset;
        }
    }
    const age = Number(profile.appearance?.age);
    return { key, name, age: Number.isFinite(age) && age > 0 ? age : null, appearance: profile.appearance?.text || '', refs, sd_prompt: sdPrompt };
}
