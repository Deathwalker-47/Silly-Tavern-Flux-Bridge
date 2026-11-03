// One popup for a character or persona: References, Appearance, Gallery. A story card's
// popup also switches between the card and its side characters, each with their own.

import { bridge, bridgeBase, cachedCapabilities } from './bridge.js';
import { uploadAsset, windowProblem } from './assets.js';
import { runDetached } from './jobs.js';
import { SLOTS } from './profile.js';
import { blobToBase64, saveFile } from './store.js';

const ctx = () => SillyTavern.getContext();
const esc = (s) => $('<span>').text(String(s ?? '')).html();
const stars = (rating) => [1, 2, 3, 4, 5].map(n => `<i class="fa-${n <= (rating || 0) ? 'solid' : 'regular'} fa-star fm-star" data-star="${n}"></i>`).join('');

function limitsText() {
    const w = cachedCapabilities()?.reference_window;
    if (!w) return '';
    const mb = (n) => `${Math.round(n / 1048576)} MB`;
    return `${w.formats.join(', ')}. ${Math.round(w.min_bytes / 1024)} KB to ${mb(w.max_bytes)}. Shortest side at least ${w.min_side} px, longest up to ${w.max_side} px, at most ${w.max_pixels / 1e6} MP, shape between 2:5 and 5:2.`;
}

function costText(g) {
    const credits = g.cost_credits != null ? `${g.cost_credits} cr` : '';
    const usd = g.cost_usd != null ? `$${Number(g.cost_usd).toFixed(g.cost_usd < 0.1 ? 3 : 2)}` : '';
    return [usd, credits].filter(Boolean).join(' + ');
}

function galleryItem(g) {
    const file = g.url || (g.file_url ? `${bridgeBase()}${g.file_url}` : '');
    const preview = !file ? '<div class="fm-gen-empty"><i class="fa-solid fa-clock-rotate-left"></i></div>'
        : g.kind === 'video' ? `<video src="${esc(file)}" controls muted preload="metadata"></video>`
            : `<img src="${esc(file)}" loading="lazy" alt="">`;
    const meta = [g.family || g.model_id, g.provider_label, g.aspect_ratio, g.seconds ? `${g.seconds} s` : ''].filter(Boolean).join(' · ');
    const seed = [g.seed != null ? `seed ${g.seed}` : '', costText(g)].filter(Boolean).join(' · ');
    const refs = (g.refs || []).map(r => `${r.person.replace(/^(char|persona):/, '').replace(/\.png$/, '')}: ${r.slot}`).join(', ');
    const loras = (g.loras || []).map(l => l.id).join(', ');
    return `
        <div class="fm-gen ${g.rating ? '' : 'fm-gen-unrated'}" data-id="${esc(g.id)}">
            <div class="fm-gen-media">${preview}</div>
            <div class="fm-gen-body">
                <div class="fm-gen-meta">${esc(meta)}</div>
                <div class="fm-gen-meta">${esc(seed)}</div>
                <div class="fm-stars">${stars(g.rating)}</div>
                <input class="text_pole fm-comment" type="text" placeholder="Add a comment" value="${esc(g.comment || '')}">
                <details><summary>Prompt and references</summary>
                    <div class="fm-gen-prompt">${esc(g.prompt || '')}</div>
                    ${refs ? `<div class="fm-gen-meta">References: ${esc(refs)}</div>` : ''}
                    ${loras ? `<div class="fm-gen-meta">LoRAs: ${esc(loras)}</div>` : ''}
                    <div class="menu_button menu_button_icon fm-trace"><i class="fa-solid fa-list-check"></i><span>Pipeline steps</span></div>
                    <pre class="fm-trace-out fm-hidden"></pre>
                </details>
            </div>
        </div>`;
}

function traceText(trace) {
    return trace.steps.map((s) => {
        const mark = s.ok === false ? '✗' : s.ok ? '✓' : '·';
        const time = s.ms != null ? ` ${s.ms} ms` : '';
        const facts = s.data ? ` ${JSON.stringify(s.data)}` : '';
        return `${String(s.t).padStart(7)}s ${mark} ${s.stage}/${s.event}${time}${facts}`;
    }).join('\n');
}

/**
 * @param {object} card
 * @param {string} card.name
 * @param {string} card.key            person key, as the director and gallery use it
 * @param {string} card.folder         SillyTavern image folder
 * @param {() => Promise<object>} card.load
 * @param {(patch) => Promise<object>} card.save
 * @param {string} [card.sdPrompt]     for a character: finds its LoRA
 * @param {string} [card.tab]
 * @param {object} [card.side]         a character's side characters: list, add, remove, who(id, name), suggestions
 */
export async function openMediaPanel(card) {
    const c = ctx();
    let who = card;                    // the card, or the side character being edited
    let profile = await who.load();
    let page = 0;
    let filter = '';
    let uploadSlot = '';
    let castPeople = [];
    const $root = $(`
        <div class="fm-popup">
            <div class="fm-popup-head">
                <h3><i class="fa-solid fa-images"></i> <span class="fm-who">${esc(who.name)}</span></h3>
                <div class="fm-tabs">
                    <div class="menu_button fm-tab" data-tab="refs">References</div>
                    <div class="menu_button fm-tab" data-tab="appearance">Appearance</div>
                    <div class="menu_button fm-tab" data-tab="gallery">Gallery</div>
                </div>
            </div>
            <div class="fm-people fm-hidden">
                <select class="text_pole fm-person" title="Side characters"></select>
                <input class="text_pole fm-person-name fm-hidden" type="text" placeholder="Name">
                <div class="menu_button menu_button_icon fm-find-people" title="Ask the bridge who else is in this card, and add them all"><i class="fa-solid fa-users-viewfinder"></i><span>Find people</span></div>
                <div class="menu_button fa-solid fa-trash fm-person-remove fm-hidden" title="Remove this side character and their references"></div>
            </div>
            <small class="fm-hint fm-people-hint fm-hidden"></small>
            <div class="fm-pane" data-pane="refs">
                <div class="fm-lora-status"></div>
                <div class="fm-migrated fm-hidden"></div>
                <div class="fm-avatars"></div>
                <label class="checkbox_label"><input type="checkbox" class="fm-confirm"> Fictional adult, or my own photos on my own persona</label>
                <small class="fm-hint">${esc(limitsText())} Location data is removed. Nothing is sent to a provider until this box is ticked.</small>
            </div>
            <div class="fm-pane fm-hidden" data-pane="appearance">
                <label>Age <small>(required, 18 or over)</small> <input class="text_pole fm-age" type="number" min="18" max="120"></label>
                <label>Appearance <small>(wins over anything in the chat)</small>
                    <textarea class="text_pole fm-appearance" rows="6" placeholder="Build, height, hair, eyes, skin, marks, usual style"></textarea>
                </label>
                <label class="checkbox_label fm-auto-row"><input type="checkbox" class="fm-auto"> Auto-send photos and videos for this character</label>
            </div>
            <div class="fm-pane fm-hidden" data-pane="gallery">
                <div class="fm-gallery-bar">
                    <div class="fm-tabs">
                        <div class="menu_button fm-filter fm-tab-active" data-kind="">All</div>
                        <div class="menu_button fm-filter" data-kind="image">Image</div>
                        <div class="menu_button fm-filter" data-kind="video">Video</div>
                    </div>
                </div>
                <div class="fm-gens"><i>Loading…</i></div>
                <div class="menu_button fm-more fm-hidden">Load more</div>
            </div>
            <input type="file" class="fm-file" accept="image/jpeg,image/png,image/webp,image/heic,image/heif" hidden>
        </div>`);
    const $file = $root.find('.fm-file');
    let uploadWho = who;

    /** Saves to whoever was showing when the edit began, even if the dropdown has moved on since. */
    const save = async (patch, target = who) => {
        const next = await target.save(patch);
        if (target === who) profile = next;
        return next;
    };

    const renderSlots = () => {
        $root.find('.fm-avatars').html(SLOTS.map((slot) => {
            const ref = profile.refs[slot.id];
            return `
                <div class="fm-slot ${slot.required ? 'fm-slot-main' : ''} ${ref?.rejected ? 'fm-slot-bad' : ''}" data-slot="${slot.id}">
                    <div class="fm-slot-img">${ref?.url ? `<img src="${esc(ref.url)}" alt="">` : '<i class="fa-solid fa-user-plus"></i>'}</div>
                    <div class="fm-slot-label">${esc(slot.label)}${slot.required ? ' *' : ''}</div>
                    ${ref?.rejected ? `<div class="fm-slot-error">${esc(ref.rejected)} Replace this image.</div>` : ''}
                    <div class="fm-slot-actions">
                        <div class="menu_button fa-solid fa-upload fm-upload" title="Upload an image"></div>
                        <div class="menu_button fa-solid fa-wand-magic-sparkles fm-generate" title="Generate from the active image model"></div>
                        <div class="menu_button fa-solid fa-trash fm-clear ${ref ? '' : 'fm-hidden'}" title="Clear slot"></div>
                    </div>
                </div>`;
        }).join(''));
    };

    const setSlot = async (slot, ref, target = who) => {
        await save(current => ({ refs: { ...current.refs, [slot]: ref } }), target);
        if (target === who) renderSlots();
    };

    const saveFromBlob = async (slot, blob, target = who) => {
        const asset = await uploadAsset(blob, 'reference');
        const ext = (blob.type.split('/')[1] || 'jpg').replace('jpeg', 'jpg');
        const url = await saveFile(await blobToBase64(blob), ext, target.folder, `ref_${slot}_${Date.now()}`);
        await setSlot(slot, { url, asset }, target);
    };

    const renderGallery = async (append = false) => {
        const asked = who;
        const $list = $root.find('.fm-gens');
        if (!append) {
            page = 0;
            $list.html('<i>Loading…</i>');
        }
        try {
            const { generations } = await bridge(`/media/generations?person=${encodeURIComponent(asked.key)}&kind=${filter}&limit=30&offset=${page * 30}`, { timeout: 20000 });
            if (asked !== who) return;
            if (!append) $list.empty();
            if (!generations.length && !append) $list.html('<i>Nothing generated yet. Every image and video lands here with its settings.</i>');
            $list.append(generations.map(galleryItem).join(''));
            $root.find('.fm-more').toggleClass('fm-hidden', generations.length < 30);
        } catch (err) {
            $list.html(`<i>${esc(err?.message || err)}</i>`);
        }
    };

    const showTab = (tab) => {
        $root.find('.fm-tab[data-tab]').removeClass('fm-tab-active').filter(`[data-tab="${tab}"]`).addClass('fm-tab-active');
        $root.find('.fm-pane').addClass('fm-hidden').filter(`[data-pane="${tab}"]`).removeClass('fm-hidden');
        if (tab === 'gallery') renderGallery();
    };

    const patchGeneration = (id, body) => bridge(`/media/generations/${id}`, { method: 'PATCH', body, timeout: 15000 });

    const loraStatus = () => {
        const $status = $root.find('.fm-lora-status').empty();
        if (!who.key.startsWith('char:')) return;
        const asked = who;
        bridge('/media/lora-status', { method: 'POST', body: { names: [who.name], character_prompts: { [who.name]: who.sdPrompt || '' } }, timeout: 15000 })
            .then((r) => {
                if (asked !== who) return;
                $status.html(r.characters?.[asked.name]
                    ? '<i class="fa-solid fa-circle-check fm-ok"></i> This character has a LoRA. Flux images use it.'
                    : '<i class="fa-solid fa-circle-info fm-warn"></i> No LoRA for this character. Flux switches to Seedream 4.5 with these references.');
            })
            .catch(() => asked === who && $status.text('Could not reach the bridge to check for a LoRA.'));
    };

    /** The dropdown: the card first, then its side characters, then ways to add one. */
    const renderPeople = async () => {
        if (!card.side) return;
        const people = await card.side.list();
        const taken = new Set([...people.map(p => p.name.toLowerCase()), card.name.toLowerCase()]);
        castPeople = card.side.suggestions().filter(p => !taken.has(p.name.toLowerCase()));
        $root.find('.fm-person').html([
            `<option value="">${esc(card.name)} (the story card, used by default)</option>`,
            ...people.map(p => `<option value="${esc(p.id)}">${esc(p.name || 'Unnamed')}</option>`),
            ...(castPeople.length > 1 ? [`<option value="+cast-all">+ Add all ${castPeople.length} from the voice cast</option>`] : []),
            ...castPeople.map((p, i) => `<option value="+cast" data-cast="${i}">+ Add ${esc(p.name)} from the voice cast</option>`),
            '<option value="+new">+ Add a side character</option>',
        ].join('')).val(who.id || '');
        $root.find('.fm-people').removeClass('fm-hidden');
    };

    /** Adds found or cast people, says what happened, and returns the first new id. */
    const addPeople = async (found) => {
        const { added, skipped, voice } = await card.side.importAll(found);
        const parts = [added.length ? `Added ${added.map(p => p.name).join(', ')}.` : 'They are all here already.'];
        if (skipped && added.length) parts.push(`${skipped} already here.`);
        if (voice) parts.push(`${voice} also added to the voice cast.`);
        const described = added.filter(p => p.appearance).length;
        if (described) parts.push(`${described} got an appearance from the card.`);
        if (added.length) parts.push("Set each one's age before they can be generated.");
        toastr[added.length ? 'success' : 'info'](parts.join(' '), 'Media Studio', { timeOut: 12000 });
        await renderPeople();
        return added.length ? (await card.side.list()).find(p => p.name === added[0].name)?.id || '' : '';
    };

    const fillPerson = () => {
        const side = !!who.id;
        $root.find('.fm-who').text(who.name);
        $root.find('.fm-person-name').toggleClass('fm-hidden', !side).val(side ? who.name : '');
        $root.find('.fm-person-remove').toggleClass('fm-hidden', !side);
        $root.find('.fm-people-hint').toggleClass('fm-hidden', !card.side).text(side
            ? `${who.name}'s references are sent only when the director puts ${who.name} in the scene. Set ${who.name}'s age in Appearance.`
            : 'The story card\'s own references are the default. Add side characters to give other people in the story their own.');
        $root.find('.fm-confirm').prop('checked', !!profile.fictional_confirmed);
        $root.find('.fm-age').val(profile.appearance?.age || '');
        $root.find('.fm-appearance').val(profile.appearance?.text || '');
        $root.find('.fm-auto').prop('checked', profile.auto !== false);
        $root.find('.fm-auto-row').toggleClass('fm-hidden', side || (!who.sdPrompt && who.key.startsWith('persona:')));
        renderSlots();
        loraStatus();
        if (!$root.find('[data-pane="gallery"]').hasClass('fm-hidden')) renderGallery();
    };

    const showPerson = async (id) => {
        const person = id ? (await card.side.list()).find(p => p.id === id) : null;
        const target = person ? card.side.who(person.id, person.name) : card;
        who = target;
        const next = await target.load();
        if (target !== who) return;            // a later pick won
        profile = next;
        await renderPeople();
        fillPerson();
    };

    $root.on('click', '.fm-tab[data-tab]', function () { showTab(this.dataset.tab); });
    $root.on('change', '.fm-person', async function () {
        let id = this.value;
        if (id === '+cast-all') return showPerson(await addPeople(castPeople));
        if (id === '+cast') {
            const person = castPeople[Number($(this).find(':selected').attr('data-cast'))];
            if (!person) return renderPeople();
            id = await card.side.add(person.name, person);
        } else if (id === '+new') {
            const name = window.prompt('Name of the side character', '');
            if (!name?.trim()) return renderPeople();
            id = await card.side.add(name.trim());
        }
        await showPerson(id);
    });
    $root.on('click', '.fm-find-people', async function () {
        const $btn = $(this).addClass('fm-busy');
        const $label = $btn.find('span').text('Looking...');
        try {
            const found = await card.side.find();
            if (!found.length) return toastr.info('No other named people on this card.', 'Media Studio');
            await showPerson(await addPeople(found));
        } catch (err) {
            toastr.error(err?.detail || err?.message || String(err), 'Media Studio: could not read the card');
        } finally {
            $btn.removeClass('fm-busy');
            $label.text('Find people');
        }
    });
    $root.on('change', '.fm-person-name', async function () {
        const target = who;
        const name = this.value.trim();
        if (!name || !target.id) return $(this).val(target.name);
        await save({ name }, target);
        if (target === who) who = card.side.who(target.id, name);
        await renderPeople();
        fillPerson();
    });
    $root.on('click', '.fm-person-remove', async function () {
        if (!who.id || !window.confirm(`Remove ${who.name} and their references from this card?`)) return;
        await card.side.remove(who.id);
        await showPerson('');
    });
    $root.on('click', '.fm-upload', function () {
        uploadSlot = $(this).closest('.fm-slot').data('slot');
        uploadWho = who;
        $file.val('').trigger('click');
    });
    $file.on('change', async function () {
        const file = this.files?.[0];
        if (!file) return;
        const problem = await windowProblem(file);
        if (problem) return toastr.warning(problem, 'Media Studio');
        const $slot = $root.find(`.fm-slot[data-slot="${uploadSlot}"]`).addClass('fm-busy');
        try {
            await saveFromBlob(uploadSlot, file, uploadWho);
        } catch (err) {
            toastr.warning(err?.detail || err?.message || String(err), 'Media Studio');
        } finally {
            $slot.removeClass('fm-busy');
        }
    });
    $root.on('click', '.fm-generate', async function () {
        const slot = $(this).closest('.fm-slot').data('slot');
        const target = who;
        const prompt = window.prompt('Describe the image for this slot', `${target.name}, ${profile.appearance?.text || 'portrait photo'}, face clearly visible, neutral background`);
        if (!prompt) return;
        const $slot = $(this).closest('.fm-slot').addClass('fm-busy');
        try {
            const s = c.extensionSettings.flux_media;
            const refs = profile.fictional_confirmed ? Object.values(profile.refs).map(r => r?.asset).filter(Boolean) : [];
            const job = await runDetached({ kind: 'image', selection: s.image, prompt, refs, folder: target.folder, origin: 'panel', person_key: target.key, lora_for: { name: target.name, sd_prompt: target.sdPrompt || '' } });
            const blob = await (await fetch(`${bridgeBase()}${job.file_url}`)).blob();
            await saveFromBlob(slot, blob, target);
        } catch (err) {
            toastr.error(err?.detail || err?.message || String(err), 'Media Studio: image failed');
        } finally {
            $slot.removeClass('fm-busy');
        }
    });
    $root.on('click', '.fm-clear', async function () { await setSlot($(this).closest('.fm-slot').data('slot'), null); });
    $root.on('change', '.fm-confirm', async function () { await save({ fictional_confirmed: this.checked }); });
    $root.on('change', '.fm-auto', async function () { await save({ auto: this.checked }); });
    $root.on('change', '.fm-age, .fm-appearance', async function () {
        const age = Number($root.find('.fm-age').val());
        if (age && age < 18) {
            toastr.warning('Age must be 18 or over.', 'Media Studio');
            return $root.find('.fm-age').val(profile.appearance?.age || '');
        }
        await save({ appearance: { age: age || '', text: String($root.find('.fm-appearance').val() || '') } });
    });

    $root.on('click', '.fm-filter', function () {
        filter = this.dataset.kind;
        $root.find('.fm-filter').removeClass('fm-tab-active');
        $(this).addClass('fm-tab-active');
        renderGallery();
    });
    $root.on('click', '.fm-more', () => { page += 1; renderGallery(true); });
    $root.on('click', '.fm-star', async function () {
        const $gen = $(this).closest('.fm-gen');
        const rating = Number(this.dataset.star);
        $gen.find('.fm-stars').html(stars(rating));
        $gen.removeClass('fm-gen-unrated');
        await patchGeneration($gen.data('id'), { rating }).catch(err => toastr.error(err?.message || String(err), 'Media Studio'));
    });
    $root.on('change', '.fm-comment', async function () {
        await patchGeneration($(this).closest('.fm-gen').data('id'), { comment: this.value }).catch(err => toastr.error(err?.message || String(err), 'Media Studio'));
    });
    $root.on('click', '.fm-trace', async function () {
        const $out = $(this).siblings('.fm-trace-out').toggleClass('fm-hidden');
        if ($out.hasClass('fm-hidden') || $out.text()) return;
        const id = $(this).closest('.fm-gen').data('id');
        try {
            $out.text(traceText(await bridge(`/media/generations/${id}/trace`, { timeout: 20000 })));
        } catch (err) {
            $out.text(err?.message || String(err));
        }
    });

    if (profile.migrated) {
        $root.find('.fm-migrated').removeClass('fm-hidden').text('These came from your old avatar slots. Check each one is in the right slot.');
        save({ migrated: false });
    }
    await renderPeople();
    fillPerson();
    showTab(card.tab || 'refs');
    await c.callGenericPopup($root, c.POPUP_TYPE.TEXT, '', { okButton: 'Close', wide: true, large: false, allowVerticalScrolling: true, leftAlign: true });
}
