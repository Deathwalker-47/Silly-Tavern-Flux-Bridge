// Character media popup: avatar slots (Main + 4 extras) and the character's gallery.

import { listFolder, deleteFile, saveFile, blobToBase64, isVideoUrl } from './store.js';

const ctx = () => SillyTavern.getContext();
export const AVATAR_SLOTS = 5;
const esc = (s) => $('<span>').text(String(s ?? '')).html();

/**
 * @param {number|string} chid character id
 * @param {object} deps
 * @param {(chid) => Promise<object>} deps.getCharMedia
 * @param {(chid, patch) => Promise<object>} deps.setCharMedia
 * @param {(chid, prompt) => Promise<string>} deps.generateAvatar returns the saved file URL
 * @param {(chid) => Promise<boolean|null>} deps.loraStatus
 */
export async function openCharacterMedia(chid, deps) {
    const c = ctx();
    const ch = c.characters[chid];
    if (!ch) return;
    let cfg = await deps.getCharMedia(chid);
    let pickSlot = null; // when set, clicking a gallery item assigns it to this avatar slot

    const $root = $(`
        <div class="fm-popup">
            <div class="fm-popup-head">
                <h3><i class="fa-solid fa-images"></i> ${esc(ch.name)}: media</h3>
                <div class="fm-tabs">
                    <div class="menu_button fm-tab fm-tab-active" data-tab="avatars"><i class="fa-solid fa-user"></i> Avatars</div>
                    <div class="menu_button fm-tab" data-tab="gallery"><i class="fa-solid fa-photo-film"></i> Gallery</div>
                </div>
            </div>
            <div class="fm-pane" data-pane="avatars">
                <div class="fm-lora-status"><i class="fa-solid fa-spinner fa-spin"></i> Checking for a LoRA on the bridge…</div>
                <div class="fm-avatars"></div>
                <label class="fm-gen-label">Prompt for generated avatars
                    <textarea class="text_pole fm-avatar-prompt" rows="2" placeholder="Portrait of …, face clearly visible, neutral background"></textarea>
                </label>
                <div class="fm-char-options">
                    <label>Image mode for this character
                        <select class="text_pole fm-char-mode">
                            <option value="">Use global setting</option>
                            <option value="auto">Auto (LoRA if found, else avatar)</option>
                            <option value="lora">Text-to-image with LoRA</option>
                            <option value="avatar">Image-to-image from avatar</option>
                        </select>
                    </label>
                    <label class="checkbox_label"><input type="checkbox" class="fm-char-auto"> Auto-send photos/videos for this character</label>
                </div>
                <small class="fm-hint">Avatars are used to generate this character's images when there is no LoRA for them on the bridge (or when the image mode is set to avatar). Only the main avatar is needed today; the extra slots are for multi-reference models later.</small>
            </div>
            <div class="fm-pane fm-hidden" data-pane="gallery">
                <div class="fm-gallery-bar">
                    <span class="fm-pick-banner fm-hidden"></span>
                    <div class="menu_button menu_button_icon fm-refresh"><i class="fa-solid fa-rotate"></i><span>Refresh</span></div>
                </div>
                <div class="fm-gallery"><i>Loading…</i></div>
            </div>
            <input type="file" class="fm-file" accept="image/*" hidden>
        </div>`);

    const $file = $root.find('.fm-file');
    let uploadSlot = 0;

    $root.find('.fm-avatar-prompt').val(cfg.avatarPrompt || defaultAvatarPrompt(ch));
    $root.find('.fm-char-mode').val(cfg.imageMode || '');
    $root.find('.fm-char-auto').prop('checked', cfg.auto !== false);

    const renderAvatars = () => {
        const slots = Array.from({ length: AVATAR_SLOTS }, (_, i) => cfg.avatars?.[i] || null);
        $root.find('.fm-avatars').html(slots.map((url, i) => `
            <div class="fm-slot ${i === 0 ? 'fm-slot-main' : ''}" data-slot="${i}">
                <div class="fm-slot-img">${url ? `<img src="${esc(url)}" alt="">` : '<i class="fa-solid fa-user-plus"></i>'}</div>
                <div class="fm-slot-label">${i === 0 ? 'Main' : `Extra ${i}`}</div>
                <div class="fm-slot-actions">
                    <div class="menu_button fa-solid fa-upload fm-upload" title="Upload an image"></div>
                    <div class="menu_button fa-solid fa-wand-magic-sparkles fm-generate" title="Generate (text-to-image)"></div>
                    <div class="menu_button fa-solid fa-photo-film fm-pick" title="Pick from gallery"></div>
                    <div class="menu_button fa-solid fa-trash fm-clear ${url ? '' : 'fm-hidden'}" title="Clear slot"></div>
                </div>
            </div>`).join(''));
    };

    const setSlot = async (slot, url) => {
        const avatars = Array.from({ length: AVATAR_SLOTS }, (_, i) => cfg.avatars?.[i] || null);
        avatars[slot] = url;
        cfg = await deps.setCharMedia(chid, { avatars });
        renderAvatars();
    };

    const renderGallery = async () => {
        const $g = $root.find('.fm-gallery').html('<i>Loading…</i>');
        const items = await listFolder(ch.name);
        if (!items.length) {
            $g.html('<i>No images or videos yet. Generated media for this character lands here.</i>');
            return;
        }
        $g.html(items.map(item => `
            <div class="fm-item" data-url="${esc(item.url)}" data-type="${item.type}">
                ${item.type === 'video'
                    ? `<video src="${esc(item.url)}" muted loop preload="metadata"></video><i class="fa-solid fa-film fm-item-badge"></i>`
                    : `<img src="${esc(item.url)}" loading="lazy" alt="">`}
                <div class="fm-item-actions">
                    <div class="menu_button fa-solid fa-up-right-from-square fm-open" title="Open"></div>
                    ${item.type === 'image' ? '<div class="menu_button fa-solid fa-user-check fm-set-main" title="Use as main avatar"></div>' : ''}
                    <div class="menu_button fa-solid fa-trash fm-delete" title="Delete"></div>
                </div>
            </div>`).join(''));
    };

    const showTab = (tab) => {
        $root.find('.fm-tab').removeClass('fm-tab-active').filter(`[data-tab="${tab}"]`).addClass('fm-tab-active');
        $root.find('.fm-pane').addClass('fm-hidden').filter(`[data-pane="${tab}"]`).removeClass('fm-hidden');
        if (tab === 'gallery') renderGallery();
        if (tab !== 'gallery') {
            pickSlot = null;
            $root.find('.fm-pick-banner').addClass('fm-hidden');
        }
    };

    $root.on('click', '.fm-tab', function () { showTab(this.dataset.tab); });
    $root.on('click', '.fm-refresh', () => renderGallery());

    $root.on('click', '.fm-upload', function () {
        uploadSlot = Number($(this).closest('.fm-slot').data('slot'));
        $file.val('').trigger('click');
    });
    $file.on('change', async function () {
        const file = this.files?.[0];
        if (!file) return;
        try {
            const base64 = await blobToBase64(file);
            const ext = (file.type.split('/')[1] || 'png').replace('jpeg', 'jpg');
            const url = await saveFile(base64, ext, ch.name, `avatar${uploadSlot}_${Date.now()}`);
            await setSlot(uploadSlot, url);
        } catch (err) {
            toastr.error(err?.message || String(err), 'Media Studio');
        }
    });

    $root.on('click', '.fm-generate', async function () {
        const slot = Number($(this).closest('.fm-slot').data('slot'));
        const prompt = String($root.find('.fm-avatar-prompt').val() || '').trim() || defaultAvatarPrompt(ch);
        const $slot = $(this).closest('.fm-slot').addClass('fm-busy');
        try {
            await deps.setCharMedia(chid, { avatarPrompt: prompt });
            const url = await deps.generateAvatar(chid, prompt);
            await setSlot(slot, url);
        } catch (err) {
            toastr.error(err?.message || String(err), 'Media Studio: avatar generation failed');
        } finally {
            $slot.removeClass('fm-busy');
        }
    });

    $root.on('click', '.fm-pick', function () {
        pickSlot = Number($(this).closest('.fm-slot').data('slot'));
        showTab('gallery');
        pickSlot = Number($(this).closest('.fm-slot').data('slot'));
        $root.find('.fm-pick-banner').removeClass('fm-hidden').text(`Click an image to use it for the ${pickSlot === 0 ? 'main' : `extra ${pickSlot}`} avatar`);
    });

    $root.on('click', '.fm-clear', async function () {
        await setSlot(Number($(this).closest('.fm-slot').data('slot')), null);
    });

    $root.on('click', '.fm-item', async function (e) {
        if ($(e.target).closest('.fm-item-actions').length) return;
        const { url, type } = this.dataset;
        if (pickSlot !== null && type === 'image') {
            const slot = pickSlot;
            pickSlot = null;
            await setSlot(slot, url);
            showTab('avatars');
            return;
        }
        if (type === 'video') {
            const video = $(this).find('video')[0];
            video.paused ? video.play() : video.pause();
        }
    });
    $root.on('click', '.fm-open', function () { window.open($(this).closest('.fm-item').data('url'), '_blank'); });
    $root.on('click', '.fm-set-main', async function () { await setSlot(0, $(this).closest('.fm-item').data('url')); toastr.success('Main avatar updated', 'Media Studio'); });
    $root.on('click', '.fm-delete', async function () {
        const url = $(this).closest('.fm-item').data('url');
        if (!confirm('Delete this file permanently?')) return;
        try {
            await deleteFile(url);
            if ((cfg.avatars || []).includes(url)) {
                const avatars = (cfg.avatars || []).map(a => (a === url ? null : a));
                cfg = await deps.setCharMedia(chid, { avatars });
                renderAvatars();
            }
            renderGallery();
        } catch (err) {
            toastr.error(err?.message || String(err), 'Media Studio');
        }
    });

    $root.on('change', '.fm-char-mode', async function () { cfg = await deps.setCharMedia(chid, { imageMode: this.value }); });
    $root.on('change', '.fm-char-auto', async function () { cfg = await deps.setCharMedia(chid, { auto: this.checked }); });
    $root.on('change', '.fm-avatar-prompt', async function () { cfg = await deps.setCharMedia(chid, { avatarPrompt: this.value }); });

    renderAvatars();
    deps.loraStatus(chid).then((has) => {
        const text = has === null
            ? '<i class="fa-solid fa-plug-circle-xmark"></i> Could not reach the bridge to check for a LoRA.'
            : has
                ? '<i class="fa-solid fa-circle-check fm-ok"></i> This character has a LoRA on the bridge: images use text-to-image with the LoRA (Auto mode).'
                : '<i class="fa-solid fa-circle-info fm-warn"></i> No LoRA found for this character: images are generated from the main avatar (Auto mode).';
        $root.find('.fm-lora-status').html(text);
    });

    await c.callGenericPopup($root, c.POPUP_TYPE.TEXT, '', { okButton: 'Close', wide: true, large: false, allowVerticalScrolling: true, leftAlign: true });
}

function defaultAvatarPrompt(ch) {
    const sd = ch?.data?.extensions?.sd_character_prompt || '';
    return `${ch.name}${sd ? `, ${sd}` : ''}, portrait photo, face and upper body clearly visible, looking at the camera, soft natural light, plain background`;
}

export { isVideoUrl };
