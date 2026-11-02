// SillyTavern picks a slash command (or macro) on the first touch of its suggestion list,
// so on a phone the touch that starts a scroll picks whatever is under the finger, usually
// /bgcol near the top. Touches on the list are held back here: a scroll stays a scroll, and
// a tap still picks the item, by replaying it as the mouse press SillyTavern listens for.

const MOVE_PX = 10;

export function fixTouchAutocomplete() {
    let pending = null;

    document.addEventListener('pointerdown', (e) => {
        if (e.pointerType === 'mouse' || !e.isTrusted) return;
        const item = e.target.closest?.('.autoComplete li');
        if (!item) return;
        // Capture phase: SillyTavern's own listener sits on the item and never sees this touch.
        e.stopPropagation();
        e.preventDefault();         // keeps the chat box focused, as SillyTavern does
        pending = { item, x: e.clientX, y: e.clientY, id: e.pointerId };
    }, true);

    document.addEventListener('pointermove', (e) => {
        if (pending && e.pointerId === pending.id && Math.hypot(e.clientX - pending.x, e.clientY - pending.y) > MOVE_PX) pending = null;
    }, true);

    document.addEventListener('pointercancel', () => { pending = null; }, true);   // the browser took the touch for scrolling

    document.addEventListener('pointerup', (e) => {
        if (!pending || e.pointerId !== pending.id) return;
        const { item } = pending;
        pending = null;
        if (!item.isConnected) return;
        item.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, cancelable: true, pointerType: 'mouse', isPrimary: true }));
    }, true);
}
