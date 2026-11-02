// What only the browser can see goes to the bridge's event record: attached or missed,
// why an auto image was skipped, how a group round ended. It never bothers the user.

import { bridge } from './bridge.js';

const queue = [];
let timer = null;

async function flush() {
    timer = null;
    const events = queue.splice(0, 50);
    if (!events.length) return;
    try {
        await bridge('/media/events', { method: 'POST', body: { events }, timeout: 10000 });
    } catch { /* a lost event is fine */ }
    if (queue.length) timer = setTimeout(flush, 3000);
}

export function track(event, facts = {}) {
    queue.push({ event, chat_id: SillyTavern.getContext().getCurrentChatId?.() ?? null, ...facts });
    timer ??= setTimeout(flush, 3000);
}
