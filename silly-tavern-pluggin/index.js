/**
 * Auto Image Generator - Universal with LLM Context
 * FIXED: Mutation Observer approach - Works from empty chat
 * Waits for complete messages before triggering generation
 */

(async function() {
    console.log('🎨 Auto Image Generator with LLM loading...');

    const DEFAULT_CONFIG = {
        BRIDGE_URL: "http://localhost:7861/sdapi/v1/txt2img",
        OPENROUTER_API_KEY: "",
        OPENROUTER_URL: "https://openrouter.ai/api/v1/chat/completions",
        LLM_MODEL: "deepseek/deepseek-chat",
        MESSAGE_COMPLETE_DELAY: 4000, // 4 seconds after last change
        CHECK_INTERVAL: 1000, // Fallback check every second
        MIN_MESSAGE_LENGTH: 20,
        // When true: send raw narrative to bridge (bridge's DeepSeek handles summarization).
        // When false: use OpenRouter LLM in plugin first (legacy behaviour).
        USE_BRIDGE_SUMMARIZATION: true
    };

    function loadConfig() {
        const runtime = (typeof window !== 'undefined' && window.AUTO_IMAGE_UNIVERSAL_CONFIG)
            ? window.AUTO_IMAGE_UNIVERSAL_CONFIG
            : {};

        let persisted = {};
        try {
            const raw = localStorage.getItem('autoImageUniversalConfig');
            persisted = raw ? JSON.parse(raw) : {};
        } catch (_) {
            persisted = {};
        }

        return { ...DEFAULT_CONFIG, ...persisted, ...runtime };
    }

    const CONFIG = loadConfig();

    function normalizeBridgeUrl(rawUrl) {
        if (!rawUrl || typeof rawUrl !== 'string') {
            return DEFAULT_CONFIG.BRIDGE_URL;
        }

        const trimmed = rawUrl.trim().replace(/\/$/, '');
        if (!trimmed) return DEFAULT_CONFIG.BRIDGE_URL;

        return trimmed.endsWith('/sdapi/v1/txt2img')
            ? trimmed
            : `${trimmed}/sdapi/v1/txt2img`;
    }

    /**
     * Resolve bridge URL at call time:
     *  1. SillyTavern's built-in Image Generation SD Web UI URL (auto_url)
     *  2. Plugin config (localStorage / window override / default)
     */
    function resolveBridgeUrl() {
        try {
            if (typeof SillyTavern !== 'undefined' && SillyTavern.getContext) {
                const ctx = SillyTavern.getContext();
                const stUrl = ctx?.extensionSettings?.sd?.auto_url;
                if (stUrl && typeof stUrl === 'string' && stUrl.trim().length > 0) {
                    return normalizeBridgeUrl(stUrl);
                }
            }
        } catch (_) {}

        return normalizeBridgeUrl(CONFIG.BRIDGE_URL);
    }

    let lastProcessedMessage = '';
    let messageCompleteTimer = null;
    let isProcessing = false;
    let observerActive = false;

    // ============================================
    // CORE MESSAGE DETECTION
    // ============================================

    /**
     * Get the latest AI message from chat
     */
    function getLatestAIMessage() {
        const chat = document.querySelector('#chat');
        if (!chat) return null;

        const messages = chat.querySelectorAll('.mes');
        if (messages.length === 0) return null;

        for (let i = messages.length - 1; i >= 0; i--) {
            const msg = messages[i];
            const isUser = msg.getAttribute('is_user') === 'true';
            const isSystem = msg.getAttribute('is_system') === 'true';

            // Get AI message (not user, not system)
            if (!isUser && !isSystem) {
                const messageText = msg.querySelector('.mes_text');
                const text = messageText ? messageText.innerText.trim() : '';
                
                if (text.length >= CONFIG.MIN_MESSAGE_LENGTH) {
                    return {
                        text: text,
                        element: msg,
                        name: msg.querySelector('.ch_name')?.textContent || 'Unknown'
                    };
                }
            }
        }
        return null;
    }

    /**
     * Handle message completion - triggers image generation
     */
    async function onMessageComplete() {
        const message = getLatestAIMessage();

        if (!message || message.text === lastProcessedMessage) {
            return;
        }

        // Check if image already exists for this message
        if (message.element.querySelector('.auto-generated-image')) {
            console.log('[AutoImageGen] ⏭️ Image already exists for this message');
            lastProcessedMessage = message.text;
            return;
        }

        console.log('[AutoImageGen] ✅ Message complete, triggering generation');
        lastProcessedMessage = message.text;

        await processNewMessage(message);
    }

    /**
     * Reset message complete timer (debounced)
     */
    function resetMessageCompleteTimer() {
        clearTimeout(messageCompleteTimer);
        messageCompleteTimer = setTimeout(onMessageComplete, CONFIG.MESSAGE_COMPLETE_DELAY);
    }

    // ============================================
    // MUTATION OBSERVER SETUP
    // ============================================

    /**
     * Setup MutationObserver to watch for new messages and text changes
     */
    function setupMessageObserver() {
        if (observerActive) return;

        const chatContainer = document.querySelector('#chat');
        if (!chatContainer) {
            console.log('[AutoImageGen] Chat container not found, retrying...');
            setTimeout(setupMessageObserver, 1000);
            return;
        }

        console.log('[AutoImageGen] Setting up message observer');
        observerActive = true;

        const observer = new MutationObserver((mutations) => {
            let shouldResetTimer = false;

            for (const mutation of mutations) {
                // ✅ CHECK 1: New message nodes being added (works from empty chat)
                if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                    for (const node of mutation.addedNodes) {
                        if (node.classList && node.classList.contains('mes')) {
                            const isUser = node.getAttribute('is_user') === 'true';
                            const isSystem = node.getAttribute('is_system') === 'true';

                            // New AI message detected
                            if (!isUser && !isSystem) {
                                console.log('[AutoImageGen] 📨 New AI message detected');
                                shouldResetTimer = true;
                            }
                        }
                    }
                }

                // ✅ CHECK 2: Text changes in existing messages (streaming text)
                if (mutation.type === 'characterData' || mutation.type === 'childList') {
                    const target = mutation.target.nodeType === Node.TEXT_NODE
                        ? mutation.target.parentElement
                        : mutation.target;

                    if (target && target.closest('.mes_text')) {
                        const mesElement = target.closest('.mes');
                        if (mesElement) {
                            const isUser = mesElement.getAttribute('is_user') === 'true';
                            const isSystem = mesElement.getAttribute('is_system') === 'true';

                            // Text change in AI message
                            if (!isUser && !isSystem) {
                                console.log('[AutoImageGen] ✏️ AI message text updated');
                                shouldResetTimer = true;
                            }
                        }
                    }
                }
            }

            if (shouldResetTimer) {
                resetMessageCompleteTimer();
            }
        });

        // Configure observer to catch all changes
        observer.observe(chatContainer, {
            childList: true,           // ✅ Watch for new message nodes
            subtree: true,             // ✅ Watch all descendants
            characterData: true,       // ✅ Watch text content changes
            characterDataOldValue: false,
            attributes: false
        });

        console.log('[AutoImageGen] ✅ Observer active - watching for messages');
    }

    /**
     * Fallback polling in case observer misses something
     */
    function startFallbackCheck() {
        setInterval(() => {
            if (!isProcessing && !messageCompleteTimer) {
                const message = getLatestAIMessage();
                if (message && message.text !== lastProcessedMessage) {
                    // New message detected but observer didn't catch it
                    console.log('[AutoImageGen] 🔄 Fallback detected new message');
                    resetMessageCompleteTimer();
                }
            }
        }, CONFIG.CHECK_INTERVAL);
    }

    // ============================================
    // CHARACTER & PROMPT HANDLING
    // ============================================

    function getChatContext(numMessages = 5) {
        const chat = document.querySelector('#chat');
        const messages = chat?.querySelectorAll('.mes');
        if (!messages || messages.length === 0) return [];

        const context = [];
        const startIndex = Math.max(0, messages.length - numMessages);

        for (let i = startIndex; i < messages.length; i++) {
            const msg = messages[i];
            const name = msg.querySelector('.ch_name')?.textContent || 'Unknown';
            const text = msg.querySelector('.mes_text')?.textContent || '';

            if (text.trim()) {
                context.push({
                    role: name === 'You' ? 'user' : 'character',
                    name: name,
                    content: text.trim()
                });
            }
        }

        return context;
    }

    function getActiveCharacterName() {
        try {
            if (typeof SillyTavern !== 'undefined' && SillyTavern.getContext) {
                const context = SillyTavern.getContext();
                if (context.name2) return context.name2;
                if (context.characters && context.characterId >= 0) {
                    return context.characters[context.characterId]?.name;
                }
            }

            const chat = document.querySelector('#chat');
            const messages = chat?.querySelectorAll('.mes');
            if (messages && messages.length > 0) {
                for (let i = messages.length - 1; i >= 0; i--) {
                    const msg = messages[i];
                    const isUser = msg.getAttribute('is_user') === 'true';
                    if (!isUser) {
                        const name = msg.querySelector('.ch_name')?.textContent;
                        if (name) return name;
                    }
                }
            }

            return null;
        } catch (e) {
            console.error('[AutoImageGen] ❌ Error getting character name:', e);
            return null;
        }
    }

    function getCharacterImagePrompt(characterName) {
        try {
            console.log(`[AutoImageGen] 🔍 Looking for image prompt for: ${characterName}`);

            let character = null;

            if (typeof SillyTavern !== 'undefined' && SillyTavern.getContext) {
                const context = SillyTavern.getContext();
                if (context.characters && context.characterId >= 0) {
                    character = context.characters[context.characterId];
                }
            }

            if (!character && typeof characters !== 'undefined' && typeof this_chid !== 'undefined') {
                character = characters[this_chid];
            }

            if (!character) {
                console.log('[AutoImageGen] ⚠️ Could not access character data');
                return getFallbackPrompt(characterName);
            }

            console.log('[AutoImageGen] ✅ Got character data');

            // Check sd_character_prompt FIRST (correct SillyTavern field)
            const possiblePaths = [
                character.data?.extensions?.sd_character_prompt,
                character.data?.extensions?.sd_api_prompt,
                character.extensions?.sd_character_prompt,
                character.extensions?.sd_api_prompt,
                character.sd_character_prompt,
                character.sd_prompt
            ];

            for (let i = 0; i < possiblePaths.length; i++) {
                const prompt = possiblePaths[i];
                if (prompt && typeof prompt === 'string' && prompt.trim().length > 0) {
                    console.log(`[AutoImageGen] ✅ Found prompt: ${prompt.substring(0, 80)}...`);
                    return prompt.trim();
                }
            }

            console.log('[AutoImageGen] ⚠️ No prompt found, using fallback');
            return getFallbackPrompt(characterName);

        } catch (e) {
            console.error('[AutoImageGen] ❌ Error reading character data:', e);
            return getFallbackPrompt(characterName);
        }
    }

    function getFallbackPrompt(characterName) {
        console.log(`[AutoImageGen] ⚠️ Using fallback prompt for: ${characterName}`);
        const fallbackPrompts = {
            'Nimya': 'nimya, nimya_face, nimya_alt, 33 year-old south woman, nsfw2, warm brown skin, round face, long wavy black hair, beautiful face, elegant jewelry',
            'Shreya': 'Shreya, beautiful woman, long black hair, warm brown skin, natural beauty',
            'Sai Pallavi': 'Sai Pallavi, gorgeous actress, natural beauty, curly hair, radiant smile',
            'Shraddha Kapoor': 'Shraddha Kapoor, beautiful actress, round face, long black hair, elegant'
        };
        return fallbackPrompts[characterName] || `${characterName}, beautiful woman, natural beauty`;
    }

    function cleanPrompt(text) {
        return text
            .replace(/[^\x00-\x7F\u00C0-\u024F]/g, ' ')
            .replace(/\s+/g, ' ')
            .trim();
    }

    /**
     * Collect all unique character names visible in recent chat messages.
     * Also checks SillyTavern group chat membership if available.
     */
    function getAllVisibleCharacterNames(context) {
        const names = new Set();

        for (const msg of context) {
            if (msg.role === 'character' && msg.name && msg.name !== 'Unknown') {
                names.add(msg.name);
            }
        }

        try {
            if (typeof SillyTavern !== 'undefined' && SillyTavern.getContext) {
                const ctx = SillyTavern.getContext();
                if (ctx.groups && ctx.selectedGroupId) {
                    const group = ctx.groups.find(g => g.id === ctx.selectedGroupId);
                    if (group && group.members) {
                        for (const memberId of group.members) {
                            const char = ctx.characters?.find(c => c.avatar === memberId);
                            if (char?.name) names.add(char.name);
                        }
                    }
                }
            }
        } catch (_) {}

        return Array.from(names);
    }

    /**
     * Fetch SD prompt / trigger words for each named character from ST character data.
     * Returns an object: { "Nimya": "nimya33 south indian woman...", ... }
     */
    function getAllCharacterPrompts(characterNames) {
        const prompts = {};

        try {
            if (typeof SillyTavern !== 'undefined' && SillyTavern.getContext) {
                const ctx = SillyTavern.getContext();
                if (ctx.characters) {
                    for (const char of ctx.characters) {
                        if (characterNames.includes(char.name)) {
                            const sdPrompt =
                                char.data?.extensions?.sd_character_prompt ||
                                char.data?.extensions?.sd_api_prompt ||
                                char.extensions?.sd_character_prompt ||
                                char.extensions?.sd_api_prompt ||
                                null;
                            if (sdPrompt && sdPrompt.trim()) {
                                prompts[char.name] = sdPrompt.trim();
                            }
                        }
                    }
                }
            }
        } catch (_) {}

        return prompts;
    }

    /**
     * Build a raw narrative payload for the bridge.
     * Includes character SD prompts (trigger words), all visible character names,
     * the current message, and a short window of recent context.
     * The bridge's DeepSeek summarizer will process this — no LLM call here.
     */
    function buildRawPromptPayload(currentMessage, context, activeCharPrefix, allCharNames) {
        const parts = [];

        // Character SD prompt (trigger words) for the active character
        if (activeCharPrefix) {
            parts.push(activeCharPrefix);
        }

        // Explicit character list so bridge keyword matcher can find all LoRAs
        if (allCharNames.length > 0) {
            parts.push(`Characters present: ${allCharNames.join(', ')}`);
        }

        // Current AI message — this is the scene we're illustrating
        parts.push(currentMessage.text);

        // Add 2 preceding messages for scene continuity (capped to avoid flooding)
        const recentContext = context.slice(-3, -1);
        if (recentContext.length > 0) {
            const contextText = recentContext
                .map(m => m.content)
                .join(' ')
                .substring(0, 500);
            if (contextText.trim()) {
                parts.push(`Recent context: ${contextText}`);
            }
        }

        return parts.join('\n\n');
    }

    // ============================================
    // LLM VISUAL PROMPT GENERATION
    // ============================================

    async function generateVisualPrompt(context, characterPrefix) {
        console.log('[AutoImageGen] 🤖 Using LLM to generate visual prompt...');

        if (!CONFIG.OPENROUTER_API_KEY) {
            const lastMsg = (context && context.length > 0) ? context[context.length - 1] : { content: '' };
            const cleanContent = cleanPrompt((lastMsg.content || '').substring(0, 150));
            const fallback = cleanContent || 'portrait, cinematic lighting, detailed face, realistic skin, high detail';
            return characterPrefix ? `${characterPrefix}, ${fallback}` : fallback;
        }

        const conversationText = context.map(msg =>
            `${msg.name}: ${msg.content}`
        ).join('\n');

        const systemPrompt = `You are a visual scene descriptor for Flux image generation. Given a conversation, create a detailed visual description.

CRITICAL RULES:
1. Output ONLY in English (no other languages)
2. Focus on: pose, expression, clothing, gesture, setting, mood
3. Be specific and detailed (use descriptive adjectives)
4. Use comma-separated keywords
5. Maximum 150 words
6. Do NOT include dialogue or text
7. Do NOT include non-English characters

Format: descriptive keywords separated by commas
Example: "smiling warmly, playful expression, standing in kitchen, wearing casual dress, hands on hips, bright lighting, relaxed pose"`;

        const userPrompt = `Conversation:\n${conversationText}\n\nDescribe this scene visually in English only (pose, expression, action, setting):`;

        try {
            const response = await fetch(CONFIG.OPENROUTER_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${CONFIG.OPENROUTER_API_KEY}`,
                    'HTTP-Referer': window.location.href,
                    'X-Title': 'SillyTavern Auto-Image'
                },
                body: JSON.stringify({
                    model: CONFIG.LLM_MODEL,
                    messages: [
                        { role: 'system', content: systemPrompt },
                        { role: 'user', content: userPrompt }
                    ],
                    temperature: 0.5,
                    max_tokens: 200
                })
            });

            if (!response.ok) {
                throw new Error(`OpenRouter returned ${response.status}`);
            }

            const data = await response.json();
            let visualPrompt = data.choices[0].message.content.trim();
            visualPrompt = cleanPrompt(visualPrompt);

            const fullPrompt = characterPrefix
                ? `${characterPrefix}, ${visualPrompt}`
                : visualPrompt;

            console.log(`[AutoImageGen] ✅ LLM generated: ${visualPrompt}`);
            console.log(`[AutoImageGen] 📝 Full prompt: ${fullPrompt.substring(0, 100)}...`);

            return fullPrompt;

        } catch (e) {
            console.error('[AutoImageGen] ❌ LLM failed:', e);
            const lastMsg = (context && context.length > 0) ? context[context.length - 1] : { content: '' };
            const cleanContent = cleanPrompt((lastMsg.content || '').substring(0, 100));
            const fallback = cleanContent || 'portrait, cinematic lighting, detailed face, realistic skin, high detail';
            return characterPrefix
                ? `${characterPrefix}, ${fallback}`
                : fallback;
        }
    }

    // ============================================
    // IMAGE GENERATION & DISPLAY
    // ============================================

    async function generateImage(prompt, metadata = {}) {
        const bridgeUrl = resolveBridgeUrl();
        console.log('[AutoImageGen] 🎨 Sending to bridge...');
        console.log(`[AutoImageGen] 📝 Prompt: ${prompt.substring(0, 100)}...`);
        console.log(`[AutoImageGen] 🌉 URL: ${bridgeUrl}`);

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 90000);

        try {
            const response = await fetch(bridgeUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                signal: controller.signal,
                body: JSON.stringify({
                    prompt: prompt,
                    negative_prompt: "ugly, deformed, blurry, low quality, bad anatomy, text, watermark, words, letters, anatomical diagram, medical diagram",
                    steps: 40,
                    cfg_scale: 3.5,
                    width: 1024,
                    height: 1024,
                    seed: -1,
                    // Multi-char metadata — bridge uses these for LoRA pre-matching
                    character_prompts: metadata.characterPrompts || {},
                    visible_characters: metadata.visibleCharacters || []
                })
            });

            if (!response.ok) {
                throw new Error(`Bridge returned ${response.status}`);
            }

            const data = await response.json();
            console.log('[AutoImageGen] ✅ Image generated successfully');
            return data.images[0];

        } catch (e) {
            if (e.name === 'AbortError') {
                console.error('[AutoImageGen] ❌ Image generation timed out after 90 seconds.');
            } else if (e instanceof TypeError) {
                console.error(
                    '[AutoImageGen] ❌ Failed to reach bridge. Make sure Flux LoRA Bridge is running and reachable at:',
                    bridgeUrl
                );
            } else {
                console.error('[AutoImageGen] ❌ Image generation failed:', e);
            }
            return null;
        } finally {
            clearTimeout(timeoutId);
        }
    }

    function displayImage(base64Image, messageElement) {
        if (messageElement.querySelector('.auto-generated-image')) {
            console.log('[AutoImageGen] ⏭️ Image already displayed');
            return;
        }

        const imgContainer = document.createElement('div');
        imgContainer.className = 'auto-generated-image';

        const img = document.createElement('img');
        img.src = `data:image/png;base64,${base64Image}`;
        img.style.maxWidth = '100%';
        img.style.borderRadius = '8px';
        img.style.boxShadow = '0 2px 8px rgba(0,0,0,0.1)';
        img.style.cursor = 'pointer';
        img.onclick = () => window.open(img.src, '_blank');

        imgContainer.appendChild(img);

        const messageText = messageElement.querySelector('.mes_text');
        if (messageText) {
            messageText.parentNode.insertBefore(imgContainer, messageText.nextSibling);
        } else {
            messageElement.appendChild(imgContainer);
        }

        console.log('[AutoImageGen] 🖼️ Image displayed');
    }

    // ============================================
    // MESSAGE PROCESSING
    // ============================================

    async function processNewMessage(message) {
        if (isProcessing) {
            console.log('[AutoImageGen] ⏳ Already processing, skipping...');
            return;
        }

        isProcessing = true;

        try {
            console.log(`\n[AutoImageGen] 📝 Processing message from ${message.name}`);

            const activeCharacterName = getActiveCharacterName();
            console.log(`[AutoImageGen] 👤 Active character: ${activeCharacterName}`);

            if (CONFIG.USE_BRIDGE_SUMMARIZATION) {
                // ── New path: send raw narrative to bridge, let DeepSeek summarize ──
                const context = getChatContext(10);
                const visibleCharacters = getAllVisibleCharacterNames(context);
                const characterPrompts = getAllCharacterPrompts(visibleCharacters);
                const activeCharPrefix = getCharacterImagePrompt(activeCharacterName);

                console.log(`[AutoImageGen] 👥 Visible characters: ${visibleCharacters.join(', ')}`);
                console.log(`[AutoImageGen] 📎 Character prompts found: ${Object.keys(characterPrompts).join(', ')}`);

                const rawPrompt = buildRawPromptPayload(message, context, activeCharPrefix, visibleCharacters);
                const base64Image = await generateImage(rawPrompt, { characterPrompts, visibleCharacters });

                if (base64Image) {
                    displayImage(base64Image, message.element);
                }
            } else {
                // ── Legacy path: OpenRouter summarization in plugin ──
                const characterPrefix = getCharacterImagePrompt(activeCharacterName);
                console.log(`[AutoImageGen] 🎨 Character prefix: ${characterPrefix.substring(0, 80)}...`);

                const context = getChatContext(5);
                const visualPrompt = await generateVisualPrompt(context, characterPrefix);
                const base64Image = await generateImage(visualPrompt);

                if (base64Image) {
                    displayImage(base64Image, message.element);
                }
            }

        } catch (e) {
            console.error('[AutoImageGen] ❌ Error processing message:', e);
        } finally {
            isProcessing = false;
        }
    }

    // ============================================
    // INITIALIZATION
    // ============================================

    function init() {
        console.log('[AutoImageGen] 🚀 Initializing...');
        setupMessageObserver();
        startFallbackCheck();
        console.log('[AutoImageGen] ✅ Ready - watching for messages');
        console.log(`[AutoImageGen] ⏱️ Message completion delay: ${CONFIG.MESSAGE_COMPLETE_DELAY}ms`);
        console.log(`[AutoImageGen] 🌉 Bridge: ${resolveBridgeUrl()}`);
        console.log(`[AutoImageGen] 🤖 OpenRouter key configured: ${Boolean(CONFIG.OPENROUTER_API_KEY)}`);
        console.log(`[AutoImageGen] 🧠 Bridge summarization: ${CONFIG.USE_BRIDGE_SUMMARIZATION ? 'enabled (raw text → DeepSeek)' : 'disabled (OpenRouter in plugin)'}`);
    }

    // Wait for page load
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
