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
        MIN_MESSAGE_LENGTH: 20
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

    function collectCandidateBridgeUrls() {
        const context = (typeof SillyTavern !== 'undefined' && SillyTavern.getContext)
            ? SillyTavern.getContext()
            : null;

        const candidates = [
            context?.extensionSettings?.sd?.url,
            context?.extensionSettings?.sd?.webui_url,
            context?.extensionSettings?.sd?.api_url,
            context?.extensionSettings?.sd?.sd_url,
            context?.extension_settings?.sd?.url,
            context?.extension_settings?.sd?.webui_url,
            window?.extension_settings?.sd?.url,
            window?.extension_settings?.sd?.webui_url,
            window?.extension_settings?.sd?.api_url,
            window?.extension_settings?.sd?.sd_url,
            window?.SillyTavern?.getContext?.()?.extensionSettings?.sd?.url,
            window?.SillyTavern?.getContext?.()?.extensionSettings?.sd?.webui_url,
            document.querySelector('#sd_webui_url')?.value,
            document.querySelector('input[name="sd_webui_url"]')?.value,
            document.querySelector('#extensions_settings2 #sd_webui_url')?.value
        ];

        const likelyUrlInputs = Array.from(document.querySelectorAll('input[type="text"], input:not([type])'));
        for (const input of likelyUrlInputs) {
            const hint = `${input.id || ''} ${input.name || ''} ${input.placeholder || ''} ${input.className || ''}`.toLowerCase();
            if ((hint.includes('sd') || hint.includes('webui') || hint.includes('bridge') || hint.includes('url')) && input.value) {
                candidates.push(input.value);
            }
        }

        return candidates;
    }

    function normalizeBridgeUrl(rawUrl) {
        if (!rawUrl || typeof rawUrl !== 'string') {
            return null;
        }

        const trimmed = rawUrl.trim().replace(/\/+$/, '');
        if (!trimmed || !/^https?:\/\//i.test(trimmed)) return null;

        if (trimmed.endsWith('/sdapi/v1/txt2img')) {
            return trimmed;
        }

        return `${trimmed}/sdapi/v1/txt2img`;
    }

    function isLocalhostUrl(url) {
        try {
            const parsed = new URL(url);
            return ['localhost', '127.0.0.1', '::1'].includes(parsed.hostname);
        } catch (_) {
            return false;
        }
    }

    function deriveHostedBridgeUrl() {
        const host = window.location.hostname;
        if (!host || ['localhost', '127.0.0.1', '::1'].includes(host)) return null;

        if (host.endsWith('midnighttavern.online')) {
            return `${window.location.protocol}//bridge.midnighttavern.online/sdapi/v1/txt2img`;
        }

        return null;
    }

    function resolveBridgeTxt2ImgUrl() {
        const rawCandidates = [
            ...collectCandidateBridgeUrls(),
            CONFIG.BRIDGE_URL,
            DEFAULT_CONFIG.BRIDGE_URL
        ];

        const normalizedCandidates = rawCandidates
            .map(normalizeBridgeUrl)
            .filter(Boolean);

        const isHostedPage = !['localhost', '127.0.0.1', '::1'].includes(window.location.hostname);

        if (isHostedPage) {
            const remoteCandidate = normalizedCandidates.find((url) => !isLocalhostUrl(url));
            if (remoteCandidate) {
                return remoteCandidate;
            }

            const derivedHostedBridge = deriveHostedBridgeUrl();
            if (derivedHostedBridge) {
                return derivedHostedBridge;
            }
        }

        return normalizedCandidates[0] || DEFAULT_CONFIG.BRIDGE_URL;
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

    async function generateImage(prompt) {
        console.log('[AutoImageGen] 🎨 Sending to bridge...');
        console.log(`[AutoImageGen] 📝 Prompt: ${prompt.substring(0, 100)}...`);
        const bridgeTxt2ImgUrl = resolveBridgeTxt2ImgUrl();
        console.log(`[AutoImageGen] 🌉 URL: ${bridgeTxt2ImgUrl}`);

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000);

        try {
            const response = await fetch(bridgeTxt2ImgUrl, {
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
                    seed: -1
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
                console.error('[AutoImageGen] ❌ Image generation timed out after 30 seconds.');
            } else if (e instanceof TypeError) {
                console.error(
                    '[AutoImageGen] ❌ Failed to reach bridge. Make sure Flux LoRA Bridge is running and reachable at:',
                    bridgeTxt2ImgUrl
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

            const characterPrefix = getCharacterImagePrompt(activeCharacterName);
            console.log(`[AutoImageGen] 🎨 Character prefix: ${characterPrefix.substring(0, 80)}...`);

            const context = getChatContext(5);
            const visualPrompt = await generateVisualPrompt(context, characterPrefix);
            const base64Image = await generateImage(visualPrompt);

            if (base64Image) {
                displayImage(base64Image, message.element);
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
        console.log(`[AutoImageGen] 🌉 Bridge: ${resolveBridgeTxt2ImgUrl()}`);
        console.log(`[AutoImageGen] 🤖 OpenRouter key configured: ${Boolean(CONFIG.OPENROUTER_API_KEY)}`);
    }

    // Wait for page load
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
