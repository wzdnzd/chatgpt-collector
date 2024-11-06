/**
 * Filename: worker.js
 * Description: Cloudflare Worker Script for converting NextChat API to standard OpenAI API
 * Date: 2024-03-27
 * Version: 0.0.1
 * Author: wzdnzd
 * License: MIT License
 * 
 * usage:
 * 1. create a KV namespace and bind it to the script variable `openAPIs`
 * 2. add a secret key named `SECRET_KEY = 'your-secret-key'` as environment variable
 * 3. add the environment variable `GPT35_ONLY`, with a value of true or false
 * 4. deploy the script to Cloudflare Worker
 * 5. add NextChat API endpoints and access tokens to the KV namespace
 * 6. use the Cloudflare Worker URL as the API endpoint and add the secret key set in step 2 as the Authorization header
 * 7. use ${CLOUDFLARE_WORKER_URL}/v1/chat/completions as the request URL, and then send the request to the endpoint with the same body
 */

addEventListener('fetch', event => {
    event.respondWith(handleRequest(event.request))
});

addEventListener('scheduled', event => {
    event.waitUntil(doSomeTaskOnASchedule());
});

async function doSomeTaskOnASchedule() {
    await handleSyncFromRemote();
}

const KV = openAPIs;
const maxRetries = 3;
const gpt35Only = (GPT35_ONLY || '').trim().toLowerCase() === 'true';
const userAgent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36';

async function handleRequest(request) {
    const corsHeaders = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'OPTIONS,POST,GET',
        'Access-Control-Allow-Headers': '*',
    };

    if (request.method === 'OPTIONS') {
        return new Response(null, { headers: corsHeaders });
    }

    const accessToken = request.headers.get('Authorization');
    if (!accessToken
        || !accessToken.startsWith('Bearer ')
        || accessToken.substring(7) !== (SECRET_KEY || '')) {
        return new Response(JSON.stringify({ message: 'Unauthorized', success: false }), {
            status: 401,
            headers: { 'Content-Type': 'application/json' }
        });
    }

    const url = new URL(request.url);
    let response;

    if (url.pathname === '/v1/models' && request.method === 'GET') {
        response = await handleListModels();
    } else if (url.pathname === '/v1/chat/completions' && request.method === 'POST') {
        response = await handleProxy(request);
    } else if (url.pathname === '/v1/sync' && request.method === 'POST') {
        response = await handleSyncFromRemote();
    } else {
        response = new Response(JSON.stringify({ message: 'Invalid request method or path', success: false }), {
            status: 405,
            headers: { 'Content-Type': 'application/json' }
        });
    }

    return response;
}

async function handleListModels() {
    // list and return all openai models
    return new Response(JSON.stringify({
        "data": listSupportModels(),
        "object": "list"
    }), {
        status: 200, headers: {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': '*',
            'Content-Type': 'application/json',
        }
    });
}

async function handleProxy(request) {
    const keys = await KV.list();
    const count = keys.keys.length;

    if (count <= 0) {
        return new Response(JSON.stringify({ message: 'Service is temporarily unavailable', success: false }), {
            status: 503,
            headers: { 'Content-Type': 'application/json' }
        });
    }

    const requestBody = await request.json();
    const isStreamReq = requestBody.stream || false;

    if (gpt35Only) {
        requestBody.model = "gpt-3.5-turbo";
    }

    const headers = new Headers(request.headers);

    // add custom headers
    headers.set('Content-Type', 'application/json');
    headers.set('Path', 'v1/chat/completions');
    headers.set('Accept-Language', 'zh-CN,zh;q=0.9');
    headers.set('Accept-Encoding', 'gzip, deflate, br, zstd');
    headers.set('Accept', 'application/json, text/event-stream');
    headers.set('User-Agent', userAgent);

    let response;
    let targetURL;

    for (let retry = 0; retry < maxRetries; retry++) {
        requestBody.stream = isStreamReq;

        targetURL = keys.keys[Math.floor(Math.random() * count)].name;
        const content = (await KV.get(targetURL) || '').trim();
        let accessToken;

        if (content) {
            try {
                const data = JSON.parse(content);

                // target service don't support stream
                const enableStream = data?.stream || data?.stream === undefined;
                if (!enableStream) {
                    requestBody.stream = false;
                }

                // use default model instead of the request model if default model is set
                if (data.defaultModel) {
                    requestBody.model = data.defaultModel;
                }

                const text = data?.token || '';
                const accessTokens = text.trim()
                    .split(',')
                    .map(s => s.trim())
                    .filter(s => s !== '');

                if (accessTokens.length > 0) {
                    const index = Math.floor(Math.random() * accessTokens.length);
                    accessToken = accessTokens[index];
                }
            } catch (error) {
                console.error(`Invalid config for ${targetURL}: `, error);
            }
        }

        let proxyURL = targetURL;
        if (!proxyURL.endsWith('/api/chat-stream')
            && !proxyURL.endsWith('/v1/chat/completions')) {
            const url = new URL(proxyURL);
            if (!url.pathname.startsWith('/api/openai')) {
                proxyURL += '/api/chat-stream';
            } else {
                proxyURL += '/v1/chat/completions';
            }
        }

        headers.set('Referer', targetURL + '/');
        headers.set('Origin', targetURL);

        // remove old authorization header if exist
        headers.delete('Authorization');
        if (accessToken) {
            headers.set('Authorization', `Bearer ${accessToken}`);
        }

        try {
            response = await fetch(proxyURL, {
                method: 'POST',
                headers: headers,
                body: JSON.stringify(requestBody),
            });

            if (response && response.ok) {
                break;
            }
        } catch (error) {
            console.error(`Error during fetch with ${targetURL}: `, error);
        }
    }

    // no valid response after retries
    if (!response) {
        return new Response(JSON.stringify({ message: 'Internal server error', success: false }), {
            status: 500,
            headers: { 'Content-Type': 'application/json' }
        });
    }

    // return the original response
    const newHeaders = new Headers(response.headers);
    newHeaders.set('Access-Control-Allow-Origin', '*');
    newHeaders.set('Access-Control-Allow-Methods', '*');

    let newBody = response.body;
    const contentType = response.headers.get('Content-Type') || '';

    if (response.status === 200) {
        if (isStreamReq && !contentType.includes('text/event-stream')) {
            // need text/event-stream but got others
            if (contentType.includes('application/json')) {
                try {
                    const data = await response.json();
                    const model = data?.model || requestBody.model || 'gpt-3.5-turbo';

                    // 'chatcmpl-'.length = 10
                    const messageId = (data?.id || '').slice(10);

                    const choices = data?.choices
                    if (!choices || choices.length === 0) {
                        return new Response(JSON.stringify({ message: 'No effective response', success: false }), {
                            status: 503,
                            headers: { 'Content-Type': 'application/json' }
                        });
                    }

                    const record = choices[0].message || {};
                    const message = record?.content || '';
                    const content = transformToJSON(message, model, messageId);
                    const text = `data: ${content}\n\ndata: [Done]`;

                    newBody = new ReadableStream({
                        start(controller) {
                            controller.enqueue(new TextEncoder().encode(text));
                            controller.close();
                        }
                    });

                } catch (error) {
                    console.error(`Error during parse response body with ${targetURL}, error: `, error);
                    return new Response(JSON.stringify({ message: 'Internal server error', success: false }), {
                        status: 500,
                        headers: { 'Content-Type': 'application/json' }
                    });
                }
            } else {
                const { readable, writable } = new TransformStream();

                // transform chunk data to event-stream
                streamResponse(response, writable, requestBody.model, generateUUID());
                newBody = readable;
            }

            newHeaders.set('Content-Type', 'text/event-stream');
        } else if (!isStreamReq && !contentType.includes('application/json')) {
            // need application/json
            try {
                const content = (await response.text())
                    .replace(/^\`\`\`json\n/, "").replace(/\n\`\`\`$/, "");

                // compress json data
                const text = JSON.stringify(JSON.parse(content));
                newBody = new ReadableStream({
                    start(controller) {
                        controller.enqueue(new TextEncoder().encode(text));
                        controller.close();
                    }
                });

                newHeaders.set('Content-Type', 'application/json');
            } catch (error) {
                console.error(`Error during parse response body with ${targetURL}, Content-Type: ${contentType}, error: `, error);
                return new Response(JSON.stringify({ message: 'Internal server error', success: false }), {
                    status: 500,
                    headers: { 'Content-Type': 'application/json' }
                });
            }
        }
    }

    const newResponse = new Response(newBody, {
        ...response,
        headers: newHeaders
    });

    return newResponse;
}

async function handleSyncFromRemote() {
    const subscribeLink = (REMOTE_SUBLINK || '').trim();
    if (!subscribeLink) {
        return new Response(JSON.stringify({ message: 'Skip sync due to no subscribe link found', success: true }), { status: 200 });
    }

    const headers = {
        'User-Agent': userAgent,
        'Accept': '*/*',
        'Accept-Encoding': 'gzip, deflate, br, zstd'
    }
    const content = await fetchRemoteLinks(subscribeLink, headers, 3, 250);
    if (!content) {
        return new Response(JSON.stringify({
            message: 'Failed to sync because of fetch remote data error',
            success: false
        }), { status: 500 });
    }

    // split with comma
    const targets = mergeMultiTokens(content.split(',').filter(s => s.trim() !== ''));
    if (targets.length <= 0) {
        return new Response(JSON.stringify({
            message: 'Failed to sync due to remote data is empty',
            success: false
        }), { status: 500 });
    }

    const apiPaths = new Set();
    for (const t of targets) {
        try {
            const result = parseURL(t);
            if (!result) continue;

            // write to kv namespace
            await KV.put(result.apiPath, JSON.stringify({ stream: result.stream, defaultModel: result.defaultModel, unstable: result.unstable, token: result.token }));
            apiPaths.add(result.apiPath);
        } catch {
            console.warn(`Storage to KV failed, url: ${t}`);
        }
    }

    // remove invalid data
    const keys = await KV.list();
    for (let key of keys.keys) {
        let deleted = false;

        if (!apiPaths.has(key.name)) {
            const text = await KV.get(key.name);
            try {
                deleted = JSON.parse(text).unstable || false;
            } catch (error) {
                console.warn(`Found invalid config and will be deleted, key: ${key.name}`);
                deleted = true;
            }
        }

        if (deleted) {
            await KV.delete(key.name);
        }
    }

    return new Response(JSON.stringify({ message: `sync finished, found ${apiPaths.size} data`, success: true }), { status: 200 });
}

function listSupportModels() {
    if (gpt35Only) {
        return [
            {
                "id": "gpt-3.5-turbo",
                "object": "model",
                "created": 1685474247,
                "owned_by": "openai",
                "permission": [],
                "root": "gpt-3.5-turbo",
                "parent": null
            }
        ]
    }

    return [
        {
            "id": "dall-e-2",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "dall-e-2",
            "parent": null
        },
        {
            "id": "dall-e-3",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "dall-e-3",
            "parent": null
        },
        {
            "id": "whisper-1",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "whisper-1",
            "parent": null
        },
        {
            "id": "tts-1",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "tts-1",
            "parent": null
        },
        {
            "id": "tts-1-1106",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "tts-1-1106",
            "parent": null
        },
        {
            "id": "tts-1-hd",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "tts-1-hd",
            "parent": null
        },
        {
            "id": "tts-1-hd-1106",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "tts-1-hd-1106",
            "parent": null
        },
        {
            "id": "gpt-3.5-turbo",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "gpt-3.5-turbo",
            "parent": null
        },
        {
            "id": "gpt-3.5-turbo-0301",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "gpt-3.5-turbo-0301",
            "parent": null
        },
        {
            "id": "gpt-3.5-turbo-0613",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "gpt-3.5-turbo-0613",
            "parent": null
        },
        {
            "id": "gpt-3.5-turbo-16k",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "gpt-3.5-turbo-16k",
            "parent": null
        },
        {
            "id": "gpt-3.5-turbo-16k-0613",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "gpt-3.5-turbo-16k-0613",
            "parent": null
        },
        {
            "id": "gpt-3.5-turbo-1106",
            "object": "model",
            "created": 1699593571,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "gpt-3.5-turbo-1106",
            "parent": null
        },
        {
            "id": "gpt-3.5-turbo-instruct",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "gpt-3.5-turbo-instruct",
            "parent": null
        },
        {
            "id": "gpt-4",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "gpt-4",
            "parent": null
        },
        {
            "id": "gpt-4-0314",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "gpt-4-0314",
            "parent": null
        },
        {
            "id": "gpt-4-0613",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "gpt-4-0613",
            "parent": null
        },
        {
            "id": "gpt-4-32k",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "gpt-4-32k",
            "parent": null
        },
        {
            "id": "gpt-4-32k-0314",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "gpt-4-32k-0314",
            "parent": null
        },
        {
            "id": "gpt-4-32k-0613",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "gpt-4-32k-0613",
            "parent": null
        },
        {
            "id": "gpt-4-1106-preview",
            "object": "model",
            "created": 1699593571,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "gpt-4-1106-preview",
            "parent": null
        },
        {
            "id": "gpt-4-vision-preview",
            "object": "model",
            "created": 1699593571,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "gpt-4-vision-preview",
            "parent": null
        },
        {
            "id": "text-embedding-ada-002",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "text-embedding-ada-002",
            "parent": null
        },
        {
            "id": "text-davinci-003",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "text-davinci-003",
            "parent": null
        },
        {
            "id": "text-davinci-002",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "text-davinci-002",
            "parent": null
        },
        {
            "id": "text-curie-001",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "text-curie-001",
            "parent": null
        },
        {
            "id": "text-babbage-001",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "text-babbage-001",
            "parent": null
        },
        {
            "id": "text-ada-001",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "text-ada-001",
            "parent": null
        },
        {
            "id": "text-moderation-latest",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "text-moderation-latest",
            "parent": null
        },
        {
            "id": "text-moderation-stable",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "text-moderation-stable",
            "parent": null
        },
        {
            "id": "text-davinci-edit-001",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "text-davinci-edit-001",
            "parent": null
        },
        {
            "id": "code-davinci-edit-001",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "code-davinci-edit-001",
            "parent": null
        },
        {
            "id": "davinci-002",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "davinci-002",
            "parent": null
        },
        {
            "id": "babbage-002",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai",
            "permission": [
                {
                    "id": "modelperm-LwHkVFn8AcMItP432fKKDIKJ",
                    "object": "model_permission",
                    "created": 1626777600,
                    "allow_create_engine": true,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ],
            "root": "babbage-002",
            "parent": null
        }
    ]
}

function transformToJSON(text, model, messageId) {
    return JSON.stringify({
        'id': `chatcmpl-${messageId}`,
        "object": "chat.completion.chunk",
        'model': model || 'gpt-3.5-turbo',
        "created": Math.floor(Date.now() / 1000),
        'choices': [{
            "index": 0,
            "delta": {
                "content": text || ''
            },
            "logprobs": null,
            "finish_reason": null
        }],
        "system_fingerprint": null
    });
}

function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
        const r = (Math.random() * 16) | 0;
        const v = c === 'x' ? r : (r & 0x3) | 0x8;
        return v.toString(16);
    });
}

async function streamResponse(response, writable, model, messageId) {
    const reader = response.body.getReader();
    const writer = writable.getWriter();
    const encoder = new TextEncoder();
    const decoder = new TextDecoder("utf-8");

    function push() {
        reader.read().then(({ done, value }) => {
            if (done) {
                writer.close();
                return;
            }

            const chunk = decoder.decode(value, { stream: true });
            const toSend = `data: ${transformToJSON(chunk, model, messageId)}\n\n`;

            writer.write(encoder.encode(toSend));
            push();
        }).catch(error => {
            console.error(error);
            writer.close();
        });
    }

    push();
}

async function fetchRemoteLinks(url, headers, retries = 3, delay = 250) {
    try {
        const response = await fetch(url, { headers: headers });
        if (!response.ok) {
            throw new Error('Fetch remote links failed');
        }

        return await response.text();
    } catch (error) {
        if (retries > 1) {
            console.warn(`Failed to request network, will retry after ${delay / 1000} seconds`);

            await new Promise(resolve => setTimeout(resolve, delay));
            return fetchRemoteLinks(url, retries - 1, delay);
        } else {
            console.error('Maximum number of failed retries reached');
            return '';
        }
    }
}

function mergeMultiTokens(urls) {
    if (!Array.isArray(urls) || urls.length === 0) {
        return [];
    }

    // merge urls with the same apiPath but different tokens into a single url, with the tokens concatenated by commas
    const map = new Map();
    for (const url of urls) {
        const target = parseURL(url)
        if (!target) continue;

        if (map.has(target.apiPath)) {
            const item = map.get(target.apiPath);
            if (item.token !== target.token) {
                item.token += ',' + target.token;
            }
        } else {
            map.set(target.apiPath, { token: target.token, stream: target.stream, defaultModel: target.defaultModel, unstable: target.unstable });
        }
    }

    const result = [];
    for (const [apiPath, item] of map.entries()) {
        const u = new URL(apiPath);
        u.searchParams.set("token", item.token);
        u.searchParams.set("unstable", item.unstable.toString());
        u.searchParams.set("stream", item.stream.toString());
        u.searchParams.set("model", item.defaultModel);
        result.push(u.toString());
    }

    return result;
}

function parseURL(url) {
    if (!url || typeof url !== 'string') return {};

    try {
        const u = new URL(url);
        const apiPath = u.origin + u.pathname;
        const token = u.searchParams.get("token") || '';
        const stream = (u.searchParams.get("stream") || 'true').toLowerCase() === 'true';
        const defaultModel = u.searchParams.get("model") || '';

        const category = (u.searchParams.get("unstable") || '').toLowerCase();
        const unstable = category === '' || category === 'true';

        return { apiPath, token, stream, defaultModel, unstable };
    } catch {
        console.error(`Ignore invalid link: ${url}`);
        return {};
    }
}