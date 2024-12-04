# -*- coding: utf-8 -*-

# @Author  : wzdnzd
# @Time    : 2024-04-07

import asyncio
import base64
import gzip
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from urllib import error, parse, request

import aiofiles
import aiohttp
from tqdm import tqdm

import utils
from logger import logger
from urlvalidator import isurl

# 默认检测关键词
DEFAULT_KEYWORD = "ChatGPT is"

# 默认问题
DEFAULT_QUESTION = f"Tell me what ChatGPT is in English, your answer should contain a maximum of 20 words and must start with '{DEFAULT_KEYWORD}'!"

# 默认接口
DEFAULT_API = "/v1/chat/completions"

# 常见接口集合
COMMON_APIS = [
    "/v1/chat/completions",
    "/api/openai/v1/chat/completions",
    "/api/chat-stream",
    "/api/chat/openai",
]


@dataclass
class CheckResult(object):
    # 是否可用
    available: bool

    # 是否应该停止尝试
    terminate: bool = False

    # 是否支持流式响应
    stream: bool = False

    # 支持的模型版本
    version: int = 0

    # 模型不存在
    notfound: bool = False


@dataclass
class RequestParams(object):
    # 请求地址
    url: str = ""

    # 是否支持流式输出
    stream: int = -1

    # 模型版本
    version: int = -1

    # 访问令牌
    token: str = ""

    # 模型名称
    model: str = "gpt-3.5-turbo"

    # 模型映射
    mapping: dict = None


def get_headers(domain: str, token: str = "", customize_headers: dict = None) -> dict:
    # TODO: need new method to generate headers due to this method is deprecated
    def create_jwt(access_code: str = "") -> str:
        """see: https://github.com/lobehub/lobe-chat/blob/main/src/utils/jwt.ts"""
        now = int(time.time())
        payload = {
            "accessCode": utils.trim(access_code),
            "apiKey": "",
            "endpoint": "",
            "iat": now,
            "exp": now + 100,
            "userId": str(uuid.uuid4()),
        }

        text = base64.b64encode(json.dumps(payload).encode()).decode()
        return f"http_nosafe.{text}"

    headers = {
        "Content-Type": "application/json",
        "Path": "v1/chat/completions",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json, text/event-stream",
        "Referer": domain,
        "Origin": domain,
        "User-Agent": utils.USER_AGENT,
    }

    token = utils.trim(token)

    # TODO: need enhance detect whether the domain is lobe-chat
    # model provider see: https://github.com/lobehub/lobe-chat/blob/main/src/libs/agent-runtime/types/type.ts
    if re.match(r".*/(web)?api/chat/(openai|bedrock)$", domain, flags=re.I):
        # lobe-chat
        headers["X-Lobe-Chat-Auth"] = create_jwt(access_code=token)
    elif token:
        headers["Authorization"] = f"Bearer {token}"

    if isinstance(customize_headers, dict):
        headers.update(customize_headers)

    return headers


def get_payload(question: str, model: str, stream: bool = True, style: int = 0) -> dict:
    question = utils.trim(question) or DEFAULT_QUESTION

    payload = {
        "model": model or "gpt-3.5-turbo",
        "temperature": 0,
        "max_tokens": 1000,
        "top_p": 1,
        "frequency_penalty": 1,
        "presence_penalty": 1,
        "stream": stream,
    }

    if style == 1:
        # microsoft azure style
        payload["prompt"] = question
        payload["options"] = {}
    else:
        payload["messages"] = [{"role": "user", "content": question}]

    return payload


def get_model_name(provider: str, version: int) -> str:
    provider = utils.trim(provider).lower()
    if provider.startswith("claude"):
        return "claude-3-opus-20240229" if version == 4 else "claude-3-haiku-20240307"

    return "gpt-4-turbo" if version == 4 else "gpt-3.5-turbo"


def get_model_version(model: str, soft: bool = False, opposite: bool = False) -> int:
    model = utils.trim(model).lower()
    if not model:
        return 0

    if model.startswith("gpt-"):
        if model == "gpt-4o-mini":
            return 3
        return 4 if model.startswith("gpt-4") else 3
    elif model.startswith("claude-"):
        return 4 if re.match(r"claude-3-(opus|5-sonnet)-", model) else 3

    if soft and re.search(r"-4", model) is not None:
        return 4

    return 3 if not opposite else 0


def need_check_version(model: str) -> bool:
    model = utils.trim(model).lower()
    return (model.startswith("gpt-") or model.startswith("claude-")) and get_model_version(model=model) == 3


def support_version() -> list[int]:
    return [3, 4]


def parse_url(url: str) -> RequestParams:
    address, stream, version, token, model, mapping = "", -1, -1, "", "", None

    url = utils.trim(url)
    if url:
        try:
            result = parse.urlparse(url)
            address = f"{result.scheme}://{result.netloc}{result.path}"

            if result.query:
                params = {k: v[0] for k, v in parse.parse_qs(result.query).items()}

                if "stream" in params:
                    stream = 1 if utils.trim(params.get("stream")).lower() in ["true", "1"] else 0

                if "version" in params:
                    text = utils.trim(params.get("version")).lower()
                    version = int(text) if text.isdigit() else -1

                if "token" in params:
                    token = utils.trim(params.get("token"))

                if "model" in params:
                    text = utils.trim(params.get("model"))
                    model = text if text else model

                if "mapping" in params:
                    text = utils.trim(params.get("mapping"))
                    words = text.split(",")
                    mapping = dict()

                    for word in words:
                        item = utils.trim(word)
                        if not item or ":" not in item:
                            continue

                        w1, w2 = item.split(":", maxsplit=1)
                        old, new = utils.trim(w1), utils.trim(w2)
                        if old and new:
                            mapping[old] = new
        except:
            logger.error(f"[Interactive] invalid url: {url}")

    return RequestParams(url=address, stream=stream, version=version, token=token, model=model, mapping=mapping)


def get_urls(domain: str, potentials: str = "", wander: bool = False) -> list[str]:
    """
    Generate candidate APIs

    Parameters
    ----------
    domain : str
        The target site domain
    potentials : str
        The potential APIs to be tested, multiple APIs separated by commas
    wander : bool
         Whether to use common APIs for testing

    Returns
    -------
    list[str]
        Targets to be checked

    """
    domain = utils.trim(domain)
    if not domain:
        return []

    try:
        result = parse.urlparse(domain)
        urls = list()

        if not result.path or result.path == "/":
            subpaths = list(set([x.strip().lower() for x in utils.trim(potentials).split(",") if x]))
            if not subpaths:
                if wander:
                    subpaths = COMMON_APIS
                else:
                    subpaths = [DEFAULT_API or "/v1/chat/completions"]

            for subpath in subpaths:
                url = parse.urljoin(domain, subpath)
                urls.append(url)
        else:
            urls.append(domain)

        return urls
    except:
        logger.error(f"[Interactive] skip due to invalid url: {domain}")
        return []


def parse_headers(content: str, delimiter: str = "|") -> dict:
    if not isinstance(delimiter, str) or delimiter == "":
        logger.error("[Interactive] cannot parse headers due to invalid delimiter")
        return None

    content = utils.trim(content)
    groups = content.split(delimiter)
    headers = dict()

    for group in groups:
        text = utils.trim(group)
        words = text.split(":", maxsplit=1)
        if not words or len(words) != 2:
            continue

        k = utils.trim(words[0])
        v = utils.trim(words[1])

        if k and v:
            headers[k] = v

    return None if not headers else headers


def chat(
    url: str,
    question: str = "",
    keyword: str = "",
    model: str = "gpt-3.5-turbo",
    stream: bool = False,
    token: str = "",
    retry: int = 3,
    timeout: int = 6,
    style: int = 0,
    headers: dict = None,
    strict: bool = True,
) -> CheckResult:
    if not isurl(url=url):
        return CheckResult(available=False, terminate=True)

    customize_headers = get_headers(domain=url, token=token, customize_headers=headers)
    model = utils.trim(model) or "gpt-3.5-turbo"
    retry = 3 if retry < 0 else retry
    timeout = 15 if timeout <= 0 else timeout

    payload = get_payload(question=question, model=model, stream=stream, style=style)
    data = json.dumps(payload).encode(encoding="UTF8")

    response, count = None, 0
    while not response and count < retry:
        try:
            opener = request.build_opener(utils.NoRedirect)
            response = opener.open(
                request.Request(
                    url=url,
                    data=data,
                    headers=customize_headers,
                    method="POST",
                ),
                timeout=timeout,
            )
        except error.HTTPError as e:
            if e.code == 400:
                return CheckResult(available=False, terminate=True)

            try:
                content = e.read().decode("UTF8")
            except UnicodeDecodeError:
                content = gzip.decompress(e.read()).decode("UTF8")
            except:
                content = ""

            if e.code == 401:
                terminate = not not_ternimate(content=content)
                if not terminate:
                    # authorization success but no provider, see: https://github.com/lobehub/lobe-chat/blob/main/src/app/api/chat/%5Bprovider%5D/route.ts
                    logger.warning(f"[Interactive] authorization success but no provider, url: {url}, model: {model}")
                return CheckResult(available=False, terminate=terminate)

            if no_model(content=content, model=model):
                return CheckResult(available=False, terminate=True, notfound=True)
        except error.URLError as e:
            return CheckResult(available=False, terminate=True)
        except Exception:
            pass

        count += 1

    if not response or response.getcode() != 200:
        return CheckResult(available=False)

    try:
        text = response.read()
    except:
        text = b""

    try:
        content = text.decode(encoding="UTF8")
    except UnicodeDecodeError:
        content = gzip.decompress(text).decode("UTF8")
    except:
        content = ""

    content_type = response.headers.get("content-type", "")
    keyword = utils.trim(keyword) or DEFAULT_KEYWORD
    return verify(
        content=content,
        content_type=content_type,
        stream=stream,
        model=model,
        keyword=keyword,
        strict=strict,
    )


async def chat_async(
    url: str,
    question: str = "",
    keyword: str = "",
    model: str = "gpt-3.5-turbo",
    stream: bool = False,
    token: str = "",
    retry: int = 3,
    timeout: int = 6,
    interval: float = 3,
    style: int = 0,
    headers: dict = None,
    strict: bool = True,
) -> CheckResult:
    if not isurl(url=url):
        return CheckResult(available=False, terminate=True)
    if retry <= 0:
        return CheckResult(available=False)

    model = model.strip() or "gpt-3.5-turbo"
    customize_headers = get_headers(domain=url, token=token, customize_headers=headers)
    payload = get_payload(question=question, model=model, stream=stream, style=style)
    timeout = 6 if timeout <= 0 else timeout
    interval = max(0, interval)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=customize_headers, timeout=timeout) as response:
                if response.status != 200:
                    terminal = response.status in [400, 401]
                    try:
                        content = await response.text()
                        notfound = no_model(content=content, model=model)

                        # authorization success but no provider, see: https://github.com/lobehub/lobe-chat/blob/main/src/app/api/chat/%5Bprovider%5D/route.ts
                        if response.status == 401 and not_ternimate(content=content):
                            logger.warning(
                                f"[Interactive] authorization success but no provider, url: {url}, model: {model}"
                            )
                            terminal = False
                    except asyncio.TimeoutError:
                        notfound = False

                    return CheckResult(available=False, terminate=terminal, notfound=notfound)

                content = ""
                try:
                    content = await response.text()
                except asyncio.TimeoutError:
                    await asyncio.sleep(interval)
                    return await chat_async(
                        url=url,
                        question=question,
                        keyword=keyword,
                        model=model,
                        stream=stream,
                        token=token,
                        retry=retry - 1,
                        timeout=min(timeout + 10, 90),
                        interval=interval,
                        style=style,
                        headers=headers,
                        strict=strict,
                    )

                content_type = response.headers.get("content-type", "")
                keyword = utils.trim(keyword) or DEFAULT_KEYWORD
                return verify(
                    content=content,
                    content_type=content_type,
                    stream=stream,
                    model=model,
                    keyword=keyword,
                    strict=strict,
                )
    except:
        await asyncio.sleep(interval)
        return await chat_async(
            url=url,
            question=question,
            keyword=keyword,
            model=model,
            stream=stream,
            token=token,
            retry=retry - 1,
            timeout=timeout,
            interval=interval,
            style=style,
            headers=headers,
            strict=strict,
        )


def check(
    domain: str,
    filename: str = "",
    potentials: str = "",
    wander: bool = False,
    question: str = "",
    keyword: str = "",
    model: str = "gpt-3.5-turbo",
    detect: bool = True,
    style: int = 0,
    headers: dict = None,
    strict: bool = True,
) -> str:
    target, urls = "", get_urls(domain=domain, potentials=potentials, wander=wander)
    stream, version, params, real_model = False, 0, None, ""

    for url in urls:
        params = parse_url(url=url)
        if not params or not params.url:
            continue

        real_model = params.model or model or "gpt-3.5-turbo"

        # available check
        result = chat(
            url=params.url,
            question=question,
            keyword=keyword,
            model=real_model,
            stream=params.stream == 1,
            token=params.token,
            timeout=30,
            style=style,
            headers=headers,
            strict=strict,
        )
        if result.available:
            target, stream, version = params.url, result.stream, result.version
            break
        elif result.notfound:
            logger.warning(f"[Interactive] url {params.url} available but no {real_model} model exists")
            break
        elif result.terminate:
            # if terminal is true, all attempts should be aborted immediately
            break

    if not target:
        return ""

    if detect:
        if params.stream == -1:
            # stream support check
            result = chat(
                url=target,
                question=question,
                keyword=keyword,
                model=real_model,
                stream=True,
                token=params.token,
                timeout=30,
                style=style,
                headers=headers,
                strict=strict,
            )
            stream = result.stream or stream
        else:
            stream = stream or params.stream == 1

        if params.version == -1 and need_check_version(model=real_model):
            # version check
            result = chat(
                url=target,
                question=question,
                keyword=keyword,
                model=get_model_name(provider=real_model, version=4),
                stream=False,
                token=params.token,
                timeout=30,
                style=style,
                headers=headers,
                strict=strict,
            )
            version = result.version if result.available else version
        else:
            version = params.version

    target = concat_url(
        url=target,
        stream=stream,
        version=version,
        token=params.token,
        model=params.model,
        mapping=params.mapping,
    )
    filename = utils.trim(filename)

    if target and filename:
        utils.write_file(filename=filename, lines=target, overwrite=False)

    return target


def check_concurrent(
    sites: list[str],
    filename: str = "",
    potentials: str = "",
    wander: bool = False,
    question: str = "",
    keyword: str = "",
    model: str = "gpt-3.5-turbo",
    num_threads: int = -1,
    show_progress: bool = False,
    index: int = 0,
    style: int = 0,
    headers: str = "",
    strict: bool = True,
) -> list[str]:
    if not sites or not isinstance(sites, list):
        logger.warning(f"[Interactive] skip process due to lines is empty")
        return []

    filename = utils.trim(filename)
    model = utils.trim(model) or "gpt-3.5-turbo"
    index = max(0, index)

    # parse customize headers
    customize_headers = parse_headers(content=headers, delimiter="|")

    tasks = [
        [x, filename, potentials, wander, question, keyword, model, True, style, customize_headers, strict]
        for x in sites
        if x
    ]

    result = utils.multi_thread_run(
        func=check,
        tasks=tasks,
        num_threads=num_threads,
        show_progress=show_progress,
        description=f"Chunk-{index}",
    )

    return [x for x in result if x]


async def check_async(
    sites: list[str],
    filename: str = "",
    potentials: str = "",
    wander: bool = False,
    question: str = "",
    keyword: str = "",
    model: str = "gpt-3.5-turbo",
    concurrency: int = 512,
    show_progress: bool = True,
    style: int = 0,
    headers: str = "",
    strict: bool = True,
) -> list[str]:
    async def checkone(
        domain: str,
        filename: str,
        semaphore: asyncio.Semaphore,
        model: str = "gpt-3.5-turbo",
        detect: bool = True,
        style: int = 0,
        headers: dict = None,
    ) -> str:
        async with semaphore:
            target, urls = "", get_urls(domain=domain, potentials=potentials, wander=wander)
            stream, version, params, real_model = False, 0, None, ""

            for url in urls:
                params = parse_url(url=url)
                if not params or not params.url:
                    continue

                real_model = params.model or model or "gpt-3.5-turbo"

                # available check
                result = await chat_async(
                    url=params.url,
                    question=question,
                    keyword=keyword,
                    model=real_model,
                    stream=params.stream == 1,
                    token=params.token,
                    timeout=30,
                    interval=3,
                    style=style,
                    headers=headers,
                    strict=strict,
                )
                if result.available:
                    target, stream, version = params.url, result.stream, result.version
                    break
                elif result.notfound:
                    logger.warning(f"[Interactive] url {params.url} available but no {real_model} model exists")
                    break
                elif result.terminate:
                    # if terminal is true, all attempts should be aborted immediately
                    break

            if not target:
                return ""

            if detect:
                if params.stream == -1:
                    # stream support check
                    result = await chat_async(
                        url=target,
                        question=question,
                        keyword=keyword,
                        model=real_model,
                        stream=True,
                        token=params.token,
                        timeout=30,
                        interval=3,
                        style=style,
                        headers=headers,
                        strict=strict,
                    )
                    stream = result.stream or stream
                else:
                    stream = stream or params.stream == 1

                if params.version == -1 and need_check_version(model=real_model):
                    # version check
                    result = await chat_async(
                        url=url,
                        question=question,
                        keyword=keyword,
                        model=get_model_name(provider=real_model, version=4),
                        stream=False,
                        token=params.token,
                        timeout=30,
                        interval=3,
                        style=style,
                        headers=headers,
                        strict=strict,
                    )
                    version = result.version if result.available else version
                else:
                    version = params.version if params.version >= 0 else version

            target = concat_url(
                url=target,
                stream=stream,
                version=version,
                token=params.token,
                model=params.model,
                mapping=params.mapping,
            )
            if target and filename:
                directory = os.path.dirname(filename)
                if not os.path.exists(directory) or not os.path.isdir(directory):
                    os.makedirs(directory, exist_ok=True)

                async with aiofiles.open(filename, mode="a+", encoding="utf8") as f:
                    await f.write(target + "\n")
                    await f.flush()

            return target

    if not sites or not isinstance(sites, (list, tuple)):
        logger.error(f"[Interactive] skip check due to domains is empty")
        return []

    filename = utils.trim(filename)
    model = utils.trim(model) or "gpt-3.5-turbo"

    # parse customize request headers
    customize_headers = parse_headers(content=headers, delimiter="|")

    concurrency = 256 if concurrency <= 0 else concurrency
    semaphore = asyncio.Semaphore(concurrency)

    logger.info(f"[Interactive] start asynchronous execution of the detection tasks, total: {len(sites)}")
    starttime = time.time()

    if show_progress:
        # show progress bar
        tasks, targets = [], []
        for site in sites:
            tasks.append(
                asyncio.create_task(
                    checkone(
                        domain=site,
                        filename=filename,
                        semaphore=semaphore,
                        model=model,
                        detect=True,
                        style=style,
                        headers=customize_headers,
                    )
                )
            )

        pbar = tqdm(total=len(tasks), desc="Processing", unit="task")
        for task in asyncio.as_completed(tasks):
            target = await task
            if target:
                targets.append(target)

            pbar.update(1)

        pbar.close()
    else:
        # no progress bar
        tasks = [
            checkone(
                domain=domain,
                filename=filename,
                semaphore=semaphore,
                model=model,
                detect=True,
                style=style,
                headers=customize_headers,
            )
            for domain in sites
            if domain
        ]
        targets = [x for x in await asyncio.gather(*tasks)]

    logger.info(f"[Interactive] async check completed, cost: {time.time()-starttime:.2f}s")
    return targets


def batch_probe(
    candidates: list[str],
    question: str = "",
    keyword: str = "",
    model: str = "gpt-3.5-turbo",
    filename: str = "",
    potentials: str = "",
    wander: bool = False,
    run_async: bool = True,
    show_progress: bool = False,
    num_threads: int = 0,
    chunk: int = 512,
    style: int = 0,
    headers: str = "",
    strict: bool = True,
) -> list[str]:
    if not candidates or not isinstance(candidates, list):
        return []

    question = utils.trim(question)
    keyword = utils.trim(keyword)
    if (question and not keyword) or (not question and keyword):
        logger.error("[Interactive] question and keyword must be used together")
        return []

    # run as async
    if run_async:
        sites = asyncio.run(
            check_async(
                sites=candidates,
                filename=filename,
                potentials=potentials,
                wander=wander,
                question=question,
                keyword=keyword,
                model=model,
                concurrency=num_threads,
                show_progress=show_progress,
                style=style,
                headers=headers,
                strict=strict,
            )
        )
    # concurrent check
    elif len(candidates) <= chunk:
        sites = check_concurrent(
            sites=candidates,
            filename=filename,
            potentials=potentials,
            wander=wander,
            question=question,
            keyword=keyword,
            model=model,
            num_threads=num_threads,
            show_progress=show_progress,
            style=style,
            headers=headers,
            strict=strict,
        )
    # sharding
    else:
        slices = [candidates[i : i + chunk] for i in range(0, len(candidates), chunk)]
        tasks = [
            [
                x,
                filename,
                potentials,
                wander,
                question,
                keyword,
                model,
                num_threads,
                show_progress,
                style,
                headers,
                strict,
            ]
            for x in slices
        ]
        results = utils.multi_process_run(func=check_concurrent, tasks=tasks)
        sites = [x for r in results if r for x in r]

    return [x for x in sites if x]


def no_model(content: str, model: str = "") -> bool:
    content = utils.trim(content)
    if not content:
        return False

    pattern = r"model_not_found|对于模型.*?无可用渠道"
    model = utils.trim(model).lower()
    if not model.startswith("claude-"):
        pattern = rf"{pattern}|Anthropic|Claude"

    return re.search(pattern, content, flags=re.I) is not None


def not_ternimate(content: str) -> bool:
    content = utils.trim(content)
    if not content:
        return False

    return re.search(r'"errorType":(\s+)?401', content, flags=re.I) is not None


def concat_url(url: str, stream: bool, version: int, token: str = "", model: str = "", mapping: dict = None) -> str:
    url = utils.trim(url)
    if not url:
        return ""

    version = 3 if version <= 0 else version
    target = f"{url}?stream={str(stream).lower()}&version={version}"

    token = utils.trim(token)
    if token:
        target = f"{target}&token={token}"

    model = utils.trim(model)
    if model:
        target = f"{target}&model={model}"

    records = set()
    if isinstance(mapping, dict):
        for key, value in mapping.items():
            k, v = utils.trim(key), utils.trim(value)
            if k and v:
                records.add(f"{k}:{v}")

    if records:
        text = ",".join(list(records))
        target += f"&mapping={text}"
    return target


def verify(
    content: str,
    content_type: str,
    stream: bool,
    model: str,
    keyword: str = DEFAULT_KEYWORD,
    strict: bool = False,
) -> CheckResult:
    def extract_message(content: str) -> tuple[bool, str, str]:
        if not content:
            return False, "", ""

        try:
            data = json.loads(content)
            name = data.get("model", "") or ""

            # extract message
            choices = data.get("choices", [])
            if not choices or not isinstance(choices, list):
                return False, "", name

            delta, support = None, False
            if "delta" in choices[0]:
                delta = choices[0].get("delta", {})
                support = True
            elif "message" in choices[0]:
                delta = choices[0].get("message", {})
                support = False

            if delta is None or not isinstance(delta, dict):
                return False, "", name

            message = delta.get("content", "") or ""
            return support, message, name
        except:
            return False, "", ""

    def support_stream(content: str, strict: bool) -> tuple[bool, str, str]:
        if not content:
            return False, "", ""

        support, words, model = True, [], ""
        flag, lines = "data:", content.split("\n")

        for i in range(len(lines)):
            line = utils.trim(lines[i])
            if not line:
                continue

            texts = line.split(sep=flag, maxsplit=1)
            if strict and len(texts) != 2:
                support = False

            text = utils.trim(texts[1] if len(texts) == 2 else texts[0])

            # ignore the last line because usually it is "data: [DONE]"
            if text == "[DONE]":
                continue

            success, message, name = extract_message(content=text)
            if not success:
                support = len(texts) == 2
                if len(texts) == 2:
                    message = texts[1]

            model = model or name
            words.append(message)

        return support, "".join(words), model

    content = utils.trim(content)
    if not content:
        return CheckResult(available=False)

    content_type = utils.trim(content_type)
    req_model = utils.trim(model)
    text, support, resp_model = content, False, ""

    if "text/event-stream" in content_type:
        support, text, resp_model = support_stream(content=content, strict=strict)
    elif "application/json" in content_type or "text/plain" in content_type:
        content = re.sub(r"^```json\n|\n```$", "", text, flags=re.MULTILINE)
        support, message, resp_model = extract_message(content=content)
        text = message or content

    available = re.search(keyword, text, flags=re.I) is not None and re.search(r'"error":', text, flags=re.I) is None
    if available and strict:
        # set available to false if request model is not equal to response model in strict mode
        available = req_model == resp_model

    model = "" if not available else (req_model if strict else (resp_model or req_model))
    version = get_model_version(model=model, soft=True, opposite=False)

    # adapt old nextweb, content_type is empty but support stream
    if stream and available:
        support = True
    elif not available:
        support = False

    not_exists = False if available else (req_model != resp_model if strict else no_model(content=content, model=model))
    terminal = available or not_exists
    return CheckResult(available=available, stream=support, version=version, terminate=terminal, notfound=not_exists)
