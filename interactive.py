# -*- coding: utf-8 -*-

# @Author  : wzdnzd
# @Time    : 2024-04-07

import asyncio
import os
import re
import time
from urllib import parse

import aiofiles
import aiohttp
from tqdm import tqdm

import utils
from logger import logger


def get_headers(domain: str) -> dict:
    return {
        "Content-Type": "application/json",
        "Path": "v1/chat/completions",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json, text/event-stream",
        "Referer": domain,
        "Origin": domain,
        "User-Agent": utils.USER_AGENT,
    }


def get_payload(model: str, stream: bool = True) -> dict:
    return {
        "model": model or "gpt-3.5-turbo",
        "temperature": 0,
        "max_tokens": 1000,
        "top_p": 1,
        "frequency_penalty": 1,
        "presence_penalty": 1,
        "stream": stream,
        "messages": [
            {
                "role": "user",
                "content": "Tell me what ChatGPT is in English, your answer should contain a maximum of 20 words and must start with 'ChatGPT'!",
            }
        ],
    }


def get_urls(domain: str, standard: bool = False) -> list[str]:
    domain = utils.trim(domain)
    if not domain:
        return []

    try:
        result = parse.urlparse(domain)
        urls = list()

        if not result.path or result.path == "/":
            subpaths = (
                ["/v1/chat/completions"]
                if standard
                else [
                    "/v1/chat/completions",
                    "/api/chat-stream",
                    "/api/openai/v1/chat/completions",
                ]
            )

            for subpath in subpaths:
                url = parse.urljoin(domain, subpath)
                urls.append(url)
        else:
            urls.append(f"{result.scheme}://{result.netloc}{result.path}")

        return urls
    except:
        logger.error(f"[Interactive] skip due to invalid url: {domain}")
        return []


def check(domain: str, filename: str = "", standard: bool = False, model: str = "gpt-3.5-turbo") -> str:
    target, urls = "", get_urls(domain=domain, standard=standard)

    for url in urls:
        success, terminate = chat(url=url, model=model, timeout=30)
        if success:
            target = f"{url}?mode=openai&stream=true"
            break
        elif terminate:
            # if terminal is true, all attempts should be aborted immediately
            break

    filename = utils.trim(filename)
    if target and filename:
        utils.write_file(filename=filename, lines=target, overwrite=False)

    return target


def check_concurrent(
    sites: list[str],
    filename: str = "",
    standard: bool = False,
    model: str = "gpt-3.5-turbo",
    num_threads: int = -1,
    show_progress: bool = False,
    index: int = 0,
) -> list[str]:
    if not sites or not isinstance(sites, list):
        logger.warning(f"[Interactive] skip process due to lines is empty")
        return []

    filename = utils.trim(filename)
    model = utils.trim(model) or "gpt-3.5-turbo"
    index = max(0, index)
    tasks = [[x, filename, standard, model] for x in sites if x]

    result = utils.multi_thread_collect(
        func=check,
        tasks=tasks,
        num_threads=num_threads,
        show_progress=show_progress,
        description=f"Chunk-{index}",
    )

    return [x for x in result if x]


def chat(
    url: str,
    model: str = "gpt-3.5-turbo",
    token: str = "",
    retry: int = 3,
    timeout: int = 6,
) -> tuple[bool, bool]:
    """the first return value indicates whether the URL is available, and the second return value indicates whether subsequent attempts should be terminated"""

    if not url:
        return False, True

    headers = get_headers(domain=url)
    token = utils.trim(token)
    if token:
        headers["Authorization"] = f"Bearer {token}"

    model = utils.trim(model) or "gpt-3.5-turbo"
    retry = 3 if retry < 0 else retry
    timeout = 15 if timeout <= 0 else timeout

    payload = get_payload(model=model, stream=False)
    try:
        response, exitcode = utils.http_post(
            url=url,
            headers=headers,
            params=payload,
            retry=retry,
            timeout=timeout,
            allow_redirects=False,
        )
        if not response or response.getcode() != 200:
            return False, exitcode == 2

        content = response.read().decode("UTF8")
        if not content:
            return False, False
        elif re.search("ChatGPT", content, flags=re.I) is not None:
            return True, True
        elif re.search("model_not_found", content, flags=re.I) is not None:
            logger.warning(f"[Interactive] API can be used but not found model: {model}, url: {url}")
            return False, True
        else:
            return False, False
    except:
        return False, False


async def chat_async(
    url: str,
    model: str = "gpt-3.5-turbo",
    token: str = "",
    retry: int = 3,
    timeout: int = 6,
    interval: float = 3,
) -> tuple[bool, bool]:
    if not url or retry <= 0:
        return False, False

    headers = get_headers(domain=url)
    token = token.strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    model = model.strip() or "gpt-3.5-turbo"
    payload = get_payload(model=model, stream=False)
    timeout = 6 if timeout <= 0 else timeout
    interval = max(0, interval)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=timeout) as response:
                if response.status != 200:
                    return False, response.status in [400, 401]

                content = ""
                try:
                    content = await response.text()
                except asyncio.TimeoutError:
                    await asyncio.sleep(interval)
                    return await chat_async(url, model, token, retry - 1, min(timeout + 10, 90), interval)
                if not content:
                    return False, False
                elif re.search("ChatGPT", content, flags=re.I) is not None:
                    return True, True
                elif re.search("model_not_found", content, flags=re.I) is not None:
                    logger.warning(f"[Check] API can be used but not found model: {model}, url: {url}")
                    return False, True

                return False, False
    except:
        await asyncio.sleep(interval)
        return await chat_async(url, model, token, retry - 1, timeout, interval)


async def check_async(
    sites: list[str],
    filename: str = "",
    standard: bool = False,
    model: str = "gpt-3.5-turbo",
    concurrency: int = 512,
    show_progress: bool = True,
) -> list[str]:
    async def checkone(
        domain: str,
        filename: str,
        semaphore: asyncio.Semaphore,
        model: str = "gpt-3.5-turbo",
    ) -> str:
        async with semaphore:
            target, urls = "", get_urls(domain=domain, standard=standard)

            for url in urls:
                success, terminal = await chat_async(url=url, model=model, timeout=30, interval=3)
                if success:
                    target = f"{url}?mode=openai&stream=true"
                    break
                elif terminal:
                    break

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

    concurrency = 256 if concurrency <= 0 else concurrency
    semaphore = asyncio.Semaphore(concurrency)

    logger.info(f"[Interactive] start asynchronous execution of the detection tasks, total: {len(sites)}")
    starttime = time.time()

    if show_progress:
        # show progress bar
        tasks, targets = [], []
        for site in sites:
            tasks.append(asyncio.create_task(checkone(site, filename, semaphore, model)))

        pbar = tqdm(total=len(tasks), desc="Processing", unit="task")
        for task in asyncio.as_completed(tasks):
            target = await task
            if target:
                targets.append(target)

            pbar.update(1)

        pbar.close()
    else:
        # no progress bar
        tasks = [checkone(domain, filename, semaphore, model) for domain in sites if domain]
        targets = [x for x in await asyncio.gather(*tasks)]

    logger.info(f"[Interactive] async check completed, cost: {time.time()-starttime:.2f}s")
    return targets
