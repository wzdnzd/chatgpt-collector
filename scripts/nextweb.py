# -*- coding: utf-8 -*-

# @Author  : wzdnzd
# @Time    : 2023-06-30

import asyncio
import itertools
import json
import math
import os
import re
import time
import warnings
from datetime import datetime, timedelta, timezone
from urllib import parse as parse

import aiofiles
import aiohttp
from tqdm import tqdm

import push
import utils
from logger import logger

# date format
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# last modified key name
LAST_MODIFIED = "lastModified"

# github rest api prefix
GITHUB_API = "https://api.github.com"

# github username
OWNER = "ChatGPTNextWeb"

# github repository name
REPO = "ChatGPT-Next-Web"

# deployments filepath
DEPLOYMENTS_FILE = os.path.join(utils.PATH, "data", "deployments.txt")

# domains filepath
MATERIAL_FILE = os.path.join(utils.PATH, "data", "material.txt")

# candidates filepath
CANDIDATES_FILE = os.path.join(utils.PATH, "data", "candidates.txt")

# result filepath
SITES_FILE = os.path.join(utils.PATH, "data", "sites.txt")

# request headers
DEFAULT_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": utils.USER_AGENT,
}

# github token
GITHUB_TOKEN = utils.trim(os.environ.get("GITHUB_TOKEN", ""))
if GITHUB_TOKEN:
    DEFAULT_HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"

# running on local
LOCAL_MODE = utils.trim(os.environ.get("LOCAL_MODE", "")).lower() in ["true", "1"]


def last_history(url: str, refresh: bool) -> datetime:
    last = datetime(year=1970, month=1, day=1, tzinfo=timezone.utc)
    if not refresh and url:
        content = utils.http_get(url=url) if url else ""
        if content:
            modified = ""
            try:
                modified = json.loads(content).get(LAST_MODIFIED, "")
                if modified:
                    date = datetime.strptime(modified, DATE_FORMAT)
                    last = (date + timedelta(minutes=-10)).replace(tzinfo=timezone.utc)
            except Exception:
                logger.error(f"[NextWeb] invalid date format: {modified}")

    return last


def query_forks_count(retry: int = 3) -> int:
    url = f"{GITHUB_API}/search/repositories?q=user:{OWNER}+repo:{REPO}+{REPO}"
    content = utils.http_get(url=url, headers=DEFAULT_HEADERS, retry=retry)
    if not content:
        logger.error(f"[NextWeb] failed to query forks count")
        return -1

    try:
        items = json.loads(content).get("items", [])
        if not items or type(items) != list:
            logger.error(f"[NextWeb] cannot get forks count, message: {content}")
            return -1

        return items[0].get("forks_count", 0)
    except:
        logger.error(f"[NextWeb] occur error when parse forks count, message: {content}")
        return -1


def list_deployments(history: datetime, sort: str = "newest") -> list[str]:
    last = history or datetime(year=1970, month=1, day=1, tzinfo=timezone.utc)
    count, peer = query_forks_count(retry=3), 100
    total = int(math.ceil(count / peer))

    # see: https://docs.github.com/en/rest/repos/forks?apiVersion=2022-11-28
    if sort not in ["newest", "oldest", "stargazers", "watchers"]:
        sort = "newest"

    # concurrent
    if last.year == 1970:
        tasks = [[x, last, peer, sort] for x in range(1, total + 1)]
        results = utils.multi_thread_collect(func=query_deployments_page, tasks=tasks)
        return list(itertools.chain.from_iterable([x[0] for x in results]))

    # serial
    deployments, over, page = [], False, 1
    while not over and page <= total:
        array, over = query_deployments_page(page=page, last=last, peer=peer, sort=sort)
        deployments.extend(array)
        page += 1

    return deployments


def parse_site(
    url: str,
    page: int = 1,
    peer: int = 5,
    session: str = "",
    rest: bool = True,
    last: datetime = None,
    clean: bool = True,
) -> list[str]:
    if not rest and session:
        return extract_target(url=url, session=session)

    if not url or page <= 0:
        return []

    peer, domains = min(max(1, peer), 100), set()
    if not last:
        last = datetime(year=1970, month=1, day=1, tzinfo=timezone.utc)

    url = f"{url}?environment=production&per_page={peer}&page={page}"
    content = utils.http_get(url=url, headers=DEFAULT_HEADERS, interval=2.0)

    if content:
        try:
            deployments = json.loads(content)
            for item in deployments:
                if not item or not isinstance(item, dict):
                    continue

                # get deployment updat time
                text = utils.trim(item.get("updated_at", ""))

                # updated = datetime.fromisoformat(text.replace("Z", "+00:00")) if text else None
                updated = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc) if text else None

                # it has been processed last time
                if updated and updated <= last:
                    continue

                site = query_deployment_status(url=item.get("statuses_url", ""), clean=clean)
                if site:
                    domains.add(site)
        except:
            logger.error(f"[NextWeb] failed to parse target url due to cannot query deployments, message: {content}")

    targets = [] if not domains else list(domains)

    if LOCAL_MODE and targets:
        utils.write_file(filename=MATERIAL_FILE, lines=targets, overwrite=False)

    return targets


def extract_target(url: str, session: str) -> list[str]:
    if not url or not session:
        return []

    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "User-Agent": utils.USER_AGENT,
        "Cookie": f"user_session={session}",
    }

    url = url.replace(f"{GITHUB_API}/repos", "https://github.com")
    content = utils.http_get(url=url, headers=headers, interval=1.0)
    regex = '<a href="(https://.*?)" data-testid="deployments-list-environment-url" tabindex="-1">'
    groups = re.findall(regex, content, flags=re.I)
    return [] if not groups else groups


def query_deployment_status(url: str, clean: bool = True) -> str:
    if not url:
        return ""

    content = utils.http_get(url=url, headers=DEFAULT_HEADERS, interval=1.0)
    try:
        statuses = json.loads(content)
        if not statuses or type(statuses) != list:
            return ""

        state = statuses[0].get("state", "")
        target = statuses[0].get("target_url", "") or ""
        if clean and (target.startswith("https://dash.zeabur.com/") or target.startswith("https://github.com/")):
            return ""

        success = state == "success"
        if not success:
            description = utils.trim(statuses[0].get("description", ""))
            success = re.search("checks for deployment have failed", description, flags=re.I) is not None

        return target if success else ""
    except:
        return ""


def query_deployments_page(page: int, last: datetime, peer: int = 100, sort: str = "newest") -> tuple[list[str], bool]:
    if page <= 0 or not last:
        return [], False

    peer = min(max(peer, 1), 100)
    url = f"{GITHUB_API}/repos/{OWNER}/{REPO}/forks?sort={sort}&per_page={peer}&page={page}"
    deployments, over, starttime = list(), False, time.time()

    content, retry = "", 5
    while not content and retry > 0:
        content = utils.http_get(url=url, headers=DEFAULT_HEADERS, interval=1.0)
        retry -= 1
        if not content:
            time.sleep(2)

    try:
        forks = json.loads(content)
        for fork in forks:
            if not fork or type(fork) != dict:
                continue

            datetext = fork.get("created_at", "")
            updated = datetime.strptime(datetext, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

            # last time has already been collected
            if updated <= last:
                over = True
                break

            deployment = fork.get("deployments_url", "")
            if deployment:
                deployments.append(deployment)
    except:
        logger.error(f"[NextWeb] cannot fetch deployments for page: {page}, message: {content}")

    cost = "{:.2f}s".format(time.time() - starttime)
    logger.info(f"[NextWeb] finished query deployments for page: {page}, cost: {cost}")

    return deployments, over


def auth(url: str) -> str:
    if not url:
        return ""

    response = utils.http_post_noerror(url=f"{url}/api/config", allow_redirects=False, retry=2)
    if not response or response.getcode() != 200:
        return ""

    try:
        content = response.read().decode("UTF8")
        data = json.loads(content)
        target = "" if data.get("needCode", False) else url

        if LOCAL_MODE and target:
            filename = os.path.join(utils.PATH, "data", CANDIDATES_FILE)
            utils.write_file(filename=filename, lines=target, overwrite=False)

        return target
    except:
        return ""


def check(domain: str, model: str = "gpt-3.5-turbo", filename: str = "") -> str:
    target, urls = "", get_urls(domain=domain)
    filename = utils.trim(filename) or SITES_FILE

    for url in urls:
        success, terminate = chat(url=url, model=model, timeout=30)
        if success:
            target = f"{url}?mode=openai&stream=true"
            break
        elif terminate:
            # if terminal is true, all attempts should be aborted immediately
            break

    if LOCAL_MODE and target:
        utils.write_file(filename=filename, lines=target, overwrite=False)

    return target


def check_concurrent(
    sites: list[str],
    model: str = "gpt-3.5-turbo",
    num_threads: int = -1,
    show_progress: bool = False,
    index: int = 0,
    filename: str = "",
) -> list[str]:
    if not sites or not isinstance(sites, list):
        logger.warning(f"[NextWeb] skip process due to lines is empty")
        return []

    model = utils.trim(model) or "gpt-3.5-turbo"
    index = max(0, index)
    tasks = [[x, model, filename] for x in sites if x]

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

    payload = get_payload(model=model)
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
            logger.warning(f"[NextWeb] API can be used but not found model: {model}, url: {url}")
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
    payload = get_payload(model=model, stream=True)
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
                    return await chat_async(url, model, token, retry - 1, min(timeout + 10, 90))
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
        return await chat_async(url, model, token, retry - 1, timeout)


async def check_async(
    sites: list[str],
    model: str = "gpt-3.5-turbo",
    concurrency: int = 512,
    filename: str = "",
    show_progress: bool = True,
) -> list[str]:
    async def checkone(
        domain: str,
        semaphore: asyncio.Semaphore,
        model: str = "gpt-3.5-turbo",
        filename: str = "",
    ) -> str:
        async with semaphore:
            target, urls = "", get_urls(domain=domain)
            filename = utils.trim(filename) or SITES_FILE

            for url in urls:
                success, terminal = await chat_async(url=url, model=model, timeout=30, interval=3)
                if success:
                    target = f"{url}?mode=openai&stream=true"
                    break
                elif terminal:
                    break

            if LOCAL_MODE and target:
                directory = os.path.dirname(filename)
                if not os.path.exists(directory) or not os.path.isdir(directory):
                    os.makedirs(directory, exist_ok=True)

                async with aiofiles.open(filename, mode="a+", encoding="utf8") as f:
                    await f.write(target + "\n")
                    await f.flush()

            return target

    if not sites or not isinstance(sites, (list, tuple)):
        logger.error(f"[NextWeb] skip check due to domains is empty")
        return []

    model = utils.trim(model) or "gpt-3.5-turbo"
    filename = utils.trim(filename) or SITES_FILE

    concurrency = 256 if concurrency <= 0 else concurrency
    semaphore = asyncio.Semaphore(concurrency)

    logger.info(f"[NextWeb] start asynchronous execution of the detection tasks, total: {len(sites)}")
    starttime = time.time()

    if show_progress:
        # show progress bar
        tasks, targets = [], []
        for site in sites:
            tasks.append(asyncio.create_task(checkone(site, semaphore, model, filename)))

        pbar = tqdm(total=len(tasks), desc="Processing", unit="task")
        for task in asyncio.as_completed(tasks):
            target = await task
            if target:
                targets.append(target)

            pbar.update(1)

        pbar.close()
    else:
        # no progress bar
        tasks = [checkone(domain, semaphore, model, filename) for domain in sites if domain]
        targets = [x for x in await asyncio.gather(*tasks)]

    logger.info(f"[NextWeb] async check completed, cost: {time.time()-starttime:.2f}s")
    return targets


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


def get_urls(domain: str) -> list[str]:
    domain = utils.trim(domain)
    if not domain:
        return []

    try:
        result = parse.urlparse(domain)
        urls = list()

        if not result.path or result.path == "/":
            for subpath in ["/api/openai/v1/chat/completions", "/api/chat-stream"]:
                url = parse.urljoin(domain, subpath)
                urls.append(url)
        else:
            urls.append(f"{result.scheme}://{result.netloc}{result.path}")

        return urls
    except:
        logger.error(f"[NextWeb] skip due to invalid url: {domain}")
        return []


def check_billing(domain: str) -> bool:
    warnings.warn("This API has been disabled upstream", DeprecationWarning)

    if not domain:
        return False

    url = f"{domain}/api/openai/dashboard/billing/subscription"
    headers = {
        "Accept": "*/*",
        "Content-Type": "application/json",
        "Referer": url,
        "Origin": url,
        "User-Agent": utils.USER_AGENT,
    }

    content = utils.http_get(url=url, headers=headers, interval=0.5, retry=2)
    try:
        limit = json.loads(content).get("hard_limit_usd", "")
        return utils.is_number(limit)
    except:
        return False


def read(filepath: str) -> list[str]:
    filepath, collections = utils.trim(filepath), []

    # if file not exist, read from it's backup
    for file in [filepath, f"{filepath}.bak"]:
        if not file or not os.path.exists(file) or not os.path.isfile(file):
            continue

        with open(file, "r", encoding="utf8") as f:
            for line in f.readlines():
                collections.append(line.replace("\n", "").strip().lower())
        break

    return collections


def load(url: str, overlay: bool) -> list[str]:
    url, sites = utils.trim(url), []

    # load local file if exist
    sites.extend(read(SITES_FILE))

    # add local existing material if necessary
    if overlay:
        sites.extend(read(MATERIAL_FILE))

    # load remote sites
    if url:
        content = utils.http_get(url=url)
        sites.extend(content.split(","))

    return [utils.extract_domain(url=x, include_protocal=True) for x in sites]


def backup_file(filepath: str) -> None:
    if not filepath or not os.path.exists(filepath) or not os.path.isfile(filepath):
        return

    newfile = f"{filepath}.bak"
    if os.path.exists(newfile):
        os.remove(newfile)

    os.rename(filepath, newfile)


def collect(params: dict) -> list:
    if not params or type(params) != dict:
        return []

    persist = params.get("persist", {})
    storage = {} if not persist or type(persist) != dict else persist

    server = os.environ.get("COLLECT_CONF", "").strip()
    pushtool = push.get_instance(domain=server)
    if not LOCAL_MODE and not pushtool.validate(storage.get("modified", {})):
        logger.error(f"[NextWeb] invalid persist config, must config modified store if running on remote")
        return []

    # store config
    modified, database = storage.get("modified", {}), storage.get("sites", {})

    # github user session
    session = utils.trim(os.environ.get("USER_SESSION", ""))

    # sort type for forks
    sort = utils.trim(params.get("sort", "newest")).lower()

    # check only
    checkonly = params.get("checkonly", False)

    # re-generate deployments
    refresh = False if checkonly else params.get("refresh", False)

    # add local existing material if necessary
    overlay = params.get("overlay", False)

    # number of concurrent threads
    num_threads = params.get("num_threads", 0)

    # run type
    run_async = params.get("async", True)

    mode, starttime = "LOCAL" if LOCAL_MODE else "REMOTE", time.time()
    logger.info(
        f"[NextWeb] start to collect sites from {OWNER}/{REPO}, mode: {mode}, checkonly: {checkonly}, refresh: {refresh}"
    )

    # load exists
    candidates = [] if refresh else load(pushtool.raw_url(database), overlay)

    if not checkonly:
        begin = datetime.now(timezone.utc).strftime(DATE_FORMAT)

        # fetch last run time
        last = last_history(pushtool.raw_url(push_conf=modified), refresh)

        # TODO: if it is possible to bypass the rate limiting measures of GitHub, asynchronous requests can be used
        deployments = list_deployments(last, sort)

        # add local existing deployments if necessary
        if overlay:
            deployments.extend(read(filepath=DEPLOYMENTS_FILE))

        # deduplication
        deployments = list(set(deployments))
        logger.info(f"[NextWeb] collect completed, found {len(deployments)} deployments")

        if deployments:
            # save deployments to file
            if LOCAL_MODE:
                # backup exist files
                for filename in [DEPLOYMENTS_FILE, MATERIAL_FILE, CANDIDATES_FILE]:
                    backup_file(filepath=filename)

                utils.write_file(DEPLOYMENTS_FILE, deployments, True)

            clean = params.get("clean", True)
            args = [[x, 1, 100, session, True, last, clean] for x in deployments]
            logger.info(f"[NextWeb] extract target domain begin, count: {len(args)}")

            materials = utils.multi_thread_collect(
                func=parse_site,
                tasks=args,
                show_progress=True,
                num_threads=num_threads,
            )
            newsites = list(itertools.chain.from_iterable(materials))
            candidates.extend([x for x in newsites if x])

        # save last modified time
        if pushtool.validate(modified):
            content = json.dumps({LAST_MODIFIED: begin})
            pushtool.push_to(content=content, push_conf=modified, group="modified")

    logger.info(f"[NextWeb] candidate target collection completed, found {len(candidates)} sites")

    if params.get("skip_check", False) or not candidates:
        logger.warning("[NextWeb] availability testing steps will be skipped")
        return list(set(candidates))

    # backup sites.txt if file exists
    backup_file(filepath=SITES_FILE)

    candidates = list(set(candidates))
    chunk = max(params.get("chunk", 256), 1)
    model = params.get("model", "") or "gpt-3.5-turbo"

    logger.info(f"[NextWeb] start to check available, sites: {len(candidates)}, model: {model}")

    # run as async
    if run_async:
        sites = asyncio.run(check_async(sites=candidates, model=model, concurrency=num_threads, show_progress=True))
    # concurrent check
    elif len(candidates) <= chunk:
        results = check_concurrent(sites=candidates, model=model, num_threads=num_threads, show_progress=True)
        sites = [x for x in results if x]
    else:
        # sharding
        slices = [candidates[i : i + chunk] for i in range(0, len(candidates), chunk)]
        tasks = [[x, model, num_threads, False, 0] for x in slices]
        results = utils.multi_process_collect(func=check_concurrent, tasks=tasks)
        sites = [x for r in results if r for x in r if x]

    # save sites
    if sites and pushtool.validate(database):
        success = pushtool.push_to(content=",".join(sites), push_conf=database, group="sites")
        if not success:
            logger.error(f"[NextWeb] push {len(sites)} sites to remote failed")

    cost = "{:.2f}s".format(time.time() - starttime)
    logger.info(f"[NextWeb] finished check {len(results)} candidates, got {len(sites)} avaiable sites, cost: {cost}")
    return sites
