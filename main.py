import argparse
import importlib
import json
import multiprocessing
import os
import re
import time
import traceback
from enum import Enum
from http.client import HTTPResponse
from multiprocessing.managers import DictProxy, ListProxy
from multiprocessing.synchronize import Semaphore
from urllib import parse

import push
import utils
from logger import logger
from urlvalidator import isurl


class GPTProvider(Enum):
    AZURE = 1
    CHATGPT = 2
    OPENAI = 3
    UNKNOWN = 4


COMMON_PATH_MODE = {
    "/api/chat-process": GPTProvider.OPENAI,
    "/api/chat-stream": GPTProvider.AZURE,
    "/api/chat": GPTProvider.OPENAI,
    "/api": GPTProvider.OPENAI,
}

COMMON_PAYLOAD = {
    GPTProvider.AZURE: {"prompt": "What is ChatGPT?", "options": {}},
    GPTProvider.CHATGPT: {},
    GPTProvider.OPENAI: {
        "messages": [{"role": "user", "content": "What is ChatGPT?"}],
        "stream": True,
        "model": "gpt-3.5-turbo",
        "temperature": 1,
        "presence_penalty": 0,
    },
}

HEADERS = {"Content-Type": "application/json", "path": "v1/chat/completions"}


def query_provider(name: str) -> GPTProvider:
    if not name:
        return GPTProvider.UNKNOWN
    for item in GPTProvider:
        if name.upper() == item.name:
            return item

    return GPTProvider.UNKNOWN


def execute_script(script: str, params: dict = {}) -> list[str]:
    try:
        # format: a.b.c#function or a-b.c#_function or a#function and so on
        regex = r"^([a-zA-Z0-9_]+|([0-9a-zA-Z_]+([a-zA-Z0-9_\-]+)?\.)+)[a-zA-Z0-9_\-]+#[a-zA-Z_]+[0-9a-zA-Z_]+$"
        if not re.match(regex, script):
            logger.info(
                f"[ScriptError] script execute error because script: {script} is invalidate"
            )
            return []

        path, func_name = script.split("#", maxsplit=1)
        path = f"scripts.{path}"
        module = importlib.import_module(path)
        if not hasattr(module, func_name):
            logger.error(f"script: {path} not exists function {func_name}")
            return []

        func = getattr(module, func_name)

        starttime = time.time()
        logger.info(f"[ScriptInfo] start execute script: scripts.{script}")

        sites = func(params)
        if type(sites) != list:
            logger.error(
                f"[ScriptError] return value error, need a list, but got a {type(sites)}"
            )
            return []

        endtime = time.time()
        logger.info(
            "[ScriptInfo] finished execute script: scripts.{}, cost: {:.3}s".format(
                script, endtime - starttime
            )
        )

        return sites
    except:
        logger.error(
            f"[ScriptError] occur error run script: {script}, message: \n{traceback.format_exc()}"
        )
        return []


def call(
    script: str, params: dict, availables: ListProxy, semaphore: Semaphore
) -> None:
    try:
        if not script:
            return

        subscribes = execute_script(script=script, params=params)
        if subscribes and type(subscribes) == list:
            availables.extend(subscribes)
    finally:
        if semaphore is not None and isinstance(semaphore, Semaphore):
            semaphore.release()


def batch_call(tasks: dict) -> list[str]:
    if not tasks:
        return []

    try:
        thread_num = max(min(len(tasks), 50), 1)
        with multiprocessing.Manager() as manager:
            availables, processes = manager.list(), []
            semaphore = multiprocessing.Semaphore(thread_num)
            for k, v in tasks.items():
                semaphore.acquire()
                p = multiprocessing.Process(
                    target=call, args=(k, v, availables, semaphore)
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

            return list(availables)
    except:
        traceback.print_exc()
        return []


def process(url: str) -> None:
    if not url or not isurl(url):
        logger.error("[ConfigError] cannot process because config is invalid")
        return
    try:
        headers = {"User-Agent": utils.USER_AGENT, "Referer": url}
        content = utils.http_get(url=url, headers=headers)
        if not content:
            logger.error(f"[ConfigError] cannot fetch configuration from url: {url}")
            return
        else:
            configuration = json.loads(content)
            pushtool = push.get_instance(domain=url)
            tasks = regularized(data=configuration, pushtool=pushtool)

            if not tasks:
                logger.warning("[ConfigWarn] no valid crawler task found")
                return

            urls = batch_call(tasks.get("scripts", {}))
            # exist urls
            candidates = fetct_exist(tasks.get("persists"), pushtool)

            data = batch_probe(
                urls=urls, candidates=candidates, threshold=tasks.get("threshold")
            )
            if not data:
                logger.warning("[ProcessWarn] cannot found any domains")
                return

            params = [
                [v, tasks.get("persists").get(k), k, 5] for k, v in data.items() if v
            ]
            cpu_count = multiprocessing.cpu_count()
            num = len(params) if len(params) <= cpu_count else cpu_count

            pool = multiprocessing.Pool(num)
            pool.starmap(func=pushtool.push_to, iterable=params)
            pool.close()
    except:
        traceback.print_exc()
        logger.error("[ConfigError] occur error when load task config")


def regularized(data: dict, pushtool: push.PushTo) -> dict:
    if not data or not pushtool:
        return {}

    scripts, persists = data.get("scripts", {}), data.get("persists", {})
    tasks, groups = {}, {}

    if type(scripts) != dict or type(persists) != dict:
        logger.error("[ConfigError] scripts or persists not ilegal")
        return {}

    for k, v in scripts.items():
        if utils.isblank(k) or type(v) != dict or not v.get("enable", True):
            continue
        tasks[k] = v.get("params", {})

    if not tasks:
        return {}

    for k, v in persists.items():
        if pushtool.validate(v):
            groups[k] = v

    if "candidates" not in groups or "availables" not in groups:
        return {}

    threshold = max(int(data.get("threshold", 72)), 1)
    return {"threshold": threshold, "scripts": tasks, "persists": groups}


def fetct_exist(persists: dict, pushtool: push.PushTo) -> dict:
    if not persists or not pushtool:
        return {}

    candidates = {}
    for k, v in persists.items():
        content = utils.http_get(url=pushtool.raw_url(v))
        if not content:
            continue
        if k != "candidates":
            for site in content.split(","):
                if not utils.isblank(site):
                    candidates[site] = {"defeat": 0}

            continue

        try:
            candidates.update(json.loads(content))
        except:
            logger.error("[CandidatesError] fetch candidates failed")

    return candidates


def deduplicate(sites: list[str]) -> list[str]:
    if not sites:
        return []

    dataset = {}
    for site in sites:
        hostname = parse.urlparse(url=site).netloc
        url = dataset.get(hostname, "")
        if not url or len(url) < len(site):
            dataset[hostname] = site

    return list(dataset.values())


def batch_probe(urls: list[str], candidates: dict, threshold: int) -> dict:
    if not urls and not candidates:
        return {}

    urls = urls if urls is not None else []
    candidates = candidates if candidates is not None else {}
    urls.extend(candidates.keys())
    sites = deduplicate(sites=urls)
    logger.info(f"[ProbeSites] starting probe count: {len(sites)} sites: {sites}")

    with multiprocessing.Manager() as manager:
        collections, potentials, processes = manager.list(), manager.dict(), []
        semaphore, starttime = multiprocessing.Semaphore(max(50, 1)), time.time()
        for site in sites:
            semaphore.acquire()
            p = multiprocessing.Process(
                target=check,
                args=(
                    site,
                    candidates,
                    threshold,
                    collections,
                    potentials,
                    semaphore,
                ),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        availables = list(collections)
        cost = "{:.3}s".format(time.time() - starttime)
        logger.info(
            f"[ProbeInfo] collect finished, cost: {cost} count: {len(availables)} sites: {availables}"
        )
        data = {
            "availables": ",".join(availables),
            "candidates": json.dumps(dict(potentials)),
        }
        return data


def check(
    url: str,
    candidates: dict,
    threshold: int,
    availables: ListProxy,
    potentials: DictProxy,
    semaphore: Semaphore,
) -> None:
    try:
        status, apipath = judge(url=url, retry=2)
        # reachable
        if status:
            availables.append(apipath)
            return

        if url not in candidates:
            return

        defeat = candidates.get(url).get("defeat", 0) + 1
        if defeat > threshold:
            return

        potentials[url] = {"defeat": defeat}
    finally:
        if semaphore is not None:
            semaphore.release()


def parse_url(url: str) -> tuple[bool, str, GPTProvider]:
    if not isurl(url):
        return False, "", GPTProvider.UNKNOWN

    result = parse.urlparse(url=url)
    full, mode = result.path and result.path != "/", GPTProvider.UNKNOWN
    apipath = f"{result.scheme}://{result.netloc}"
    if full:
        apipath = f"{apipath}{result.path}"
        mode = COMMON_PATH_MODE.get(result.path, mode)

    if result.query:
        params = {k: v[0] for k, v in parse.parse_qs(result.query).items()}
        provider = query_provider(params.get("mode", ""))
        mode = provider if provider != GPTProvider.UNKNOWN else mode

    return full, apipath, mode


def generate_tasks(url: str) -> list[tuple[str, GPTProvider]]:
    full, apipath, mode = parse_url(url=url)
    if not apipath:
        return []

    flag = mode != GPTProvider.UNKNOWN
    if full:
        if flag:
            return [(apipath, mode)]
        else:
            return [(apipath, x) for x in COMMON_PAYLOAD.keys()]
    if flag:
        return [(f"{apipath}{x}", mode) for x in COMMON_PATH_MODE.keys()]

    return [
        (f"{apipath}{x}", y)
        for x in COMMON_PATH_MODE.keys()
        for y in COMMON_PAYLOAD.keys()
    ]


def judge(url: str, retry: int = 2) -> tuple[bool, str]:
    """
    Judge whether the website is valid and sniff api path
    """
    tasks = generate_tasks(url=url)
    if not tasks:
        return False, ""

    for apipath, mode in tasks:
        body = COMMON_PAYLOAD.get(mode)
        link = f"{apipath}?mode={mode.name.lower()}"
        if not body:
            continue
        try:
            response = utils.http_post(
                url=apipath, headers=HEADERS, params=body, retry=retry
            )

            if not response or response.getcode() != 200:
                continue

            content_type = response.headers.get("content-type", "")
            if not content_type or "text/plain" in content_type:
                return True, link

            content = response.read().decode("UTF8")
            if not content:
                continue
            try:
                index = content.rfind("\n", 0, len(content) - 2)
                text = content if index < 0 else content[index + 1 :]
                data = json.loads(text)
                if data.get("id", ""):
                    return True, link
            except:
                if not re.search(
                    r'"role":(\s+)?".*",(\s+)?"id":(\s+)?"[A-Za-z0-9\-]+"', content
                ):
                    return True, link
                else:
                    logger.error(
                        f"[JudgeError] url: {apipath} mode: {mode.name} message: {content}"
                    )
        except Exception as e:
            logger.error(
                f"[JudgeError] url: {apipath} mode: {mode.name} message: {str(e)}"
            )

    return False, ""


def debug(url: str, response: HTTPResponse) -> None:
    status = response.getcode() if response else 000
    message = response.read().decode("UTF8") if response else ""
    logger.info(f"url: {url} code: {status} response: {message}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=False,
        default=os.environ.get(
            "SUBSCRIBE_CONF",
            "https://pastebin.enjoyit.ml/api/file/raw/clgh9ryvd0000l308s7e38h4y",
        ).strip(),
        help="remote config file",
    )

    args = parser.parse_args()
    process(url=args.config)
