import argparse
import importlib
import json
import multiprocessing
import os
import re
import time
import traceback
from http.client import HTTPResponse
from multiprocessing.managers import DictProxy, ListProxy
from multiprocessing.synchronize import Semaphore

import push
import utils
from logger import logger
from urlvalidator import isurl


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

            domians = batch_call(tasks.get("scripts", {}))
            # exists sites
            texts, candidates = fetct_exist(tasks.get("persists"), pushtool)
            domians.extend(texts)

            data = batch_probe(
                domains=domians,
                candidates=candidates,
                threshold=tasks.get("threshold"),
                chatgptweb="chatgptweb" in tasks.get("persists"),
                chatgptnextweb="chatgptnextweb" in tasks.get("persists"),
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

    if "candidates" not in groups or (
        "chatgptweb" not in groups and "chatgptnextweb" not in groups
    ):
        return {}

    threshold = max(int(data.get("threshold", 72)), 1)
    return {"threshold": threshold, "scripts": tasks, "persists": groups}


def fetct_exist(persists: dict, pushtool: push.PushTo) -> tuple[list, dict]:
    if not persists or not pushtool:
        return [], {}

    sites, candidates = [], {}
    for k, v in persists.items():
        content = utils.http_get(url=pushtool.raw_url(v))
        if not content:
            continue
        if k != "candidates":
            sites.extend([x for x in content.split(",") if not utils.isblank(x)])
            continue

        try:
            candidates = json.loads(content)
        except:
            logger.error("[CandidatesError] fetch candidates failed")

    return sites, candidates


def batch_probe(
    domains: list[str],
    candidates: dict,
    threshold: int,
    chatgptweb: bool = True,
    chatgptnextweb: bool = True,
) -> dict:
    if (not domains and not candidates) or (not chatgptweb and not chatgptnextweb):
        return {}

    domains = domains if domains is not None else []
    candidates = candidates if candidates is not None else {}
    domains.extend(candidates.keys())
    sites = list(set(domains))
    logger.info(f"[ProbeSites] {len(sites)} sites: {sites}")

    with multiprocessing.Manager() as manager:
        collections, potentials, processes = manager.dict(), manager.dict(), []
        semaphore = multiprocessing.Semaphore(max(50, 1))
        for domain in sites:
            semaphore.acquire()
            p = multiprocessing.Process(
                target=check,
                args=(
                    domain,
                    candidates,
                    threshold,
                    collections,
                    potentials,
                    semaphore,
                    chatgptweb,
                    chatgptnextweb,
                ),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        data = {"candidates": json.dumps(dict(potentials))}
        for k, v in collections.items():
            logger.info(f"[ProbeInfo] category: {k} count: {len(v)} sites: {v}")
            data[k] = ",".join(v)

        return data


def check(
    domain: str,
    candidates: dict,
    threshold: int,
    availables: DictProxy,
    potentials: DictProxy,
    semaphore: Semaphore,
    chatgptweb: bool = True,
    chatgptnextweb: bool = True,
) -> None:
    try:
        status, category = judge(
            domain=domain, chatgptweb=chatgptweb, chatgptnextweb=chatgptnextweb
        )
        # reachable
        if status:
            keys = category.lower()
            sites = availables.get(keys, [])
            sites.append(domain)
            availables[keys] = sites
            return

        # invalid domain if category is empty
        if category == "" or domain not in candidates:
            return

        defeat = candidates.get(domain).get("defeat", 0) + 1
        if defeat > threshold:
            return

        potentials[domain] = {"defeat": defeat, "category": category}
    finally:
        if semaphore is not None:
            semaphore.release()


def judge(
    domain: str, retry: int = 2, chatgptweb: bool = True, chatgptnextweb: bool = True
) -> tuple[bool, str]:
    """
    Judge whether the website is ChatGPTWeb, ChatGPTNextWeb or neither of them
    """
    if not isurl(domain):
        return False, ""

    try:
        # TODO check /api/config for ChatGPTWeb and /api/openai?_vercel_no_cache=1 for ChatGPTNextWeb
        # Reference: https://github.com/Chanzhaoyu/chatgpt-web/blob/main/service/src/index.ts#L48
        # Reference: https://github.com/Yidadaa/ChatGPT-Next-Web/blob/main/app/requests.ts#L51

        # check with ChatGPTWeb mode
        if chatgptweb:
            body = {"prompt": "What is ChatGPT?", "options": {}}
            url = f"{domain}/api/chat-process"
            response = utils.http_post(url=url, params=body, retry=retry)
            if response and response.getcode() == 200:
                content = response.read().decode("UTF8")
                avaiable = r'"id":' in content
                return avaiable, "ChatGPTWeb"

        # check with ChatGPTNextWeb mode
        if chatgptnextweb:
            headers = {
                "Content-Type": "application/json",
                "path": "v1/chat/completions",
            }
            url = f"{domain}/api/chat-stream"
            body = {
                "messages": [{"role": "user", "content": "What is ChatGPT?"}],
                "stream": False,
                "model": "gpt-3.5-turbo",
                "temperature": 1,
                "presence_penalty": 0,
            }

            response = utils.http_post(url=url, headers=headers, params=body, retry=2)
            avaiable = response and response.getcode() == 200
            return avaiable, "ChatGPTNextWeb"

        return False, "ChatGPTWeb"
    except:
        return False, "Unknown"


def debug(domain: str, response: HTTPResponse, chatgptweb: bool) -> None:
    status = response.getcode() if response else 000
    message = response.read().decode("UTF8") if response else ""
    category = "ChatGPTWeb" if chatgptweb else "ChatGPTNextWeb"
    logger.info(
        f"domain: {domain} code: {status} category: {category} response: {message}"
    )


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
